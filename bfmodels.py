import os
from pathlib import Path
import time
import datetime
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import BertForPreTraining, BertTokenizer, RobertaTokenizer, RobertaModel, BertForSequenceClassification
from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from dataloader import make_storage_label, get_dataloader_for_encoded_data

class BF_Model():
    """
    Wrapper round baseline bert model, fine-tuned bert models or fine-tuned roberta models,
    exposes fit, predict methods so can be used with sklearn utilities.
    
    Parameters
    ==========
    model_weight_name,  bert_model=True, fine_tune_model=True
    model, num_labels, batch_size = 32,
    """
    def __init__(self, model_weight_name,  bert_model, fine_tune_model, label_to_class, class_to_label, anli_type,
                 rounds, lr, batch_size, epochs, **kwargs):
        self.model_weight_name = model_weight_name
        self.bert_model = bert_model
        self.fine_tune_model = fine_tune_model
        self.num_labels = len(label_to_class)
        self.label_to_class = label_to_class
        self.class_to_label = class_to_label
        self.anli_type = anli_type
        self.rounds = rounds
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        if fine_tune_model:
            if bert_model:
                self.model = self.create_bert_model()
            else:
                self.model = self.create_roberta_model()
        else:
            self.model = self.create_baseline_bert_model()
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        self.params = [
            'model_weight_name', 
            'bert_model', 
            'fine_tune_model', 
            'label_to_class', 
            'class_to_label',
            'anli_type',
            'rounds',
            'batch_size',
            'epochs',
            'lr'
        ]

    def get_params(self, deep=True):
        params = self.params.copy()
        return {p: getattr(self, p) for p in params}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
   
            
    def fit(self, X, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters. If 'X_dev' is included,
            then performance is monitored every 10 epochs; use
            `dev_iter` to control this number.

        Returns
        -------
        self

        """
        # get other parameters:
        self.lr = kwargs.get("lr", 2e-5)
        self.epochs = kwargs.get("epochs", self.epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        X_dev = kwargs.get("X_dev")
        y_dev = kwargs.get("y_dev")
        
        print("batch_size:",self.batch_size)
        
        #todo: add asserts for above parameter
        
        if not self.fine_tune_model:
            # save model for later.
            output_filename = self.make_output_path()
            self.to_pickle(output_filename)
            return self
        
        storage_label = make_storage_label(self.anli_type, self.rounds, self.model_weight_name)
        train_dataloader = get_dataloader_for_encoded_data(X, y, self.model_weight_name, self.class_to_label, 
                             self.bert_model, storage_label, self.batch_size)
        
        
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(self.model.parameters(),
                          lr = self.lr, # args.learning_rate
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        
        
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        
        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        
        # For each epoch...
        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.model.train()
            
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 500 batches.
                if step % 100 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains four pytorch tensors:
                #   [0]: input ids 
                #   [1]: token_type_ids
                #   [2]: attention masks
                #   [3]: labels 
                if self.bert_model:
                    b_input_ids = batch[0].to(self.device, non_blocking=True)
                    b_token_type_ids = batch[1].to(self.device, non_blocking=True)
                    b_input_mask = batch[2].to(self.device, non_blocking=True)
                    b_labels = batch[3].to(self.device, non_blocking=True)
                else:
                    b_input_ids = batch[0].to(self.device, non_blocking=True)
                    b_input_mask = batch[1].to(self.device, non_blocking=True)
                    b_labels = batch[2].to(self.device, non_blocking=True)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                if self.bert_model:
                    loss, logits = self.model(b_input_ids, 
                                         token_type_ids=b_token_type_ids, 
                                         attention_mask=b_input_mask, 
                                         labels=b_labels)
                else:
                    loss, logits = self.model(b_input_ids, 
                                         attention_mask=b_input_mask, 
                                         labels=b_labels)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                loss = loss.sum()
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()
                
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            

            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            
            storage_label = make_storage_label("dev", self.rounds, self.model_weight_name)
            dev_dataloader = get_dataloader_for_encoded_data(X_dev, y_dev, self.model_weight_name, self.class_to_label, 
                                 self.bert_model, storage_label, self.batch_size)
            
            if X_dev is not None:
                # Evaluate data for one epoch
                for batch in dev_dataloader:
                    # Unpack this training batch from our dataloader. 
                    #
                    # As we unpack the batch, we'll also copy each tensor to the GPU using 
                    # the `to` method.
                    #
                    # `batch` contains four pytorch tensors:
                    #   [0]: input ids 
                    #   [1]: token_type_ids
                    #   [2]: attention masks
                    #   [3]: labels 
                    if self.bert_model:
                        b_input_ids = batch[0].to(self.device, non_blocking=True)
                        b_token_type_ids = batch[1].to(self.device, non_blocking=True)
                        b_input_mask = batch[2].to(self.device, non_blocking=True)
                        b_labels = batch[3].to(self.device, non_blocking=True)
                    else:
                        b_input_ids = batch[0].to(self.device, non_blocking=True)
                        b_input_mask = batch[1].to(self.device, non_blocking=True)
                        b_labels = batch[2].to(self.device, non_blocking=True)

                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    with torch.no_grad():        
                        # Forward pass, calculate logit predictions.
                        # token_type_ids is the same as the "segment ids", which 
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        # The documentation for this `model` function is here: 
                        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                        # Get the "logits" output by the model. The "logits" are the output
                        # values prior to applying an activation function like the softmax.
                        if self.bert_model:
                            (loss, logits) = self.model(b_input_ids, 
                                                   token_type_ids=b_token_type_ids, 
                                                   attention_mask=b_input_mask,
                                                   labels=b_labels)
                        else:
                            (loss, logits) = self.model(b_input_ids,  
                                                   attention_mask=b_input_mask,
                                                   labels=b_labels)

                    # Accumulate the validation loss.
                    loss = loss.sum()
                    total_eval_loss += loss.item()

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += self.flat_accuracy(logits, label_ids)

                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
                print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(dev_dataloader)

                # Measure how long the validation run took.
                validation_time = self.format_time(time.time() - t0)

                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print("  Validation took: {:}".format(validation_time))

                # Record all statistics from this epoch.
                training_stats.append(
                    {
                        'epoch': epoch_i + 1,
                        'Training Loss': avg_train_loss,
                        'Valid. Loss': avg_val_loss,
                        'Valid. Accur.': avg_val_accuracy,
                        'Training Time': training_time,
                        'Validation Time': validation_time
                    }
                )

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        
        # save model for later.
        output_filename = self.make_output_path()
        self.to_pickle(output_filename)
        return self
        
    def predict(self, X, **kwargs):
        """Predicted labels for the examples in `X`. These are converted
        from the integers that PyTorch needs back to their original
        values in `self.classes_`.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        list of length len(X)

        """
        
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        pred_array = []
        self.model.eval()
        
        dataloader = get_dataloader_for_encoded_data(X, None, self.model_weight_name, self.class_to_label, 
                                 self.bert_model, None, self.batch_size)
        for batch in dataloader:
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains four pytorch tensors:
            #   [0]: input ids 
            #   [1]: token_type_ids
            #   [2]: attention masks
            #   [3]: labels 
            if self.bert_model:
                b_input_ids = batch[0].to(self.device, non_blocking=True)
                b_token_type_ids = batch[1].to(self.device, non_blocking=True)
                b_input_mask = batch[2].to(self.device, non_blocking=True)
            else:
                b_input_ids = batch[0].to(self.device, non_blocking=True)
                b_input_mask = batch[1].to(self.device, non_blocking=True)
                

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                if self.bert_model:
                    logits = self.model(b_input_ids, 
                                           token_type_ids=b_token_type_ids, 
                                           attention_mask=b_input_mask)
                else:
                    logits = self.model(b_input_ids, 
                                           attention_mask=b_input_mask)
                
                pred_array.append(logits[0])
                
                
        preds = torch.cat(pred_array).cpu().numpy()
        return [self.label_to_class[i] for i in preds.argmax(axis=1)]

        
    def create_baseline_bert_model(self): 
        model = BertForPreTraining.from_pretrained(
            self.model_weight_name, 
            num_labels = self.num_labels,    
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        return model
    
    def create_bert_model(self):
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            self.model_weight_name, 
            num_labels = self.num_labels,    
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        return model
        
    def create_roberta_model(self):
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = RobertaForSequenceClassification.from_pretrained(
            self.model_weight_name, 
            num_labels = self.num_labels,    
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        return model
    
    
    def make_output_path(self):
        model_path = "models"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        op = "{}_{}{}_{}_{}_{}.pkl".format(self.model_weight_name, str(int(self.bert_model)),  
                                           str(int(self.fine_tune_model)), self.epochs, self.batch_size, self.lr)
        return os.path.join(model_path, op)
    
    
    def to_pickle(self, output_filename):
        """Serialize the entire class instance. Importantly, this
        is different from using the standard `torch.save` method:

        torch.save(self.model.state_dict(), output_filename)

        The above stores only the underlying model parameters. In
        contrast, the current method ensures that all of the model
        parameters are on the CPU and then stores the full instance.
        This is necessary to ensure that we retain all the information
        needed to read new examples and make predictions.

        """
        #self.model = self.model.cpu()
        with open(output_filename, 'wb') as f:
            pickle.dump(self, f)
            

    @staticmethod
    def from_pickle(src_filename):
        """Load an entire class instance onto the CPU. This also sets
        `self.warm_start = True` so that the loaded parameters are used
        if `fit` is called.

        Importantly, this is different from recommended PyTorch method:

        self.model.load_state_dict(torch.load(src_filename))

        We cannot reliably do this with new instances, because we need
        to see new examples in order to set some of the model
        dimensionalities and obtain information about what the class
        labels are. Thus, the current method loads an entire serialized
        class as created by `to_pickle`.

        The training and prediction code move the model parameters to
        `self.device`.

        Parameters
        ----------
        src_filename : str
            Full path to the serialized model file.

        """
        with open(src_filename, 'rb') as f:
            return pickle.load(f)
    
        
    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))