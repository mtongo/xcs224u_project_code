import os
from pathlib import Path
import json
from itertools import repeat
import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler

num_classes = 3
class_to_label = {'e':0, 'c':1, 'n':2}
label_to_class = {0:'e', 1:'c', 2:'n'}

def get_anli_raw_data(anli_type = 'train', rounds = (1, 2, 3)):
    DATA_HOME = os.path.join("data", "nlidata")
    ANLI_HOME = os.path.join(DATA_HOME, "anli_v0.1")
    src_filenames = []
    for r in rounds:
        src_filenames.append(
            os.path.join(
                ANLI_HOME,
                "R{}".format(r),
                "{}.jsonl".format(anli_type)))
        
    for src_filename in src_filenames:
        for line in open(src_filename, encoding='utf8'):
            d = json.loads(line)
            yield ((d['context'], d['hypothesis']), d['label'])
            
            
def get_eval_raw_data():
    EVAL_DATA_HOME = os.path.join("data", "financial-statement-extracts")
    src_filenames = ['ALPHABET INC.csv', 'AMERICAN EXPRESS CO.csv', 'SOUTHWEST AIRLINES CO.csv', 'ZOETIS INC.csv']
    for src_filename in src_filenames:
        df = pd.read_csv(os.path.join(EVAL_DATA_HOME, src_filename))
        for _, row in df.iterrows():
            yield ((row['plabel'], row['tlabel']), 'e')
            
            
def make_storage_label(anli_type, rounds, model_weight_name):
    return "{0}_{1}_R{2}".format(model_weight_name, anli_type, "".join(map(str, rounds)))


def get_anli_data_max_seq_len_for_tokenizer(X, tokenizer):
    max_len = 0
    for c,h in X:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(c, h, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    return max_len


def _encode_anli_data(X, y, model_weight_name, class_to_label_map, bert_model=True):
    # encode and store on disk for muliple use.
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    # storage_label should in the following format(not enforced), only to save time, 'bert_dev_R123' for 
    # bertTokenizer and Rounds (1, 2, 3) of anl_data
    
    hard_max_len = 128
    
    if bert_model:
        tokenizer = BertTokenizer.from_pretrained(model_weight_name, do_lower_case=True)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(model_weight_name, do_lower_case=True)
        
    max_len = get_anli_data_max_seq_len_for_tokenizer(X, tokenizer)
    
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []
    
    if y is None:
        yy = repeat(-1)
    else:
        yy = y
    
    # For every sentence...
    for (c,h),l in zip(X,yy):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            c,   # Sentence to encode.
                            h,
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = min(max_len, hard_max_len),           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its token_type_ids
        if bert_model:
            token_type_ids.append(encoded_dict['token_type_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        if y is not None:
            labels.append(class_to_label_map[l])
        else:
            labels.append(l)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    if bert_model:
        token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    labels = torch.tensor(labels, dtype=torch.long)

    return (input_ids, token_type_ids, attention_masks, labels)

def encoded_anli_data(X, y, model_weight_name, class_to_label_map, bert_model=True, storage_label=None):
    # encode and store on disk for muliple use.
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    # storage_label should in the following format(not enforced), only to save time, 'bert_dev_R123' for 
    # bertTokenizer and Rounds (1, 2, 3) of anl_data
    
    if not storage_label:
        return _encode_anli_data(X, y, model_weight_name, class_to_label_map, bert_model)
    
    storage_path = os.path.join('enc_data', storage_label)
    Path(storage_path).mkdir(parents=True, exist_ok=True)
    input_ids_path = os.path.join(storage_path, 'input_ids')
    token_type_ids_path = os.path.join(storage_path, 'token_type_ids')
    attention_masks_path = os.path.join(storage_path, 'attention_masks')
    labels_path = os.path.join(storage_path, 'labels')
    
    input_ids_exist = Path(input_ids_path).is_file()
    token_type_ids_exist = Path(token_type_ids_path).is_file()
    attention_masks_exist = Path(attention_masks_path).is_file()
    labels_path_exist = Path(labels_path).is_file()
    
    if bert_model:
        files_exist = input_ids_exist and token_type_ids_exist and attention_masks_path and labels_path_exist
    else:
        files_exist = input_ids_exist and attention_masks_path and labels_path_exist
        
    
    if files_exist:
        input_ids = torch.load(input_ids_path)
        if bert_model: 
            token_type_ids =  torch.load(token_type_ids_path)
        else:
            token_type_ids = None
        attention_masks = torch.load(attention_masks_path)
        labels = torch.load(labels_path)
    else:
        input_ids, token_type_ids, attention_masks, labels = \
            _encode_anli_data(X, y, model_weight_name, class_to_label_map, bert_model)
            
        torch.save(input_ids, input_ids_path)
        if bert_model: 
            torch.save(token_type_ids, token_type_ids_path)
        torch.save(attention_masks, attention_masks_path)
        torch.save(labels, labels_path)

    return (input_ids, token_type_ids, attention_masks, labels)


def get_dataloader_for_encoded_data(X, y, model_weight_name, class_to_label_map, bert_model=True, 
                                    storage_label=None, batch_size = 32):
    
    input_ids, token_type_ids, attention_masks, labels = \
        encoded_anli_data(X, y, model_weight_name, class_to_label_map, bert_model, storage_label)

    if bert_model:
        dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
    else:
        dataset = TensorDataset(input_ids, attention_masks, labels)
        
    """ll = len(dataset)
    sample_size = int(0.1 * ll)
    dataset, _ = random_split(dataset, [sample_size, ll-sample_size])"""
 
    dataloader = DataLoader(
                dataset,
                sampler = RandomSampler(dataset), # Select batches randomly
                batch_size = batch_size, # Trains with this batch size.
                pin_memory=True
            )

    return dataloader
