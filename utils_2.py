import random
import os
import sys
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def fit_classifier_with_crossvalidation(
        X, y, basemod, cv, param_grid, scoring='f1_macro', verbose=True):
    """Fit a classifier with hyperparameters set via cross-validation.

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
    y : list
        The list of labels for rows in `X`.
    basemod : an sklearn model class instance
        This is the basic model-type we'll be optimizing.
    cv : int
        Number of cross-validation folds.
    param_grid : dict
        A dict whose keys name appropriate parameters for `basemod` and
        whose values are lists of values to try.
    scoring : value to optimize for (default: f1_macro)
        Other options include 'accuracy' and 'f1_micro'. See
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose : bool
        Whether to print some summary information to standard output.

    Prints
    ------
    To standard output (if `verbose=True`)
        The best parameters found.
        The best macro F1 score obtained.

    Returns
    -------
    An instance of the same class as `basemod`.
        A trained model instance, the best model found.

    """
    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)
    # Report some information:
    if verbose:
        print("Best params: {}".format(crossvalidator.best_params_))
        print("Best score: {0:0.03f}".format(crossvalidator.best_score_))
    # Return the best model found:
    return crossvalidator.best_estimator_