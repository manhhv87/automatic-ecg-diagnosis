import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)

from densenet1d import _DenseNet


def ecg_feature_extractor(input_layer=None, stages=None):
    backbone_model = _DenseNet(input_layer=input_layer,
                               num_outputs=None,
                               blocks=(6, 4, 6, 0)[:stages],  # Own model
                               # blocks=(6, 12, 24, 16)[:stages],   # DenseNet-121
                               # blocks=(6, 12, 32, 32)[:stages],   # DenseNet-169
                               # blocks=(6, 12, 48, 32)[:stages],   # DenseNet-201
                               # blocks=(6, 12, 64, 48)[:stages],   # DenseNet-264
                               first_num_channels=16,
                               # first_num_channels=64,
                               growth_rate=8,
                               # growth_rate=32,
                               kernel_size=(8, 6, 8, 4),
                               # kernel_size=(3, 3, 3, 3),
                               bottleneck=True,
                               dropout_rate=None,
                               include_top=False).model()

    return backbone_model


# %% Auxiliary functions
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T


def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""

    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])

        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))

        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index - 1] if index != 0 else threshold[0] - 1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


def affer_results(y_true, y_pred):
    """Return true positives, false positives, true negatives, false negatives.

    Args:
        y_true : ndarray
            True value
        y_pred : ndarray
            Predicted value

    Returns:
        tn, tp, fn, fp: ndarray
            Boolean matrices containing true negatives, true positives, false negatives and false positives.
        cm : ndarray
            Matrix containing: 0 - true negative, 1 - true positive,
                               2 - false negative, and 3 - false positive.
    """

    tn = (y_true == y_pred) & (y_pred == 0)  # True negative
    tp = (y_true == y_pred) & (y_pred == 1)  # True positive
    fp = (y_true != y_pred) & (y_pred == 1)  # False positive
    fn = (y_true != y_pred) & (y_pred == 0)  # False negative

    # Generate matrix of "tp, fp, tn, fn"
    m, n = np.shape(y_true)
    cm = np.zeros((m, n), dtype=int)
    cm[tn] = 0
    cm[tp] = 1
    cm[fn] = 2
    cm[fp] = 3
    return tn, tp, fn, fp, cm
