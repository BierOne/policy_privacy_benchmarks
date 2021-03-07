# @Time : 2021/3/6 23:22
# @Author : BierOne
# @File : evaluation.py
import torch
import torch.nn as nn
from utilities import config, utils
import torch.nn.functional as F
EPS = 1e-7


def compute_f1_score(pred, target, threshold=config.threshold):
    # True if the predicted score is greater than threshold
    pred = F.sigmoid(pred)
    pred = pred.ge(threshold).float().data
    TP = (pred * target).sum(dim=1) # [batch, num_label] -> [batch]
    TN = ((1 - pred) * (1 - target)).sum(dim=1)
    FP = (pred * (1 - target)).sum(dim=1)
    FN = ((1 - pred) * target).sum(dim=1)

    # Computation of F1 score, precision and recall
    F1, Recall, Precision = utils.compute_f1scores_with_terms(TP, FP, FN)
    metrics = {
        'TP': TP, 'TN': TN,
        'FP': FP, 'FN': FN,
        'F1': F1, 'Precision': Precision, 'Recall': Recall
    }
    return metrics, pred.long()


def binary_cross_entropy_with_logits(input, target, mean=False):
    """
    Function that measures Binary Cross Entropy between target and output logits:
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    loss = loss.sum(dim=1)
    return loss.mean() if mean else loss