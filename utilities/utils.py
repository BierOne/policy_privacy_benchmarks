# @Time : 2021/3/6 13:11
# @Author : BierOne
# @File : utils.py
import errno
import os, sys
import numpy as np
from os import path, listdir

from train.evaluation import *
from utilities import config

EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def load_raw_data(folder):
    files = []
    for f in listdir(folder):
        site_idx, site_name = f.split('_')
        site_name = site_name[:-4]
        file_path = path.join(folder, f)
        if path.isfile(file_path):
            files.append([site_idx, site_name, file_path])
    return files


def assert_array_eq(real, expected):
    assert (np.abs(real - expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def compute_f1scores_with_terms(TP, FP, FN):
    # Computation of F1 score, precision and recall
    Precision = TP / (TP + FP + EPS)
    Recall = TP / (TP + FN + EPS)
    F1 = (2 * Precision * Recall) / (Precision + Recall + EPS)
    return F1, Recall, Precision


def calculate_f1_per_label(pred, target):
    TP = (pred * target).sum(0)
    TN = ((1 - pred) * (1 - target)).sum(0)
    FP = (pred * (1 - target)).sum(0)
    FN = ((1 - pred) * target).sum(0)

    score_list = compute_f1scores_with_terms(TP, FP, FN)
    micro_score_list = compute_f1scores_with_terms(TP.sum(), FP.sum(), FN.sum())

    return score_list, micro_score_list


def print_results(pred, target):
    ''' print scores(f1, recall, precision) per label

    :param pred: [seg_num, num_categories]
    :param target: [seg_num, num_categories]
    :return
    '''

    print("{} Segments".format(pred.shape[0]))
    print("\n" + "Score per label with " + str(config.threshold) + " threshold")
    print("-" * 35 * 3)
    row_format = "{:<48}" + "{:<15}" * 4
    print(row_format.format("Label", "Num", "F1", "Precision", "Recall"""))
    print("-" * 35 * 3)
    scores_list, micro_score_list = calculate_f1_per_label(pred, target)
    label_num_list =  target.sum(0)
    for label, index in config.CATEGORY_TO_LABEL.items():
        f1_label = round(scores_list[0][index], 2)
        recall_label = round(scores_list[1][index], 2)
        precision_label = round(scores_list[2][index], 2)
        print(row_format.format(label, int(label_num_list[index]), f1_label, precision_label, recall_label))

    micro_score_list = list(map(lambda x: round(x, 2), micro_score_list))
    macro_score_list = list(map(lambda x: round(x.mean(), 2), scores_list))
    print("-" * 35 * 3)
    print(row_format.format('micro_average', '-', micro_score_list[0], micro_score_list[2], micro_score_list[1]))
    print(row_format.format('macro_average', '-', macro_score_list[0], macro_score_list[2], macro_score_list[1]))


class Tracker:
    """
        Keep track of results over time, while having access to
        monitors to display information about them.
    """

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l) # {name: l}
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        """ Storage of data points that updates the given monitors """

        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor) # ListStorage.mean = monitor

        def append(self, item, length=1):
            for monitor in self.monitors:
                monitor.update(item, length)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value, length=1):
            self.total += value
            self.n += length

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value, length=1):
            value = value / length
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value