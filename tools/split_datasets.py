# @Time : 2021/3/6 20:42
# @Author : BierOne
# @File : split_datasets.py
import os, sys
sys.path.append(os.getcwd())

import random, json
from utilities import config
TRAIN_SET_LENGTH = 75
TEST_SET_LENGTH = 115 - TRAIN_SET_LENGTH


def main():
    train_set_idxes = random.sample(range(115), TRAIN_SET_LENGTH)
    test_set_idxes = [i for i in range(115) if i not in train_set_idxes]
    set_idxes = {
        'train_set_idxes': train_set_idxes,
        'test_set_idxes': test_set_idxes
    }
    json.dump(set_idxes, open(os.path.join(config.dataroot, 'set_idxes.json'), 'w'))


if __name__ == '__main__':
    main()

