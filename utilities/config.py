# @Time : 2021/3/6 13:05
# @Author : BierOne
# @File : config.py
from collections import OrderedDict


dataset = 'OPP-115'
CATEGORY_TO_LABEL = OrderedDict([('First Party Collection/Use', 0),
                                 ('Third Party Sharing/Collection', 1),
                                 ('User Access, Edit and Deletion', 2),
                                 ('Data Retention', 3),
                                 ('Data Security', 4),
                                 ('International and Specific Audiences', 5),
                                 ('Do Not Track', 6),
                                 ('Policy Change', 7),
                                 ('User Choice/Control', 8),
                                 ('Introductory/Generic', 9),
                                 ('Practice not covered', 10),
                                 ('Privacy contact information', 11)])

dataroot = '/home/liuyibing/policy_privacy/data/'
processed_data_path = dataroot + 'processed_data.json'

raw_data_path = dataroot + dataset + '/raw_data/'
glove_path = dataroot + 'word_embed/glove/'
num_categories = 12
hid_dim = 1024
max_segment_len = 300
threshold = 0.5
workers = 2