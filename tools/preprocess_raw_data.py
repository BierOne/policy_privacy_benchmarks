# @Time : 2021/3/6 14:02
# @Author : BierOne
# @File : preprocess_raw_data.py
import os, sys
sys.path.append(os.getcwd())

import pandas as pd
import json
from utilities import config, utils


def main():
    files = utils.load_raw_data(config.raw_data_path)
    processed_data = []
    print('processing data...')
    for site_idx, site_name, file_path in files:
        site = {
            'site_idx': site_idx,
            'site_name': site_name,
            'segments': {}
        }
        segments = site['segments']
        with open(file_path, 'r') as opened_file:
            reader = pd.read_csv(opened_file, delimiter=',', names=["seg_idx", "segment", "category"])
            for i, row in reader.iterrows():
                seg_idx = row['seg_idx']
                if seg_idx in segments.keys():
                    segments[seg_idx]['category'].append(row['category'])
                else:
                    segments[seg_idx] = {
                        'segment': row['segment'],
                        'category': [row['category']]
                    }
        processed_data.append(site)
    json.dump(processed_data, open(config.processed_data_path, 'w'))
    print('done, the data path is: %s' % config.processed_data_path)

if __name__ == '__main__':
    main()