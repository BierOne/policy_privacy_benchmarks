# @Time : 2021/3/6 13:13
# @Author : BierOne
# @File : dataset.py
import os, json
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from utilities import config, utils


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('-', ' ') \
            .replace('.', '').replace('"', '').replace('n\'t', ' not').replace('$', ' dollar ')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word2vocab(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(len(self.word2idx))
        return tokens

    def dump_to_file(self, path):
        json.dump([self.word2idx, self.idx2word], open(path, 'w'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = json.load(open(path, 'r'))
        d = cls(word2idx, idx2word)
        return d

    def add_word2vocab(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_loader(split, batch_size=32):
    """ Returns a data loader for the desired split """
    assert split in ['train', 'test']
    dataset = PolicyPrivacyDataset(split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split != 'test' else False,  # only shuffle the data in training
        pin_memory=True,
        collate_fn=collate_fn,
        num_workers=config.workers,
    )
    return loader


def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


class PolicyPrivacyDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.dictionary = Dictionary.load_from_file(os.path.join(config.dataroot, 'dictionary.json'))
        self.entries = self._load_entries()

    def _load_entries(self):
        data = json.load(open(config.processed_data_path, 'r'))
        set_idxes = json.load(open(os.path.join(config.dataroot, 'set_idxes.json'), 'r'))
        set_idxes = set_idxes['train_set_idxes'] if self.split == 'train' else set_idxes['test_set_idxes']
        set_idxes = list(map(int, set_idxes))

        entries = []
        for idx in set_idxes:
            policy = data[idx]
            site_idx = int(policy['site_idx'])
            for seg_idx, seg in policy['segments'].items():
                seg_vec, seg_len = self.encode_segment(seg['segment'])
                entry = {
                    'site_idx': site_idx,
                    'seg_idx': int(seg_idx),
                    'segment': seg_vec,
                    'seg_len': seg_len,
                    'label': self.encode_categories(seg['category'])
                }
                entries.append(entry)
        print('length of entries: {}'.format(len(entries)))

        return entries

    def encode_segment(self, segment, max_length=config.max_segment_len):
        tokens = self.dictionary.tokenize(segment, False)
        tokens = tokens[:max_length]
        seg_len = len(tokens)
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        return torch.LongTensor(tokens), min(seg_len, max_length)

    def encode_categories(self, categories):
        target = torch.zeros(config.num_categories)
        labels = []
        for c in categories:
            label = config.CATEGORY_TO_LABEL.get(c)
            if label is not None:
                labels.append(label)
        labels = torch.LongTensor(labels)
        if len(labels):
            target.scatter_(0, labels, 1.0)
        return target

    def __getitem__(self, index):
        entry = self.entries[index]
        site_idx = entry['site_idx']
        seg_idx = entry['seg_idx']
        seg_vec = entry['segment']
        seg_len = entry['seg_len']
        label = entry['label']
        return site_idx, seg_idx, seg_vec, label, seg_len

    def __len__(self):
        return len(self.entries)
