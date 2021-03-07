import torch
import torch.nn as nn
from torch.nn.utils import rnn
import utilities.config as config


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, init_emb, freeze=False, dropout=0.0):
        super(WordEmbedding, self).__init__()
        weights = torch.from_numpy(init_emb)
        ntoken, emb_dim = weights.shape
        self.emb = nn.Embedding(ntoken, emb_dim, padding_idx=-1) # padding_idx= ntoken
        self.emb.weight.data = weights
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.emb(x))


class SentenceEmbedding(nn.Module):
    def __init__(self, w_dim, hid_dim, nlayers, bidirect, dropout=0.0, rnn_type='GRU'):
        """Module for question embedding
        """
        super(SentenceEmbedding, self).__init__()
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn_type = rnn_type
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.ndirections = 1 + int(bidirect)
        self.num_hid = hid_dim
        self.rnn = rnn_cls(
            w_dim, hid_dim, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

    def forward(self, x, x_len):
        """
        x: [batch, sequence, in_dim]
        return: if ndirections==1: [batch, num_hid] else [batch, 2*num_hid]
        """
        packed = rnn.pack_padded_sequence(x, x_len, batch_first=True)
        if self.rnn_type == 'GRU':
            output, hidden = self.rnn(packed) # output->[b, seq, h*ndirections], hidden->[1, b, h]
        elif self.rnn_type == 'LSTM':
            output, (hidden, _) = self.rnn(packed) # output->[b, seq, h*ndirections], hidden->[1, b, h]
        if self.ndirections == 1:
            return hidden.squeeze(0)
        output = rnn.pad_packed_sequence(output, batch_first=True, total_length=config.max_segment_len)[0]
        # get the last hidden unit in forward direction
        forward_ = output[:, -1, :self.num_hid]
        # get the first hidden unit in backward direction
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)
