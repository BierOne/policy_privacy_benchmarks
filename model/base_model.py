# @Time : 2021/3/7 12:59
# @Author : BierOne
# @File : base_model.py
from model.language_model import WordEmbedding, SentenceEmbedding
from utilities import config
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, w_emb, s_emb, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.s_emb = s_emb
        self.classifier = classifier

    def forward(self, seg, label, seg_len):
        """Forward
        seg: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(seg)
        s_emb = self.s_emb(w_emb, seg_len)  # [batch, s_dim]
        logits = self.classifier(s_emb)

        return logits





def build_baseline_with_dl(embeddings):
    bidirect = False
    hidden_features = config.hid_dim
    setence_features = config.hid_dim
    w_emb = WordEmbedding(
        embeddings,
        dropout=0.5
    )
    s_emb = SentenceEmbedding(
        w_dim=300,
        hid_dim=setence_features,
        nlayers=1,
        bidirect=bidirect,
        dropout=0,
        rnn_type="LSTM"
    )

    classifier = nn.Sequential(
            nn.Linear(setence_features + setence_features*bidirect, hidden_features),
            nn.ReLU(),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(hidden_features, config.num_categories)
    )

    return BaseModel(w_emb, s_emb, classifier)