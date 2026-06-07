import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class TextRNN(nn.Module):
    """Configurable RNN: LSTM | BiLSTM | GRU"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes,
                 rnn_type='bilstm', n_layers=2, dropout=0.4,
                 pad_idx=0, pretrained_emb=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        bidirectional = rnn_type in ('bilstm',)
        use_lstm = rnn_type in ('lstm', 'bilstm')
        RNNClass = nn.LSTM if use_lstm else nn.GRU

        self.rnn = RNNClass(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0
        )
        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * direction_factor, n_classes)
        self._use_lstm = use_lstm

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        if self._use_lstm:
            out, (hidden, _) = self.rnn(emb)
        else:
            out, hidden = self.rnn(emb)

        # Concat last forward + backward hidden
        if self.rnn.bidirectional:
            h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            h = hidden[-1]
        return self.fc(self.dropout(h))


class TextCNN(nn.Module):
    """Kim (2014) TextCNN with multiple filter sizes."""
    def __init__(self, vocab_size, embed_dim, n_classes,
                 filter_sizes=(2, 3, 4, 5), n_filters=256,
                 dropout=0.5, pad_idx=0, pretrained_emb=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embed_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_sizes), n_classes)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)  # (B, 1, L, D)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb).squeeze(3))  # (B, F, L-k+1)
            p = F.max_pool1d(c, c.shape[2]).squeeze(2)  # (B, F)
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))