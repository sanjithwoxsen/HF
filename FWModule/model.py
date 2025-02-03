import torch.nn as nn
import numpy as np
import torch
from transformers import PreTrainedModel
from .config import FeelWiseConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        self.pos_encoding = torch.tensor(pos_encoding, dtype=torch.float32)

    def forward(self, x):
        x = x * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:x.size(1), :].to(x.device)
        return x

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sub_layer_x):
        return self.layer_norm(x + sub_layer_x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, n_head)
        self.add_norm1 = AddNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sub_layer_x = self.multi_head_attention(x, x, x)[0]
        sub_layer_x = self.dropout(sub_layer_x)
        x = self.add_norm1(x, sub_layer_x)
        sub_layer_x = self.feed_forward(x)
        sub_layer_x = self.dropout(sub_layer_x)
        x = self.add_norm2(x, sub_layer_x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, max_len, input_vocab_size, n_head, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

class FeelWiseModel(PreTrainedModel):
    config_class = FeelWiseConfig
    base_model_prefix = "FeelWiseEmotion"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config.n_layers, config.d_model, config.max_len, config.input_vocab_size, config.n_head, config.d_ff, config.dropout)
        self.fc = nn.Linear(config.d_model, config.num_classes)  # Final classification layer

    def forward(self, input_ids):
        x = self.encoder(input_ids)  # Include attention_mask if your encoder uses it
        x = x.mean(dim=1)
        logits = self.fc(x)
        return logits