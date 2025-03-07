import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from .AnomalyTransformer import AnomalyTransformer, Time_Positional_Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

class  Conv1dAnomalyTransformer(nn.Module):
    d_model = 8
    transformer_out = 16
    pr = True
    use_emb = False

    emb_size = 16
    win_size = 16
    def __init__(self, d_model = 16, layer = 6, num_class=6, with_time = False, use_emb=True, add_norm=False, in_dim=12, emb_size=128, win_size=15):
        print(f"init Conv1dTransformerEncoder with {num_class} classes")
        super(Conv1dAnomalyTransformer, self).__init__()
        self.transformer_out = d_model
        self.layer = layer
        self.use_emb = use_emb
        self.emb_size = emb_size
        self.win_size = win_size

        self.encoder = AnomalyTransformer(win_size=self.win_size, enc_in=self.transformer_out, c_out=self.transformer_out, d_model=self.transformer_out, n_heads=self.transformer_out, e_layers=self.layer, d_ff=512, dropout=0.5, output_attention=True, use_tse=with_time, max_time_position=10000)

        self.with_time = with_time
        self.add_norm = add_norm
        self.num_class = num_class

        self.cnn = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(960, self.transformer_out)
        )
        # self.cnn = nn.Sequential(
        #     nn.Linear(in_dim,  self.transformer_out),
        #     nn.ReLU(),
        # )
        self.emb = nn.Sequential(
            nn.Linear(self.win_size*self.transformer_out, self.win_size*self.transformer_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(self.win_size*self.transformer_out),
            nn.Linear(self.win_size*self.transformer_out, self.emb_size),
        ) if(self.use_emb) else None
    
    def forward(self, x, time=None, device='cuda:0'):
        # CNN
        ae_out = torch.empty((x.shape[0], self.transformer_out, 0)).to(device)

        for i in range(x.shape[1]):
            row = x[:, i, None ,:]
            tmp = self.cnn(row)
            tmp = tmp.view(ae_out.size(0), -1).unsqueeze(2)
            ae_out = torch.concat((ae_out, tmp), dim=2)
        
        x = ae_out.permute(0, 2, 1)
        # Transformer
        if (self.with_time):
            x = self.encoder(x, time)
        else:
            x = self.encoder(x)

        x = self.emb(x.view(x.size(0), -1)) if (self.use_emb) else torch.sum(x, 1)
        x = F.normalize(x, p=2, dim=-1) if (self.add_norm) else x
        
        return x
    
    def calculate_attention(self, features):
        max_pool = torch.max(features, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(features, dim=1, keepdim=True)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = torch.sigmoid(self.fc_att(combined))
        return attention_map
    
    def calculate_prototype(self, support_features):
        prototypes = []
        for i in range(self.num_class):
            class_features = support_features[i]
            attention_map = self.calculate_attention(class_features)
            weighted_features = class_features * attention_map
            prototype = weighted_features.mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)
        return prototypes
