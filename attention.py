import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, emb_size=512, head_size=8):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.att_size = self.emb_size // self.head_size
        self.query = nn.Linear(self.emb_size, self.att_size)
        self.key = nn.Linear(self.emb_size, self.att_size)
        self.value = nn.Linear(self.emb_size, self.att_size)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        query = self.query(x) # query.shape: (batch_size, seq_len, att_size)
        key = self.key(x)
        value = self.value(x)
        score = torch.matmul(query, torch.transpose(key, -1, -2)) # score.shape: (batch_size, seq_len, seq_len)
        distribution = F.softmax(score, -1) # distribution.shape: (batch_size, seq_len, seq_len)
        attention = torch.matmul(distribution, value) # attention.shape: (batch_size, seq_len, att_size)
        return attention

