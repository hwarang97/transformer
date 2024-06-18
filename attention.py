import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, emb_size=512, head_size=8):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.att_size = self.emb_size // self.head_size
        self.query = nn.Linear(self.emb_size, self.att_size)
        self.key = nn.Linear(self.emb_size, self.att_size)
        self.value = nn.Linear(self.emb_size, self.att_size)

    def forward(self, x):
        query = self.query(x) #(batch_size, slef.att_size, 1)
        key = self.key(x)
        value = self.value(x)
