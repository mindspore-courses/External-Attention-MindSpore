"""
MindSpore implementation of 'SelfAttention'
Refer to "Attention Is All You Need"
"""

import math
import mindspore as ms
from mindspore import nn


class SelfAttention(nn.Cell):
    """ Self Attention """
    def __init__(self, d_model, d_k=None, d_v=None, h=1, drop_rate=0.1):
        super().__init__()
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.fc_q = nn.Dense(d_model, h * d_k)
        self.fc_k = nn.Dense(d_model, h * d_k)
        self.fc_v = nn.Dense(d_model, h * d_v)
        self.fc_o = nn.Dense(h * d_v, d_model)
        self.dropout = nn.Dropout(p=drop_rate)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def construct(self, x, attention_mask=None, attention_weights=None):
        B, N = x.shape[:2]

        q = self.fc_q(x).view(B, N, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(B, N, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(x).view(B, N, self.h, self.d_v).permute(0, 2, 1, 3)

        att = ms.ops.matmul(q, k) / math.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -float('inf'))

        att = ms.ops.softmax(att, axis=-1)
        att = self.dropout(att)
        att = ms.ops.matmul(att, v).permute(0, 2, 1, 3).view(B, N, self.h * self.d_v)
        return self.fc_o(att)


if __name__ == "__main__":
    dummy_input = ms.ops.rand(4, 128, 768)
    attention = SelfAttention(768)
    out = attention(dummy_input)
    print(out.shape)
