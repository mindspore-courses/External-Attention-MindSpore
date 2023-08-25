"""
MindSpore implementation of 'SimplifiedSelfAttention'
"""
import math
import mindspore as ms
from mindspore import nn


class SimplifiedScaledDotProductAttention(nn.Cell):
    """ Simplified Scale Dot Product Attention """
    def __init__(self, d_model, h=1, drop_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Dense(h * self.d_v, d_model)
        self.dropout = nn.Dropout(p=drop_rate)

    def construct(self, queries, keys=None, values=None, attention_mask=None, attention_weights=None):
        if keys is None:
            keys = queries
        if values is None:
            values = queries

        B, Nq = queries.shape[:2]
        Nk = keys.shape[1]
        q = queries.view(B, Nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = keys.view(B, Nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = values.view(B, Nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = ms.ops.matmul(q, k) / math.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -float('inf'))

        att = ms.ops.softmax(att, axis=-1)
        att = self.dropout(att)
        att = ms.ops.matmul(att, v).permute(0, 2, 1, 3).view(B, Nq, self.h * self.d_v)
        return self.fc_o(att)


if __name__ == '__main__':
    dummy_input = ms.ops.randn(50, 49, 512)
    ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
    output = ssa(dummy_input, dummy_input, dummy_input)
    print(output.shape)
