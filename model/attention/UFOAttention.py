"""
MindSpore implementation of 'UFOAttention'
Refer to "UFO-ViT: High Performance Linear Vision Transformer without Softmax"
"""
import mindspore as ms
from mindspore import nn


def XNorm(x, gamma):
    """ XNorm """
    norm_tensor = ms.ops.norm(x, 2, -1, True)
    return x * gamma / norm_tensor


class UFOAttention(nn.Cell):
    """ UFOAttention """
    def __init__(self, d_model, d_k=None, d_v=None, h=1, dropout=.1):
        super().__init__()
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.fc_q = nn.Dense(d_model, h * d_k)
        self.fc_k = nn.Dense(d_model, h * d_k)
        self.fc_v = nn.Dense(d_model, h * d_v)
        self.fc_o = nn.Dense(h * d_v, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.gamma = ms.Parameter(ms.ops.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def construct(self, queries, keys=None, values=None):
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        B, N = queries.shape[:2]
        q = self.fc_q(queries).view(B, N, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(B, N, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(B, N, self.h, self.d_v).permute(0, 2, 1, 3)

        kv = ms.ops.matmul(k, v)
        kv_norm = XNorm(kv, self.gamma)
        q_norm = XNorm(q, self.gamma)
        out = ms.ops.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).view(B, N, self.h * self.d_v)
        return self.fc_o(out)


if __name__ == '__main__':
    dummy_input = ms.ops.randn((50, 49, 512))
    ufo = UFOAttention(d_model=512, h=8)
    output = ufo(dummy_input)
    print(output.shape)
