"""
MindSpore implementation of 'gMLP'
Refer to "Pay Attention to MLPs"
"""
from collections import OrderedDict
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer


class Residual(nn.Cell):
    """ Residual Block """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x):
        return self.fn(x) + x


class SpatialGatingUnit(nn.Cell):
    """ Spatial Gating Unit """
    def __init__(self, dim, len_sen):
        super().__init__()
        self.ln = nn.LayerNorm([dim])
        self.proj = nn.Conv1d(len_sen, len_sen, 1, has_bias=True)

        self.proj.weight.set_data(initializer('zeros', self.proj.weight.shape, self.proj.weight.dtype))
        self.proj.bias.set_data(initializer('ones', self.proj.bias.shape, self.proj.bias.dtype))

    def construct(self, x):
        res, gate = ms.ops.chunk(x, 2, -1)
        gate = self.ln(gate)
        gate = self.proj(gate)
        return res * gate


class gMLP(nn.Cell):
    """ gMLP """
    def __init__(self, num_tokens, len_sen=49, dim=512, d_ff=1024, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_tokens, dim) if num_tokens else nn.Identity()

        self.gmlp = nn.SequentialCell([Residual(nn.SequentialCell(OrderedDict([
            (f'ln1_{i}', nn.LayerNorm([dim])),
            (f'fc1_{i}', nn.Dense(dim, d_ff * 2)),
            (f'gelu_{i}', nn.GELU()),
            (f'sgu_{i}', SpatialGatingUnit(d_ff, len_sen)),
            (f'fc2_{i}', nn.Dense(d_ff, dim))
        ]))) for i in range(num_layers)])

        self.to_logits = nn.SequentialCell(
            nn.LayerNorm([dim]),
            nn.Dense(dim, num_tokens),
            nn.Softmax(-1)
        )

    def construct(self, x):
        embeded = self.embedding(x)

        y = self.gmlp(embeded)
        logits = self.to_logits(y)
        return logits


if __name__ == '__main__':
    dummy_input = ms.ops.randint(0, 10000, (50, 49))
    gmlp = gMLP(10000, 49, dim=512, d_ff=1024)
    output = gmlp(dummy_input)
    print(output.shape)
