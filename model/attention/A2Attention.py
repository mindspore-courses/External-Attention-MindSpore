"""
DoubleAttention
"""

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, HeNormal, Normal


class DoubleAttention(nn.Cell):
    """
    Double Attention
    """
    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        self.c_m = c_m
        self.c_n = c_n

        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)

        if reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, 1)

        self.apply(self.init_weights)

    def init_weights(self, cell):
        """ init weight """
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(mode='fan_out'), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.beta.set_data(initializer('ones', cell.beta.shape, cell.beta.dtype))
            cell.gamma.set_data(initializer('zeros', cell.gamma.shape, cell.gamma.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(sigma=0.001), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        a = self.convA(x)  # b, c_m, h, w
        b = self.convB(x)  # b, c_n, h, w
        v = self.convV(x)  # b, c_n, h, w
        tmpA = a.view(b, self.c_m, -1)
        attention_maps = ms.ops.softmax(b.view(b, self.c_n, -1))
        attention_vectors = ms.ops.softmax(v.view(b, self.c_n, -1))

        global_descriptors = ms.ops.bmm(tmpA, attention_maps.permute(0, 2, 1))

        tmpZ = ms.ops.matmul(global_descriptors, attention_vectors)
        tmpZ = tmpZ.view(b, self.c_m, h, w)
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


if __name__ == "__main__":
    in_tensor = ms.ops.randn([12, 512, 7, 7])
    a2 = DoubleAttention(512, 128, 128)
    output = a2(in_tensor)
    print(output.shape)
