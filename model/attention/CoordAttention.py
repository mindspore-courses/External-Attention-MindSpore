"""
MindSpore implementation of 'CoordAttention'
Refer to "Coordinate Attention for Efficient Mobile Network Design"
"""
import mindspore as ms
from mindspore import nn


class h_sigmoid(nn.Cell):
    """ h sigmoid """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6()

    def construct(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Cell):
    """ h swish """
    def __init__(self):
        super().__init__()
        self.sigmoid = h_sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Cell):
    """ Coordinate Attention """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1)

    def construct(self, x):
        identity = x

        _, _, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = ms.ops.cat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = ms.ops.split(y, [H, W], axis=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


if __name__ == "__main__":
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    ca = CoordAtt(512, 512)
    output = ca(dummy_input)
    print(output.shape)
