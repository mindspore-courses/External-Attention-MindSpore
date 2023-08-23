"""
MindSpore implementation of 'TripletAttention'
Refer to "Rotate to Attend: Convolutional Triplet Attention Module"
"""
import mindspore as ms
from mindspore import nn


class BasicConv(nn.Cell):
    """ BasicConv """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad', padding=padding, dilation=dilation, group=groups, has_bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01) if bn else nn.Identity()
        self.relu = nn.ReLU() if relu else nn.Identity()

    def construct(self, x):
        return self.relu(self.bn(self.conv(x)))


class ZPool(nn.Cell):
    """ ZPool """
    def construct(self, x):
        return ms.ops.cat((ms.ops.max(x, 1)[0].unsqueeze(1), ms.ops.mean(x, 1).unsqueeze(1)), axis=1)


class AttentionGate(nn.Cell):
    """ AttentionGate """
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def construct(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = ms.ops.sigmoid(x_out)
        return x * scale


class TripletAttention(nn.Cell):
    """ TripletAttention """
    def __init__(self, no_spatial=False):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def construct(self, x):
        x_out1 = self.cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_out2 = self.hc(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = (x_out + x_out1 + x_out2) / 3
        else:
            x_out = (x_out1 + x_out2) / 2
        return x_out


if __name__ == '__main__':
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    triplet = TripletAttention()
    output = triplet(dummy_input)
    print(output.shape)
