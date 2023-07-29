""" involution """
import mindspore as ms
from mindspore import nn


class Involution(nn.Cell):
    """ Involution
    """
    def __init__(self, kernel_size, in_channels=4, stride=1, group=1, ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.stride = stride
        self.group = group

        assert self.in_channels % group == 0

        self.group_channel = self.in_channels // group
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels//ratio, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels // ratio)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.in_channels // ratio,
                               self.group * self.kernel_size * self.kernel_size, kernel_size=1)
        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def construct(self, x):
        B, C, H, W = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.avgpool(x)))))
        b, _, h, w = weight.shape
        weight = weight.reshape(b, self.group, self.kernel_size*self.kernel_size, h, w).unsqueeze(2)

        x_unfold = ms.ops.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride)
        x_unfold = x_unfold.reshape(B, self.group, C // self.group,
                                    self.kernel_size * self.kernel_size, H // self.stride, W // self.stride)

        out = (x_unfold * weight).sum(axis=3)
        out = out.reshape(B, C, H // self.stride, W // self.stride)
        return out


if __name__ == "__main__":
    in_tensor = ms.ops.randn(1, 4, 64, 64)
    involution = Involution(3, in_channels=4, stride=1)
    output = involution(in_tensor)
    print(output.shape)
