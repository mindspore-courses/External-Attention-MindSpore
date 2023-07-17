""" Depthwise and Separable Convolution """
import mindspore as ms
from mindspore import nn


class DepthwiseSeparableConvolution(nn.Cell):
    """ DepthwiseSeparableConvolution """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        pad_mode='pad',
                                        padding=padding)

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=1,
                                        stride=1,
                                        group=1)

    def construct(self, x):
        x = self.depthwise_conv(x)
        out = self.pointwise_conv(x)
        return out


if __name__ == '__main__':
    in_tensor = ms.ops.randn((1, 3, 224, 224), dtype=ms.float32)
    conv = DepthwiseSeparableConvolution(3, 64)
    output = conv(in_tensor)
    print(output.shape)
