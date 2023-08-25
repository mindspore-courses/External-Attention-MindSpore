"""
MindSpore implementation of 'ParNetAttention'
Refer to "Non-deep Networks"
"""
import mindspore as ms
from mindspore import nn


class ParNetAttention(nn.Cell):
    """ ParNetAttention """
    def __init__(self, channels=512):
        super().__init__()
        self.sse = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.SequentialCell(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )
        self.conv3x3 = nn.SequentialCell(
            nn.Conv2d(channels, channels, kernel_size=3, pad_mode='pad', padding=1),
            nn.BatchNorm2d(channels)
        )
        self.silu = nn.SiLU()

    def construct(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        out = self.silu(x1 + x2 + x3)
        return out


if __name__ == "__main__":
    dummy_input = ms.ops.randn((50, 512, 7, 7))
    pna = ParNetAttention(channels=512)
    output = pna(dummy_input)
    print(output.shape)
