"""
MindSpore implementation of 'SEAttention'
Refer to "Squeeze-and-Excitation Networks"
"""
import mindspore as ms
from mindspore import nn


class SEAttention(nn.Cell):
    """ SEAttention """
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(channels, channels // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(channels // reduction, channels, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    se = SEAttention(channels=512, reduction=8)
    output = se(dummy_input)
    print(output.shape)
