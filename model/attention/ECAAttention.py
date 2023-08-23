"""
MindSpore implementation of 'ECAAttention'
Refer to "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
"""
import mindspore as ms
from mindspore import nn


class ECAAttention(nn.Cell):
    """ ECA Attention """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, pad_mode='pad', padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


if __name__ == "__main__":
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    eca = ECAAttention()
    output = eca(dummy_input)
    print(output.shape)
