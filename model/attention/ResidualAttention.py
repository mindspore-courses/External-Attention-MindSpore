"""
MindSpore implementation of 'ResidualAttention'
Refer to "Residual Attention: A Simple but Effective Method for Multi-Label Recognition"
"""
import mindspore as ms
from mindspore import nn


class ResidualAttention(nn.Cell):
    """ ResidualAttention """
    def __init__(self, channels=512, num_classes=1000, la=.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channels, out_channels=num_classes, kernel_size=1, stride=1)

    def construct(self, x):
        y_raw = self.fc(x).flatten(start_dim=2)
        y_avg = y_raw.mean(axis=2)
        y_max = y_raw.max(axis=2)[0]
        score = y_avg + self.la * y_max
        return score


if __name__ == '__main__':
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    model = ResidualAttention(channels=512, num_classes=1000, la=0.2)
    output = model(dummy_input)
    print(output.shape)
