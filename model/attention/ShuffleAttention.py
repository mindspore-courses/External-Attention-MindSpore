"""
MindSpore implementation of 'ShuffleAttention'
Refer to "SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS"
"""
import mindspore as ms
from mindspore import nn


class ShuffleAttention(nn.Cell):
    """ ShuffleAttention """
    def __init__(self, channels=512, G=8):
        super().__init__()
        self.G = G
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channels // (2 * G), channels // (2 * G))
        self.cweight = ms.Parameter(ms.ops.zeros((1, channels // (2 * G), 1, 1)))
        self.cbias = ms.Parameter(ms.ops.ones((1, channels // (2 * G), 1, 1)))
        self.sweight = ms.Parameter(ms.ops.zeros((1, channels // (2 * G), 1, 1)))
        self.sbias = ms.Parameter(ms.ops.ones((1, channels // (2 * G), 1, 1)))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        """ channel shuffle """
        B, _, H, W = x.shape
        x = x.reshape(B, groups, -1, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, -1, H, W)
        return x

    def construct(self, x):
        B, _, H, W = x.shape
        x = x.view(B * self.G, -1, H, W)

        x_0, x_1 = x.chunk(2, axis=1)

        # channel attention
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)

        out = ms.ops.cat((x_channel, x_spatial), axis=1)
        out = out.view(B, -1, H, W)
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == '__main__':
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    se = ShuffleAttention(channels=512, G=8)
    output = se(dummy_input)
    print(output.shape)
