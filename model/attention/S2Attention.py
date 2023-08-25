"""
MindSpore implementation of 'S2Attention'
Refer to "S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision"
"""
import mindspore as ms
from mindspore import nn


def spatial_shift1(x):
    """ spatial shift1 """
    _, W, H, C = x.shape
    x[:, 1:, :, :C // 4] = x[:, :W - 1, :, :C // 4]
    x[:, :W - 1, :, C // 4:C // 2] = x[:, 1:, :, C // 4:C // 2]
    x[:, :, 1:, C // 2:C * 3 // 4] = x[:, :, :H - 1, C // 2:C * 3 // 4]
    x[:, :, :H - 1, 3 * C // 4:] = x[:, :, 1:, 3 * C // 4:]
    return x


def spatial_shift2(x):
    """ spatial shift2 """
    _, H, W, C = x.shape
    x[:, :, 1:, :C // 4] = x[:, :, :H - 1, :C // 4]
    x[:, :, :H - 1, C // 4:C // 2] = x[:, :, 1:, C // 4:C // 2]
    x[:, 1:, :, C // 2:C * 3 // 4] = x[:, :W - 1, :, C // 2:C * 3 // 4]
    x[:, :W - 1, :, 3 * C // 4:] = x[:, 1:, :, 3 * C // 4:]
    return x


class SplitAttention(nn.Cell):
    """ Split Attention """
    def __init__(self, channels=512, k=3):
        super().__init__()
        self.channels = channels
        self.k = k
        self.mlp1 = nn.Dense(channels, channels)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Dense(channels, channels * k)
        self.softmax = nn.Softmax(1)

    def construct(self, x_all):
        B, K, H, W, C = x_all.shape
        x_all = x_all.reshape(B, K, -1, C)
        a = x_all.sum(1).sum(1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(B, self.k, C)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all

        out = out.sum(1).reshape(B, H, W, C)
        return out


class S2Attention(nn.Cell):
    """ S2 Attention """
    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Dense(channels, channels * 3)
        self.mlp2 = nn.Dense(channels, channels)
        self.split_attention = SplitAttention()

    def construct(self, x):
        _, C, _, _ = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :C])
        x2 = spatial_shift2(x[:, :, :, C:C * 2])
        x3 = x[:, :, :, C * 2:]
        x_all = ms.ops.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x = x.permute(0, 3, 1, 2)
        return x


if __name__ == '__main__':
    dummy_input = ms.ops.randn((50, 512, 7, 7))
    s2att = S2Attention(channels=512)
    output = s2att(dummy_input)
    print(output.shape)
