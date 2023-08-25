"""
MindSpore implementation of 'CoTAttention'
Refer to "Contextual Transformer Networks for Visual Recognition"
"""
import mindspore as ms
from mindspore import nn


class CoTAttention(nn.Cell):
    """ CoTAttention """
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.key_embed = nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, pad_mode='pad', padding=kernel_size // 2, group=4),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.value_embed = nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.SequentialCell(
            nn.Conv2d(2 * dim, 2 * dim // factor, kernel_size=1),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, kernel_size=1)
        )

    def construct(self, x):
        B, C, H, W = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(B, C, -1)

        y = ms.ops.cat([k1, x], axis=1)
        att = self.attention_embed(y)
        att = att.reshape(B, C, self.kernel_size * self.kernel_size, H, W)
        att = att.mean(2, keep_dims=False).view(B, C, -1)
        k2 = ms.ops.softmax(att, axis=-1) * v
        k2 = k2.view(B, C, H, W)

        return k1 + k2


if __name__ == "__main__":
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    cot = CoTAttention()
    output = cot(dummy_input)
    print(output.shape)
