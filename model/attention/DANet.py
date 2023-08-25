"""
MindSpore implementation of 'DANet'
Refer to "Dual Attention Network for Scene Segmentation"
"""
# pylint: disable=E0401
import mindspore as ms
from mindspore import nn
from SelfAttention import SelfAttention
from SimplifiedSelfAttention import SimplifiedScaledDotProductAttention


class PositionAttentionCell(nn.Cell):
    """ Position Attention """
    def __init__(self, d_model=512, kernel_size=3):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, pad_mode='pad', padding=(kernel_size - 1) // 2)
        self.pa = SelfAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def construct(self, x):
        B, C, _, _ = x.shape
        y = self.cnn(x)
        y = y.view(B, C, -1).permute(0, 2, 1)
        y = self.pa(y)
        return y


class ChannelAttentionCell(nn.Cell):
    """ Channel Attention """
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, pad_mode='pad', padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)

    def construct(self, x):
        B, C, _, _ = x.shape
        y = self.cnn(x)
        y = y.view(B, C, -1)
        y = self.pa(y)
        return y


class DACell(nn.Cell):
    """ DANet """
    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.position_attention = PositionAttentionCell(d_model=d_model, kernel_size=kernel_size)
        self.channel_attention = ChannelAttentionCell(d_model=d_model, kernel_size=kernel_size, H=H, W=W)

    def construct(self, x):
        B, C, H, W = x.shape
        p_out = self.position_attention(x)
        c_out = self.channel_attention(x)
        p_out = p_out.permute(0, 2, 1).view(B, C, H, W)
        c_out = c_out.view(B, C, H, W)
        return p_out + c_out


if __name__ == '__main__':
    dummy_input = ms.ops.randn((50, 512, 7, 7))
    model = DACell(d_model=512, kernel_size=3, H=7, W=7)
    print(model(dummy_input).shape)
