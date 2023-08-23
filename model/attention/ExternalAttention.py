"""
MindSpore implementation of 'ExternalAttention'
Refer to "Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"
"""
import mindspore as ms
from mindspore import nn


class ExternalAttention(nn.Cell):
    """ External Attention """
    def __init__(self, in_channel, S=64):
        super().__init__()
        self.mk = nn.Dense(in_channel, S, has_bias=False)
        self.mv = nn.Dense(S, in_channel, has_bias=False)
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        attn = self.mk(x)
        attn = self.softmax(attn)
        attn = attn / attn.sum(axis=2, keepdims=True)
        out = self.mv(attn)
        return out


class MultiHeadExternalAttention(nn.Cell):
    """ Multi Head External Attention """
    def __init__(self, in_channel, S, head_num):
        super().__init__()
        self.H = head_num
        self.mk = nn.Dense(in_channel//head_num, S, has_bias= False)
        self.softmax = nn.Softmax(axis=2)
        self.mv = nn.Dense(S, in_channel//head_num, has_bias=False)

    def construct(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.H, C//self.H)
        x = x.permute(0, 2, 1, 3)

        attn = self.mk(x)
        attn = self.softmax(attn)
        attn = attn / attn.sum(axis=2, keepdims=True)
        out = self.mv(attn)

        out = out.permute(0, 2, 1, 3)
        out = out.view(B, N, C)
        return out


if __name__ == '__main__':
    dummy_input = ms.ops.randn([12, 48, 512], dtype=ms.float32)
    ea = ExternalAttention(in_channel=512, S=8)
    mhea = MultiHeadExternalAttention(in_channel=512, S=8, head_num=8)
    out1 = ea(dummy_input)
    out2 = mhea(dummy_input)
    print(f"External Attention: output shape is {out1.shape}")
    print(f"Multiple Heads External Attention: output shape is {out2.shape}")
