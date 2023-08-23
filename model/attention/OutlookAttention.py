"""
MindSpore implementation of 'OutlookAttention'
Refer to "VOLO: Vision Outlooker for Visual Recognition"
"""
import math
import mindspore as ms
from mindspore import nn


class OutlookAttention(nn.Cell):
    """ Outlook Attention """
    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1, qkv_bias=False,
                 attn_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = self.head_dim ** (-0.5)

        self.v_pj = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn = nn.Dense(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=attn_drop)

        self.unfold = nn.Unfold(ksizes=[1, kernel_size, kernel_size, 1],
                                strides=[1, stride, stride, 1],
                                rates=[1, 1, 1, 1], padding='same')
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def construct(self, x):
        B, H, W, C = x.shape

        # 映射到新的特征v
        v = self.v_pj(x).permute(0, 3, 1, 2)  # B,C,H,W
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v)
        nb, nw, _, _ = v.shape
        v = v.reshape(nb, nw, -1)
        v = v.reshape(B, self.num_heads, self.head_dim, self.kernel_size * self.kernel_size, h * w)\
            .permute(0, 1, 4, 3, 2)  # B,num_head,H*W,kxk,head_dim

        # 生成Attention Map
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # B,H,W,C
        attn = self.attn(attn).reshape(B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
                                       self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = self.scale * attn
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # 获取weighted特征
        out = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
        out = ms.ops.fold(out, output_size=ms.Tensor((H, W)), kernel_size=self.kernel_size,
                          padding=self.padding, stride=self.stride)  # B,C,H,W

        out = self.proj(out.transpose(0, 2, 3, 1))  # B,H,W,C
        out = self.proj_drop(out)

        return out


if __name__ == '__main__':
    dummy_input = ms.ops.randn((50, 28, 28, 512))
    outlook = OutlookAttention(dim=512)
    output = outlook(dummy_input)
    print(output.shape)
