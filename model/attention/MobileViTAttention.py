"""
MindSpore implementation of 'MobileViTAttention'
Refer to "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
"""
import mindspore as ms
from mindspore import nn


class PreNorm(nn.Cell):
    """ PreNorm """
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm((dim, ))
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Cell):
    """ FeedForward """
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Dense(mlp_dim, dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        return self.net(x)


class Attention(nn.Cell):
    """ Attention Layer """
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(axis=-1)
        self.to_qkv = nn.Dense(dim, inner_dim * 3, has_bias=False)

        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

    def construct(self, x):
        q, k, v = self.to_qkv(x).chunk(3, axis=-1)
        b, p, n, hd = q.shape
        q = q.reshape(b, p, n, self.heads, hd // self.heads).transpose(0, 1, 3, 2, 4)
        k = k.reshape(b, p, n, self.heads, hd // self.heads).transpose(0, 1, 3, 2, 4)
        v = v.reshape(b, p, n, self.heads, hd // self.heads).transpose(0, 1, 3, 2, 4)

        dots = ms.ops.matmul(q, k.transpose(0, 1, 2, 4, 3)) * self.scale
        attn = self.attend(dots)
        out = ms.ops.matmul(attn, v)
        b, p, h, n, d = out.shape
        out = out.transpose(0, 1, 3, 2, 4).reshape(b, p, n, h*d)
        return self.to_out(out)


class Transformer(nn.Cell):
    """ Transformer Layer """
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.SequentialCell([])
        for _ in range(depth):
            self.layers.append(nn.SequentialCell([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def construct(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class MobileViTAttention(nn.Cell):
    """ MobileViT Attention """
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size,
                               pad_mode='pad', padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size,
                               pad_mode='pad', padding=kernel_size // 2)

    def construct(self, x):
        y = x.copy()  # bs,c,h,w

        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        bs, dim, h, w = y.shape
        y = y.reshape(bs, dim, h//self.ph, self.ph, w//self.pw, self.pw).transpose(0, 3, 5, 2, 4, 1)
        bs, ph, pw, nh, nw, dim = y.shape
        y = y.reshape(bs, ph*pw, nh*nw, dim)
        y = self.trans(y)
        bs, _, _, dim = y.shape
        y = y.reshape(bs, self.ph, self.pw, h//self.ph, w//self.pw, dim).transpose(0, 5, 3, 1, 4, 2)\
            .reshape(bs, dim, h, w)

        ## Fusion
        y = self.conv3(y)  # bs,dim,h,w
        y = ms.ops.cat([x, y], axis=1)  # bs,2*dim,h,w
        y = self.conv4(y)  # bs,c,h,w

        return y


if __name__ == '__main__':
    m = MobileViTAttention()
    dummy_input = ms.ops.randn(1, 3, 49, 49)
    output = m(dummy_input)
    print(output.shape)
