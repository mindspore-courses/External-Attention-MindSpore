"""
MindSpore implementation of 'gfnet'
Refer to "Global Filter Networks for Image Classification"
"""
# pylint: disable=E0401
import math
import mindspore as ms
from mindspore import nn
from model.layers import DropPath


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose((0, 2, 1))
        return x


class GlobalFilter(nn.Cell):
    """ global filter """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = ms.Parameter(ms.ops.randn((h, w, dim, 2), dtype=ms.float32) * 0.02)
        self.w = w
        self.h = h
        self.complex = ms.ops.Complex()
        # self.rfft2 = ms.ops.FFTWithSize(signal_ndim=2, inverse=False, real=True, norm='ortho', onesided=True,
        #                                 signal_sizes=( ))

    def construct(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a = b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(ms.float32)

        rfft2 = ms.ops.FFTWithSize(signal_ndim=3, inverse=False, real=True, onesided=True)
        irfft2 = ms.ops.FFTWithSize(signal_ndim=3, inverse=True, real=True, norm='ortho', onesided=True)

        x = rfft2(x.view(*x.shape[1:]).transpose(0, 2, 1)).transpose(0, 2, 1)

        weight = self.complex(self.complex_weight[:, :, :, 0], self.complex_weight[:, :, :, -1])
        x *= weight

        x = irfft2(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        x = x.reshape(B, N, C)
        return x


class Mlp(nn.Cell):
    """ mlp """
    def __init__(self, in_chans, hidden_chans=None, out_chans=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_chans = out_chans or in_chans
        hidden_chans = hidden_chans or in_chans
        self.fc1 = nn.Dense(in_chans, hidden_chans)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_chans, out_chans)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Cell):
    """ Block for GFNet """
    def __init__(self, dim, mlp_ratio=4., drop=1e-6, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14,
                 w=8):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.filter = GlobalFilter(dim, h, w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        y = self.norm1(x)

        y = self.filter(y)
        y = self.norm2(y)
        y = self.mlp(y)
        y = self.drop_path(y)
        return x + y


class GFNet(nn.Cell):
    """ GFNet """
    def __init__(self, embed_dim=384, img_size=224, patch_size=16, mlp_ratio=4, depth=4, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        self.embedding = nn.Dense((patch_size ** 2) * 3, embed_dim)

        h = img_size // patch_size
        w = h // 2 + 1

        self.blocks = nn.SequentialCell()
        for _ in range(depth):
            self.blocks.append(Block(dim=embed_dim, mlp_ratio=mlp_ratio, h=h, w=w))

        self.head = nn.Dense(embed_dim, num_classes)
        self.softmax = nn.Softmax(1)

    def construct(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)

        x = x.mean(axis=1)

        x = self.softmax(self.head(x))
        return x


if __name__ == "__main__":
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
    output = gfnet(dummy_input)
    print(output.shape)
