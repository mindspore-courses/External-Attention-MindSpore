# pylint: disable=E0401
"""
MindSpore implementation of `HATNet`.
Refer to Vision Transformers with Hierarchical Attention
"""
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer as init, TruncatedNormal

from model.layers import DropPath


class InvertedResidual(nn.Cell):
    """ InvertedResidual """
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.SequentialCell(
            nn.GroupNorm(1, in_dim, eps=1e-6),
            nn.Conv2d(in_dim, hidden_dim, 1, has_bias=False),
            act_layer()
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
                      pad_mode='pad', padding=pad, group=hidden_dim, has_bias=False),
            act_layer()
        )
        self.conv3 = nn.SequentialCell(
            nn.Conv2d(hidden_dim, out_dim, 1, has_bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )
        self.drop = nn.Dropout2d(p=drop)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x


class Attention(nn.Cell):
    """ Attention """
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, drop=0.):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.drop = nn.Dropout2d(p=drop)

        if grid_size > 1:
            self.grid_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride=ds_ratio)
            self.ds_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Conv2d(dim, dim * 2, 1)

    def construct(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))

        if self.grid_size > 1:
            grid_h, grid_w = H // self.grid_size, W // self.grid_size
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, grid_h,
                              self.grid_size, grid_w, self.grid_size)
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3)
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q * self.scale) @ k.transpose(0, 2, 1)
            attn = ms.ops.softmax(attn, axis=-1)
            grid_x = (attn @ v).reshape(B, self.num_heads, grid_h, grid_w,
                                        self.grid_size, self.grid_size, self.head_dim)
            grid_x = grid_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, C, H, W)
            grid_x = self.grid_norm(x + grid_x)

            q = self.q(grid_x).reshape(B, self.num_heads, self.head_dim, -1)
            q = q.transpose(0, 1, 3, 2)
            kv = self.kv(self.ds_norm(self.avg_pool(grid_x)))
            kv = kv.reshape(B, 2, self.num_heads, self.head_dim, -1)
            kv = kv.permute(1, 0, 2, 4, 3)
            k, v = kv[0], kv[1]
        else:
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = ms.ops.softmax(attn, axis=-1)
        global_x = (attn @ v).transpose(0, 1, 3, 2).reshape(B, C, H, W)
        if self.grid_size > 1:
            global_x = global_x + grid_x
        x = self.drop(self.proj(global_x))

        return x


class Block(nn.Cell):
    """ Block """
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, expansion=4,
                 drop=0., drop_path=0., kernel_size=3, act_layer=nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(dim, head_dim, grid_size=grid_size, ds_ratio=ds_ratio, drop=drop)
        self.conv = InvertedResidual(dim, hidden_dim=dim * expansion, out_dim=dim,
                                     kernel_size=kernel_size, drop=drop, act_layer=act_layer)

    def construct(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x


class Downsample(nn.Cell):
    """ Downsample """
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, pad_mode='pad', padding=1, stride=2)
        self.norm = nn.GroupNorm(1, out_dim, eps=1e-6)

    def construct(self, x):
        x = self.norm(self.conv(x))
        return x


class HATNet(nn.Cell):
    """ HATNet """
    def __init__(self, num_classes=1000, dims=None,
                 head_dim=64, expansions=None, grid_sizes=None,
                 ds_ratios=None, depths=None, drop_rate=0.,
                 drop_path_rate=0., act_layer=nn.SiLU, kernel_sizes=None):
        super().__init__()
        if dims is None:
            dims = [64, 128, 256, 512]
        if expansions is None:
            expansions = [4, 4, 6, 6]
        if grid_sizes is None:
            grid_sizes = [1, 1, 1, 1]
        if ds_ratios is None:
            ds_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [3, 4, 8, 3]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]
        self.depths = depths
        self.patch_embed = nn.SequentialCell(
            nn.Conv2d(3, 16, 3, pad_mode='pad', padding=1, stride=2),
            nn.GroupNorm(1, 16, eps=1e-6),
            act_layer(),
            nn.Conv2d(16, dims[0], 3, pad_mode='pad', padding=1, stride=2),
        )

        self.blocks = []
        dpr = list(ms.ops.linspace(0, drop_path_rate, sum(depths)))
        for stage in range(len(dims)):
            self.blocks.append(nn.SequentialCell([Block(
                dims[stage], head_dim, grid_size=grid_sizes[stage], ds_ratio=ds_ratios[stage],
                expansion=expansions[stage], drop=drop_rate, drop_path=dpr[sum(depths[:stage]) + i],
                kernel_size=kernel_sizes[stage], act_layer=act_layer)
                for i in range(depths[stage])]))
        self.blocks = nn.SequentialCell(self.blocks)

        self.ds2 = Downsample(dims[0], dims[1])
        self.ds3 = Downsample(dims[1], dims[2])
        self.ds4 = Downsample(dims[2], dims[3])
        self.classifier = nn.SequentialCell(
            nn.Dropout(p=0.2),
            nn.Dense(dims[-1], num_classes),
        )

        # init weights
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        """ reset drop path """
        dpr = list(ms.ops.linspace(0, drop_path_rate, sum(self.depths)))
        cur = 0
        for stage, block in enumerate(self.blocks):
            for idx in range(self.depths[stage]):
                block[idx].drop_path.drop_prob = dpr[cur + idx]
            cur += self.depths[stage]

    def _init_weights(self, cell):
        """ initialize weight """
        if isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.set_data(init(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(init('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            cell.gamma.set_data(init('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(init('zeros', cell.beta.shape, cell.beta.dtype))

    def construct(self, x):
        x = self.patch_embed(x)
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds4(x)
        for block in self.blocks[3]:
            x = block(x)
        x = ms.ops.adaptive_avg_pool2d(x, (1, 1)).flatten(start_dim=1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    hat = HATNet(dims=[48, 96, 240, 384], head_dim=48, expansions=[8, 8, 4, 4],
                 grid_sizes=[8, 7, 7, 1], ds_ratios=[8, 4, 2, 1], depths=[2, 2, 6, 3])
    output = hat(dummy_input)
    print(hat)
    print(output.shape)
