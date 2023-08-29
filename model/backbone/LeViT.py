# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `LeViT`.
Refer to "LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference"
"""
import itertools

import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore import context
from mindspore import numpy as np
from mindspore.common import initializer as init


from model.helper import load_pretrained
from model.registry import register_model


__all__ = [
    "LeViT",
    "LeViT_128S",
    "LeViT_128",
    "LeViT_192",
    "LeViT_256",
    "LeViT_384",
]


def _cfg(url='', **kwargs):  # need to check for
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'LeViT_128S': _cfg(url=''),
    'LeViT_128': _cfg(url=''),
    'LeViT_192': _cfg(url=''),
    'LeViT_256': _cfg(url=''),
    'LeViT_384': _cfg(url='')
}

FLOPS_COUNTER = 0


class Conv2d_BN(nn.SequentialCell):
    """ Conv2d and BatchNorm """
    def __init__(self,
                 a: int,
                 b: int,
                 ks: int = 1,
                 stride: int = 1,
                 pad: int = 0,  # pad=1
                 dilation: int = 1,
                 group: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=a,
                              out_channels=b,
                              kernel_size=ks,
                              stride=stride,
                              padding=pad,
                              dilation=dilation,
                              group=group,
                              has_bias=False,
                              pad_mode="pad")

        self.bn = nn.BatchNorm2d(num_features=b,
                                 gamma_init="ones",
                                 beta_init="zeros",
                                 use_batch_statistics=True,
                                 momentum=0.9)  # 0.1

    def construct(self, input_data: Tensor) -> Tensor:
        x = self.conv(input_data)
        x = self.bn(x)
        return x


class Linear_BN(nn.SequentialCell):
    """ Dense and BatchNorm """
    def __init__(self,
                 a: int,
                 b: int) -> None:
        super().__init__()

        self.linear = nn.Dense(a,
                               b,
                               weight_init='Uniform',
                               bias_init='Uniform',
                               has_bias=False)

        self.bn1d = nn.BatchNorm1d(num_features=b,
                                   gamma_init="ones",
                                   beta_init="zeros",
                                   momentum=0.9)

    def construct(self, input_data: Tensor) -> Tensor:
        x = self.linear(input_data)
        x1, x2, x3 = x.shape
        new_x = ms.ops.reshape(x, (x1 * x2, x3))
        x = self.bn1d(new_x).reshape(x.shape)
        return x


class BN_Linear(nn.SequentialCell):
    """ BatchNorm and Dense """
    def __init__(self,
                 a: int,
                 b: int,
                 bias: bool = True,
                 std: float = 0.02) -> None:
        super().__init__()

        self.bn1d = nn.BatchNorm1d(num_features=a,
                                   gamma_init="ones",
                                   beta_init="zeros",
                                   momentum=0.9)

        self.linear = nn.Dense(a,
                               b,
                               weight_init=init.TruncatedNormal(sigma=std),
                               bias_init='zeros',
                               has_bias=bias)

    def construct(self, input_data: Tensor) -> Tensor:
        x = self.bn1d(input_data)
        x = self.linear(x)
        return x


class Residual(nn.Cell):
    """ Residual """
    def __init__(self,
                 m: type = None,
                 drop: int = 0):
        super().__init__()
        self.m = m
        self.drop = drop

    def construct(self, x: Tensor) -> Tensor:
        if self.training and self.drop > 0:
            return x + self.m(x) * ms.Tensor.to_tensor(
                (np.randn((x.shape[0], 1, 1)) > self.drop) / (1 - self.drop))

        y = self.m(x)
        x = x + y
        return x


def b16(n, activation=nn.HSwish):
    """ b16 """
    return nn.SequentialCell(
        Conv2d_BN(3, n // 8, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 8, n // 4, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 4, n // 2, 3, 2, 1),
        activation(),
        Conv2d_BN(n // 2, n, 3, 2, 1))


class Subsample(nn.Cell):
    """ DownSample """
    def __init__(self,
                 stride: int,
                 resolution: int):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def construct(self, x: Tensor) -> Tensor:
        B, _, C = x.shape
        x = x.view(B, self.resolution, self.resolution, C)[
            :, ::self.stride, ::self.stride].reshape(B, -1, C)
        return x


class Attention(nn.Cell):
    """ Attention """
    def __init__(self,
                 dim: int,
                 key_dim: int,
                 num_heads: int = 8,
                 attn_ratio: int = 4,
                 activation: type = None,
                 resolution: int = 14) -> None:

        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, h)
        self.proj = nn.SequentialCell(activation(), Linear_BN(self.dh, dim))

        points = list(itertools.product(range(resolution), range(resolution)))  # 迭代两个不同大小的列表来获取新列表
        self.N = len(points)
        self.softmax = nn.Softmax(axis=-1)

        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = ms.Parameter(
            Tensor(np.zeros([num_heads, len(attention_offsets)], np.float32)))

        attention_bias_idxs = ms.Tensor(idxs, dtype=ms.int64).view(self.N, self.N)
        self.attention_bias_idxs = ms.Parameter(attention_bias_idxs, requires_grad=False)

        self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def construct(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        atte = self.qkv(x).view(B, N, self.num_heads, -1)
        q, k, v = ms.ops.split(atte, [self.key_dim, self.key_dim, 2 * self.key_dim], axis=3)
        q = ms.ops.transpose(q, (0, 2, 1, 3))
        k = ms.ops.transpose(k, (0, 2, 1, 3))
        v = ms.ops.transpose(v, (0, 2, 1, 3))

        attn = (
                (ms.ops.matmul(q, ms.ops.transpose(k, (-4, -3, -1, -2)))) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        attn = self.softmax(attn)

        x = ms.ops.transpose((ms.ops.matmul(attn, v)), (0, 2, 1, 3))

        x = x.reshape(B, N, self.dh)

        x = self.proj(x)

        return x


class AttentionSubsample(nn.Cell):
    """ Attention SubSample """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 key_dim: int,
                 num_heads: int = 8,
                 attn_ratio: int = 2,
                 activation: type = None,
                 stride: int = 2,
                 resolution: int = 14,
                 resolution_: int = 7) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_ ** 2
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h)

        self.q = nn.SequentialCell(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd))
        self.proj = nn.SequentialCell(activation(), Linear_BN(self.dh, out_dim))
        self.softmax = nn.Softmax(axis=-1)
        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(range(resolution_), range(resolution_)))

        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = ms.Parameter(
            Tensor(np.zeros([num_heads, len(attention_offsets)], np.float32)))

        attention_bias_idxs = (ms.Tensor(idxs, dtype=ms.int64)).view((N_, N))
        self.attention_bias_idxs = ms.Parameter(attention_bias_idxs, requires_grad=False)

        self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def construct(self,
                  x: Tensor) -> Tensor:

        B, N, _ = x.shape
        atte = self.kv(x).view(B, N, self.num_heads, -1)
        k, v = ms.ops.split(atte, [self.key_dim, atte.shape[3] - self.key_dim], axis=3)
        v = ms.ops.transpose(v, (0, 2, 1, 3))
        k = ms.ops.transpose(k, (0, 2, 1, 3))

        q = self.q(x).view(B, self.resolution_2, self.num_heads, self.key_dim)
        q = ms.ops.transpose(q, (0, 2, 1, 3))

        attn = (
                ms.ops.matmul(q, ms.ops.transpose(k, (-4, -3, -1, -2))) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )

        attn = self.softmax(attn)

        x = ms.ops.transpose((ms.ops.matmul(attn, v)), (0, 2, 1, 3))
        x = x.reshape(B, -1, self.dh)
        x = self.proj(x)
        return x


class LeViT(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 num_classes: int = 1000,
                 embed_dim=None,
                 key_dim=None,
                 depth=None,
                 num_heads=None,
                 attn_ratio=None,
                 mlp_ratio=None,
                 hybrid_backbone: type = b16(128, activation=nn.HSwish),
                 down_ops=None,
                 attention_activation: type = nn.HSwish,
                 mlp_activation: type = nn.HSwish,
                 distillation: bool = True,
                 drop_path: int = 0):
        super().__init__()

        if embed_dim is None:
            embed_dim = [128, 256, 384]
        if key_dim is None:
            key_dim = [16, 16, 16]
        if depth is None:
            depth = [2, 3, 4]
        if num_heads is None:
            num_heads = [4, 6, 8]
        if attn_ratio is None:
            attn_ratio = [2, 2, 2]
        if mlp_ratio is None:
            mlp_ratio = [2, 2, 2]
        if down_ops is None:
            down_ops = [['Subsample', 16, 128 // 16, 4, 2, 2], ['Subsample', 16, 256 // 16, 4, 2, 2]]
        self.num_classes = num_classes
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.blocks = []

        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(nn.SequentialCell(
                            Linear_BN(ed, h),
                            mlp_activation(),
                            Linear_BN(h, ed),
                        ), drop_path))

            if do[0] == 'Subsample':
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(nn.SequentialCell(
                            Linear_BN(embed_dim[i + 1], h),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1]),
                        ), drop_path))
        self.blocks = nn.SequentialCell(*self.blocks)

        # Classifier head
        if num_classes > 0:
            self.head = BN_Linear(embed_dim[-1], num_classes)
            if distillation:
                self.head_dist = BN_Linear(embed_dim[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weights """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
            if cell.bias is not None:
                cell.bias.set_data(init.initializer('zeros', cell.bias.shape))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))

    def construct(self, x: Tensor) -> Tensor:

        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = ms.ops.transpose(x, (0, 2, 1))
        x = self.blocks(x)
        x = x.mean(1)
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


@register_model
def LeViT_128S(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    """ LeViT_128S """
    default_cfg = default_cfgs['LeViT_128S']
    model = LeViT(num_classes=num_classes,
                  embed_dim=[128, 256, 384],
                  num_heads=[4, 6, 8],
                  key_dim=[16, 16, 16],
                  depth=[2, 3, 4],
                  down_ops=[
                      ['Subsample', 16, 128 // 16, 4, 2, 2],
                      ['Subsample', 16, 256 // 16, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(128),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_128(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    """ LeViT_128 """
    default_cfg = default_cfgs['LeViT_128']
    model = LeViT(num_classes=num_classes,
                  embed_dim=[128, 256, 384],
                  num_heads=[4, 8, 12],
                  key_dim=[16, 16, 16],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 16, 128 // 16, 4, 2, 2],
                      ['Subsample', 16, 256 // 16, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(128),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_192(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    """ LeViT_192 """
    default_cfg = default_cfgs['LeViT_192']
    model = LeViT(num_classes=num_classes,
                  embed_dim=[192, 288, 384],
                  num_heads=[3, 5, 6],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 192 // 32, 4, 2, 2],
                      ['Subsample', 32, 288 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(192),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_256(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    """ LeViT_256 """
    default_cfg = default_cfgs['LeViT_256']
    model = LeViT(num_classes=num_classes,
                  embed_dim=[256, 384, 512],
                  num_heads=[4, 6, 8],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 256 // 32, 4, 2, 2],
                      ['Subsample', 32, 384 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(256),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


@register_model
def LeViT_384(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> LeViT:
    """ LeViT_384 """
    default_cfg = default_cfgs['LeViT_384']
    model = LeViT(num_classes=num_classes,
                  embed_dim=[384, 512, 768],
                  num_heads=[6, 9, 12],
                  key_dim=[32, 32, 32],
                  depth=[4, 4, 4],
                  down_ops=[
                      ['Subsample', 32, 384 // 32, 4, 2, 2],
                      ['Subsample', 32, 512 // 32, 4, 2, 2],
                  ],
                  hybrid_backbone=b16(384),
                  **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  # context.PYNATIVE_MODE

    net = LeViT_128S()
    # print(net)
    dummy_input = ms.ops.rand((4, 3, 224, 224))
    output = net(dummy_input)
    print(output.shape)
