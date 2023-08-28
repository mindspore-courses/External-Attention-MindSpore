# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `PatchConvnet`.
Refer to Augmenting Convolutional networks with attention-based aggregation
"""
from functools import partial
from typing import Optional

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal
from model.layers import DropPath, to_2tuple
from model.registry import register_model

__all__ = ['S60', 'S120', 'B60', 'B120', 'L60', 'L120', 'S60_multi']


def make_divisible(v, divisor=8, min_value=None):
    """ make divisible """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Cell):
    """ Squeeze Excite """
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=ms.ops.sigmoid, divisor=1, **_):
        super().__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, has_bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, has_bias=True)
        self.gate_fn = gate_fn

    def construct(self, x):
        x_se = x.mean((2, 3), keep_dims=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class Mlp(nn.Cell):
    """ Mlp """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Cell = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Learned_Aggregation_Layer(nn.Cell):
    """ Learned Aggregation Layer """
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(0, 1, 3, 2)
        attn = self.id(attn)
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Learned_Aggregation_Layer_multi(nn.Cell):
    """ Learned Aggregation Layer multi """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.num_classes = num_classes

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        B, N, C = x.shape
        q = (
            self.q(x[:, : self.num_classes])
            .reshape(B, self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x[:, self.num_classes:])
            .reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x[:, self.num_classes:])
            .reshape(B, N - self.num_classes, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(0, 1, 3, 2)
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(0, 2, 1, 3).reshape(B, self.num_classes, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class Layer_scale_init_Block_only_token(nn.Cell):
    """ Layer scale init Block for token """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Cell = nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Learned_Aggregation_Layer,
        Mlp_block=Mlp,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = ms.Parameter(init_values * ms.ops.ones(dim), requires_grad=True)
        self.gamma_2 = ms.Parameter(init_values * ms.ops.ones(dim), requires_grad=True)

    def construct(self, x: ms.Tensor, x_cls: ms.Tensor) -> ms.Tensor:
        u = ms.ops.cat((x_cls, x), axis=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Conv_blocks_se(nn.Cell):
    """ Conv Blocks with se """
    def __init__(self, dim: int):
        super().__init__()

        self.qkv_pos = nn.SequentialCell(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, group=dim, kernel_size=3, pad_mode='pad', padding=1, stride=1, has_bias=True),
            nn.GELU(),
            SqueezeExcite(dim, rd_ratio=0.25),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(0, 2, 1)
        x = x.reshape(B, C, H, W)
        x = self.qkv_pos(x)
        x = x.reshape(B, C, N)
        x = x.transpose(0, 2, 1)
        return x


class Layer_scale_init_Block(nn.Cell):
    """ Layer scale init Block """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        Attention_block=None,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention_block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_1 = ms.Parameter(init_values * ms.ops.ones((dim)), requires_grad=True)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.SequentialCell:
    """3x3 convolution with padding"""
    return nn.SequentialCell(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False),
    )


class ConvStem(nn.Cell):
    """Image to Patch Embedding"""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.SequentialCell(
            conv3x3(in_chans, embed_dim // 8, 2),
            nn.GELU(),
            conv3x3(embed_dim // 8, embed_dim // 4, 2),
            nn.GELU(),
            conv3x3(embed_dim // 4, embed_dim // 2, 2),
            nn.GELU(),
            conv3x3(embed_dim // 2, embed_dim, 2),
        )

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.proj(x).flatten(start_dim=2).transpose(0, 2, 1)
        return x


class PatchConvnet(nn.Cell):
    """ PatchConvnet """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        block_layers=Layer_scale_init_Block,
        block_layers_token=Layer_scale_init_Block_only_token,
        Patch_layer=ConvStem,
        act_layer: nn.Cell = nn.GELU,
        Attention_block=Conv_blocks_se,
        dpr_constant: bool = True,
        init_scale: float = 1e-4,
        Attention_block_token_only=Learned_Aggregation_Layer,
        Mlp_block_token_only=Mlp,
        depth_token_only: int = 1,
        mlp_ratio_clstk: float = 3.0,
        multiclass: bool = False,
    ):
        super().__init__()

        self.multiclass = multiclass
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        if not self.multiclass:
            self.cls_token = ms.Parameter(ms.ops.zeros((1, 1, int(embed_dim))))
        else:
            self.cls_token = ms.Parameter(ms.ops.zeros((1, num_classes, int(embed_dim))))

        if not dpr_constant:
            dpr = [x.item() for x in ms.ops.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.SequentialCell(
            [
                block_layers(
                    dim=embed_dim,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    init_values=init_scale,
                )
                for i in range(depth)
            ]
        )

        self.blocks_token_only = nn.SequentialCell(
            [
                block_layers_token(
                    dim=int(embed_dim),
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio_clstk,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block_token_only,
                    Mlp_block=Mlp_block_token_only,
                    init_values=init_scale,
                )
                for i in range(depth_token_only)
            ]
        )

        self.norm = norm_layer((int(embed_dim), ))

        self.total_len = depth_token_only + depth

        self.feature_info = [{"num_chs": int(embed_dim), "reduction": 0, "module": 'head'}]
        if not self.multiclass:
            self.head = nn.Dense(int(embed_dim), num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.SequentialCell([nn.Dense(int(embed_dim), 1) for _ in range(num_classes)])

        self.rescale: float = 0.02

        self.cls_token = initializer(TruncatedNormal(sigma=self.rescale), self.cls_token.shape, self.cls_token.dtype)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weights """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=self.rescale), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def get_classifier(self):
        """ get classifier """
        return self.head

    def get_num_layers(self):
        """ get layers num """
        return len(self.blocks)

    def reset_classifier(self, num_classes: int):
        """ reset classifier """
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct_features(self, x: ms.Tensor) -> ms.Tensor:
        """ construct features """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        for _, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = ms.ops.cat((cls_tokens, x), axis=1)

        x = self.norm(x)

        if not self.multiclass:
            return x[:, 0]

        return x[:, : self.num_classes].reshape(B, self.num_classes, -1)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        B = x.shape[0]
        x = self.construct_features(x)
        if not self.multiclass:
            x = self.head(x)
            return x

        all_results = []
        for i in range(self.num_classes):
            all_results.append(self.head[i](x[:, i]))
        return ms.ops.cat(all_results, axis=1).reshape(B, self.num_classes)


@register_model
def S60(**kwargs):
    """ PatchConvnet-S60 """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def S120(**kwargs):
    """ PatchConvnet-S120 """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384,
        depth=120,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def B60(**kwargs):
    """ PatchConvnet-B60 """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=768,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        **kwargs
    )

    return model


@register_model
def B120(**kwargs):
    """ PatchConvnet-B120 """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=768,
        depth=120,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        **kwargs
    )

    return model


@register_model
def L60(**kwargs):
    """ PatchConvnet-L60 """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=1024,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def L120(**kwargs):
    """ PatchConvnet-L120 """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=1024,
        depth=120,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        init_scale=1e-6,
        mlp_ratio_clstk=3.0,
        **kwargs
    )

    return model


@register_model
def S60_multi(**kwargs):
    """ PatchConvnet-S60-multi """
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        Attention_block_token_only=Learned_Aggregation_Layer_multi,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,
        multiclass=True,
        **kwargs
    )

    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = S60()
    output = net(dummy_input)
    print(net)
    print(output.shape)
