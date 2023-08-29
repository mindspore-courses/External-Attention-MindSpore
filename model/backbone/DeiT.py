# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `DeiT`.
Refer to "Training data-efficient image transformers & distillation through attention"
"""
from enum import Enum
from functools import partial
from typing import Union, Tuple, Optional, Callable

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal, Normal

from model.layers import DropPath, to_2tuple
from model.registry import register_model
from model.helper import load_pretrained

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Format(str, Enum):
    """ image format """
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def nchw_to(x: ms.Tensor, fmt: Format):
    """ switch image format """
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(start_dim=2).transpose(0, 2, 1)
    elif fmt == Format.NCL:
        x = x.flatten(start_dim=2)
    return x


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Dense

        self.fc1 = linear_layer(in_features, hidden_features, has_bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer((hidden_features, )) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, has_bias=bias[1])
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchDropout(nn.Cell):
    """ https://arxiv.org/abs/2212.00794
    """
    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
            return_indices: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices

    def construct(self, x):
        if not self.training or self.prob == 0.:
            if self.return_indices:
                return x, None
            return x

        if self.num_prefix_tokens:
            prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
        else:
            prefix_tokens = None

        B = x.shape[0]
        L = x.shape[1]
        num_keep = max(1, int(L * (1. - self.prob)))
        keep_indices = ms.ops.argsort(ms.ops.randn((B, L)), axis=-1)[:, :num_keep]
        if self.ordered:
            # NOTE does not need to maintain patch order in typical transformer use,
            # but possibly useful for debug / visualization
            keep_indices = keep_indices.sort(dim=-1)[0]
        x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

        if prefix_tokens is not None:
            x = ms.ops.cat((prefix_tokens, x), axis=1)

        if self.return_indices:
            return x, keep_indices
        return x


class LayerScale(nn.Cell):
    """ Layer Scale """
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = ms.Parameter(init_values * ms.ops.ones((dim, )))

    def construct(self, x):
        return x * self.gamma


class Attention(nn.Cell):
    """ Attention """
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 3, 2)
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ Block """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim, ))
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(s // p for (s, p) in zip(self.img_size, self.patch_size))
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias)
        self.norm = norm_layer((embed_dim, )) if norm_layer else nn.Identity()

    def construct(self, x):
        _, _, H, W = x.shape
        if self.img_size is not None:
            assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(start_dim=2).transpose(0, 2, 1)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Cell):
    """ Vision Transformer
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,
            mlp_layer: Callable = Mlp,
    ):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(ms.ops.zeros((1, 1, embed_dim))) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = ms.Parameter(ms.ops.randn((1, embed_len, embed_dim)) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = list(ms.ops.linspace(0, drop_path_rate, depth))  # stochastic depth decay rule
        self.blocks = nn.SequentialCell(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer((embed_dim, )) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer((embed_dim, )) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(p=drop_rate)
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = initializer(TruncatedNormal(sigma=.02), self.pos_embed.shape, self.pos_embed.dtype)
        if self.cls_token is not None:
            self.cls_token = initializer(Normal(sigma=1e-6), self.cls_token.shape, self.cls_token.dtype)
        if weight_init != 'skip':
            self.apply(self.init_weights)

    def init_weights(self, cell):
        """ initialize weight """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def reset_classifier(self, num_classes: int, global_pool=None):
        """ reset classifier """
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        """ position embedding """
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = ms.ops.cat((self.cls_token, x), axis=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = ms.ops.cat((self.cls_token, x), axis=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: ms.Tensor,
            n=1,
    ):
        """ intermediate layers """
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: ms.Tensor,
            n=1,
            reshape: bool = False,
            return_class_token: bool = False,
            norm: bool = False,
    ):
        """ Intermediate layer accessor
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def construct_features(self, x):
        """ construct features """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def construct_head(self, x, pre_logits: bool = False):
        """ construct head """
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def construct(self, x):
        x = self.construct_features(x)
        x = self.construct_head(x)
        return x


class DistilledVisionTransformer(VisionTransformer):
    """ Distilled Vision Transformer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = ms.Parameter(ms.ops.zeros((1, 1, self.embed_dim)))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = ms.Parameter(ms.ops.zeros((1, num_patches + 2, self.embed_dim)))
        self.head_dist = nn.Dense(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.dist_token = initializer(TruncatedNormal(sigma=.02), self.dist_token.shape, self.dist_token.dtype)
        self.pos_embed = initializer(TruncatedNormal(sigma=.02), self.pos_embed.shape, self.pos_embed.dtype)
        self.head_dist.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weight """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def construct_features(self, x):
        """ construct features """
        x = self.patch_embed(x)

        x = ms.ops.cat((self.cls_token, self.dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def construct(self, x):
        x, x_dist = self.construct_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ deit-tiny-patch16 with image size 224 """
    default_cfg = _cfg()
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    """ deit-small-patch16 with image size 224 """
    default_cfg = _cfg()
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    """ deit-base-patch16 with image size 224 """
    default_cfg = _cfg()
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ deit-tiny-distilled-patch16 with image size 224 """
    default_cfg = _cfg()
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ deit-small-distilled-patch16 with image size 224 """
    default_cfg = _cfg()
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ deit-base-distilled-patch16 with image size 224 """
    default_cfg = _cfg()
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    """ deit-base-patch16 with image size 384 """
    default_cfg = _cfg()
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ deit-base-distilled-patch16 with image size 384 """
    default_cfg = _cfg()
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = deit_base_distilled_patch16_224()
    output = net(dummy_input)
    print(output.shape)
