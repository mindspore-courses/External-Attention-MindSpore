# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `CaiT`.
Refer to Class-Attention in Image Transformers
"""
from functools import partial

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.layers import DropPath, to_4tuple, to_2tuple
from model.registry import register_model
from model.helper import load_pretrained


__all__ = [
    'cait_M48', 'cait_M36',
    'cait_S36', 'cait_S24', 'cait_S24_224',
    'cait_XS24', 'cait_XXS24', 'cait_XXS24_224',
    'cait_XXS36', 'cait_XXS36_224'
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        # 'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Cell):
    """ mlp """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_4tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_dim=2).transpose(0, 2, 1)
        return x


class Class_Attention(nn.Cell):
    """
    Class Attention
    refer to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications to do CA
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(0, 1, 3, 2)
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScale_Block_CA(nn.Cell):
    """
    Layer Scale Block and CA
    refer to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications to add CA and LayerScale
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = ms.Parameter(init_values * ms.ops.ones((dim, )), requires_grad=True)
        self.gamma_2 = ms.Parameter(init_values * ms.ops.ones((dim, )), requires_grad=True)

    def construct(self, x, x_cls):
        u = ms.ops.cat((x_cls, x), axis=1)

        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls


class Attention_talking_head(nn.Cell):
    """
    Attention talking head
    refer to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)

        self.proj = nn.Dense(dim, dim)

        self.proj_l = nn.Dense(num_heads, num_heads)
        self.proj_w = nn.Dense(num_heads, num_heads)

        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(0, 1, 3, 2)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = ms.ops.softmax(attn, axis=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Cell):
    """
    Layer Scale Block
    refer to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications to add layerScale
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention_talking_head,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = ms.Parameter(init_values * ms.ops.ones((dim, )), requires_grad=True)
        self.gamma_2 = ms.Parameter(init_values * ms.ops.ones((dim, )), requires_grad=True)

    def construct(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CaiT(nn.Cell):
    """
    CaiT
    refer to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications to adapt to our cait models
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, block_layers=LayerScale_Block,
                 block_layers_token=LayerScale_Block_CA,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention_talking_head, Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=Class_Attention,
                 Mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_clstk=4.0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = ms.Parameter(ms.ops.zeros((1, 1, embed_dim)))
        self.pos_embed = ms.Parameter(ms.ops.zeros((1, num_patches, embed_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.SequentialCell([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.SequentialCell([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale)
            for i in range(depth_token_only)])

        self.norm = norm_layer((embed_dim, ))

        self.feature_info = [{"num_chs": embed_dim, "reduction": 0, "module": 'head'}]
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = initializer(TruncatedNormal(sigma=.02), self.pos_embed.shape, self.pos_embed.dtype)
        self.cls_token = initializer(TruncatedNormal(sigma=.02), self.cls_token.shape, self.cls_token.dtype)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ init weights """
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

        cls_tokens = self.cls_token
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        for _, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = ms.ops.cat((cls_tokens, x), axis=1)

        x = self.norm(x)
        return x[:, 0]

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


@register_model
def cait_XXS24_224(pretrained=False, **kwargs):
    """ cait_XXS24 with the image size 224 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=224, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_XXS24(pretrained=False, **kwargs):
    """ cait_XXS24 with the image size 384 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=384, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_XXS36_224(pretrained=False, **kwargs):
    """ cait_XXS36 with the image size 224 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=224, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_XXS36(pretrained=False, **kwargs):
    """ cait_XXS24 with the image size 384 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=384, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_XS24(pretrained=False, **kwargs):
    """ cait_XS24 with the image size 384 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=384, patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_S24_224(pretrained=False, **kwargs):
    """ cait_S24 with the image size 224 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=224, patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_S24(pretrained=False, **kwargs):
    """ cait_XXS24 with the image size 384 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=384, patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_S36(pretrained=False, **kwargs):
    """ cait_S36 with the image size 384 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=384, patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-6,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_M36(pretrained=False, **kwargs):
    """ cait_M36 with the image size 384 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        init_scale=1e-6,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def cait_M48(pretrained=False, **kwargs):
    """ cait_M48 with the image size 448 """
    default_cfg = _cfg()
    model = CaiT(
        img_size=448, patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm,
        init_scale=1e-6,
        depth_token_only=2, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(model, default_cfg)

    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    cait = cait_XXS24_224()
    output = cait(dummy_input)
    print(output.shape)
