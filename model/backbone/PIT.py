# pylint: disable=E0401
"""
MindSpore implementation of `PIT`.
Refer to Rethinking Spatial Dimensions of Vision Transformers
"""
import math

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.layers import DropPath
from model.registry import register_model


__all__ = [
    'pit_b', 'pit_s', 'pit_xs', 'pit_ti', 'pit_b_distilled', 'pit_s_distilled', 'pit_xs_distilled', 'pit_ti_distilled'
]


class Mlp(nn.Cell):
    """ Mlp """
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


class Block(nn.Cell):
    """ Block """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Cell):
    """ Attention """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer(nn.Cell):
    """ Transformer layer """
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super().__init__()
        self.layers = nn.SequentialCell([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.SequentialCell([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=nn.LayerNorm
            )
            for i in range(depth)])

    def construct(self, x, cls_tokens):
        b, c, h, w = x.shape
        # h, w = x.shape[2:4]
        x = x.transpose(0, 2, 3, 1).reshape(b, -1, c)
        # x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = ms.ops.cat((cls_tokens, x), axis=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.reshape(b, h, w, c).transpose(0, 3, 1, 2)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Cell):
    """ conv head pooling """
    def __init__(self, in_feature, out_feature, stride):
        super().__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1, pad_mode='pad',
                              padding=stride // 2, stride=stride, group=in_feature)
        self.fc = nn.Dense(in_feature, out_feature)

    def construct(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Cell):
    """ conv embedding """
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, pad_mode='pad', padding=padding, has_bias=True)

    def construct(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Cell):
    """ Pooling Transformer """
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super().__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = ms.Parameter(
            ms.ops.randn((1, base_dims[0] * heads[0], width, width)),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.cls_token = ms.Parameter(
            ms.ops.randn((1, 1, base_dims[0] * heads[0])),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.SequentialCell([])
        self.pools = nn.SequentialCell([])

        for stage, d in enumerate(depth):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + d)]
            block_idx += d

            self.transformers.append(
                Transformer(base_dims[stage], d, heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm((base_dims[-1] * heads[-1], ), epsilon=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Dense(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        self.pos_embed = initializer(TruncatedNormal(sigma=.02), self.pos_embed.shape, self.pos_embed.dtype)
        self.cls_token = initializer(TruncatedNormal(sigma=.02), self.cls_token.shape, self.cls_token.dtype)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weights """
        if isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def get_classifier(self):
        """ get classifier """
        return self.head

    def reset_classifier(self, num_classes):
        """ reset classifier """
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Dense(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def construct_features(self, x):
        """ construct features """
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token

        for stage, pool in enumerate(self.pools):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = pool(x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def construct(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    """ Distilled Pooling Transformer """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = ms.Parameter(
            ms.ops.randn((1, 2, self.base_dims[0] * self.heads[0])),
            requires_grad=True)
        if self.num_classes > 0:
            self.head_dist = nn.Dense(self.base_dims[-1] * self.heads[-1],
                                       self.num_classes)
        else:
            self.head_dist = nn.Identity()

        self.cls_token = initializer(TruncatedNormal(sigma=.02), self.cls_token.shape, self.cls_token.dtype)
        self.head_dist.apply(self._init_weights)

    def construct(self, x):
        cls_token = self.construct_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        return (x_cls + x_dist) / 2


@register_model
def pit_b(**kwargs):
    """ pit-b """
    return PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_s(**kwargs):
    """ pit-s """
    return PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_xs(**kwargs):
    """ pit-xs """
    return PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_ti(**kwargs):
    """ pit-ti """
    return PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_b_distilled(**kwargs):
    """ pit-b-distilled """
    return DistilledPoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_s_distilled(**kwargs):
    """ pit-s-distilled """
    return DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_xs_distilled(**kwargs):
    """ pit-xs-distilled """
    return DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs)


@register_model
def pit_ti_distilled(**kwargs):
    """ pit-ti-distilled """
    return DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs)


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    model = pit_ti_distilled()
    output = model(dummy_input)
    print(model)
    print(output.shape)
