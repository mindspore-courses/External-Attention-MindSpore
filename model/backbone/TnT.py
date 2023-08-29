# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `TnT`.
Refer to "Transformer in Transformer"
"""
import math
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.layers import DropPath, to_2tuple
from model.registry import register_model
from model.helper import load_pretrained
from model.backbone.resnet import BottleNeck, ResNet

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def resnet26(num_classes):
    """ resnet26 """
    return ResNet(BottleNeck, [2, 2, 2, 2], num_classes=num_classes)


def reset50(num_classes):
    """ resnet50 """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def make_divisible(v, divisor=8, min_value=None):
    """ make divisible """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class SE(nn.Cell):
    """ SE """
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.SequentialCell(
            nn.LayerNorm(dim),
            nn.Dense(dim, hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, dim),
            nn.Tanh()
        )

    def construct(self, x):
        a = x.mean(dim=1, keepdim=True)
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Cell):
    """ Attention """
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Dense(dim, hidden_dim * 2, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, _ = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ TNT Block
    """

    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer((inner_dim, ))
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer((inner_dim, ))
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer((num_words * inner_dim, ))
            self.proj = nn.Dense(num_words * inner_dim, outer_dim, has_bias=False)
            self.proj_norm2 = norm_layer((outer_dim, ))
        # Outer
        self.outer_norm1 = norm_layer((outer_dim, ))
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer((outer_dim, ))
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def construct(self, inner_tokens, outer_tokens):
        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))  # B*N, k*k, c
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))  # B*N, k*k, c
            B, N, _ = outer_tokens.shape
            outer_tokens[:, 1:] = outer_tokens[:, 1:] + self.proj_norm2(
                self.proj(self.proj_norm1(inner_tokens.reshape(B, N - 1, -1))))  # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens


class PatchEmbed(nn.Cell):
    """ Image to Visual Word Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, inner_dim=24, inner_stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)
        _patch_size = (1, ) + patch_size + (1, )
        self.unfold = nn.Unfold(ksizes=_patch_size, strides=_patch_size, rates=(1, 1, 1, 1))
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, pad_mode='pad', padding=3, stride=inner_stride)

    def construct(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B, Ck2, N
        x = x.transpose(0, 2, 1)
        x = x.reshape(B * self.num_patches, C, self.patch_size[0], self.patch_size[1])  # B*N, C, 16, 16
        x = self.proj(x)  # B*N, C, 8, 8
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(0, 2, 1)  # B*N, 8*8, C
        return x


class TNT(nn.Cell):
    """ TNT (Transformer in Transformer) for computer vision
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, outer_dim=768, inner_dim=48,
                 depth=12, outer_num_heads=12, inner_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, inner_stride=4, se=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer((num_words * inner_dim, ))
        self.proj = nn.Dense(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer((outer_dim, ))

        self.cls_token = ms.Parameter(ms.ops.zeros((1, 1, outer_dim)))
        self.outer_tokens = ms.Parameter(ms.ops.zeros((1, num_patches, outer_dim)), requires_grad=False)
        self.outer_pos = ms.Parameter(ms.ops.zeros((1, num_patches + 1, outer_dim)))
        self.inner_pos = ms.Parameter(ms.ops.zeros((1, num_words, inner_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = list(ms.ops.linspace(0, drop_path_rate, depth))  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.SequentialCell(blocks)
        self.norm = norm_layer((outer_dim, ))

        # Classifier head
        self.head = nn.Dense(outer_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.cls_token.set_data(initializer(TruncatedNormal(sigma=.02), self.cls_token.shape, self.cls_token.dtype))
        self.outer_tokens.set_data(initializer(TruncatedNormal(sigma=.02),
                                               self.outer_tokens.shape, self.outer_tokens.dtype))
        self.inner_pos.set_data(initializer(TruncatedNormal(sigma=.02), self.inner_pos.shape, self.inner_pos.dtype))
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weights """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def get_classifier(self):
        """ get classifier """
        return self.head

    def reset_classifier(self, num_classes):
        """ reset classifier """
        self.num_classes = num_classes
        self.head = nn.Dense(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct_features(self, x):
        """ construct features """
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C

        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        outer_tokens = ms.ops.cat((self.cls_token.broadcast_to((B, -1, -1)), outer_tokens), axis=1)

        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        outer_tokens = self.norm(outer_tokens)
        return outer_tokens[:, 0]

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def tnt_s_patch16_224(pretrained=False, **kwargs):
    """ tnt-s-patch16-224 """
    patch_size = 16
    inner_stride = 4
    outer_dim = 384
    inner_dim = 24
    outer_num_heads = 6
    inner_num_heads = 4
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, filter_fn=_conv_filter)
    return model


@register_model
def tnt_b_patch16_224(pretrained=False, **kwargs):
    """ tnt-b-patch16-224 """
    patch_size = 16
    inner_stride = 4
    outer_dim = 640
    inner_dim = 40
    outer_num_heads = 10
    inner_num_heads = 4
    outer_dim = make_divisible(outer_dim, outer_num_heads)
    inner_dim = make_divisible(inner_dim, inner_num_heads)
    model = TNT(img_size=224, patch_size=patch_size, outer_dim=outer_dim, inner_dim=inner_dim, depth=12,
                outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads, qkv_bias=False,
                inner_stride=inner_stride, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, filter_fn=_conv_filter)
    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = tnt_s_patch16_224()
    output = net(dummy_input)
    print(output.shape)
