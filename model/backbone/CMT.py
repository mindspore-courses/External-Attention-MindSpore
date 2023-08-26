# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `CMT`.
Refer to CMT: Convolutional Neural Networks Meet Vision Transformers
"""
import math
from functools import partial
from collections import OrderedDict
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal, HeNormal

from model.helper import load_pretrained
from model.layers import DropPath, to_2tuple
from model.registry import register_model
from model.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class SwishImplementation(nn.Cell):
    """A memory-efficient implementation of Swish function"""

    def construct(self, i):
        result = i * ms.ops.sigmoid(i)
        return result


class MemoryEfficientSwish(nn.Cell):
    """ memory efficient swish """
    def construct(self, x):
        return SwishImplementation()(x)


class Mlp(nn.Cell):
    """ mlp """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, has_bias=True),
            act_layer(),
            nn.BatchNorm2d(hidden_features),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1,
                              pad_mode='pad', padding=1, group=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features)
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, has_bias=True),
            nn.BatchNorm2d(out_features)
        )
        self.drop = nn.Dropout(p=drop)

    def construct(self, x, H, W):
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj_bn(self.proj_act(self.proj(x) + x))
        x = self.conv2(x)
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    """ attention """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Dense(dim, self.qk_dim, has_bias=qkv_bias)
        self.k = nn.Dense(dim, self.qk_dim, has_bias=qkv_bias)
        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.SequentialCell(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, group=dim, has_bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def construct(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale + relative_pos
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ block """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, pad_mode='pad', padding=1, group=dim)

    def construct(self, x, H, W, relative_pos):
        B, _, C = x.shape
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm((embed_dim,))

    def construct(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_dim=2).transpose(0, 2, 1)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class CMT(nn.Cell):
    """ cmt """
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=None, stem_channel=16,
                 fc_dim=1280,
                 num_heads=None, mlp_ratios=None, qkv_bias=True, qk_scale=None,
                 representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 depths=None, qk_ratio=1, sr_ratios=None, dp=0.1):
        super().__init__()
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [2, 2, 10, 2]
        if mlp_ratios is None:
            mlp_ratios = [3.6, 3.6, 3.6, 3.6]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        if embed_dims is None:
            embed_dims = [46, 92, 184, 368]

        self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)

        self.stem_conv1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=2,
                                    pad_mode='pad', padding=1, has_bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1,
                                    pad_mode='pad', padding=1, has_bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1,
                                    pad_mode='pad', padding=1, has_bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = ms.Parameter(ms.ops.randn(
            num_heads[0], self.patch_embed_a.num_patches,
            self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = ms.Parameter(ms.ops.randn(
            num_heads[1], self.patch_embed_b.num_patches,
            self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = ms.Parameter(ms.ops.randn(
            num_heads[2], self.patch_embed_c.num_patches,
            self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = ms.Parameter(ms.ops.randn(
            num_heads[3], self.patch_embed_d.num_patches,
            self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))

        dpr = list(ms.ops.linspace(0, drop_path_rate, sum(depths)))  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.SequentialCell([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.SequentialCell([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.SequentialCell([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.SequentialCell([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # Representation layer
        if representation_size:
            self.pre_logits = nn.SequentialCell(OrderedDict([
                ('fc', nn.Dense(self.embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._drop = nn.Dropout(p=dp)
        self.head = nn.Dense(fc_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(mode='fan_out'), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def update_temperature(self):
        """ update temperature """
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    def get_classifier(self):
        """ get classifier """
        return self.head

    def reset_classifier(self, num_classes):
        """ reset classifier """
        self.head = nn.Dense(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def construct_features(self, x):
        """ construct features """
        B = x.shape[0]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        x, (H, W) = self.patch_embed_a(x)
        for _, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, (H, W) = self.patch_embed_b(x)
        for _, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, (H, W) = self.patch_embed_c(x)
        for _, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x, (H, W) = self.patch_embed_d(x)
        for _, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)

        B, _, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)
        x = self._drop(x)
        x = self.pre_logits(x)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def resize_pos_embed(posemb, posemb_new):
    """ resize position embedding """
    ntok_new = posemb_new.shape[1]
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = ms.ops.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = ms.ops.cat([posemb_tok, posemb_grid], axis=1)
    return posemb


def _create_cmt_model(pretrained=False, **kwargs):
    default_cfg = _cfg()
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = CMT(img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(
            model, num_classes=num_classes, in_channels=kwargs.get('in_chans', 3))
    return model


@register_model
def cmt_tiny(pretrained=False, **kwargs):
    """
    CMT-Tiny
    """
    model_kwargs = {'qkv_bias': True,  **kwargs}
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def cmt_xs(pretrained=False, **kwargs):
    """
    CMT-XS: dim x 0.9, depth x 0.8, input 192
    """
    model_kwargs = {'qkv_bias': True, 'embed_dims': [52, 104, 208, 416], 'stem_channel': 16, 'num_heads': [1, 2, 4, 8],
                    'depths': [3, 3, 12, 3], 'mlp_ratios': [3.77, 3.77, 3.77, 3.77], 'qk_ratio': 1,
                    'sr_ratios': [8, 4, 2, 1], **kwargs}
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def cmt_small(pretrained=False, **kwargs):
    """
    CMT-Small
    """
    model_kwargs = {'qkv_bias': True, 'embed_dims': [64, 128, 256, 512], 'stem_channel': 32, 'num_heads': [1, 2, 4, 8],
                    'depths': [3, 3, 16, 3], 'mlp_ratios': [4, 4, 4, 4], 'qk_ratio': 1, 'sr_ratios': [8, 4, 2, 1],
                    **kwargs}
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


@register_model
def cmt_base(pretrained=False, **kwargs):
    """
    CMT-Base
    """
    model_kwargs = {'qkv_bias': True, 'embed_dims': [76, 152, 304, 608], 'stem_channel': 38, 'num_heads': [1, 2, 4, 8],
                    'depths': [4, 4, 20, 4], 'mlp_ratios': [4, 4, 4, 4], 'qk_ratio': 1, 'sr_ratios': [8, 4, 2, 1],
                    'dp': 0.3, **kwargs}
    model = _create_cmt_model(pretrained=pretrained, **model_kwargs)
    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    cmt = cmt_tiny()
    output = cmt(dummy_input)
    print(output.shape)
