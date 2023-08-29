# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `CeiT`.
Refer to "Incorporating Convolution Designs into Visual Transformers"
"""
import math
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.layers import DropPath, to_2tuple
from model.helper import load_pretrained
from model.registry import register_model

__all__ = [
    'ceit_tiny_patch16_224', 'ceit_small_patch16_224', 'ceit_base_patch16_224',
    'ceit_tiny_patch16_384', 'ceit_small_patch16_384',
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


class Image2Tokens(nn.Cell):
    """ image to tokens """
    def __init__(self, in_chans=3, out_chans=64, kernel_size=7, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride,
                              pad_mode='pad',  padding=kernel_size // 2, has_bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=1)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x


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


class LocallyEnhancedFeedForward(nn.Cell):
    """ locally enhanced feed forward """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            pad_mode='pad', padding=(kernel_size - 1) // 2, group=hidden_features
        )
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)

        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def construct(self, x):
        b, n, k = x.shape
        cls_token, tokens = ms.ops.split(x, [1, n - 1], axis=1)
        x = tokens.reshape(b, int(math.sqrt(n - 1)), int(math.sqrt(n - 1)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        tokens = x.flatten(start_dim=2).permute(0, 2, 1)
        out = ms.ops.cat((cls_token, tokens), axis=1)
        return out


class Attention(nn.Cell):
    """ self attention """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.attention_map = None

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionLCA(Attention):
    """ attention lca """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dim = dim
        self.qkv_bias = qkv_bias

    def construct(self, x):
        q_weight = self.qkv.weight[:self.dim, :]
        q_bias = None if not self.qkv_bias else self.qkv.bias[:self.dim]
        kv_weight = self.qkv.weight[self.dim:, :]
        kv_bias = None if not self.qkv_bias else self.qkv.bias[self.dim:]

        B, N, C = x.shape
        _, last_token = ms.ops.split(x, [N - 1, 1], axis=1)

        q = (last_token @ q_weight.T + q_bias) \
            .reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = (x @ kv_weight.T + kv_bias) \
            .reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        # self.attention_map = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ lca blocks """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, with_bn=True,
                 feedforward_type='leff'):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer((dim, ))
        self.feedforward_type = feedforward_type

        if feedforward_type == 'leff':
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.leff = LocallyEnhancedFeedForward(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                kernel_size=kernel_size, with_bn=with_bn,
            )
        else:  # LCA
            self.attn = AttentionLCA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.feedforward = Mlp(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
            )

    def construct(self, x):
        if self.feedforward_type == 'leff':
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.leff(self.norm2(x)))
            return x, x[:, 0]
        # LCA
        _, last_token = ms.ops.split(x, [x.shape[1] - 1, 1], axis=1)
        x = last_token + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.feedforward(self.norm2(x)))
        return x


class HybridEmbed(nn.Cell):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, patch_size=16, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Cell)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            # map for all networks, the feature metadata has reliable channel and stride info, but using
            # stride to calc feature dim requires info about padding of each stage that isn't captured.
            training = backbone.training
            if training:
                backbone.eval()
            o = self.backbone(ms.ops.zeros((1, in_chans, img_size[0], img_size[1])))
            if isinstance(o, (list, tuple)):
                o = o[-1]  # last feature if backbone outputs list/tuple of features
            feature_size = o.shape[-2:]
            feature_dim = o.shape[1]
            # backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = (feature_size[0] // patch_size) * (feature_size[1] // patch_size)
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(start_dim=2).transpose(0, 2, 1)
        return x


class CeIT(nn.Cell):
    """ CeIT """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=nn.LayerNorm,
                 leff_local_size=3,
                 leff_with_bn=True):
        """
        args:
            - img_size (:obj:`int`): input image size
            - patch_size (:obj:`int`): patch size
            - in_chans (:obj:`int`): input channels
            - num_classes (:obj:`int`): number of classes
            - embed_dim (:obj:`int`): embedding dimensions for tokens
            - depth (:obj:`int`): depth of encoder
            - num_heads (:obj:`int`): number of heads in multi-head self-attention
            - mlp_ratio (:obj:`float`): expand ratio in feedforward
            - qkv_bias (:obj:`bool`): whether to add bias for mlp of qkv
            - qk_scale (:obj:`float`): scale ratio for qk, default is head_dim ** -0.5
            - drop_rate (:obj:`float`): dropout rate in feedforward module after linear operation
                and projection drop rate in attention
            - attn_drop_rate (:obj:`float`): dropout rate for attention
            - drop_path_rate (:obj:`float`): drop_path rate after attention
            - hybrid_backbone (:obj:`nn.Module`): backbone e.g. resnet
            - norm_layer (:obj:`nn.Module`): normalization type
            - leff_local_size (:obj:`int`): kernel size in LocallyEnhancedFeedForward
            - leff_with_bn (:obj:`bool`): whether add bn in LocallyEnhancedFeedForward
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.i2t = HybridEmbed(
            hybrid_backbone, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.i2t.num_patches

        self.cls_token = ms.Parameter(ms.ops.zeros((1, 1, embed_dim)))
        self.pos_embed = ms.Parameter(ms.ops.zeros((1, num_patches + 1, embed_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = list(ms.ops.linspace(0, drop_path_rate, depth))  # stochastic depth decay rule
        self.blocks = nn.SequentialCell([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                kernel_size=leff_local_size, with_bn=leff_with_bn)
            for i in range(depth)])

        # without droppath
        self.lca = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0., norm_layer=norm_layer,
            feedforward_type='lca'
        )
        self.pos_layer_embed = ms.Parameter(ms.ops.zeros((1, depth, embed_dim)))

        self.norm = norm_layer((embed_dim, ))

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # Classifier head
        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = initializer(TruncatedNormal(sigma=.02), self.pos_embed.shape, self.pos_embed.dtype)
        self.cls_token = initializer(TruncatedNormal(sigma=.02), self.cls_token.shape, self.cls_token.dtype)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
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
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct_features(self, x):
        """ construct features """
        B = x.shape[0]
        x = self.i2t(x)

        cls_tokens = self.cls_token
        x = ms.ops.cat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        cls_token_list = []
        for blk in self.blocks:
            x, curr_cls_token = blk(x)
            cls_token_list.append(curr_cls_token)

        all_cls_token = ms.ops.stack(cls_token_list, axis=1)  # B*D*K
        all_cls_token = all_cls_token + self.pos_layer_embed
        # attention over cls tokens
        last_cls_token = self.lca(all_cls_token)
        last_cls_token = self.norm(last_cls_token)

        return last_cls_token.view(B, -1)

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


@register_model
def ceit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    default_cfg = _cfg(**kwargs)

    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t,
        patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)
    return model


@register_model
def ceit_small_patch16_224(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    default_cfg = _cfg()
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t,
        patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def ceit_base_patch16_224(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    default_cfg = _cfg()
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t,
        patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def ceit_tiny_patch16_384(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    default_cfg = _cfg()
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t, img_size=384,
        patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def ceit_small_patch16_384(pretrained=False, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    default_cfg = _cfg()
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t, img_size=384,
        patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    ceit = ceit_tiny_patch16_224()
    output = ceit(dummy_input)
    print(output.shape)
