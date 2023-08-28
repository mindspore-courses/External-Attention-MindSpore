# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `EfficientFormer`.
Refer to EfficientFormer: Vision Transformers at MobileNet Speed
"""
import itertools
import os

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.layers import to_2tuple, to_4tuple, DropPath
from model.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.registry import register_model
from model.helper import load_pretrained

EfficientFormer_width = {
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}

EfficientFormer_depth = {
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}


class Attention(nn.Cell):
    """ Attention """
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** 0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = nn.Dense(dim, h)
        self.proj = nn.Dense(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = ms.Parameter(ms.ops.zeros((num_heads, len(attention_offsets))))
        self.attention_bias_idxs = ms.Tensor(idxs, dtype=ms.intp).view(N, N)

        self.ab = self.attention_biases[:, self.attention_bias_idxs]
        if self.training:
            del self.ab

    def train(self, mode=True):
        """ train mode """
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
            self.ab.requires_grad = False

    def construct(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], axis=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = ((q @ k.transpose(0, 1, 3, 2)) * self.scale +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab))
        attn = ms.ops.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


def stem(in_chs, out_chs):
    """ stem """
    return nn.SequentialCell(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, pad_mode='pad', padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, pad_mode='pad', padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU()
    )


class Embedding(nn.Cell):
    """
        Patch Embedding that is implemented by a layer of conv.
        Input: tensor in shape [B, C, H, W]
        Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_4tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, pad_mode='pad', padding=padding)

        if isinstance(norm_layer, nn.BatchNorm2d):
            self.norm = norm_layer(embed_dim)
        elif isinstance(norm_layer, nn.LayerNorm):
            self.norm = norm_layer((embed_dim,))
        else:
            self.norm = nn.Identity()

    def construct(self, x):
        x = self.norm(self.proj(x))
        return x


class Flat(nn.Cell):
    """ Flat """
    def __init__(self, ):
        super().__init__()

    def construct(self, x):
        x = x.flatten(start_dim=2).transpose(0, 2, 1)
        return x


class Pooling(nn.Cell):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, pad_mode='pad', padding=pool_size // 2, count_include_pad=False)

    def construct(self, x):
        return self.pool(x) - x


class LinearMlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop2 = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class Mlp(nn.Cell):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(p=drop)
        self.apply(self._init_weights)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, cell):
        """ initialize weight """
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        x = self.act(self.norm1(self.fc1(x)))
        x = self.drop(x)
        x = self.drop(self.norm2(self.fc2(x)))
        return x


class Meta3D(nn.Cell):
    """ Meta3D """
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale1 = ms.Parameter(layer_scale_init_value * ms.ops.ones((dim,)), requires_grad=True)
            self.layer_scale2 = ms.Parameter(layer_scale_init_value * ms.ops.ones((dim,)), requires_grad=True)

    def construct(self, x):
        if self.use_layer_scale:
            x += self.drop_path(self.layer_scale1.unsqueeze(0).unsqueeze(0)) \
                 * self.token_mixer(self.norm1(x))
            x += self.drop_path(self.layer_scale2.unsqueeze(0).unsqueeze(0)) \
                 * self.mlp(self.norm2(x))
        else:
            x += self.drop_path(self.token_mixer(self.norm1(x)))
            x += self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Cell):
    """ Meta4D """
    def __init__(self, dim, pool_size=3, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale1 = ms.Parameter(layer_scale_init_value * ms.ops.ones((dim,)), requires_grad=True)
            self.layer_scale2 = ms.Parameter(layer_scale_init_value * ms.ops.ones((dim,)), requires_grad=True)

    def construct(self, x):
        if self.use_layer_scale:
            x += self.drop_path(self.layer_scale1.unsqueeze(-1).unsqueeze(-1)) \
                 * self.token_mixer(x)
            x += self.drop_path(self.layer_scale2.unsqueeze(-1).unsqueeze(-1)) \
                 * self.mlp(x)
        else:
            x += self.drop_path(self.token_mixer(self.norm1(x)))
            x += self.drop_path(self.mlp(self.norm2(x)))
        return x


def meta_blocks(dim, index, layers, pool_size=3, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0., use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    """ Meta blocks """
    blocks = []
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers)-1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(Meta3D(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, norm_layer=norm_layer, drop=drop_rate,
                                 drop_path=block_dpr, use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value))
        else:
            blocks.append(Meta4D(dim, pool_size=pool_size, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop_rate,
                                 drop_path=block_dpr, use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())
    blocks = nn.SequentialCell(*blocks)
    return blocks


class EfficientFormer(nn.Cell):
    """ EfficientFormer """
    def __init__(self, layers, embed_dims=None, mlp_ratio=4., downsamples=None, pool_size=3,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, num_classes=1000, down_patch_size=3,
                 down_stride=2, down_pad=1, drop_rate=0., drop_path_rate=0., use_layer_scale=True,
                 layer_scale_init_value=1e-5, fork_feat=False, vit_num=0, distillation=True):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers, pool_size=pool_size, mlp_ratio=mlp_ratio,
                                act_layer=act_layer, norm_layer=norm_layer, drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate, use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value, vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                network.append(Embedding(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                                         in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
        self.network = nn.SequentialCell(*network)

        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer((embed_dims[i_emb], ))
                layer_name = f'norm{i_layer}'
                self.insert_child_to_cell(layer_name, layer)
        else:
            self.norm = norm_layer((embed_dims[-1], ))
            self.head = nn.Dense(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Dense(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)

    def cls_init_weights(self, cell):
        """ cls initialize weight """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct_tokens(self, x):
        """ construct tokens """
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def construct(self, x):
        x = self.patch_embed(x)
        x = self.construct_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))

        return cls_out.unsqueeze(dim=0)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


@register_model
def efficientformer_l1(pretrained=False, **kwargs):
    """ efficientformer_l1 """
    default_cfg = _cfg(crop_pct=0.9)
    model = EfficientFormer(layers=EfficientFormer_depth['l1'],
                            embed_dims=EfficientFormer_width['l1'],
                            downsamples=[True, True, True, True],
                            vit_num=1,
                            **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def efficientformer_l3(pretrained=False, **kwargs):
    """ efficientformer_l3 """
    default_cfg = _cfg(crop_pct=0.9)
    model = EfficientFormer(layers=EfficientFormer_depth['l3'],
                            embed_dims=EfficientFormer_width['l3'],
                            downsamples=[True, True, True, True],
                            vit_num=4,
                            **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


@register_model
def efficientformer_l7(pretrained=False, **kwargs):
    """ efficientformer_l7 """
    default_cfg = _cfg(crop_pct=0.9)
    model = EfficientFormer(layers=EfficientFormer_depth['l7'],
                            embed_dims=EfficientFormer_width['l7'],
                            downsamples=[True, True, True, True],
                            vit_num=8,
                            **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg)

    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = efficientformer_l1()
    output = net(dummy_input)
    print(net)
    print(output[0].shape)
