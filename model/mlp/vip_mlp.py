# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of 'vip_mlp'
Refer to "Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"
"""
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal, initializer

from model.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.layers import DropPath
from model.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ViP_S': _cfg(crop_pct=0.9),
    'ViP_M': _cfg(crop_pct=0.9),
    'ViP_L': _cfg(crop_pct=0.875),
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
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.act(self.fc2(x)))


class WeightedPermuteMLP(nn.Cell):
    """ weighted permute mlp """
    def __init__(self, dim, segment_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim
        self.mlp_c = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.mlp_h = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.mlp_w = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.flatten = nn.Flatten(2, 3)

        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
        w = self.mlp_h(h).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = self.flatten((h + w + c).permute(0, 3, 1, 2)).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1)
        a = ms.ops.softmax(a, axis=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        return self.proj_drop(self.proj(x))


class PermutatorBlock(nn.Cell):
    """ permutator block """
    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=False, attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = mlp_fn(dim, segment_dim, qkv_bias, attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        return x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam


class PatchEmbed(nn.Cell):
    """ patch embed """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        return self.proj(x)


class Downsample(nn.Cell):
    """ downsample """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        return self.proj(x)


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=False, attn_drop=0,
                 drop_path_rate=0., skip_lam=1.0, mlp_fn=WeightedPermuteMLP):
    """ basic blocks """
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio, qkv_bias,
                                      attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.SequentialCell(*blocks)
    return blocks


class VisionPermutator(nn.Cell):
    """ vision permutator """
    def __init__(self, layers, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, skip_lam=skip_lam)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))

        self.network = nn.SequentialCell(network)
        self.norm = norm_layer((embed_dims[-1],))

        self.head = nn.Dense(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weights """
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(0.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        if isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('zeros', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('ones', cell.beta.shape, cell.beta.dtype))

    def forward_embeddings(self, x):
        """ forward embeddings """
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        """ forward tokens """
        for _, block in enumerate(self.network):
            x = block(x)
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def construct(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))

@register_model
def vip_s14(**kwargs):
    """ vip s14 """
    layers = [4, 3, 8, 3]
    transitions = [False, False, False, False]
    segment_dim = [16, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [384, 384, 384, 384]
    model = VisionPermutator(layers=layers, embed_dims=embed_dims, patch_size=14, transitions=transitions,
                             segment_dim=segment_dim, mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['ViP_S']
    return model

@register_model
def vip_s7(**kwargs):
    """ vip s7 """
    layers = [4, 3, 8, 3]
    transitions = [True, False, False, False]
    segment_dim = [32, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [192, 384, 384, 384]
    model = VisionPermutator(layers=layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                             segment_dim=segment_dim, mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['ViP_S']
    return model

@register_model
def vip_m7(**kwargs):
    """ vip m7 """
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    segment_dim = [32, 32, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 256, 512, 512]
    model = VisionPermutator(layers=layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                             segment_dim=segment_dim, mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['ViP_M']
    return model

@register_model
def vip_l7(**kwargs):
    """ vip l7 """
    layers = [8, 8, 16, 4]
    transitions = [True, False, False, False]
    segment_dim = [32, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 512, 512, 512]
    model = VisionPermutator(layers=layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                             segment_dim=segment_dim, mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['ViP_L']
    return model


if __name__ == "__main__":
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    vip = vip_s14()
    output = vip(dummy_input)
    print(output.shape)
