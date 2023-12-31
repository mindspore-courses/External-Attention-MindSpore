# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `VOLO`.
Refer to " VOLO: Vision Outlooker for Visual Recognition"
"""
import math
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.layers import DropPath, to_2tuple
from model.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'volo': _cfg(crop_pct=0.96),
    'volo_large': _cfg(crop_pct=1.15),
}


class OutlookAttention(nn.Cell):
    """ OutlookAttention """
    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim ** -0.5

        self.v = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn = nn.Dense(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        unfold_kernel_size = (1, ) + to_2tuple(kernel_size) + (1, )
        unfold_stride = (1, ) + to_2tuple(stride) + (1, )
        self.unfold = nn.Unfold(ksizes=unfold_kernel_size, rates=(1, 1, 1, 1), padding="same", strides=unfold_stride)
        self.pool = nn.AvgPool2d(kernel_size=to_2tuple(stride), stride=to_2tuple(stride), ceil_mode=True)

    def construct(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = ms.ops.fold(x, output_size=ms.Tensor((H, W)), kernel_size=self.kernel_size,
                        padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Cell):
    """
    Implementation of outlooker layer: which includes outlook attention + MLP
    Outlooker is the first stage in our VOLO
    --dim: hidden dim
    --num_heads: number of heads
    --mlp_ratio: mlp ratio
    --kernel_size: kernel size in each window for outlook attention
    return: outlooker layer
    """

    def __init__(self, dim, kernel_size, padding, stride=1,
                 num_heads=1, mlp_ratio=3., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
                                     padding=padding, stride=stride,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Cell):
    """Implementation of MLP"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
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


class Attention(nn.Cell):
    """ Implementation of self-attention
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, H, W, C = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Cell):
    """
    Implementation of Transformer,
    Transformer is the second stage in our VOLO
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ClassAttention(nn.Cell):
    """
    Class attention layer from CaiT, see details in CaiT
    Class attention is the post stage in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Dense(dim,
                           self.head_dim * self.num_heads * 2,
                           has_bias=qkv_bias)
        self.q = nn.Dense(dim, self.head_dim * self.num_heads, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, _ = x.shape

        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        cls_embed = (attn @ v).transpose(0, 2, 1, 3).reshape(
            B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_drop(cls_embed)
        return cls_embed


class ClassBlock(nn.Cell):
    """
    Class attention block from CaiT, see details in CaiT
    We use two-layers class attention in our VOLO, which is optional.
    """

    def __init__(self, dim, num_heads, head_dim=None, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = ClassAttention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def construct(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
        cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
        return ms.ops.cat([cls_embed, x[:, 1:]], axis=1)


def get_block(block_type, **kwargs):
    """
    get block by name, specifically for class attention block in here
    """
    if block_type == 'ca':
        return ClassBlock(**kwargs)
    raise TypeError("Error block type")


def rand_bbox(size, lam, scale=1):
    """
    get bounding box as token labeling (https://github.com/zihangJiang/TokenLabeling)
    return: bounding box
    """
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PatchEmbed(nn.Cell):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
                 patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]

        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.SequentialCell(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                          pad_mode='pad', padding=3, has_bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          pad_mode='pad', padding=1, has_bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          pad_mode='pad', padding=1, has_bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            )

        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def construct(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Cell):
    """
    Image to Patch Embedding, downsampling between stage1 and stage2
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def outlooker_blocks(block_fn, index, dim, layers, num_heads=1, kernel_size=3,
                     padding=1, stride=1, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                     attn_drop=0, drop_path_rate=0.):
    """
    generate outlooker layer in stage1
    return: outlooker layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(block_fn(dim, kernel_size=kernel_size, padding=padding,
                               stride=stride, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                               drop_path=block_dpr))

    blocks = nn.SequentialCell(*blocks)

    return blocks


def transformer_blocks(block_fn, index, dim, layers, num_heads, mlp_ratio=3.,
                       qkv_bias=False, qk_scale=None, attn_drop=0,
                       drop_path_rate=0.):
    """
    generate transformer layers in stage2
    return: transformer layers
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx +
                                      sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            block_fn(dim, num_heads,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=block_dpr))

    blocks = nn.SequentialCell(*blocks)

    return blocks


class VOLO(nn.Cell):
    """
    Vision Outlooker, the main class of our model
    --layers: [x,x,x,x], four blocks in two stages, the first block is outlooker, the
              other three are transformer, we set four blocks, which are easily
              applied to downstream tasks
    --img_size, --in_chans, --num_classes: these three are very easy to understand
    --patch_size: patch_size in outlook attention
    --stem_hidden_dim: hidden dim of patch embedding, d1-d4 is 64, d5 is 128
    --embed_dims, --num_heads: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios, --qkv_bias, --qk_scale, --drop_rate: easy to undertand
    --attn_drop_rate, --drop_path_rate, --norm_layer: easy to undertand
    --post_layers: post layers like two class attention layers using [ca, ca],
                  if yes, return_mean=False
    --return_mean: use mean of all feature tokens for classification, if yes, no class token
    --return_dense: use token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --mix_token: mixing tokens as token labeling, details are here:
                    https://github.com/zihangJiang/TokenLabeling
    --pooling_scale: pooling_scale=2 means we downsample 2x
    --out_kernel, --out_stride, --out_padding: kerner size,
                                               stride, and padding for outlook attention
    """

    def __init__(self, layers, img_size=224, in_chans=3, num_classes=1000, patch_size=8,
                 stem_hidden_dim=64, embed_dims=None, num_heads=None, downsamples=None,
                 outlook_attention=None, mlp_ratios=None, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 post_layers=None, return_mean=False, return_dense=True, mix_token=True,
                 pooling_scale=2, out_kernel=3, out_stride=2, out_padding=1):

        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(stem_conv=True, stem_stride=2, patch_size=patch_size,
                                      in_chans=in_chans, hidden_dim=stem_hidden_dim,
                                      embed_dim=embed_dims[0])

        # inital positional encoding, we add positional encoding after outlooker blocks
        self.pos_embed = ms.Parameter(
            ms.ops.zeros((1, img_size // patch_size // pooling_scale,
                          img_size // patch_size // pooling_scale,
                          embed_dims[-1])))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            if outlook_attention[i]:
                # stage 1
                stage = outlooker_blocks(Outlooker, i, embed_dims[i], layers, num_heads=num_heads[i],
                                         kernel_size=out_kernel, stride=out_stride,
                                         padding=out_padding, mlp_ratio=mlp_ratios[i],
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_rate)
                network.append(stage)
            else:
                # stage 2
                stage = transformer_blocks(Transformer, i, embed_dims[i], layers,
                                           num_heads[i], mlp_ratio=mlp_ratios[i],
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop_path_rate=drop_path_rate,
                                           attn_drop=attn_drop_rate)
                network.append(stage)

            if downsamples[i]:
                # downsampling between two stages
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], 2))

        self.network = nn.SequentialCell(network)

        # set post block, for example, class attention layers
        self.post_network = None
        if post_layers is not None:
            self.post_network = nn.SequentialCell([
                get_block(post_layers[i],
                          dim=embed_dims[-1],
                          num_heads=num_heads[-1],
                          mlp_ratio=mlp_ratios[-1],
                          qkv_bias=qkv_bias,
                          qk_scale=qk_scale,
                          attn_drop=attn_drop_rate,
                          drop_path=0.,
                          norm_layer=norm_layer)
                for i in range(len(post_layers))
            ])
            self.cls_token = ms.Parameter(ms.ops.zeros((1, 1, embed_dims[-1])))
            self.cls_token = initializer(TruncatedNormal(sigma=.02), self.cls_token.shape, self.cls_token.dtype)

        # set output type
        self.return_mean = return_mean  # if yes, return mean, not use class token
        self.return_dense = return_dense  # if yes, return class token and all feature tokens
        if return_dense:
            assert not return_mean, "cannot return both mean and dense"
        self.mix_token = mix_token
        self.pooling_scale = pooling_scale
        if mix_token:  # enable token mixing, see token labeling for details.
            self.beta = 1.0
            assert return_dense, "return all tokens if mix_token is enabled"
        if return_dense:
            self.aux_head = nn.Dense(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        self.norm = norm_layer((embed_dims[-1], ))

        # Classifier head
        self.head = nn.Dense(
            embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.pos_embed = initializer(TruncatedNormal(sigma=.02), self.pos_embed.shape, self.pos_embed.dtype)
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
        self.head = nn.Dense(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct_embeddings(self, x):
        """ construct embedding """
        # patch embedding
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def construct_tokens(self, x):
        """ construct tokens """
        for idx, block in enumerate(self.network):
            if idx == 2:  # add positional encoding after outlooker blocks
                x = x + self.pos_embed
                x = self.pos_drop(x)
            x = block(x)

        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def construct_cls(self, x):
        """ construct cls """
        x = ms.ops.cat((self.cls_token, x), axis=1)
        for block in self.post_network:
            x = block(x)
        return x

    def construct(self, x):
        # step1: patch embedding
        x = self.construct_embeddings(x)

        # mix token, see token labeling for details.
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                2] // self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
            temp_x = x.clone()
            sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale * bbx1, self.pooling_scale * bby1, \
                                         self.pooling_scale * bbx2, self.pooling_scale * bby2
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        # step2: tokens learning in the two stages
        x = self.construct_tokens(x)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.construct_cls(x)
        x = self.norm(x)

        if self.return_mean:  # if no class token, return mean
            return self.head(x.mean(1))

        x_cls = self.head(x[:, 0])
        if not self.return_dense:
            return x_cls

        x_aux = self.aux_head(
            x[:, 1:]
        )  # generate classes in all feature tokens, see token labeling

        if not self.training:
            return x_cls + 0.5 * x_aux.max(1)[0]

        if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

            temp_x = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x

            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

        # return these: 1. class token, 2. classes from all feature tokens, 3. bounding box
        return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)


@register_model
def volo_d1(**kwargs):
    """
    VOLO-D1 model, Params: 27M
    --layers: [x,x,x,x], four blocks in two stages, the first stage(block) is outlooker,
            the other three blocks are transformer, we set four blocks, which are easily
             applied to downstream tasks
    --embed_dims, --num_heads,: embedding dim, number of heads in each block
    --downsamples: flags to apply downsampling or not in four blocks
    --outlook_attention: flags to apply outlook attention or not
    --mlp_ratios: mlp ratio in four blocks
    --post_layers: post layers like two class attention layers using [ca, ca]
    See detail for all args in the class VOLO()
    """
    layers = [4, 4, 8, 2]  # num of layers in the four blocks
    embed_dims = [192, 384, 384, 384]
    num_heads = [6, 12, 12, 12]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]  # do downsampling after first block
    outlook_attention = [True, False, False, False]
    # first block is outlooker (stage1), the other three are transformer (stage2)
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model


@register_model
def volo_d2(**kwargs):
    """
    VOLO-D2 model, Params: 59M
    """
    layers = [6, 4, 10, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model


@register_model
def volo_d3(**kwargs):
    """
    VOLO-D3 model, Params: 86M
    """
    layers = [8, 8, 16, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model


@register_model
def volo_d4(**kwargs):
    """
    VOLO-D4 model, Params: 193M
    """
    layers = [8, 8, 16, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)

    model.default_cfg = default_cfgs['volo_large']
    return model


@register_model
def volo_d5(**kwargs):
    """
    VOLO-D5 model, Params: 296M
    stem_hidden_dim=128, the dim in patch embedding is 128 for VOLO-D5
    """
    layers = [12, 12, 20, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 stem_hidden_dim=128,
                 **kwargs)
    model.default_cfg = default_cfgs['volo_large']
    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn(1, 3, 224, 224)
    net = volo_d1()
    output = net(dummy_input)
    print(output.shape)
