# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `CrossViT`.
Refer to CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification
"""
import math
from functools import partial
import mindspore as ms
from mindspore import numpy as np
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal, Orthogonal


from model.layers import DropPath, to_2tuple
from model.helper import load_pretrained
from model.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


_model_urls = {
    'crossvit_15_224': '',
    'crossvit_15_dagger_224': '',
    'crossvit_15_dagger_384': '',
    'crossvit_18_224': '',
    'crossvit_18_dagger_224': '',
    'crossvit_18_dagger_384': '',
    'crossvit_9_224': '',
    'crossvit_9_dagger_224': '',
    'crossvit_base_224': '',
    'crossvit_small_224': '',
    'crossvit_tiny_224': '',
}


def get_sinusoid_encoding(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return ms.Tensor(sinusoid_table, dtype=ms.float32).unsqueeze(0)


class Token_transformer(nn.Cell):
    """ Token Transformer """
    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim,
                       act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Token_performer(nn.Cell):
    """ Token Performer """
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Dense(dim, 3 * self.emb)
        self.dp = nn.Dropout(p=dp1)
        self.proj = nn.Dense(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm((dim, ))
        self.norm2 = nn.LayerNorm((self.emb, ))
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.SequentialCell(
            nn.Dense(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Dense(1 * self.emb, self.emb),
            nn.Dropout(p=dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        # self.w = ms.ops.randn((self.m, self.emb))
        self.w = initializer(Orthogonal(), (self.m, self.emb))
        self.w = ms.Parameter(self.w * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        """ pre_exp """
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = x.float() @ self.w.T
        # wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return ms.ops.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        """ single attention """
        k, q, v = ms.ops.split(self.kqv(x), self.emb, axis=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = (qp @ kp.sum(axis=1).T).unsqueeze(axis=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = v.float().T @ kp  # (B, emb, m)
        y = qp @ kptv.T / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def construct(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class T2T(nn.Cell):
    """ Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, patch_size=16, tokens_type='transformer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if patch_size == 12:
            kernel_size = ((7, 4, 2), (3, 3, 1), (3, 1, 1))
        elif patch_size == 16:
            kernel_size = ((7, 4, 2), (3, 2, 1), (3, 2, 1))
        else:
            raise ValueError(f"Unknown patch size {patch_size}")

        self.soft_split0 = nn.Unfold(ksizes=to_2tuple(kernel_size[0][0]), strides=to_2tuple(kernel_size[0][1]),
                                     rates=to_2tuple(kernel_size[0][2]))
        self.soft_split1 = nn.Unfold(ksizes=to_2tuple(kernel_size[1][0]), strides=to_2tuple(kernel_size[1][1]),
                                     rates=to_2tuple(kernel_size[1][2]))
        self.soft_split2 = nn.Unfold(ksizes=to_2tuple(kernel_size[2][0]), strides=to_2tuple(kernel_size[2][1]),
                                     rates=to_2tuple(kernel_size[2][2]))

        if tokens_type == 'transformer':
            self.attention1 = Token_transformer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim,
                                                num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim,
                                                num_heads=1, mlp_ratio=1.0)
            self.project = nn.Dense(token_dim * (kernel_size[2][0] ** 2), embed_dim)

        elif tokens_type == 'performer':
            self.attention1 = Token_performer(dim=in_chans * (kernel_size[0][0] ** 2), in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * (kernel_size[1][0] ** 2), in_dim=token_dim,
                                              kernel_ratio=0.5)
            self.project = nn.Dense(token_dim * (kernel_size[2][0] ** 2), embed_dim)

        self.num_patches = (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1])) * \
                           (img_size // (kernel_size[0][1] * kernel_size[1][1] * kernel_size[2][1]))

    def construct(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(0, 2, 1).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(0, 2, 1, 3)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(0, 2, 1).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(0, 2, 1, 3)

        # final tokens
        x = self.project(x)

        return x


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
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Cell):
    """ Block """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, multi_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, pad_mode='pad', padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, pad_mode='pad', padding=1),
                )
            elif patch_size[0] == 16:
                self.proj = nn.SequentialCell(
                    nn.Conv2d(in_chans, embed_dim // 4, kernel_size=7, stride=4, pad_mode='pad', padding=3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, pad_mode='pad', padding=1),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, pad_mode='pad', padding=1),
                )
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def construct(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_dim=2).transpose(0, 2, 1)
        return x


class CrossAttention(nn.Cell):
    """ Cross Attention """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.wk = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.wv = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Cell):
    """ Cross Attention Block """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.norm1 = norm_layer((dim, ))
        self.attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer((dim, ))
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleBlock(nn.Cell):
    """ Multi-Scale Block """
    def __init__(self, dim, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        num_branches = len(dim)
        self.num_branches = num_branches
        # different branch could have different embedding size, the first one is the base
        self.blocks = nn.SequentialCell()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    Block(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.SequentialCell(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None

        self.projs = nn.SequentialCell()
        for d in range(num_branches):
            if dim[d] == dim[(d+1) % num_branches]:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer((dim[d], )), act_layer(), nn.Dense(dim[d], dim[(d+1) % num_branches])]
            self.projs.append(nn.SequentialCell(*tmp))

        self.fusion = nn.SequentialCell()
        for d in range(num_branches):
            d_ = (d+1) % num_branches
            nh = num_heads[d_]
            if depth[-1] == 0:  # backward capability:
                self.fusion.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d],
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                                                       attn_drop=attn_drop, drop_path=drop_path[-1],
                                                       norm_layer=norm_layer, has_mlp=False))
            else:
                tmp = []
                for _ in range(depth[-1]):
                    tmp.append(CrossAttentionBlock(dim=dim[d_], num_heads=nh, mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                                   drop_path=drop_path[-1], norm_layer=norm_layer, has_mlp=False))
                self.fusion.append(nn.SequentialCell(*tmp))

        self.revert_projs = nn.SequentialCell()
        for d in range(num_branches):
            if dim[(d+1) % num_branches] == dim[d]:
                tmp = [nn.Identity()]
            else:
                tmp = [norm_layer((dim[(d+1) % num_branches], )), act_layer(), nn.Dense(dim[(d+1) % num_branches],
                                                                                        dim[d])]
            self.revert_projs.append(nn.SequentialCell(*tmp))

    def construct(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]
        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]
        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = ms.ops.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), axis=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = ms.ops.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), axis=1)
            outs.append(tmp)
        return outs


def _compute_num_patches(img_size, patches):
    """ compute patches """
    return [i // p * i // p for i, p in zip(img_size, patches)]


class VisionTransformer(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=(224, 224), patch_size=(8, 16), in_chans=3, num_classes=1000, embed_dim=(192, 384),
                 depth=([1, 3, 1], [1, 3, 1], [1, 3, 1]), num_heads=(6, 12), mlp_ratio=(2., 2., 4.), qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None,
                 norm_layer=nn.LayerNorm, multi_conv=False):
        super().__init__()

        self.num_classes = num_classes
        if not isinstance(img_size, list):
            img_size = to_2tuple(img_size)
        self.img_size = img_size

        num_patches = _compute_num_patches(img_size, patch_size)
        self.num_branches = len(patch_size)

        self.patch_embed = nn.SequentialCell()
        if hybrid_backbone is None:
            self.pos_embed = [ms.Parameter(ms.ops.zeros((1, 1 + num_patches[i], embed_dim[i])))
                              for i in range(self.num_branches)]
            for im_s, p, d in zip(img_size, patch_size, embed_dim):
                self.patch_embed.append(PatchEmbed(img_size=im_s, patch_size=p, in_chans=in_chans, embed_dim=d,
                                                   multi_conv=multi_conv))
        else:
            self.pos_embed = nn.SequentialCell()
            # from .t2t import T2T, get_sinusoid_encoding
            tokens_type = 'transformer' if hybrid_backbone == 't2t' else 'performer'
            for idx, (im_s, p, d) in enumerate(zip(img_size, patch_size, embed_dim)):
                self.patch_embed.append(T2T(im_s, tokens_type=tokens_type, patch_size=p, embed_dim=d))
                self.pos_embed.append(ms.Parameter(get_sinusoid_encoding(n_position=1 + num_patches[idx],
                                                                         d_hid=embed_dim[idx]),
                                                   requires_grad=False))

            del self.pos_embed
            self.pos_embed = nn.SequentialCell([ms.Parameter(ms.ops.zeros((1, 1 + num_patches[i], embed_dim[i])))
                                               for i in range(self.num_branches)])

        self.cls_token = [ms.Parameter(ms.ops.zeros((1, 1, embed_dim[i])))
                          for i in range(self.num_branches)]
        self.pos_drop = nn.Dropout(p=drop_rate)

        total_depth = sum(sum(x[-2:]) for x in depth)
        dpr = list(ms.ops.linspace(0, drop_path_rate, total_depth))  # stochastic depth decay rule
        dpr_ptr = 0
        self.blocks = nn.SequentialCell()
        for idx, block_cfg in enumerate(depth):
            curr_depth = max(block_cfg[:-1]) + block_cfg[-1]
            dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
            blk = MultiScaleBlock(embed_dim, block_cfg, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                  drop_path=dpr_, norm_layer=norm_layer)
            dpr_ptr += curr_depth
            self.blocks.append(blk)

        self.norm = nn.SequentialCell([norm_layer((embed_dim[i], )) for i in range(self.num_branches)])
        self.head = nn.SequentialCell([nn.Dense(embed_dim[i], num_classes) if num_classes > 0 else nn.Identity()
                                       for i in range(self.num_branches)])

        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                self.pos_embed[i] = initializer(TruncatedNormal(sigma=.02),
                                                self.pos_embed[i].shape, self.pos_embed[i].dtype)
            self.cls_token[i] = initializer(TruncatedNormal(sigma=.02),
                                            self.cls_token[i].shape, self.cls_token[i].dtype)

        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weight """
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
        _, _, H, _ = x.shape
        xs = []
        for i in range(self.num_branches):
            x_ = ms.ops.interpolate(x, size=(self.img_size[i], self.img_size[i]), mode='bicubic')\
                if H != self.img_size[i] else x
            tmp = self.patch_embed[i](x_)
            tmp = ms.ops.cat((self.cls_token[i], tmp), axis=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.blocks:
            xs = blk(xs)

        xs = [self.norm[i](x) for i, x in enumerate(xs)]
        out = [x[:, 0] for x in xs]

        return out

    def construct(self, x):
        xs = self.construct_features(x)
        ce_logits = [self.head[i](x) for i, x in enumerate(xs)]
        ce_logits = ms.ops.mean(ms.ops.stack(ce_logits, axis=0), axis=0)
        return ce_logits


@register_model
def crossvit_tiny_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-tiny with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_tiny_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[96, 192], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[3, 3], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit_small_224(pretrained=False, num_classes: int = 1000, in_channels=3,  **kwargs):
    """ crossvit-small with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_small_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[6, 6], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit_base_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-base with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_base_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[384, 768], depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
                              num_heads=[12, 12], mlp_ratio=[4, 4, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit_9_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-9 with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_9_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit_15_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-15 with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_15_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit_18_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-18 with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_18_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def crossvit_9_dagger_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-9-dagger with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_9_dagger_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[128, 256], depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
                              num_heads=[4, 4], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model

@register_model
def crossvit_15_dagger_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-15-dagger with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_15_dagger_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model

@register_model
def crossvit_15_dagger_384(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-15-dagger with image size 384 """
    default_cfg = _cfg(_model_urls['crossvit_15_dagger_384'])
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[192, 384], depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
                              num_heads=[6, 6], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model

@register_model
def crossvit_18_dagger_224(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-18-dagger with image size 224 """
    default_cfg = _cfg(_model_urls['crossvit_18_dagger_224'])
    model = VisionTransformer(img_size=[240, 224],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model

@register_model
def crossvit_18_dagger_384(pretrained=False, num_classes: int = 1000, in_channels=3, **kwargs):
    """ crossvit-18-dagger with image size 384 """
    default_cfg = _cfg(_model_urls['crossvit_18_dagger_384'])
    model = VisionTransformer(img_size=[408, 384],
                              patch_size=[12, 16], embed_dim=[224, 448], depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
                              num_heads=[7, 7], mlp_ratio=[3, 3, 1], qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, epsilon=1e-6), multi_conv=True, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = crossvit_tiny_224()
    output = net(dummy_input)
    print(output.shape)
