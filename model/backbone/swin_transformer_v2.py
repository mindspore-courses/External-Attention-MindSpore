# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `swin transformer v2`.
Refer to "Swin Transformer V2: Scaling Up Capacity and Resolution"
"""
import mindspore as ms
from mindspore import nn
from mindspore import numpy as np
from mindspore.common.initializer import initializer, TruncatedNormal

from model.layers import DropPath, to_2tuple
from model.helper import build_model_with_cfg
from model.registry import register_model


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).view(B, H, W, -1)
    return x


class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=None):

        super().__init__()
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = ms.Parameter(ms.ops.log(10 * ms.ops.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.SequentialCell(nn.Dense(2, 512, has_bias=True),
                                     nn.ReLU(),
                                     nn.Dense(512, num_heads, has_bias=False))

        # get relative_coords_table
        relative_coords_h = ms.ops.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=ms.float32)
        relative_coords_w = ms.ops.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=ms.float32)
        relative_coords_table = ms.ops.stack(
            ms.ops.meshgrid(relative_coords_h,
                            relative_coords_w)).permute(1, 2, 0).unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = ms.ops.sign(relative_coords_table) * ms.ops.log2(
            ms.ops.abs(relative_coords_table) + 1.0) / np.log2(ms.Tensor(8, dtype=ms.float32))

        self.relative_coords_table = ms.Parameter(relative_coords_table,
                                                  name="relative_coords_table",
                                                  requires_grad=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = ms.ops.arange(self.window_size[0])
        coords_w = ms.ops.arange(self.window_size[1])
        coords = ms.ops.stack(ms.ops.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = ms.ops.flatten(coords, start_dim=1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.relative_position_index = ms.Parameter(relative_position_index,
                                                    name="relative_position_index",
                                                    requires_grad=False)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=False)
        if qkv_bias:
            self.q_bias = ms.Parameter(ms.ops.zeros(dim))
            self.v_bias = ms.Parameter(ms.ops.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.normalize = ms.ops.LpNorm(axis=-1, keep_dims=True)

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = ms.ops.cat((self.q_bias, ms.ops.zeros_like(self.v_bias), self.v_bias))
        qkv = ms.ops.dense(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (self.normalize(q) @ self.normalize(k).transpose(0, 1, 3, 2))

        logit_scale = ms.ops.clamp(self.logit_scale, max=ms.ops.log(ms.Tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * ms.ops.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 1, 3, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Cell):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer((dim, ))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(float(drop_path)) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = ms.ops.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = ms.Parameter(attn_mask, name="attn_mask", requires_grad=False)
        else:
            self.attn_mask = None

    def construct(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = ms.ops.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = ms.ops.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Cell):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Dense(4 * dim, 2 * dim, has_bias=False)
        self.norm = norm_layer((2 * dim, ))

    def construct(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = ms.ops.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Cell):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.SequentialCell([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])
        self._init_respostnorm()

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            blk.norm1.gamma.set_data(initializer('ones', blk.norm1.gamma.shape, blk.norm1.gamma.dtype))
            blk.norm1.beta.set_data(initializer('zeros', blk.norm1.beta.shape, blk.norm1.beta.dtype))
            blk.norm2.gamma.set_data(initializer('ones', blk.norm2.gamma.shape, blk.norm2.gamma.dtype))
            blk.norm2.beta.set_data(initializer('zeros', blk.norm2.beta.shape, blk.norm2.beta.dtype))


class PatchEmbed(nn.Cell):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer((embed_dim, ))
        else:
            self.norm = None

    def construct(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_dim=2).transpose(0, 2, 1)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Cell):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=None, num_heads=None,
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=None):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if pretrained_window_sizes is None:
            pretrained_window_sizes = [0, 0, 0, 0]

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = ms.Parameter(ms.ops.zeros((1, num_patches, embed_dim)))
            self.absolute_pos_embed = initializer(TruncatedNormal(sigma=.02),
                                                  self.absolute_pos_embed.shape,
                                                  self.absolute_pos_embed.dtype)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x for x in ms.ops.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.SequentialCell()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer((self.num_features, ))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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

    def construct_features(self, x):
        """ construct features """
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(0, 2, 1))  # B C 1
        x = ms.ops.flatten(x, start_dim=1)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def _create_swin_transformer_v2(**kwargs):
    """ create swin transformer v2 """
    return build_model_with_cfg(SwinTransformerV2, **kwargs)


@register_model
def swinv2_tiny_window16_256(**kwargs):
    """ swinv2_tiny_window16_256 """
    model_kwargs = {"window_size": 16, "embed_dim": 96, "depths": (2, 2, 6, 2), "num_heads": (3, 6, 12, 24), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_tiny_window8_256(**kwargs):
    """ swinv2_tiny_window8_256 """
    model_kwargs = {"window_size": 8, "embed_dim": 96, "depths": (2, 2, 6, 2), "num_heads": (3, 6, 12, 24), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_small_window16_256(**kwargs):
    """ swinv2_small_window16_256 """
    model_kwargs = {"window_size": 16, "embed_dim": 96, "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_small_window8_256(**kwargs):
    """ swinv2_small_window8_256 """
    model_kwargs = {"window_size": 8, "embed_dim": 96, "depths": (2, 2, 18, 2), "num_heads": (3, 6, 12, 24), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_base_window16_256(**kwargs):
    """ swinv2_base_window16_256 """
    model_kwargs = {"window_size": 16, "embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_base_window8_256(**kwargs):
    """ swinv2_base_window8_256 """
    model_kwargs = {"window_size": 8, "embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_base_window12_192_22k(**kwargs):
    """ swinv2_base_window12_192_22k """
    model_kwargs = {"window_size": 12, "embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_base_window12to16_192to256_22kft1k(**kwargs):
    """ swinv2_base_window12to16_192to256_22kft1k """
    model_kwargs = {"window_size": 16, "embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32),
                    "pretrained_window_sizes": (12, 12, 12, 6), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_base_window12to24_192to384_22kft1k(**kwargs):
    """ swinv2_base_window12to24_192to384_22kft1k """
    model_kwargs = {"window_size": 24, "embed_dim": 128, "depths": (2, 2, 18, 2), "num_heads": (4, 8, 16, 32),
                    "pretrained_window_sizes": (12, 12, 12, 6), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_large_window12_192_22k(**kwargs):
    """ swinv2_large_window12_192_22k """
    model_kwargs = {"window_size": 12, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48),
                    **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_large_window12to16_192to256_22kft1k(**kwargs):
    """ swinv2_large_window12to16_192to256_22kft1k """
    model_kwargs = {"window_size": 16, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48),
                    "pretrained_window_sizes": (12, 12, 12, 6), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


@register_model
def swinv2_large_window12to24_192to384_22kft1k(**kwargs):
    """ swinv2_large_window12to24_192to384_22kft1k """
    model_kwargs = {"window_size": 24, "embed_dim": 192, "depths": (2, 2, 18, 2), "num_heads": (6, 12, 24, 48),
                    "pretrained_window_sizes": (12, 12, 12, 6), **kwargs}
    return _create_swin_transformer_v2(**model_kwargs)


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = SwinTransformerV2()
    output = net(dummy_input)
    print(output.shape)
