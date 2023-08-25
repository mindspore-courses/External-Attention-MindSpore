"""
MindSpore implementation of 'Crossformer'
Refer to "CROSSFORMER: A VERSATILE VISION TRANSFORMER HINGING ON CROSS-SCALE ATTENTION"
"""
# pylint: disable=E0401
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal
from model.layers import to_2tuple, DropPath


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


class DynamicPosBias(nn.Cell):
    """ DynamicPosBias """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Dense(2, self.pos_dim)
        self.pos1 = nn.SequentialCell(
            nn.LayerNorm((self.pos_dim,)),
            nn.ReLU(),
            nn.Dense(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.SequentialCell(
            nn.LayerNorm((self.pos_dim,)),
            nn.ReLU(),
            nn.Dense(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.SequentialCell(
            nn.LayerNorm((self.pos_dim,)),
            nn.ReLU(),
            nn.Dense(self.pos_dim, self.num_heads)
        )

    def construct(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        """ get flops """
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Cell):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)

            # generate mother-set
            position_bias_h = ms.ops.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = ms.ops.arange(1 - self.group_size[1], self.group_size[1])
            biases = ms.ops.stack(ms.ops.meshgrid(position_bias_h, position_bias_w))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(start_dim=1).transpose(1, 0).float()
            self.biases = ms.Parameter(biases, name='biases', requires_grad=False)

            # get pair-wise relative position index for each token inside the group
            coords_h = ms.ops.arange(self.group_size[0])
            coords_w = ms.ops.arange(self.group_size[1])
            coords = ms.ops.stack(ms.ops.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
            coords_flatten = ms.ops.flatten(coords, start_dim=1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.relative_position_index = ms.Parameter(relative_position_index, name='relative_position_index',
                                                        requires_grad=False)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 3, 2)

        if self.position_bias:
            pos = self.pos(self.biases)  # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1) # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        """ get flops """
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops


class CrossFormerBlock(nn.Cell):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, group_size=7, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        if min(self.input_resolution) <= self.group_size:
            # if group size is larger than input resolution, we don't partition groups
            self.lsda_flag = 0
            self.group_size = min(self.input_resolution)

        self.norm1 = norm_layer((dim, ))

        self.attn = Attention(
            dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer((dim, ))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn_mask = None
        # self.attn_mask = ms.Parameter(attn_mask, name='attn_mask', requires_grad=False)

    def construct(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L}, {H}, {W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # group embeddings
        G = self.group_size
        if self.lsda_flag == 0:  # 0 for SDA
            x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5)
        else:  # 1 for LDA
            x = x.reshape(B, G, H // G, G, W // G, C).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B * H * W // G ** 2, G ** 2, C)

        # multi-head self-attention
        x = self.attn(x, mask=self.attn_mask)  # nW*B, G*G, C

        # ungroup embeddings
        x = x.reshape(B, H // G, W // G, G, G, C)
        if self.lsda_flag == 0:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def flops(self):
        """ get flops """
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # LSDA
        nW = H * W / self.group_size / self.group_size
        flops += nW * self.attn.flops(self.group_size * self.group_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Cell):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, patch_size=None):
        super().__init__()
        if patch_size is None:
            patch_size = [2]
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.SequentialCell()
        self.patch_size = patch_size
        self.norm = norm_layer((dim, ))

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(nn.Conv2d(dim, out_dim, kernel_size=ps,
                                             stride=stride, pad_mode='pad', padding=padding))

    def construct(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        xs = []
        for i, _ in enumerate(self.reductions):
            tmp_x = self.reductions[i](x).flatten(start_dim=2).transpose(0, 2, 1)
            xs.append(tmp_x)
        x = ms.ops.cat(xs, axis=2)
        return x

    def flops(self):
        """ get flops """
        H, W = self.input_resolution
        flops = H * W * self.dim
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                out_dim = 2 * self.dim // 2 ** i
            else:
                out_dim = 2 * self.dim // 2 ** (i + 1)
            flops += (H // 2) * (W // 2) * ps * ps * out_dim * self.dim
        return flops


class Stage(nn.Cell):
    """ CrossFormer blocks for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): variable G in the paper, one group has GxG embeddings
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, group_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, patch_size_end=None, num_patch_size=None):

        super().__init__()
        if patch_size_end is None:
            patch_size_end = [4]
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.SequentialCell()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1
            self.blocks.append(CrossFormerBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, group_size=group_size,
                                                lsda_flag=lsda_flag,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=float(drop_path[i])
                                                if isinstance(drop_path, list) else drop_path,
                                                norm_layer=norm_layer,
                                                num_patch_size=num_patch_size))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        """ get flops """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Cell):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: [4].
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=None, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if patch_size is None:
            patch_size = [4]
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[0] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.SequentialCell()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, pad_mode='pad', padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer((embed_dim, ))
        else:
            self.norm = None

    def construct(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        xs = []
        for i, _ in enumerate(self.projs):
            tx = self.projs[i](x).flatten(start_dim=2).transpose(0, 2, 1)
            xs.append(tx)  # B Ph*Pw C
        x = ms.ops.cat(xs, axis=2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """ get flops """
        Ho, Wo = self.patches_resolution
        flops = 0
        for i, _ in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                dim = self.embed_dim // 2 ** i
            else:
                dim = self.embed_dim // 2 ** (i + 1)
            flops += Ho * Wo * dim * self.in_chans * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class CrossFormer(nn.Cell):
    r""" CrossFormer
        A PyTorch impl of : `CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention`  -

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        group_size (int): Group size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=None, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=None, num_heads=None,
                 group_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 merge_size=None):
        super().__init__()

        if merge_size is None:
            merge_size = [[2], [2], [2]]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        if patch_size is None:
            patch_size = [4]
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
            self.absolute_pos_embed.set_data(initializer(TruncatedNormal(sigma=.02), self.absolute_pos_embed.shape,
                                                         self.absolute_pos_embed.dtype))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = list(ms.ops.linspace(0, drop_path_rate, sum(depths)))

        # build layers
        self.layers = nn.SequentialCell()

        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        for i_layer in range(self.num_layers):
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          group_size=group_size[i_layer],
                          mlp_ratio=self.mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          norm_layer=norm_layer,
                          downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                          patch_size_end=patch_size_end,
                          num_patch_size=num_patch_size)
            self.layers.append(layer)

        self.norm = norm_layer((self.num_features, ))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def construct_features(self, x):
        """ get features """
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

    def flops(self):
        """ get flops """
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    model = CrossFormer(img_size=224,
                        patch_size=[4, 8, 16, 32],
                        in_chans=3,
                        num_classes=1000,
                        embed_dim=48,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        group_size=[7, 7, 7, 7],
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.0,
                        drop_path_rate=0.1,
                        ape=False,
                        patch_norm=True,
                        merge_size=[[2, 4], [2, 4], [2, 4]]
                        )
    output = model(dummy_input)
    print(output.shape)
