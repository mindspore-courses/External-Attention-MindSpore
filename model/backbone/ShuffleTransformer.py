# pylint: disable=E0401
"""
MindSpore implementation of `ShuffleTransformer`.
Refer to "Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer"
"""
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal

from model.layers import DropPath


class Mlp(nn.Cell):
    """ Mlp """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0., stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, has_bias=True)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Cell):
    """ Attention """
    def __init__(self, dim, num_heads, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shuffle = shuffle

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(p=proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = ms.Parameter(
                ms.ops.zeros(((2 * window_size - 1) * (2 * window_size - 1), num_heads)))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = ms.ops.arange(self.ws)
            coords_w = ms.ops.arange(self.ws)
            coords = ms.ops.stack(ms.ops.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
            coords_flatten = ms.ops.flatten(coords, start_dim=1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

            self.relative_position_bias_table = initializer(TruncatedNormal(sigma=.02),
                                                            self.relative_position_bias_table.shape,
                                                            self.relative_position_bias_table.dtype)

    def construct(self, x):
        _, _, H, _ = x.shape
        qkv = self.to_qkv(x)
        b, _, h, w = qkv.shape
        if self.shuffle:
            qkv = qkv.reshape(b, 3, self.num_heads, -1, self.ws, h//self.ws, self.ws, w//self.ws)
            b, qkv_dim, h, d, ws1, hh, ws2, ww = qkv.shape
            qkv = qkv.transpose(1, 0, 5, 7, 2, 4, 6, 3)
            q = k = v = ms.ops.sum(qkv.reshape(qkv_dim, b*hh*ww, h, ws1*ws2, d), dim=0)

        else:
            qkv = qkv.reshape(b, 3, self.num_heads, -1, h//self.ws, self.ws, w//self.ws, self.ws)
            b, qkv_dim, h, d, hh, ws1, ww, ws2 = qkv.shape

            qkv = qkv.transpose(1, 0, 4, 6, 2, 5, 7, 3)
            q = k = v = ms.ops.sum(qkv.reshape(qkv_dim, b*hh*ww, h, ws1*ws2, d), dim=0)

        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = ms.ops.softmax(dots, axis=-1)
        out = attn @ v

        if self.shuffle:
            out = out.reshape(b, H//self.ws, out.shape[0]//b//(H//self.ws), self.num_heads, self.ws, self.ws, -1)
            b, hh, ww, h, ws1, ws2, d = out.shape
            out = out.transpose(0, 3, 6, 4, 1, 5, 2).reshape(b, h*d, ws1*hh, ws2*ww).sum(axis=0, keepdims=True)

        else:
            out = out.reshape(b, H//self.ws, out.shape[0]//b//(H//self.ws), self.num_heads, self.ws, self.ws, -1)
            b, hh, ww, h, ws1, ws2, d = out.shape
            out = out.transpose(0, 3, 6, 1, 4, 2, 5).reshape(b, h*d, hh*ws1, ww*ws2).sum(axis=0, keepdims=True)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Block(nn.Cell):
    """ Block """
    def __init__(self, dim, out_dim, num_heads, window_size=1, shuffle=False, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, stride=False,
                 relative_pos_embedding=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=relative_pos_embedding)
        self.local = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, pad_mode='pad', padding=window_size // 2,
                               group=dim, has_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer,
                       drop=drop, stride=stride)
        self.norm3 = norm_layer(dim)

    def construct(self, x):
        y = self.norm1(x)

        y = self.attn(y)
        y = self.drop_path(y)
        x += y

        x = x + self.local(self.norm2(x))  # local connection
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class PatchMerging(nn.Cell):
    """ PatchMerging """
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2, has_bias=False)

    def construct(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class StageModule(nn.Cell):
    """ Stage Module """
    def __init__(self, layers, dim, out_dim, num_heads, window_size=1, shuffle=True, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., relative_pos_embedding=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        if dim != out_dim:
            self.patch_partition = PatchMerging(dim, out_dim)
        else:
            self.patch_partition = None

        num = layers // 2
        self.layers = nn.SequentialCell([])
        for _ in range(num):
            self.layers.append(nn.SequentialCell([
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=False,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding),
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding)
            ]))

    def construct(self, x):
        if self.patch_partition:
            x = self.patch_partition(x)

        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x


class PatchEmbedding(nn.Cell):
    """ Patch Embedding """
    def __init__(self, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU6()
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, pad_mode='pad', padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def construct(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x


class ShuffleTransformer(nn.Cell):
    """ Shuffle Transformer """
    def __init__(self, num_classes=1000, token_dim=32, embed_dim=96, mlp_ratio=4., layers=None, num_heads=None,
                 relative_pos_embedding=True, shuffle=True, window_size=7, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., has_pos_embed=False):
        super().__init__()
        if layers is None:
            layers = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.has_pos_embed = has_pos_embed
        dims = [i * 32 for i in num_heads]

        self.to_token = PatchEmbedding(inter_channel=token_dim, out_channels=embed_dim)

        dpr = list(ms.ops.linspace(0, drop_path_rate, 4))  # stochastic depth decay rule
        self.stage1 = StageModule(layers[0], embed_dim, dims[0], num_heads[0], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[0],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[1],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[2],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                  attn_drop=attn_drop_rate, drop_path=dpr[3],
                                  relative_pos_embedding=relative_pos_embedding)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier head
        self.head = nn.Dense(dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, cell):
        """ initialize weights """
        if isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))
        elif isinstance(cell, (nn.Dense, nn.Conv2d)):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def get_classifier(self):
        """ get classifier """
        return self.head

    def reset_classifier(self, num_classes):
        """ reset classifier """
        self.num_classes = num_classes
        self.head = nn.Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def construct_features(self, x):
        """ construct features """
        x = self.to_token(x)
        _, c, h, w = x.shape

        if self.has_pos_embed:
            x = x + self.pos_embed.view(1, h, w, c).permute(0, 3, 1, 2)
            x = self.pos_drop(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = ms.ops.flatten(x, start_dim=1)
        return x

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    sft = ShuffleTransformer()
    output = sft(dummy_input)
    print(output.shape)
