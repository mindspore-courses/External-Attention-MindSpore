""" HorNet """
from functools import partial
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, TruncatedNormal, XavierUniform


def trunc_norm(tensor, std=.02):
    """ truncated normalization
    """
    return initializer(TruncatedNormal(sigma=std), tensor.shape, tensor.dtype)


def get_dwconv(dim, kernel, bias):
    """ get dwconv
    """
    return nn.Conv2d(dim, dim, kernel_size=kernel, pad_mode='pad', padding=(kernel - 1) // 2, has_bias=bias, group=dim)


class DropPath(nn.Cell):
    """ Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob, ndim=2):
        super().__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = ms.Tensor(ms.ops.ones(shape), dtype=ms.float32)

    def construct(self, x):
        if not self.training:
            return x
        mask = ms.ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class GlobalLocalFilter(nn.Cell):
    """ GlobalLocalFilter
        https://arxiv.org/abs/2207.14284
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, pad_mode='pad', padding=1, group=dim // 2)
        self.complex_weight = ms.Parameter(ms.ops.randn((dim // 2, h, w, 2), dtype=ms.float32) * 0.02)
        self.complex_weight = trunc_norm(self.complex_weight)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

        self.rfft = ms.ops.FFTWithSize(signal_ndim=2, inverse=False, real=True, norm='ortho', onesided=True,
                                       signal_sizes=(2, 3))
        self.irfft = ms.ops.FFTWithSize(signal_ndim=2, inverse=True, real=True, norm='ortho', onesided=True,
                                        signal_sizes=(2, 3))

    def construct(self, x):
        x = self.pre_norm(x)
        x1, x2 = ms.ops.chunk(x, 2, axis=1)
        x1 = self.dw(x1)

        x2 = x2.to(ms.float32)
        B, C, H, W = x2.shape
        x2 = self.rfft(x2)

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = ms.ops.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                        align_corners=True).permute(1, 2, 3, 0)

        weight = weight.to(ms.complex64)
        x2 *= weight
        x2 = self.irfft(x2)

        x = ms.ops.cat([x1.unsqueeze(2), x2.unsqueeze(2)], axis=2).reshape(B, 2 * C, H, W)
        x = self.post_norm(x)
        return x


class Gnconv(nn.Cell):
    """gnconv
    """
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.SequentialCell(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s

    def construct(self, x):
        fused_x = self.proj_in(x)
        pwa, abc = ms.ops.split(fused_x, (self.dims[0], sum(self.dims)), axis=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = ms.ops.split(dw_abc, self.dims, axis=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        return self.proj_out(x)


class Block(nn.Cell):
    """HorNet's Block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=Gnconv):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)

        self.gamma1 = ms.Parameter(layer_scale_init_value * ms.ops.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = ms.Parameter(layer_scale_init_value * ms.ops.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        _, C, _, _ = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x += self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        inputs = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma2 is not None:
            x *= self.gamma2
        x = x.permute(0, 3, 1, 2)
        x = inputs + self.drop_path(x)
        return x


class HorNet(nn.Cell):
    """HorNet
    """
    def __init__(self, depths, in_chans=3, num_classes=1000, base_dim=96, drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1., gnconv=Gnconv, block=Block, uniform_init=False):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.downsample_layers = nn.SequentialCell()
        stem = nn.SequentialCell(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            dowmsample_layer = nn.SequentialCell(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(dowmsample_layer)

        self.stages = nn.SequentialCell()
        dp_rates = list(x for x in ms.ops.linspace(0, drop_path_rate, sum(depths)))

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        assert len(gnconv) == 4

        cur = 0
        for i in range(4):
            stage = nn.SequentialCell(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm((dims[-1],), epsilon=1e-6)
        self.head = nn.Dense(dims[-1], num_classes)
        self.uniform_init = uniform_init

        weight = self.head.weight * head_init_scale
        self.head.weight.set_data(weight)

        if self.head.bias is not None:
            bias = self.head.bias * head_init_scale
            self.head.bias.set_data(bias)

        self.apply(self.init_weight)

    def init_weight(self, cell):
        """ init weight
        """
        if not self.uniform_init:
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(trunc_norm(cell.weight))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        else:
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct_features(self, x):
        """construct features
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            for blk in self.stages[i]:
                x = blk(x)
        return self.norm(x.mean([-2, -1]))

    def construct(self, x):
        x = self.construct_features(x)
        return self.head(x)


class LayerNorm(nn.Cell):
    """LayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = ms.Parameter(ms.ops.ones(normalized_shape))
        self.bias = ms.Parameter(ms.ops.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_first', 'channels_last']:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)
        self.layer_norm = nn.LayerNorm(self.normalized_shape, epsilon=self.eps)

    def construct(self, x):
        if self.data_format == 'channels_last':
            self.layer_norm.gamma.set_data(self.weight)
            self.layer_norm.beta.set_data(self.bias)
            x = self.layer_norm(x)
        if self.data_format == 'channels_first':
            u = x.mean(1, keep_dims=True)
            s = (x - u).pow(2).mean(1, keep_dims=True)
            x = (x - u) / ms.ops.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def hornet_tiny_7x7(**kwargs):
    """hornet_tiny_7x7
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s),
                       partial(Gnconv, order=5, s=s)
                   ], **kwargs)
    return model


def hornet_tiny_gf(**kwargs):
    """hornet_tiny_gf
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(Gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter)
                   ], **kwargs)
    return model


def hornet_small_7x7(**kwargs):
    """hornet_small_7x7
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=96, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s),
                       partial(Gnconv, order=5, s=s)
                   ], **kwargs)
    return model


def hornet_small_gf(**kwargs):
    """hornet_small_gf
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=96, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(Gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter)
                   ], **kwargs)
    return model


def hornet_base_7x7(**kwargs):
    """hornet_base_7x7
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s),
                       partial(Gnconv, order=5, s=s)
                   ], **kwargs)
    return model


def hornet_base_gf(**kwargs):
    """hornet_base_gf
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(Gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter)
                   ], **kwargs)
    return model


def hornet_large_7x7(**kwargs):
    """hornet_large_7x7
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s),
                       partial(Gnconv, order=5, s=s)
                   ], **kwargs)
    return model


def hornet_large_gf(**kwargs):
    """hornet_large_gf
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
                       partial(Gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter)
                   ], **kwargs)
    return model


def hornet_large_gf_img384(**kwargs):
    """hornet_large_gf_img384
    """
    s = 1 / 3.
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
                   gnconv=[
                       partial(Gnconv, order=2, s=s),
                       partial(Gnconv, order=3, s=s),
                       partial(Gnconv, order=4, s=s, h=24, w=13, gflayer=GlobalLocalFilter),
                       partial(Gnconv, order=5, s=s, h=12, w=7, gflayer=GlobalLocalFilter)
                   ], **kwargs)
    return model


if __name__ == "__main__":
    in_tensor = ms.ops.randn((1, 3, 224, 224))
    hornet = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
                    gnconv=[
                       partial(Gnconv, order=2, s=1/3),
                       partial(Gnconv, order=3, s=1/3),
                       partial(Gnconv, order=4, s=1/3),
                       partial(Gnconv, order=5, s=1/3)
                   ])
    output = hornet(in_tensor)
    print(output.shape)
