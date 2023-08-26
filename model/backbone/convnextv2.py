# pylint: disable=E0401
"""
MindSpore implementation of `ConvNextV2`.
Refer to ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders
"""
import mindspore as ms
from mindspore import nn
from model.layers import DropPath


class Block(nn.Cell):
    """ Block """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, pad_mode='pad', padding=3, group=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Dense(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def construct(self, x):
        y = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)

        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)
        x = y + self.drop_path(x)

        return x


class LayerNorm(nn.Cell):
    """ LayerNorm """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        if data_format == "channels_first":
            self.weight = ms.Parameter(ms.ops.ones(normalized_shape))
            self.bias = ms.Parameter(ms.ops.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError()
        self.normalized_shape = (normalized_shape,)
        self.layer_norm = nn.LayerNorm(self.normalized_shape, gamma_init='ones', beta_init='zeros', epsilon=self.eps)

    def construct(self, x):
        if self.data_format == "channels_last":
            # layer_norm = nn.LayerNorm()
            # return ms.ops.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return self.layer_norm(x)
        u = x.mean(1, keep_dims=True)
        s = (x - u).pow(2).mean(1, keep_dims=True)
        x = (x - u) / ms.ops.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Cell):
    """ GRN """
    def __init__(self, dim):
        super().__init__()
        self.gamma = ms.Parameter(ms.ops.zeros((1, 1, 1, dim)))
        self.beta = ms.Parameter(ms.ops.zeros((1, 1, 1, dim)))

    def construct(self, x):
        Gx = ms.ops.norm(x, ord=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(axis=-1, keep_dims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextV2(nn.Cell):
    """ ConvNextV2 """
    def __init__(self, in_channels=3, num_classes=1000,
                 depth=None, dims=None,
                 drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        if depth is None:
            depth = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]
        self.depth = depth
        self.downsample_layers = nn.SequentialCell()
        stem = nn.SequentialCell(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.SequentialCell()
        dp_rates = ms.ops.linspace(0, drop_path_rate, sum(depth))
        cur = 0
        for i in range(4):
            stage = nn.SequentialCell(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depth[i])]
            )
            self.stages.append(stage)
            cur += depth[i]

        self.norm = nn.LayerNorm((dims[-1],), epsilon=1e-6)
        self.head = nn.Dense(dims[-1], num_classes)

        self.head.weight.set_data(self.head.weight.data * head_init_scale)
        if self.head.bias is not None:
            self.head.bias.set_data(self.head.bias.data * head_init_scale)

    def construct(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    """ ConvNextV2-atto """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    """ ConvNextV2-femto """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    """ ConvNextV2-pico """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    """ ConvNextV2-nano """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    """ ConvNextV2-tiny """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    """ ConvNextV2-base """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextv2_large(**kwargs):
    """ ConvNextV2-large """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextv2_huge(**kwargs):
    """ ConvNextV2-huge """
    model = ConvNextV2(depth=[2, 2, 6, 2], dims=[352, 704, 1408, 2816], **kwargs)
    return model


if __name__ == "__main__":
    convnextv2 = convnextv2_atto()
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    output = convnextv2(dummy_input)
    print(output.shape)
