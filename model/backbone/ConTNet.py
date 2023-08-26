# pylint: disable=E0401
# pylint: disable=W0201
"""
MindSpore implementation of `ConTNet`.
Refer to ConTNet: Why not use convolution and transformer at the same time?
"""
from collections import OrderedDict
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, HeNormal, TruncatedNormal, Constant

from model.layers import _trunc_normal_

__all__ = ['ConTBlock', 'ConTNet']


def fixed_padding(inputs, kernel_size, dilation):
    """ fixed padding """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = ms.ops.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class ConvBN(nn.SequentialCell):
    """ Conv and BN """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, group=1, bn=True):
        padding = (kernel_size - 1) // 2
        if bn:
            super().__init__(OrderedDict([
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                   pad_mode='pad', padding=padding, group=group)),
            ]))
        else:
            super().__init__(OrderedDict([
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad',
                                   padding=padding, group=group, has_bias=False))
            ]))


class MHSA(nn.Cell):
    """ Multi-head Self Attention """
    def __init__(self, planes, head_num, dropout, patch_size, qkv_bias, relative):
        super().__init__()
        self.head_num = head_num
        head_dim = planes // head_num
        self.qkv = nn.Dense(planes, 3 * planes, has_bias=qkv_bias)
        self.relative = relative
        self.patch_size = patch_size
        self.scale = head_dim ** -0.5

        if self.relative:
            # print('### relative position embedding ###')
            self.relative_position_bias_table = ms.Parameter(
                ms.ops.zeros(((2 * patch_size - 1) * (2 * patch_size - 1), head_num)))
            coords_w = coords_h = ms.ops.arange(patch_size)
            coords = ms.ops.stack(ms.ops.meshgrid(coords_h, coords_w))
            coords_flatten = ms.ops.flatten(coords, start_dim=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0)
            relative_coords[:, :, 0] += patch_size - 1
            relative_coords[:, :, 1] += patch_size - 1
            relative_coords[:, :, 0] *= 2 * patch_size - 1
            self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            _trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(p=dropout)
        self.proj = nn.Dense(planes, planes)
        self.proj_drop = nn.Dropout(p=dropout)

    def construct(self, x):
        B, N, C, H = *x.shape, self.head_num
        qkv = self.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)  # x: (3, B, H, N, C//H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # x: (B, H, N, C//N)

        q = q * self.scale
        attn = q @ k.transpose(0, 1, 3, 2)  # attn: (B, H, N, N)

        if self.relative:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.patch_size ** 2, self.patch_size ** 2, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1)
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = ms.ops.softmax(attn, axis=-1)
        # attn = attn.softmax(axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Cell):
    """ Build a Multi-Layer Perceptron """

    def __init__(self, planes, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Dense(planes, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Dense(mlp_dim, planes)
        self.drop = nn.Dropout(p=dropout)

    def construct(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class STE(nn.Cell):
    """ Build a Standard Transformer Encoder(STE) """

    def __init__(self, planes, mlp_dim, head_num, dropout, patch_size,
                 relative, qkv_bias, pre_norm):
        super().__init__()
        self.patch_size = patch_size
        self.pre_norm = pre_norm
        self.relative = relative

        if not relative:
            self.pe = ms.ParameterTuple(
                [ms.Parameter(ms.ops.zeros((1, patch_size, 1, planes // 2))),
                 ms.Parameter(ms.ops.zeros((1, 1, patch_size, planes // 2)))]
            )
        self.attn = MHSA(planes, head_num, dropout, patch_size, qkv_bias, relative)
        self.mlp = MLP(planes, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm((planes,))
        self.norm2 = nn.LayerNorm((planes,))

        self.unfold1 = nn.Unfold((1, 1, self.patch_size, 1), strides=[1, 1, self.patch_size, 1], rates=[1, 1, 1, 1])
        self.unfold2 = nn.Unfold((1, self.patch_size, 1, 1), strides=[1, self.patch_size, 1, 1], rates=[1, 1, 1, 1])

    def construct(self, x):
        B, C, H, W = x.shape
        patch_size = self.patch_size
        patch_num_h, patch_num_w = H // patch_size, W // patch_size

        x = x.unfold(kernel_size=(self.patch_size, 1), stride=(self.patch_size, 1)).reshape(B, C, self.patch_size, -1)
        x = x.unfold(kernel_size=(1, self.patch_size), stride=(1, self.patch_size))
        x = x.reshape(B, C, patch_num_h, patch_num_w, patch_size, patch_size)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(-1, patch_size, patch_size, C)
        if not self.relative:
            x_h, x_w = x.split(C // 2, axis=3)
            x = ms.ops.cat((x_h + self.pe[0], x_w + self.pe[1]), axis=3)

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        if self.pre_norm:
            x += self.attn(self.norm1(x))
            x += self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x))
            x = self.norm2(x + self.mlp(x))

        b_pnh_pnw, _, c = x.shape
        b = b_pnh_pnw // (patch_num_h * patch_num_w)
        x = x.reshape(b, patch_num_h, patch_num_w, patch_size, patch_size, c).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(b, c, patch_num_h * patch_size, patch_num_w * patch_size)
        return x


class ConTBlock(nn.Cell):
    """ Build a ContBlock """

    def __init__(self, planes, out_planes, mlp_dim, head_num, dropout, patch_size,
                 downsample, stride=1, last_dropout=.3, **kwargs):
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Identity()
        self.dropout = nn.Identity()

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.ste1 = STE(planes, mlp_dim, head_num, dropout, patch_size[0], **kwargs)
        self.ste2 = STE(planes, mlp_dim, head_num, dropout, patch_size[1], **kwargs)

        if stride == 1 and downsample is not None:
            self.dropout = nn.Dropout(p=last_dropout)
            kernel_size = 1
        else:
            kernel_size = 3

        self.out_conv = ConvBN(planes, out_planes, kernel_size, stride, bn=False)

    def construct(self, x):
        x_preact = self.relu(self.bn(x))
        identity = self.identity(x)

        if self.downsample is not None:
            identity = self.downsample(x_preact)

        residual = self.ste1(x_preact)
        residual = self.ste2(residual)
        residual = self.out_conv(residual)
        out = self.dropout(residual + identity)
        return out


class ConTNet(nn.Cell):
    """ Build a ConTNet backbone """

    def __init__(self, block, layers, mlp_dim, head_num, dropout, in_channels=3,
                 inplanes=64, num_classes=1000, init_weights=True, first_embedding=False,
                 tweak_C=False, **kwargs):
        super().__init__()
        self.inplanes = inplanes
        self.block = block

        if tweak_C:
            self.layer0 = nn.SequentialCell(OrderedDict([
                ('conv_bn1', ConvBN(in_channels, inplanes//2, kernel_size=3, stride=2)),
                ('relu1', nn.ReLU()),
                ('conv_bn2', ConvBN(inplanes//2, inplanes//2, kernel_size=3, stride=1)),
                ('relu2', nn.ReLU()),
                ('conv_bn3', ConvBN(inplanes//2, inplanes, kernel_size=3, stride=1)),
                ('relu3', nn.ReLU()),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=1))
            ]))
        elif first_embedding:
            self.layer0 = nn.SequentialCell(OrderedDict([
                ('conv', nn.Conv2d(in_channels, inplanes, kernel_size=4, stride=4)),
                ('norm', nn.LayerNorm((inplanes, )))
            ]))
        else:
            self.layer0 = nn.SequentialCell(OrderedDict([
                ('conv', ConvBN(in_channels, inplanes, kernel_size=7, stride=2, bn=False)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=1))
            ]))

        self.cont_layers = []
        self.out_channels = OrderedDict()

        for i, layer in enumerate(layers):
            stride = 2
            patch_size = [7, 14]
            if i == len(layers) - 1:
                stride, patch_size[1] = 1, 7
            cont_layer = self._make_layer(inplanes * 2**i, layer, stride=stride, mlp_dim=mlp_dim[i],
                                          head_num=head_num[i], dropout=dropout[i], patch_size=patch_size, **kwargs)
            layer_name = f'layer{i+1}'
            self.insert_child_to_cell(layer_name, cont_layer)
            self.cont_layers.append(layer_name)
            self.out_channels[layer_name] = 2 * inplanes * 2**i

        self.last_out_channels = next(reversed(self.out_channels.values()))
        self.fc = nn.Dense(self.last_out_channels, num_classes)

        if init_weights:
            self.apply(self._init_weights)

    def _make_layer(self, planes, blocks, stride, mlp_dim, head_num, dropout, patch_size, use_avg_down=False, **kwargs):
        """ make layer """
        layers = OrderedDict()
        for i in range(0, blocks-1):
            layers[f'{self.block.__name__}{i}'] = self.block(
                planes, planes, mlp_dim, head_num, dropout, patch_size, **kwargs
            )
        # downsample = None
        if stride != 1:
            if use_avg_down:
                downsample = nn.SequentialCell(OrderedDict([
                    ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2)),
                    ('conv', ConvBN(planes, planes*2, kernel_size=1, stride=1, bn=False))
                ]))
            else:
                downsample = ConvBN(planes, planes*2, kernel_size=1, stride=2, bn=False)
        else:
            downsample = ConvBN(planes, planes*2, kernel_size=1, stride=1, bn=False)
        layers[f'{self.block.__name__}{blocks-1}'] = self.block(
            planes, planes*2, mlp_dim, head_num, dropout, patch_size, downsample, stride, **kwargs
        )
        return nn.SequentialCell(layers)

    def _init_weights(self, cell):
        """ initialize weight """
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                                             cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(TruncatedNormal(sigma=.02),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer(Constant(0), cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm)):
            cell.gamma.set_data(initializer(Constant(1), cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer(Constant(0), cell.beta.shape, cell.beta.dtype))

    def construct(self, x):
        x = self.layer0(x)

        for _, layer_name in enumerate(self.cont_layers):
            cont_layer = getattr(self, layer_name)
            x = cont_layer(x)
        x = x.mean([2, 3])
        x = self.fc(x)

        return x


def create_ConTNet_Ti(kwargs):
    """ ConTNet-Ti """
    return ConTNet(block=ConTBlock,
                   mlp_dim=[196, 392, 768, 768],
                   head_num=[1, 2, 4, 8],
                   dropout=[0, 0, 0, 0],
                   inplanes=48,
                   layers=[1, 1, 1, 1],
                   last_dropout=0,
                   **kwargs)

def create_ConTNet_S(kwargs):
    """ ConTNet-S """
    return ConTNet(block=ConTBlock,
                   mlp_dim=[256, 512, 1024, 1024],
                   head_num=[1, 2, 4, 8],
                   dropout=[0, 0, 0, 0],
                   inplanes=64,
                   layers=[1, 1, 1, 1],
                   last_dropout=0,
                   **kwargs)

def create_ConTNet_M(kwargs):
    """ ConTNet-M """
    return ConTNet(block=ConTBlock,
                   mlp_dim=[256, 512, 1024, 1024],
                   head_num=[1, 2, 4, 8],
                   dropout=[0, 0, 0, 0],
                   inplanes=64,
                   layers=[2, 2, 2, 2],
                   last_dropout=0,
                   **kwargs)

def create_ConTNet_B(kwargs):
    """ ConTNet-B """
    return ConTNet(block=ConTBlock,
                   mlp_dim=[256, 512, 1024, 1024],
                   head_num=[1, 2, 4, 8],
                   dropout=[0, 0, 0.1, 0.1],
                   inplanes=64,
                   layers=[3, 4, 6, 3],
                   last_dropout=0.2,
                   **kwargs)

def build_model(relative, qkv_bias, pre_norm):
    """ build model """
    kwargs = {"relative": relative, "qkv_bias": qkv_bias, 'pre_norm': pre_norm}
    return create_ConTNet_Ti(kwargs)


if __name__ == "__main__":
    model = build_model(relative=True, qkv_bias=True, pre_norm=True)
    dummy_input = ms.ops.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)
