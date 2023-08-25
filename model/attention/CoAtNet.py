"""
MindSpore implementation of 'CoAtNet'
Refer to "CoAtNet: Marrying Convolution and Attention for All Data Sizes"
"""
import math
from functools import partial
import mindspore as ms
from mindspore import nn


class SelfAttention(nn.Cell):
    """ Self Attention """

    def __init__(self, d_model, d_k=None, d_v=None, h=1, drop_rate=0.1):
        super().__init__()
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.fc_q = nn.Dense(d_model, h * d_k)
        self.fc_k = nn.Dense(d_model, h * d_k)
        self.fc_v = nn.Dense(d_model, h * d_v)
        self.fc_o = nn.Dense(h * d_v, d_model)
        self.dropout = nn.Dropout(p=drop_rate)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def construct(self, x, attention_mask=None, attention_weights=None):
        B, N = x.shape[:2]

        q = self.fc_q(x).view(B, N, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(B, N, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(x).view(B, N, self.h, self.d_v).permute(0, 2, 1, 3)

        att = ms.ops.matmul(q, k) / math.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -float('inf'))

        att = ms.ops.softmax(att, axis=-1)
        att = self.dropout(att)
        att = ms.ops.matmul(att, v).permute(0, 2, 1, 3).view(B, N, self.h * self.d_v)
        return self.fc_o(att)


class Swish(nn.Cell):
    """Swish activation
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


def drop_connect(inputs, p, training):
    """drop connect
    """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += ms.ops.randn((batch_size, 1, 1, 1), dtype=inputs.dtype)
    binary_tensor = ms.ops.floor(random_tensor)
    out = inputs / keep_prob * binary_tensor
    return out


def get_same_padding_conv2d(image_size=None):
    """"get same padding conv2d
    """
    return partial(Conv2dStaticSamePadding, image_size=image_size)


def get_width_and_height_from_size(x):
    """
    :param x: image size
    :return: image's width and height
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, (list, tuple)):
        return x
    raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """
    :param input_image_size:
    :param stride:
    :return: output image size
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class Conv2dStaticSamePadding(nn.Conv2d, nn.Cell):
    """Conv2dStaticSamePadding
    """

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        # super().__init__()
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[0] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 and pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def construct(self, x):
        x = self.static_padding(x)
        x = ms.ops.conv2d(x, self.weight, self.bias, self.stride, pad_mode='pad',
                          padding=self.padding, dilation=self.dilation, groups=self.group)
        return x


class MBConvBlock(nn.Cell):
    """MBConvBlock
    """

    def __init__(self, ksize, input_filters, out_filters, expand_ratio=1, stride=1, image_size=224):
        super().__init__()
        self._bn_mom = 0.1
        self._bn_eps = 0.01
        self._se_ratio = 0.25
        self._input_filters = input_filters
        self._output_filters = out_filters
        self._expand_ratio = expand_ratio
        self._kernel_size = ksize
        self._stride = stride

        inp = self._input_filters
        oup = self._input_filters * self._expand_ratio
        if self._expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, has_bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._kernel_size
        s = self._stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, group=oup,
                                      kernel_size=k, stride=s, has_bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        Conv2d = get_same_padding_conv2d(image_size=(1, 1))
        num_squeezed_channels = max(1, int(self._input_filters * self._se_ratio))
        self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, has_bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = Swish()

    def construct(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._expand_ratio != 1:
            expand = self._expand_conv(inputs)
            bn0 = self._bn0(expand)
            x = self._swish(bn0)
        depthwise = self._depthwise_conv(x)
        bn1 = self._bn1(depthwise)
        x = self._swish(bn1)

        x_squeezed = ms.ops.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._swish(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = ms.ops.sigmoid(x_squeezed) * x

        input_filters, output_filters = self._input_filters, self._output_filters
        if self._stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class CoAtNet(nn.Cell):
    """ CoAtNet """

    def __init__(self, in_channels, image_size, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = [64, 96, 192, 384, 768]
        self.out_chs = out_channels
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.s0 = nn.SequentialCell(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, pad_mode='pad', padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, pad_mode='pad', padding=1)
        )
        self.mlp0 = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1)
        )

        self.s1 = MBConvBlock(ksize=3, input_filters=out_channels[0], out_filters=out_channels[0],
                              image_size=image_size // 2)
        self.mlp1 = nn.SequentialCell(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1)
        )

        self.s2 = MBConvBlock(ksize=3, input_filters=out_channels[1], out_filters=out_channels[1],
                              image_size=image_size // 4)
        self.mlp2 = nn.SequentialCell(
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1)
        )

        self.s3 = SelfAttention(out_channels[2], out_channels[2] // 8, out_channels[2] // 8, 8)
        self.mlp3 = nn.SequentialCell(
            nn.Dense(out_channels[2], out_channels[3]),
            nn.ReLU(),
            nn.Dense(out_channels[3], out_channels[3])
        )

        self.s4 = SelfAttention(out_channels[3], out_channels[3] // 8, out_channels[3] // 8, 8)
        self.mlp4 = nn.SequentialCell(
            nn.Dense(out_channels[3], out_channels[4]),
            nn.ReLU(),
            nn.Dense(out_channels[4], out_channels[4])
        )

    def construct(self, x):
        B = x.shape[0]
        # stage0
        y = self.mlp0(self.s0(x))
        y = self.maxpool2d(y)
        # stage1
        y = self.mlp1(self.s1(y))
        y = self.maxpool2d(y)
        # stage2
        y = self.mlp2(self.s2(y))
        y = self.maxpool2d(y)
        # stage3
        y = y.reshape(B, self.out_chs[2], -1).permute(0, 2, 1)  # B,N,C
        y = self.mlp3(self.s3(y))
        y = self.maxpool1d(y.permute(0, 2, 1)).permute(0, 2, 1)
        # stage4
        y = self.mlp4(self.s4(y))
        y = self.maxpool1d(y.permute(0, 2, 1))
        N = y.shape[-1]
        y = y.reshape(B, self.out_chs[4], int(math.sqrt(N)), int(math.sqrt(N)))
        return y


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    model = CoAtNet(3, 224)
    output = model(dummy_input)
    print(output.shape)
