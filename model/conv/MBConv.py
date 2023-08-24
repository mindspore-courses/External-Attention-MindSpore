""" MBConv """
import math
from functools import partial
import mindspore as ms
from mindspore import nn


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
    output = inputs / keep_prob * binary_tensor
    return output


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


if __name__ == "__main__":
    in_tensor = ms.ops.randn((1, 3, 112, 112))
    mbcblock = MBConvBlock(ksize=3, input_filters=3, out_filters=3, image_size=112)
    out = mbcblock(in_tensor)
    print(out.shape)
