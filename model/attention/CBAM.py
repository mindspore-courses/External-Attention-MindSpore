"""
MindSpore implementation of 'CBAM'
Refer to "CBAM: Convolutional Block Attention Module"
"""
import mindspore as ms
from mindspore import nn


class BasicConv(nn.Cell):
    """ Basic Convolution """
    def __init__(self, in_planes, out_planes,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', padding=padding,
                              dilation=dilation, group=groups, has_bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Cell):
    """ Flatten """
    def construct(self, x):
        return x.view(x.shape[0], -1)


class ChannelGate(nn.Cell):
    """ Channel Attention """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None):
        super().__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.gate_channels = gate_channels
        self.mlp = nn.SequentialCell(
            Flatten(),
            nn.Dense(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Dense(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def construct(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = ms.ops.avg_pool2d(x, kernel_size=(x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = ms.ops.max_pool2d(x, kernel_size=(x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = ms.ops.lp_pool2d(x, norm_type=1, kernel_size=(x.shape[2], x.shape[3]),
                                           stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = ms.ops.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    """ 2d logsumexp """
    tensor_flatten = tensor.view(tensor.shape[0], tensor.shape[1], -1)
    s, _ = ms.ops.max(tensor_flatten, axis=2, keepdims=True)
    outputs = s + (tensor_flatten - s).exp().sum(axis=2, keepdims=True).log()
    return outputs


class ChannelPool(nn.Cell):
    """ Channel Pool """
    def construct(self, x):
        return ms.ops.cat((ms.ops.max(x, 1)[0].unsqueeze(1), ms.ops.mean(x, 1).unsqueeze(1)), axis=1)


class SpatialGate(nn.Cell):
    """ Spatial Attention """
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def construct(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = ms.ops.sigmoid(x_out)
        return x * scale


class CBAM(nn.Cell):
    """ CBAM """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, no_spatial=False):
        super().__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def construct(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


if __name__ == "__main__":
    dummy_input = ms.ops.randn(12, 128, 14, 14)
    model = CBAM(128)
    output = model(dummy_input)
    print(output.shape)
