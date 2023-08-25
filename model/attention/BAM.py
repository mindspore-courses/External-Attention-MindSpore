"""
MindSpore implementation of 'BAM'
Refer to "BAM: Bottleneck Attention Module"
"""
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, HeNormal, Normal


class Flatten(nn.Cell):
    """ Flatten """
    def construct(self, x):
        return x.view(x.shape[0], -1)


class ChannelGate(nn.Cell):
    """ Channel Attention """
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super().__init__()
        # self.gate_activation = gate_channel
        self.gate_c = nn.SequentialCell()
        self.gate_c.append(Flatten())

        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.append(nn.Dense(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.append(nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.append(nn.ReLU())

        self.gate_c.append(nn.Dense(gate_channels[-2], gate_channels[-1]))

    def construct(self, in_tensor):
        avg_pool = ms.ops.avg_pool2d(in_tensor, in_tensor.shape[2], stride=in_tensor.shape[2])
        out = self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
        return out


class SpatialGate(nn.Cell):
    """ Spatial Attention """
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super().__init__()
        self.gate_s = nn.SequentialCell()
        self.gate_s.append(nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.append(nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.append(nn.ReLU())
        for _ in range(dilation_conv_num):
            self.gate_s.append(nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,
                                         kernel_size=3, pad_mode='pad',
                                         padding=dilation_val, dilation=dilation_val))
            self.gate_s.append(nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.append(nn.ReLU())
        self.gate_s.append(nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def construct(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAMBlock(nn.Cell):
    """ BAM """
    def __init__(self, gate_channel):
        super().__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(mode='fan_out'), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(sigma=0.001), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        sa_out = self.spatial_att(x)
        ca_out = self.channel_att(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out


if __name__ == "__main__":
    dummy_input = ms.ops.randn(12, 128, 14, 14)
    bam = BAMBlock(128)
    output = bam(dummy_input)
    print(output.shape)
