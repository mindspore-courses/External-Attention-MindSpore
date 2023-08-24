""" DynamicConv """
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, HeNormal, HeUniform


class Attention(nn.Cell):
    """Attention with temperature
    """
    def __init__(self, in_channels, ratio, K, temperature=30, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.temperature = temperature
        assert in_channels > ratio
        hidden_channels = in_channels // ratio
        self.net = nn.SequentialCell(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, K, kernel_size=1, has_bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        if init_weight:
            self.apply(self.init_weight)

    def update_temperature(self):
        """update temperature
        """
        if self.temperature > 1:
            self.temperature -= 1

    def init_weight(self, cell):
        """init Conv2d and BatchNorm2d
        """
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

    def construct(self, x):
        att = self.avgpool(x)
        att = self.net(att).view(x.shape[0], -1)
        return self.sigmoid(att)


class DynamicConv(nn.Cell):
    """DynamicConv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 dilation=1, groups=1, bias=True, K=4, temperature=30, ratio=4, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = Attention(in_channels, ratio=ratio, K=K, temperature=temperature, init_weight=init_weight)

        self.weight = ms.Parameter(
            ms.ops.randn((K, out_channels, in_channels // groups, kernel_size, kernel_size)), requires_grad=True)
        if bias:
            self.bias = ms.Parameter(ms.ops.randn((K, out_channels)), requires_grad=True)
        else:
            self.bias = None

        if init_weight:
            self.init_weight()

    def init_weight(self):
        """init weight with HeUniform
        """
        for i in range(self.K):
            self.weight[i] = initializer(HeUniform(), self.weight[i].shape, self.weight[i].dtype)

    def construct(self, x):
        B, _, H, W = x.shape
        softmax_att = self.attention(x)
        x = x.view(1, -1, H, W)
        weight = self.weight.view(self.K, -1)
        aggregate_weight = ms.ops.mm(softmax_att, weight)\
            .view(B*self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size)

        if self.bias is not None:
            bias = self.bias.view(self.K, -1)
            aggregate_bias = ms.ops.mm(softmax_att, bias).view(-1)
            output = ms.ops.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride,
                                   pad_mode='pad', padding=self.padding, groups=self.groups*B, dilation=self.dilation)
        else:
            output = ms.ops.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride,
                                   pad_mode='pad', padding=self.padding, groups=self.groups*B, dilation=self.dilation)
        output = output.view(B, self.out_channels, H, W)
        return output


if __name__ == "__main__":
    in_tensor = ms.ops.randn((2, 32, 64, 64))
    cond = DynamicConv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    out = cond(in_tensor)
    print(out.shape)
