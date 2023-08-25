"""
MindSpore implementation of 'SGE'
Refer to "Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks"
"""
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, HeNormal, Normal


class SpatialGroupEnhance(nn.Cell):
    """ Spatial Group Enhance """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = ms.Parameter(ms.ops.zeros((1, groups, 1, 1)))
        self.bias = ms.Parameter(ms.ops.zeros((1, groups, 1, 1)))
        self.sig = nn.Sigmoid()
        self.apply(self.init_weight)

    def init_weight(self, cell):
        """ initializer weight """
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(mode='fan_out'), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data('ones', cell.gamma.shape, cell.gamma.dtype)
            cell.beta.set_data('zeros', cell.beta.shape, cell.beta.dtype)
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(sigma=0.001), cell.weight.shape, cell.bias.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        B, C, H, W = x.shape
        x = x.view(B * self.groups, -1, H, W)
        xn = x * self.avg_pool(x)
        xn = xn.sum(axis=1, keepdims=True)
        t = xn.view(B * self.groups, -1)

        t -= t.mean(axis=1, keep_dims=True)
        std = t.std(axis=1, keepdims=True) + 1e-5
        t /= std
        t = t.view(B, self.groups, H, W)

        t *= self.weight + self.bias
        t = t.view(B * self.groups, 1, H, W)
        x *= self.sig(t)
        x = x.view(B, C, H, W)
        return x


if __name__ == "__main__":
    dummy_input = ms.ops.randn((50, 512, 7, 7))
    sge = SpatialGroupEnhance(groups=8)
    output = sge(dummy_input)
    print(output.shape)
