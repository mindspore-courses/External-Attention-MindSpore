"""
MindSpore implementation of `ConvMixer`.
Refer to Patches Are All You Need?
"""
import mindspore as ms
from mindspore import nn


class Residual(nn.Cell):
    """ Residual """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x):
        return self.fn(x)


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, num_classes=1000):
    """ ConvMixer """
    return nn.SequentialCell(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.SequentialCell(
            Residual(nn.SequentialCell(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, group=dim, pad_mode='pad', padding=kernel_size // 2),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dense(dim, num_classes)
    )


if __name__ == "__main__":
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    convmixer = ConvMixer(dim=512, depth=12)
    output = convmixer(dummy_input)
    print(output.shape)
