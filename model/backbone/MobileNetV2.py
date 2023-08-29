"""
MindSpore implementation of `MobileNetV2`.
Refer to "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
"""
import mindspore as ms
from mindspore import nn


class InvertedBlock(nn.Cell):
    """ Inverted Block """
    def __init__(self, in_chans, out_chans, expand_ratio, stride):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = in_chans * expand_ratio
        self.use_res_connect = self.stride == 1 and in_chans == out_chans

        layers = []
        if expand_ratio != 1:
            layers.append(nn.SequentialCell(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=3,
                          pad_mode='pad', padding=1, stride=stride, group=in_chans, has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6()
            ))
        layers.append(nn.SequentialCell(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      pad_mode='pad', padding=1, stride=stride, group=in_chans, has_bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6()
        ))
        layers.append(nn.SequentialCell(
            nn.Conv2d(hidden_dim, out_chans, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU6()
        ))

        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        if self.use_res_connect:
            return x + self.layers(x)

        return self.layers(x)


class MobileNetV2(nn.Cell):
    """ MobileNet V2 """
    def __init__(self, in_chans=3, num_classes=1000):
        super().__init__()
        self.configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = nn.SequentialCell(
            nn.Conv2d(in_chans, 32, kernel_size=3, pad_mode='pad', padding=1, stride=2, has_bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(input_channel, c, expand_ratio=t, stride=stride))
                input_channel = c
        self.layers = nn.SequentialCell(*layers)

        self.last_conv = nn.SequentialCell(
            nn.Conv2d(input_channel, 1280, kernel_size=1, padding=0, stride=1, has_bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )

        self.classifier = nn.SequentialCell(
            nn.Dropout2d(p=0.2),
            nn.Dense(1280, num_classes)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def construct(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    net = MobileNetV2(in_chans=3, num_classes=1000)
    output = net(dummy_input)
    print(output.shape)
