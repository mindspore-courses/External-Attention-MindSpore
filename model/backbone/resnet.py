"""
MindSpore implementation of `ResNet`.
Refer to "Deep Residual Learning for Image Recognition"
"""
import mindspore as ms
from mindspore import nn


class BottleNeck(nn.Cell):
    """ Bottle Neck """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode='pad',
                               padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * BottleNeck.expansion,
                               kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        return self.relu(out)


class ResNet(nn.Cell):
    """ ResNet """
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode='pad', padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Dense(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(-1)

    def _make_layer(self, block, channels, blocks, stride=1):
        """ make layer """
        if stride != 1 or self.in_channels != channels * block.expansion:
            self.downsample = nn.Conv2d(self.in_channels, channels * block.expansion, stride=stride, kernel_size=1)

        layers = [block(self.in_channels, channels, downsample=self.downsample, stride=stride)]
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.classifier(out)
        out = self.softmax(out)

        return out


def ResNet50(num_classes=1000):
    """ resnet50 """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=1000):
    """ resnet101 """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=1000):
    """ resnet152 """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)


if __name__ == "__main__":
    dummy_input = ms.ops.randn(50, 3, 224, 224)
    resnet50 = ResNet50()
    resnet101 = ResNet101()
    resnet152 = ResNet152()
    print(f"resnet50 output shape: {resnet50(dummy_input).shape}")
    print(f"resnet101 output shape: {resnet101(dummy_input).shape}")
    print(f"resnet152 output shape: {resnet152(dummy_input).shape}")
