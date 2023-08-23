"""
MindSpore implementation of 'SKAttention'
Refer to "Selective Kernel Networks"
"""
from collections import OrderedDict
import mindspore as ms
from mindspore import nn


class SKAttention(nn.Cell):
    """ SKAttention """
    def __init__(self,
                 channels=512,
                 kernels=None,
                 reduction=16,
                 group=1,
                 L=32):
        super().__init__()
        if kernels is None:
            kernels = [1, 3, 5, 7]
        self.d = max(L, channels // reduction)
        self.convs = nn.SequentialCell()
        for k in kernels:
            self.convs.append(
                nn.SequentialCell(OrderedDict([
                    ('conv', nn.Conv2d(channels, channels, kernel_size=k, pad_mode='pad', padding=k // 2, group=group)),
                    ('bn', nn.BatchNorm2d(channels)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Dense(channels, self.d)
        self.fcs = nn.SequentialCell()
        for _ in range(len(kernels)):
            self.fcs.append(nn.Dense(self.d, channels))
        self.softmax = nn.Softmax(axis=0)

    def construct(self, x):
        B, C, _, _ = x.shape
        conv_outs = []

        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = ms.ops.stack(conv_outs, 0)

        # fuse
        U = sum(conv_outs)

        # reduction channels
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)

        # calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(B, C, 1, 1))
        attention_weights = ms.ops.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)

        # fuse
        V = (attention_weights * feats).sum(0)
        return V


if __name__ == "__main__":
    dummy_input = ms.ops.randn((50, 512, 7, 7))
    se = SKAttention(channels=512, reduction=8)
    output = se(dummy_input)
    print(output.shape)
