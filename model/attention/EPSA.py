"""
MindSpore implementation of 'EPSA'
Refer to "EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network"
"""
import mindspore as ms
from mindspore import nn


class EPSA(nn.Cell):
    """ EPSANet """
    def __init__(self, channels=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        self.convs = []
        for i in range(S):
            self.convs.append(
                nn.Conv2d(channels // S, channels // S, kernel_size=2 * (i + 1) + 1, pad_mode='pad', padding=i + 1))

        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.SequentialCell(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels // S, channels // (S * reduction), kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channels // (S * reduction), channels // S, kernel_size=1),
                nn.Sigmoid()
            ))
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        B, C, H, W = x.shape

        SPC_out = x.view(B, self.S, C // self.S, H, W)
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = ms.ops.stack(se_out, axis=1)
        SE_out = SE_out.expand_as(SPC_out)

        softmax_out = self.softmax(SE_out)

        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(B, -1, H, W)
        return PSA_out


if __name__ == '__main__':
    dummy_input = ms.ops.randn((50, 512, 7, 7))
    epsa = EPSA(channels=512, reduction=8)
    output = epsa(dummy_input)
    print(output.shape)
