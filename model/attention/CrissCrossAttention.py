"""
MindSpore implementation of 'CrissCrossAttention'
Refer to "CCNet: Criss-Cross Attention for Semantic Segmentation"
"""
import mindspore as ms
from mindspore import nn


def INF(B, H, W):
    """ return the diagonal matrix, with the value inf """
    return ms.ops.diag(ms.Tensor(float("inf")).repeat(H)).unsqueeze(0).repeat(B * W, 0)


class CrissCrossAttention(nn.Cell):
    """ CrissCrossAttention """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.softmax = nn.Softmax(axis=3)

        self.gamma = ms.Parameter(ms.ops.zeros(1))

    def construct(self, x):
        B, _, H, W = x.shape

        proj_query = self.query_conv(x)
        # print(proj_query.permute(0, 3, 1, 2)view(B*W, -1, ).shape)
        proj_query_h = proj_query.permute(0, 3, 1, 2).view(B * W, -1, H).permute(0, 2, 1)

        proj_query_w = proj_query.permute(0, 2, 1, 3).view(B * H, -1, W).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_h = proj_key.permute(0, 3, 1, 2).view(B * W, -1, H)
        proj_key_w = proj_key.permute(0, 2, 1, 3).view(B * H, -1, W)

        proj_value = self.value_conv(x)
        proj_value_h = proj_value.permute(0, 3, 1, 2).view(B * W, -1, H)
        proj_value_w = proj_value.permute(0, 2, 1, 3).view(B * H, -1, W)

        energy_h = ms.ops.bmm(proj_query_h, proj_key_h).view(B, H, W, W).permute(0, 2, 1, 3) + INF(B, H, W).view(B, W,
                                                                                                                 H,
                                                                                                                 H).permute(
            0, 2, 1, 3)
        energy_w = ms.ops.bmm(proj_query_w, proj_key_w).view(B, H, W, W)
        concate = self.softmax(ms.ops.cat([energy_h, energy_w], axis=3))

        att_h = concate[:, :, :, 0: H].permute(0, 2, 1, 3).view(B * W, H, H)
        att_w = concate[:, :, :, H: H + W].view(B * H, W, W)

        out_h = ms.ops.bmm(proj_value_h, att_h.permute(0, 2, 1)).view(B, W, -1, H).permute(0, 2, 3, 1)
        out_w = ms.ops.bmm(proj_value_w, att_w.permute(0, 2, 1)).view(B, H, -1, W).permute(0, 2, 1, 3)

        return self.gamma * (out_h + out_w) + x


if __name__ == "__main__":
    dummy_input = ms.ops.randn(50, 512, 7, 7)
    att = CrissCrossAttention(512)
    output = att(dummy_input)
    print(output.shape)
