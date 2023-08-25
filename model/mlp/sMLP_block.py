"""
MindSpore implementation of 'sMLP'
Refer to "Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?"
"""
import mindspore as ms
from mindspore import nn


class sMLPBlock(nn.Cell):
    """ sMLPBlock """
    def __init__(self, h=224, w=224, c=3):
        super().__init__()
        self.proj_h = nn.Dense(h, h)
        self.proj_w = nn.Dense(w, w)
        self.fuse = nn.Dense(3 * c, c)

    def construct(self, x):
        x_h = self.proj_h(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = ms.ops.cat([x_h, x_w, x_id], axis=1)
        out = self.fuse(x_fuse.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


if __name__ == "__main__":
    dummy_input = ms.ops.randn((50, 3, 224, 224))
    smlp = sMLPBlock(h=224, w=224)
    output = smlp(dummy_input)
    print(output.shape)
