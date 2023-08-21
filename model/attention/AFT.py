"""
MindSpore implementation of 'AFT Attention'
Refer to "An Attention Free Transformer"
"""
import mindspore as ms
from mindspore import nn
from mindspore import ops


class AFT_FULL(nn.Cell):
    """ AFT Attention """
    def __init__(self, d_model, n=49, simple=False):
        super().__init__()
        self.fc_q = nn.Dense(d_model, d_model)
        self.fc_k = nn.Dense(d_model, d_model)
        self.fc_v = nn.Dense(d_model, d_model)
        if simple:
            self.position_biases = ms.ops.zeros((n, n))
        else:
            self.position_biases = ms.Parameter(ms.ops.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        B, N, D = x.shape

        q = self.fc_q(x)
        k = self.fc_k(x).view(1, B, N, D)
        v = self.fc_v(x).view(1, B, N, D)

        numerator = ops.sum(ops.exp(k + self.position_biases.view(N, 1, -1, 1)) * v, dim=2)
        denominator = ops.sum(ops.exp(k + self.position_biases.view(N, 1, -1, 1)), dim=2)

        out = numerator / denominator
        out = self.sigmoid(q) * (out.permute(1, 0, 2))
        return out


if __name__ == "__main__":
    dummy_input = ms.ops.randn((50, 49, 512))
    aft_full = AFT_FULL(d_model=512, n=49)
    output = aft_full(dummy_input)
    print(output.shape)
