"""
MindSpore implementation of 'MobileViTv2Attention'
Refer to "Separable Self-attention for Mobile Vision Transformers"
"""
import mindspore as ms
from mindspore import nn


class MobileViTv2Attention(nn.Cell):
    """ MobileViTv2Attention """
    def __init__(self, d_model):
        super().__init__()
        self.fc_i = nn.Dense(d_model, 1)
        self.fc_k = nn.Dense(d_model, d_model)
        self.fc_v = nn.Dense(d_model, d_model)
        self.fc_o = nn.Dense(d_model, d_model)

        self.d_model = d_model

    def construct(self, x):
        i = self.fc_i(x)
        weight_i = ms.ops.softmax(i, axis=1)
        context_score = weight_i * self.fc_k(x)
        context_vec = ms.ops.sum(context_score, dim=1, keepdim=True)
        v = self.fc_v(x) * context_vec
        out = self.fc_o(v)
        return out


if __name__ == '__main__':
    dummy_input = ms.ops.randn((50, 49, 512))
    sa = MobileViTv2Attention(d_model=512)
    output = sa(dummy_input)
    print(output.shape)
