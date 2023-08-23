"""
MindSpore implementation of 'SimAM'
Refer to "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"
"""
import mindspore as ms
from mindspore import nn


class SimAM(nn.Cell):
    """ SimAM """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def construct(self, x):
        _, _, H, W = x.shape
        n = W * H - 1

        x_minus_mu_square = (x - x.mean(axis=(2, 3), keep_dims=True)).pow(2)
        y = x_minus_mu_square / (4 * x_minus_mu_square.sum(axis=(2, 3), keepdims=True) / n + self.e_lambda) + 0.5
        return x * self.activation(y)


if __name__ == '__main__':
    dummy_input = ms.ops.randn(3, 64, 7, 7)
    model = SimAM(64)
    outputs = model(dummy_input)
    print(outputs.shape)
