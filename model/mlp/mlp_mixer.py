"""
MindSpore implementation of 'MLP-Mixer'
Refer to "MLP-Mixer: An all-MLP Architecture for Vision"
"""
import mindspore as ms
from mindspore import nn


class MlpBlock(nn.Cell):
    """ Mlp Block """
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Dense(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Dense(mlp_dim, input_dim)

    def construct(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Cell):
    """ Mixer Block """
    def __init__(self, tokens_mlp_dim=16, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super().__init__()
        self.ln = nn.LayerNorm([channels_mlp_dim])
        self.tokens_mlp_block = MlpBlock(tokens_mlp_dim, tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock(channels_mlp_dim, channels_hidden_dim)

    def construct(self, x):
        out = self.ln(x).transpose((0, 2, 1))
        out = self.tokens_mlp_block(out)

        out = out.transpose((0, 2, 1))
        y = x + out
        out = self.ln(y)
        out = y + self.channels_mlp_block(out)
        return out


class MlpMixer(nn.Cell):
    """ Mlp Mixer """
    def __init__(self, num_classes, num_blocks, patch_size, tokens_hidden_dim,
                 channels_hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.patch_size = patch_size
        self.token_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.embedding = nn.Conv2d(3, channels_mlp_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm([channels_mlp_dim])
        self.mlp_blocks = []
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim))
        self.fc = nn.Dense(channels_mlp_dim, num_classes)

    def construct(self, x):
        y = self.embedding(x)
        B, C, _, _ = y.shape
        y = y.view(B, C, -1).transpose((0, 2, 1))
        assert self.token_mlp_dim == y.shape[1]

        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)

        y = self.ln(y)
        y = ms.ops.mean(y, axis=1, keep_dims=False)
        probs = self.fc(y)
        return probs


if __name__ == '__main__':
    dummy_input = ms.ops.randn(50, 3, 40, 40)
    mlp_mixer = MlpMixer(num_classes=1000, num_blocks=10, patch_size=10, tokens_hidden_dim=32, channels_hidden_dim=1024,
                         tokens_mlp_dim=16, channels_mlp_dim=1024)
    output = mlp_mixer(dummy_input)
    print(output.shape)
