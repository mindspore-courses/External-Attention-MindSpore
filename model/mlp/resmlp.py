"""
MindSpore implementation of 'resmlp'
Refer to "ResMLP: Feedforward networks for image classification with data-efficient training"
"""
import mindspore as ms
from mindspore import nn


class Rearrange(nn.Cell):
    """ rearrange """
    def __init__(self, image_size=14, patch_size=7):
        super().__init__()
        self.h = patch_size
        self.w = patch_size
        self.nw = image_size // patch_size
        self.nh = image_size // patch_size

    def construct(self, x):
        B, C, _, _ = x.shape
        out = x.reshape(B, C, self.h, self.nh, self.w, self.nw)
        out = out.permute(0, 3, 5, 2, 4, 1)
        out = out.view(B, self.nh * self.nw, -1)
        return out


class Affine(nn.Cell):
    """ affine """
    def __init__(self, channels):
        super().__init__()
        self.g = ms.Parameter(ms.ops.ones((1, 1, channels)))
        self.b = ms.Parameter(ms.ops.zeros((1, 1, channels)))

    def construct(self, x):
        return x * self.g * self.b


class PreAffinePostLayerScale(nn.Cell):
    """ pre affine post layer scale """
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 < depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = ms.ops.fill(ms.float32, (1, 1, dim), init_eps)
        self.scale = ms.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def construct(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class ResMLP(nn.Cell):
    """ resmlp """
    def __init__(self, dim=128, image_size=14, patch_size=7, expansion_factor=4, depth=4, class_num=1000):
        super().__init__()
        self.flatten = Rearrange(image_size, patch_size)
        # num_patches = (image_size // patch_size) ** 2

        def wrapper(idx, fn):
            return PreAffinePostLayerScale(dim, idx + 1, fn)
        # wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)
        self.embedding = nn.Dense((patch_size ** 2) * 3, dim)
        self.mlp = nn.SequentialCell()
        for i in range(depth):
            self.mlp.insert_child_to_cell(f'fc1_{i}', wrapper(i, nn.Conv1d(patch_size ** 2, patch_size ** 2, 1)))
            self.mlp.insert_child_to_cell(f'fc1_{i}', wrapper(i, nn.SequentialCell(
                                                                    nn.Dense(dim, dim * expansion_factor),
                                                                    nn.GELU(),
                                                                    nn.Dense(dim * expansion_factor, dim)
                                                                )))
        self.aff = Affine(dim)
        self.classifier = nn.Dense(dim, class_num)
        self.softmax = nn.Softmax(1)

    def construct(self, x):
        y = self.flatten(x)
        y = self.embedding(y)
        y = self.mlp(y)
        y = self.aff(y)
        y = y.mean(axis=1)
        return self.softmax(self.classifier(y))


if __name__ == "__main__":
    dummy_input = ms.ops.randn((50, 3, 14, 14))
    resmlp = ResMLP(dim=128, image_size=14, patch_size=7, class_num=1000)
    output = resmlp(dummy_input)
    print(output.shape)
