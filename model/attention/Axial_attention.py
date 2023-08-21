"""
MindSpore implementation of 'Axial Attention'
Refer to " Axial Attention in Multidimensional Transformers"
"""
from operator import itemgetter
import mindspore as ms
from mindspore import nn


def calculate_permutations(num_dimensions, emb_dim):
    """ calculate permutations """
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


def map_el_ind(arr, ind):
    """ get item from arr with indices """
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    """ sort and return indices """
    indices = list(range(len(arr)))
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


class PermuteToFrom(nn.Cell):
    """ PermuteToFrom """
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def construct(self, x, **kwargs):
        axial = x.permute(*self.permutation)

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation)
        return axial


class SelfAttention(nn.Cell):
    """ Self Attention of Axial Attention """
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Dense(dim, dim_hidden, has_bias=False)
        self.to_k = nn.Dense(dim, dim_hidden, has_bias=False)
        self.to_v = nn.Dense(dim, dim_hidden, has_bias=False)
        self.to_out = nn.Dense(dim_hidden, dim)

    def construct(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        b, _, d, h, e = *q.shape, self.heads, self.dim_heads

        def merge_heads(item):
            return item.reshape(b, -1, h, e).transpose(0, 2, 1, 3).reshape(b * h, -1, e)

        # merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(0, 2, 1, 3).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = q @ k.transpose(0, 2, 1) * (e ** 0.5)
        dots = ms.ops.softmax(dots, axis=-1)
        out = dots @ v

        out = out.reshape(b, h, -1, e).transpose(0, 2, 1, 3).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class AxialAttention(nn.Cell):
    """ Axial Attention """
    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.SequentialCell(attentions)
        self.sum_axial_out = sum_axial_out

    def construct(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


if __name__ == '__main__':
    dummy_input = ms.ops.randn((1, 8, 256, 256))
    model = AxialAttention(dim=256)
    output = model(dummy_input)
    print(output.shape)
