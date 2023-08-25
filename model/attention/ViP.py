"""
MindSpore implementation of 'ViP'
Refer to "Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"
"""
import mindspore as ms
from mindspore import nn


class MLP(nn.Cell):
    """ mlp """
    def __init__(self, in_chans, hidden_chans, out_chans, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Dense(in_chans, hidden_chans)
        self.act_layer = act_layer()
        self.fc2 = nn.Dense(hidden_chans, out_chans)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        return self.drop(self.fc2(self.act_layer(self.fc1(x))))


class WeightedPermuteMLP(nn.Cell):
    """ Weighted Permute MLP """
    def __init__(self, dim, seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim = seg_dim
        self.mlp_c = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.mlp_h = nn.Dense(dim, dim, has_bias=qkv_bias)
        self.mlp_w = nn.Dense(dim, dim, has_bias=qkv_bias)

        self.reweighting = MLP(dim, dim // 4, dim * 3)

        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x):
        B, H, W, C = x.shape
        c_embed = self.mlp_c(x)

        S = C // self.seg_dim
        h_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.seg_dim, W, H * S)
        h_embed = self.mlp_h(h_embed).reshape(B, self.seg_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 1, 2, 4).reshape(B, self.seg_dim, H, W * S)
        w_embed = self.mlp_h(h_embed).reshape(B, self.seg_dim, H, W, S).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        weight = (c_embed + h_embed + w_embed).permute(0, 3, 1, 2).flatten(start_dim=2, end_dim=-1).mean(2)
        weight = ms.ops.softmax(self.reweighting(weight).reshape(B, C, 3).permute(2, 0, 1), axis=0).unsqueeze(
            2).unsqueeze(2)

        x = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]
        x = self.proj_drop(self.proj(x))

        return x


if __name__ == '__main__':
    dummy_input = ms.ops.randn((64, 8, 8, 512))
    vip = WeightedPermuteMLP(512, 8)
    out = vip(dummy_input)
    print(out.shape)
