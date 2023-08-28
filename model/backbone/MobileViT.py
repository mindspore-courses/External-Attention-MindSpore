"""
MindSpore implementation of `MobileViT`.
Refer to MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
"""
import mindspore as ms
from mindspore import nn


def conv_bn(inp, oup, kernel_size=3, stride=1):
    """ conv and batchnorm """
    return nn.SequentialCell(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, pad_mode='pad', padding=kernel_size//2),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Cell):
    """ PreNorm """
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm((dim, ))
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Cell):
    """ FeedForward """
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Dense(mlp_dim, dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        return self.net(x)


class Attention(nn.Cell):
    """ Attention """
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not(heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(axis=-1)
        self.to_qkv = nn.Dense(dim, inner_dim*3, has_bias=False)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        ) if project_out else nn.Identity()

    def construct(self, x):
        q, k, v = self.to_qkv(x).chunk(3, axis=-1)
        b, p, n, hd = q.shape
        h, d = self.heads, hd // self.heads
        q = q.view(b, p, n, h, d).permute(0, 1, 3, 2, 4)
        k = k.view(b, p, n, h, d).permute(0, 1, 3, 2, 4)
        v = v.view(b, p, n, h, d).permute(0, 1, 3, 2, 4)

        dots = ms.ops.matmul(q, k.transpose(0, 1, 2, 4, 3)) * self.scale
        attn = self.attend(dots)
        out = ms.ops.matmul(attn, v)

        out = out.permute(0, 1, 3, 2, 4).view(b, p, n, -1)
        return self.to_out(out)


class Transformer(nn.Cell):
    """ Transformer layer """
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.SequentialCell()
        for _ in range(depth):
            self.layers.append(nn.SequentialCell(
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ))

    def construct(self, x):
        out = x
        for att, ffn in self.layers:
            out += att(out)
            out += ffn(out)
        return out


class MobileViTAttention(nn.Cell):
    """ MobileViT Attention """
    def __init__(self, in_channels=3, dim=512, kernel_size=3, patch_size=7, depth=3, mlp_dim=1024):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               pad_mode='pad', padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=depth, heads=8, head_dim=64, mlp_dim=mlp_dim)

        self.conv3 = nn.Conv2d(dim, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(2*in_channels, in_channels, kernel_size=kernel_size,
                               pad_mode='pad', padding=kernel_size//2)

    def construct(self, x):
        y = self.conv1(x)
        y = self.conv2(y)

        B, C, H, W = y.shape

        nH, pH, nW, pW = H // self.ph, self.ph, W // self.pw, self.pw

        y = y.view(B, C, nH, pH, nW, pW).permute(0, 3, 5, 2, 4, 1)
        y = y.view(B, pH * pW, nH * nW, C)

        y = self.trans(y)
        y = y.view(B, pH, pW, nH, nW, C).permute(0, 3, 5, 2, 4, 1)
        y = y.view(B, C, H, W)

        y = self.conv3(y)
        y = ms.ops.cat([x, y], 1)
        y = self.conv4(y)

        return y


class MV2Block(nn.Cell):
    """ MV2 Block """
    def __init__(self, inp, out, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expansion
        self.use_res_connection = stride == 1 and inp == out

        if expansion == 1:
            self.conv = nn.SequentialCell(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride,
                          pad_mode='pad', padding=1, group=hidden_dim, has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out, kernel_size=1, stride=1, has_bias=False),
                nn.BatchNorm2d(out)
            )
        else:
            self.conv = nn.SequentialCell(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          pad_mode='pad', padding=1, group=hidden_dim, has_bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out, kernel_size=1, stride=1, has_bias=False),
                nn.SiLU(),
                nn.BatchNorm2d(out)
            )

    def construct(self, x):
        if self.use_res_connection:
            return x + self.conv(x)
        return self.conv(x)


class MobileViT(nn.Cell):
    """ MobileViT """
    def __init__(self, image_size, dims, channels, num_classes, depths=None,
                 kernel_size=3, patch_size=2):
        super().__init__()
        if depths is None:
            depths = [2, 4, 3]
        ih, iw = image_size, image_size
        ph, pw = patch_size, patch_size
        assert iw % pw == 0 and ih % ph == 0

        self.conv1 = conv_bn(3, channels[0], kernel_size=3, stride=patch_size)
        self.mv2 = nn.SequentialCell()
        self.m_vits = nn.SequentialCell()

        self.mv2.append(MV2Block(channels[0], channels[1], 1))
        self.mv2.append(MV2Block(channels[1], channels[2], 2))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))
        self.mv2.append(MV2Block(channels[3], channels[4], 2))
        self.m_vits.append(MobileViTAttention(channels[4], dim=dims[0], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[0], mlp_dim=int(2 * dims[0])))
        self.mv2.append(MV2Block(channels[4], channels[5], 2))
        self.m_vits.append(MobileViTAttention(channels[5], dim=dims[1], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[1], mlp_dim=int(4 * dims[1])))
        self.mv2.append(MV2Block(channels[5], channels[6], 2))
        self.m_vits.append(MobileViTAttention(channels[6], dim=dims[2], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[2], mlp_dim=int(4 * dims[2])))

        self.conv2 = conv_bn(channels[-2], channels[-1], kernel_size=1)
        self.pool = nn.AvgPool2d(image_size // 2, 1)
        self.fc = nn.Dense(channels[-1], num_classes)

    def construct(self, x):
        y = self.conv1(x)
        for i in range(5):
            y = self.mv2[i](y)

        y = self.m_vits[0](y)
        y = self.mv2[5](y)
        y = self.m_vits[1](y)
        y = self.mv2[6](y)
        y = self.m_vits[2](y)

        y = self.conv2(y)
        y = self.pool(y).view(y.shape[0], -1)
        y = self.fc(y)
        return y


def mobilevit_xxs(image_size=224, num_classes=1000):
    """ mobilevit-xxs """
    dims = [60, 80, 96]
    channels = [16, 16, 24, 24, 48, 64, 80, 320]
    return MobileViT(image_size, dims, channels, num_classes)


def mobilevit_xs(image_size=224, num_classes=1000):
    """ mobilevit-xs """
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 80, 96, 384]
    return MobileViT(image_size, dims, channels, num_classes)


def mobilevit_s(image_size=224, num_classes=1000):
    """ mobilevit-s """
    dims = [144, 196, 240]
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    return MobileViT(image_size, dims, channels, num_classes)


if __name__ == "__main__":
    dummy_input = ms.ops.randn((1, 3, 224, 224))
    mvit_xxs = mobilevit_xxs()
    output = mvit_xxs(dummy_input)
    print(output.shape)
