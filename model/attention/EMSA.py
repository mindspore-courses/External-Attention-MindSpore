"""
MindSpore implementation of 'EMSA'
Refer to "ResT: An Efficient Transformer for Visual Recognition"
"""
import mindspore as ms
from mindspore import nn


class EMSA(nn.Cell):
    """ EMSA """
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=3, apply_transform=True):
        super().__init__()
        self.H = H
        self.W = W
        self.fc_q = nn.Dense(d_model, h * d_k)
        self.fc_k = nn.Dense(d_model, h * d_k)
        self.fc_v = nn.Dense(d_model, h * d_v)
        self.fc_o = nn.Dense(h * d_v, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.ratio = ratio

        if self.ratio > 1:
            self.sr = nn.SequentialCell()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, pad_mode='pad',
                                     padding=ratio // 2, group=d_model)
            self.sr_ln = nn.LayerNorm([d_model])

        self.apply_transform = apply_transform
        if self.apply_transform:
            self.transform = nn.SequentialCell(
                nn.Conv2d(h, h, kernel_size=1, stride=1),
                nn.Softmax(-1),
                nn.InstanceNorm2d(h)
            )

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def construct(self, queries, keys, values, attantion_mask=None, attention_weights=None):
        B, Nq, C = queries.shape
        Nk = keys.shape[1]

        q = self.fc_q(queries).view(B, Nq, self.h, self.d_k).permute(0, 2, 1, 3)

        if self.ratio > 1:
            x = queries.permute(0, 2, 1).view(B, C, self.H, self.W)
            x = self.sr_conv(x)
            x = x.view(B, C, -1).permute(0, 2, 1)
            x = self.sr_ln(x)
            k = self.fc_k(keys).view(B, -1, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.fc_v(values).view(B, Nk, self.h, self.d_v).permute(0, 2, 1, 3)
        else:
            k = self.fc_k(keys).view(B, Nk, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.fc_v(values).view(B, Nk, self.h, self.d_v).permute(0, 2, 1, 3)

        if self.apply_transform:
            att = ms.ops.matmul(q, k) / ms.numpy.sqrt(self.d_k)
            att = self.transform(att)
        else:
            att = ms.ops.matmul(q, k) / ms.numpy.sqrt(self.d_k)
            att = ms.ops.softmax(att, -1)

        if attention_weights is not None:
            att = att * attention_weights
        if attantion_mask is not None:
            att = att.masked_fill(attantion_mask, -float('inf'))

        att = self.dropout(att)
        out = ms.ops.matmul(att, v).permute(0, 2, 1, 3).view(B, Nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


if __name__ == "__main__":
    dummy_input = ms.ops.randn(50, 64, 512)
    emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
    output = emsa(dummy_input, dummy_input, dummy_input)
    print(output.shape)
