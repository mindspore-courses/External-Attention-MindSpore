""" ACmix Attention """
import mindspore as ms
from mindspore import nn


def position(H, W):
    """ get position encode """
    loc_w = ms.ops.linspace(-1., 1., W).unsqueeze(0).repeat(H, axis=0)
    loc_h = ms.ops.linspace(-1., 1., H).unsqueeze(1).repeat(W, axis=1)
    loc = ms.ops.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0)
    loc = loc.reshape(-1, *loc.shape)
    # print(loc)
    return loc


def stride(x, strides):
    """ split x with strides in last two dimension """
    # B, C, H, W = x.shape
    return x[:, :, ::strides, ::strides]


def init_rate_half(tensor):
    """ fill data with 0.5 """
    if isinstance(tensor, ms.Parameter):
        fill_data = ms.ops.ones(tensor.shape) * 0.5
        tensor.set_dtype(fill_data.dtype)
        tensor.data.set_data(fill_data)
    return tensor


def init_rate_0(tensor):
    """ fill data with 0 """
    if isinstance(tensor, ms.Parameter):
        fill_data = ms.ops.zeros(tensor.shape)
        tensor.set_dtype(fill_data.dtype)
        tensor.data.set_data(fill_data)
    return tensor


class ACmix(nn.Cell):
    """ ACmix """
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, strides=1, dilation=1):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = strides
        self.dilation = dilation
        self.rate1 = ms.Parameter(ms.Tensor(1))
        self.rate2 = ms.Parameter(ms.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(ksizes=[1, self.kernel_att, self.kernel_att, 1],
                                strides=[1, self.stride, self.stride, 1],
                                rates=[1, 1, 1, 1])
        self.softmax = nn.Softmax(axis=1)

        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv ** 2, kernel_size=1)
        self.dep_conv = nn.Conv2d(self.kernel_conv ** 2 * self.head_dim, out_planes, kernel_size=self.kernel_conv,
                                  stride=strides, pad_mode='pad', padding=1, has_bias=True, group=self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """ reset parameters """
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = ms.ops.zeros((self.kernel_conv ** 2, self.kernel_conv, self.kernel_conv))
        for i in range(self.kernel_conv ** 2):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.reshape(1, *kernel.shape).repeat(self.out_planes, axis=0)
        self.dep_conv.weight = ms.Parameter(default_input=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def construct(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** 0.5
        B, _, H, W = q.shape
        h_out, w_out = H // self.stride, W // self.stride

        pe = self.conv_p(position(H, W))

        q_att = q.view(B * self.head, self.head_dim, H, W) * scaling
        k_att = k.view(B * self.head, self.head_dim, H, W)
        v_att = v.view(B * self.head, self.head_dim, H, W)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(B * self.head, self.head_dim,
                                                         self.kernel_att ** 2, h_out, w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att ** 2,
                                                        h_out, w_out)

        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(B * self.head, self.head_dim, self.kernel_att ** 2,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(B, self.out_planes, h_out, w_out)

        f_all = self.fc(
            ms.ops.cat([q.view(B, self.head, self.head_dim, H * W), k.view(B, self.head, self.head_dim, H * W),
                        v.view(B, self.head, self.head_dim, H * W)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)
        return self.rate1 * out_att + self.rate2 * out_conv


if __name__ == "__main__":
    in_tensor = ms.ops.randn((50, 256, 7, 7))
    acmix = ACmix(in_planes=256, out_planes=256)
    out = acmix(in_tensor)
    print(out.shape)
