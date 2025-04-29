from matplotlib.pyplot import sca
from torch._prims_common import Dim
from torchinfo import summary
import torch
from torch import nn
from torch.nn import Softmax, functional as F
import numpy as np
from scipy.signal import butter
from bnn_ops import Binarize, LBitTanh, LBit, lbit

STEP_MEMORY = False

LOG = False


# 一种稀疏的类softmax方法，用于替换nn.Softmax，提取上半部分的概率密度
class SoftMax(nn.Module):
    def __init__(self, dim=1):
        super(SoftMax, self).__init__()
        self.dim = dim

    def expi32(self, x):
        x = (x * 64.).to(torch.int32)
        x0 = torch.tensor(64, dtype=torch.int32)
        x1 = x  # 6
        x2 = torch.div(x1 * x1, 2, rounding_mode='trunc')  # 12
        x3 = torch.div(x1 * x2, 3, rounding_mode='trunc')  # 18
        x4 = torch.div(x1 * x3, 4, rounding_mode='trunc')  # 24
        temp = x0 * 64 ** 3 + x1 * 64 ** 3 + x2 * 64 ** 2 + x3 * 64 ** 1 + x4
        o = torch.div(64 * x0 * 64 ** 3, temp, rounding_mode='trunc')
        return o / 64.

    def expf32_n4_f(self, x):
        x1 = x
        x2 = x1 * x1 / 2
        x3 = x2 * x1 / 3
        x4 = x3 * x1 / 4
        temp = 1 + x1 + x2 + x3 + x4
        o = lbit(1 / temp, 2 ** 6)
        return o

    def forward(self, x):
        x = LBit.apply(x)
        x = torch.max(x, dim=self.dim, keepdim=True)[0] - x
        x = self.expf32_n4_f(x)
        x = x / x.sum(dim=self.dim, keepdim=True)
        return LBit.apply(x)


class QConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 b_weight=True):
        super(QConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias, dtype=torch.float32)
        self.b_weight = b_weight

        # Initialize weights and biases
        nn.init.xavier_normal_(self.weight)
        if bias:
            nn.init.constant_(self.bias, 0)

        self.mem = None
        self.mem_length = kernel_size - stride

    def forward(self, x):
        b = self.bias
        w = self.weight

        x = LBitTanh.apply(x)
        b = LBitTanh.apply(b)

        if self.b_weight:
            w = Binarize.apply(w)
        else:
            w = LBitTanh.apply(w)

        if self.mem != None:
            x = torch.cat([self.mem, x], dim=-1)

        self.mem = x[..., x.shape[-1] - self.mem_length:]

        y = F.conv1d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

        (LOG) and print("QConv", list(x.shape[1:])[::-1], x.shape[1] * x.shape[2], list(y.shape[1:])[::-1], )
        y = LBit.apply(y)
        return y


class Upsample(nn.Upsample):
    def __init__(self, scale_factor=2, mode='linear'):
        super(Upsample, self).__init__(scale_factor=scale_factor, mode=mode)
        self.l = scale_factor // 2
        self.r = scale_factor - self.l
        self.mem = None
        self.mem_length = 1

    def forward(self, x):
        x = LBit.apply(x)
        if self.mem != None:
            x = torch.cat([self.mem, x], dim=-1)

        self.mem = x[..., x.shape[-1] - self.mem_length:]

        y = super().forward(x)[..., self.l:-self.r]
        (LOG and self.mem_length > 0) and print("UpSample", list(x.shape[1:])[::-1], x.shape[1] * x.shape[2])
        return LBit.apply(y)


class AvgPool1d(nn.AvgPool1d):
    def __init__(self, kernel_size, stride):
        super().__init__(kernel_size, stride)
        self.mem = None
        self.mem_length = kernel_size - stride

    def forward(self, x):
        x = LBit.apply(x)
        if self.mem != None:
            x = torch.cat([self.mem, x], dim=-1)
        self.mem = x[..., x.shape[-1] - self.mem_length:]

        y = super().forward(x)

        (LOG and self.mem_length > 0) and print("Pool ", list(x.shape[1:])[::-1], x.shape[1] * x.shape[2])

        return LBit.apply(y)


class ECGSegMCULBit(torch.nn.Module):
    def __init__(self):
        super(ECGSegMCULBit, self).__init__()

        softmax = SoftMax

        self.down = nn.Sequential(
            AvgPool1d(4, stride=4),
            QConv1d(12, 4, 1, 1, b_weight=False),
            QConv1d(4, 4, 3, 1, b_weight=False),
            QConv1d(4, 64, 3, 1, ),
            AvgPool1d(5, stride=5),
            softmax(dim=1),
            QConv1d(64, 4, 1, 1, ),
            QConv1d(4, 4, 3, 1, b_weight=False),
            QConv1d(4, 64, 3, 1, ),
            AvgPool1d(3, stride=1),
            softmax(dim=1),

            QConv1d(64, 4, 1, ),
            QConv1d(4, 4, 3, b_weight=False),
            QConv1d(4, 64, 3),
            AvgPool1d(5, stride=1),
            softmax(dim=1),
        )
        self.up = nn.Sequential(
            Upsample(5),
            QConv1d(64, 8, 1),
            QConv1d(8, 8, 1, b_weight=False),
            QConv1d(8, 64, 1),
            softmax(dim=1),
            Upsample(4),
            QConv1d(64, 8, 1),
            QConv1d(8, 4, 1, b_weight=False),
        )

    def forward(self, x):
        if STEP_MEMORY:
            return self.step(x).transpose(1, 2)
        else:
            x = F.pad(x, (236, 100))
            return self.step(x)[..., 16:].transpose(1, 2)

    def step(self, x):
        x = x / 8
        y = self.down(x)
        y = self.up(y)
        return y

    def test_step(self, x):
        x = x / 8
        y = self.down(x)
        y = self.up(y)
        return y.transpose(1, 2)

    def step_init(self):
        x = torch.zeros((1, 12, 336 + 20))
        return self.step(x)[..., 16:]



if __name__ == "__main__":
    net = ECGSegMCULBit()
    net.load_state_dict(torch.load('model.pth'))
    x = torch.ones((1, 12, 20))
    y = net.step_init()
    print(x.shape, y.shape)
    LOG = True
    print()
    # 遍历net的层
    for n1, c1 in net.named_children():
        for n2, c2 in c1.named_children():
            if hasattr(c2, 'mem_length') and c2.mem_length > 0:
                # print(n1, n2, c2.mem.shape, c2.mem.mean())
                mem = (c2.mem * 64).to(torch.int32)[0].numpy().transpose(0, 1).T
                data = "{" + ', '.join([str(i) for i in mem.flatten().tolist()]) + "}"
                l = mem.shape[0] * mem.shape[1]
                print(f"i32 _{n1}{n2}_mem[{l}] = {data};")

    for n1, c1 in net.named_children():
        for n2, c2 in c1.named_children():
            if hasattr(c2, 'mem_length'):
                if c2.mem_length > 0:
                    mem = c2.mem.detach()[0].numpy()
                    l = mem.shape[0] * mem.shape[1]
                    print(f"MatI32 {n1}{n2}_mem = (MatI32)" + "{" + f"_{n1}{n2}_mem, {mem.shape[1]}, {mem.shape[0]}" + "};")

    mem8 = net.down[8].mem.clone()
    y = net.test_step(x)
    print(x.shape, y.shape)
    print("=> x:")
    print(y.argmax(dim=-1))
