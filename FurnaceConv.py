import torch
from torch.nn import (
    Conv1d, Conv2d, Conv3d,
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    Sequential
)


class FurnaceConvBN(Sequential):
    def __init__(self, dims, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        Conv = [Conv1d, Conv2d, Conv3d][dims-1]
        BN = [BatchNorm1d, BatchNorm2d, BatchNorm3d][dims-1]
        self.dims = dims
        rate_s = 2  # 丹炉压缩比
        rate_e = 4  # 丹炉释放比
        s1 = a // rate_s
        e = rate_e * a
        s2 = b // rate_s
        self.add_module('c_is', Conv(a, s1, ks, stride, pad, dilation, groups, bias=False))  # 特征提取，压缩加速传输
        self.add_module('c_se', Conv(s1, e, 1, 1, 0, 1, groups, bias=False))  # 压缩-释放 炼丹
        self.add_module('bn', BN(e))  # 标准化
        self.add_module('c_es', Conv(e, s2, 1, 1, 0, 1, groups, bias=False))  # 释放-压缩 炼丹
        self.add_module('c_so', Conv(s2, b, 1, 1, 0, 1, groups, bias=False))  # 压缩加速传输
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
        self.groups = groups
        self.fuseing = False

    def fuse(self):
        self.fuseing = True

    def unfuse(self):
        self.fuseing = False

    @torch.no_grad()
    def _fuse(self):
        g = self.groups
        c_is, c_se, bn, c_es, c_so = self._modules.values()
        w_is = c_is.weight.reshape(g, c_is.weight.shape[0]//g, c_is.weight.shape[1], -1)  # g s/g i/g *k
        w_se = c_se.weight.reshape(g, c_se.weight.shape[0]//g, c_se.weight.shape[1], -1)  # g e/g s1/g *[1...]
        w_es = c_es.weight.reshape(g, c_es.weight.shape[0]//g, c_es.weight.shape[1], -1)  # g s2/g e/g *[1...]
        w_so = c_so.weight.reshape(g, c_so.weight.shape[0]//g, c_so.weight.shape[1], -1)  # g o/g s2/g *[1...]

        w_bn1 = (bn.weight / (bn.running_var + bn.eps)**0.5).reshape(g, bn.weight.shape[0]//g, 1)   # g,e/g,1,1,1
        b1 = (bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5).reshape(g,  bn.bias.shape[0]//g, 1)  # g,e/g,1,1,1
        # g c2 c1 *k => g *k c1 c2
        [w_is, w_se, w_es, w_so, w_bn1, b1] = [i.flatten(3).permute(0, 3, 2, 1) for i in [w_is, w_se, w_es, w_so, w_bn1, b1]]
        b = ((b1 @ w_es) @ w_so).flatten()                                                # o
        w = (w_is @  ((w_se*w_bn1)  @ w_es) @ w_so).permute(0, 4, 3, 1, 2).flatten(0, 1)  # g *k i o/g => o i *k
        conv = torch.nn.Conv2d(w.size(1) * self.groups,  w.size(0),  w.shape[2:],
                               stride=self.c.stride,
                               padding=self.c.padding,
                               dilation=self.c.dilation,
                               groups=self.c.groups,
                               device=w.device)
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

    def forward(self, input):
        if self.fuseing:
            return super().forward(input)
        else:
            return self._fuse()(input)


if __name__ == '__main__':
    conv1 = FurnaceConvBN(8, 8, 3, 1, 1, 1, groups=2,)
    x = torch.rand(1, 8, 5, 5)
    conv1.eval()
    y1 = conv1(x)
    conv1.fuseing = True
    y2 = conv1(x)

    print
