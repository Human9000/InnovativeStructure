import torch
from torch.nn import (
    Conv1d, Conv2d, Conv3d,
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    Sequential
)
from timm.layers import SqueezeExcite


class FurnaceConv(Sequential):
    rate_s = 2  # 炼丹炉的输入输出的加速比
    rate_e = 4  # 炼丹炉的膨胀比

    def __init__(self, dims, c_in, c_out, kernal_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        # ===== Conv 参数 ========
        Conv = [Conv1d, Conv2d, Conv3d][dims-1]
        self.dims = dims
        self.c_in = c_in
        self.c_out = c_out
        self.ks = kernal_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        # ==== Furnace 参数 ======
        _speed_in = c_in // self.rate_s   # 炼丹炉的入口流速
        _size = self.rate_e * c_in     # 炼丹炉的丹炉大小
        _speed_out = c_out // self.rate_s  # 炼丹炉的出口流速
        # ===== 记录 Furance 参数 ========
        self.c_in = _speed_in
        # ===== 炼丹炉-模块 ========
        # 炼丹 1：快速入药
        self.add_module('ingredient_addition', Conv(c_in, _speed_in, kernal_size, stride, padding, dilation, groups, bias=False))
        # 炼丹 2：药材配比
        self.add_module('proportioning', SqueezeExcite(_speed_in, 0.25))
        # 炼丹 3：熔炼药材（炼丹炉）
        self.add_module('melting', Conv(_speed_in, _size, 1, 1, 0, 1, groups, bias=False))
        # 炼丹 4：压缩成丹（炼丹炉）
        self.add_module('compression', Conv(_size, _speed_out, 1, 1, 0, 1, groups, bias=False))
        # 炼丹 5：快速出丹（炼丹炉）
        self.add_module('extraction', Conv(_speed_out, c_out, 1, 1, 0, 1, groups, bias=bias))


    @torch.no_grad()
    def fuse(self):
        # 炼丹炉 压缩
        g = self.groups
        ci, cp, cm, cc, ce = self._modules.values()
        w1 = cm.weight.reshape(g, cm.weight.shape[0]//g, cm.weight.shape[1], -1)
        w2 = cc.weight.reshape(g, cc.weight.shape[0]//g, cc.weight.shape[1], -1)
        w3 = ce.weight.reshape(g, ce.weight.shape[0]//g, ce.weight.shape[1], -1)
        [w1, w2, w3, ] = [i.flatten(3).permute(0, 3, 2, 1) for i in [w1, w2, w3]]
        w = ((w1  @ w2) @ w3).permute(0, 3, 2, 1).reshape(self.c_out,  self.c_in//g,  *[1,]*self.dims)
        conv = torch.nn.Conv2d(self.c_in,  self.c_out,  1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=self.groups,
                               bias=self.bias,
                               device=w.device)
        conv.weight.data.copy_(w)
        if self.bias:
            conv.bias.data.copy_(ce.bias)
        return Sequential(ci, cp, conv)  # 炼丹炉化简 => (入药，配比，出丹)


class FurnaceConvBN(Sequential):
    rate_s = 2  # 炼丹炉的输入输出的加速比
    rate_e = 4  # 炼丹炉的膨胀比

    def __init__(self, dims, c_in, c_out, kernal_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, bn_weight_init=1):
        super().__init__()
        # ===== Conv 参数 ========
        Conv = [Conv1d, Conv2d, Conv3d][dims-1]
        BN = [BatchNorm1d, BatchNorm2d, BatchNorm3d][dims-1]
        self.dims = dims
        self.cin = c_in
        self.cout = c_out
        self.ks = kernal_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        # ==== Furnace 参数 ======
        s1 = c_in // self.rate_s  # 炼丹炉的输入口
        e = self.rate_e * c_in   # 炼丹炉的丹炉容量
        s2 = c_out // self.rate_s  # 炼丹炉的输口
        # === 记录 Furance 参数 ===
        self.cin = s1
        # ===== 模块 ========
        self.add_module('c_is', Conv(c_in, s1, kernal_size, stride, padding, dilation, groups, bias=False))  # 特征提取，压缩加速传输
        self.add_module('se', SqueezeExcite(s1, 0.25))
        self.add_module('c_se', Conv(s1, e, 1, 1, 0, 1, groups, bias=False))  # 压缩-释放 炼丹
        self.add_module('bn', BN(e))  # 标准化
        self.add_module('c_es', Conv(e, s2, 1, 1, 0, 1, groups, bias=False))  # 释放-压缩 炼丹
        self.add_module('c_so', Conv(s2, c_out, 1, 1, 0, 1, groups, bias=bias))  # 压缩加速传输
        # ===== 初始化 ========
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        g = self.groups
        c_is, se, c_se, bn, c_es, c_so = self._modules.values()
        w_se = c_se.weight.reshape(g, c_se.weight.shape[0]//g, c_se.weight.shape[1], -1)  # g e/g s1/g *[1...]
        w_es = c_es.weight.reshape(g, c_es.weight.shape[0]//g, c_es.weight.shape[1], -1)  # g s2/g e/g *[1...]
        w_so = c_so.weight.reshape(g, c_so.weight.shape[0]//g, c_so.weight.shape[1], -1)  # g o/g s2/g *[1...]
        w_bn1 = (bn.weight / (bn.running_var + bn.eps)**0.5).reshape(g, bn.weight.shape[0]//g, 1, 1)   # g,e/g,1,1
        b1 = (bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5).reshape(g,  bn.bias.shape[0]//g, 1, 1)  # g,e/g, 1,1

        [w_se, w_es, w_so, w_bn1, b1] = [i.flatten(3).permute(0, 3, 2, 1) for i in [w_se, w_es, w_so, w_bn1, b1]]
        b = ((b1 @ w_es) @ w_so).flatten()                                                # o
        w = (((w_se*w_bn1)  @ w_es) @ w_so).permute(0, 3, 2, 1).reshape(self.cout,  self.cin//g,  *[1,]*self.dims)  # g *1 s/g o/g => o s/g *1

        conv = torch.nn.Conv2d(self.cin, self.cout, 1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               groups=self.groups,
                               device=w.device)
        if self.bias:
            b = b + c_so.bias
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return Sequential(c_is, se, conv)


def FurnaceConv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=True):
    return FurnaceConv(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def FurnaceConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=True):
    return FurnaceConv(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def FurnaceConv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=True):
    return FurnaceConv(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def FurnaceConvBN1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=True):
    return FurnaceConvBN(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def FurnaceConvBN2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=True):
    return FurnaceConvBN(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


def FurnaceConvBN3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=1, bias=True):
    return FurnaceConvBN(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


if __name__ == '__main__':
    x = torch.rand(1, 8, 5, 5)

    conv1 = FurnaceConvBN2d(8, 8, 3, 1, 1, 1, groups=4, bias=True).eval()
    conv2 = conv1.fuse()
    conv3 = Conv2d(8, 8, 3, 1, 1, 1, groups=4, bias=True).eval()
    # y1 = conv1(x)
    # y2 = conv2(x)
    # print(conv1)
    # print(conv2)
    # print("err:",(y1-y2).abs().mean())

    import ptflops
    res1 = ptflops.get_model_complexity_info(conv1, (8, 5, 5), print_per_layer_stat=True)
    res2 = ptflops.get_model_complexity_info(conv2, (8, 5, 5), print_per_layer_stat=True)
    res3 = ptflops.get_model_complexity_info(conv3, (8, 5, 5), print_per_layer_stat=True)
    print("Conv2d", res3)
    print("FurnaceConvBN2d", res1)
    print("FurnaceConvBN2d-fuse", res2)
