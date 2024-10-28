import torch
from torch.nn import (
    Conv1d, Conv2d, Conv3d,
    BatchNorm1d, BatchNorm2d, BatchNorm3d,
    Sequential
)
# 炼丹炉卷积

class FurnaceConv(Sequential):
    def __init__(self, dims, c_in, c_out, kernal_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        # ===== Conv 参数 ========
        Conv = [Conv1d, Conv2d, Conv3d][dims-1]
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
        rate_s = 2  # 炼丹炉的输入输出的加速比
        rate_e = 4  # 炼丹炉的膨胀比
        s1 = c_in // rate_s # 炼丹炉的输入口
        e = rate_e * c_in   # 炼丹炉的丹炉容量
        s2 = c_out // rate_s # 炼丹炉的输口 
        # ===== 模块 ========
        self.add_module('c_is', Conv(c_in, s1, kernal_size, stride, padding, dilation, groups, bias=False))  # 特征提取，压缩加速传输
        self.add_module('c_se', Conv(s1, e, 1, 1, 0, 1, groups, bias=False))  # 压缩-释放 炼丹
        self.add_module('c_es', Conv(e, s2, 1, 1, 0, 1, groups, bias=False))  # 释放-压缩 炼丹
        self.add_module('c_so', Conv(s2, c_out, 1, 1, 0, 1, groups, bias=bias))  # 压缩加速传输
 
    @torch.no_grad()
    def fuse(self):
        g = self.groups
        c_is, c_se, c_es, c_so = self._modules.values()
        w_is = c_is.weight.reshape(g, c_is.weight.shape[0]//g, c_is.weight.shape[1], -1)  # g s/g i/g *k
        w_se = c_se.weight.reshape(g, c_se.weight.shape[0]//g, c_se.weight.shape[1], -1)  # g e/g s1/g *[1...]
        w_es = c_es.weight.reshape(g, c_es.weight.shape[0]//g, c_es.weight.shape[1], -1)  # g s2/g e/g *[1...]
        w_so = c_so.weight.reshape(g, c_so.weight.shape[0]//g, c_so.weight.shape[1], -1)  # g o/g s2/g *[1...]
        [w_is, w_se, w_es, w_so, ] = [i.flatten(3).permute(0, 3, 2, 1) for i in [w_is, w_se, w_es, w_so]]                  # o
        w = (w_is @  (w_se  @ w_es) @ w_so).permute(0, 3, 2, 1).reshape(self.cout,  self.cin//g,  *c_is.weight.shape[2:])  # g *k i o/g => o i *k 
        conv = torch.nn.Conv2d(self.cout,  self.cin,  self.ks,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups,
                               bias=self.bias,
                               device=w.device)
        conv.weight.data.copy_(w)
        if self.bias:
            conv.bias.data.copy_(c_so.bias)
        return conv 

class FurnaceConvBN(Sequential):
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
        rate_s = 2  # 炼丹炉的输入输出的加速比
        rate_e = 4  # 炼丹炉的膨胀比
        s1 = c_in // rate_s # 炼丹炉的输入口
        e = rate_e * c_in   # 炼丹炉的丹炉容量
        s2 = c_out // rate_s # 炼丹炉的输口 
        # ===== 模块 ========
        self.add_module('c_is', Conv(c_in, s1, kernal_size, stride, padding, dilation, groups, bias=False))  # 特征提取，压缩加速传输
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
        c_is, c_se, bn, c_es, c_so = self._modules.values()
        w_is = c_is.weight.reshape(g, c_is.weight.shape[0]//g, c_is.weight.shape[1], -1)  # g s/g i/g *k
        w_se = c_se.weight.reshape(g, c_se.weight.shape[0]//g, c_se.weight.shape[1], -1)  # g e/g s1/g *[1...]
        w_es = c_es.weight.reshape(g, c_es.weight.shape[0]//g, c_es.weight.shape[1], -1)  # g s2/g e/g *[1...]
        w_so = c_so.weight.reshape(g, c_so.weight.shape[0]//g, c_so.weight.shape[1], -1)  # g o/g s2/g *[1...]

        w_bn1 = (bn.weight / (bn.running_var + bn.eps)**0.5).reshape(g, bn.weight.shape[0]//g, 1, 1)   # g,e/g,1,1,1
        b1 = (bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5).reshape(g,  bn.bias.shape[0]//g, 1, 1)  # g,e/g,1,1,1
        # g c2 c1 *k => g *k c1 c2
        [w_is, w_se, w_es, w_so, w_bn1, b1] = [i.flatten(3).permute(0, 3, 2, 1) for i in [w_is, w_se, w_es, w_so, w_bn1, b1]]
        b = ((b1 @ w_es) @ w_so).flatten()                                                # o
        w = (w_is @  ((w_se*w_bn1)  @ w_es) @ w_so).permute(0, 3, 2, 1).reshape(self.cout,  self.cin//g,  *c_is.weight.shape[2:])  # g *k i o/g => o i *k

        conv = torch.nn.Conv2d(self.cin, self.cout, self.ks,
                               stride=self.stride,
                               padding=self.padding,
                               dilation=self.dilation,
                               groups=self.groups,
                               device=w.device)
        if self.bias:
            b = b + c_so.bias
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
 

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
    
    conv1 = FurnaceConv2d(8, 8, 3, 1, 1, 1, groups=4, bias=True).eval() 
    conv2 = conv1.fuse()
    y1 = conv1(x) 
    y2 = conv2(x) 
    print(conv1)
    print(conv2)
    print("err:",(y1-y2).abs().mean()) 

    conv1 = FurnaceConvBN2d(8, 8, 3, 1, 1, 1, groups=4, bias=True).eval() 
    conv2 = conv1.fuse()
    y1 = conv1(x) 
    y2 = conv2(x) 
    print(conv1)
    print(conv2)
    print("err:",(y1-y2).abs().mean())


    import ptflops 
    res1 = ptflops.get_model_complexity_info(conv1, (8, 5, 5), print_per_layer_stat=False)
    res2 = ptflops.get_model_complexity_info(conv2, (8, 5, 5), print_per_layer_stat=False) 
    print("Conv1",res1, "Conv2",res2) 