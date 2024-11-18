
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# 注册到ptflop的参数量计算中 
from ptflops.pytorch_engine import MODULES_MAPPING  
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func


def autopad(k, p=None, d=1):  # kernel, padding, dilation 
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module): 
    default_act = nn.SiLU()  # default activation

    def __init__(self, dim, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True): 
        super().__init__()
        _Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]
        _BN = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim-1]

        self.conv = _Conv(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = _BN(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x): 
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x): 
        return self.act(self.conv(x))


class FlashAttnFlopsHook(nn.Module):
    def __init__(self, ): 
        super().__init__() 
        def pass_through(*args, **kwargs):  return 
        MODULES_MAPPING[FlashAttnFlopsHook] = pass_through
  
    def forward(self, qkv):  
        b,l,_,h,d = qkv.shape
        print(b,l,h,d)
        self.__flops__ += 2 * b * l * l * h * d  # 统计qkv的计算量
        return qkv
    

class LogFlashAttentionNd(nn.Module):
    def __init__(self, dim, in_channels, num_heads=8, bases=[16, 5], down_factor=1): 
        super().__init__()
        self.dim = dim
        head_dim = in_channels // num_heads
        self.static_params = [num_heads, head_dim,]
        h = (head_dim * num_heads) * 2  # 2 for query and key
        self.bases = bases 
        self.down_factor = down_factor
        self.qks = nn.ModuleList(Conv(dim, in_channels, h, 1, act=False) for _ in self.bases)
        self.vs = nn.ModuleList(Conv(dim, in_channels, in_channels, 1, act=False) for _ in self.bases)
        self.pes = nn.ModuleList(Conv(dim, in_channels, in_channels, 3, 1, g=in_channels, act=False) for _ in self.bases)
        self.proj = Conv(dim, in_channels, in_channels, 1, act=False) 
        self.cuda_batch_size = 2**16 - 1
 
        self.flash_attn_flops_hook = FlashAttnFlopsHook()
        

    def _log_attn_func(self, x, index):
        h, d = self.static_params
        size0 = x.shape[2:]
        base = self.bases[index]
        qk = self.qks[index]
        v = self.vs[index]
        pe = self.pes[index]
        # 对2取对数，求多少个长度向上取整
        qk = qk(x)  # (batch, seqlen, nheads*headdim, )
        v2 = v0 = v(x)  # (batch, nheads*headdim, seqlen, )
        if self.down_factor > 1:
            v2 = [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d][self.dim-1](v2, self.down_factor, self.down_factor)
            qk = [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d][self.dim-1](qk, self.down_factor, self.down_factor)
        size_down = qk.shape[2:]

        qk = qk.flatten(2)
        v2 = v2.flatten(2)
        
        b, c, w = v2.shape
        n = int(math.log(w, base) - 1e-9) + 1
        qk = F.pad(qk, (0, base**n - w), value=0.0)  # pad到base的整数次幂
        v2 = F.pad(v2, (0, base**n - w), value=0.0)  # pad到base的整数次幂 
        qk = qk.transpose(1, 2).view(b, *[base,]*n, 2, h, d)
        v2 = v2.transpose(1, 2).view(b, *[base,]*n, 1, h, d,)   # （batch, *seqlen,  nheads, headdim )  
        cbs =   self.cuda_batch_size
        for i in range(n):  
            qkv = torch.cat([qk, v2], dim=-3).transpose(i+1, -4).flatten(0, -5).half()  # (batch * batch_len, seqlen, nheads, headdim,) 
            ibs = qkv.shape[0]
            v2_list = []
            for j in range((ibs + cbs - 1)//cbs):  # 向上取整，不超出cuda的范围
                v2_list.append(flash_attn_qkvpacked_func(qkv[j*cbs:(j+1)*cbs], 0.0)) 
                self.flash_attn_flops_hook(qkv)
            v2 = torch.cat(v2_list, dim=0) .unflatten(0, [b, ]+[base,]*(n-1))  
            v2 = v2.transpose(i+1, -4).unsqueeze(-3)

        # 删除pad的部分，恢复到原来的形状
        v2 = v2.reshape(b, -1, c)[..., :w, :].transpose(1, 2).unflatten(2, size_down)

        if self.down_factor > 1:
            v2 = F.interpolate(v2, size0, mode=['linear', 'bilinear', 'trilinear'][self.dim-1])
        print(v2.shape, v0.shape)
        # 添加位置编码
        return v2 + pe(v0)

    def forward(self, x):
        y = x
        # 多个base下，多次计算累加
        for index in range(len(self.bases)):
            y = y + self._log_attn_func(x, index)
        # 最后输出做通道投影 
        return self.proj(y) 

 
if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    ma = LogFlashAttentionNd(3, 1, 1, [8,], down_factor=4).cuda() 
    res = get_model_complexity_info(ma, (1, 128*4, 128*4, 128*4), as_strings=True, print_per_layer_stat=True)
    print(res)
    # LogFlashAttentionNd(
    #   4.7 k, 100.000% Params, 370.25 MMac, 100.000% MACs, 
    #   (qks): ModuleList(
    #     (0): Conv(
    #       2.18 k, 46.259% Params, 109.18 MMac, 29.489% MACs, 
    #       (conv): Conv2d(2.05 k, 43.537% Params, 102.76 MMac, 27.755% MACs, 32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn): BatchNorm2d(128, 2.721% Params, 6.42 MMac, 1.735% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    #     )
    #   )
    #   (vs): ModuleList(
    #     (0): Conv(
    #       1.09 k, 23.129% Params, 54.59 MMac, 14.745% MACs, 
    #       (conv): Conv2d(1.02 k, 21.769% Params, 51.38 MMac, 13.877% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (bn): BatchNorm2d(64, 1.361% Params, 3.21 MMac, 0.867% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    #     )
    #   )
    #   (pes): ModuleList(
    #     (0): Conv(
    #       352, 7.483% Params, 17.66 MMac, 4.770% MACs, 
    #       (conv): Conv2d(288, 6.122% Params, 14.45 MMac, 3.903% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    #       (bn): BatchNorm2d(64, 1.361% Params, 3.21 MMac, 0.867% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #       (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    #     )
    #   )
    #   (proj): Conv(
    #     1.09 k, 23.129% Params, 54.59 MMac, 14.745% MACs, 
    #     (conv): Conv2d(1.02 k, 21.769% Params, 51.38 MMac, 13.877% MACs, 32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn): BatchNorm2d(64, 1.361% Params, 3.21 MMac, 0.867% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (act): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
    #   )
    #   (flash_attn_flops_hook): FlashAttnFlopsHook(0, 0.000% Params, 134.22 MMac, 36.251% MACs, )
    # )
    # ('370.25 MMac', '4.7 k')