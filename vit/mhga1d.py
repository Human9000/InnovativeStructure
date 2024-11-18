
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self,  c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        _Conv = nn.Conv1d 
        _BN =  nn.BatchNorm1d 

        self.conv = _Conv(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = _BN(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class MultiHeadGridAttention1d(nn.Module):
    def __init__(self, in_channels, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = nh_kd * 2
        self.qk1 = Conv(in_channels, h, 1, act=False)
        self.qk2 = Conv(in_channels, h, 1, act=False) 
        self.qk3 = Conv(in_channels, h, 1, act=False) 
        self.qk4 = Conv(in_channels, h, 1, act=False) 

        self.v = Conv(in_channels, in_channels, 1, act=False)
        self.pe = Conv(in_channels, in_channels, 3, 1, g=in_channels, act=False)
        self.proj = Conv(in_channels, in_channels, 1, act=False)

    def forward(self, x):
        b, c, w = x.shape
        nh = self.num_heads
        kd = self.key_dim
        hd = self.head_dim

        w0  = int(w**(1/4)+1-1e-9)  # 开根号并向上取整得到 h0 h1

        # padding 图像使宽高可以开根号为整数, => (B, C, h0*h1, w0*w1)
        x = F.pad(x, (0, w0**4 - w), value=0.0)
         
        qk1 = self.qk1(x).view(b, nh, 2, kd, *[w0,]*4).unbind(2)
        qk2 = self.qk2(x).view(b, nh, 2, kd, *[w0,]*4).unbind(2) 
        v0 = self.v(x)
        v = v0.view(b, nh, hd, *[w0,]*4)

        # 在多个维度求注意力
        a1 = (torch.einsum('bHdijkl,bhdIjkl->bHIijkl', qk1[0], qk1[1]) * self.scale).softmax(dim=3)
        a2 = (torch.einsum('bHdijkl,bhdiJkl->bHJijkl', qk2[0], qk2[1]) * self.scale).softmax(dim=4) 
        a3 = (torch.einsum('bHdijkl,bhdijKl->bHKijkl', qk2[0], qk2[1]) * self.scale).softmax(dim=5) 
        a4 = (torch.einsum('bHdijkl,bhdijkL->bHLijkl', qk2[0], qk2[1]) * self.scale).softmax(dim=6) 


        print(a1.shape, a2.shape, a3.shape, a4.shape)
        # 依次叠加各个维度注意力
        x = torch.einsum('bHdijkl,bHIijkl,bHJIjkl,bHKIJkl,bHLIJKl->bHdIJkL', v, a1, a2, a3, a4)

        # 添加位置编码的残差
        x = x.reshape(b, nh*hd,  w0**4) + self.pe(v0)

        # 去除padding，并做输出投影
        x = self.proj(x[..., :w])
        return x


if __name__ == '__main__':

    from ptflops import get_model_complexity_info
    res = get_model_complexity_info(MultiHeadGridAttention1d(16,4),
                                    (16, 1024*1024),
                                    as_strings=True,
                                    print_per_layer_stat=True)
    print(res)
