from typing import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


def silu(x):
    return x * F.sigmoid(x)


class RMSNorm2d(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d,1,1)))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x * silu(residual) 
        return x * torch.rsqrt(x.pow(2).mean([-2, -1], keepdim=True) + self.eps) * self.weight


class DWConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(OrderedDict([
            ('depthwise', nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias, padding_mode)),
            ('pointwise', nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False))
        ]))



class SDWConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(OrderedDict([
            ("d", nn.Conv2d(in_channels, in_channels*groups, kernel_size, stride, padding, dilation, in_channels, bias, padding_mode)),
            ("s", nn.ChannelShuffle(groups)), 
            ("p", nn.Conv2d(in_channels*groups, out_channels, 1, groups=groups, bias=False))
        ]))
# class SDWConv2d(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
#         super().__init__(OrderedDict([
#             ("d", nn.Conv2d(in_channels, in_channels*groups, kernel_size, stride, padding, dilation, in_channels, bias, padding_mode)),
#             ("s1", nn.ChannelShuffle(groups)),
#             ("p1", nn.Conv2d(in_channels*groups, in_channels*groups, 1, groups=groups**2, bias=False)),
#             ("s2", nn.ChannelShuffle(groups)),
#             ("p2", nn.Conv2d(in_channels*groups, out_channels, 1, groups=groups, bias=False))
#         ]))


class SDWNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SDWNBlock, self).__init__()
        self.dsw_conv = SDWConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False)
        self.norm = RMSNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.dsw_conv(x),
                         self.residual_conv(x))


if __name__ == '__main__':
    conv = SDWConv2d(64, 128, 3, 1, 1, groups=4,)
    print(conv)
    block = SDWNBlock(64, 128, 3, 1, 1, groups=4,)
    print(block)
    block = DWConv2d(64, 128, 3, 1, 1, groups=4,)
    print(block)

    from ptflops import get_model_complexity_info
    from thop import profile
    from thop import clever_format

    x = torch.randn(1, 64, 32, 32) 
    macs, params = profile(block, inputs=(x,)) 
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs: {macs}, params: {params}")
    # macs: 1.560G, params: 1.060M

    res = get_model_complexity_info( SDWConv2d(64, 128, 3, 1, 1, groups=128, bias=False), 
                                    ( 64, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)

    res = get_model_complexity_info( DWConv2d(64, 128, 3, 1, 1, groups=1, bias=False),
                                     ( 64, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)
    res = get_model_complexity_info( nn.Conv2d(64, 128, 3, 1, 1, groups=1, bias=False),
                                     ( 64, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)

