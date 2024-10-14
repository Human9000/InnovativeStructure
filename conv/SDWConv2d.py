from typing import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


def silu(x):
    return x * F.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x, residual=None):
        if residual is not None:
            x = x * silu(residual)
        norm = torch.rsqrt(x.pow(2).flatten(1).mean(-1) + self.eps) * self.weight
        for _ in x.shape[2:]:
            norm = norm.unsqueeze(-1)
        return x * norm


class DimShuffle(nn.Module):
    def __init__(self, dim: int, groups: int):
        super().__init__()
        self.dim = dim
        self.groups = groups

    def forward(self, x):
        src_shape = list(x.shape)
        dst_shape = src_shape[:self.dim] + [self.groups] + [src_shape[self.dim] // self.groups] + src_shape[self.dim + 1:]
        return x.view(*dst_shape).transpose(self.dim, self.dim + 1).contiguous().view(*src_shape)


class DWConv(nn.Sequential):
    def __init__(self, dims, in_channels, out_channels):
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        super().__init__(OrderedDict([
            ('depthwise', Conv(in_channels, in_channels, 3, 1, 1, groups=in_channels)),
            ('pointwise', Conv(in_channels, out_channels, 1, bias=False))
        ]))


class SDWConv(nn.Sequential):
    def __init__(self, dims, in_channels, out_channels, groups=1):
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1] 
        h1 = in_channels * groups
        h2 = (in_channels + groups**2 + 1 )//(groups**2) * groups**2  # 向上取整
        super().__init__(OrderedDict([
            ("d", Conv(in_channels, h1, 3, 1, 1, groups=in_channels)),
            ("s", DimShuffle(1, in_channels)),
            ("r", Conv(h1, h2, 1, groups=groups **2, bias=False)),
            ("p", Conv(h2, out_channels, 1, groups=groups, bias=False))
        ]))


class SDWNBlock(nn.Module):
    def __init__(self, dims, in_channels, out_channels, groups=4, ):
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        super(SDWNBlock, self).__init__()
        self.dsw_conv = SDWConv(dims, in_channels, out_channels, groups)
        self.residual_conv = Conv(in_channels, out_channels, 1, groups=groups, bias=False)
        self.norm = RMSNorm(out_channels)

    def forward(self, x):
        return self.norm(self.dsw_conv(x),  self.residual_conv(x))



if __name__ == '__main__':
    conv = SDWConv(2, 64, 128,   groups=4,)
    print(conv)
    block = SDWNBlock (2, 64, 128,  groups=4,)
    print(block)
    block = DWConv (2, 64, 128,   )
    print(block)

    from ptflops import get_model_complexity_info
    from thop import profile
    from thop import clever_format

    x = torch.randn(1, 64, 32, 32)
    macs, params = profile(block, inputs=(x,))
    macs, params = clever_format([macs, params], "%.3f")
    print(f"macs: {macs}, params: {params}")
    # macs: 1.560G, params: 1.060M

    res = get_model_complexity_info(SDWNBlock(2, 32, 128, groups=16,   ),
                                    (32, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)

    res = get_model_complexity_info(SDWConv(2, 64, 128, groups=4,  ),
                                    (64, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)

    res = get_model_complexity_info(DWConv(2, 64, 128 ),
                                    (64, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)
    res = get_model_complexity_info(nn.Conv2d(64, 128, 3, 1, 1, groups=4 ),
                                    (64, 32, 32), as_strings=True, print_per_layer_stat=True)
    print(res)
