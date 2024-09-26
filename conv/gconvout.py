import torch
from torch import nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


# 3*255*255 =》 27*(255/3)*(255/3)
class GSUBase(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super(GSUBase, self).__init__()
        assert dim in [1, 2, 3]
        self.g = groups
        # ============= 根据dim生成相关参数 ===========
        self.Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim-1]

        # ============= 生成可训练的相关层 ===========
        self.center = self.Conv(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False)
        self.border = self.Conv(in_channels, in_channels*2, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.grus = nn.ModuleList(nn.GRUCell(in_channels//groups*2, out_channels//groups) for _ in range(groups))

    def split_c_b_groups(self, x):
        c = self.center(x)
        b = self.border(x)
        assert c.shape == b.shape
        self._bs = [b.shape[0],] + list(b.shape[2:])  # 记录除了通道的形状，batch,*size

        # 分组
        c = c.flatten(2).unflatten(1, (self.g, -1)).transpose(1, 3).flatten(0, 1)  # bs, c2/g, g
        b = b.flatten(2).unflatten(1, (self.g, -1)).transpose(1, 3).flatten(0, 1)  # bs, c1/g, g

        return c, b

    def marge_groups(self, x):
        x = x.transpose(0, 1).unflatten(1, self._bs).transpose(0, 1)  # bs, c2/g, g -> b, c2, *s
        return x

    def forward(self, x):
        c, b = self.split_c_b_groups(x)  # 分离中心和边界
        y = [gru(b[..., g], c[..., g]) for g, gru in enumerate(self.grus)]  # 分组调用不同的gru卷积: g*(bs, c2/g)
        y = torch.stack(y, dim=-1).flatten(1)  # shuffle_group : g*(bs, c2/g) => bs, c2/g, g => bs, c2
        y = self.marge_groups(y)  # 合并组
        return y


class GGSUBase(GSUBase):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super(GGSUBase, self).__init__(dim, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        assert dim in [1, 2, 3]
        # ============= 根据dim生成相关参数 ===========
        self.Pool = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d][dim-1]
        # max = 11        
        s = [[1, 3, 5, 11],  # 1d
             [(1, 11), (11, 1), (3, 5), (5, 3)],  # 2d
             [(1, 3, 11), (3, 11, 1), (11, 5, 1), (5, 3, 5)],  # 3d
             ][dim-1]

        if dim == 1:
            c = s
        elif dim == 2:
            c = [s[0]*s[1] for s in s]
        elif dim == 3:
            c = [s[0]*s[1]*s[2] for s in s]

        # ============= 创建模型的可训练参数层 ===========
        # 注意力块 分离层
        self.attn0 = nn.Sequential(self.Pool(s[0]), nn.Flatten(2), nn.Linear(c[0], c[0]), nn.Unflatten(2, s[0]))
        self.attn1 = nn.Sequential(self.Pool(s[1]), nn.Flatten(2), nn.Linear(c[1], c[1]), nn.Unflatten(2, s[1]))
        self.attn2 = nn.Sequential(self.Pool(s[2]), nn.Flatten(2), nn.Linear(c[2], c[2]), nn.Unflatten(2, s[2]))
        self.attn3 = nn.Sequential(self.Pool(s[3]), nn.Flatten(2), nn.Linear(c[3], c[3]), nn.Unflatten(2, s[3]))

        # 注意力块 融合层
        self.merge = self.Conv(in_channels*4, in_channels*2, 1, groups=in_channels, bias=False)

    # 替换掉父类的border模块
    def border(self, x):
        interplate_argvs = {'size': x.shape[2:], 'mode': 'bilinear', 'align_corners': True}
        # 分离成 1x11, 11x1 , 2x5, 5x2 四个级别的注意力块
        attn0 = F.interpolate(self.attn0(x), **interplate_argvs)
        attn1 = F.interpolate(self.attn1(x), **interplate_argvs)
        attn2 = F.interpolate(self.attn2(x), **interplate_argvs)
        attn3 = F.interpolate(self.attn3(x), **interplate_argvs)
        # 融合分离的四个注意力块
        y = self.merge(torch.cat([attn0, attn1, attn2, attn3, ], dim=1)) 
        return y


class GSU1d(GSUBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super().__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class GGSU1d(GGSUBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super().__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class GSU2d(GSUBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class GGSU2d(GGSUBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class GSU3d(GSUBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super().__init__(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


class GGSU3d(GGSUBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super().__init__(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    in_channle = 64
    out_channle = 128
    kernel_size = 3
    groups = 1
    stride = 1
    padding = 1
    dilation = 1
    conv_out = GSU2d(in_channle, out_channle, kernel_size,  stride=stride, padding=padding, dilation=dilation, groups=groups, )
    g_conv_out = GGSU2d(in_channle, out_channle, kernel_size,  stride=stride, padding=padding, dilation=dilation, groups=groups, )
    conv2d = nn.Conv2d(in_channle, out_channle, kernel_size,  stride=stride, padding=padding, dilation=dilation, groups=groups, ) 

    g_out_res = get_model_complexity_info(g_conv_out,
                                          (in_channle, 256, 256),
                                          as_strings=True,
                                          print_per_layer_stat=True,
                                          verbose=False)

    out_res = get_model_complexity_info(conv_out,
                                        (in_channle, 256, 256),
                                        as_strings=True,
                                        print_per_layer_stat=True,
                                        verbose=False)

    c_res = get_model_complexity_info(conv2d,
                                      (in_channle, 256, 256),
                                      as_strings=True,
                                      print_per_layer_stat=True,
                                      verbose=False)
    print("Conv2d", c_res)
    print("ConvOut2d", out_res)
    print("GConvOut2d", g_out_res)

    # y = conv_out(torch.randn(1, in_channle, 256))
    # print(y.shape)
