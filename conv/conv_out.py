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


class ConvOut2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1):
        super(ConvOut2d, self).__init__()
        # ====
        _cin = in_channels//groups
        _cout = out_channels//groups
        _hidden = (kernel_size**2-1) * _cin
        # ====
        self.in_channels = _cin
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 分离中心点和周围点
        self.h_index = list(range(kernel_size**2))
        self.c_index = [self.h_index.pop(kernel_size**2//2)]

        self.fcs = nn.ModuleList(nn.Linear(_hidden, _cout) for g in range(groups))
        self.grus = nn.ModuleList(nn.GRUCell(_cin, _cout) for g in range(groups))

        self._batch = 1
        self._out_size = (1, 1)
        # 计算滑动窗口后的特征图大小的函数
        self._lsize = lambda l: (l + 2 * padding - 2 * (kernel_size//2) * dilation) // stride
        self._whsize = lambda w, h: (self._lsize(w), self._lsize(h))

    def slide(self, x):
        # 记录batch和特征图大小，用于滑动窗口后的还原
        self._batch = x.shape[0]
        self._out_size = self._whsize(x.size(2), x.size(3))
        # 滑动窗口展开
        x = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.transpose(1, 2).flatten(0, 1).unflatten(1, (self.groups, self.in_channels, -1))  # b*win_num, g, cin, k**2
        # 分离出中心点和边界点
        c = x[..., self.c_index].flatten(-2)  # b*win_num, g, cin
        h = x[..., self.h_index].flatten(-2)  # b*win_num, g, cin*(k**2 - 1)
        return c, h

    def unslide(self, x):
        x = x.unflatten(0, (self._batch, -1)).transpose(1, 2).unflatten(-1,  self._out_size)
        return x

    def forward(self, x):
        c, h = self.slide(x)  # 滑动窗口展开
        y = []
        # 分组调用不同的gru卷积
        for g, (fc, gru) in enumerate(zip(self.fcs, self.grus)):
            _c = c[..., g, :]  # 取第g组  b*win_num, cin
            _h = h[..., g, :]  # 取第g组  b*win_num, cin
            _h = fc(_h)        # b*win_num, cout
            _y = gru(_c, _h)   # b*win_num, cout
            y.append(_y)
        y = torch.cat(y, dim=-1)  # b*win_num, g * cout
        y = self.unslide(y)  # 融合滑动窗口
        return y


class ConvOut1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvOut1d, self).__init__()
        # ====
        _cin = in_channels//groups
        _cout = out_channels//groups
        _hidden = (kernel_size-1) * _cin
        # ====
        self.in_channels = _cin
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 分离中心点和周围点
        self.h_index = list(range(kernel_size))
        self.c_index = [self.h_index.pop(kernel_size//2)]

        self.fcs = nn.ModuleList(nn.Linear(_hidden, _cout, bias=bias) for _ in range(groups))
        self.grus = nn.ModuleList(nn.GRUCell(_cin, _cout, bias=bias) for _ in range(groups))

        self._batch = 1
        self._out_size = (1, 1)
        # 计算滑动窗口后的特征图大小的函数
        self._lsize = lambda l: (l + 2 * padding - 2 * (kernel_size//2) * dilation) // stride
     

    def slide(self, x):
        # 记录batch和特征图大小，用于滑动窗口后的还原
        self._batch = x.shape[0]
        self._out_size = (self._lsize(x.shape[2]),)
        # 滑动窗口展开
        x = F.unfold(x[..., None], (self.kernel_size, 1), (self.dilation, 1), (self.padding, 0), (self.stride, 1))
 
        x = x.transpose(1, 2).flatten(0, 1).unflatten(1, (self.groups, self.in_channels, -1))  # b*win_num, g, cin, k**2
        # 分离出中心点和边界点
        c = x[..., self.c_index].flatten(-2)  # b*win_num, g, cin
        h = x[..., self.h_index].flatten(-2)  # b*win_num, g, cin*(k**2 - 1)
        return c, h

    def unslide(self, x):
        x = x.unflatten(0, (self._batch, -1)).transpose(1, 2).unflatten(-1,  self._out_size)
        return x

    def forward(self, x):
        c, h = self.slide(x)  # 滑动窗口展开
        y = []
        # 分组调用不同的gru卷积
        for g, (fc, gru) in enumerate(zip(self.fcs, self.grus)):
            _c = c[..., g, :]  # 取第g组  b*win_num, cin
            _h = h[..., g, :]  # 取第g组  b*win_num, cin
            _h = fc(_h)        # b*win_num, cout
            _y = gru(_c, _h)   # b*win_num, cout
            y.append(_y)
        y = torch.cat(y, dim=-1)  # b*win_num, g * cout
        y = self.unslide(y)  # 融合滑动窗口
        return y

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    in_channle = 64
    out_channle = 64
    kernel_size = 3
    groups = 32
    stride = 1
    padding = 1
    dilation = 1
    # conv_out = ConvOut2d(in_channle, out_channle, kernel_size,  stride=stride, padding=padding, dilation=dilation, groups=groups, )    
    conv_out = ConvOut1d(in_channle, out_channle, kernel_size,  stride=stride, padding=padding, dilation=dilation, groups=groups, )    

    res = get_model_complexity_info(conv_out,
                                    (in_channle, 256),
                                    as_strings=True,
                                    print_per_layer_stat=True,
                                    verbose=False)
    print(res) 

    y = conv_out(torch.randn(1, in_channle, 256))
    print(y.shape)
