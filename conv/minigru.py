import torch

from torch import nn
from torch.nn import functional as F


# class Residual1x12d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
#     super().__init__()
#     self.conv = nn.Conv2d(in_channels, out_channels, 1, stride,
#                           padding, dilation, groups, bias, padding_mode, device, dtype)
#     self.left = kernel_size//2

#     def forward(self, x):
#         l = self.left
#         y = self.conv(x)
#         w, h = y.shape[2:]
#         return y[:, :, l:w-l, l:h-l]


# class MiniGRU(nn.Module):
# def __init__(self, in_channels, hidden_size):
#     super().__init__()
#     self.z = nn.Linear(in_channels, hidden_size)  # 更新门
#     self._h = nn.Linear(in_channels, hidden_size)  # 隐状态

# def forward(self, xs, h0=None):
#     '''
#     输入：xs 输入序列，形状为 (batch_size, seq_len, in_channels)
#     输入：h0 初始的隐状态，形状为 (batch_size, hidden_size)
#     输出：h 更新后的隐状态，形状为 (batch_size, seq_len, hidden_size)
#     '''
#     # 并行计算更新门和候选隐状态
#     zs = torch.sigmoid(self.z(xs))
#     _hs = self._h(xs)

#     # 初始化隐状态
#     if h0 is None:
#         h = _hs[:, :1, :].clone() * 0
#     else:
#         h = h0

#     hs = []

#     # 循环更新隐状态
#     for i in range(xs.shape[1]):
#         h = zs[:, i, :] * _hs[:, i, :] + (1-zs[:, i, :]) * h
#         hs.append(h)

#     return torch.stack(hs, dim=1)


class MiniGRUConv2d8(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.z = nn.Conv2d(in_channels, out_channels * 8, kernel_size, stride, padding)  # 更新门
        self._h = nn.Conv2d(in_channels, out_channels * 8, kernel_size, stride, padding)  # 隐状态
        self.s = nn.Conv2d(in_channels, out_channels * 8, kernel_size, stride, padding)  # 选择门

        # 初始化隐状态, 4个正方向
        self.init_h20 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h21 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h30 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h31 = nn.Parameter(torch.zeros(1, out_channels, 1))

        # 初始化隐状态, 4个斜方向
        self.init_h00 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h10 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h11 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h01 = nn.Parameter(torch.zeros(1, out_channels, 1))

    # 更新隐状态, 上方向
    def d20(self, zs, _hs):
        hs = torch.zeros_like(_hs) 
        # 初始化隐状态传递
        z = zs[:, :, 0, :]
        _h = _hs[:, :, 0, :]
        h_1 = self.init_h20
        hs[:, :, 0, :] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(1, zs.shape[2]):
            z = zs[:, :, i, :]
            _h = _hs[:, :, i, :]
            h_1 = hs[:, :, i-1, :]
            hs[:, :, i, :] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 下方向
    def d21(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递
        z = zs[:, :, -1, :]
        _h = _hs[:, :, -1, :]
        h_1 = self.init_h21
        hs[:, :, -1, :] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(zs.shape[2]-2, -1, -1):
            z = zs[:, :, i, :]
            _h = _hs[:, :, i, :]
            h_1 = hs[:, :, i+1, :]
            hs[:, :, i, :] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 左方向
    def d30(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递
        z = zs[:, :, :, 0] 
        _h = _hs[:, :, :, 0]
        h_1 = self.init_h30
        hs[:, :, :, 0] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(1, zs.shape[3]):
            z = zs[:, :, :, i]
            _h = _hs[:, :, :, i]
            h_1 = hs[:, :, :, i-1]
            hs[:, :, :, i] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 右方向
    def d31(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递
        z = zs[:, :, :, -1]
        _h = _hs[:, :, :, -1]
        h_1 = self.init_h31
        hs[:, :, :, -1] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(zs.shape[3]-2, -1, -1):
            z = zs[:, :, :, i]
            _h = _hs[:, :, :, i]
            h_1 = hs[:, :, :, i+1]
            hs[:, :, :, i] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 左上方向
    def d00(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递，第一行
        z = zs[:, :, 0,  :]
        _h = _hs[:, :, 0, :]
        h_1 = self.init_h00
        hs[:, :, 0, ] = z * _h + (1-z) * h_1  
        # 初始化隐状态传递，第一列
        z = zs[:, :, :, 0]
        _h = _hs[:, :, :, 0]
        h_1 = self.init_h00
        hs[:, :, :, 0] = z * _h + (1-z) * h_1  
        # 循环更新隐状态
        for i in range(1, zs.shape[2]):
            for j in range(1, zs.shape[3]): 
                z = zs[:, :, i, j]
                _h = _hs[:, :, i, j]
                h_1 = hs[:, :, i-1, j-1]
                hs[:, :, i, j] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 右上方向
    def d01(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递，第一行
        z = zs[:, :, 0, :]
        _h = _hs[:, :, 0, :]
        h_1 = self.init_h01
        hs[:, :, 0, :] = z * _h + (1-z) * h_1          
        # 初始化隐状态传递，最后一列
        z = zs[:, :, :, -1]
        _h = _hs[:, :, :, -1]
        h_1 = self.init_h01
        hs[:, :, :, -1] = z * _h + (1-z) * h_1  

        # 循环更新隐状态
        for i in range(1, zs.shape[2]):
            for j in range(zs.shape[3]-2, -1, -1):
                z = zs[:, :, i, j]
                _h = _hs[:, :, i, j]
                h_1 = hs[:, :, i-1, j+1]
                hs[:, :, i, j] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 左下方向
    def d10(self, zs, _hs):
        hs = torch.zeros_like(_hs)  
        # 初始化隐状态传递，最后一行
        z = zs[:, :, -1, :]
        _h = _hs[:, :, -1, :]
        h_1 = self.init_h10
        hs[:, :, -1, :] = z * _h + (1-z) * h_1  
        # 初始化隐状态传递，第一列
        z = zs[:, :, :, 0]
        _h = _hs[:, :, :, 0]
        h_1 = self.init_h10
        hs[:, :, :, 0] = z * _h + (1-z) * h_1  
        # 循环更新隐状态
        for i in range(zs.shape[2]-2, -1, -1):
            for j in range(1, zs.shape[3]): 
                z = zs[:, :, i, j]
                _h = _hs[:, :, i, j]
                h_1 = hs[:, :, i+1, j-1]
                hs[:, :, i, j] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 右下方向
    def d11(self, zs, _hs):
        hs = torch.zeros_like(_hs)  
        # 初始化隐状态传递，最后一行
        z = zs[:, :, -1, :]
        _h = _hs[:, :, -1, :]
        h_1 = self.init_h11
        hs[:, :, -1, :] = z * _h + (1-z) * h_1  
        # 初始化隐状态传递，最后一列
        z = zs[:, :, :, -1]
        _h = _hs[:, :, :, -1]
        h_1 = self.init_h11
        hs[:, :, :, -1] = z * _h + (1-z) * h_1  
        # 循环更新隐状态
        for i in range(zs.shape[2]-2, -1, -1):
            for j in range(zs.shape[3]-2, -1, -1):  
                z = zs[:, :, i, j]
                _h = _hs[:, :, i, j]
                h_1 = hs[:, :, i+1, j+1]
                hs[:, :, i, j] = z * _h + (1-z) * h_1
        return hs

    def forward(self, xs):
        '''
        输入：xs 输入序列，形状为 (batch_size, seq_len, in_channels)
        输出：h 更新后的隐状态，形状为 (batch_size, seq_len, hidden_size)
        '''
        # 1.并行计算8个方向的更新门
        zs8 = torch.sigmoid(self.z(xs)).unflatten(1, (8, -1)).unbind(dim=1)

        # 2.并行计算8个方向的候选隐状态
        _hs8 = self._h(xs).unflatten(1, (8, -1)).unbind(dim=1)

        # 3.计算8方向选择门
        ss8 = torch.sigmoid(self.s(xs)).unflatten(1, (8, -1)).unbind(dim=1)

        # 4.依次更新8个方向的隐状态
        hs8 = [self.d20(zs8[0], _hs8[0]),
               self.d21(zs8[1], _hs8[1]),
               self.d30(zs8[2], _hs8[2]),
               self.d31(zs8[3], _hs8[3]),
               self.d00(zs8[4], _hs8[4]),
               self.d01(zs8[5], _hs8[5]),
               self.d10(zs8[6], _hs8[6]),
               self.d11(zs8[7], _hs8[7])]

        # 5.将8个方向的隐状态选择性相加
        hs = torch.sum(torch.stack(ss8) * torch.stack(hs8), dim=0)

        # 6.返回最终的隐状态
        return hs

from conv.FurnaceConvV2 import FurnaceConvBN2d

class MiniGRUConv2d4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.z = FurnaceConvBN2d(in_channels, out_channels * 4, kernel_size, stride, padding)  # 更新门
        self._h = FurnaceConvBN2d(in_channels, out_channels * 4, kernel_size, stride, padding)  # 隐状态
        self.s = FurnaceConvBN2d(in_channels, out_channels * 4, kernel_size, stride, padding)  # 选择门

        # 初始化隐状态, 4个正方向
        self.init_h20 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h21 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h30 = nn.Parameter(torch.zeros(1, out_channels, 1))
        self.init_h31 = nn.Parameter(torch.zeros(1, out_channels, 1))
    
    def fuse(self):
        self.z = self.z.fuse()
        self._h = self._h.fuse()
        self.s = self.s.fuse()
        self.back_fuse = [self.z, self._h, self.s]
    
    def back_fuse(self):
        self.z, self._h, self.s = self.back_fuse

    # 更新隐状态, 上方向
    def d20(self, zs, _hs):
        hs = torch.zeros_like(_hs) 
        # 初始化隐状态传递
        z = zs[:, :, 0, :]
        _h = _hs[:, :, 0, :]
        h_1 = self.init_h20
        hs[:, :, 0, :] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(1, zs.shape[2]):
            z = zs[:, :, i, :]
            _h = _hs[:, :, i, :]
            h_1 = hs[:, :, i-1, :]
            hs[:, :, i, :] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 下方向
    def d21(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递
        z = zs[:, :, -1, :]
        _h = _hs[:, :, -1, :]
        h_1 = self.init_h21
        hs[:, :, -1, :] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(zs.shape[2]-2, -1, -1):
            z = zs[:, :, i, :]
            _h = _hs[:, :, i, :]
            h_1 = hs[:, :, i+1, :]
            hs[:, :, i, :] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 左方向
    def d30(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递
        z = zs[:, :, :, 0] 
        _h = _hs[:, :, :, 0]
        h_1 = self.init_h30
        hs[:, :, :, 0] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(1, zs.shape[3]):
            z = zs[:, :, :, i]
            _h = _hs[:, :, :, i]
            h_1 = hs[:, :, :, i-1]
            hs[:, :, :, i] = z * _h + (1-z) * h_1
        return hs

    # 更新隐状态, 右方向
    def d31(self, zs, _hs):
        hs = torch.zeros_like(_hs)
        # 初始化隐状态传递
        z = zs[:, :, :, -1]
        _h = _hs[:, :, :, -1]
        h_1 = self.init_h31
        hs[:, :, :, -1] = z * _h + (1-z) * h_1 
        # 循环更新隐状态
        for i in range(zs.shape[3]-2, -1, -1):
            z = zs[:, :, :, i]
            _h = _hs[:, :, :, i]
            h_1 = hs[:, :, :, i+1]
            hs[:, :, :, i] = z * _h + (1-z) * h_1
        return hs
 
    def forward(self, xs): 
        # 1.并行计算4个方向的更新门
        zs4 = torch.sigmoid(self.z(xs)).unflatten(1, (4, -1)).unbind(dim=1)

        # 2.并行计算4个方向的候选隐状态
        _hs4 = self._h(xs).unflatten(1, (4, -1)).unbind(dim=1)

        # 3.计算4方向选择门
        ss4 = torch.sigmoid(self.s(xs)).unflatten(1, (4, -1)).unbind(dim=1)

        # 4.依次更新4个方向的隐状态
        hs4 = [self.d20(zs4[0], _hs4[0]),
               self.d21(zs4[1], _hs4[1]),
               self.d30(zs4[2], _hs4[2]),
               self.d31(zs4[3], _hs4[3])  ]

        # 5.将4个方向的隐状态选择性相加
        hs = torch.sum(torch.stack(ss4) * torch.stack(hs4), dim=0)

        # 6.返回最终的隐状态
        return hs
    

if __name__ == '__main__':
    x = torch.randn(1, 16, 10, 16)
    gru8 = MiniGRUConv2d8(16, 16)
    h = gru8(x)
    print(h.shape)
    gru4 = MiniGRUConv2d4(16, 16)
    h = gru4(x)
    print(h.shape)
    from torchsummary import summary
    summary(gru8.cuda(), (16, 10, 16))
    summary(gru4.cuda(), (16, 10, 16))
