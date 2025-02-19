import torch
from torch import nn


class GruDirection2d(nn.Module):
    """ 动态规划实现2d多方向更新, 支持水平, 垂直, 倾斜, 跨步长 的更新策略"""

    def __init__(self, dh=1, dw=1):
        super().__init__()
        if dh == 0 and dw == 0:
            raise ValueError("dh and dw cannot be both 0")

        self.dh = dh
        self.dw = dw
        # 根据dh和dw的符号来决定是否需要翻转输入
        self.fh = (lambda x: torch.flip(x, dims=[2])) if self.dh < 0 else (lambda x: x)
        self.fw = (lambda x: torch.flip(x, dims=[3])) if self.dw < 0 else (lambda x: x)

        # 根据dh和dw的值来选择合适的forward函数
        if self.dh == 0:
            self.forward = self.forward_w
        elif self.dw == 0:
            self.forward = self.forward_h
        else:
            self.forward = self.forward_wh

    def forward_h(self, z, _h, h0):         # 定义forward_h函数
        B, C, H, W = z.shape                # 获取输入z的形状
        dh = abs(self.dh)                   # 计算dh的绝对值
        _h = self.fh(_h)                    # 根据dh的符号翻转输入_h
        z = self.fh(z)                      # 根据dh的符号翻转输入z
        h = torch.empty(B, C, H + dh, W, device=z.device)  # 初始化输出h
        h[:, :, :dh, :] = h0                # 初始化h的前dh行
        for th in range(H):                 # 循环更新H
            h_1 = h[:, :, th, :]            # 获取当前时间步的h
            zt = z[:, :, th, :]             # 获取当前时间步的z
            _ht = _h[:, :, th, :]           # 获取当前时间步的_h
            h[:, :, th + dh, :] = zt * _ht + (1 - zt) * h_1  # 更新h
        return self.fh(h[:, :, dh:, :])     # 返回翻转后的h

    def forward_w(self, z, _h, h0):         # 定义forward_w函数
        B, C, H, W = z.shape                # 获取输入z的形状
        dw = abs(self.dw)                   # 计算dw的绝对值
        _h = self.fw(_h)                    # 根据dw的符号翻转输入_h
        z = self.fw(z)                      # 根据dw的符号翻转输入z
        h = torch.empty(B, C, H, W + dw, device=z.device)  # 初始化输出h
        h[:, :, :, :dw] = h0                # 初始化h的前dw列
        for tw in range(W):                 # 循环更新W
            h_1 = h[:, :, :, tw]            # 获取当前时间步的h
            zt = z[:, :, :, tw]             # 获取当前时间步的z
            _ht = _h[:, :, :, tw]           # 获取当前时间步的_h
            h[:, :, :, tw + dw] = zt * _ht + (1 - zt) * h_1  # 更新h
        return self.fw(h[:, :, :, dw:])     # 返回翻转后的h

    def forward_wh(self, z, _h, h0):        # 定义forward_wh函数
        B, C, H, W = z.shape                # 获取输入z的形状
        dh, dw = abs(self.dh), abs(self.dw) # 计算dh和dw的绝对值
        h = torch.empty(B, C, H + dh, W + dw, device=z.device)  # 初始化输出h
        _h = self.fw(self.fh(_h))           # 根据dh和dw的符号翻转输入_h
        z = self.fw(self.fh(z))             # 根据dh和dw的符号翻转输入z
        h[:, :, :dh, :dw] = h0              # 初始化h的前dh行和前dw列

        for t in range(min(H, W)):
            Ht = torch.tensor(list(range(t, H, 1)) + [t,] * (W-t-1))  # 可以并行计算的Ht索引
            Wt = torch.tensor([t,] * (H-t) + list(range(t+1, W, 1)))  # 可以并行计算的Wt索引
            h_1 = h[:, :, Ht, Wt]           # 获取当前时间步的h
            zt = z[:, :, Ht, Wt]            # 获取当前时间步的z
            _ht = _h[:, :, Ht, Wt]          # 获取当前时间步的_h
            h[:, :, Ht + dh, Wt + dw] = zt * _ht + (1 - zt) * h_1  # 更新当前时间步的h
        h = h[:, :, dh:, dw:]               # 截取有效部分的h
        return self.fw(self.fh(h))          # 返回翻转后的h

    # def forward_wh2(self, z, _h, h0):        # 定义forward_wh函数
    #     B, C, H, W = z.shape                # 获取输入z的形状
    #     dh, dw = abs(self.dh), abs(self.dw) # 计算dh和dw的绝对值
    #     h = torch.empty(B, C, H + dh, W + dw, device=z.device)  # 初始化输出h
    #     _h = self.fw(self.fh(_h))           # 根据dh和dw的符号翻转输入_h
    #     z = self.fw(self.fh(z))             # 根据dh和dw的符号翻转输入z
    #     h[:, :, :dh, :dw] = h0              # 初始化h的前dh行和前dw列
    #     for th in range(H):                 # 循环更新H
    #         for tw in range(W):             # 循环更W
    #             h_1 = h[:, :, th, tw]       # 获取当前时间步的h
    #             zt = z[:, :, th, tw]        # 获取当前时间步的z
    #             _ht = _h[:, :, th, tw]      # 获取当前时间步的_h
    #             h[:, :, th + dh, tw + dw] = zt * _ht + (1 - zt) * h_1  # 更新h
    #     h = h[:, :, dh:, dw:]               # 截取有效部分的h
    #     return self.fw(self.fh(h))          # 返回翻转后的h


class GRUConv2d(nn.Module):
    """ 多方向2dGRUConv模块 """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 direction_num=8,):
        super().__init__()
        if direction_num == 8:  # 8 方向更新
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        elif direction_num == 4:  # 4 方向更新
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            raise ValueError(f"direction_num must be 4 or 8, but got {direction_num}")

        self.h0 = nn.ParameterList()  # 初始隐状态
        self._h = nn.ModuleList()   # 隐状态
        self.z = nn.ModuleList()  # 更新门
        self.s = nn.ModuleList()  # 选择门
        self.d = nn.ModuleList()  # 方向更新策略

        for dh, dw in directions:
            self.d.append(GruDirection2d(dh, dw))
            self.h0.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1,)))
            self._h.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
            self.z.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.Sigmoid()))
            self.s.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.Sigmoid()))

    def forward(self, x):
        _hs = [h_conv(x) for h_conv in self._h]  # 并行计算多方向隐状态
        zs = [z_conv(x) for z_conv in self.z]  # 并行计算多方向更新门输出
        ss = [s_conv(x) for s_conv in self.s]  # 并行计算多方向选择门输出
        hs = [d(z, _h, h0) * s for h0, _h, z, s, d in zip(self.h0, _hs, zs, ss, self.d)]  # 动规更新计算多方向隐状态
        return torch.sum(torch.stack(hs), dim=0)  # 返回所有方向输出的选择门输出和


if __name__ == '__main__':
    x = torch.randn(1, 10, 16, 32)
    gru = GRUConv2d(10, 16, direction_num=4)
    print(gru)
    h = gru(x)
    print(h.shape)

    gru = GRUConv2d(10, 16, direction_num=8)
    print(gru)
    h = gru(x)
    print(h.shape)