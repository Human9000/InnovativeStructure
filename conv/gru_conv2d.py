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
        self.fh = (lambda x: torch.flip(x, dims=[2])) if self.dh < 0 else (lambda x: x)
        self.fw = (lambda x: torch.flip(x, dims=[3])) if self.dw < 0 else (lambda x: x)

        if self.dh == 0:
            self.forward = self.forward_w
        elif self.dw == 0:
            self.forward = self.forward_h
        else:
            self.forward = self.forward_wh

    def forward_h(self, z, _h, h0):
        B, C, H, W = z.shape
        dh = abs(self.dh)
        _h = self.fh(_h)
        z = self.fh(z)
        h = torch.empty(B, C, H + dh, W, device=z.device)
        h[:, :, :dh, :] = h0

        for th in range(H):
            h_1 = h[:, :, th, :]
            zt = z[:, :, th, :]
            _ht = _h[:, :, th, :]
            h[:, :, th + dh, :] = zt * _ht + (1 - zt) * h_1

        return self.fh(h[:, :, dh:, :])

    def forward_w(self, z, _h, h0):
        B, C, H, W = z.shape
        dw = abs(self.dw)
        _h = self.fw(_h)
        z = self.fw(z)
        h = torch.empty(B, C, H, W + dw, device=z.device)
        h[:, :, :, :dw] = h0

        for tw in range(W):
            h_1 = h[:, :, :, tw]
            zt = z[:, :, :, tw]
            _ht = _h[:, :, :, tw]
            h[:, :, :, tw + dw] = zt * _ht + (1 - zt) * h_1

        return self.fw(h[:, :, :, dw:])

    def forward_wh(self, z, _h, h0):
        B, C, H, W = z.shape
        dh, dw = abs(self.dh), abs(self.dw)
        h = torch.empty(B, C, H + dh, W + dw, device=z.device)
        _h = self.fw(self.fh(_h))
        z = self.fw(self.fh(z))
        h[:, :, :dh, :dw] = h0

        for th in range(H):
            for tw in range(W):
                h_1 = h[:, :, th, tw]
                zt = z[:, :, th, tw]
                _ht = _h[:, :, th, tw]
                h[:, :, th + dh, tw + dw] = zt * _ht + (1 - zt) * h_1

        h = h[:, :, dh:, dw:]
        return self.fw(self.fh(h))


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

        self._h = nn.ModuleList()   # 隐状态
        self.h0 = nn.ParameterList()  # 初始隐状态
        self.z = nn.ModuleList()  # 更新门
        self.s = nn.ModuleList()  # 选择门
        self.d = nn.ModuleList()  # 方向更新策略

        for dh, dw in directions:
            self._h.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
            self.h0.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1,)))
            self.z.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.Sigmoid()))
            self.s.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.Sigmoid()))
            self.d.append(GruDirection2d(dh, dw))

    def forward(self, x):
        hs = []
        _hs = [h_conv(x) for h_conv in self._h]
        z_outputs = [z_conv(x) for z_conv in self.z]
        s_outputs = [s_conv(x) for s_conv in self.s]

        for h0, _h, z, s, d in zip(self.h0, _hs, z_outputs, s_outputs, self.d):
            hs.append(d(z, _h, h0) * s)

        return torch.sum(torch.stack(hs), dim=0)


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
