import torch
from torch import nn


class GruDirection3d(nn.Module):
    """ 动态规划实现3d多方向更新, 支持水平, 垂直, 倾斜, 跨步长 的更新策略"""

    def __init__(self, d1=1, d2=1, d3=1):
        super().__init__()
        ds = torch.tensor([d1, d2, d3])
        nd = (ds != 0).sum()
        if nd == 0:
            raise ValueError("d1, d2, d3 cannot be all 0")

        if nd == 1:  # 1个维度不为0
            idx = torch.where(ds != 0)[0]  # 获取不为0的维度索引
            self.flip = lambda x: torch.flip(x, dims=[idx+2]) if ds[idx] < 0 else (lambda x: x)
            self.trans = lambda x: x.transpose(2, idx+2)  # 将不为0的维度移动到2维度
            self.d = ds[idx]
            self.forward = self.forward_1
        elif nd == 2:  # 2个维度不为0
            flip_idxs = torch.where(ds < 0)[0]    # 获取需要翻转的维度索引
            self.flip = (lambda x: torch.flip(x, dims=flip_idxs+2)) if flip_idxs.shape[0] > 0 else (lambda x: x)
            idx = torch.where(ds == 0)[0]  # 获取为0的维度索引
            self.trans = lambda x: x.transpose(-1, idx+2)  # 将为0的维度移动到末尾
            # idx = 0 = > 2, 1, 0
            # idx = 1 = > 0, 2, 1
            # idx = 2 = > 0, 1, 2
            self.d = [(ds[2], ds[1]),
                      (ds[0], ds[2]),
                      (ds[0], ds[1]),][idx]
            self.forward = self.forward_2
        else:
            flip_idxs = torch.where(ds < 0)[0]    # 获取需要翻转的维度索引
            self.flip = lambda x: torch.flip(x, dims=flip_idxs+2) if flip_idxs.shape[0] > 0  else (lambda x: x)
            self.trans = lambda x: x
            self.d = ds 
            self.forward = self.forward_3

    def forward_1(self, z, _h, h0):         # 定义forward_h函数
        _h = self.trans(self.flip(_h))       # 根据dd的符号翻转输入_h
        z = self.trans(self.flip(z))         # 根据dd的符号翻转输入z
        B, C, D, H, W = z.shape             # 获取输入z的形状
        dd = abs(self.d)                     # 计算d的绝对值
        h = torch.empty(B, C, D + dd, H, W, device=z.device)  # 初始化输出h
        h[:, :, :dd] = h0                # 初始化h的前dd行
        for td in range(D):              # 循环更新D
            h_1 = h[:, :, td]            # 获取当前时间步的h
            zt = z[:, :, td]             # 获取当前时间步的z
            _ht = _h[:, :, td]           # 获取当前时间步的_h       
            h[:, :, td + dd,] = zt * _ht + (1 - zt) * h_1  # 更新h
        return self.flip(self.trans(h[:, :, dd:]))     # 返回翻转后的h

    def forward_2(self, z, _h, h0):        # 定义forward_wh函数
        _h = self.trans(self.flip(_h))           # 根据dh和dw的符号翻转输入_h
        z = self.trans(self.flip(z))             # 根据dh和dw的符号翻转输入z 
        
        # 将维度数最小的维度，移动到 4(W) 维度
        min_dim = torch.argmin(torch.tensor(z.shape[2:])).item()
        z = z.transpose(4, min_dim+2)
        _h = _h.transpose(4, min_dim+2) 

        B, C, H, W, D = z.shape             # 获取输入z的形状
        dh, dw = abs(self.d[0]), abs(self.d[1])  # 计算dh和dw的绝对值
        h = torch.empty(B, C, H + dh, W + dw, D, device=z.device)  # 初始化输出h
        h[:, :, :dh, :dw] = h0              # 初始化h的前dh行和前dw列

        for t in range(W): 
            # 更新H维度
            h_1 = h[:, :, t, t:]           # 获取当前时间步的h
            zt = z[:, :, t, t:]            # 获取当前时间步的z
            _ht = _h[:, :, t, t:]          # 获取当前时间步的_h
            h[:, :, t + dh, t + dw:] = zt * _ht + (1 - zt) * h_1  # 更新当前时间步的h

            # 更新W维度
            h_1 = h[:, :, t+1:, t]           # 获取当前时间步的h
            zt = z[:, :, t+1:, t]            # 获取当前时间步的z
            _ht = _h[:, :, t+1:, t]          # 获取当前时间步的_h
            h[:, :, t + dh + 1:, t + dw] = zt * _ht + (1 - zt) * h_1  # 更新当前时间步的h

        h = h[:, :, dh:, dw:]               # 截取有效部分的h 
        h = h.transpose(4, min_dim+2)           # 还原“维度数最小的维度，移动到4”
        return self.flip(self.trans(h[:, :, dh:, dw:]))          # 返回翻转后的h

    def forward_3(self, z, _h, h0):        # 定义forward_wh函数
        _h = self.trans(self.flip(_h))           # 根据dh和dw的符号翻转输入_h
        z = self.trans(self.flip(z))             # 根据dh和dw的符号翻转输入z

        # 将维度数最小的维度，移动到 5(D) 维度
        min_dim = torch.argmin(torch.tensor(z.shape[2:])).item()
        z = z.transpose(5, min_dim+2)
        _h = _h.transpose(5, min_dim+2)

        B, C, H, W, D = z.shape             # 获取输入z的形状
        dh, dw, dd = abs(self.d[0]), abs(self.d[1]), abs(self.d[2])  # 计算dh和dw的绝对值
        h = torch.empty(B, C, H + dh, W + dw, D + dd, device=z.device)  # 初始化输出h
        h[:, :, :dh, :dw, :dd] = h0              # 初始化h的前dh行和前dw列
        for t in range(D):
            # 每个维度分别并行计算 ===
            # 更新H维度
            h_1 = h[:, :, t, t:, t:]            # 获取当前时间步的h
            zt = z[:, :, t, t:, t:]             # 获取当前时间步的z
            _ht = _h[:, :, t, t:, t:]           # 获取当前时间步的_h
            h[:, :, t + dh, t + dw:, t + dd:] = zt * _ht + (1 - zt) * h_1  # 更新当前时间步的h

            # 更新W维度
            h_1 = h[:, :, t+1:, t, t:]           # 获取当前时间步的h
            zt = z[:, :, t+1:, t, t:]            # 获取当前时间步的z
            _ht = _h[:, :, t+1:, t, t:]          # 获取当前时间步的_h
            h[:, :, t + dh + 1, t + dw, t + dd:] = zt * _ht + (1 - zt) * h_1  # 更新当前时间步的h

            # 更新D维度
            h_1 =  h[:, :, t+1:, t+1:, t]           # 获取当前时间步的h
            _ht = _h[:, :, t+1:, t+1:, t]          # 获取当前时间步的_h
            zt = z[:, :, t+1:, t+1:, t]            # 获取当前时间步的z
            h[:, :, t + dh + 1, t + dw + 1, t + dd:] = zt * _ht + (1 - zt) * h_1  # 更新当前时间步的h

        h = h[:, :, dh:, dw:, dd:]              # 截取有效部分的h 
        h = h.transpose(5, min_dim+2)           # 还原“维度数最小的维度，移动到5”
        return self.flip(self.trans(h[:, :, dh:, dw:, dd:]))          # 返回翻转后的h


class GRUConv3d(nn.Module):
    """ 多方向3dGRUConv模块 """ 
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 directions = None, # 自定义方向更新策略, 比如[T,H,W]的情况下[[1,0,0],[-1,0,0]] 表示在T维度上正负步长更新
                 direction_num=8, # 默认的方向更新策略
                 ):
        super().__init__()
        if directions is None:
            if direction_num == 26:  # 26 方向更新
                directions = [(i, j, k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if not sum([i==0,j==0,k==0])==3] # 排除2个维度为0的情况
            elif direction_num == 20:  # 20 方向更新 
                directions = [(i, j, k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if not sum([i==0,j==0,k==0])>=2] # 排除2-3个维度为0的情况
            elif direction_num == 8:  # 8 方向更新
                directions = [(i, j, k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if not sum([i==0,j==0,k==0])>=1] # 排除1-3个维度为0的情况
            elif direction_num == 12:  # 18 方向更新
                directions = [(i, j, k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if sum([i==0,j==0,k==0])==1] # 包含1个维度为0的情况
            elif direction_num == 6:  # 6 方向更新
                directions = [(i, j, k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if sum([i==0,j==0,k==0])==2] # 包含2个维度为0的情况
            else:
                raise ValueError(f"direction_num must be 4 or 8, but got {direction_num}")

        self.h0 = nn.ParameterList()  # 初始隐状态
        self._h = nn.ModuleList()   # 隐状态
        self.z = nn.ModuleList()  # 更新门
        self.s = nn.ModuleList()  # 选择门
        self.d = nn.ModuleList()  # 方向更新策略

        for dh, dw in directions:
            self.d.append(GruDirection3d(dh, dw))
            self.h0.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1,)))
            self._h.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
            self.z.append(nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.Sigmoid()))
            self.s.append(nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias), nn.Sigmoid()))

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
