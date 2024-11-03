import torch
from torch import nn


class GRUParallelDirection2d(nn.Module):
    """ 并行的2d GRU 多方向更新策略, 支持水平, 垂直, 倾斜, 跨步长 的更新策略"""

    def __init__(self, d1=1, d2=1):
        super().__init__()
        ds = torch.tensor([d1, d2])
        nd = (ds != 0).sum()
        if nd == 0:
            raise ValueError("d1, d2 cannot be all 0")

        if nd == 1:  # 1个维度不为0
            idx = torch.where(ds != 0)[0]  # 获取不为0的维度索引
            self.flip = (lambda x: torch.flip(x, dims=[idx+2])) if ds[idx] < 0 else (lambda x: x)
            self.trans = lambda x: x.transpose(-1, (idx+2).item())  # 将不为0的维度移动到末尾
            self.d = ds[idx].item()
            self.forward = self.forward_1
        elif nd == 2:  # 2个维度不为0
            flip_idxs = torch.where(ds < 0)[0]    # 获取需要翻转的维度索引
            self.flip = (lambda x: torch.flip(x, dims=(flip_idxs+2).numpy().tolist())) if flip_idxs.shape[0] > 0 else (lambda x: x)
            self.d = ds
            self.forward = self.forward_2

    def forward_1(self, z, _h, h0):     # 定义forward_h函数
        _h = self.trans(self.flip(_h))  # 根据d调整输入_h的形状
        z = self.trans(self.flip(z))    # 根据d调整输入z的形状
        B, C, H, W = z.shape            # 获取输入z的形状
        d = abs(self.d)                 # 计算d的绝对值
        h = torch.empty(B, C, H, W+d, device=z.device)  # 初始化输出h
        h[..., :d] = h0                 # 初始化h的前d行 
        t_num = (H+d-1)//d              # 计算时间步数
        ts = torch.arange(t_num+2)*d    # 计算时间步列表 
        for ti in range(t_num):         # 循环更新h
            t = ts[ti:ti+3]             # 获取当前时间步的索引
            h_1 =  h[..., t[0]:t[1]]    # H维度，获取当前时间步的h_1
            zt =   z[..., t[0]:t[1]]    # H维度，获取当前时间步的z
            _ht = _h[..., t[0]:t[1]]    # H维度，获取当前时间步的_h
            h[..., t[1]:t[2]] = zt * _ht + (1 - zt) * h_1  # H维度，更新h
        h = h[..., d:]                  # 截取有效部分的h
        return self.flip(self.trans(h)) # 返回翻转后的h

    def forward_2(self, z, _h, h0):     # 定义forward_h函数
        _h = self.flip(_h)              # 根据d调整输入_h的形状
        z  = self.flip(z)               # 根据d调整输入z的形状 
        B, C, H, W = z.shape            # 获取输入z的形状
        dh, dw = abs(self.d[0]), abs(self.d[1])  # 计算dh和dw的绝对值
        h = torch.empty(B, C, H + dh, W + dw, device=z.device)  # 初始化输出h
        h[..., :dh, :dw] = h0                           # 初始化h的前dh行和前dw列
        t_num = min((H+dh-1)//dh , (W+dw-1)//dw  )      # 计算时间步数
        ts = torch.arange(t_num+2)                      # 生成时间步列表
        thws = torch.stack((ts*dh, ts*dw), dim=0)       # 生成宽高时间步列表
        for ti in range(t_num):                         # 循环更新h隐状态
            t = thws[:,ti:ti+3]                         # 获取当前时间步的宽高索引
            h_1 =  h[..., t[0,0]:t[0,1], t[1,0]:-dw]    # H维度，获取当前时间步的h_1
            zt =   z[..., t[0,0]:t[0,1], t[1,0]:   ]    # H维度，获取当前时间步的z
            _ht = _h[..., t[0,0]:t[0,1], t[1,0]:   ]    # H维度，获取当前时间步的_h
            h[..., t[0,1]:t[0,2], t[1,1]:] = zt * _ht + (1 - zt) * h_1  # H维度，更新h
            h_1 =  h[..., t[0,1]:-dh, t[1,0]:t[1,1]]    # W维度，获取当前时间步的h_1
            zt =   z[..., t[0,1]:   , t[1,0]:t[1,1]]    # W维度，获取当前时间步的z
            _ht = _h[..., t[0,1]:   , t[1,0]:t[1,1]]    # W维度，获取当前时间步的_h
            h[..., t[0,2]:, t[1,1]:t[1,2]] = zt * _ht + (1 - zt) * h_1  # W维度，更新h 
        h = h[..., dh:, dw:]            # 截取有效部分的h（去掉初始的dh行和dw列）
        return self.flip(h)  # 返回翻转后的h


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
                 directions=None,  # 自定义方向更新策略
                 direction_num=8,  # 默认的方向更新策略
                 ):
        super().__init__()
        if directions is None:
            if direction_num == 8:      # 8 方向更新
                directions = [(1, 2), (1, -1), (-1, 1), (-2, -1), (1, 0), (-2, 0), (0, 1), (0, -1)]
            elif direction_num == 4:    # 4 方向更新
                directions = [(2, 0), (-3, 0), (0, 1), (0, -2)]
            else:
                raise ValueError(f"direction_num must be 4 or 8, but got {direction_num}")

        self.h0 = nn.ParameterList()    # 多方向初始隐状态
        self._h = nn.ModuleList()       # 多方向候选隐状态
        self.z = nn.ModuleList()        # 多方向更新门
        self.s = nn.ModuleList()        # 多方向选择门
        self.d = nn.ModuleList()        # 多方向更新策略

        self.z_en = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)  # 更新编码器
        self.s_en = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)  # 选择编码器
        self._h_en = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)  # 候选隐状态编码器

        for dh, dw in directions:
            self.d.append(GRUParallelDirection2d(dh, dw))                                                    # 更新策略
            self.h0.append(nn.Parameter(torch.zeros(1, out_channels, 1, 1,)))                                # 初始隐状态
            self._h.append(nn.Conv2d(out_channels, out_channels, 1, bias=bias))                              # 候选隐状态
            self.z.append(nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=bias), nn.Sigmoid()))  # 更新门
            self.s.append(nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=bias), nn.Sigmoid()))  # 选择门

    def forward(self, x):
        z_feature = self.z_en(x)                            # 并行计算更新特征
        s_feature = self.s_en(x)                            # 并行计算选择特征
        _h_feature = self._h_en(x)                          # 并行计算候选隐状态特征
        _hs = [h_conv(_h_feature) for h_conv in self._h]    # 并行计算多方向隐状态
        zs = [z_conv(z_feature) for z_conv in self.z]       # 并行计算多方向更新门输出
        ss = [s_conv(s_feature) for s_conv in self.s]       # 并行计算多方向选择门输出
        hs = [d(z, _h, h0) * s for h0, _h, z, s, d in zip(self.h0, _hs, zs, ss, self.d)]  # 动规并行更新计算多方向隐状态+选择门输出
        return torch.sum(torch.stack(hs), dim=0)            # 多方向隐状态选择门输出和


if __name__ == '__main__':
    # x = torch.randn(1, 10, 512, 512).cuda()
    # gru = GRUConv2d(10, 16, direction_num=4).cuda()
    # # print(gru)
    # h = gru(x)
    # print(h.shape)
    gru = GRUConv2d(64, 64, direction_num=8).cuda()
    # # print(gru)
    # h = gru(x)
    # print(h.shape)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(gru, (64, 64, 64),
                                             as_strings=True,
                                             print_per_layer_stat=False)
    print('macs: ', macs, 'params: ', params)
