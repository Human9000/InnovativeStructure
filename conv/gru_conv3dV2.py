import torch
from torch import nn


class SortAttn3d(nn.Module):
    def __init__(self, in_channels_list, fact=4, group=2):
        super(SortAttn3d, self).__init__()
        g = len(in_channels_list) * group
        in_channels = sum(in_channels_list) 
        hidden_size = in_channels//fact 
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels*2, hidden_size, 1, groups=2*g, bias=False), 
            nn.ChannelShuffle(g),
            nn.Conv3d(hidden_size, hidden_size, 1, groups=g, bias=False),
            nn.ChannelShuffle(g),
            nn.Conv3d(hidden_size, hidden_size, 1, groups=g, bias=False),
            nn.ReLU(),  # 非线性激活 
            nn.ChannelShuffle(g),
            nn.Conv3d(hidden_size, in_channels, 1, groups=g, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, xs):
        # 对隐藏层进行通道排序
        n = len(xs)
        x  = torch.cat(xs, dim=1)
        sorted_x, indices = torch.sort(x, dim=1, stable=True)
        # 残差链接排序通道
        se_hidden = torch.cat((x, sorted_x), dim=1)   # b, 2*c, ...
        print(se_hidden.shape)
        print(self.attention)
        # 生成注意力分数
        attention_scores = self.attention(se_hidden)  # b,c
        print(attention_scores.shape)
        print(x.shape)
        # 叠加注意力结果 
        y = ( x * attention_scores).unflatten(1, (n, -1)).sum(1)
        return y


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
            self.flip = (lambda x: torch.flip(x, dims=[idx+2])) if ds[idx] < 0 else (lambda x: x)
            self.trans = lambda x: x.transpose(-1, (idx+2).item())  # 将不为0的维度移动到-1维度
            self.d = ds[idx].item()
            self.forward = self.forward_1
        elif nd == 2:  # 2个维度不为0
            flip_idxs = torch.where(ds < 0)[0]    # 获取需要翻转的维度索引
            self.flip = (lambda x: torch.flip(x, dims=(flip_idxs+2).numpy().tolist())) \
                if flip_idxs.shape[0] > 0 else (lambda x: x)
            idx = torch.where(ds == 0)[0]  # 获取为0的维度索引
            self.trans = lambda x: x.transpose(2, (idx+2).item())  # 将为0的维度移动到末尾 
            self.d = torch.cat([ds[:idx] ,ds[idx+1:]], dim=0)
            self.forward = self.forward_2
        else:
            flip_idxs = torch.where(ds < 0)[0]    # 获取需要翻转的维度索引
            self.flip = (lambda x: torch.flip(x, dims=(flip_idxs+2).numpy().tolist())) \
                if flip_idxs.shape[0] > 0  else (lambda x: x)
       
            self.d = ds 
            self.forward = self.forward_3

    def forward_1(self, z, _h, h0):     # 定义forward_h函数
        _h = self.trans(self.flip(_h))  # 根据d调整输入_h的形状
        z = self.trans(self.flip(z))    # 根据d调整输入z的形状
        B, C, D, H, W = z.shape         # 获取输入z的形状
        d = abs(self.d)                 # 计算d的绝对值
        h = torch.ones((B, C, D, H, W+d), device=z.device) * h0  # 初始化输出h
        t_num = (W+d-1)//d              # 计算时间步数
        ts = torch.arange(t_num+2)*d    # 计算时间步列表 
        for ti in range(t_num):         # 循环更新h
            t = ts[ti:ti+3]             # 获取当前时间步的索引
            h_1 =  h[..., t[0]:t[1]]    # H维度，获取当前时间步的h_1
            zt =   z[..., t[0]:t[1]]    # H维度，获取当前时间步的z
            _ht = _h[..., t[0]:t[1]]    # H维度，获取当前时间步的_h
            h[..., t[1]:t[2]] = zt * _ht + (1 - zt) * h_1  # H维度，更新h
        h = h[..., d:]                  # 截取有效部分的h
        return self.flip(self.trans(h)) # 返回翻转后的h

    def forward_2(self, z, _h, h0):     # 定义forward_wh函数
        _h = self.trans(self.flip(_h))  # 根据dh和dw的符号翻转输入_h
        z = self.trans(self.flip(z))    # 根据dh和dw的符号翻转输入z   
        B, C, D, H, W = z.shape         # 获取输入z的形状
        dh, dw = abs(self.d)            # 计算dh和dw的绝对值
        h = torch.ones((B, C, D, H + dh, W + dw), device=z.device) * h0  # 初始化输出h
        t_num = min((H+dh-1)//dh, (W+dw-1)//dw)         # 计算时间步数
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
        return self.flip(self.trans(h[..., dh:, dw:] )) # 截取有效部分的h（去掉初始的dh行和dw列） ,返回翻转后的h

    def forward_3(self, z, _h, h0):  # 定义forward_wh函数
        _h = self.flip(_h)           # 根据dh和dw的符号翻转输入_h
        z = self.flip(z)             # 根据dh和dw的符号翻转输入z
        B, C, D, H, W = z.shape      # 获取输入z的形状
        dd, dh, dw = abs(self.d)     # 计算dh和dw的绝对值
        h = torch.ones((B, C, D+dd, H + dh, W + dw), device=z.device) * h0  # 初始化输出h 
        t_num = min((D+dd-1)//dd, (H+dh-1)//dh, (W+dw-1)//dw) # 计算时间步数
        _ts = torch.arange(t_num+2)                           # 生成时间步列表
        ts = torch.stack((_ts*dd, _ts*dh, _ts*dw), dim=0)     # 生成多维度时间步列表
        for ti in range(t_num):                               # 循环更新h隐状态
            t = ts[:,ti:ti+3]                                 # 获取当前时间步的宽高索引
            h_1 =  h[..., t[0,0]:t[0,1], t[1,0]:-dh   , t[2,0]:-dw   ]    # D维度，获取当前时间步的h_1
            zt =   z[..., t[0,0]:t[0,1], t[1,0]:      , t[2,0]:      ]    # D维度，获取当前时间步的z
            _ht = _h[..., t[0,0]:t[0,1], t[1,0]:      , t[2,0]:      ]    # D维度，获取当前时间步的_h
            h[..., t[0,1]:t[0,2], t[1,1]:, t[2,1]: ] = zt * _ht + (1 - zt) * h_1  # D维度，更新h
            h_1 =  h[..., t[0,1]:-dd   , t[1,0]:t[1,1], t[2,0]:-dw   ]    # H维度，获取当前时间步的h_1
            zt =   z[..., t[0,1]:      , t[1,0]:t[1,1], t[2,0]:      ]    # H维度，获取当前时间步的z
            _ht = _h[..., t[0,1]:      , t[1,0]:t[1,1], t[2,0]:      ]    # H维度，获取当前时间步的_h
            h[..., t[0,2]:, t[1,1]:t[1,2], t[2,1]:] = zt * _ht + (1 - zt) * h_1  # H维度，更新h
            h_1 =  h[..., t[0,1]:-dd   , t[1,1]:-dh   , t[2,0]:t[2,1]]    # W维度，获取当前时间步的h_1
            zt =   z[..., t[0,1]:      , t[1,1]:      , t[2,0]:t[2,1]]    # W维度，获取当前时间步的z
            _ht = _h[..., t[0,1]:      , t[1,1]:      , t[2,0]:t[2,1]]    # W维度，获取当前时间步的_h 
            h[..., t[0,2]:, t[1,2]:, t[2,1]:t[2,2]] = zt * _ht + (1 - zt) * h_1  # W维度，更新h  
        h = h[..., dd:, dh:, dw:]            # 截取有效部分的h（去掉初始的dh行和dw列） 
        return self.flip(h)          # 返回翻转后的h


class SE3d(nn.Module):
    def __init__(self, in_channels, fact=4 ):
        super(SE3d, self).__init__() 
        hidden_size = in_channels//fact
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels*2, hidden_size, 1, bias=False),  
            nn.ReLU(),
            nn.Conv3d(hidden_size, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_mean = x.mean(dim=[2,3,4], keepdim=True)
        x_max = x.amax(dim=[2,3,4], keepdim=True)
        x_se = torch.cat((x_mean, x_max), dim=1)
        return x * self.attention(x_se)    

class GRUConv3d(nn.Module):
    """ 多方向3dGRUConv模块 """ 
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
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
        n,c = len(directions), out_channels  # 获取方向数和输出通道数 
        self.h0 = nn.Parameter(torch.zeros(n, 1, c, 1, 1, 1)) # 初始隐状态
        self._h = nn.ModuleList() # 隐状态
        self.z = nn.ModuleList()  # 更新门 
        self.d = nn.ModuleList()  # 方向更新策略
        self.nd_attn = SortAttn3d([out_channels,]*n, fact=4) # n方向注意力
        self.z_en = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)  # 更新编码器
        self._h_en = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)  # 候选隐状态编码器

        for d in directions:
            self.d.append(GruDirection3d(*d))
            self._h.append(SE3d(out_channels))
            self.z.append(nn.Sequential(SE3d(out_channels), nn.Sigmoid()))

    def forward(self, x):
        z_feature = self.z_en(x)                            # 并行计算更新特征
        _h_feature = self._h_en(x)                          # 并行计算候选隐状态特征
        _hs = [h_conv(_h_feature) for h_conv in self._h]    # 并行计算多方向隐状态
        zs = [z_conv(z_feature) for z_conv in self.z]       # 并行计算多方向更新门输出
        hs = [d(z, _h, h0)  for h0, _h, z,  d in zip(self.h0, _hs, zs, self.d)]  # 并行更新计算多方向隐状态
        return self.nd_attn(hs)  # 返回所有方向的注意力选择输出


if __name__ == '__main__': 
    gru = GRUConv3d(16, 16, 3, 1, 1, groups=1, direction_num=26).cuda()
    conv = nn.Conv3d(16, 16, 3, 1, 1 ).cuda()
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(gru, (16, 64, 64, 64),
                                             as_strings=True,
                                             print_per_layer_stat=True)
    print('macs: ', macs, 'params: ', params)
    macs, params = get_model_complexity_info(conv, (16, 64, 64, 64),
                                             as_strings=True,
                                             print_per_layer_stat=False)
    print('macs: ', macs, 'params: ', params)
