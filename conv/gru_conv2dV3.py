import torch
from torch import nn 


class SortAttn2d(nn.Sequential):
    def __init__(self, in_channels_list, fact=4, group=2):
        g = len(in_channels_list) * group  # 计算组数，等于输入通道数的数量乘以组数
        in_channels = sum(in_channels_list)  # 计算总的输入通道数
        hidden_size = in_channels // fact  # 计算隐藏层通道数，等于输入通道数除以压缩因子

        super().__init__(  # 调用父类的初始化方法
            nn.Conv2d(in_channels * 2, hidden_size, 1, groups=2 * g, bias=False),  # 第一个卷积层，输入通道数为总输入通道数的两倍，输出通道数为隐藏层通道数，卷积核大小为1，分组数为2*g
            nn.ChannelShuffle(g),  # 通道混洗层，按组数g对通道进行混洗
            nn.Conv2d(hidden_size, hidden_size, 1, groups=g, bias=False),  # 第二个卷积层，输入和输出通道数均为隐藏层通道数，卷积核大小为1，分组数为g
            nn.ChannelShuffle(g),  # 通道混洗层，按组数g对通道进行混洗
            nn.Conv2d(hidden_size, hidden_size, 1, groups=g, bias=False),  # 第三个卷积层，输入和输出通道数均为隐藏层通道数，卷积核大小为1，分组数为g
            nn.ReLU(),  # 非线性激活函数，使用ReLU激活
            nn.ChannelShuffle(g),  # 通道混洗层，按组数g对通道进行混洗
            nn.Conv2d(hidden_size, in_channels, 1, groups=g, bias=False),  # 第四个卷积层，输入通道数为隐藏层通道数，输出通道数为总输入通道数，卷积核大小为1，分组数为g
            nn.Sigmoid(),  # Sigmoid激活函数，输出注意力分数
        )
        
    def forward(self, xs):
        n = len(xs)
        x = torch.cat(xs, dim=1)
        # sorted_x, indices = torch.topk(x, k=x.shape[1], dim=1)  # 对隐藏层进行通道排序, 返回排序后的张量和索引，支持反向传播，慢，不稳定 
        sorted_x, indices = torch.sort(x, dim=1, stable=True)  # 对隐藏层进行通道排序, 返回排序后的张量和索引, 不支持反向传播，快，稳定
        se_hidden = torch.cat((x, sorted_x), dim=1)   # 残差链接排序通道 b, 2*c, ...
        attention_scores = super().forward(se_hidden)  # 生成注意力分数b,c
        y = (x * attention_scores).unflatten(1, (n, -1)).sum(1)  # 叠加注意力结果
        return y


class SE2d(nn.Module):
    def __init__(self, 
                 c_in,  # 输入通道数 in_channels
                 c_out, # 输出通道数 out_channels
                 k_s,   # 卷积核大小 kernel_size
                 s,     # 步长 stride
                 p,     # 填充 padding
                 d,     # 空洞 dilation
                 g,     # 分组 groups
                 b,     # 偏置 bias
                 fact=4,# 通道压缩因子
                 n=1,   # 多方向的方向数
                 ):
        super().__init__()
        self.c = c_out # 输出通道数
        h = c_out//fact # 隐藏层通道数
        self.en = nn.Conv2d(c_in, c_out, k_s, s, p, d, g, b) # 输入卷积层
        self.s = nn.Conv2d(self.c * 2, h * n, 1, bias=False) # 通道压缩层
        self.relu = nn.ReLU() # ReLU激活函数
        self.e = nn.Conv2d(h*n, self.c*n, 1, groups=n, bias=False) # 注意力层
        self.sigmoid = nn.Sigmoid()   # Sigmoid激活函数

    def forward(self, input: torch.Tensor):   
        x = self.en(input)  # 通过输入卷积层
        x_se = torch.cat((x.mean(dim=[2, 3], keepdim=True),
                          x.amax(dim=[2, 3], keepdim=True)), dim=1)  # 计算全局平均池化和全局最大池化
        x_se = self.relu(self.s(x_se))  # 通过通道压缩层和ReLU激活
        attns = self.sigmoid(self.e(x_se)).split(self.c, dim=1)  # 计算注意力权重并分割
        
        return [x * attn for attn in attns]  # 返回加权后的输出

class SEGate2d(SE2d):
    def __init__(self, 
                 c_in,  # 输入通道数 in_channels
                 c_out, # 输出通道数 out_channels
                 k_s,   # 卷积核大小 kernel_size
                 s,     # 步长 stride
                 p,     # 填充 padding
                 d,     # 空洞 dilation
                 g,     # 分组 groups
                 b,     # 偏置 bias
                 fact=4,# 通道压缩因子
                 n=1,   # 多方向的方向数
                 ):
        super().__init__(c_in, c_out, k_s, s, p, d, g, b, fact, n)
        self.gate = nn.Sigmoid()# 更新门

    def forward(self, x: torch.Tensor):
        return [self.gate(y) for y in super().forward(x)] # 返回经过更新门的输出


class GRUDirection2d(nn.Module):
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
        h = torch.ones(B, C, H, W+d, device=z.device) * h0  # 初始化输出h 
        t_num = (W+d-1)//d              # 计算时间步数
        ts = torch.arange(t_num+2)*d    # 计算时间步列表
        for ti in range(t_num):         # 循环更新h
            t = ts[ti:ti+3]             # 获取当前时间步的索引
            h_1 = h[..., t[0]:t[1]]     # H维度，获取当前时间步的h_1
            zt = z[..., t[0]:t[1]]      # H维度，获取当前时间步的z
            _ht = _h[..., t[0]:t[1]]    # H维度，获取当前时间步的_h
            h[..., t[1]:t[2]] = zt * _ht + (1 - zt) * h_1  # H维度，更新h
        h = h[..., d:]                  # 截取有效部分的h
        return self.flip(self.trans(h))  # 返回翻转后的h

    def forward_2(self, z, _h, h0):     # 定义forward_h函数
        _h = self.flip(_h)              # 根据d调整输入_h的形状
        z = self.flip(z)                # 根据d调整输入z的形状
        B, C, H, W = z.shape            # 获取输入z的形状
        dh, dw = abs(self.d[0]), abs(self.d[1])         # 计算dh和dw的绝对值
        h = torch.ones(B, C, H + dh, W + dw, device=z.device) *  h0  # 初始化输出h 
        t_num = min((H+dh-1)//dh, (W+dw-1)//dw)         # 计算时间步数
        ts = torch.arange(t_num+2)                      # 生成时间步列表
        thws = torch.stack((ts*dh, ts*dw), dim=0)       # 生成宽高时间步列表
        for ti in range(t_num):                         # 循环更新h隐状态
            t = thws[:, ti:ti+3]                        # 获取当前时间步的宽高索引
            h_1 = h[..., t[0, 0]:t[0, 1], t[1, 0]:-dw]  # H维度，获取当前时间步的h_1
            zt = z[..., t[0, 0]:t[0, 1], t[1, 0]:]      # H维度，获取当前时间步的z
            _ht = _h[..., t[0, 0]:t[0, 1], t[1, 0]:]    # H维度，获取当前时间步的_h
            h[..., t[0, 1]:t[0, 2], t[1, 1]:] = zt * _ht + (1 - zt) * h_1  # H维度，更新h
            h_1 = h[..., t[0, 1]:-dh, t[1, 0]:t[1, 1]]  # W维度，获取当前时间步的h_1
            zt = z[..., t[0, 1]:, t[1, 0]:t[1, 1]]      # W维度，获取当前时间步的z
            _ht = _h[..., t[0, 1]:, t[1, 0]:t[1, 1]]    # W维度，获取当前时间步的_h
            h[..., t[0, 2]:, t[1, 1]:t[1, 2]] = zt * _ht + (1 - zt) * h_1  # W维度，更新h
        h = h[..., dh:, dw:]    # 截取有效部分的h（去掉初始的dh行和dw列）
        return self.flip(h)     # 返回翻转后的h


class GRU2d(nn.Module):
    """ 多方向2dGRU模块 """ 
    def __init__(self,
                 c_in, # 输入通道数 in_channels
                 c_out, # 输出通道数 out_channels
                 k_s=1,  # 卷积核大小 kernel_size
                 s=1,  # 步长 stride
                 p=0,  # 填充 padding
                 d=1,  # 空洞 dilation
                 g=1,  # 分组 groups
                 b=False,  # 偏置 bias
                 directions=None,  # 自定义方向更新策略
                 direction_num=8,  # 默认的方向更新策略
                 ):
        super().__init__()
        if directions is None:
            if direction_num == 8:      # 8 方向更新
                directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
            elif direction_num == 4:    # 4 方向更新
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            else:
                raise ValueError(f"direction_num must be 4 or 8, but got {direction_num}")
        n, c = len(directions), c_out                           # 获取方向数和输出通道数
        self.h0 = nn.Parameter(torch.zeros(n, 1, c, 1, 1))      # 多方向初始隐状态
        self.nd_attn = SortAttn2d([c_out,]*n, fact=4)           # 多方向注意力机制(使用SortAttn2d快速计算)
        self._h = SE2d(c_in, c_out, k_s, s, p, d, g, b, fact=4, n=n)      # 多方向候选隐状态(使用SE2d快速计算)
        self.z = SEGate2d(c_in, c_out, k_s, s, p, d, g, b, fact=4, n=n)   # 多方向更新门(使用SEGate2d快速计算)
        self.d = nn.ModuleList([GRUDirection2d(*d) for d in directions])  # 多方向更新策略

    def forward(self, x):
        _hs = self._h(x)                           # 并行计算多方向隐状态
        zs = self.z(x)                             # 并行计算多方向更新门输出
        hs = [d(z, _h, h0) for h0, _h, z,   d in zip(self.h0, _hs, zs,  self.d)]  # 动规并行更新计算多方向隐状态
        return self.nd_attn(hs)


if __name__ == '__main__':
    from ptflops.pytorch_engine import MODULES_MAPPING
    # 自定义一个函数来计算 GRUParallelDirection2d 的 FLOPs 
    def flops_counter_hook(module, input, output, flops_per_element=2):  
        tensor_size = torch.tensor(input[0].size())
        module.__flops__ += int(torch.prod(tensor_size) * flops_per_element)  

    # 注册自定义的 FLOPs 计算函数
    MODULES_MAPPING[GRUDirection2d] = lambda module, input, output: flops_counter_hook(module, input, output, flops_per_element=2)
    MODULES_MAPPING[nn.Sigmoid] = lambda module, input, output: flops_counter_hook(module, input, output, flops_per_element=10)
    from ptflops import get_model_complexity_info
    gru = GRU2d(64, 64, g=64, direction_num=8).cuda()
    macs, params = get_model_complexity_info(gru, (64, 64, 64),
                                             as_strings=True,
                                             print_per_layer_stat=True)
    print('macs: ', macs, 'params: ', params)

    macs, params = get_model_complexity_info(nn.Conv2d(64, 64, 3, 1, 1), (64, 64, 64),
                                             as_strings=True,
                                             print_per_layer_stat=True)
    print('macs: ', macs, 'params: ', params)
