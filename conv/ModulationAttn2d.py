import torch
from torch import nn
import torch.nn.functional as F
  

class Transform2d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.qkv = nn.Sequential(nn.Conv2d(cin, cout*3, kernel_size=3, padding=1),
                                 nn.Flatten(2), nn.Unflatten(1, (3, cout)))
        # 残差门控和缩放
        self.res_gate = nn.Sequential(nn.Conv2d(cin+cout, cout, kernel_size=1), nn.Sigmoid())
        self.res = nn.Conv2d(cin, cout, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).unbind(dim=1)             # qkv 提取 [B, C, HW]
        q = q.transpose(-1, -2)                         # q [B, HW, C]
        v = v.transpose(-1, -2)                         # v [B, HW, C]
        attn = F.softmax((q @ k) / (C), dim=-1)         # 注意力 [B, HW, HW]
        y = (attn @ v).transpose(-1, -2)                # 注意力后 y [B, C, HW]
        y = y.unflatten(2, (H, W))                      # 注意力后 y [B, C, H, W]
        g = self.res_gate(torch.cat([y, x], dim=1))     # 残差门控 [B, C, H, W]
        return g*self.res(x) + (1-g)*y                  # 残差连接


class MAttn2dForward(nn.Module):
    def __init__(self, c, register_ptflops=True):
        super().__init__()
        self.c = c
        if register_ptflops:  # 注册ptflops
            self.register_ptflops()

    def register_ptflops(self):
        from ptflops.pytorch_ops import MODULES_MAPPING 
        def ptflops(model, input, output, extra_per_position_flops=0):
            input = input[0]
            B, C, H, W = input.shape 
            model.__flops__ += 10*B*C*H*W             # _q = F.softmax(-F.relu(q * mq), dim=-1)   
            model.__flops__ += B*C*H*W                # _k = F.relu(k * mk).transpose(-1, -2)                 
            model.__flops__ += B*C*C*H*W + 10*B*C*C   # _attn = F.softmax((_q @ _k)/(self.c**0.5), dim=-1)    
            model.__flops__ += B*C*C*H*W              # y = (_attn @ v).unflatten(2, (H, W))                  
            model.__flops__ += 2*B*C*H*W              # y = g*self.res(x) + (1-g)*y                   
        MODULES_MAPPING[self.__class__] = ptflops

    def forward(self, x: torch.Tensor,
                qkv: torch.nn.Module,
                m_qk: torch.nn.Module,
                res_gate: torch.nn.Module,
                res: torch.nn.Module):
        B, C, H, W = x.shape                                # 获取输入维度
        q, k, v = qkv(x).unbind(dim=1)                      # qkv 提取,并压缩空间维度 [B, C, HW]
        mq, mk = m_qk(v).unbind(dim=1)                      # 生成空间调制参数 [B, C, HW]
        _q = F.softmax(-F.relu(q * mq), dim=-1)             # 调制q [B, C, HW], 用softmax生成空间概率分布，用-relu防止softmax里面的exp(x)导致梯度爆炸
        _k = F.relu(k * mk).transpose(-1, -2)               # 调制k [B, C, HW], 用relu激活选择空间有效区域
        _attn = F.softmax((_q @ _k)/(self.c**0.5), dim=-1)  # 空间调制的线性通道注意力 [B, C, C]
        y = (_attn @ v).unflatten(2, (H, W))                # 叠加注意力，并还原空间维度 y [B, C, H, W]
        g = res_gate(torch.cat([y, x], dim=1))              # 残差门控 [B, C, H, W]
        y = g * res(x) + (1-g)*y                            # 残差连接
        return y, _attn


class MAttn2d(nn.Module):
    def __init__(self, cin, cout, bias=False):
        super().__init__()
        # 标准注意力的qkv提取
        self.qkv = nn.Sequential(nn.Conv2d(cin, cout*3, kernel_size=3, padding=1, bias=bias),
                                 nn.Flatten(2),
                                 nn.Unflatten(1, (3, cout)))
        # 条件性权重机制（空间调制器） Modulation
        self.m_qk = nn.Sequential(nn.Conv1d(cout, cout*2, kernel_size=3, padding=1, bias=bias),
                                 nn.Sigmoid(),
                                 nn.Unflatten(1, (2, cout)))
        # 残差门控和缩放
        self.res = nn.Conv2d(cin, cout, kernel_size=1, bias=bias)
        self.res_gate = nn.Sequential(nn.Conv2d(cin+cout, cout, kernel_size=1, bias=bias),
                                      nn.Sigmoid())
        # 注册ptflops的函数
        self._forward = MAttn2dForward(cout, register_ptflops=True)

    def forward(self, x): 
        return self._forward(x, self.qkv, self.m_qk, self.res_gate, self.res)


if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = MAttn2d(32, 32)
    y, attn = model(x)

    from ptflops import get_model_complexity_info 
    flops, params = get_model_complexity_info(model, (32, 32, 32), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    # model = Transform2d(32, 32)
    # y = model(x)
    # print(y.shape)
