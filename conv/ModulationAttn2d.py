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
        q,k,v = self.qkv(x).unbind(dim=1)               # qkv 提取 [B, C, HW] 
        q = q.transpose(-1, -2)                         # q [B, HW, C] 
        v = v.transpose(-1, -2)                         # q [B, HW, C] 
        attn = F.softmax((q @ k) / (C), dim=-1)         # 注意力 [B, HW, HW]
        y = (attn @ v).transpose(-1, -2)                # 注意力后 y [B, C, HW]
        y = y.unflatten(2, (H, W))                      # 注意力后 y [B, C, H, W]
        g = self.res_gate(torch.cat([y, x], dim=1))     # 残差门控 [B, C, H, W]
        return g*self.res(x) + (1-g)*y                  # 残差连接


class ModulationAttn2d(nn.Module):
    def __init__(self, cin, cout, bias=False):
        super().__init__()
        # 标准注意力的qkv提取
        self.qkv = nn.Sequential(
            nn.Conv2d(cin, cout*3, kernel_size=3, padding=1, bias=bias), 
            nn.Unflatten(1, (3, cout)))  

        # 条件性权重机制（空间调制器）
        self.modulation = nn.Sequential(
            nn.Conv1d(cout, cout*2, kernel_size=3, padding=1, bias=bias),
            nn.Unflatten(1, (2, cout)))
        
        # 残差门控和缩放
        self.res = nn.Conv2d(cin, cout, kernel_size=1, bias=bias)
        self.res_gate = nn.Sequential(
            nn.Conv2d(cin+cout, cout, kernel_size=1, bias=bias),
            nn.Sigmoid())
        

    def forward(self, x):
        B, C, H, W = x.shape
        q,k,v = self.qkv(x).unbind(dim=1)                   # qkv 提取 [B, C, HW]
        qm, km = self.modulation(v).unbind(dim=1)           # 生成调制参数 [B, C, HW]
        m_q = F.softmax(-F.relu(q * qm), dim=-1)            # 调制q [B, C, HW], 用softmax生成空间概率分布 
        m_k = F.softmax(-F.relu(k * km), dim=-1).transpose(-1, -2)   # 调制k [B, C, HW], 用softmax生成空间概率分布
        m_attn = F.softmax((m_q @ m_k)/(C**0.5), dim=-1)    # 空间调制的线性通道注意力 [B, C, C], 用softmax生成通道概率分布
        y = (m_attn @ v).unflatten(2, (H, W))               # 叠加注意力 y [B, C, H, W] 
        g = self.res_gate(torch.cat([y, x], dim=1))         # 残差门控 [B, C, H, W]
        return g*self.res(x) + (1-g)*y                      # 残差连接


if __name__ == "__main__":
    x = torch.randn(1, 32, 32, 32)
    model = ModulationAttn2d(32, 32)
    y = model(x)
    print(y.shape)

    model = Transform2d(32, 32)
    y = model(x)
    print(y.shape)
