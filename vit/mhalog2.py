
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self,  c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        _Conv = nn.Conv1d 
        _BN =  nn.BatchNorm1d 

        self.conv = _Conv(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = _BN(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class LogAttention1d(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__() 
        # hidden_dim = in_channels
        head_dim = in_channels // num_heads 
        
        assert hidden_dim % num_heads == 0, 'hidden_dim must be divisible by num_heads'
        key_dim = hidden_dim
        # key_dim = int(head_dim * attn_ratio)
        scale = key_dim**-0.5

        self.static_params = [num_heads, head_dim, key_dim, scale]
        
        h = (key_dim * num_heads) * 2  # 2 for query and key


        self.bases = [5,4]
        self.qks = nn.ModuleList(Conv(in_channels, h, 1, act=False)  for _ in self.bases)
        self.vs =  nn.ModuleList(Conv(in_channels, in_channels, 1, act=False) for _ in self.bases)
        self.pes = nn.ModuleList(Conv(in_channels, in_channels, 3, 1, g=in_channels, act=False)for _ in self.bases)

        self.proj = Conv(in_channels, in_channels, 1, act=False)

    def _log_attn_func1(self, x, index):  
        b,c,w = x.shape
        base = self.bases[index]
        qk = self.qks[index]
        v = self.vs[index]
        pe = self.pes[index]
        nh, hd, kd, scale = self.static_params
        # 对2取对数，求多少个长度向上取整 
        n = int(math.log(w, base) - 1e-9) + 1 
        x = F.pad(x, (0, base**n - w), value=0.0) # pad到2的整数次幂 
        qk = qk(x) # 
        v2 = v1 = v0 = v(x) # 
        # 从小到大搜索
        for i in range(n-1,-1,-1):
            size = [base**i, base, base**(n-i-1)]
            print(size)
            q,k = qk.view(b, nh, 2, kd,*size).unbind(2)  
            attn = (torch.einsum('bHdijk,bHdiJk->bHJijk', q,k) * scale).softmax(dim=4) 
            v1 = torch.einsum('bHJijk,bHdijk->bHdiJk', attn, v1.reshape(b, nh,  hd,*size))
        v1 = v1.reshape(b, c, -1)

        # 从大到小搜索
        for i in range(n):
            size = [base**i, base, base**(n-i-1)]
            print(size)
            q,k = qk.view(b, nh, 2, kd,*size).unbind(2) 
            attn = (torch.einsum('bHdijk,bHdiJk->bHJijk', q,k) * scale).softmax(dim=4) 
            v2 = torch.einsum('bHJijk,bHdijk->bHdiJk', attn, v2.reshape(b, nh,  hd,*size))
        v2 = v2.reshape(b, c, -1)
        
        # 添加位置编码，并删除pad的部分
        return (v1 + v2 + pe(v0)) [..., :w]


    def _log_attn_func(self, x, index):  
        b,c,w = x.shape
        base = self.bases[index]
        qk = self.qks[index]
        v = self.vs[index]
        pe = self.pes[index]
        nh, hd, kd, scale = self.static_params
        # 对2取对数，求多少个长度向上取整 
        n = int(math.log(w, base) - 1e-9) + 1 
        x = F.pad(x, (0, base**n - w), value=0.0) # pad到2的整数次幂 
        qk = qk(x) # 
        v0 = v(x) # 
        q,k = qk.view(b, nh, 2, kd, *[base,]*n).unbind(2)  
        v2 = v1 = v0.view(b, nh,  hd, *[base,]*n)
        

        einsum_str = 'defghijklmnopqrstuvwxyz'[:n] 
        einsum_STR = 'DEFGHIJKLMNOPQRSTUVWXYZ'[:n]
        # 计算q,k,a,v 的einsum字符串
        q_strs = []
        k_strs = []
        a_strs = []
        v_strs = []
        for i in range(n):
            q_str = 'abc' + einsum_str[:n]
            k_str = 'abc' + einsum_str[:i]  + einsum_STR[i] + einsum_str[i+1:n]
            a_str = 'ab'  + einsum_STR[i] + einsum_str[:n]   
            v_str = 'abc' + einsum_str[:i] + einsum_STR[i] + einsum_str[i+1:n]
            q_strs.append(q_str)
            k_strs.append(k_str)
            a_strs.append(a_str)
            v_strs.append(v_str)

        # 提前计算每个attn
        # attns = []
        # for i in range(n):  
        #     attns.append( (torch.einsum(f'{q_strs[i]},{k_strs[i]}->{a_strs[i]}', q,k) * scale).softmax(dim=i+3) )
        #     print(f'q[{q_strs[i]}],k[{k_strs[i]}]->attn[{a_strs[i]}]')

        # # 从小到大叠加注意力
        # for i in range(n-1,-1,-1):     
        #     v1 = torch.einsum(f'{a_strs[i]},{q_strs[i]}->{v_strs[i]}', attns[i], v1 )  
        #     print(f'attn[{a_strs[i]}],v[{q_strs[i]}]->v[{v_strs[i]}]')
        # v1 = v1.reshape(b, c, -1)

        # 从大到小叠加注意力
        for i in range(n):   
            attn = (torch.einsum(f'{q_strs[i]},{k_strs[i]}->{a_strs[i]}', q,k) * scale).softmax(dim=i+3) 
            # print(f'q[{q_strs[i]}],k[{k_strs[i]}],attn[{a_strs[i]}]')
            v2 = torch.einsum(f'{a_strs[i]},{q_strs[i]}->{v_strs[i]}', attn, v2) 
            # print(f'q[{q_strs[i]}],k[{k_strs[i]}],attn[{a_strs[i]}]')
            print(f'(q[{q_strs[i]}],k[{k_strs[i]}]->attn[{a_strs[i]}]),v[{q_strs[i]}]->v[{v_strs[i]}]')
        v2 = v2.reshape(b, c, -1)
        
        # 添加位置编码，并删除pad的部分
        return ( v2 + pe(v0)) [..., :w]



    def forward(self, x):  
        # 不同的base，不同的搜索方向 
        for index  in range(len(self.bases)):
            x = x + self._log_attn_func(x, index)
        # 最后输出做通道投影
        x = self.proj(x)
        return x


if __name__ == '__main__': 
    from ptflops import get_model_complexity_info
    ma = LogAttention1d(16, 32, 4).cuda()
    ma.train()
    # with torch.no_grad():
    res = get_model_complexity_info(ma,
                                    (16, 512*512),
                                as_strings=True,
                                print_per_layer_stat=True)
    print(res)
