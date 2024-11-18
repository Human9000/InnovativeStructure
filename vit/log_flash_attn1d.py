
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from flash_attn import flash_attn_func ,flash_attn_qkvpacked_func


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


class LogFlashAttention1d(nn.Module):
    def __init__(self, in_channels, num_heads=8 ):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()  
        head_dim = in_channels // num_heads    
        scale = head_dim**-0.5

        self.static_params = [num_heads, head_dim,  scale]
        
        h = (head_dim * num_heads) * 2  # 2 for query and key


        self.bases = [2, 3 ]
        self.qks = nn.ModuleList(Conv(in_channels, h, 1, act=False)  for _ in self.bases)
        self.vs =  nn.ModuleList(Conv(in_channels, in_channels, 1, act=False) for _ in self.bases)
        self.pes = nn.ModuleList(Conv(in_channels, in_channels, 3, 1, g=in_channels, act=False)for _ in self.bases)

        self.proj = Conv(in_channels, in_channels, 1, act=False)

        self.cuda_batch_size = 2**16 - 1

    def _log_attn_func(self, x, index):   
        b,c,w = x.shape

        base = self.bases[index]
        qk = self.qks[index]
        v = self.vs[index]
        pe = self.pes[index]
        h, d,  scale = self.static_params
        # 对2取对数，求多少个长度向上取整 
        n = int(math.log(w, base) - 1e-9) + 1 
        x = F.pad(x, (  0, base**n - w), value=0.0) # pad到2的整数次幂 
        qk = qk(x).transpose(1,2)  # (batch, seqlen, nheads*headdim, )
        v0 = v(x)  # (batch, nheads*headdim, seqlen, )
        q,k = qk.view(b, *[base,]*n, h, 2, d,).unbind(-2)  # （batch, *seqlen, nheads, headdim )
        v2 = v0.transpose(1,2).view(b, *[base,]*n, h, d,)  # （batch, *seqlen,  nheads, headdim )
         
        for i in range(n):
            # 将q,k,v 转换为flash_attn的输入格式(batch, seqlen, nheads, headdim)
            # 找到对应的维度i+3 作为seqlen，其他的维度作为batch 
            q_i = q.transpose(i+1, -3).flatten(0, -4).half() # (batch * batch_len, seqlen, nheads, headdim,)
            k_i = k.transpose(i+1, -3).flatten(0, -4).half() # (batch * batch_len, seqlen, nheads, headdim,)
            v_i = v2.transpose(i+1, -3).flatten(0, -4).half() # (batch * batch_len, seqlen, nheads, headdim,) 
            # print(q_i.shape, k_i.shape, v_i.shape)
            qkv = torch.stack([q_i, k_i, v_i], dim=2)
            # print(qkv.shape)
            batch_size = qkv.shape[0]
            v2_list = []
            for j in range((batch_size + self.cuda_batch_size - 1)//self.cuda_batch_size):# 向上取整  
                v2_list.append(flash_attn_qkvpacked_func(qkv[j*self.cuda_batch_size:(j+1)*self.cuda_batch_size], 0.0)) 
                print(v2_list[-1].shape)
            v2 = torch.cat(v2_list, dim=0)
            v2 = v2.unflatten(0, [b, ]+[base,]*(n-1)).transpose(i+1, -3)  #（batch, *seqlen,  nheads, headdim )
            
        v2 = v2.reshape(b, -1, c)
        
        # 添加位置编码，并删除pad的部分
        return ( v2.transpose(1,2) + pe(v0))[..., :w]



    def forward(self, x):  
        x = x.transpose(1,2)
        # 不同的base，不同的搜索方向 
        for index  in range(len(self.bases)):
            x = x + self._log_attn_func(x, index)
        # 最后输出做通道投影
        x = self.proj(x)
        return x.transpose(1,2)


if __name__ == '__main__': 
    from ptflops import get_model_complexity_info
    ma = LogFlashAttention1d(32, 4).cuda()
    ma.train()
    res = get_model_complexity_info(ma, (1024*1024, 32), as_strings=True, print_per_layer_stat=True)
    print(res)
