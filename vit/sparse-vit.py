import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from einops import repeat

# 先对k进行稀疏，我们需要的是关键词，非关键词的内容要首先过滤掉
# 然后按照k的稀疏规则
# 然后对attn进行稀疏，只注意力关注到的大部分地方，少部分地方直接跳过
# 然后对v按照attn的稀疏规则
class SparseMHA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., sparse_ratio = 0.5):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.sparse_ratio = sparse_ratio

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim = -1)

        attn = F.dropout(attn, p = self.sparse_ratio, training = self.training)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
