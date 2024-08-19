import torch
from torch import nn
from torch.nn import functional as F
from ptflops import get_model_complexity_info


class ChannelAttention2d(nn.Module):
    def __init__(self, c1, c2, pool, dim, head) -> None:
        super().__init__()
        self.c1 = c1 # 输入通道数
        self.c2 = c2 # 输出通道数
        self.pool = pool # 池化大小
        self.dim = dim # 

        self.qkv = nn.Linear(2*pool**2, dim*3)
        self.weight = nn.Linear(dim, c2)

        self.mha = nn.MultiheadAttention(dim, head, batch_first=True)

    def forward(self, x):  # b,c,w,h
        x1 = F.adaptive_avg_pool2d(x, self.pool).flatten(-2)  # b c pool^2
        x2 = F.adaptive_max_pool2d(x, self.pool).flatten(-2)  # b c pool^2

        x12 = torch.cat((x1, x2), dim=-1)

        q, k, v = self.qkv(x12).reshape(-1, self.c1, self.dim, 3).unbind(-1)
        res, attn = self.mha(q, k, v)
        weight = self.weight(res)
        y = x.permute(0, 2, 3, 1) @ weight.unsqueeze(1)
        return y.permute(0, 3, 1, 2), weight


# 输入的通道大小和图像尺度大小
cin1,w1,h1 = 51,55,55
cin2,w2,h2 = 71,100,100
cin3,w3,h3 = 140,200,200


# 输出的通道大小
b,c,w,h = 4, 81, 100, 100

# 其余超参数
pool = 3 # 全局池化大小
dim = 16 # 注意力机制的单词大小
hade = 4 # 注意力机制的头数

# 定义模型
ca2d = ChannelAttention2d(cin1+cin2+cin3, c, pool, dim,hade)

# 三种特征
f1 = torch.randn(b,cin1,w1,h1)
f2 = torch.randn(b,cin2,w2,h2)
f3 = torch.randn(b,cin3,w3,h3)

# 尺寸统一
f1 = F.interpolate(f1,(w,h))
f2 = F.interpolate(f2,(w,h))
f3 = F.interpolate(f3,(w,h))

# 特征通道合并
f123 = torch.cat((f1,f2,f3),dim=1)

# 特征注意力融合
f, weight = a(f123)

print('特征融合后的大小', f.shape)
print('特征融合的注意力权重', weight.shape)

get_model_complexity_info(ca2d, (cin1+cin2+cin3, w, h))
