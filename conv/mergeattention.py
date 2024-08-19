
import torch

from torch import nn
from torch.nn import functional as F


class MergeAttention(nn.Module):
    def __init__(self, out_channel, out_size, dim, head) -> None:
        super().__init__()
        self.c = out_channel
        self.s = out_size
        self.d = dim
        self.h = head

        self.qkv = nn.Linear(dim*2, dim*3*head)
        self.wb = nn.Linear(dim*head, out_channel*2)
        self.drop = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def batchlinear(self,
                    x,  # [b,c1,...]
                    weight,  # [b,c1,c2]
                    bias=None,  # [b,1,c2]
                    dim=-1,
                    ):
        if dim != -1: # 将需要全连接的维度交换到最后
            x = x.transpose(dim, -1)

        for _ in range(x.dense_dim()-3):  # 补充特征全维度数
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)

        if bias is not None: # 是否有偏置
            y = x @ weight + bias
        else: # 是否有偏置
            y = x @ weight

        if dim != -1:  # 将需要全连接的维度交换到原来位置
            y = y.transpose(dim, -1)
        return y

    def weight_bias_attention(self, data):
        b, c = data.shape[:2]
        h, d = self.h, self.d

        scale = c**(-0.5)

        qkv = self.qkv(data).reshape(b, c, h, d, 3)

        # b h c d
        q, k, v = qkv.transpose(1, 2).unbind(-1)

        # b h c c
        attn = (q@k.transpose(-2, -1))*scale
        attn = self.drop(attn)
        attn = self.softmax(attn)

        # b h c d
        v = attn@v
        
        # b c h*d
        v = v.transpose(1, 2).reshape(b, c, -1)
 
        # [b,c1,c2] * 2
        weight, bias = self.wb(v).reshape(b, c, self.c, 2).unbind(-1)
        
        # [b,1,c2] 
        bias = bias.mean(dim=1, keepdim=True) 

        return weight, bias
 

    def attn_data(self, features):
        f1d = features.flatten(2)
        _avg = F.adaptive_avg_pool1d(f1d, self.d)
        _max = F.adaptive_max_pool1d(f1d, self.d)
        data = torch.cat((_avg, _max), dim=-1)
        return data

    def forward(self, *features):
        if len(features) == 1:  # 单输入兼容
            features = features[0]
        if isinstance(features, (list, tuple)):
            features = [F.interpolate(f, self.s) for f in features]  # 尺度统一
            features = torch.cat(features, dim=1)  # 通道合并
        else:
            features = F.interpolate(features, self.s)  # 尺度统一

        data = self.attn_data(features) # 获取降维特征

        weight, bias = self.weight_bias_attention(data) # 求解注意力权重和注意力偏置

        y = self.batchlinear(features, weight, bias, dim=1) # 批全连接叠加注意力信息

        return y, weight, bias


ma = MergeAttention(out_channel=8,
          out_size=(10, 10),
          dim=10,
          head=4)

# 三个任意尺度和通道数的特征
x1 = torch.zeros(4, 10, 4, 4)
x2 = torch.zeros(4, 5, 18, 4)
x3 = torch.zeros(4, 11, 35, 4)

y,w,b = ma(x1)
print(y.shape)
y,w,b = ma(x1, x2, x3)
print(y.shape)
