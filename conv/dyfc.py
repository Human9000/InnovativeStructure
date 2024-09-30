import torch
from torch import nn 


class DyFC(nn.Module):
    def __init__(self,
                 cin, # 输入通道数
                 cout, # 输出通道数
                 k, # 基向量个数
                 channel_dim=-1 # 通道维度对应的维度索引
                 ) -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.channel_dim = channel_dim

        self.attention = nn.Sequential( # 获取基向量注意力
            nn.Linear(cin*2, cin*k),
            nn.ReLU(),
            nn.Linear(cin*k, k),
            nn.Softmax(1),
        )

        # 定义 K 个基向量权重和基向量偏置
        self.weight = nn.Parameter(torch.ones((1, cin*cout, k), requires_grad=True))
        self.bias = nn.Parameter(torch.zeros((1, cout, k), requires_grad=True))

    def forward(self, x:torch.Tensor):
        batch_size = x.shape[0] # 获取batch_size信息
        x1 = x.transpose(-1, self.channel_dim) # 通道维度调整至最后 
        with torch.no_grad():
            # 拉直除batch_size 和 channel 的维度
            x2 = x1.flatten(1,-2)
            # 获取channel特征均值
            x2_mean = torch.mean(x2, dim=1)
            # 获取channel特征方差 
            x2_std = torch.std(x2, dim=1)
            # 构建服从输入特征的正泰分布的特征分布数
            xlr = torch.cat((x2_mean - x2_std, x2_mean + x2_std), dim=-1) 

        attn = self.attention(xlr) # 根据原特征正泰分布数计算基向量注意力
        weights = self.weight @ attn.unsqueeze(-1) # 生成注意力合成向量的权重
        bias = self.bias @ attn.unsqueeze(-1) # 生成注意力合成向量的偏置
        weights = weights.view(batch_size, self.cin, self.cout)  # 调整权重形状进行全连接操作
        bias = bias.view(batch_size, 1, self.cout) # 调整偏置形状进行全连接操作
        y = x2 @ weights + bias # 全连接
        
        # 调整输出形状与原始形状相同
        y = y.view(*x1.shape[:-1], self.cout).transpose(-1, self.channel_dim)

        return y


dyfc = DyFC(11, 7, k=3, channel_dim=2)
x = torch.randn((3, 12, 11, 10))
print(x.shape)
y = dyfc(x)
print(y.shape)
