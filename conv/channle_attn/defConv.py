
import torch
import torch.nn as nn 


class DefConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DefConv, self).__init__()
        # 根据当前的像素位置，搜索卷积核邻域内与当前像素最相关的特征，按照特征相关度排序，然后取前k个最相关的特征，按照相关度排序顺序进行加权求和，得到当前像素的输出

        # unfold  负责将卷积核邻域内的特征提取出来
        self.unfold_k = nn.Conv2d(in_channels, in_channels*kernel_size**2, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.unfold_r = nn.Conv2d(in_channels, in_channels*kernel_size**2, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)

        # conv  负责将相关系数编码和卷积核编码进行卷积
        self.conv = nn.Conv2d(in_channels*kernel_size * 2, out_channels, 1, groups=groups, bias=bias)
        self.groups = groups
        self.topk = in_channels*kernel_size

    def forward(self, x):
        r = self.unfold_r(x) # 相关系数
        r = r.unflatten(1, (self.groups, -1)) # 相关系数 
        r = torch.softmax(r, dim=2) # 相关系数归一化
        k = self.unfold_k(x)  # 卷积核
        k = k.unflatten(1, (self.groups, -1)) # 卷积核
        topk_r, topk_i = torch.topk(r, k=self.topk, dim=2) # 取前k个最相关的特征
        topk_k = torch.gather(k, dim=2, index = topk_i) # 根据索引取对应的卷积核
        topk_y = torch.cat([topk_r, topk_k], dim=2).flatten(1,2) # 拼接相关系数编码和卷积核编码
        y = self.conv(topk_y) # 卷积
        return y


x = torch.randn(1, 3, 32, 32)
model = DefConv(3, 1, 1)
print(model(x).shape)
