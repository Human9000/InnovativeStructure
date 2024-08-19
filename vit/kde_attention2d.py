import torch.nn as nn
import torch
from einops import rearrange
from torch.nn import functional as F


class KDEAttention(nn.Module):
    def __init__(self, channel, reduction=4, gamma=0.5, group=4, window=8, kernel='gaussian', h=1.0):
        super(KDEAttention, self).__init__()
        self.channel = channel  # 输入的通道数
        self.gamma = gamma  # 残差注意力权重缩放因子
        self.group = group  # 分组数
        self.window = window  # 窗口大小
        self.kernel = kernel  # 核函数类型
        self.h = h  # 带宽
        # 确保核函数是支持的
        if self.kernel not in ['gaussian', 'laplacian', 'epanechnikov']:
            raise ValueError(
                f"Unsupported kernel: {self.kernel}. Supported options are 'gaussian', 'laplacian', 'epanechnikov'.")
        self.flatten = nn.Flatten(start_dim=3, end_dim=-1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.window)

    def forward(self, x):
        # 计算数据点之间的距离矩阵, 添加分组机制和窗口机制
        ghx = rearrange(x,
                        'b (G c) (W1 h) (W2 w) -> (b G W1 W2) c (h w)',
                        W1=self.window, W2=self.window, G=self.group)
        diff = ghx.unsqueeze(1) - ghx.unsqueeze(2)  # (bGWW, c2, c2, hw2)
        dist = torch.norm(self.flatten(diff), dim=3)  # (bGWW, c2, c2)

        # 根据核函数类型计算核矩阵
        if self.kernel == 'gaussian':
            K = torch.exp(-0.5 * ((dist / self.h) ** 2))
        elif self.kernel == 'laplacian':
            K = torch.exp(-(dist / self.h))
        elif self.kernel == 'epanechnikov':
            u = dist / self.h
            mask = (u <= 1).float()
            K = 0.75 * (1 - u ** 2) * mask

        # 计算密度估计值
        density = K.mean(dim=-1)  # 对每一行求平均，(bGWW,c2)
        density = rearrange(density, '(b G W1 W2) c -> b (G c) W1 W2', W1=self.window, W2=self.window, G=self.group)
        attn = self.fc(density) + self.gamma
        attn = F.interpolate(attn,
                             size=x.size()[2:],
                             mode='bilinear',
                             align_corners=False)
        return x * attn
class KDEAttentionV2(nn.Module):
    def __init__(self, channel, reduction=4, gamma=0.5, group=4, window=8, kernel='gaussian', h=1.0):
        super(KDEAttentionV2, self).__init__()
        self.channel = channel  # 输入的通道数
        self.gamma = gamma  # 残差注意力权重缩放因子
        self.group = group  # 分组数
        self.window = window  # 窗口大小
        self.kernel = kernel  # 核函数类型
        self.h = h  # 带宽
        # 确保核函数是支持的
        if self.kernel not in ['gaussian', 'laplacian', 'epanechnikov']:
            raise ValueError(
                f"Unsupported kernel: {self.kernel}. Supported options are 'gaussian', 'laplacian', 'epanechnikov'.")
        self.flatten = nn.Flatten(start_dim=3, end_dim=-1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=self.window)

    def forward(self, x):
        # 计算window后的特征图大小
        h, w = x.size()[-2:]
        h//=self.window
        w//=self.window

        # 计算数据点之间的距离矩阵, 添加分组机制和窗口机制
        ghx = rearrange(x,
                        'b (G c) (h W1 ) (w W2) -> (b G h w) c (W1 W2)',
                        W1=self.window, W2=self.window, G=self.group)
        diff = ghx.unsqueeze(1) - ghx.unsqueeze(2)  # (bGWW, c2, c2, hw2)
        dist = torch.norm(self.flatten(diff), dim=3)  # (bGWW, c2, c2)

        # 根据核函数类型计算核矩阵
        if self.kernel == 'gaussian':
            K = torch.exp(-0.5 * ((dist / self.h) ** 2))
        elif self.kernel == 'laplacian':
            K = torch.exp(-(dist / self.h))
        elif self.kernel == 'epanechnikov':
            u = dist / self.h
            mask = (u <= 1).float()
            K = 0.75 * (1 - u ** 2) * mask

        # 计算密度估计值
        density = K.mean(dim=-1)  # 对每一行求平均，(bGWW,c2)
        density = rearrange(density, '(b G h w) c -> b (G c) h w', h=h , w=w, G=self.group)
        attn = self.fc(density) + self.gamma
        attn = self.up_sample(attn)
        return x * attn


if __name__ == '__main__':
    batch = 32

    import time

    for size,group,window in [
       [ (batch, 32, 512, 512),    4 ,   32],
       [ (batch, 64, 256, 256),    4 ,   16],
       [ (batch, 128, 128, 128),   8 ,   16 ],
       [ (batch, 256, 64, 64),     8 ,   8 ],
       [ (batch, 512, 32, 32),     16 ,  8 ],
       [ (batch, 1024, 16, 16),    16 ,  4 ],
       [ (batch, 2048, 8, 8),      32 ,  2 ],
    ]:
        kdea = KDEAttention(size[1], group=group, window=window).cuda()
        x = torch.randn(*size).cuda()
        start = time.time()
        res1 = kdea(x)
        print("KDEAttention", time.time() - start, size, group, window)

    for size,group,window in [
        [(batch, 32, 512, 512),  4,  16],
        [(batch, 64, 256, 256),  4,  16],
        [(batch, 128, 128, 128), 8,  8 ],
        [(batch, 256, 64, 64),   8,  8 ],
        [(batch, 512, 32, 32),   16, 4 ],
        [(batch, 1024, 16, 16),  16, 4 ],
        [(batch, 2048, 8, 8),    32, 2 ],
    ]:
        kdea = KDEAttentionV2(size[1], group=group, window=window).cuda()
        x = torch.randn(*size).cuda()
        start = time.time()
        res1 = kdea(x)
        print("KDEAttentionV2", time.time() - start, size, group, window)
