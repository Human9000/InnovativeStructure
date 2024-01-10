import torch.nn as nn
import torch


class KDEAttention(nn.Module):
    def __init__(self, channel, reduction=4, gamma=0.5, kernel='gaussian', h=1.0):
        super(KDEAttention, self).__init__()
        self.channel = channel  # 输入的通道数
        self.gamma = gamma  # 残差注意力权重缩放因子
        self.kernel = kernel  # 核函数类型
        self.h = h  # 带宽
        # 确保核函数是支持的
        if self.kernel not in ['gaussian', 'laplacian', 'epanechnikov']:
            raise ValueError(
                f"Unsupported kernel: {self.kernel}. Supported options are 'gaussian', 'laplacian', 'epanechnikov'.")
        self.flatten = nn.Flatten(start_dim=3, end_dim=-1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        # 计算数据点之间的距离矩阵
        diff = x.unsqueeze(1) - x.unsqueeze(2)  # (b, c, c, w, h)
        dist = torch.norm(self.flatten(diff), dim=3)  # (b, c, c)

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
        density = K.sum(dim=-1) / c  # 对每一行求和然后除以N
        attn = self.fc(density).view(b, c, 1, 1) + self.gamma
        return x * attn


class KDEAttentionV2(nn.Module):
    def __init__(self, channel, reduction=4, gamma=0.5, kernel='gaussian', h=1.0):
        super(KDEAttentionV2, self).__init__()
        self.channel = channel  # 输入的通道数
        self.gamma = gamma  # 残差注意力权重缩放因子
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

    def forward(self, x):
        b, c = x.shape[:2]
        # 计算数据点之间的距离矩阵
        diff = x.unsqueeze(1) - x.unsqueeze(2)  # (b, c, c, w, h)
        dist = torch.norm(self.flatten(diff), dim=3)  # (b, c, c)

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
        density = K.sum(dim=-1) / c  # 对每一行求和然后除以N
        attn = self.fc(x) + self.gamma
        return density.view(b, c, 1, 1) * attn


if __name__ == '__main__':
    # 将KDE作为注意力输出
    kdea = KDEAttention(32, reduction=4)
    x = torch.randn(1, 32, 128, 128)
    kdea(x)
    # 将KDE作为主干输出
    kdeaV2 = KDEAttentionV2(32, reduction=4)
    x = torch.randn(1, 32, 128, 128)
    kdeaV2(x)
