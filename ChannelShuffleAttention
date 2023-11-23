import torch
from torch import nn
from torch.nn import functional as F


def channel_shuffle_2d(x, groups):
    # 获取输入特征图的shape=[b,c,h,w]
    batch_size, num_channels, height, wight = x.size()
    # 均分通道，获得每个组对应的通道数
    channels_per_group = num_channels // groups
    # 特征图shape调整 [b,c,h,w]==>[b,g,c_g,h,w]
    x = x.view(batch_size, groups, channels_per_group, height, wight)
    # 维度调整 [b,g,c_g,h,w]==>[b,c_g,g,h,w]；将调整后的tensor以连续值的形式保存在内存中
    x = torch.transpose(x, 1, 2).contiguous()
    # 将调整后的通道拼接回去 [b,c_g,g,h,w]==>[b,c,h,w]
    x = x.view(batch_size, -1, height, wight)
    # 完成通道重排
    return x

def channel_shuffle_1d(x, groups):
    # 获取输入特征图的shape=[b,c,h]
    batch_size, num_channels, height = x.size()
    # 均分通道，获得每个组对应的通道数
    channels_per_group = num_channels // groups
    # 特征图shape调整 [b,c,h]==>[b,g,c_g,h]
    x = x.view(batch_size, groups, channels_per_group, height)
    # 维度调整 [b,g,c_g,h]==>[b,c_g,g,h]；将调整后的tensor以连续值的形式保存在内存中
    x = torch.transpose(x, 1, 2).contiguous()
    # 将调整后的通道拼接回去 [b,c_g,g,h]==>[b,c,h]
    x = x.view(batch_size, -1, height)
    # 完成通道重排
    return x

class ChannelAttention2d(nn.Module):
    def __init__(self, cin, kernel_size=(7, 7)):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(kernel_size)
        self.pool2 = nn.AdaptiveAvgPool2d(kernel_size)
        self.groups = 1
        for i in kernel_size:
            self.groups *= i

        self.conv1 = nn.Conv2d(in_channels=cin * 2, out_channels=cin * self.groups, kernel_size=kernel_size, groups=cin)
        self.conv2 = nn.Conv2d(in_channels=cin * self.groups, out_channels=cin * self.groups, kernel_size=1,
                               groups=self.groups)

    def pool(self, x):
        p1 = self.pool1(x)
        xi = F.interpolate(p1, x.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.pool2(torch.pow(x - xi, 2))  # 添加一个方差特征
        y = torch.cat([p1, p2])
        return y

    def conv(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = channel_shuffle_2d(y, self.groups)  # 采用channel_shuffle方法
        y = self.conv2(y)
        y = F.sigmoid(y)
        return y

    def forward(self, x):
        xp = self.pool(x)
        y = self.conv(xp)
        return x * F.interpolate(y, x.shape[2:3], mode='linear', align_corners=False)

class ChannelAttention1d(nn.Module):
    def __init__(self, cin, kernel_size=(7,)):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool1d(kernel_size)
        self.pool2 = nn.AdaptiveAvgPool1d(kernel_size)
        self.groups = 1
        for i in kernel_size:
            self.groups *= i

        self.conv1 = nn.Conv1d(in_channels=cin * 2, out_channels=cin * self.groups, kernel_size=kernel_size, groups=cin)
        self.conv2 = nn.Conv1d(in_channels=cin * self.groups, out_channels=cin * self.groups, kernel_size=1,
                               groups=self.groups)

    def pool(self, x):
        p1 = self.pool1(x)
        xi = F.interpolate(p1, x.shape[2:], mode='linear', align_corners=False)
        p2 = self.pool2(torch.pow(x - xi, 2))  # 添加一个方差特征
        y = torch.cat([p1, p2])
        return y

    def conv(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = channel_shuffle_1d(y, self.groups)  # 采用channel_shuffle方法
        y = self.conv2(y)
        y = F.sigmoid(y)
        return y

    def forward(self, x):
        xp = self.pool(x)
        y = self.conv(xp)
        return x * F.interpolate(y, x.shape[2:3], mode='linear', align_corners=False)
