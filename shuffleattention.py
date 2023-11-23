import torch
from torch import nn
from torch.nn import functional as F


def channel_shuffle(x, groups):
    shape = x.shape
    # 获取输入特征图的shape=[b,c,*]
    batch_size, num_channels, = shape[:2]
    # 均分通道，获得每个组对应的通道数
    channels_per_group = num_channels // groups
    # 特征图shape调整 [b,c,*]==>[b,g,c_g,*]
    x = x.view(batch_size, groups, channels_per_group, *shape[2:])
    # 维度调整 [b,g,c_g,*]==>[b,c_g,g,*]；将调整后的tensor以连续值的形式保存在内存中
    x = torch.transpose(x, 1, 2).contiguous()
    # 将调整后的通道拼接回去 [b,c_g,g,*]==>[b,c,*]
    x = x.view(batch_size, -1, *shape[2:])
    # 完成通道重排
    return x


class ShuffleAttention(nn.Module):
    def __init__(self, cin, kernel_size=(5,), ratio=4):
        super().__init__()
        dims = len(kernel_size)
        assert dims in [1, 2, 3], 'ShuffleAttention only supports from 1 to 3 dimensions'
        Pool = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d][dims - 1]
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        self.interpolate_mode = ['linear', 'bilinear', 'trilinear'][dims - 1]
        self.kernel_size = kernel_size
        self.groups = 1
        for i in kernel_size:
            self.groups *= i

        self.pool1 = Pool(kernel_size)
        self.pool2 = Pool(kernel_size)

        self.conv1 = Conv(in_channels=cin * 2,
                          out_channels=cin // ratio * self.groups,
                          kernel_size=kernel_size,
                          bias=False,
                          # groups=cin // ratio
                          )
        self.conv2 = Conv(in_channels=cin // ratio * self.groups,
                          out_channels=cin * self.groups,
                          kernel_size=1,
                          bias=False,
                          groups=self.groups
                          )

    def pool(self, x):
        p1 = self.pool1(x)
        xi = F.interpolate(p1, x.shape[2:], mode=self.interpolate_mode, align_corners=False)
        # 添加方差特征
        p2 = self.pool2(torch.pow(x - xi, 2))
        y = torch.cat([p1, p2], dim=1)
        return y

    def conv(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        # 采用channel_shuffle方法
        y = channel_shuffle(y, self.groups)
        y = self.conv2(y)
        y = F.sigmoid(y)
        # batch_size, *kernel_size×channel -> batch_size, channel, *kernel_size
        print(y.shape, x.shape[0], self.kernel_size)
        y = y.reshape(x.shape[0], *self.kernel_size, -1).unsqueeze(1).transpose(1, -1).squeeze(-1)
        return y

    def forward(self, x):
        xp = self.pool(x)
        y = self.conv(xp)
        return x * F.interpolate(y, x.shape[2:], mode=self.interpolate_mode, align_corners=False)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn(1, 128, 224, 224)
    sa1 = ShuffleAttention(128, (3, 7))
    sa2 = ShuffleAttention(128, (7, 3))
    conv = torch.nn.Conv2d(128, 128, (1, 1))
    se = SE_Block(128, )
    y = sa1(x)
    print(y.shape)
    from ptflops import get_model_complexity_info

    res = get_model_complexity_info(se, (128, 128, 128))
    print(res)
    res = get_model_complexity_info(sa1, (128, 128, 128))
    print(res)
    res = get_model_complexity_info(sa2, (128, 128, 128))
    print(res)
