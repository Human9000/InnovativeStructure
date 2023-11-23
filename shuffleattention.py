import torch
from torch import nn
from torch.nn import functional as F


class ChannelShuffle(nn.Module):
    def __init__(self, groups) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x):
        shape = x.shape 
        batch_size, num_channels, = shape[:2] 
        channels_per_group = num_channels // self.groups 
        x = x.view(batch_size, self.groups, channels_per_group, *shape[2:]) 
        x = torch.transpose(x, 1, 2).contiguous() 
        x = x.view(batch_size, -1, *shape[2:]) 
        return x


class SA_Block(nn.Module):
    def __init__(self,
                 cin,
                 blocksize=[5, 5],
                 ratio=16, 
                 ):
        super().__init__()
        dims = len(blocksize)
        assert dims in [
            1, 2, 3], 'SA_Block only supports from 1 to 3 dimensions'
        blocksize = torch.tensor(blocksize)
        Pool = [nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
                nn.AdaptiveAvgPool3d][dims - 1]
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims - 1]
        self.interpolate_mode = ['linear', 'bilinear', 'trilinear'][dims - 1]
        self.pool_size = blocksize

        self.pool1 = Pool(blocksize)
        self.pool2 = Pool(blocksize)

        groups = torch.cumprod(blocksize, 0)[-1]

        self.sa = nn.Sequential(
            Conv(in_channels=cin*2,
                 out_channels=cin // ratio * groups,
                 kernel_size=blocksize,
                 bias=False,
                 groups=cin // ratio,
                 ), nn.ReLU(),
            Conv(in_channels=cin // ratio * groups,
                 out_channels=cin // ratio * groups,
                 kernel_size=1,
                 bias=False,
                 groups=groups
                 ), nn.ReLU(), ChannelShuffle(groups),
            Conv(in_channels=cin // ratio * groups,
                 out_channels=cin * groups,
                 kernel_size=1,
                 bias=False,
                 groups=groups
                 ), nn.Sigmoid(),
        )

    def pool(self, x):
        p1 = self.pool1(x)
        xi = F.interpolate(
            p1, x.shape[2:], mode=self.interpolate_mode, align_corners=False)
        # 添加方差特征
        p2 = self.pool2(torch.pow(x - xi, 2))
        y = torch.cat([p1, p2], dim=1)
        return y

    def conv(self, x):
        # batch_size, *kernel_size×channel -> batch_size, channel, *kernel_size 
        y = self.sa(x).reshape(x.shape[0], *self.pool_size, -
                       1).unsqueeze(1).transpose(1, -1).squeeze(-1)
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
    sa1 = SA_Block(128, (3, 7))
    sa2 = SA_Block(128, (7, 3))
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
