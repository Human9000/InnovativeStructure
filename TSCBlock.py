# 自动池化2d
class AutoPool2d(nn.Module):
    def __init__(self, in_channels, out_sizes):
        super().__init__()
        # 自适应挑选特征从而进行池化
        self.auto_pool_w1 = nn.Parameter(torch.rand(1, out_sizes, in_channels))
        self.auto_pool_w2 = nn.Parameter(torch.rand(1, in_channels, in_channels))
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        xf = x.flatten(2)  # b t hw
        x1 = self.auto_pool_w1 @ xf  # b so hw
        x2 = self.auto_pool_w2 @ xf  # b cin hw
        x1 = self.softmax(x1)
        return x2 @ x1.transpose(-1, 1)  # b cin so


# 值域映射
class MapExp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.sigmoid(x)
        x = (x - 0.5) * 2  # b 1 t , 值域[-1,1]
        x = torch.exp(x)  # b 1 t , 值域[1/e,e]
        return x


class TSCBlock(nn.Module):  # (TSCBlock, Time Self-Convolution Block) 时间自卷积模块
    def __init__(self,
                 in_channels,
                 pool_size=8,
                 AUTOPOOL=True,  # 可控制的消融参数，是否启用自动池化
                 MAPEXP=True):  # 可控制的消融参数，是否值域映射
        super().__init__()
        channel_1 = pool_size * pool_size  # 池化后保留的宽高形成的新的通道数
        channel_2 = pool_size  # 时间注意力机制，中间保留的特征通道数

        # 池化
        self.pool = AutoPool2d(in_channels, channel_1) if AUTOPOOL else nn.Sequential(
            nn.AdaptiveAvgPool2d((1, channel_1)), nn.Flatten(2))

        # 时间自卷积
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels=channel_1, out_channels=channel_2, kernel_size=5, padding=2),  # 压缩空间特征
            nn.GELU(),
            nn.Conv1d(in_channels=channel_2, out_channels=1, kernel_size=5, padding=2),  # 再次压缩空间特征
            MapExp() if MAPEXP else nn.Sigmoid(),
        )

    def forward(self, input):  # b t h w
        x = self.pool(input)  # b t c
        x = x.transpose(1, -1)  # b c t
        w = self.time_conv(x)  # b 1 t
        w = w.transpose(1, -1).unsqueeze(-1)  # b t 1 1
        return input * w
