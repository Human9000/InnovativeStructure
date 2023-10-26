
class SEAttention(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channel // reduction, in_channel * out_channel, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, w, h = x.size()
        rate = 2 / c
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, -1)
        y = self.relu(y - 0.2) * rate  # 屏蔽不明显的特征
        x = x.reshape(b, c, -1).transpose(1, -1)
        y = x @ y
        return y.transpose(1, -1).reshape(b, -1, w, h)


class MConv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.p = nn.AvgPool2d(in_channels, out_channels, kernel_size, stride, padding, )
        self.cc = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, )
        self.cp = nn.AvgPool2d(in_channels, out_channels, kernel_size, 1, padding, )
        self.pc = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, )
        self.se = SEAttention(in_channels * 4, in_channels)

    def forward(self, x):
        c = self.c(x)
        p = self.p(x)
        cc = self.cc(c)
        cp = self.cp(c)
        pc = self.pc(p)
        y = torch.cat([c, cc, cp, pc], dim=1)
        return self.se(y)
