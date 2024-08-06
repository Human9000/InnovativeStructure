import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class LocalAttn(nn.Module):
    def __init__(self, cin, cout, deep=0):
        super(LocalAttn, self).__init__()
        fact = 4 ** deep
        c = cin * 4 ** deep
        kernal = fact + 1 if fact > 1 else 1
        stride = fact if fact > 1 else 1
        padding = kernal // 2

        self.encode = nn.Sequential(
            # *[nn.Conv2d(c * 4 ** d, c * 4 ** (d + 1), 5, 4, 2) for d in range(deep)],  # down_sample
            nn.Conv2d(cin, c, kernal, stride, padding),  # down_sample
            nn.Conv2d(c, c, 5, 1, 2, groups=fact),  # encode
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, cout, 3, 1, 1),  # decode
            *[nn.Upsample(scale_factor=fact, mode='bilinear', align_corners=True) for _ in range(deep > 0)],  # up_sample
        ) 

    def forward(self, x):
        y = self.encode(x)
        return y


class GlobalAttn(nn.Module):
    def __init__(self, cin, cout, deep=0, ):
        super(GlobalAttn, self).__init__()
        kernal = [5, 9, ][deep]
        self.r = kernal
        self.conv = nn.Conv2d(cin, cout, 1, 1,0)
        self.a = nn.Linear(kernal ** 2, kernal ** 2)
        self.b = nn.Linear(kernal ** 2, kernal ** 2)
        self.c = nn.Linear(kernal ** 2, kernal ** 2)

    def forward(self, x):
        # padding,让h和w都是5的倍数
        x = self.conv(x)
        r = self.r
        padh = (r - x.shape[-2] % r) % r
        padw = (r - x.shape[-1] % r) % r
        x = F.pad(x, (0, padw, 0, padh), mode='constant', value=0)
        # 拆成5x5的大块，然后在大块内部进行注意力计算
        h, w = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b c (ch h) (cw w) -> b c (h w) (ch cw)', ch=r, cw=r)  # 做成了5x5的大块
        a = self.a(x)  # b c (h w) 25
        b = self.b(x).transpose(-1, -2)  # b c 25 (h w)
        c = self.c(x)  # b c (h w) 25
        y = a @ (b @ c) / r  # b c (h w) 25
        y = rearrange(y, 'b c (h w) (ch cw) -> b c (ch h) (cw w)', ch=r, h=h // r)
        return y[:, :, :h - padh, :w - padw]


class LGAttn2d(nn.Module):
    def __init__(self, cin, cout):
        super(LGAttn2d, self).__init__()
        # 局部注意力
        self.l1 = LocalAttn(cin, cout, deep=0)  # c, c*4
        self.l2 = LocalAttn(cin, cout, deep=1)  # c*4, c*16
        # 全局注意力
        self.g1 = GlobalAttn(cin, cout, deep=0)
        self.g2 = GlobalAttn(cin, cout, deep=1)
        # 注意力融合
        self.out = nn.Conv2d(cout * 4, cout, 1, 1)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(x)
        g1 = self.g1(x)
        g2 = self.g2(x)
        print(l1.shape, l2.shape, g1.shape, g2.shape)
        y = self.out(torch.cat([l1, l2, g1, g2,], dim=1))
        return y


if __name__ == "__main__":
    model = LGAttn2d(12, 13)
    from ptflops import get_model_complexity_info
    res = get_model_complexity_info(model, (12, 512, 768), as_strings=True, print_per_layer_stat=True)
    print(res)
