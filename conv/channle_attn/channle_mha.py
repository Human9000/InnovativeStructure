import torch
from torch import nn


class SE(nn.Sequential):
    def __init__(self, c, r=16):
        super(SE, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * super().forward(x)


class MhAConv(nn.Module):
    def __init__(self, c, k, h, s,  dropout=0.1, bias=False,):
        super(MhAConv, self).__init__()
        assert c % 4 == 0, "channel must be divisible by 4"
        self.q = nn.Conv2d(c, c*k*h, k, s, k//2, bias=False, groups=c)
        self.k = nn.Conv2d(c, c*k*h, k, s, k//2, bias=False, groups=c)
        self.v = nn.Conv2d(c, c*h, k, s, k//2, bias=False, groups=c)
        self.softmax = nn.Softmax(dim=-2)
        self.drop = nn.Dropout(dropout)

        self.inProj = nn.Conv2d(c, c, 1, stride=1, padding=0, bias=bias)
        self.pe = nn.Conv2d(c, c*h, k, s, k//2, bias=False, groups=c)
        self.outProj = nn.Sequential(SE(c*h, h*4),
                                     nn.Conv2d(c*h*2, c, 1, stride=1, padding=0, bias=bias, groups=c))
        self.h = h
        self.scale = k**-0.5

    def forward(self, x):
        x = self.inProj(x)
        pe = self.pe(x)  # [B, C, H, W]
        B, C, H, W = x.size()
        q = self.q(x).view(B, self.h, C, -1, H*W).transpose(-1, -3)  # [B, h, H*W, kernel, C]
        k = self.k(x).view(B, self.h, C, -1, H*W).transpose(-1, -3)  # [B, h, H*W, kernel, C]
        v = self.v(x).view(B, self.h, C, 1, H*W).transpose(-1, -3)  # [B, h, H*W, 1, C]
        attn = self.softmax(k.transpose(-2, -1) @ q * self.scale)  # [B, h, H*W, C, C]
        x = (v @ attn).transpose(-1, -3).view(B, self.h * C, H, W)  # [B, h*C, H, W]
        x = self.outProj(torch.cat([self.drop(x), pe], dim=1))  # [B, C, H, W]
        return x
