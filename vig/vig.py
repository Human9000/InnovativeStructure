from mingru import minGRU
from einops import rearrange
import torch
from torch import nn
from torch.nn import Module


class ViGBlock(Module):
    def __init__(self, channle, patch_size=1, dim=96, head=2, BiGRU=False):
        super().__init__()

        # 初始化ViGBlock类，设置输入通道数cin，输出通道数cout，维度dim，补丁大小patch_size，头数head，是否使用双向GRU BiGRU
        self.head = head  # 设置头数
        assert dim % head == 0, 'dim must be divisible by head'  # 确保维度可以被头数整除

        # 初始化行和列的正向和反向GRU
        self.r_gru_p = minGRU(dim//self.head)  # 行正向GRU
        self.r_gru_n = minGRU(dim//self.head) if BiGRU else None  # 行反向GRU，如果BiGRU为True则初始化，否则为None
        self.c_gru_p = minGRU(dim//self.head)  # 列正向GRU
        self.c_gru_n = minGRU(dim//self.head) if BiGRU else None  # 列反向GRU，如果BiGRU为True则初始化，否则为None
 
        self.r_patch_embadding = nn.Conv2d(channle, dim, kernel_size=patch_size*2 + 1, stride=patch_size, padding=patch_size)
        self.c_patch_embadding = nn.Conv2d(channle, dim, kernel_size=patch_size*2 + 1, stride=patch_size, padding=patch_size)
        self.r_unpatch_embadding = nn.ConvTranspose2d(dim, channle, kernel_size=patch_size, stride=patch_size)
        self.c_unpatch_embadding = nn.ConvTranspose2d(dim, channle, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # ======== 行列分别patch_embadding============
        r = rearrange(self.r_patch_embadding(x),  # 行卷积，将相邻的patch_size行，升维到通道上，即patch_size行的通道数变为dim，保持列数不变，行数缩小patch_size倍，后续在列数上做minGRU
                      'B (H d) r c -> B H c r d', H=self.head)  # 行 patch

        c = rearrange(self.c_patch_embadding(x),  # 列卷积，将相邻的patch_size列，升维到通道上，即patch_size列的通道数变为dim，保持行数不变，列数缩小patch_size倍，后续在行数上做minGRU
                      'B (H d) r c -> B H r c d', H=self.head)  # 列 patch 
        # ======= 行列正向GRU 和 逆向GRu =======
        r = self.r_gru_p(r)
        c = self.c_gru_p(c)

        if self.r_gru_n is not None:  # 逆向
            r = self.r_gru_p(r) + self.r_gru_n(r.flip(dims=[-1])).flip(dims=[-1])
            c = self.c_gru_p(c) + self.c_gru_n(c.flip(dims=[-1])).flip(dims=[-1])

        # ======= 行/列 unpatch_embadding=======
        r = self.r_unpatch_embadding(rearrange(r, 'B H c r d -> B (H d) r c'))  # 行反卷积
        c = self.c_unpatch_embadding(rearrange(c, 'B H r c d -> B (H d) r c'))  # 列反卷积

        # ======= 残差融合 =====================
        # y = self.residual_out(torch.cat([r, c, x], dim=1))
        y = r+c+x
        return y
