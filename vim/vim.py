from mamba2 import Mamba2 
from torch import nn
from torch.nn import Module
import torch

class ViMBlock(Module):
    def __init__(self, channle, patch_size=1, dim=96, head=2, Bi=False):
        super().__init__() 
        # 初始化ViGBlock类，设置输入通道数cin，输出通道数cout，维度dim，补丁大小patch_size，头数head，是否使用双向 ViM
        self.head = head  # 设置头数
        assert dim % head == 0, 'dim must be divisible by head'  # 确保维度可以被头数整除

        # 初始化行和列的正向和反向ViM
        self.r_p = Mamba2(d_model=dim, d_state=dim, headdim=self.head, chunk_size=8)  # 行正向mamba2
        self.r_n = Mamba2(d_model=dim, d_state=dim, headdim=self.head, chunk_size=8)  if Bi else None  # 行反向ViM，如果Bi为True则初始化，否则为None
        self.c_p = Mamba2(d_model=dim, d_state=dim, headdim=self.head, chunk_size=8)  # 列正向mamba2
        self.c_n = Mamba2(d_model=dim, d_state=dim, headdim=self.head, chunk_size=8) if Bi else None  # 列反向ViM，如果Bi为True则初始化，否则为None
 
        self.r_patch_embadding = nn.Conv2d(channle, dim, kernel_size=patch_size*2 + 1, stride=patch_size, padding=patch_size)
        self.c_patch_embadding = nn.Conv2d(channle, dim, kernel_size=patch_size*2 + 1, stride=patch_size, padding=patch_size)
        self.r_unpatch_embadding = nn.ConvTranspose2d(dim, channle, kernel_size=patch_size, stride=patch_size)
        self.c_unpatch_embadding = nn.ConvTranspose2d(dim, channle, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # ======== 行/列 patch_embadding ============  
        r =self.r_patch_embadding(x).permute(0, 3, 2, 1)
        c =self.c_patch_embadding(x).permute(0, 2, 3, 1)
        rs3 = r.shape[:2]
        cs3 = c.shape[:2]
        c = c.flatten(0, 1) # (br, c, d)
        r = r.flatten(0, 1) # (bc, r, d)
 
        # ======= 行列正向 和 逆向=======
        r = self.r_p(r)
        c = self.c_p(c)

        if self.r_n is not None:  # 逆向 
            r = r + self.r_n(r.flip(dims=[-1])).flip(dims=[-1])
            c = c + self.c_n(c.flip(dims=[-1])).flip(dims=[-1])

        # ======= 行/列 unpatch_embadding=======
        r = r.unflatten(0, rs3).permute(0, 3, 1, 2) # (b, d, r, c)
        c = c.unflatten(0, cs3).permute(0, 3, 2, 1) # (b, d, r, c)
        r = self.r_unpatch_embadding(r)  # 行 (b, c, H, W)
        c = self.c_unpatch_embadding(c)  # 列 (b, c, H, W)

        # ======= 残差融合 ===================== 
        y = r + c + x
        return y

if __name__ == '__main__': # 测试ViMBlock类
    x = torch.randn(32, 1, 256, 256).cuda()
    model = ViMBlock(channle=1,
                    dim=96,
                    head=4,
                    patch_size=8,
                    Bi=True,
                    ).cuda()
 
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.eval()
    with torch.no_grad():
        for i in range(100):
            out = model(x)
            opt.zero_grad()
            opt.step()
            print("Output shape:", out.shape)
