from mingru import minGRU 
import torch
from torch import nn
from torch.nn import Module



class ViGBlock(Module):
    def __init__(self, channle, patch_size=1, dim=96, head=2, Bi=False):
        super().__init__()

        # 初始化ViGBlock类，设置输入通道数cin，输出通道数cout，维度dim，补丁大小patch_size，头数head，是否使用双向GRU BiGRU
        self.head = head  # 设置头数
        assert dim % head == 0, 'dim must be divisible by head'  # 确保维度可以被头数整除

        # 初始化行和列的正向和反向GRU
        self.r_gru_p = minGRU(dim//self.head)  # 行正向GRU
        self.r_gru_n = minGRU(dim//self.head) if Bi else None  # 行反向GRU，如果BiGRU为True则初始化，否则为None
        self.c_gru_p = minGRU(dim//self.head)  # 列正向GRU
        self.c_gru_n = minGRU(dim//self.head) if Bi else None  # 列反向GRU，如果BiGRU为True则初始化，否则为None
 
        self.r_patch_embadding = nn.Conv2d(channle, dim, kernel_size=patch_size*2 + 1, stride=patch_size, padding=patch_size)
        self.c_patch_embadding = nn.Conv2d(channle, dim, kernel_size=patch_size*2 + 1, stride=patch_size, padding=patch_size)
        self.r_unpatch_embadding = nn.ConvTranspose2d(dim, channle, kernel_size=patch_size, stride=patch_size)
        self.c_unpatch_embadding = nn.ConvTranspose2d(dim, channle, kernel_size=patch_size, stride=patch_size)
        self.r_pe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=channle) # 位置编码
        self.c_pe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=channle) # 位置编码

    def forward(self, x):
        #  行/列 patch_embadding   
        r = self.r_patch_embadding(x)
        c = self.c_patch_embadding(x)

        #  行/列 位置编码 
        r_pe = self.r_pe(r)
        c_pe = self.c_pe(c)

        #  行/列 调整形状   
        r = r.unflatten(1, (self.head, -1)).permute(0, 1, 4, 3, 2)
        c = c.unflatten(1, (self.head, -1)).permute(0, 1, 3, 4, 2) 
        rs3 = r.shape[:3]
        cs3 = c.shape[:3]
        c = c.flatten(0, 2) # (bhr, c, d//h)
        r = r.flatten(0, 2) # (bhc, r, d//h)

        #  行/列 单向/双向 
        if self.r_gru_n is None:  # 单向 
            r = self.r_gru_p(r)
            c = self.c_gru_p(c)
        else: # 双向
            r = self.r_gru_p(r) + self.r_gru_n(r.flip(dims=[-1])).flip(dims=[-1])
            c = self.c_gru_p(c) + self.c_gru_n(c.flip(dims=[-1])).flip(dims=[-1]) 

        #  行/列 调整形状
        r = r.unflatten(0, rs3).permute(0, 1, 4, 3, 2).flatten(1,2) # (b, h*d//h, r, c)
        c = c.unflatten(0, cs3).permute(0, 1, 4, 2, 3).flatten(1,2) # (b, h*d//h, r, c)

        #  行/列 融合位置编码 + unpatch_embadding
        r = self.r_unpatch_embadding(r + r_pe)  # 行 (b, c, H, W)
        c = self.c_unpatch_embadding(c + c_pe)  # 列 (b, c, H, W)
 
        y = r + c
        return y
    
    

if __name__ == '__main__': # 测试ViGBlock类
    x = torch.randn(32, 1, 256, 256).cuda()
    model = ViGBlock(channle=1,
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
