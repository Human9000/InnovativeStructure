import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class ConvAttn(nn.Module):
    def __init__(self,
                 in_channels,  # 通道数
                 out_channels,  # 通道数
                 heads=1,  # 分组数
                 patch_n=2,  # embadding patch 的数量
                 add_maxpool=False,
                 bias=True,
                 ):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.heads = heads
        self.p = patch_n
        L = patch_n * patch_n

        # 初始化三个卷积层，用于计算query、key和value
        self.q = nn.Conv2d(in_channels, in_channels*L, kernel_size=patch_n, bias=bias, groups=heads)
        self.k = nn.Conv2d(in_channels, in_channels*L, kernel_size=patch_n, bias=bias, groups=heads)

        self.v = nn.Sequential(  # 使用 shuffle conv 来减少参数量
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias, groups=heads),
            nn.ChannelShuffle(heads),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias, groups=heads),
        )

        # 使用门控机制来控制信息流动
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=bias)

        # 初始化输出卷积层
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.p
        H = self.heads
        L = p*p
        D = c // H

        # 将输入x分割成小的patch，用于后续的自注意力计算
        patch_embadding = rearrange(x, 'b c (p1 h) (p2 w) -> b c p1 p2 h w', p1=p, p2=p)  # b c p1 p2 h w

        # patch reduce 层， 减少attn的计算量

        x_p_reduce = patch_embadding.mean((3, 4))  # b c p p
        if self.add_maxpool:
            # 如果需要添加最大池化，则计算最大池化后的结果，并与平均池化后的结果进行加权平均
            max_patch_reduce = patch_embadding.amax((3, 4))
            x_p_reduce = 0.5 * x_p_reduce + 0.5 * max_patch_reduce  # b g P c

        # 计算query、key和value
        q = self.q(x_p_reduce)  # b cL 1 1
        k = self.k(x_p_reduce)  # b cL 1 1
        v = self.v(x)           # b c h w

        # 调整query、key和value的形状，用于后续的矩阵乘法
        q = rearrange(q[..., 0, 0], 'b (H D L) -> b H L D', H=H, D=D)  # b H L D
        k = rearrange(k[..., 0, 0], 'b (H D L) -> b H D L', H=H, D=D)  # b H D L
        v = rearrange(v, 'b (H D) (p1 h) (p2 w) -> b H (p1 p2) (D h w)', H=H, D=D, p1=p, p2=p)  # b H L Dhw

        # 计算patch之间的全局自注意力
        attn = (q/D@k)/L

        # 叠加注意力权重
        y = attn @ v  # b H L Dhw

        # 将自注意力计算的结果调整回原始形状
        y = rearrange(y, 'b H (p1 p2) (D h w) -> b (H D) (p1 h) (p2 w)', D=D, p1=p, h=h//p)  # b c h w

        # 信息融合
        gate = torch.sigmoid(self.gate_conv(torch.cat([x, y], dim=1)))
        y = gate * x + (1 - gate) * y

        # 最终输出，通过卷积层将自注意力计算的结果与原始输入x进行融合
        y = self.out(y)
        return y


class ConvGate(nn.Module):
    def __init__(self,
                 in_channels,  # 通道数
                 out_channels,  # 通道数
                 heads=1,  # 分组数
                 patch_n=2,  # embadding patch 的数量
                 add_maxpool=False,
                 bias=True,
                 ):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.heads = heads
        self.p = patch_n
        L = patch_n * patch_n

        self.se_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*L, kernel_size=patch_n, bias=bias, groups=heads),
            nn.ReLU(),
            nn.Conv2d(in_channels*L, heads*L*L*2, kernel_size=1, bias=bias, groups=heads),
            nn.Sigmoid(),
            nn.Flatten(1),
            nn.Unflatten(1, (heads, L, L, 2)),
        )
 
        self.v = nn.Sequential(  # 使用 shuffle conv 来减少参数量
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias, groups=heads),
            nn.ChannelShuffle(heads),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias, groups=heads), 
        )

        # 使用门控机制来控制信息流动
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=bias)

        # 初始化输出卷积层
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.p
        H = self.heads
        L = p*p
        D = c // H

        # 将输入x分割成小的patch
        patch = x.view(b, c, p, h//p, p, w//p).contiguous()  # b c p h//p p w//p

        # 计算平均池化后的结果
        x_reduce = patch.mean((3, 5))  # b c 1 1
        if self.add_maxpool:
            # 如果需要添加最大池化，则计算最大池化后的结果，并与平均池化后的结果进行加权平均
            max_reduce = patch.amax((3, 5))
            x_reduce = 0.5 * x_reduce + 0.5 * max_reduce  # b g P c

        # 计算 attn 和 value
        attn1, attn2 = self.se_block(x_reduce).unbind(-1)  # b H L L , b H L L 
        attn = (attn1 @ attn2 )/ L**1.5 
        v = rearrange(self.v(x), 'b (H D) (p1 h) (p2 w) -> b H (p1 p2) (D h w)', H=H, D=D, p1=p, p2=p)  # b H L Dhw

        # 叠加注意力权重
        y = attn @ v  # b H L Dhw

        # 将自注意力计算的结果调整回原始形状
        y = rearrange(y, 'b H (p1 p2) (D h w) -> b (H D) (p1 h) (p2 w)', D=D, p1=p, h=h//p)  # b c h w

        # 信息融合
        gate = torch.sigmoid(self.gate_conv(torch.cat([x, y], dim=1)))
        y = gate * x + (1 - gate) * y

        # 最终输出，通过卷积层将自注意力计算的结果与原始输入x进行融合
        y = self.out(y)
        return y

class ConvGate(nn.Module):
    def __init__(self,
                 in_channels,  # 通道数
                 out_channels,  # 通道数
                 patch_n=2,  # embadding patch 的数量
                 reduce_rate= 0.25, # se block 的压缩率
                 add_maxpool=False,
                 bias=True,
                 ):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.p = patch_n
        L = patch_n * patch_n
        reduce_channels = int(in_channels *L * reduce_rate)  
        patch_channels = in_channels * L 

        self.gate_patch = nn.Sequential(
            nn.Conv2d(in_channels, reduce_channels, kernel_size=patch_n, bias=bias,),
            nn.ReLU(),
            nn.Conv2d(reduce_channels, patch_channels, kernel_size=1, bias=bias,),
            nn.Sigmoid(),
        )
 
        # 使用门控机制来控制信息流动
        self.gate_channel = nn.Sequential(
            nn.Conv2d(in_channels * 2, int(in_channels*reduce_rate), kernel_size=1, bias=bias),
            nn.ReLU(),
            nn.Conv2d(int(in_channels*reduce_rate), in_channels, kernel_size=1, bias=bias), 
            nn.Sigmoid(),
        )

        # 初始化输出卷积层
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.p
        H = self.heads
        L = p*p
        D = c // H

        # 将输入x分割成小的patch
        patch = x.view(b, c, p, h//p, p, w//p).contiguous()  # b c p h//p p w//p

        # 计算平均池化后的结果
        x_reduce = patch.mean((3, 5))  # b c 1 1
        if self.add_maxpool:
            # 如果需要添加最大池化，则计算最大池化后的结果，并与平均池化后的结果进行加权平均
            max_reduce = patch.amax((3, 5))
            x_reduce = 0.5 * x_reduce + 0.5 * max_reduce  # b g P c
        

        # 计算 attn 和 value
        gate_patch  = self.gate_patch(x_reduce)   # b cL 1 1 


        gate_channel = torch.sigmoid(self.gate_channel()
        gate_patch  = self.gate_patch(x_reduce)   # b c L

        v = rearrange(self.v(x), 'b (H D) (p1 h) (p2 w) -> b H (p1 p2) (D h w)', H=H, D=D, p1=p, p2=p)  # b H L Dhw

        # 叠加注意力权重
        y = attn @ v  # b H L Dhw

        # 将自注意力计算的结果调整回原始形状
        y = rearrange(y, 'b H (p1 p2) (D h w) -> b (H D) (p1 h) (p2 w)', D=D, p1=p, h=h//p)  # b c h w

        # 信息融合
        gate = torch.sigmoid(self.gate_channel(torch.cat([x, y], dim=1)))
        y = gate * x + (1 - gate) * y

        # 最终输出，通过卷积层将自注意力计算的结果与原始输入x进行融合
        y = self.out(y)
        return y

class AtnnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(AtnnConv2d, self).__init__()
        nn.Unfold
        
    def forward(self, x):
        torch.unfold(x, dimesion)
    



if __name__ == '__main__':
    convattn = ConvAttn(40, 40, heads=8, patch_n=4, add_maxpool=True, bias=True).cuda()
    x = torch.randn(1, 40, 32, 32).cuda()
    y = convattn(x)
    print(y.shape)

    pass
