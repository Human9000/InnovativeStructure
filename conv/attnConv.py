
import torch
from torch import nn
from torch.nn import functional as F


class AtnnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(AtnnConv2d, self).__init__()
        self.gama = 1 / (in_channels * kernel_size ** 2)  # gamma 参数，用于缩放注意力分数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.qkv = nn.Conv2d(in_channels, in_channels*3, 1, 1, 0, bias=bias)  # qkv 编码器
        self.select_q = nn.Linear(kernel_size**2, 1)  # 问题选择器
        self.residule = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias, groups=in_channels)  # 残差
        self.gate = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 1, 1, 0, bias=bias), nn.Sigmoid())  # 残差门控单元
        self.trans_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)  # 通道调整器

    def forward(self, x):
        # qkv 编码器, 将输入x编码为每个滑动窗口的qkv
        qkv = F.unfold(self.qkv(x),
                       kernel_size=self.kernel_size,
                       stride=self.stride,
                       padding=self.padding
                       ).unflatten(1, (3, self.in_channels, -1))
        # q,k,v 分离
        q, k, v = qkv.transpose(2, -1).unbind(1)    # b wh l c  #

        # 问题选择器
        q = self.select_q(q.transpose(-1, -2))      # b wh c 1     #

        # 核注意力计算
        attn = (k@q).transpose(-1, -2) * self.gama  # b wh 1 l

        # 注意力卷积
        y = (attn @ v).transpose(1, -1)             # b c wh

        # 计算残差
        x = self.residule(x)

        # 调整y的形状与x相同
        y = y.reshape(*x.shape)  # b c w h

        # 残差门控单元
        gate = self.gate(torch.cat([x, y], dim=1))

        # 调整输出通道数
        y = self.trans_channel(y*gate + x*(1-gate))

        return y

    def flpos(self,x):
        flops = 0
        with torch.no_grad():
            b,c,w,h = self.residule(x).shape
        b,c,w,h = x.shape
        # qkv 编码器
        flops += b*c*w*h*c*3

        # 问题选择器
        flops += b*c*w*h*self.kernel_size**2

        # 核注意力计算
        flops += b*w*h*c*self.kernel_size**2
 
        # 注意力卷积
        flops += b*c*w*h*self.kernel_size**2

        # 残差门控单元
        flops += b*c*w*h*c

        # 调整输出通道数
        flops += b*c*w*h*self.out_channels
        return flops
        



if __name__ == '__main__':
    block = AtnnConv2d(10, 10, 9, 1, 4).cuda()
    # x = torch.randn(1, 10, 512, 512).cuda()
    # print(x.shape)
    # y = block(x)
    # print(y.shape)
    flops = block.flpos(torch.randn(1, 10, 512, 512).cuda())
    print(flops/2**30, 'G')
    print(1*10*512*512*10*9**2/2**30, 'G')
    from ptflops  import get_model_complexity_info

    res = get_model_complexity_info(block, (10, 512, 512), as_strings=True, print_per_layer_stat=True, verbose=True)
    res = get_model_complexity_info(nn.Conv2d(10, 10, 9, 1, 4).cuda(), (10, 512, 512), as_strings=True, print_per_layer_stat=True, verbose=True)
     