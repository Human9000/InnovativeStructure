import ptflops
import torch

from torch import nn

from torch.nn import functional as F

 

class Mean(nn.Module):
    def __init__(self, dim=(2, 3)):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class Conv2dMaker(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros'):
        super(Conv2dMaker, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # 卷积核生成器，根据原始特征图生成卷积核
        self.kernel = nn.Sequential(nn.Conv2d(in_channels,
                                              in_channels*kernel_size[0]*kernel_size[1],
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=False,
                                              padding_mode=padding_mode,),
                                    Mean((2, 3)),
                                    nn.Unflatten(1, (in_channels, kernel_size[0], kernel_size[1])),
                                    )
        # 卷积核权重生成器，根据生成的卷积核生成卷积核的输入权重
        self.weight_in = nn.Conv2d(in_channels, in_channels//groups, 1)
        # 卷积核权重生成器，根据生成的卷积核生成卷积核的输出权重
        self.weight_out = nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=False)

        # 偏执生成器，根据生成的卷积核生成卷积核的偏执
        if bias:
            self.bias_out = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=False),
                nn.Flatten(1),
            )
        else:
            self.bias_out = lambda *args,**argv: None

    def forward(self, x):
        kernel = self.kernel(x) 
        w_in = self.weight_in(kernel).unsqueeze(1)
        w_out = self.weight_out(kernel).unsqueeze(2)
        weight = w_in * w_out
        bias = self.bias_out(kernel)
        return weight, bias


class ConvPlus2d(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 ):
        super(ConvPlus2d, self).__init__()
        self.stride = stride
        if padding == 0:
            self.padding = 'valid'
        elif padding == kernel_size // 2:
            self.padding = 'same'
        else:
            self.padding = padding

        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.maker = Conv2dMaker(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias,
                                 )

    def forward(self, x): 
        weight, bias = self.maker(x) 
        batch = x.shape[0]
        print(x.shape, weight.shape, bias.shape,batch, self.groups)
        y = F.conv2d(
            torch.flatten(x, 0, 1)[None],  # b c w h -> 1 bc w h
            torch.flatten(weight, 0, 1),  # bc2 c 3 3
            torch.flatten(bias, 0, 1),  # bc2
            groups=batch*self.groups,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        y = torch.unflatten(y[0], 0, (batch, -1)) 
        return x


conv2dp = ConvPlus2d(32, 64, 3, 1, 1, 1, groups=1, bias=True)
conv2dm = Conv2dMaker(32, 64, 3, 1, 1, 1, groups=1, bias=True)
conv2d = nn.Conv2d(32, 64, 3, 1, 1, 1, groups=1, bias=True)


res = ptflops.get_model_complexity_info(conv2dm, (32, 64, 64), print_per_layer_stat=True)
print(res)

res = ptflops.get_model_complexity_info(conv2d, (32, 64, 64), print_per_layer_stat=True)
print(res)


yp = conv2dp(torch.rand(1, 32,  64, 64))
y = conv2d(torch.rand(1, 32, 64, 64))
print(conv2d.weight.shape)
print(y.shape)
print(yp.shape)
