
import torch
from torch import nn
from torch.nn import functional as F


class Mean(nn.Module):
    def __init__(self, dim=(2, 3)):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class ConvMaker(nn.Module):
    def __init__(self,
                 dims,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros'):
        super(ConvMaker, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for _ in range(dims)]

        kernel_cumsum = 1
        for k in kernel_size:
            kernel_cumsum *= k

        mean_dim = [2, [2, 3], [2, 3, 4]][dims-1]
        Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]

        # 卷积核生成器，根据原始特征图生成卷积核
        self.kernel = nn.Sequential(Conv(in_channels,
                                         in_channels*kernel_cumsum,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels,
                                         bias=False,
                                         padding_mode=padding_mode,),
                                    Mean(mean_dim),
                                    nn.Unflatten(1, [in_channels,] + kernel_size),
                                    )
        # 卷积核权重生成器，根据生成的卷积核生成卷积核的输入权重
        self.weight_in = Conv(in_channels, in_channels//groups, 1)
        # 卷积核权重生成器，根据生成的卷积核生成卷积核的输出权重
        self.weight_out = Conv(in_channels, out_channels, kernel_size, groups=groups, bias=False)

        # 偏执生成器，根据生成的卷积核生成卷积核的偏执
        if bias:
            self.bias_out = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size, groups=groups, bias=False),
                nn.Flatten(1),
            )
        else:
            self.bias_out = lambda *args, **argv: None

    def forward(self, x):
        kernel = self.kernel(x)
        w_in = self.weight_in(kernel).unsqueeze(1)
        w_out = self.weight_out(kernel).unsqueeze(2)
        weight = w_in * w_out
        bias = self.bias_out(kernel)
        return weight, bias


class BaseConvPlus(nn.Module):
    def __init__(self,
                 dims,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 ):
        super(BaseConvPlus, self).__init__()
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
        self.conv = [F.conv1d, F.conv2d, F.conv3d][dims-1]

        self.maker = ConvMaker(dims,
                               in_channels,
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
        y = self.conv(
            torch.flatten(x, 0, 1)[None],  # b c l -> 1 bc l
            torch.flatten(weight, 0, 1),  # bc2 c k
            torch.flatten(bias, 0, 1),  # bc2
            groups=batch*self.groups,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        y = torch.unflatten(y[0], 0, (batch, -1))
        return y


class ConvPlus1d(BaseConvPlus):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 ):
        super(ConvPlus1d, self).__init__(1, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


class ConvPlus2d(BaseConvPlus):
    def __init__(self, in_channels, out_channels,  kernel_size,
                 stride=2,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 ):
        super(ConvPlus2d, self).__init__(2, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
class ConvPlus3d(BaseConvPlus):
    def __init__(self, in_channels, out_channels,  kernel_size,
                 stride=3,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 ):
        super(ConvPlus3d, self).__init__(3, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        


def test1d():
    print('========== test 1d ===========')
    conv1dp = ConvPlus1d(32, 64, 3, 1, 1, 1, groups=1, bias=True)
    conv1d = nn.Conv1d(32, 64, 3, 1, 1, 1, groups=1, bias=True)

    # res = ptflops.get_model_complexity_info(conv1dp, (32, 64), print_per_layer_stat=True)
    # print(res)

    # res = ptflops.get_model_complexity_info(conv1d, (32, 64), print_per_layer_stat=True)
    # print(res)

    yp = conv1dp(torch.rand(1, 32,  64))
    print(yp.shape)
    y = conv1d(torch.rand(1, 32, 64))
    # print(conv1d.weight.shape)
    print(y.shape)


def test2d():
    print('========== test 2d ===========')

    conv2dp = ConvPlus2d(32, 64, 3, 1, 1, 1, groups=1, bias=True)
    conv2d = nn.Conv2d(32, 64, 3, 1, 1, 1, groups=1, bias=True)

    # res = ptflops.get_model_complexity_info(conv2dp, (32, 64, 64), print_per_layer_stat=True)
    # print(res)

    # res = ptflops.get_model_complexity_info(conv2d, (32, 64, 64), print_per_layer_stat=True)
    # print(res)

    yp = conv2dp(torch.rand(1, 32,  64, 64))
    y = conv2d(torch.rand(1, 32, 64, 64))
    print(conv2d.weight.shape)
    print(y.shape)
    print(yp.shape)

def test3d():
    print('========== test 3d ===========')

    conv3dp = ConvPlus3d(32, 64, 3, 1, 1, 1, groups=1, bias=True)
    conv3d = nn.Conv3d(32, 64, 3, 1, 1, 1, groups=1, bias=True)

    # res = ptflops.get_model_complexity_info(conv3dp, (32, 64, 64, 64), print_per_layer_stat=True)
    # print(res)

    # res = ptflops.get_model_complexity_info(conv3d, (32, 64, 64, 64), print_per_layer_stat=True)
    # print(res)

    yp = conv3dp(torch.rand(1, 32,  64, 64, 64))
    y = conv3d(torch.rand(1, 32, 64, 64, 64))
    print(conv3d.weight.shape)
    print(y.shape)
    print(yp.shape)


if __name__ == '__main__':
    import ptflops
    test1d()
    # test2d()
    # test3d()
