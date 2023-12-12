from torch import nn
from torch.nn import functional as F


class MultiHeadConvBase(nn.Module):
    def __init__(self,
                 conv,
                 attention_heads,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None):
        super().__init__()
        self.conv = conv(in_channels,
                         out_channels * attention_heads + attention_heads,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         device,
                         dtype)
        self.head = attention_heads
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        shape = x.shape
        soft_attn = x[:, :self.head, ].reshape(shape[0], 1, self.head, *shape[2:])
        data = x[:, self.head:, ].reshape(shape[0], self.out_channels, self.head, *shape[2:])
        soft_attn = F.softmax(soft_attn, dim=2)
        y = (soft_attn * data).sum(dim=2)
        return y


class MultiHeadConv1d(MultiHeadConvBase):
    def __init__(self, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(nn.Conv1d, attention_heads, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)


class MultiHeadConv2d(MultiHeadConvBase):
    def __init__(self, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(nn.Conv2d, attention_heads, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)


class MultiHeadConv3d(MultiHeadConvBase):
    def __init__(self, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(nn.Conv3d, attention_heads, in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    MHConv2d = MultiHeadConv2d(2, 32, 64, 3)
    conv64 = nn.Conv2d(32, 64, 3)
    conv130 = nn.Conv2d(32, 130, 3)

    res = get_model_complexity_info(MHConv2d, (32, 74, 64))
    print(res, '\n')
    res = get_model_complexity_info(conv64, (32, 74, 64))
    print(res, '\n')
    res = get_model_complexity_info(conv130, (32, 74, 64))
    print(res, '\n')
