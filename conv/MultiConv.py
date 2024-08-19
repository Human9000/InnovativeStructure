import torch
import torch.nn as nn
import math


class MutilConv(nn.Module):
    def __init__(self,
                 num_heads,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 conv=nn.Conv2d,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.query_proj = conv(in_channels, 1, kernel_size, stride, padding, bias=bias)
        self.key_proj = conv(in_channels, num_heads, kernel_size, stride, padding, bias=bias)
        self.value_proj = conv(in_channels, out_channels * num_heads, kernel_size, stride, padding, bias=bias) 

    def forward(self, x):
        b = x.shape[0]
        h = self.num_heads
        c = self.out_channels
        query = self.query_proj(x).reshape(b, 1, -1)  # (b, h, *l)
        key = self.key_proj(x).reshape(b, h, -1)  # (b, 1, *l)
        value = self.value_proj(x)  # (b, h*c, *l)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.num_heads)  # (b, 1, h)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (b, 1, h)

        attn_output = torch.matmul(attn_weights, value.reshape(b, h, -1))  # (b, 1, c*l)
        attn_output = attn_output.reshape(b, c, *value.shape[2:]) 
        return attn_output, attn_weights


class MutilConv1d(MutilConv):
    def __init__(self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(num_heads, in_channels, out_channels, kernel_size, stride, padding, bias, conv=nn.Conv1d)


class MutilConv2d(MutilConv):
    def __init__(self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(num_heads, in_channels, out_channels, kernel_size, stride, padding, bias, conv=nn.Conv2d)


class MutilConv3d(MutilConv):
    def __init__(self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(num_heads, in_channels, out_channels, kernel_size, stride, padding, bias, conv=nn.Conv3d)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    net = MutilConv2d(8, 16, 16, 3)
    res = get_model_complexity_info(net, (16, 98, 98))
    print(res)
