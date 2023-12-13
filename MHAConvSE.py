

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s1 = x.shape
        s2 = s1[:2] + [1,] * len(s1[2:])
        y = self.avg_pool(x).view(*s1[:2])
        y = self.fc(y).view(*s2)
        return x * y

class MHAConvSE(nn.Module):
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
        assert in_channels % num_heads == 0, f"in_channels {in_channels} needs to be divisible by num_heads {num_heads}."
        assert out_channels % num_heads == 0, f"out_channels {in_channels} needs to be divisible by num_heads {num_heads}."
        self.d_model = out_channels // num_heads
        self.num_heads = num_heads
        self.query_proj = conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.key_proj = conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.value_proj = conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.out_proj = SE(in_channels, out_channels)

    def forward(self, x):
        b = x.shape[0]
        h = self.num_heads
        d = self.d_model

        query = self.query_proj(x).reshape(b, h, d, -1)  # (b, h, 1, *l)
        key = self.key_proj(x).reshape(b, h, d, -1)  # (b, h, d, *l)
        value = self.value_proj(x)  # (b, c, *l)

        scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(self.num_heads)  # (b, h, 1, d)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)  # (b, h, 1, d)
        attn_output = torch.matmul(attn_weights, value.reshape(b, h, d, -1))  # (b, h, 1, d*l)
        attn_output = attn_output.reshape(*value.shape)  # (b, h*d, *l)
        attn_output = self.out_proj(attn_output) + value  # (b, h*d, *l)
        return attn_output, attn_weights


class MHAConvSE1d(MHAConvSE):
    def __init__(self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(num_heads, in_channels, out_channels, kernel_size, stride, padding, bias, conv=nn.Conv1d)


class MHAConvSE2d(MHAConvSE):
    def __init__(self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(num_heads, in_channels, out_channels, kernel_size, stride, padding, bias, conv=nn.Conv2d)


class MHAConvSE3d(MHAConvSE):
    def __init__(self, num_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__(num_heads, in_channels, out_channels, kernel_size, stride, padding, bias, conv=nn.Conv3d)


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    net = MHAConvSE2d(8, 16, 16, 3)
    res = get_model_complexity_info(net, (16, 98, 98))
    print(res)
