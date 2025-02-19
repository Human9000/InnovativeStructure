
import torch
from torch import nn


# 分组线性层
class GroupLinear(nn.Linear):
    def __init__(self,
                 i_dims: int,
                 o_dims: int,
                 groups: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if i_dims % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if o_dims % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        super().__init__(i_dims//groups, o_dims, bias, device, dtype)
        self.in_features = i_dims
        self.groups = groups

    def forward(self, input):
        g, i, o = self.groups, self.in_features, self.out_features
        i_shape_1 = list(input.shape[:-1])
        i_v = input.view(-1, g, i//g, 1)
        w_v = self.weight.view(1, g, o//g, i//g)
        if self.bias is not None:
            b_v = self.bias.view(1, g, o // g, 1)
            y = (w_v @ i_v + b_v).view(i_shape_1 + [self.out_features])
        else:
            y = (w_v @ i_v) .view(i_shape_1 + [self.out_features])
        return y

    def extra_repr(self) -> str:
        return f'i_dims={self.in_features}, o_dims={self.out_features}, groups={self.groups}, bias={self.bias is not None}'

# shuffle 线性层


class ShuffleLinear(nn.Linear):
    def __init__(self,
                 i_dims: int,
                 o_dims: int,
                 groups: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if i_dims % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if o_dims % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        h_dims = min(i_dims, o_dims)
        super().__init__(i_dims//groups, h_dims, bias, device, dtype)
        self.in_features = i_dims
        self.hid_features = h_dims
        self.out_features = o_dims
        self.groups = groups

        self.weight2 = nn.Parameter(torch.empty(
            (o_dims, h_dims//groups),
            device=device,
            dtype=dtype,
        ))

    def forward(self, input):
        g, i, h, o = (self.groups,  self.in_features,
                      self.hid_features, self.out_features)
        y_shape = list(input.shape[:-1]) + [o, ]

        # ===
        i_v = input.view(-1, g, i//g, 1)
        w_v1 = self.weight.view(1, g, h//g, i//g)
        w_v2 = self.weight2.view(1, g, o//g, h//g)

        # ===
        y_v1 = (w_v1 @ i_v)
        y_v1_shuffle = y_v1.transpose(1, 2).contiguous().view(-1, g, h//g, 1)
        y_v = w_v2 @ y_v1_shuffle

        # ===
        if self.bias is not None:
            b_v = self.bias.view(1, g, o // g, 1)
            y = (y_v + b_v).view(y_shape)
        else:
            y = y_v.view(y_shape)

        return y

    def extra_repr(self) -> str:
        return f'i_dims={self.in_features}, o_dims={self.out_features}, groups={self.groups}, bias={self.bias is not None}'


if __name__ == '__main__':
    glinear = ShuffleLinear(12, 128, groups=4, bias=False,)
    x = torch.randn(1, 50, 12)
    print(glinear)

    y = glinear(x)
    print(y.shape)
