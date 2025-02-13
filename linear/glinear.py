import torch
from torch import nn
from torch.nn import functional as F
 

class GLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 groups: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None: 
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_features % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_channels must be divisible by groups') 
        super().__init__(in_features//groups, out_features, bias, device, dtype) 
        self.in_features = in_features
        self.out_features = out_features
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
        return f'in_features={self.in_features}, out_features={self.out_features}, groups={self.groups}, bias={self.bias is not None}'


if __name__ == '__main__':

    glinear = GLinear(12, 128, groups=4, bias=False,)
    x = torch.randn(1, 50, 12)
    print(glinear)

    y = glinear(x)
    print(y.shape)
