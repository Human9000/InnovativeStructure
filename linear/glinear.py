
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
import math

class GLinear(nn.Module):
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
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

        self.weight = Parameter(torch.empty((out_features, in_features//groups), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
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

    y = glinear(x)
    print(y.shape)
