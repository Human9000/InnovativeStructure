import torch
from torch import nn
from torch.nn import functional as F
 
class PReluPlus(nn.PReLU):
    def __init__(self, channels, dims=2, bias=False, lrw=1, lrb=1):
        super().__init__()
        assert dims in [
            2, 3, 4, 5], "dims not in [2, 3, 4, 5]:[FC, Conv1d, Conv2d, Conv3d]"

        self.lrw = lrw  # 用于优化训练weight 和 bias的偏执项
        self.lrb = lrb  # 用于优化训练weight 和 bias的偏执项

        self._fc = [F.linear, F.conv1d, F.conv2d, F.conv3d][dims-2]
        self._relu = self.relu_dims2 if dims == 2 else self.relu_dims345

        self.weight = nn.Parameter(torch.cat([torch.eye(channels),
                                              torch.eye(channels)], dim=1))/self.lrw
        self.fc = self.fc_bias if bias else self.fc_no_bias

        if bias:
            self.bias = nn.Parameter(torch.empty(channels))
        else:
            self.register_parameter('bias', None)

    def relu_dims2(x):
        return F.relu(torch.cat((x, -x), dim=-1), inplace=True)

    def relu_dims345(x):
        return F.relu(torch.cat((x, -x), dim=1), inplace=True)

    def fc_bias(self, x):
        return self._fc(self._relu(x), self.weight*self.lrw, self.bias*self.lrb)

    def fc_no_bias(self, x):
        return self._fc(self._relu(x), self.weight*self.lrw, None)
 
    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    x2 = torch.randn(1, 10)
    x3 = torch.randn(1, 10, 1)
    x4 = torch.randn(1, 10, 1, 1)
    x5 = torch.randn(1, 10, 1, 1, 1)

    p_relu_p2 = PReluPlus(10, dims=2, lrw=1e-1, lrb=1e-2)
    p_relu_p3 = PReluPlus(10, dims=3, lrw=1e-1, lrb=1e-2)
    p_relu_p4 = PReluPlus(10, dims=4, lrw=1e-1, lrb=1e-2)
    p_relu_p5 = PReluPlus(10, dims=5, lrw=1e-1, lrb=1e-2)

    y2 = p_relu_p2(x2)
    y3 = p_relu_p3(x3)
    y4 = p_relu_p4(x4)
    y5 = p_relu_p5(x5)

    print(p_relu_p2)
    print(p_relu_p3)
    print(p_relu_p4)
    print(p_relu_p5)
