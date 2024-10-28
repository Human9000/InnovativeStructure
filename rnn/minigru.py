import torch

from torch import nn
from torch.nn import functional as F


def _gru_direction(z, _h, h0, d=1):
    ''' 状态转移函数, 动态规划  '''
    assert d != 0, "d must be not 0"
    print(z.shape)
    batch, length, dim = z.shape
    # h = torch.zeros((batch, length + abs(d), dim))
    h = torch.empty(batch, length + abs(d), dim, device=z.device)
    if d > 0:
        start = 0
        end = length
        h[:, :d, :] = h0  # 初始化隐状态传递
    else:
        start = length - 1
        end = -1
        h[:, d:, :] = h0  # 初始化隐状态传递

    # 循环更新隐状态
    for t in range(start, end, d):
        h_1 = h[:,  t, :]
        zt = z[:,  t, :]
        _ht = _h[:,  t, :]
        h[:,  t+d, :] = zt * _ht + (1-zt) * h_1 
        
    # 去除初始化的隐状态
    return h[:, d:]


def gru_direction1(zs, _hs, h0):
    '''正向状态转移函数, 动态规划实现'''
    return _gru_direction(zs, _hs, h0, 1)


def gru_direction_1(zs, _hs, h0):
    '''反向状态转移函数, 动态规划实现'''
    return _gru_direction(zs, _hs, h0, -1)


class BiMiniGRU(nn.Module):
    def __init__(self, in_channels, hidden_size, bidirectional=True):
        super().__init__()
        self.gru_list = []  # 存储多个方向的 GRU 层

        # 初始化隐状态, 前方向
        self._h1 = nn.Linear(in_channels, hidden_size)  # 隐状态
        self.h01 = nn.Parameter(torch.zeros(1, 1, hidden_size,))  # 初始化隐状态
        self.z1 = nn.Sequential(nn.Linear(in_channels, hidden_size), nn.Sigmoid())   # 更新门
        self.s1 = nn.Sequential(nn.Linear(in_channels, hidden_size), nn.Sigmoid())   # 选择门
        self.gru_list.append([self.h01, self._h1, self.z1, self.s1, gru_direction1])  # 存到列表中

        if bidirectional:
            self._h_1 = nn.Linear(in_channels, hidden_size)  # 隐状态
            self.h0_1 = nn.Parameter(torch.zeros(1, 1, hidden_size,))  # 初始化隐状态
            self.z_1 = nn.Sequential(nn.Linear(in_channels, hidden_size), nn.Sigmoid())  # 更新门
            self.s_1 = nn.Sequential(nn.Linear(in_channels, hidden_size), nn.Sigmoid())  # 选择门
            self.gru_list.append([self.h0_1, self._h_1, self.z_1,  self.s_1,  gru_direction_1])  # 存到列表中

    def _gru_d(self, x, d_layers_direction):
        h0, _h, z, s, d = d_layers_direction
        h = d(z(x), _h(x), h0) * s(x)
        return h

    def forward(self, xs):
        hs = [self._gru_d(xs, d_layers_direction) for d_layers_direction in self.gru_list]
        return torch.sum(torch.stack(hs), dim=0)


if __name__ == '__main__':
    x = torch.randn(1,  10, 16)
    gru2 = BiMiniGRU(16, 16)
    h = gru2(x)
    print(h.shape)
    from torchsummary import summary
    summary(gru2.cuda(), (10, 16))
