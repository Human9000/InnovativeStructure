import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class AutoSearchBase(nn.Module):
    def __init__(self, block_count: int, w0: List[float] = None, t=1):
        super().__init__()
        self.t = t
        self.count = block_count
        if w0 is not None and len(w0) == self.count:
            self.w0 = torch.tensor(w0, dtype=torch.float32)
        else:
            self.w0 = torch.ones(1, dtype=torch.float32)
        self.weights = nn.Parameter(torch.ones(self.count))
        self.mask, self.w = self._auto_search()

    def _apply(self, fn):
        self.w0 = fn(self.w0)
        self.mask = fn(self.mask)
        self.w = fn(self.w)
        return super()._apply(fn)

    def _auto_search(self):
        w1 = F.softmax(self.weights * self.w0.to(self.weights.device), dim=0)  # 引入人为倾向
        mask = w1 >= (self.t / self.count)
        return mask, w1[mask] / w1[mask].sum()

    def get_mask_w(self):
        if self.training:
            return self.auto_select()
        return self.mask, self.w


class AutoSearchBlocks(AutoSearchBase):
    def __init__(self, blocks: List[nn.Module], w0: List[float] = None, t=1):
        super().__init__(len(blocks), w0, t)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        mask, w = self.get_mask_w()
        w = w.view([1, ] * len(x.shape[:-1]) + [-1, 1])  # 与输入的特征维度对齐
        ys = [f(x) for f, flag in zip(self.blocks, mask) if flag]
        return (torch.stack(ys, dim=-1) @ w)[..., 0]


class AutoSearchFeatures(AutoSearchBase):
    def __init__(self, in_channels_list: List[int], out_channels: int, w0: List[int] = None, t=1):
        super().__init__(len(in_channels_list), w0, t)
        self.conv1x1 = nn.ModuleList([nn.Conv2d(cin, out_channels, 1) for cin in in_channels_list])

    def forward(self, features: list):
        mask, w = self.get_mask_w()
        w = w.view([1, ] * len(features[0].shape[:-1]) + [-1, 1])  # 与输入的特征维度对齐
        ys = [f(x) for f, x, flag in zip(self.conv1x1, features, mask) if flag]
        return (torch.stack(ys, dim=-1) @ w)[..., 0]


class AutoSearchChannels(AutoSearchBase):
    def __init__(self, in_channels: int, out_channels: int, groups: int, w0: List[int] = None, t=1):
        super().__init__(groups, w0, t)
        assert in_channels % out_channels == 0, "in_channels must be divisible by out_channels"
        self.conv1x1 = nn.ModuleList([nn.Conv2d(in_channels // groups, out_channels, 1) for _ in range(groups)])

    def forward(self, x):
        mask, w = self.get_mask_w()
        b, c, *size = x.shape
        xs = x.view(b, self.count, c // self.count, *size).unbind(dim=1)
        w = w.view([1, ] * len(xs[0].shape[:-1]) + [-1, 1])  # 与输入的特征维度对齐
        ys = [f(x2) for f, x2, flag in zip(self.conv1x1, xs, mask) if flag]
        return (torch.stack(ys, dim=-1) @ w)[..., 0]


if __name__ == '__main__':
    cin = 4 * 16
    cmid = 20
    cout = 4 * 16
    block = nn.Sequential(AutoSearchFeatures([cin for _ in range(5)],
                                             cmid,
                                             w0=[1, 2, 3, 4, 5]),
                          AutoSearchChannels(cmid, cmid, 4,),   # 分成4组，寻找合适的分组
                          AutoSearchBlocks([nn.Conv2d(cmid, cout, 1, padding=0, groups=1),
                                            nn.Conv2d(cmid, cout, 3, padding=1, groups=2),
                                            nn.Conv2d(cmid, cout, 3, padding=1, groups=2),
                                            nn.Conv2d(cmid, cout, 5, padding=2, groups=2),
                                            nn.Conv2d(cmid, cout, 7, padding=3, groups=4)],
                                           w0=[5, 4, 3, 2, 1])
                          ).cuda()
    xs = [
        torch.randn(1, cin, 16, 16).cuda(),
        torch.randn(1, cin, 16, 16).cuda(),
        torch.randn(1, cin, 16, 16).cuda(),
        torch.randn(1, cin, 16, 16).cuda(),
        torch.randn(1, cin, 16, 16).cuda(),
    ]
    block.train()
    block.eval()
    y = block(xs)
    print(y.shape)
