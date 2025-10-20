import torch
from torch import nn
from torch.nn import functional as F


class ActivationSparseConnection(nn.Module):
    """
    基于输入激活值绝对值大小进行Top-K选择的稀疏连接层。
    """
    def __init__(self, c1: int, c2: int, k_in: int):
        super().__init__()
        self.c1, self.c2 = c1, c2
        self.k_in = k_in

        # 只有标准的权重和偏置
        self.weight = nn.Parameter(torch.randn(c1, c2))
        self.bias = nn.Parameter(torch.randn(c2))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: [b, l, c1]
        # 1. 打分: 直接使用激活值的绝对值
        scores = x.abs() # [b, l, c1]

        # 2. Top-K 索引选择
        _, top_indices = torch.topk(scores, k=self.k_in, dim=-1) # [b, l, k_in]

        # 3. 选择输入
        x_selected = torch.gather(x, dim=-1, index=top_indices) # [b, l, k_in]

        # 4. 选择权重
        selected_weights = F.embedding(top_indices, self.weight) # [b, l, k_in, c2]

        # 5. 稀疏计算
        output = torch.einsum('blk,blko->blo', x_selected, selected_weights) # [b, l, c2]

        # 6. 添加偏置
        return output + self.bias