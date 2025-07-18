import torch
import torch.nn as nn


class MulRes(nn.Sequential):
    """
    残差乘法模块：返回 x * f(x)，用于模拟注意力加权的乘性行为
    """

    def __init__(self, *args: nn.Module):
        super().__init__(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * super().forward(x)


class AddRes(nn.Sequential):
    """
    残差加法模块：返回 x + f(x)，常用于特征增强残差连接
    """

    def __init__(self, *args: nn.Module):
        super().__init__(*args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class TopK(nn.Module):
    """
    TopK 模块：保留张量在指定维度上的前 k 个值（不包含索引）
    """

    def __init__(self, dim: int, k: int):
        super().__init__()
        self.dim = dim
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _ = torch.topk(x, self.k, dim=self.dim)
        return values


class CGSoftmax(AddRes):
    """
    通道分组 Softmax 模块（CGSoftmax）:
    1. 将通道维度分组
    2. 每组应用 Softmax（模拟注意力权重）
    3. 注意力加权后经过两层卷积融合（1x1 conv）
    """

    def __init__(self,
                 in_channels: int,
                 groups: int = 1):
        assert in_channels % groups == 0, "输入通道数必须能被分组数整除"

        super().__init__(
            MulRes(
                nn.Unflatten(1, (groups, in_channels // groups)),  # [N, C, H, W] → [N, G, Cg, H, W]
                nn.Softmax(dim=2),  # 在每组内做 softmax（通道注意力）
                nn.Flatten(1, 2)    # [N, G, Cg, H, W] → [N, C, H, W]
            ),
            nn.Conv2d(in_channels, groups, kernel_size=1, groups=groups, bias=False),  # 每组聚合
            nn.Conv2d(groups, in_channels, kernel_size=1, bias=False)  # 融合通道
        )


class CGTopK(AddRes):
    """
    通道分组 TopK 模块（CGTopK）:
    1. 分组通道
    2. 每组提取 top-K 最大值
    3. 聚合和通道融合（1x1 conv）
    """

    def __init__(self,
                 in_channels: int,
                 groups: int = 1,
                 k: int = 1):
        assert in_channels % groups == 0, "输入通道数必须能被分组数整除"

        super().__init__(
            nn.Unflatten(1, (groups, in_channels // groups)),     # [N, C, H, W] → [N, G, Cg, H, W]
            TopK(dim=2, k=k),                                     # 每组选 Top-K 通道
            nn.Flatten(1, 2),                                     # [N, G, K, H, W] → [N, G*K, H, W]
            nn.Conv2d(groups * k, groups, kernel_size=1, groups=groups, bias=False),  # 聚合每组的K个通道
            nn.Conv2d(groups, in_channels, kernel_size=1, bias=False)  # 融合通道
        )


class SEModule(MulRes):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1, 1]
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * random_tensor / keep_prob


class CGBlock(nn.Module):
    def __init__(self, in_channle, groups, t_k=1) -> None:
        super().__init__()
        self.cg_top = CGTopK(in_channle, groups, t_k)
        self.cg_soft = CGSoftmax(in_channle, groups, )
        self.r = nn.Parameter(torch.tensor((1.0, 1.0)), requires_grad=True)

    def forward(self, x):
        rt, rs = torch.softmax(self.r, dim=0)
        return self.cgtopk(x) * rt + self.cgsoftmax(x) * rs


class CGSEBlock(nn.Module):
    def __init__(self, in_channle,
                 groups, t_k=1,
                 se_reduction: int = 8,
                 drop_path_rate: float = 0.1,
                 use_se=True,
                 use_topk=True,
                 use_soft=True
                 ) -> None:
        super().__init__()
        self.cg_top = CGTopK(in_channle, groups, t_k)
        self.cg_soft = CGSoftmax(in_channle, groups)
        self.se = SEModule(in_channle, se_reduction)
        self.r = nn.Parameter(torch.ones(3), requires_grad=True)
        self.gama = nn.Parameter(torch.ones(3)*1e-6, requires_grad=True)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        r = torch.softmax(self.r, dim=0) * self.gama
        out = r[0] * self.cg_top(x)  \
            + r[1] * self.cg_soft(x)  \
            + r[2] * self.se(x)
        return x + self.drop_path(out)


# 示例使用
if __name__ == "__main__":
    # 定义输入张量大小：1个样本，8通道，10x10空间尺寸
    N, C, H, W = 1, 8, 10, 10
    groups = 2
    input_tensor = torch.randn(N, C, H, W, requires_grad=True)

    # 选择模块：CGSoftmax 或 CGTopK
    # cgs_module = CGTopK(C, groups, k=3)
    cgs_module = CGSoftmax(C, groups)

    # 前向传播
    output_tensor = cgs_module(input_tensor)
    print("输出特征图形状:", output_tensor.shape)  # 应该是 [1, 8, 10, 10]

    # 测试反向传播
    loss = output_tensor.sum()
    loss.backward()
    print("输入张量的梯度:", input_tensor.grad)
