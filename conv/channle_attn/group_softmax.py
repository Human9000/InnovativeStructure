import torch
import torch.nn as nn
import torch.nn.functional as F

class CGS(nn.Module):
    """
    Channel Group Softmax (CGS): 对输入张量在通道维度上进行分组，并在每个组内应用 softmax 操作。
    然后将结果与原始值相乘，最后通过分组卷积生成与输入通道数相同的特征，并将其加到原始输入上。
    """
    def __init__(self, in_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.conv = nn.Conv2d(groups, in_channels, kernel_size=1, groups=groups, bias=False)  # 分组卷积

        # 检查输入通道数是否能被分组数整除
        assert in_channels % groups == 0, "输入通道数必须能被分组数整除"

    def forward(self, input_tensor: torch.Tensor):
        """
        前向传播
        :param input_tensor: 输入张量，形状为 [N, C, H, W]
        :return: 输出张量，形状为 [N, C, H, W]
        """
        N, C, H, W = input_tensor.shape
        G = self.groups
        d = C // G

        # 分组
        x = input_tensor.view(N, G, d, H, W)  # [N, G, d, H, W]

        # 在每个组内应用 softmax 并加权
        x = F.softmax(x, dim=2) * x  # [N, G, d, H, W]

        # 对每个组的结果取均值
        x = x.mean(dim=2)  # [N, G, H, W]

        # 通过分组卷积生成与输入通道数相同的特征
        x = self.conv(x)  # [N, C, H, W]

        # 将生成的特征加到原始输入上
        return input_tensor + x

# 示例使用
if __name__ == "__main__":
    # 定义输入
    N, C, H, W = 2, 8, 4, 4
    groups = 4
    input_tensor = torch.randn(N, C, H, W, requires_grad=True)

    # 创建 CGS 模块
    cgs_module = CGS(in_channels=C, groups=groups)

    # 前向传播
    output_tensor = cgs_module(input_tensor)
    print("输出特征图形状:", output_tensor.shape)  # 应该是 [2, 8, 4, 4]

    # 测试反向传播
    loss = output_tensor.sum()
    loss.backward()
    print("输入张量的梯度:", input_tensor.grad)