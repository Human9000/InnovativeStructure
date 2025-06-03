import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupTopk(nn.Module):
    """
    GroupTopk 模块：对输入特征图进行分组，提取每组的前 k 个最大值，
    使用分组卷积生成与输入通道数相同的特征，然后将这些特征加到原始数据上。
    """
    def __init__(self, in_channels, groups, k=1):
        """
        初始化 GroupTopk 模块
        :param in_channels: 输入通道数
        :param groups: 分组数
        :param k: 每组提取的最大值数量，默认为 1
        """
        super(GroupTopk, self).__init__()
        self.groups = groups
        self.k = k
        self.in_channels = in_channels

        # 检查输入通道数是否能被分组数整除
        assert in_channels % groups == 0, "输入通道数必须能被分组数整除"

        # 分组卷积：用于从 topk 特征生成与输入通道数相同的特征
        self.group_conv = nn.Conv2d(groups * k, in_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, input_tensor):
        """
        前向传播
        :param input_tensor: 输入特征图，形状为 [N, C, H, W]
        :return: 输出特征图，形状为 [N, C, H, W]
        """
        N, C, H, W = input_tensor.shape
        assert C == self.in_channels, "输入通道数必须与初始化时指定的通道数一致"

        # 分组并提取每组的前 k 个最大值
        group_size = C // self.groups
        input_tensor = input_tensor.view(N, self.groups, group_size, H, W)
        topk_values, _ = torch.topk(input_tensor, self.k, dim=2)  # [N, G, k, H, W]
        topk_values = topk_values.view(N, self.groups * self.k, H, W)  # [N, G * k, H, W]

        # 使用分组卷积生成与输入通道数相同的特征
        enhanced_features = self.group_conv(topk_values)  # [N, C, H, W]

        # 将生成的特征加到原始数据上
        output_tensor = input_tensor.view(N, C, H, W) + enhanced_features

        return output_tensor

# 示例使用
if __name__ == "__main__":
    # 定义输入
    N, C, H, W = 2, 8, 4, 4
    groups = 4
    k = 1
    input_tensor = torch.randn(N, C, H, W, requires_grad=True)

    # 创建 GroupTopk 模块
    group_topk_module = GroupTopk(in_channels=C, groups=groups, k=k)

    # 前向传播
    output_tensor = group_topk_module(input_tensor)
    print("输出特征图形状:", output_tensor.shape)  # 应该是 [2, 8, 4, 4]

    # 测试反向传播
    loss = output_tensor.sum()
    loss.backward()
    print("输入张量的梯度:", input_tensor.grad)