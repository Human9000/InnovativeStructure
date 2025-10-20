
from typing import Optional, Type, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Type, Dict, Any
from moe import MoeLayer

# 自定义专家类：一个带有隐藏层和 dropout 的 MLP
class ComplexExpert(nn.Module):
    # __init__ 必须能接收 input_dim 和 output_dim
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        print(f"  - 创建 ComplexExpert (in={input_dim}, out={output_dim}, hidden={hidden_dim})")

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# 自定义路由器类


class CustomRouter(nn.Module):
    # __init__ 必须能接收 input_dim 和 num_experts
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        print(f"  - 创建 CustomRouter (in={input_dim}, num_experts={num_experts})")

    def forward(self, x):
        return self.fc(x)


class MoeConv2d(nn.Module):
    """
    一个将深度可分离卷积与通用 MoE 层相结合的二维卷积模块。

    这个模块作为包装器，处理从特征图到 token 序列的转换，
    应用 MoE，然后再转换回特征图。
    """

    def __init__(self,
                 # --- MoeConv2d 自身参数 ---
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Optional[int] = None,  
                 # --- MoeLayer 的所有参数 ---
                 n_expert: int = 10,
                 top_k: int = 2,
                 n_shared_expert: int = 1,
                 noisy_gating: bool = True,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = True,
                 load_balancing_alpha: float = 1e-2,
                 expert_class: Optional[Type[nn.Module]] = None,
                 expert_args: Optional[Dict[str, Any]] = None,
                 router_class: Optional[Type[nn.Module]] = None,
                 router_args: Optional[Dict[str, Any]] = None):
        """
        初始化 MoeConv2d。

        Args:
            in_channels (int): 输入特征图的通道数。
            out_channels (int): 输出特征图的通道数。
            kernel_size (int): 深度可分离卷积的核大小。
            stride (int): 卷积的步长。
            padding (int, optional): 卷积的填充大小。如果为 None，将自动计算。
            **moe_kwargs: 传递给 GeneralMoeLayer 的参数，例如 n_expert, top_k 等。
        """
        super().__init__()
        assert isinstance(kernel_size, int), "kernel_size 必须是一个整数。"
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels

        # MoE 层的输入维度等于深度卷积的输出通道数
        moe_input_dim = in_channels * kernel_size

        # 1. 深度可分离卷积层-空间聚合：（分组卷积实现）
        self.conv = nn.Conv2d(
            in_channels, moe_input_dim, kernel_size, stride,
            padding, bias=False, groups=in_channels,
        )

        # 2. 深度可分离卷积层-通道交互：（通用 MoE 层实现）
        self.moe_layer = MoeLayer(
            input_dim=moe_input_dim,
            output_dim=out_channels,
            n_expert=n_expert,
            top_k=top_k,
            n_shared_expert=n_shared_expert,
            noisy_gating=noisy_gating,
            capacity_factor=capacity_factor,
            drop_tokens=drop_tokens,
            load_balancing_alpha=load_balancing_alpha,
            expert_class=expert_class,
            expert_args=expert_args,
            router_class=router_class,
            router_args=router_args
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        执行模块的前向传播。

        Args:
            x (Tensor): 输入张量，形状为 (B, C, H, W)。

        Returns:
            Tensor: 输出张量，形状为 (B, out_channels, H, W)。
        """
        B, _, H, W = x.shape

        # 步骤 1: 卷积与展平
        x_conv = self.conv(x)
        x_flat = x_conv.permute(0, 2, 3, 1).flatten(0, 2)  # (B, C', H, W) -> (B*H*W, C')

        # 步骤 2: 应用 MoE 层
        y_flat = self.moe_layer(x_flat)  # (B*H*W, C') -> (B*H*W, C_out)

        # 步骤 3: 还原形状
        y = y_flat.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B*H*W, C_out) -> (B, C_out, H, W)

        return y


# --- 使用示例 ---
if __name__ == '__main__':
    print("="*20)
    print("测试1: 使用默认参数的 MoeConv2d")
    moe_conv_default = MoeConv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        n_expert=8,  # 可以继续使用默认值，但这里显式传入以作演示
        top_k=2,
    ).cuda()
    input_tensor_1 = torch.randn(4, 64, 32, 32).cuda()
    print(moe_conv_default)
    output_1 = moe_conv_default(input_tensor_1)

    print(f"默认MoeConv2d -> 输入形状: {input_tensor_1.shape}, 输出形状: {output_1.shape}\n")

    print("="*20)
    print("测试2: 使用自定义专家和路由器的 MoeConv2d")

    # 定义自定义专家的额外参数
    custom_expert_params = {
        'hidden_dim': 256,
        'dropout_rate': 0.2
    }

    moe_conv_custom = MoeConv2d(
        # Conv 参数
        in_channels=32,
        out_channels=96,
        kernel_size=5,
        stride=2,  # 测试步长不为1的情况
        # MoE 参数
        n_expert=6,
        top_k=2,
        n_shared_expert=2,
        load_balancing_alpha=0.02,
        # 自定义类和参数
        expert_class=ComplexExpert,
        expert_args=custom_expert_params,
        router_class=CustomRouter
    ).cuda()
    moe_conv_custom.train()  # 设置为训练模式以测试辅助损失

    input_tensor_2 = torch.randn(8, 32, 64, 64).cuda()
    print(moe_conv_custom)
    output_2 = moe_conv_custom(input_tensor_2)

    # 模拟损失计算和反向传播
    loss = output_2.mean()
    loss.backward()

    print(f"自定义MoeConv2d -> 输入形状: {input_tensor_2.shape}, 输出形状: {output_2.shape}")

    # 检查梯度，确认辅助损失已注入
    router_grad = moe_conv_custom.moe_layer.router.fc.bias.grad
    print(f"路由器偏置项梯度存在: {router_grad is not None}")
    assert router_grad is not None, "梯度注入失败！"
    print(f"梯度均值 (非零表示辅助损失已生效): {router_grad.mean():.6f}")
