import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair


class _MaskedConv2dFunction(Function):
    """
    底层的、可微分的函数，实现了内存高效的稀疏输出卷积。
    (我们将其命名为内部函数，以下划线开头)
    """

    @staticmethod
    def forward(ctx, x, weight, bias, mask, stride, padding):
        # 保存反向传播所需的变量
        ctx.save_for_backward(x, weight, mask)
        ctx.stride = stride
        ctx.padding = padding

        # 获取维度信息
        batch_size, out_channels = x.shape[0], weight.shape[0]
        out_h = (x.shape[2] + 2 * padding[0] - weight.shape[2]) // stride[0] + 1
        out_w = (x.shape[3] + 2 * padding[1] - weight.shape[3]) // stride[1] + 1

        # 1. 找出计算目标
        output_points = mask.squeeze(1).nonzero(as_tuple=False)
        if output_points.numel() == 0:
            return torch.zeros(batch_size, out_channels, out_h, out_w, device=x.device)

        # 2. 高效提取输入邻域 (Patches)
        # unfold 的结果是 (B, C_in*kH*kW, L)，L是所有可能的输出位置数
        x_unfolded = F.unfold(x, kernel_size=weight.shape[2:], padding=padding, stride=stride)

        # 将2D空间索引转换为unfold后的一维索引
        # batch_idx * L + y_idx * W_out + x_idx
        b_indices = output_points[:, 0]
        y_indices = output_points[:, 1]
        x_indices = output_points[:, 2]

        unfold_indices = b_indices * (out_h * out_w) + y_indices * out_w + x_indices

        # 从unfold结果中只 gather 我们需要计算的 patches
        patches = x_unfolded.permute(0, 2, 1).reshape(-1, x_unfolded.shape[1])[unfold_indices]

        # 3. 执行计算 (矩阵乘法)
        kernel_flat = weight.view(out_channels, -1)
        output_values = F.linear(patches, kernel_flat, bias)  # (N, C_out)

        # 4. 填充结果 (Scatter)
        final_output = torch.zeros(batch_size, out_channels, out_h, out_w, device=x.device)
        # 【核心修正点】
        # 我们知道 final_output[b_indices, :, y_indices, x_indices] 选出的是 (N, C_out) 的形状
        # 而 output_values 的形状正好也是 (N, C_out)
        # 所以我们不再需要对 output_values进行转置 (.T)
        final_output[b_indices, :, y_indices, x_indices] = output_values # 直接赋值

        return final_output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mask = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_x = grad_weight = grad_bias = None

        # 确保梯度只在掩码区域有效
        grad_output_masked = grad_output * mask

        # 计算关于输入 x 的梯度
        if ctx.needs_input_grad[0]:
            grad_x = F.conv_transpose2d(
                grad_output_masked,
                weight,
                stride=stride,
                padding=padding
            )

        # 计算关于权重 weight 的梯度
        if ctx.needs_input_grad[1]:
            # 为了使用conv2d计算梯度，我们需要调整输入的维度
            # 输入 (B, C_in, H, W) -> (C_in, B, H, W)
            # 梯度 (B, C_out, H, W) -> (B, C_out, H, W)
            # 卷积核 (权重) 应该是 (C_out, C_in, kH, kW)
            # 我们用 x 作为输入，grad_output 作为卷积核
            x_for_grad = x.permute(1, 0, 2, 3)  # (C_in, B, H, W)
            grad_output_for_grad = grad_output_masked.permute(1, 0, 2, 3)  # (C_out, B, H, W)

            grad_weight = F.conv2d(
                x_for_grad,
                grad_output_for_grad,
                stride=stride,
                padding=padding
            ).permute(1, 0, 2, 3)  # 结果是 (C_in, C_out, kH, kW), 需转置

        # 计算关于偏置 bias 的梯度
        if ctx.needs_input_grad[2] and ctx.saved_tensors[1] is not None:
            grad_bias = grad_output_masked.sum(dim=(0, 2, 3))

        return grad_x, grad_weight, grad_bias, None, None, None


class MaskConv2d(nn.Module):
    """
    一个实现了“密集输入，稀疏输出”的高效卷积层。

    该卷积层只计算并生成由输入`mask`指定的空间位置的输出，
    同时通过自定义反向传播来优化显存占用。

    其API设计与`torch.nn.Conv2d`保持一致。
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,  # 注意：当前实现暂不支持dilation>1
            groups=1,  # 注意：当前实现暂不支持groups>1
            bias: bool = True,
            padding_mode: str = 'zeros'  # 注意：当前实现只支持零填充
    ):
        super().__init__()

        if groups != 1:
            raise NotImplementedError("MaskedConv2d 目前不支持 groups > 1")
        if dilation not in (1, (1, 1)):
            raise NotImplementedError("MaskedConv2d 目前不支持 dilation > 1")
        if padding_mode != 'zeros':
            raise NotImplementedError("MaskedConv2d 目前只支持 'zeros' 填充模式")

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 使用 _pair 工具将 int 或 tuple 统一为 tuple 格式
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # 定义并初始化权重和偏置
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 Kaiming He 初始化，这是 PyTorch Conv2d 的标准做法
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        执行掩码卷积。

        Args:
            x (torch.Tensor): 输入的密集特征图，形状 (B, C_in, H, W)。
            mask (torch.Tensor): 指定输出位置的二值掩码，形状 (B, 1, H, W)。
                                 掩码的值应为0或1。

        Returns:
            torch.Tensor: 一个稀疏输出的密集张量，形状 (B, C_out, H', W')。
        """
        # 检查掩码形状是否正确
        if mask.size(1) != 1:
            raise ValueError("MaskedConv2d 的掩码必须是单通道的 (B, 1, H, W)")

        # 调用我们自定义的可微分函数
        return _MaskedConv2dFunction.apply(
            x, self.weight, self.bias, mask, self.stride, self.padding
        )

    def extra_repr(self) -> str:
        # 这个方法让 print(layer) 的输出更好看，和nn.Conv2d一样
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0, 0):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


# --- 使用示例 ---
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. 定义输入和掩码 ---
    input_tensor = torch.randn(4, 16, 64, 64, device=device, requires_grad=True)

    # 创建一个只在中心区域为1的掩码
    output_mask = torch.zeros(4, 1, 64, 64, device=device)
    output_mask[:, :, 16:48, 16:48] = 1

    # --- 2. 创建并使用 MaskedConv2d 层 ---
    # API 和 nn.Conv2d 一样
    masked_conv_layer = MaskConv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        padding=1,
        bias=True
    ).to(device)

    print("自定义的 MaskedConv2d 层:")
    print(masked_conv_layer)

    # --- 3. 执行前向传播 ---
    output_tensor = masked_conv_layer(input_tensor, output_mask)

    # --- 4. 验证和测试 ---
    # a. 验证输出形状
    print(f"\n输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")
    assert output_tensor.shape == (4, 32, 64, 64)

    # b. 验证掩码外的区域是否为0
    expanded_mask_for_check = (output_mask == 0).expand_as(output_tensor)
    values_outside = output_tensor[expanded_mask_for_check]
    is_zero_outside = torch.all(values_outside == 0)

    print(f"掩码外的区域是否全为零: {is_zero_outside.item()}")
    assert is_zero_outside

    # c. 验证反向传播是否正常工作
    try:
        # 计算一个标量损失并反向传播
        loss = output_tensor.mean()
        loss.backward()
        print("反向传播成功!")
        print(f"输入张量的梯度形状: {input_tensor.grad.shape}")
        print(f"权重的梯度形状: {masked_conv_layer.weight.grad.shape}")
    except Exception as e:
        print(f"反向传播失败: {e}")