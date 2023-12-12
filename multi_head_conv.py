# 导入PyTorch的nn模块和functional模块  
from torch import nn  
from torch.nn import functional as F  
  
# 定义一个基础的多头卷积模块  
class MultiHeadConvBase(nn.Module):  
    def __init__(self, conv, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):  
        super().__init__()  # 初始化模块  
        # 定义一个卷积层，其输出通道数为 out_channels * attention_heads + attention_heads，这是为了存储每个头的卷积结果和注意力权重  
        self.conv = conv(in_channels, out_channels * attention_heads + attention_heads, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)  
        self.head = attention_heads  # 注意力的头数  
        self.out_channels = out_channels  # 每个头的输出通道数  
  
    def forward(self, x):  # 前向传播函数  
        x = self.conv(x)  # 通过卷积层  
        shape = x.shape  # 获取输出形状  
        # 提取注意力权重，权重形状为 [batch_size, 1, heads, *spatial_dims]  
        soft_attn = x[:, :self.head, ].reshape(shape[0], 1, self.head, *shape[2:])  
        # 提取卷积结果数据，数据形状为 [batch_size, out_channels, heads, *spatial_dims]  
        data = x[:, self.head:, ].reshape(shape[0], self.out_channels, self.head, *shape[2:])  
        soft_attn = F.softmax(soft_attn, dim=2)  # 对注意力权重进行softmax操作，使其在头维度上的和为1  
        # 使用注意力权重对卷积结果进行加权，并在头维度上进行求和，得到最终的输出  
        y = (soft_attn * data).sum(dim=2)  
        return y  # 返回输出  
  
class MultiHeadConv1d(MultiHeadConvBase):  
    def __init__(self, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):  
        super().__init__(nn.Conv1d, attention_heads, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)  
  
class MultiHeadConv2d(MultiHeadConvBase):  
    def __init__(self, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):  
        super().__init__(nn.Conv2d, attention_heads, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)  
  
class MultiHeadConv3d(MultiHeadConvBase):  
    def __init__(self, attention_heads, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):  
        super().__init__(nn.Conv3d, attention_heads, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)  
  
# 主函数部分，用于计算模型的复杂度信息  
if __name__ == '__main__':  
    from ptflops import get_model_complexity_info  # 导入计算模型复杂度的工具函数  
  
    # 创建2D多头卷积模型实例，注意力头数为2，输入通道数为32，输出通道数为64，卷积核大小为3  
    MHConv2d = MultiHeadConv2d(2, 32, 64, 3)  
    # 创建普通2D卷积模型实例，输入通道数为32，输出通道数为64，卷积核大小为3  
    conv64 = nn.Conv2d(32, 64, 3)  
    # 创建另一个普通2D卷积模型实例，输入通道数为32，输出通道数为130，卷积核大小为3（这部分似乎是为了对比不同模型之间的复杂度）  
    conv130 = nn.Conv2d(32, 130, 3)  
  
    # 使用工具函数计算并打印多头卷积模型的复杂度信息，输入尺寸为 (32, 74, 64)  
    res = get_model_complexity_info(MHConv2d, (32, 74, 64))  
    print(res, '\n')  
    # 计算并打印普通卷积模型的复杂度信息，输入尺寸为 (32, 74, 64)  
    res = get_model_complexity_info(conv64, (32, 74, 64))  
    print(res, '\n')  
    # 计算并打印另一个普通卷积模型的复杂度信息，输入尺寸为 (32, 74, 64)  
    res = get_model_complexity_info(conv130, (32, 74, 64))  
    print(res)  # 打印结果
