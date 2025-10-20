import torch
from torch import nn

class CGS_SE_Compact(nn.Module): 
    def __init__(self, in_channel, r=16): 
        super().__init__()
        
        cg = in_channel * 2 # 拼接后的通道数

        # 将Squeeze-Excitation的完整流程封装在一个Sequential模块中
        self.attention_generator = nn.Sequential(
            # Squeeze
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            # Excitation
            nn.Linear(cg, cg // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(cg // r, cg, bias=False),
            nn.Sigmoid(),
            # Reshape
            nn.Unflatten(1, (cg, 1, 1)),
        )
        self.softmax = nn.Softmax(dim=1)
        # 1x1卷积用于最终的特征融合与降维
        self.fusion_conv = nn.Conv2d(cg, in_channel, kernel_size=1, bias=False)
        self.in_channel = in_channel

    def forward(self, x):
        """
        模块的前向传播。
        """
        # 1. 全局排序与拼接
        # 为了代码清晰，使用新变量 x_cat 存储拼接后的结果
        x_cat = torch.cat([x, self.softmax(x)], dim=1)
 
        # 2. 生成并应用注意力
        # 一步完成注意力的生成、广播和应用
        attention_score = self.attention_generator(x_cat).expand_as(x_cat)
        x_scaled = x_cat * attention_score
        
        # 3. 融合与降维
        return self.fusion_conv(x_scaled)

# --- 使用示例 ---
if __name__ == '__main__':
    input_tensor = torch.randn(4, 64, 56, 56).cuda()
    print("输入张量形状:", input_tensor.shape)
    # 实例化紧凑版本的模块
    compact_block = CGS_SE_Compact(in_channel=64, r=16).cuda() 
    print("模块参数数量:", sum(p.numel() for p in compact_block.parameters()))
    output_tensor = compact_block(input_tensor)
    print("输出张量形状:", output_tensor.shape)
    assert input_tensor.shape == output_tensor.shape