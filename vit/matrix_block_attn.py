import torch
import torch.nn as nn 


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

 
class MatrixAttentionFunc(nn.Module):
    def __init__(self,  params, layers) -> None:
        super().__init__()
        self.sp = params  # 静态参数
        self.dp = layers # 动态参数

        # 注册到ptflop的参数量计算中 
        from ptflops.pytorch_engine import MODULES_MAPPING 
        def _flops_counter_hook(module, input, output):
            b, d, h, w = module.bchw
            nh, kd = module.sp[:2] 
            module.__flops__ += b * nh * kd * h * h * w #  r = torch.softmax(torch.einsum('bHdiw,bHdjw->bHijw', rq, rk) * scale, dim=2)  # 计算行注意力权重，使用softmax归一化
            module.__flops__ += b * nh * kd * h * w * w #  c = torch.softmax(torch.einsum('bHdhi,bHdhj->bHihj', cq, ck) * scale, dim=2)  # 计算列注意力权重，使用softmax归一化
            module.__flops__ += b * d * h * w * (w+h)  #  x = torch.einsum('bHihw,bHdij,bHjhw->bHdhw', r, v, c).reshape(b, d, h, w)  # 计算输出，结合行和列的注意力
        MODULES_MAPPING[MatrixAttentionFunc] = _flops_counter_hook

    def forward(self, x):  
        nh, kd, hd, scale = self.sp  # 解包静态参数
        x = self.dp[0](x)  # 输入投影，转换输入特征
        b, d, h, w = self.bchw = x.shape # 获取输入形状，B为批量大小，d为特征维度，h和w为高和宽 
        x = self.dp[1](x)  # 行列特征提取，提取行和列的特征
        rq, rk, cq, ck, v = x.view(b, nh, kd * 4 + hd, h, w).split([kd, kd, kd, kd, hd], dim=2)  # 分割特征，得到查询、键和值
        r = torch.softmax(torch.einsum('bHdiw,bHdjw->bHijw', rq, rk) * scale, dim=2)  # 计算行注意力权重，使用softmax归一化
        c = torch.softmax(torch.einsum('bHdhi,bHdhj->bHihj', cq, ck) * scale, dim=2)  # 计算列注意力权重，使用softmax归一化

        x = torch.einsum('bHihw,bHdij,bHjhw->bHdhw', r, v, c).reshape(b, d, h, w)  # 计算输出，结合行和列的注意力
        x = x + self.dp[2](v.reshape(b, d, h, w))  # 添加位置编码，增强位置信息 
        return self.dp[3](x)  # 输出投影，生成最终输出
    
class MatrixAttention(nn.Module):  # 定义矩阵分块注意力类
    def __init__(self, dim, hidden_dim, block_size, num_heads=8, attn_ratio=0.5):  # 构造函数 
        super().__init__()  # 调用父类构造函数
        head_dim = hidden_dim // num_heads  # 计算每个头的维度
        key_dim = int(head_dim * attn_ratio)  # 计算键的维度
        scale = key_dim**-0.5  # 计算缩放因子，用于缩放注意力权重
        nh_kd = key_dim * num_heads  # 计算总键维度
        h = hidden_dim + nh_kd * 4  # 计算输出维度，包含额外的特征维度
        self.in_proj = Conv(dim, hidden_dim, block_size, block_size, g=dim, act=False)  # 输入投影卷积，大核对矩阵进行分块切分
        self.rcv = Conv(hidden_dim, h, 3, 1, act=False)  # 行列特征提取卷积，提取行和列的特征
        self.pe = Conv(hidden_dim, hidden_dim, 3, 1, g=dim, act=False)  # 位置编码卷积，用于添加位置信息
        self.out_proj = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, dim, block_size, block_size, autopad(block_size), groups=dim) , # 输出投影反卷积，大核对矩阵进行分块合并 
            Conv(dim, dim, 3, 1, act=False),# 输出投影卷积
        )

        self.func = MatrixAttentionFunc(params=[num_heads, key_dim, head_dim, scale],# 参数，存储注意力头数、键维度、头维度和缩放因子
                                        layers=[self.in_proj, self.rcv, self.pe, self.out_proj], )# 特征层，存储输入投影、行列特征提取、位置编码和输出投影
                                       
    def forward(self, x):  # 前向传播函数
        return self.func(x) 

if __name__ == '__main__': 
    ma = MatrixAttention(16, 64, 8).cuda()
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(ma, (16, 512, 512), as_strings=True, print_per_layer_stat=True)
    print('macs: ', macs, 'params: ', params)
