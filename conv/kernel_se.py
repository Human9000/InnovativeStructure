import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x): 
        x = x.unflatten(1, (self.groups, x.shape[1] // self.groups)) 
        x = x.transpose(1, 2).contiguous().flatten(1,2) 
        return x


class KernelSEBlock(nn.Module):
    def __init__(self, inchannel, kernel=3, stride=1, padding=1, ratio=16):
        super(KernelSEBlock, self).__init__()
        # 计算卷积核的通道数，等于输入通道数乘以卷积核的面积
        ck = inchannel * kernel * kernel
        self.kernel_fold = nn.Conv2d(inchannel, ck, kernel, stride, padding, groups=inchannel) # 提取每个元素的卷积核输入到通道上 
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_c = nn.Sequential(
            nn.Conv2d(ck, ck // kernel, kernel_size=1, bias=False, groups=inchannel),
            nn.ReLU(),
            nn.Conv2d(ck // kernel, ck, kernel_size=1, bias=False, groups=inchannel),
            nn.Sigmoid(),
            ChannelShuffle(inchannel),
        )
        self.fc_k = nn.Sequential(
            ChannelShuffle(inchannel),
            nn.Conv2d(ck, ck // ratio, kernel_size=1, bias=False, groups=kernel**2),
            nn.ReLU(),
            nn.Conv2d(ck // ratio, ck, kernel_size=1, bias=False, groups=kernel**2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.kernel_fold(x) # 卷积展开
        y = self.gap(x) # 全局池化 
        y = self.fc_c(y) * self.fc_k(y) # 通道注意力
        attn = y.expand_as(x) # 拓展维度
        return x * attn, attn

class KernelSEConv(nn.Module):
    def __init__(self, cin, cout, kernel=3, stride=1, padding=1, groups=1, ratio=16):
        super(KernelSEConv, self).__init__()
        self.kernel_se = KernelSEBlock(cin, kernel, stride, padding, ratio ) # 对每个卷积核输入做注意力
        self.norm_conv = nn.Conv2d(cin * kernel**2, cout, 1, 1, 0, groups=groups) # 对卷积核注意力过的数据做标准卷积

    def forward(self, x): 
        x_attn, attn = checkpoint(self.kernel_se, x, use_reentrant=False)
        x = self.norm_conv(x_attn)
        return x
    
if __name__ == '__main__':
    # --- 使用示例 ---
    model = KernelSEConv(64, 128, kernel=3).cuda()
    input_tensor = torch.randn(4, 64, 32, 32).cuda()

    # 在训练模式下，且需要计算梯度时，checkpoint 才生效
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()
    output = model(input_tensor) # 前向传播，checkpoint 包裹的模块内部激活不保存
    loss = output.mean() # 假设一个损失
    loss.backward() # 反向传播，checkpoint 包裹的模块会重新计算前向传播以获取梯度所需的激活值
    optimizer.step()

    import ptflops
    macs, params = ptflops.get_model_complexity_info(model, (64, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('macs: ', macs)
    print('params: ', params)
    macs, params = ptflops.get_model_complexity_info(nn.Conv2d(64,128, 3), (64, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('macs: ', macs)
    print('params: ', params)


    # 在评估模式下或 torch.no_grad() 中，checkpoint 不生效，行为与普通调用相同
    model.eval()
    with torch.no_grad():
        output_eval = model(input_tensor)