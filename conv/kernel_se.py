import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
# 已知 SE+Conv =（ X * attn(X) ) @ Wconv
# 其中 X是输入（cin, w, h），attn是注意力权重(cin)，Wconv是卷积核(cout, cin, k, k)
# 这里 SE+Conv, 因为这里矩阵乘法，消掉的维度为cin，所以attn 和谁先乘，结果是一样的
# 即：SE+Conv =（ X * attn(X) ) @ Wconv = X @ (attn(X) * Wconv)
# 所以标准的SE 等价与对后续的一个卷积核的输入通道维度做注意力
# 因此，先一步对x做kxk的 unfold的提取得到X'(cin*k*k, w, h), 然后用1x1的卷积核（cout, cin*k*k, 1, 1）做卷积， 这与直接卷积效果是一样的
# 更进一步，我们添加一个注意力权重，对Cin*k*k的通道做注意力
# 根据前面的推理，注意力权重先和卷积核或者和X'做乘法，效果是一样的
# 所以，这里我们直接对X'做注意力，然后和卷积核做卷积 等价于 先对卷积核做注意力，然后和X'做卷积
# 由于这里的卷积核是cin*k*k, 和k*k的卷积核是等价的
# 所以，这里我们直接对X'做注意力，然后和1*1卷积核做卷积 等价于 先对k*k卷积核做注意力，然后和X做卷积

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
        self.kernel_unfold = nn.Conv2d(inchannel, ck, kernel, stride, padding, groups=inchannel) # 提取每个元素的卷积核输入到通道上 
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
        x = self.kernel_unfold(x) # 卷积展开
        y = self.gap(x) # 全局池化 
        y = self.fc_c(y) * self.fc_k(y) # 通道注意力
        attn = y.expand_as(x) # 拓展维度
        return x * attn, attn

class KernelSEConv(nn.Module):
    def __init__(self, cin, cout, kernel=3, stride=1, padding=1, groups=1, ratio=16):
        super(KernelSEConv, self).__init__()
        self.kernel_se = KernelSEBlock(cin, kernel, stride, padding, ratio ) # 对每个卷积核输入做注意力
        self.norm_conv = nn.Conv2d(cin * kernel**2, cout, 1, 1, 0, groups=groups) # 对卷积核注意力过的数据做标准卷积
    
    def run_forward(self, x):
        x_attn, attn = self.kernel_se(x)
        x = self.norm_conv(x_attn)
        return x
        
    def forward(self, x): 
        # 用checkpoint降低显存占用，但会降低反向传播的计算速度,
        # 禁止中间数据的梯度和数据保存，反向传播的时候重新计算一遍。
        return checkpoint(self.run_forward, x, use_reentrant=False) 
        # return self.run_forward(x)
     
if __name__ == '__main__':
    # --- 使用示例 ---
    model = KernelSEConv(64, 128, kernel=7).cuda()
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