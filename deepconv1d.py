# 给下面的代码加每行的注释
import torch
from torch import nn
from torch.nn import functional as F


def unfold1d(input, kernel_size, stride=1, padding=0):
    if padding > 0:
        x = F.pad(input, [padding, padding], mode='reflect')
    else:
        x = input
    return F.unfold(x.unsqueeze(-1), (kernel_size, 1), stride=(stride, 1)).transpose(-2, -1)

# 给下面添加行注释和变量注释
class DeepShuffle1d(nn.Module):
    def __init__(self, cin, cout, gama) -> None:
        super().__init__()
        self.gama = gama
        self.c = nn.Conv1d(cin, cout, 1)

    def forward(self, x):
        y = self.c(x)
        var, mean = torch.var_mean(y, dim=2, keepdim=True)
        return self.gama*(y-mean)/(var+1e-9)


class DeepConv1d(nn.Module):
    def __init__(self, deep_channel, feature_channel_in, feature_channel_out, kernal_size, stride=1, padding=0):
        super(DeepConv1d, self).__init__()
        self.mksp = [
            kernal_size//2,
            kernal_size,
            stride,
            padding,
        ]
        self.fc = nn.Linear(feature_channel_in*kernal_size, feature_channel_out, bias=False)
        self.shuffle = DeepShuffle1d(deep_channel, feature_channel_in, gama=0.5)
        self.th = nn.Tanh()

    def get_deep_loss(self, x, b, c, n): # 根据kernalsize 计算深度掩码，屏蔽深度差大于kernalsize的核
        m, k, s, p = self.mksp
        x_resize = F.interpolate(x, size=(n,), mode='linear') # 重采样大小
        x_shuffle = self.shuffle(x_resize) # 按照需要的通道数重洗距离系数
        x1 = unfold1d(x_shuffle, k, s, p).reshape(b, -1, c, k)  # 生成对应的张量维度
        d = (x1 - x1[..., m:m+1]).abs() # 计算深度距离
        loss = (1-self.th(d)**2).reshape(b, -1, c*k) # 获取深度距离损失 loss
        alpha = 1/loss.reshape(b, -1, c, k).sum(axis=-1) # 获取深度距离均衡系数 arpha
        return loss*alpha # 均衡的 深度距离损失

    def deep_conv(self, deep_loss, y):
        _, k, s, p = self.mksp 
        y1 = unfold1d(y, k, s, p)
        y = self.fc(deep_loss * y1).transpose(-2, -1)
        return y

    def forward(self, deep, x):  # b,c1
        deep_loss = self.get_deep_loss(deep, *x.shape)  # b, n, c, k
        print((deep_loss*100).to(torch.int))
        x = self.deep_conv(deep_loss, x) 
        return x

if __name__ == "__main__":

    batch = 2
    length = 100#5000
    deep_channel = 1
    feature_channel_in = 1
    feature_channel_out = 12
    kernal_size = 5
    stride = 1
    padding = 1

    deep = torch.randn((batch, deep_channel, length))
    x = torch.ones((batch, feature_channel_in, length)) 
    conv = DeepConv1d(deep_channel,
                      feature_channel_in, 
                      feature_channel_out,
                        kernal_size,
                          stride,
                          padding)
    y = conv(deep, x)
    print(y)
    print(x.shape, y.shape)

    from thop import profile
    from thop import clever_format
    flops, params = profile(conv, inputs=(deep, x))
    
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
