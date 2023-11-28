import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        v = self.avg_pooling(x).view(*x.shape[:2])
        v = self.fc_layers(v)
        return x * v.expand_as(x)


class AssistSE(nn.Module):
    def __init__(self, main_channel, assist_channel, ratio=16, dims=1):
        super().__init__()
        # 创建一个长度为dims的列表，每个元素的值都是1，用于后续的维度扩展
        self.expand_dims = [1, ] * dims
        # 计算输入通道数，为主通道和辅助通道之和
        in_channel = main_channel + assist_channel
        # 计算中间层的通道数
        middle_channel = in_channel // ratio
        # 输出通道数与主通道数相同
        out_channel = main_channel
        # 根据dims的值选择适当的自适应平均池化层，并将其设置为1D、2D或3D
        self.aap = [nn.AdaptiveAvgPool1d,
                    nn.AdaptiveAvgPool2d,
                    nn.AdaptiveAvgPool3d][dims - 1](1)
        # 定义全连接层序列，包括两个全连接层、一个ReLU激活函数和一个Sigmoid激活函数
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channel, middle_channel, bias=False),
            nn.ReLU(),
            nn.Linear(middle_channel, out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, assist):
        # 使用自适应平均池化将x压缩为1D向量，并调整其形状以匹配x的前两个维度
        v = self.aap(x).view(*x.shape[:2])
        # 在通道维度上将压缩后的向量v和辅助信息assist拼接起来
        v = torch.cat([v, assist], dim=1)
        # 将拼接后的结果通过全连接层序列进行处理，得到每个通道的权重
        v = 2 * self.fc_layers(v)
        return x * v.view(*v.shape, *self.expand_dims)


def assist_se_1d(main_channel, assist_channel, ratio=16):
    return AssistSE(main_channel, assist_channel, ratio, dims=1)


def assist_se_2d(main_channel, assist_channel, ratio=16):
    return AssistSE(main_channel, assist_channel, ratio, dims=2)


def assist_se_3d(main_channel, assist_channel, ratio=16):
    return AssistSE(main_channel, assist_channel, ratio, dims=3)


if __name__ == '__main__':
    assist_se = assist_se_1d(1024, 12)
    x = torch.randn(1, 1024, 16)
    assist = torch.randn(1, 12)
    y = assist_se(x, assist)
    print(y)
    print(y.shape)
