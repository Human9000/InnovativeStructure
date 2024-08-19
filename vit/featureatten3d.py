from ptflops import get_model_complexity_info
import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class FeatureAttn3d(nn.Module):
    def __init__(self,
                 cin,  # 输入特征 的通道数
                 cout,  # 输出特征 的通道数
                 poolsize,  # poolsize 的全局池化特征大小
                 embedsize,  # embeding 的单词大小
                 heads,  # 多头注意力机制的头数
                 wardsize,  # 多头注意力机制的单词大小
                 ) -> None:
        super().__init__()
        self.embedsize = embedsize
        self.heads = heads
        self.wordsize = wardsize

        # embedding
        self.pool = nn.AdaptiveAvgPool3d(poolsize)
        self.qk = nn.Linear(poolsize**3, embedsize*2)

        # q,k weights
        self.w_qs = nn.Linear(embedsize, heads * wardsize)  # q
        self.w_ks = nn.Linear(embedsize, heads * wardsize)  # k

        # attention
        self.attention = ScaledDotProductAttention(temperature=wardsize ** 0.5)

        # head to one
        self.fch = nn.Linear(heads, 1)

        # out
        self.conv = nn.Conv3d(cin, cout, 1)

    def forward(self, x):
        b, c, d, w, h = x.shape

        wordsize, heads = self.wordsize,  self.heads

        # embadding q k to 1d feature, shape_qk = [b, c1+c2, embed_dim]
        xmean = F.adaptive_avg_pool3d(x, 3).reshape((b, -1, 3**3))
        q, k = self.qk(xmean).view(b, c, 2, self.embedsize).unbind(2)

        # embadding v to 1d feature , shape_v = [b, c1+c2, d*w*h]
        v = x.reshape((b, c, d*w*h))

        # q k encoder , shape_qk= [b h c1+c2 d_k]
        q = self.w_qs(q).view(b, c, heads, wordsize).transpose(1, 2)
        k = self.w_ks(k).view(b, c, heads, wordsize).transpose(1, 2)

        # attention , shape_y=[b h c1+c2 d*w*h]  shape_attn=[b h c1+c2 c1+c2]
        y, attn = self.attention(q, k, v)

        # unembadding y to 3d feature
        y = self.fch(y.permute(0, 2, 3, 1)).view(b, c, d, w, h)

        # 生长特征注意力的混淆矩阵，前c1代表特征A，后c2代表特征B
        attn = self.fch(attn.permute(0, 2, 3, 1)).view(b, c, c)

        # decoder conv 1x1x1
        y = self.conv(y+x)

        return y, attn


class ABAttn3d(nn.Module):
    def __init__(self,
                 cin1,  # 输入特征1 的通道数
                 cin2,  # 输入特征2 的通道数
                 cout,  # 输出特征 的通道数
                 poolsize,  # poolsize 的全局池化特征大小
                 embedsize,  # embeding 的单词大小
                 heads,  # 多头注意力机制的头数
                 wardsize,  # 多头注意力机制的单词大小
                 ) -> None:
        super().__init__()
        self.cin1 = cin1
        self.featureattn3d = FeatureAttn3d(cin1+cin2,  # 输入特征1 的通道数
                                           cout,  # 输出特征 的通道数
                                           poolsize,  # poolsize 的全局池化特征大小
                                           embedsize,  # embeding 的单词大小
                                           heads,  # 多头注意力机制的头数
                                           wardsize,  # 多头注意力机制的单词大小
                                           )

    def forward(self, fa, fb):
        c1 = self.cin1

        x = torch.cat((fa, fb), axis=1)

        y, attn = self.featureattn3d(x)

        fa_influence = attn[:, :c1].abs().mean(dim=-1) + attn[:, :, :c1].abs().mean(dim=-2)
        fb_influence = attn[:, c1:].abs().mean(dim=-1) + attn[:, :, c1:].abs().mean(dim=-2)

        # 输出，注意力混淆矩阵，特征a的各通道重要性系数，特征b的各通道重要性系数
        return y, attn, fa_influence, fb_influence


abattn3d = ABAttn3d(
    cin1=5,  # 输入特征 2的通道数
    cin2=10,  # 输入特征 2的通道数
    cout=5,  # 输出特征 的通道数
    poolsize=3,  # embeding 的全局池化特征大小
    embedsize=16,  # embeding 的单词大小
    heads=8,  # 多头注意力机制的头数
    wardsize=8,  # 多头注意力机制的单词大小
)

res = get_model_complexity_info(abattn3d.featureattn3d, (15, 64, 64, 64))
print(res)

y, attn, fa_influence, fb_influence = abattn3d(
    torch.randn(1, 5, 64, 64, 64),  # feature_A
    torch.randn(1, 10, 64, 64, 64),    # feature_B
)

print(attn.detach().numpy()) # 注意力混淆矩阵
print(attn.abs().detach().numpy()) # 绝对注意力混淆矩阵
print(fa_influence.detach().numpy())  # 特征a的各通道影响力
print(fb_influence.detach().numpy())  # 特征b的各通道影响力
