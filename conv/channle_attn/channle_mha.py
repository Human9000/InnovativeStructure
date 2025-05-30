from torch import nn
from torch.nn import functional as F


    
class MhAConv(nn.Module):
    def __init__(self, in_channle, out_channle, kernel_size, embadding, head, stride, dropout=0.1):
        super(MhAConv, self).__init__()
        self.inProj = nn.Conv2d(in_channle, embadding, kernel_size=1, stride=1, padding=0, bias=False)
        self.q = nn.Conv2d(embadding, embadding*kernel_size*head, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=embadding) 
        self.k = nn.Conv2d(embadding, embadding*kernel_size*head, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=embadding) 
        self.v = nn.Conv2d(embadding, embadding*head, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=embadding)
        self.pe = nn.Conv2d(embadding, embadding, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=embadding)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(dropout)
        self.proj1 = nn.Conv2d(embadding*kernel_size*head, embadding, kernel_size=1, stride=1, padding=0, bias=False, groups=embadding)
        self.outProj = nn.Conv2d(embadding, out_channle, kernel_size=1, stride=1, padding=0, bias=False)
        self.head = head
        self.kernel_size = kernel_size
        self.c = kernel_size**-0.5

    def forward(self, x): 
        x = self.inProj(x)
        B, C, H, W = x.size()
        pe = self.pe(x)
        head = self.head
        kernel_size = self.kernel_size
        qkv = self.qkv(x).view(B, 3, head, C, kernel_size, H*W)
        q,k,v = qkv.transpose(1,-1).unbind(-1) # 3*[B, H*W, head, C, kernel_size]
        attn = self.softmax(q@k.transpose(-2,-1) * self.c)  # [B, H*W, head, C, C]
        x = (attn @ v).transpose(1,-1).contiguous().view(B, C, H, W) 
        x = self.proj(self.drop(x))
        x = self.proj2(self.pe) 
        return x