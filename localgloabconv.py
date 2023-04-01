import torch
from torch import nn
from torch.nn import functional as F
from ptflops import get_model_complexity_info


class LocalConvSelfAttention(nn.Module):
    def __init__(self,
                 wordsize=64,
                 ks=8,
                 attn_dropout = 0.2
                 ) -> None:
        super().__init__()

        self.wordsize = wordsize
        self.ks = ks
        self.conv1 = nn.Conv3d(1, wordsize, ks, ks)

        self.wq = nn.Conv3d(wordsize, wordsize, 1, bias=False)
        self.wk = nn.Conv3d(wordsize, wordsize, 1, bias=False)
        self.wv = nn.Conv3d(wordsize,  ks**3, 1, bias=False)
 
        self.dropout = nn.Dropout(attn_dropout)
 
    def scaled_dot_product_attention(self, q, k, v, temperature=None):
        if temperature is None:
            temperature = q.shape[-1]**0.5
        attn = torch.matmul(q / temperature, k)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
    
    def embedding(self, x):
        b,c,d,w,h = x.shape
        y = self.conv1(x.view(b*c, 1, d, w, h))
        bc,c2,d,w,h = y.shape
        return y.view(b,c,c2,*y.shape[2:])

    def qkv(self, x): 
        b,c = x.shape[:2]
        x= x.view(b*c,*x.shape[2:])
        q = self.wq(x).view(b, c, self.wordsize, -1).permute(0, 3, 1, 2) # q 
        k = self.wk(x).view(b, c, self.wordsize, -1).permute(0, 3, 2, 1)# k 
        v = self.wv(x).view(b, c, self.ks**3, -1).permute(0, 3, 1, 2)# v 
        return q,k,v

    def unembedding(self, x, block_size, feature_size): 
        b,h,c,dwh = x.shape
        # y = self.conv2(x)
        y=x.view(b, *block_size, c, self.ks, self.ks, self.ks)  
        y = y.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()   
        b, c, d, ks1, w, ks2, h, ks3 = y.shape  
        y = F.interpolate(y.view(b, c, d*ks1, w*ks2, h*ks3),feature_size)  
        return y
     
    def forward(self, x):
        y1 = self.embedding(x)   # embedding
        q,k,v = self.qkv(y1) # q,k,v获取
        y2,attn = self.scaled_dot_product_attention(q,k,v) # self attention 
        y3 = self.unembedding(y2, y1.shape[-3:], x.shape[-3:]) # unembding
        y = y3 + x #self.conv(x)  # Residual connection 
        return y,attn
    
class GlobalConvSelfAttention(nn.Module):
    def __init__(self,
                 heads=64,
                 ks=8,
                 attn_dropout = 0.2
                 ) -> None:
        super().__init__()

        self.heads = heads
        self.ks = ks
        self.conv1 = nn.Conv3d(1, heads, ks, ks)
        self.conv2 = nn.Conv2d(heads, ks**3, 1)

        self.wq = nn.Conv3d(heads, heads, 1, bias=False)
        self.wk = nn.Conv3d(heads, heads, 1, bias=False)
        self.wv = nn.Conv3d(heads, 1, 1, bias=False)
 
        self.dropout = nn.Dropout(attn_dropout)
 
    def scaled_dot_product_attention(self, q, k, v, temperature=None):
        if temperature is None:
            temperature = q.shape[-1]**0.5
        attn = torch.matmul(q / temperature, k)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
    
    def embedding(self, x):
        b,c,d,w,h = x.shape
        y = self.conv1(x.view(b*c, 1, d, w, h))
        bc,c2,d,w,h = y.shape
        return y.view(b,c,c2,*y.shape[2:])

    def qkv(self, x): 
        b,c = x.shape[:2]
        x= x.view(b*c,*x.shape[2:])
        q = self.wq(x).view(b, c, self.heads, -1).permute(0, 2, 1, 3) # q 
        k = self.wk(x).view(b, c, self.heads, -1).permute(0, 2, 3, 1)# k 
        v = self.wv(x).view(b, c, 1, -1).permute(0, 2, 1, 3)# v 
        return q,k,v

    def unembedding(self, x, block_size, feature_size): 
        b,h,c,dwh = x.shape
        y = self.conv2(x)
        y=y.view(b, self.ks, self.ks, self.ks, c, *block_size)  
        y = y.permute(0, 4, 5, 1, 6, 2,  7,  3).contiguous()   
        b, c, d, ks1, w, ks2, h, ks3 = y.shape  
        y = F.interpolate(y.view(b, c, d*ks1, w*ks2, h*ks3),feature_size)  
        return y
     
    def forward(self, x):
        y1 = self.embedding(x)   # embedding
        q,k,v = self.qkv(y1) # q,k,v获取
        y2,attn = self.scaled_dot_product_attention(q,k,v) # self attention 
        y3 = self.unembedding(y2,y1.shape[-3:], x.shape[-3:]) # unembding
        y = y3 + x #self.conv(x)  # Residual connection 
        return y,attn
    
class LGConv(nn.Module):
    def __init__(self, kernal_size, word_size, heads, drop_rate) -> None:
        super().__init__()
        self.lconv = LocalConvSelfAttention(word_size, kernal_size, drop_rate)
        self.gconv = GlobalConvSelfAttention(heads, kernal_size, drop_rate)
    
    def forward(self, x):
        ly,lattn =self.lconv(x) 
        gy,gattn =self.gconv(x) 
        return ly+gy, lattn, gattn
    

if __name__ == '__main__':
    cin, d, w, h =  32, 64, 64, 64
    lgconv = LGConv(11, 64, 64, 0.2)  # 注意力窗口是11
    lconv = LocalConvSelfAttention (8, 64, 0.2) # 注意力窗口是8
    gconv = GlobalConvSelfAttention (8, 64, 0.2) # 注意力窗口是8
    conv = nn.Conv3d(cin,cin,5,1,2) # 卷积核大小是5
    import time
    t0 = time.time() 
    
    res = get_model_complexity_info(lgconv,(cin,d,w,h), print_per_layer_stat=False)
    t1 = time.time()
    print("lgconv", res, t1 - t0)
    res = get_model_complexity_info(lconv,(cin,d,w,h), print_per_layer_stat=False)
    t2 = time.time()
    print("lconv", res, t2 - t1)

    res = get_model_complexity_info(gconv,(cin,d,w,h), print_per_layer_stat=False)
    t3 = time.time()
    print("gconv", res, t3 - t2)

    res = get_model_complexity_info(conv,(cin,d,w,h), print_per_layer_stat=False)
    t4 = time.time()
    print("conv3d",res, t4 - t3)
 
