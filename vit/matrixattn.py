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


class MatrixAttention(nn.Module): 
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        key_dim = int(head_dim * attn_ratio)
        scale = key_dim**-0.5
        nh_kd = key_dim * num_heads
        h = dim + nh_kd * 4
        self.rcv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.func = MatrixAttentionFunc(num_heads, key_dim, head_dim, scale)

    def forward(self, x):
        return self.func(x, self.rcv, self.pe, self.proj) 
    
 
class MatrixAttentionFunc(nn.Module):
    def __init__(self, num_heads, key_dim, head_dim, scale) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.head_dim = head_dim
        self.scale = scale


        # 注册到ptflop的参数量计算中 
        from ptflops.pytorch_engine import MODULES_MAPPING
            
        def _flops_counter_hook(module, input, output):
            B, C, h, w = input[0].shape
            H = self.num_heads
            kd = self.key_dim  
            module.__flops__ += B * H * kd * h * h * w #  attn_r = torch.einsum('bHdiw,bHdjw->bHijw', rq, rk) * self.scale
            module.__flops__ += B * H * kd * h * w * w #  attn_w = torch.einsum('bHdhi,bHdhj->bHihj', cq, ck) * self.scale 
            module.__flops__ += B * C * h * w * (w+h)  #  x = torch.einsum('bHihw,bHdij,bHjhw->bHdhw', attn_r, v, attn_w).reshape(B, C, H, W)
 
        MODULES_MAPPING[MatrixAttentionFunc] = _flops_counter_hook

    def forward(self,x, rcv, pe, proj):
        B, C, H, W = x.shape
        rcv = rcv(x)
        rq, rk, cq, ck, v = rcv.view(B, self.num_heads, self.key_dim * 4 + self.head_dim, H, W).split(
            [self.key_dim,self.key_dim,self.key_dim, self.key_dim, self.head_dim], dim=2
        ) # B H d h w
        attn_r = torch.einsum('bHdiw,bHdjw->bHijw', rq, rk) * self.scale
        attn_w = torch.einsum('bHdhi,bHdhj->bHihj', cq, ck) * self.scale 
        attn_r = attn_r.softmax(dim=2)
        attn_w = attn_w.softmax(dim=2) 
        x = torch.einsum('bHihw,bHdij,bHjhw->bHdhw', attn_r, v, attn_w).reshape(B, C, H, W)
        x = x + pe(v.reshape(B, C, H, W))
        x = proj(x)
        return x
    

if __name__ == '__main__': 
    ma = MatrixAttention(16).cuda()
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(ma, (16, 512, 512), as_strings=True, print_per_layer_stat=True)
    print('macs: ', macs, 'params: ', params)
