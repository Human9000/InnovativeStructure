from torch import nn
from torch.nn import functional as F

from gru_conv2dV2 import GRUConv2d 

class FastGRU(nn.Module):
    """ FastGRU模块 """
    def __init__(self,
                 in_channels,
                 out_channels, 
                 window_size=8,
                 ):
        super().__init__() 
        self.window_size = window_size
        self.gru_2d = GRUConv2d(in_channels, out_channels, kernel_size=1,
                             directions=[(0, 1), (0, -1), (1, 0), (-1, 0),
                                        #  (1, 1), (1, -1), (-1, 1), (-1, -1)
                                         ])
    def forward(self, x): 
        B, L, d = x.shape
        w = self.window_size        
        x = F.pad(x, (0, 0, 0, (w - L % w) % w), "constant", 0)    # padding 保证L是w的整数倍     
        x = x.transpose(-1,-2).view(B, d, -1, w)# 调整x的形状         
        x = self.gru_2d(x) # 计算GRU 
        x = x.view(B, d, -1).transpose(-1,-2)# 调整x的形状 
        return x[:, :L] # 截取有效部分



if __name__ == '__main__':
    gru = FastGRU(64, 64, window_size=16).cuda()
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(gru, (320, 64),
                                             as_strings=True,
                                             print_per_layer_stat=False)
    print('macs: ', macs, 'params: ', params)
