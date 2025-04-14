from vim import ViMBlock
import torch
from torch import nn
from torch.nn import Module
  

# ===============================
# input() → ViM → ViM → output()  # res 0
#            ↓     ↑
#           ViM → ViM             # res 1
#            ↓     ↑
#           ViM → ViM             # res 2
#            ↓     ↑
#           ViM → ViM             # res 3
#            ↓     ↑
#           ViM → ViM             # max_deep
# ================================= 
class ViGUnet(nn.Module):
    def __init__(self, cin, cout, dim=96, head=4, patch_size=16, u_res_layers=[2, 4, 6, 8],  BiGRU=False):
        super().__init__()
        self.deep = max(u_res_layers) + 2
        self.u_res_layers = u_res_layers
        self.cmid = channle = max(dim//patch_size, int(cout**0.5)+1)  # 中间通道数
        self.input_layer = nn.Conv2d(cin, channle, kernel_size=1, bias=False)  # 输入层
        self.vig_en = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, BiGRU) for _ in range(self.deep)])
        self.vig_de = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, BiGRU) for _ in range(self.deep)])
        self.residual = nn.ModuleList([nn.Conv2d(channle*2, channle, kernel_size=1, bias=False) for _ in self.u_res_layers])
        self.output_layer = nn.Conv2d(channle, cout, kernel_size=3, padding=1, bias=False)  # 输出层

    def forward(self, x):
        x = self.input_layer(x)

        en = []
        # 编码器阶段
        for i in range(self.deep):
            x = self.vig_en[i](x)
            en.append(x)

        # 解码阶段
        for i in range(self.deep-1, -1, -1):
            if i in self.u_res_layers:  # 指定层的U残差链接
                print("res",i)
                x = self.residual[self.u_res_layers.index(i)](torch.cat((en[i], x), dim=1))
            else:
                
                print(i)
            x = self.vig_de[i](x) 
        y = self.output_layer(x)
        return y
 


if __name__ == '__main__':  
    x = torch.randn(32, 1, 256, 256).cuda()
    model = ViGUnet(cin=1,
                    cout=1,
                    dim=64,
                    head=4,
                    patch_size=8,
                    u_res_layers=[0, 1, 2, 3],
                    BiGRU=True,
                    ).cuda()

    mingru = minGRU(8, 1, None).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.eval()
    with torch.no_grad():
        for i in range(1):
            out = model(x)
            opt.zero_grad()
            opt.step()
            print("Output shape:", out.shape)
