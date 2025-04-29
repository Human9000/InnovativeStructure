from vim import ViMBlock
import torch
from torch import nn 


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
class ViMUnet(nn.Module):
    def __init__(self, cin, cout, dim=96, head=4, patch_size=16, u_res_layers=[2, 4, 6, 8],  Bi=False):
        super().__init__()
        self.deep = max(u_res_layers) + 2
        self.u_res_layers = u_res_layers
        self.cmid = channle = max(dim//patch_size, int(cout**0.5)+1)  # 中间通道数
        self.input_layer = nn.Conv2d(cin, channle, kernel_size=1, bias=False)  # 输入层
        self.vig_en = nn.ModuleList([ViMBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
        self.vig_de = nn.ModuleList([ViMBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
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
                # print("res",i)
                x = self.residual[self.u_res_layers.index(i)](torch.cat((en[i], x), dim=1))
            else: 
                # print(i)
                pass
            x = self.vig_de[i](x) 
        y = self.output_layer(x)
        return y
 

def test_loop_hight(): # 测试ViMUnet
    img = torch.randn(8, 3, 256, 256).cuda()
    ir = torch.randn(8, 1, 256, 256).cuda()
    enhance = torch.randn(8, 1, 256, 256).cuda()
    model = ViMUnet(cin=5, 
                    cout=1,
                    dim=96,
                    head=8,
                    patch_size=16,
                    u_res_layers=[0, 1, 2, 3],
                    Bi=True,
                    ).cuda() 
    # model.eval() 
    enhancei = enhance
    opt = torch.optim.Adam(model.parameters(), lr=0) 
    # with torch.no_grad():
    for _ in range(100):
        x = torch.cat([img, ir, enhancei.detach_()], dim=1)
        opt.zero_grad()
        enhancei = model(x)
        # enhancei.mean().backward()
        opt.step()
        print("enhancei shape:", enhancei.shape)


if __name__ == '__main__':   
    test_loop_hight()
