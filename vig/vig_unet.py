 
import torch
from torch import nn  
from vig import ViGBlock 
# ===============================
# input() → vig → vig → output()
#            ↓     ↑
#           vig → vig
#            ↓     ↑
#           vig → vig
#            ↓     ↑
#           vig → vig
#            ↓     ↑
#           vig → vig
# ================================= 
class ViGUnet(nn.Module):
    def __init__(self, cin, cout, dim=96, head=4, patch_size=16, u_res_layers=[2, 4, 6, 8],  Bi=False):
        super().__init__()
        self.deep = max(u_res_layers) + 2
        self.u_res_layers = u_res_layers
        self.cmid = channle = max(dim//patch_size, int(cout**0.5)+1)  # 中间通道数
        self.input_layer = nn.Conv2d(cin, channle, kernel_size=1, bias=False)  # 输入层
        self.vig_en = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
        self.vig_de = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
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
                x = self.residual[self.u_res_layers.index(i)](torch.cat((en[i], x), dim=1))
            else:
                pass
            x = self.vig_de[i](x) 
        y = self.output_layer(x)
        return y
 

# ============================================
#                    output()
#                       ↑
# input(1) → vig →→→→→ vig ←←←←← vig ← input(2) 
#             ↓         ↑         ↓
#             vig →→→→ vig ←←←← vig 
#              ↓        ↑        ↓
#              vig →→→ vig ←←← vig 
#               ↓       ↑       ↓
#               vig →→ vig ←← vig 
#                ↓      ↑      ↓
#                vig → vig ← vig
# ==============================================
class ViGWnet(nn.Module):
    def __init__(self, cin1, cin2, cout, dim=96, head=4, patch_size=16, u_res_layers=[2, 4, 6, 8],  Bi=False):
        super().__init__()
        self.deep = max(u_res_layers) + 2
        self.u_res_layers = u_res_layers
        self.cmid = channle = max(dim//patch_size, int(cout**0.5)+1)  # 中间通道数
        self.input_layer1 = nn.Conv2d(cin1, channle, kernel_size=1, bias=False)  # 输入层
        self.input_layer2 = nn.Conv2d(cin2, channle, kernel_size=1, bias=False)  # 输入层
        self.vig_en1 = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
        self.vig_en2 = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
        self.vig_de = nn.ModuleList([ViGBlock(channle, patch_size, dim, head, Bi) for _ in range(self.deep)])
        self.residual = nn.ModuleList([nn.Conv2d(channle*3, channle, kernel_size=1, bias=False) for _ in self.u_res_layers])
        self.output_layer = nn.Conv2d(channle, cout, kernel_size=3, padding=1, bias=False)  # 输出层

    def forward(self, x1, x2):
        x1 = self.input_layer1(x1)
        x2 = self.input_layer2(x2)

        en1 = []
        en2 = []
        # 编码器阶段
        for i in range(self.deep):
            x1 = self.vig_en1[i](x1)
            x2 = self.vig_en2[i](x2)
            en1.append(x1)
            en2.append(x2)

        x = x1 + x2
        # 解码阶段
        for i in range(self.deep-1, -1, -1):
            if i in self.u_res_layers:  # 指定层的U残差链接 
                x = self.residual[self.u_res_layers.index(i)](torch.cat((en1[i], en2[i], x), dim=1))
            else:  
                pass
            x = self.vig_de[i](x) 
        y = self.output_layer(x)
        return y

def test_wnet(): 
    x = torch.randn(32, 1, 256, 256).cuda()
    model = ViGWnet(cin1=1,
                    cin2=1,
                    cout=1,
                    dim=96,
                    head=8,
                    patch_size=16,
                    u_res_layers=[0, 1, 2, 3],
                    Bi=True,
                    ).cuda()
 
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.eval() 
    for _ in range(100):
        out = model(x, x)
        opt.zero_grad()
        opt.step()
        print("Output shape:", out.shape)



def test_unet(): # 测试ViGUnet
    x = torch.randn(32, 1, 256, 256).cuda()
    model = ViGUnet(cin=1, 
                    cout=1,
                    dim=96,
                    head=8,
                    patch_size=16,
                    u_res_layers=[0, 1, 2, 3],
                    Bi=True,
                    ).cuda()
 
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.eval() 
    for _ in range(100):
        out = model(x)
        opt.zero_grad()
        opt.step()
        print("Output shape:", out.shape)

def test_loop_hight(): # 测试ViGUnet
    img = torch.randn(32, 3, 256, 256).cuda()
    ir = torch.randn(32, 1, 256, 256).cuda()
    enhance = torch.randn(32, 1, 256, 256).cuda()
    model = ViGUnet(cin=5, 
                    cout=1,
                    dim=96,
                    head=8,
                    patch_size=16,
                    u_res_layers=[0, 1, 2, 3],
                    Bi=True,
                    ).cuda()
  
    model.eval() 
    enhancei = enhance
    with torch.no_grad():
        for _ in range(100):
            enhancei = model(torch.cat([img, ir, enhancei], dim=1))   
            print("enhancei shape:", enhancei.shape)

if __name__ == '__main__':  
    test_unet()
