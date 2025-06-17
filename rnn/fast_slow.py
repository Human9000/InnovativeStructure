import torch
import torch.nn as nn
import torch.optim as optim

# 快变量，慢变量
class FSVar(nn.Module):
    def __init__(self, fast_channels, slow_channels, ratio=5):
        super(FSVar, self).__init__()
        assert ratio>= 1,  "ratio must be greater than or equal to 1"
        self.f = fast_channels
        self.s = slow_channels
        self.ratio = ratio

    def forward(self, x):
        fast_c = self.f
        slow_c = self.s
        ratio = self.ratio
        x[:,:fast_c] = x[:,:fast_c] * ratio
        x[:,-slow_c:] = x[:,-slow_c:] / ratio
        return x
 

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fsv = FSVar(16, 16, 10)

    def forward(self, x):
        x = torch.relu(self.fsv(self.conv1(x)))
        x = torch.relu(self.fsv(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x





# 创建模型实例
model = SimpleCNN()

# 定义学习率
lr = 1e-3 
# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=lr)

# 模拟一次训练步骤
input_data = torch.randn(1, 3, 28, 28)
target = torch.randint(0, 10, (1,))

# 前向传播
output = model(input_data)
loss = output.mean()

# 反向传播
optimizer.zero_grad()
loss.backward()

# print(model.conv1.weight.grad)
# 更新参数
optimizer.step()
