import torch

model = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0, last_epoch=-1)

# lr: 学习率
# weight_decay: 权重衰减（正则化项系数）


奇数到达最低点
偶数到达最高点

2/3 
4/3
6/3 最高点
8/3   
10/3
12/3 最高点




