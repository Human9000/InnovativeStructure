import matplotlib.pyplot as plt
import os
import torch
from torch.optim import lr_scheduler, Adam
from dataset import ECGDataset
from tqdm import tqdm
from torch import mode, nn
from torch.nn import functional as F
import torch.nn.init as init
from model import ECGSegMCU,ECGSegMCULBit


def multi_loss(pred: torch.Tensor, gt: torch.Tensor, p=0.5):
    gt0 = 1-gt
    gt1 = gt
    err = (gt - pred).pow(2)
    err_p = ((err * gt1).sum(dim=0) / (gt1.sum(dim=0) + 1)).mean()  # 正样本，每个类的，正样本均衡
    err_n = ((err * gt0).sum(dim=0) / (gt0.sum(dim=0) + 1)).mean()  # 负样本，每个类的，负样本均衡
    loss = err_p * p + err_n * (1-p)  # mean代表类间均衡，+代表类内均衡
    return loss


# 初始化函数
def custom_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            # 使用He初始化（适用于ReLU激活函数）
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)  # 偏置初始化为0
        # 如果有其他层（如BatchNorm），可以在这里添加初始化代码

model = ECGSegMCULBit().cuda() 
model.apply(custom_init)
# if 'model.pth' in os.listdir():
#     model.load_state_dict(torch.load('model.pth', map_location='cuda'), strict=True)


optimizer = Adam(model.parameters(), lr=5e-3)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
dataset = ECGDataset(data_root="/home/ubuntu/dataset/phy2020segcls")
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


max_epochs = 100
model.train()


for epoch in range(max_epochs):
    progress_bar = tqdm(loader, desc=f'训练轮次 {epoch+1}/{max_epochs}', total=len(loader))
    for i, (data, target_seg) in enumerate(progress_bar):
        if torch.isnan(data).any():  # 提出空数据
            continue
        data, target_seg = data.to('cuda') , target_seg.to('cuda')   # 将数据移动到指定设备
        optimizer.zero_grad()  # 清零梯度
        # print(data.shape) 
        data = data.transpose(1, 2) 
        y = model(data)  # 前向传播
        # print(y.shape, target_seg.isnan().sum())
        loss = multi_loss(y[:, 232:-100], target_seg[:, 232:-100])  # 计算损失
        loss.backward()
        optimizer.step()
        scheduler.step()
        # 更新进度条显示当前批次损失
        progress_bar.set_postfix({'损失': f'{loss.item():.6f}'})

    for i, (data, target_seg) in enumerate(progress_bar):
        if torch.isnan(data).any():  # 提出空数据
            continue
        data, target_seg = data.to('cuda'), target_seg.to('cuda')  # 将数据移动到指定设备
        optimizer.zero_grad()  # 清零梯度 
        data = data.transpose(1, 2) 
        y = model(data)  # 前向传播
        y = torch.softmax(y, dim=2)
        y -= y.min(dim=2, keepdim=True)[0]
        y /= y.max(dim=2, keepdim=True)[0]
        y = (y==1)*1
        plt.figure(figsize=(10, 4))
        # plt.subplot(4, 1, i+1)
        for j in range(3):  
            for i0,i1 in [(0,1),(1,3),(2,2)]:
                plt.subplot(3, 3, i0*3+1+j)
                plt.plot(data[j, 0, ].detach().cpu().numpy())
                plt.plot(target_seg[j, :, i1].detach().cpu().numpy(), 'r')
                plt.plot(y[j, :, i1].detach().cpu().numpy(), 'g--')
                plt.axis('off')
        # plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig("a.png")
        plt.close()
        break
    state_dict = model.state_dict()
    torch.save(state_dict, f"model.pth")
