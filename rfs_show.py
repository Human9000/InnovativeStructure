'''
Author       : Hao Liu 33023091+Human9000@users.noreply.github.com
Date         : 2025-10-20 13:53:40
LastEditors  : Hao Liu 33023091+Human9000@users.noreply.github.com
LastEditTime : 2025-10-20 15:07:57
FilePath     : \InnovativeStructure\rfs_show.py
Description  : 
Copyright (c) 2025 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
"""
最简单的使用 timm 框架微调 rfs_resnet18 在 CIFAR-100 上的训练脚本
- 自动下载 CIFAR-100
- 单标签分类
- 包含训练、验证和模型保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
import timm
import RFS.rfs_v1  # 导入你注册的 rfs_resnet18
from RFS.rfs_v1 import plot_papper
import tqdm

if __name__ == "__main__":
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 超参数
    batch_size = 128
    epochs = 20
    lr = 1e-3
    weight_decay = 1e-4
 

    val_trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
 
    val_ds = CIFAR100(root='./data', train=False, download=True, transform=val_trans)
 
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 模型
    # model = timm.create_model('resnet18', pretrained=True, num_classes=100).to(device)
    model = timm.create_model('rfs_resnet18', pretrained=True, num_classes=100).to(device)
 
    state_dict = torch.load('rfs_resnet18_cifar100.pth')
    model.load_state_dict(state_dict) 
 
    # 验证
    model.eval()
    correct, total = 0, 0
    from matplotlib import pyplot as plt
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # break 
            plt.figure(num=2) 
            img = imgs.transpose(1,3).transpose(1,2)[0].cpu()
            plt.imshow(img)
            plt.show()
            plot_papper(model)