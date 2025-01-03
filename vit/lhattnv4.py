import torch
import torch.nn as nn
import torch.nn.functional as F 

class HLiuAttn(torch.nn.Module):
    """
    HaoLiu 注意力机制的实现。

    该机制通过分解查询 (query) 和键 (key) 向量的正负特征，
    并动态加权同向和反向注意力结果，从而捕捉序列中不同方向的关系。
    """
    def __init__(self, d_model, alpha=0.5):
        """
        初始化函数。

        Args:
            d_model (int): 输入向量维度，也等于输出向量维度。
            alpha (float, optional): 可学习参数 alpha 的初始值 (0-1)，用于动态调整同向和反向注意力权重。默认为 0.5。
        """
        super(HLiuAttn, self).__init__()
        self.sf = d_model ** -0.5  # 缩放因子，用于稳定注意力计算。
        self.la = 1. #torch.nn.Parameter(torch.tensor(alpha))  # 可学习参数 alpha，用于调整同向和反向注意力权重。
        self.eps = 1e-9  # 偏置值，防止除零错误。

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)  # 用于将输入向量转换为查询、键和值向量的线性层。

    # def forward(self, x):
    #     """
    #     前向传播计算。

    #     Args:
    #         q (torch.Tensor): 查询向量，形状为 (batch_size, seq_len, d_model)。
    #         k (torch.Tensor): 键向量，形状为 (batch_size, seq_len, d_model)。
    #         v (torch.Tensor): 值向量，形状为 (batch_size, seq_len, d_model)。

    #     Returns:
    #         torch.Tensor: 注意力加权后的输出，形状为 (batch_size, seq_len, d_model)。
    #     """
    #     q, k, v = self.qkv(x).chunk(3, dim=-1)  # 将输入向量 x 分解为查询、键和值向量。
    #     # 1. 分离正负特征：使用 softplus 提取 query 和 key 的正负信息。
    #     q_neg = F.softplus(-q)  # 查询向量的负向特征
    #     q_pos = F.softplus(q)   # 查询向量的正向特征
    #     k_neg = F.softplus(-k).transpose(-1,-2)  # 键向量的负向特征
    #     k_pos = F.softplus(k).transpose(-1,-2)   # 键向量的正向特征

    #     # 2. 缩放值向量：使用缩放因子 sf 调整值向量的尺度。
    #     sum_sf = torch.ones((k.shape[-2], 1), device=k.device) * self.sf # 创建一个缩放张量
    #     scaled_v = v * self.sf # 缩放值向量

    #     # 3. 计算加权值：分别计算正向和负向键向量与值向量的加权结果。
    #     wv_neg = k_neg @ scaled_v   # 负向键向量加权值
    #     wv_pos = k_pos @ scaled_v   # 正向键向量加权值
    #     aw_neg = k_neg @ sum_sf    # 计算负向键向量的注意力权重
    #     aw_pos = k_pos @ sum_sf   # 计算正向键向量的注意力权重

    #     # 4. 同向注意力：计算 query 和 key 方向一致的注意力结果。
    #     num_same = (q_neg @ wv_neg + q_pos @ wv_pos + self.eps) # 同向注意力分子
    #     den_same = (q_neg @ aw_neg + q_pos @ aw_pos + self.eps)  # 同向注意力分母
    #     res_same = num_same / den_same  # 同向注意力结果

    #     # 5. 反向注意力：计算 query 和 key 方向相反的注意力结果。
    #     num_diff = (q_neg @ wv_pos + q_pos @ wv_neg + self.eps) # 反向注意力分子
    #     den_diff = (q_neg @ aw_pos + q_pos @ aw_neg + self.eps)  # 反向注意力分母
    #     res_diff = num_diff / den_diff # 反向注意力结果
        
    #     # 6. 动态加权：使用可学习参数 alpha 动态加权同向和反向注意力结果。
    #     la =1 # torch.clamp(self.la, 0, 1)
    #     return res_same * la - res_diff * (1 - la)
     

    def forward(self, x):  
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 将输入向量 x 分解为查询、键和值向量。
        k = k.transpose(-1,-2)
        return F.softmax(q@k * self.sf, dim=-1) @ v

class MNISTClassifier(nn.Module):
    def __init__(self, d_model=128, num_classes=10, num_attn_layers=3):
        super(MNISTClassifier, self).__init__()
        self.conv_in = nn.Conv2d(1, d_model, kernel_size=3, stride=1, padding=2) 
        self.attn_layers = nn.ModuleList([HLiuAttn(d_model) for _ in range(num_attn_layers)]) 
        self.relu_mid = nn.ModuleList([nn.ReLU() for _ in range(num_attn_layers - 1)])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear_out = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        
    def forward(self, x):
        attn_out = self.conv_in(x)  # 通过线性层，将数据映射到 b, d, 14, 14 维度 
        b,d,h,w = attn_out.shape
        
        for i, attn in enumerate(self.attn_layers): 
             attn_out = attn_out.view(b, d, h*w).transpose(-1,-2)
             attn_out = attn(attn_out)
             attn_out = attn_out.transpose(-1, -2).view(b, d, h, w)
             if i < len(self.attn_layers) - 1 :
                 attn_out = F.relu(F.avg_pool2d(attn_out, 2, 2)) # 使用全局平均池化得到 d_model 的向量
                 b,d,h,w = attn_out.shape
             
        # print(attn_out.shape)
        gap_out = self.gap(attn_out).view(b, d) # 使用全局平均池化得到 d_model 的向量
        x = self.linear_out(gap_out)
        return x
    
    

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型，损失函数，优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier(64, 10, 4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)



# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        progress_bar.set_postfix({'loss': loss.item()})

    train_accuracy = 100 * train_correct / total_samples
    avg_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
    
    scheduler.step(avg_loss)


    # 验证循环
    model.eval()
    test_loss = 0.0
    test_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            test_correct += (predicted == labels).sum().item()
        test_accuracy = 100 * test_correct / total_samples
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%")

print("Training finished.")