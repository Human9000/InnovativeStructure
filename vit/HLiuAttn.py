import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# 定义 ViT 模型
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, num_layers, mlp_dim, dropout):
        super(ViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Patch Embedding
        self.patch_embedding = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer Encoder
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])

        # 分类头
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
      
        x = self.patch_embedding(x)  # (batch, embed_dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)
        
        x = x + self.positional_embedding # (batch, num_patches, embed_dim)
        
        x = self.transformer_encoder(x) # (batch, num_patches, embed_dim)
        
        x = x.mean(dim=1) # (batch, embed_dim)
        
        x = self.classifier(x)
        
        return x

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, mlp_dim, dropout)
    
    def forward(self, x):
        
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attention(x)
        x = x + residual
        
        residual = x
        x = self.layer_norm2(x)
        x = self.feedforward(x)
        x = x + residual
        
        return x
        

# 多头自注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, HLiu = False):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax(dim=-1) 

        # 同向和反向的自适应调节参数
        self.la = torch.nn.Parameter(torch.tensor(0.5))  # 可学习参数 alpha，用于调整同向和反向注意力权重。

        self.HLiu = HLiu

    def attn_neg_pos(self, q, k, v): 
        q_neg = F.softplus(-q)  # 查询向量的负向特征
        q_pos = F.softplus(q)   # 查询向量的正向特征
        k_neg = F.softplus(-k).transpose(-1,-2)  # 键向量的负向特征
        k_pos = F.softplus(k).transpose(-1,-2)   # 键向量的正向特征

        # 2. 缩放值向量：使用缩放因子 sf 调整值向量的尺度。
        sum_sf = torch.ones((k.shape[-2], 1), device=k.device) /np.sqrt(self.head_dim) # 创建一个缩放张量
        scaled_v = v /np.sqrt(self.head_dim) # 缩放值向量

        # 3. 计算加权值：分别计算正向和负向键向量与值向量的加权结果。
        wv_neg = k_neg @ scaled_v   # 负向键向量加权值 D N @ N D -> D D
        wv_pos = k_pos @ scaled_v   # 正向键向量加权值 D N @ N D -> D D
        aw_neg = k_neg @ sum_sf     # 计算负向键向量的注意力权重 D N @ N 1 -> D 1
        aw_pos = k_pos @ sum_sf     # 计算正向键向量的注意力权重 D N @ N 1 -> D 1

        # 4. 同向注意力：计算 query 和 key 方向一致的注意力结果。
        num_same = (q_neg @ wv_neg + q_pos @ wv_pos + 1e-12) # 同向注意力分子 N D @ D D -> N D
        den_same = (q_neg @ aw_neg + q_pos @ aw_pos + 1e-12)  # 同向注意力分母 N D @ D 1 -> N 1
        res_same = num_same / den_same  # 同向注意力结果

        # 5. 异向注意力：计算 query 和 key 方向相反的注意力结果。
        num_diff = (q_neg @ wv_pos + q_pos @ wv_neg + 1e-12) # 异向注意力分子 N D @ D D -> N D
        den_diff = (q_neg @ aw_pos + q_pos @ aw_neg + 1e-12)  # 异向注意力分母  N D @ D 1 -> N 1
        res_diff = num_diff / den_diff # 异向注意力结果
        
        # 6. 动态加权：使用可学习参数 alpha 动态加权同向和反向注意力结果。
        la = torch.clamp(self.la, 0, 1)
        return res_same * la - res_diff * (1 - la)


    def attn_norm(self, q, k, v): 
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim) # (batch, num_heads, num_patches, num_patches) 
        # print(attention_scores.shape)
        attention_probs = self.softmax(attention_scores) # (batch, num_heads, num_patches, num_patches)
        attention_probs = self.dropout(attention_probs) 
        out = torch.matmul(attention_probs, v) # (batch, num_heads, num_patches, head_dim)
        return out
    

    def forward(self, x):
        batch_size = x.size(0)

        q = self.Wq(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (batch, num_heads, num_patches, head_dim)
        k = self.Wk(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (batch, num_heads, num_patches, head_dim)
        v = self.Wv(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) # (batch, num_heads, num_patches, head_dim)
 
        if self.HLiu:
            out = self.attn_neg_pos(q,k,v) # 线性的双向注意力机制
        else:
            out = self.attn_norm(q,k,v) # 标准注意力机制
    
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim) # (batch, num_patches, embed_dim)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# 参数设置
image_size = 28
patch_size = 4
num_classes = 10
embed_dim = 128
num_heads = 16
num_layers = 6
mlp_dim = 256
dropout = 0.1
batch_size = 256
learning_rate = 0.001
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型实例化
model = ViT(image_size, patch_size, num_classes, embed_dim, num_heads, num_layers, mlp_dim, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        images, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {100 * correct / total:.2f} %')

print('Finished Training')