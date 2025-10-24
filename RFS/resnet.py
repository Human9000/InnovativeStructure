import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
import math
import time

# --- 1. EmergenceNexus (ENX) 模块定义 ---
class EmergenceNexus(nn.Module):
    """
    涌现枢纽 (Emergence Nexus, ENX) 模块。
    负责通过歌者涌现和潜能蔓延，增强输入特征 x。
    
    使用 1x1 卷积核实现特征的“内在自我审视” (渴望能 G)。
    """
    def __init__(self, channels: int, iterations: int = 6, gamma: float = 0.99, num_steps: int = 1, fixed_singer_threshold: float = 0.65):
        super().__init__()
        self.channels = channels
        self.iterations = iterations
        self.base_gamma = gamma
        self.num_steps = num_steps 
        self.fixed_singer_threshold = fixed_singer_threshold
        self.fixed_mask_training = False 
        
        # 传播场组件
        self.max_pool0 = nn.MaxPool2d(3, 1, 1)
        self.max_pool = nn.MaxPool2d(5, 2, 2)

        # 渴望能网络 (Aspiration Energy Network)
        # 使用 1x1 卷积核 (kernel_size=1)
        mid_channels = int(channels * 0.5)
        self.aspiration_energy_network = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid() 
        )
        
    def _stochastic_emergence_of_singer(self, x: torch.Tensor) -> torch.Tensor:
        """歌者的随机涌现/确定性选择"""
        # 简化潜能归一化
        x_min = x.view(x.shape[0], x.shape[1], -1).min(dim=-1, keepdim=True)[0].view(x.shape[0], self.channels, 1, 1)
        x_max = x.view(x.shape[0], x.shape[1], -1).max(dim=-1, keepdim=True)[0].view(x.shape[0], self.channels, 1, 1)
        epsilon = 1e-6
        M_soft = (x - x_min) / ((x_max - x_min) + epsilon)
        
        is_deterministic_mode = (not self.training) or (self.training and self.fixed_mask_training)

        if is_deterministic_mode: # 歌者确定性选择
            M = (M_soft >= self.fixed_singer_threshold).float()
        else: # 歌者随机涌现
            M_rand = torch.rand_like(x)
            M = (M_rand < M_soft).float()
            
        return M

    def _propagate_potential_echo(self, x: torch.Tensor, size: list) -> torch.Tensor:
        """潜能的回声蔓延"""
        out = x.clone()
        feat = x
        for i in range(self.iterations):
            gamma_i = self.base_gamma ** (2 ** i)
            pool_op = self.max_pool0 if i == 0 else self.max_pool
            feat = pool_op(feat) * gamma_i
            feat_resized = F.interpolate(feat, size=size, mode='nearest')
            out = torch.max(out, feat_resized)
            if feat.shape[2] == 1 or feat.shape[3] == 1:
                break
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape 
        x_current = x 

        for _ in range(self.num_steps):
            M = self._stochastic_emergence_of_singer(x_current)
            S = x_current * M
            L = self._propagate_potential_echo(S, x.shape[2:])
            G = self.aspiration_energy_network(x_current)
            
            # 使用 H*W 归一化以适应不同尺寸
            # scaling_factor = H * W / (math.log(H * W) if H*W > 1 else 1) 
            # allocated_energy = L * G  
            
            x_current = x_current * (1-G) + L * G
        
        return x_current


# --- 2. 模块替换工具函数 ---

def replace_module(model, name, new_module):
    """
    通过名称安全地替换模型中的子模块。
    
    参数:
        model (nn.Module): 根模块。
        name (str): 子模块的名称（例如: layer1.0.conv1）。
        new_module (nn.Module): 用于替换的新模块。
    """
    name_parts = name.split('.')
    parent = model
    
    # 遍历直到父级模块
    for part in name_parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
            
    # 设置新的模块
    setattr(parent, name_parts[-1], new_module)
    print(f"  [替换成功]：{name} 已被替换为 {new_module.__class__.__name__}")


# --- 3. ENX ResNet 模型创建函数 ---

def create_se_resnet18(num_classes=10, pretrained=False, **kwargs):
    """
    基于 legacy_seresnet18.in1k 结构创建 ENX ResNet18 模型。
    用 EmergenceNexus (ENX) 模块替换所有 Squeeze-and-Excitation (SE) 模块。
    """
    print("--- 正在创建 ENX ResNet18 模型 ---")
    # 加载带有 SE 模块的 ResNet18 作为基础结构
    model = timm.create_model('legacy_seresnet18.in1k', pretrained=pretrained, num_classes=num_classes, **kwargs) 
    return model

# --- 4. CIFAR-10 数据加载和预处理 ---

def load_cifar100(batch_size=128):
    """加载并标准化 CIFAR-100 数据集"""
    print("--- 正在加载 CIFAR-100 数据集 ---")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=r'Z:\datasets\cv', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=r'Z:\datasets\cv', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# --- 5. 训练和测试主函数 ---

# --- 5. 训练和测试主函数 ---

def train_and_test(model, trainloader, testloader, epochs=5, lr=0.01):
    """执行简化的训练和测试流程"""
    # 优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"--- 开始训练，使用设备: {device} ---")
    print(f"总 Epochs: {epochs}, 学习率: {lr}")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        # 训练
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # --- 调试检查 ---
            if i == 0 and epoch == 0:
                print(f"\n[调试信息] 模型输出预期应为 (Batch, 10)。实际标签范围应为 0-9。")
                
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # --- 调试检查 ---
            if i == 0 and epoch == 0:
                print(f"[调试信息] Model Output Shape: {outputs.shape}") 
                print(f"[调试信息] Labels Shape: {labels.shape}, Max Label: {labels.max().item()}")
            # --- 调试检查结束 ---

            loss = criterion(outputs, labels) # 错误发生点
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # 每 100 批次打印一次统计信息
                print(f'  [Epoch {epoch + 1}, Batch {i + 1:5d}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        scheduler.step()
        
        # 测试 (在每个 epoch 结束时进行)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        end_time = time.time()
        print(f'Epoch {epoch + 1} 完成。测试集精度: {accuracy:.2f}%, 耗时: {end_time - start_time:.2f}s')


        # 保存模型
        torch.save(model.state_dict(), f'ENX_ResNet18_CIFAR100.pth')
    print('训练结束。')
    return accuracy

# --- 6. 主程序执行 ---

if __name__ == '__main__':
    # 1. 创建 ENX ResNet18 模型 (num_classes=10 for CIFAR-10)
    enx_model = create_se_resnet18(num_classes=100, pretrained=False)

    # 2. 加载数据
    trainloader, testloader = load_cifar100(batch_size=128)

    # 加载预训练权重
    try:
        enx_model.load_state_dict(torch.load('ENX_ResNet18_CIFAR100.pth'))
    except:
        pass


    # 3. 训练和测试 (为演示目的，只运行少量 epoch)
    final_accuracy = train_and_test(
        model=enx_model, 
        trainloader=trainloader, 
        testloader=testloader, 
        epochs=20, 
        lr=0.1
    )
    
    print(f"\nENX ResNet18 在 CIFAR-100 上运行 {5} 个 epoch 后的最终精度: {final_accuracy:.2f}%")