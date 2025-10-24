import torch
import torch.nn as nn
import torch.nn.functional as F

class RFS(nn.Module): # RFS-V2 (Radiance Field Solver - Source/Sink)
    """
    辐射场求解器 RFS-V2 模块，引入了负激活“黑洞/吸收子”机制，
    独立计算正辐射场 L_pos 和负吸收场 L_neg，增强表达能力。
    专为 2D 图像特征 (4D Tensor: B, C, H, W) 设计。
    """
    def __init__(self, channels: int,
                 iterations: int = 8,
                 gamma: float = 0.99,            # 能量衰减因子
                 temperature: float = 100.0,     # Sigmoid 锐化温度
                 threshold_scale: float = 2.0,   # 控制光源/吸收源严格程度 (sigma 倍数)
                 min_source_strength: float = 1e-6 # 鲁棒性保证：光源/吸收源的最小强度
                 ):
        super(RFS, self).__init__() 
        self.iterations = iterations
        self.base_gamma = gamma
        self.temperature = temperature
        self.threshold_scale = threshold_scale
        self.min_source_strength = min_source_strength
        
        # MaxPool2d 模拟局部辐射和非线性积累 (针对 2D)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 辐射接收器调制器 (Radiance Receptor Modulator)
        # 注意：现在输入是 L_pos - L_neg 的结果，依然是 1x1 Conv
        self.receptor_modulator = nn.Sequential( 
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid(),
        )

    def _get_activation_mask(self, magnitude_data: torch.Tensor) -> torch.Tensor:
        """
        计算激活遮罩 (Source 或 Sink)，基于激活幅度的统计特性。
        """
        B, C, H, W = magnitude_data.shape
        
        # 展平空间维度，计算每个通道的均值和标准差
        flat_data = magnitude_data.view(B, C, -1)
        mean_val = flat_data.mean(dim=-1, keepdim=True).detach()
        std_val = flat_data.std(dim=-1, keepdim=True).detach()
        
        # 阈值 T = Mean + Sigma * Scale
        threshold_t = mean_val + self.threshold_scale * std_val 
        
        # 广播 T 到 (B, C, H, W)
        threshold_to_broadcast = threshold_t.view(B, C, 1, 1) 
        
        # Sigmoid 锐化，决定哪些区域是活跃的 Source/Sink
        M_soft = torch.sigmoid(self.temperature * (magnitude_data - threshold_to_broadcast)) 
        
        # 鲁棒性保证：添加一个微小的绝对下限 
        M_soft = M_soft + self.min_source_strength
        
        return M_soft

    def _propagate_field(self, initial_field: torch.Tensor, orig_size: tuple) -> torch.Tensor:
        """
        核心传播函数：使用多尺度 Max 融合积累场能量。
        """
        radiance_current = initial_field.clone()
        radiance_down_result = initial_field.clone() 
        stride = 1 

        # 第一次传播 (i=0): 局部辐射和基础衰减 γ^1
        radiance_pooled = self.max_pool(radiance_down_result) 
        radiance_pooled_decayed = radiance_pooled * self.base_gamma
        radiance_current = torch.max(radiance_current, radiance_pooled_decayed)
        radiance_down_result = radiance_pooled_decayed 

        # 后续多尺度传播 (i >= 1): 模拟辐射场的指数衰减传输 (γ^(2^i))
        for i in range(1, self.iterations):
            stride *= 2
            gamma_i = self.base_gamma ** stride

            # 降采样
            radiance_down_result = F.interpolate(radiance_down_result, scale_factor=0.5, mode='bilinear', align_corners=True)
            radiance_pooled = self.max_pool(radiance_down_result) 
            radiance_pooled_decayed = radiance_pooled * gamma_i
            
            # 上采样回原尺寸
            radiance_up = F.interpolate(radiance_pooled_decayed, size=orig_size, mode='bilinear', align_corners=True)
            
            # 辐射场的非线性叠加（Max 融合）
            radiance_current = torch.max(radiance_current, radiance_up)
            radiance_down_result = radiance_pooled_decayed 

        return radiance_current


    def solve_radiance_field(self, x):
        """
        求解 RFS-V2 辐射场：双场（Source/Sink）独立传播与合成。
        """
        # 1. 特征分解：分离正值和负值幅度
        x_pos = F.relu(x) # 正特征幅度 (Source Magnitude)
        x_neg_mag = F.relu(-x) # 负特征幅度 (Sink Magnitude)
        
        orig_size = x.shape[2:] 

        # 2. 生成双重遮罩
        M_source = self._get_activation_mask(x_pos) # 决定哪些是活跃光源
        M_sink = self._get_activation_mask(x_neg_mag)   # 决定哪些是活跃吸收子

        # 3. 初始化双重辐射场 L_pos 和 L_neg
        # L_pos 初始场 = 正特征 * 光源遮罩
        L_pos_init = x_pos * M_source 
        # L_neg 初始场 = 负特征幅度 * 吸收子遮罩 (代表吸收的强度)
        L_neg_init = x_neg_mag * M_sink 

        # 4. 独立传播和积累
        L_pos = self._propagate_field(L_pos_init, orig_size)
        L_neg = self._propagate_field(L_neg_init, orig_size)
        
        # 5. 场合成：最终辐射场 = 正辐射场 - 负吸收场
        final_radiance_field = L_pos - L_neg

        return final_radiance_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 求解 RFS-V2 辐射场 L
        radiance_field = self.solve_radiance_field(x)
        
        # 2. 特征接收器 R 响应并调制原始特征 x
        # Output = R * Modulator(L)
        return x * self.receptor_modulator(radiance_field)
