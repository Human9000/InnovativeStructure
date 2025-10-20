from typing import Optional, Type, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Type, Dict, Any


# --- 辅助模块：实现自动梯度注入 (保持不变) ---

class _LossAutoInjector(torch.autograd.Function):
    @staticmethod
    def forward(ctx, main_output: Tensor, *aux_losses: Tensor) -> Tensor:
        ctx.save_for_backward(*aux_losses)
        return main_output

    @staticmethod
    def backward(ctx, main_grad: Tensor) -> tuple:
        aux_losses = ctx.saved_tensors
        if main_grad.requires_grad:
            total_aux_loss = torch.tensor(0.0, device=main_grad.device, dtype=main_grad.dtype)
            for loss in aux_losses:
                total_aux_loss += loss
            total_aux_loss.backward()
        return (main_grad,) + (None,) * len(aux_losses)


class MoeLayer(nn.Module):
    """
    一个通用的、可复用的混合专家（MoE）层。

    该版本允许用户传入自定义的专家和路由器【类】进行实例化，而不是传入实例化的对象。
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_expert: int = 10,
                 top_k: int = 2,
                 n_shared_expert: int = 1,
                 noisy_gating: bool = True,
                 capacity_factor: float = 1.25,
                 drop_tokens: bool = True,
                 load_balancing_alpha: float = 1e-2,
                 # --- 新增参数：允许传入类和其构造参数 ---
                 expert_class: Optional[Type[nn.Module]] = None,
                 expert_args: Optional[Dict[str, Any]] = None,
                 router_class: Optional[Type[nn.Module]] = None,
                 router_args: Optional[Dict[str, Any]] = None):
        """
        初始化 GeneralMoeLayer。

        Args:
            input_dim (int): 输入特征维度。
            output_dim (int): 输出特征维度。
            n_expert (int): 专家总数 (共享 + 稀疏)。
            top_k (int): 每个 token 路由到的稀疏专家数量。
            n_shared_expert (int): 共享专家的数量。
            noisy_gating (bool): 是否在训练时为门控添加噪声。
            capacity_factor (float): 容量因子。
            drop_tokens (bool): 是否在训练时丢弃超出容量的 token。
            load_balancing_alpha (float): 负载均衡损失的系数。
            expert_class (Optional[Type[nn.Module]]): 自定义专家的【类】。
                它的 __init__ 方法应接受 input_dim 和 output_dim。
            expert_args (Optional[Dict]): 传递给 expert_class 构造函数的额外参数。
            router_class (Optional[Type[nn.Module]]): 自定义路由器的【类】。
                它的 __init__ 方法应接受 input_dim 和 num_experts。
            router_args (Optional[Dict]): 传递给 router_class 构造函数的额外参数。
        """
        super().__init__()

        # --- 基本参数设置 ---
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_expert = n_expert
        self.top_k = top_k
        self.n_shared_expert = n_shared_expert
        self.num_sparse_experts = n_expert - n_shared_expert
        self.noisy_gating = noisy_gating
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.load_balancing_alpha = load_balancing_alpha

        # --- 参数校验 ---
        assert n_shared_expert >= 1, "n_shared_expert 必须大于等于1"
        assert self.num_sparse_experts >= 1, "n_expert - n_shared_expert 必须大于等于1"
        assert top_k <= self.num_sparse_experts, "top_k 不能超过稀疏专家的数量"

        # --- 网络模块定义 (工厂模式) ---

        # 1. 定义专家网络 (Experts)
        expert_kwargs = expert_args or {}
        if expert_class is None:
            # 默认情况：使用简单的线性层
            self.experts = nn.ModuleList([
                nn.Linear(input_dim, output_dim, bias=True) for _ in range(n_expert)
            ])
        else:
            # 自定义情况：使用传入的类进行实例化
            assert issubclass(expert_class, nn.Module), "expert_class 必须是 nn.Module 的子类"
            final_expert_kwargs = {'input_dim': input_dim, 'output_dim': output_dim, **expert_kwargs}
            self.experts = nn.ModuleList([
                expert_class(**final_expert_kwargs) for _ in range(n_expert)
            ])

        # 2. 定义路由网络 (Router)
        router_kwargs = router_args or {}
        if router_class is None:
            # 默认情况：使用简单的线性层
            self.router = nn.Linear(input_dim, self.num_sparse_experts)
        else:
            # 自定义情况：使用传入的类进行实例化
            assert issubclass(router_class, nn.Module), "router_class 必须是 nn.Module 的子类"
            final_router_kwargs = {'input_dim': input_dim, 'num_experts': self.num_sparse_experts, **router_kwargs}
            self.router = router_class(**final_router_kwargs)

        # 3. 定义噪声层 (可选)
        if self.noisy_gating:
            self.noise_layer = nn.Linear(input_dim, self.num_sparse_experts)

    # _router_forward, _calculate_aux_losses, 和 forward 方法保持不变
    # ... (此处省略与上一版本完全相同的代码) ...
    def _router_forward(self, x: Tensor) -> tuple:
        """执行路由逻辑，为每个 token 从【稀疏专家】中选择并计算权重。"""
        num_tokens, _ = x.shape
        routing_logits = self.router(x)
        if self.noisy_gating and self.training:
            noise = torch.randn_like(routing_logits) * F.softplus(self.noise_layer(x))
            routing_logits += noise
        routing_weights = F.softmax(routing_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        expert_token_counts = None
        if self.drop_tokens and self.training:
            capacity = int(self.capacity_factor * num_tokens / self.num_sparse_experts)
            expert_token_counts = torch.bincount(topk_indices[:, 0], minlength=self.num_sparse_experts)
            position_in_expert = torch.zeros_like(topk_indices[:, 0])
            sorter = torch.argsort(topk_indices[:, 0])
            bincounts = torch.bincount(topk_indices[:, 0], minlength=self.num_sparse_experts)
            starts = torch.zeros_like(bincounts)
            starts[1:] = bincounts.cumsum(0)[:-1]
            position_in_expert[sorter] = torch.arange(num_tokens, device=x.device) - starts.gather(0, topk_indices[:, 0][sorter])
            keep_mask = (position_in_expert < capacity)
            topk_weights = topk_weights * keep_mask.unsqueeze(-1)
        norm_den = topk_weights.sum(dim=1, keepdim=True)
        topk_weights_normalized = topk_weights / (norm_den + 1e-6)
        return topk_weights_normalized, topk_indices, routing_weights, expert_token_counts

    def _calculate_aux_losses(self, num_tokens: int, routing_weights: Tensor, expert_token_counts: Tensor) -> list:
        """计算辅助损失。"""
        aux_loss_list = []
        if not self.training:
            return aux_loss_list
        if self.load_balancing_alpha > 0:
            mean_prob = routing_weights.mean(dim=0)
            token_frac = routing_weights.sum(dim=0) / num_tokens
            load_balancing_loss = self.load_balancing_alpha * self.num_sparse_experts * (mean_prob * token_frac).sum()
            aux_loss_list.append(load_balancing_loss)
        if self.drop_tokens and expert_token_counts is not None:
            capacity = int(self.capacity_factor * num_tokens / self.num_sparse_experts)
            overflow_tokens = (expert_token_counts - capacity).clamp(min=0)
            fraction_overflowed = overflow_tokens.sum() / (num_tokens + 1e-6)
            aux_loss_list.append(fraction_overflowed)
        return aux_loss_list

    def forward(self, x: Tensor) -> Tensor:
        """对一批 token 执行 MoE 前向传播。"""
        num_tokens = x.shape[0]
        y = torch.zeros(num_tokens, self.output_dim, device=x.device, dtype=x.dtype)

        # --- 1. 共享专家计算 (密集路径) ---
        for e in range(self.n_shared_expert):
            # 前 n_shared_expert 个专家是共享的，对所有 token 计算
            y = y + self.experts[e](x)
        y /= self.n_shared_expert  # 平均贡献

        # --- 2. 稀疏专家计算 (稀疏路径) ---
        # s: (N, top_k), (N, top_k), (N, num_sparse_experts), (num_sparse_experts,)
        topk_weights, topk_indices, routing_weights, expert_token_counts = self._router_forward(x)

        for i in range(self.num_sparse_experts):
            expert_mask = (topk_indices == i).any(dim=1)  # : (N,)
            if not expert_mask.any(): continue

            token_indices = expert_mask.nonzero(as_tuple=True)[0]  # : (num_tokens_i,)
            topk_pos_indices = (topk_indices[token_indices] == i).nonzero(as_tuple=True)[1]  # : (num_tokens_i,)

            current_w = topk_weights[token_indices, topk_pos_indices]  # : (num_tokens_i,)
            active_mask = current_w > 0  # : (num_tokens_i,)
            if not active_mask.any(): continue

            current_indices = token_indices[active_mask]  # : (num_active_i,)
            current_x = x[current_indices]  # : (num_active_i, C_in)
            current_w = current_w[active_mask].unsqueeze(-1)  # : (num_active_i, 1) 
            current_experts = self.experts[i + self.n_shared_expert]
            current_y = current_experts(current_x) * current_w  # : (num_active_i, C_out) 
            y.index_add_(0, current_indices, current_y)

        aux_losses = self._calculate_aux_losses(num_tokens, routing_weights, expert_token_counts)

        if aux_losses:
            return _LossAutoInjector.apply(y, *aux_losses)

        return y