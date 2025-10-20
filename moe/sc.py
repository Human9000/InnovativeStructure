import unittest
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

# 在文件顶部引入 ptflops
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False


# GatedBottleneck 类保持不变
class GatedBottleneck(nn.Module):
    """
    一个极致轻量级的门控网络 (优化版 v2)。
    """
    def __init__(self, in_features: int, out_features: int):
        super(GatedBottleneck, self).__init__()
        assert in_features % 4 == 0, "Input features must be divisible by 4 for this GatedBottleneck design."
        bottleneck_features = in_features // 4
        self.compress = nn.Conv1d(in_features, bottleneck_features, kernel_size=1, groups=bottleneck_features)
        self.norm_act = nn.Sequential(nn.BatchNorm1d(bottleneck_features), nn.GELU())
        self.expand = nn.Linear(bottleneck_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (N, in_features)
        x_3d = x.unsqueeze(-1) # (N, in_features, 1)
        x_compressed = self.compress(x_3d) # (N, bottleneck, 1)
        x_activated = self.norm_act(x_compressed) # (N, bottleneck, 1)
        x_2d = x_activated.squeeze(-1) # (N, bottleneck)
        return self.expand(x_2d) # (N, out_features)

# ChannelSparseConnectionEinsum (用于对比测试) 保持不变
class ChannelSparseConnectionEinsum(nn.Module):
    def __init__(self, c1: int, c2: int, k_in: int, k: int):
        super(ChannelSparseConnectionEinsum, self).__init__()
        assert c1 % 4 == 0, "For optimized gating, in_channel must be divisible by 4."
        self.c1, self.c2 = c1, c2
        self.k_in, self.k = k_in, k
        self.weight = nn.Parameter(torch.randn(c1, c2)) # (c1, c2)
        self.bias = nn.Parameter(torch.randn(c2)) # (c2,)
        self.score_out = GatedBottleneck(c1, c2)
        self.score_in = GatedBottleneck(c1, c1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def _output_sparse_branch(self, x: torch.Tensor) -> torch.Tensor: # x: (b, l, c1)
        b, l, _ = x.shape
        score_out_raw = self.score_out(x.view(b * l, self.c1)).view(b, l, self.c2) # (b, l, c2)
        score_out_norm = F.softmax(score_out_raw, dim=-1) # (b, l, c2)
        s_v_out, s_i_out = torch.topk(score_out_norm, k=self.k, dim=-1) # s_v_out/s_i_out: (b, l, k)
        expanded_weight = self.weight.expand(b, l, -1, -1) # (b, l, c1, c2)
        expanded_s_i_out = s_i_out.unsqueeze(2).expand(-1, -1, self.c1, -1) # (b, l, c1, k)
        selected_weights_out = torch.gather(expanded_weight, dim=3, index=expanded_s_i_out) # (b, l, c1, k)
        output_k = torch.einsum('blc,blck->blk', x, selected_weights_out) # (b, l, k)
        output_k_scaled = output_k * s_v_out # (b, l, k)
        output = torch.zeros(b, l, self.c2, device=x.device, dtype=x.dtype) # (b, l, c2)
        output.scatter_add_(dim=-1, index=s_i_out, src=output_k_scaled) # (b, l, c2)
        return output

    def _input_sparse_branch(self, x: torch.Tensor) -> torch.Tensor: # x: (b, l, c1)
        b, l, _ = x.shape
        score_in_raw = self.score_in(x.view(b * l, self.c1)).view(b, l, self.c1) # (b, l, c1)
        score_in_norm = F.softmax(score_in_raw, dim=-1) # (b, l, c1)
        s_v_in, s_i_in = torch.topk(score_in_norm, k=self.k_in, dim=-1) # s_v_in/s_i_in: (b, l, k_in)
        x_selected = torch.gather(x, -1, s_i_in) # (b, l, k_in)
        x_gated = x_selected * s_v_in # (b, l, k_in)
        selected_weights_in = F.embedding(s_i_in, self.weight) # (b, l, k_in, c2)
        output = torch.einsum('blk,blko->blo', x_gated, selected_weights_in) # (b, l, c2)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (b, l, c1)
        return self._output_sparse_branch(x) + self._input_sparse_branch(x) + self.bias # (b, l, c2)


# SimplifiedChannelSparseConnection 保持不变
class SimplifiedChannelSparseConnection(nn.Module):
    def __init__(self, c1: int, c2: int, k_in: int, k: int):
        super().__init__()
        assert c1 % 4 == 0, "For optimized gating, in_channel must be divisible by 4."
        self.c1, self.c2 = c1, c2
        self.k_in, self.k = k_in, k
        self.weight = nn.Parameter(torch.randn(c1, c2)) # (c1, c2)
        self.bias = nn.Parameter(torch.randn(c2)) # (c2,)
        self.score_out = GatedBottleneck(c1, c2)
        self.score_in = GatedBottleneck(c1, c1)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.bias, 0)

    def _get_gating_scores(self, x: torch.Tensor, gating_network: nn.Module, k: int) -> Tuple[torch.Tensor, torch.Tensor]: # x: (b, l, c)
        b, l, _ = x.shape
        scores_raw = gating_network(x.view(b * l, -1)).view(b, l, -1) # (b, l, num_scores)
        scores_norm = F.softmax(scores_raw, dim=-1) # (b, l, num_scores)
        s_v, s_i = torch.topk(scores_norm, k=k, dim=-1) # s_v/s_i: (b, l, k)
        return s_v, s_i

    def _output_sparse_branch(self, x: torch.Tensor) -> torch.Tensor: # x: (b, l, c1)
        b, l, _ = x.shape
        s_v_out, s_i_out = self._get_gating_scores(x, self.score_out, self.k) # s_v_out/s_i_out: (b, l, k)
        expanded_weight = self.weight.expand(b, l, -1, -1) # (b, l, c1, c2)
        expanded_s_i_out = s_i_out.unsqueeze(2).expand(-1, -1, self.c1, -1) # (b, l, c1, k)
        selected_weights_out = torch.gather(expanded_weight, dim=3, index=expanded_s_i_out) # (b, l, c1, k)
        output_k = torch.einsum('blc,blck->blk', x, selected_weights_out) # (b, l, k)
        output_k_scaled = output_k * s_v_out # (b, l, k)
        output = torch.zeros(b, l, self.c2, device=x.device, dtype=x.dtype) # (b, l, c2)
        output.scatter_add_(dim=-1, index=s_i_out, src=output_k_scaled) # (b, l, c2)
        return output

    def _input_sparse_branch(self, x: torch.Tensor) -> torch.Tensor: # x: (b, l, c1)
        s_v_in, s_i_in = self._get_gating_scores(x, self.score_in, self.k_in) # s_v_in/s_i_in: (b, l, k_in)
        x_selected = torch.gather(x, -1, s_i_in) # (b, l, k_in)
        x_gated = x_selected * s_v_in # (b, l, k_in)
        selected_weights_in = F.embedding(s_i_in, self.weight) # (b, l, k_in, c2)
        output = torch.einsum('blk,blko->blo', x_gated, selected_weights_in) # (b, l, c2)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: (b, l, c1)
        output_A = self._output_sparse_branch(x) # (b, l, c2)
        output_B = self._input_sparse_branch(x) # (b, l, c2)
        return output_A + output_B + self.bias # (b, l, c2)

# ==============================================================================
# ==============================  对比模块  ====================================
# ==============================================================================

class StandardLinear(nn.Module):
    """一个标准的全连接层，用于对比。"""
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.linear = nn.Linear(c1, c2)

    def forward(self, x: torch.Tensor): # x: (b, l, c1)
        return self.linear(x) # (b, l, c2)

# ==============================================================================
# ==============================  测试部分  ====================================
# ==============================================================================

class TestRefactoredCode(unittest.TestCase):
    
    def setUp(self):
        """在每个测试前运行，设置通用参数。"""
        self.batch_size = 2
        self.seq_len = 8
        self.c1 = 16  # in_channel
        self.c2 = 32  # out_channel
        self.k_in = 4 # in_top_k
        self.k = 8    # out_top_k
        
        torch.manual_seed(42)
        self.input_tensor = torch.randn( # (b, l, c1)
            self.batch_size, self.seq_len, self.c1
        )

    def test_simplified_version_shape(self):
        """测试简化版模型的前向传播和输出形状。"""
        print("\n--- Testing Simplified Version Shape ---")
        model = SimplifiedChannelSparseConnection( # nn.Module
            c1=self.c1, c2=self.c2,
            k_in=self.k_in, k=self.k
        )
        model.eval()
        output = model(self.input_tensor) # (b, l, c2)
        expected_shape = (self.batch_size, self.seq_len, self.c2) # tuple
        self.assertEqual(output.shape, expected_shape)
        print("Simplified version test passed: Output shape is correct.")

    def test_equivalence(self):
        """验证重构后的版本与原einsum版本结果是否一致。"""
        print("\n--- Testing Equivalence Between Versions ---")
        model_old = ChannelSparseConnectionEinsum( # nn.Module
            c1=self.c1, c2=self.c2, k_in=self.k_in, k=self.k
        )
        model_new = SimplifiedChannelSparseConnection( # nn.Module
            c1=self.c1, c2=self.c2, k_in=self.k_in, k=self.k
        )
        model_new.load_state_dict(model_old.state_dict())
        model_old.eval()
        model_new.eval()
        output_old = model_old(self.input_tensor) # (b, l, c2)
        output_new = model_new(self.input_tensor) # (b, l, c2)
        are_close = torch.allclose(output_old, output_new, atol=1e-6) # bool
        self.assertTrue(are_close, "Outputs of old and new models are not close enough!")
        print("Equivalence test passed: The simplified version produces the same result.")

    @unittest.skipIf(not PTFLOPS_AVAILABLE, "ptflops is not installed. Skipping complexity test.")
    def test_complexity_comparison(self):
        """使用 ptflops 对比稀疏模块和标准全连接层的参数量与计算量。"""
        print("\n--- Testing Complexity (Params and MACs) ---")
        
        # ptflops 需要输入尺寸，不包含batch维度
        input_res = (self.seq_len, self.c1) # (l, c1)

        # 实例化稀疏模型
        sparse_model = SimplifiedChannelSparseConnection(
            c1=self.c1, c2=self.c2, k_in=self.k_in, k=self.k
        )
        # 实例化标准全连接（密集）模型
        dense_model = StandardLinear(c1=self.c1, c2=self.c2)

        # 计算稀疏模型的复杂度
        macs_sparse, params_sparse = get_model_complexity_info(
            sparse_model, input_res, as_strings=True, print_per_layer_stat=False, verbose=False
        )
        
        # 计算密集模型的复杂度
        macs_dense, params_dense = get_model_complexity_info(
            dense_model, input_res, as_strings=True, print_per_layer_stat=False, verbose=False
        )

        print(f"Test Configuration:")
        print(f"  Input Shape (per item): ({self.seq_len}, {self.c1})")
        print(f"  Output Channels (c2): {self.c2}")
        print(f"  Sparse k_in: {self.k_in}, k_out: {self.k}\n")

        print(f"{'Model':<35} | {'Params':<15} | {'MACs (FLOPs)':<15}")
        print(f"{'-'*35} | {'-'*15} | {'-'*15}")
        print(f"{'Standard Linear (Dense)':<35} | {params_dense:<15} | {macs_dense:<15}")
        print(f"{'SimplifiedChannelSparseConnection':<35} | {params_sparse:<15} | {macs_sparse:<15}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)