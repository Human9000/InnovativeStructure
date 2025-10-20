import torch
from torch import nn
from torch.nn import functional as F
from maskconv import MaskConv2d


class MOERouter(nn.Module):
    def __init__(self, in_channel, experts_number, top_k):
        super().__init__()
        self.gate = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.top_k = top_k
        self.experts_number = experts_number

    def forward(self, x):  # b c w h
        router_logits = self.gate(x)  # b c w h
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)  # b c w h
        router_weights, select_expert_indices = torch.topk(router_probs, self.top_k, dim=1)  # b top_k w h , b top_k w h
        router_weights = router_weights / router_weights.sum(dim=1, keepdim=True)  # b top_k w h
        router_weights = router_weights.to(x.dtype)  # b top_k w h

        expert_mask = F.one_hot(
            select_expert_indices,
            num_classes=self.experts_number,
        )  # (b, top_k, w, h, e)

        expert_mask = expert_mask.permute(4, 1, 0, 2, 3).flatten(2)  # (e, top_k, b , w , h)

        return router_logits, router_weights, select_expert_indices, expert_mask
        # router_logits shape is (b, c, w, h)
        # router_weights shape is (b, top_k, w, h)
        # select_expert_indices shape is (b, top_k, w, h)
        # expert_mask shape is (e, top_k, b, w , h)


class SparseMOEConv(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, in_channel, out_channel, experts_number, kernal, top_k):
        super().__init__()
        self.experts = nn.ModuleList(MaskConv2d(in_channel, out_channel, kernal) for _ in range(self.experts_number))
        self.router = MOERouter(in_channel, experts_number, top_k)

    def forward(self, x):
        router_logits, router_weights, _, expert_mask = self.router(x)
        final_hidden_states = 0  # shape 是 (b, c, w, h)
        for e in range(self.experts_number):  # 遍历所有专家
            for t in range(self.top_k):  # 遍历选中的 topk 个 专家
                current_mask = expert_mask[e, t][:, None]  # 获取当前专家的掩码 [b 1 w h]
                current_weight = router_weights[:, t][:, None]  # 获取当前专家的权重 [b 1 w h]
                expert_layer = self.experts[e]
                current_hidden_states = expert_layer(x, current_mask) * current_weight  # 获取当前专家的输入数据
                final_hidden_states = final_hidden_states + current_hidden_states  # 将当前专家的输出数据保存到 final_hidden_states 中

        return final_hidden_states, router_logits  # shape 是 (b * s, experts_number)


class SharedExpertMOEConv(nn.Module):
    def __init__(self, in_channel, out_channel, experts_number, kernal, top_k, shared_expert_number):
        super().__init__()
        assert top_k > 0, "top_k must be greater than 0"
        assert experts_number > shared_expert_number, "experts_number must be greater than shared_expert_number"
        assert top_k <= experts_number - shared_expert_number, "top_k must be less than experts_number - shared_expert_number"

        self.shared_experts = nn.ModuleList(Conv2d(in_channel, out_channel, kernal) for _ in range(shared_expert_number))
        self.sparse_moe_conv = SparseMOEConv(in_channel, out_channel, experts_number - shared_expert_number, kernal, top_k)

    def forward(self, x):
        shared_expert_hidden_states = 0
        for e in range(self.shared_expert_number):
            shared_expert_hidden_states = shared_expert_hidden_states + self.shared_experts[e](x)
        sparse_moe_hidden_states, router_logits = self.sparse_moe_conv(x)
        y = shared_expert_hidden_states + sparse_moe_hidden_states
        return y, router_logits