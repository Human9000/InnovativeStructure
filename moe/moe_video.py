import torch
from torch import nn
from torch.nn import functional as F


class BasicExpert(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicExpert, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)


class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, num_experts):
        super().__init__()
        self.gate = nn.Linear(feature_in, num_experts)
        self.experts = nn.ModuleList(
            BasicExpert(
                feature_in, feature_out
            ) for _ in range(num_experts)
        )

    def forward(self, x):  # (512, 128)
        # (b, num_experts, feature_out)
        expert_output = torch.concat([expert(x).unsqueeze(1) for expert in self.experts], dim=1)

        # (b, 1, num_experts,)
        expert_weights = F.softmax(self.gate(x), dim=1).unsqueeze(1)

        #  (b, 1, n) @ (b, n, fo) -> (b, 1, fo)
        output = expert_weights @ expert_output

        return output.squeeze(1)


def test_basic_moe():
    x = torch.rand(4, 512)
    basic_moe = BasicMOE(512, 128, 4)
    print(basic_moe(x).shape)


class MOEConfig:
    def __init__(self,
                 hidden_dim,
                 experts_number,
                 top_k,
                 shared_experts_number=2, ):
        self.hidden_dim = hidden_dim
        self.experts_number = experts_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number


class MOERouter(nn.Module):
    def __init__(self, config: MOEConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_dim, config.experts_number)
        # 但是后面只会选top_k个专家
        self.top_k = config.top_k
        self.experts_number = config.experts_number

    def forward(self, x):
        router_logits = self.gate(x)

        # 计算每一个专家的概率
        router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 计算 top_k 的专家的输出
        # top_k 是可以反向传播的
        router_weights, select_expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # shape is ( batch * seq_len , top_k)

        # 重新归一化
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        router_weights = router_weights.to(x.dtype)

        expert_mask = F.one_hot(
            select_expert_indices,
            num_classes=self.experts_number
        )  # (batch*seq_len, top_k, expert_number)

        expert_mask = expert_mask.permute(2, 1, 0)  # ( experts_number, top_k, batch * seq_len)

        return router_logits, router_weights, select_expert_indices, expert_mask
        # router_weights shape is ( batch * seq_len , top_k)
        # select_expert_indices shape is ( batch * seq_len , top_k)
        # expert_mask shape is ( experts_number, top_k, batch * seq_len)
        # router_logits shape is ( batch * seq_len , experts_number)


class SparseMOE(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.experts_number = config.experts_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.experts_number)
            ]
        )

        self.router = MOERouter(config)

    # 是为了
    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # 合并前两个维度，因为不是 Sample 维度了，而是 token 维度
        hidden_states = x.view(-1, hidden_dim)  # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (experts_number, top_k, b * s)

        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in range(self.experts_number):  # 遍历所有专家
            topk_idx, bs_idx = torch.where(expert_mask[expert_idx])  # 获取当前专家被选中的 topk 维度索引 和 bs 维度索引
            expert_layer = self.experts[expert_idx]  # 获取当前专家层
            current_state = hidden_states[bs_idx, :]  # 获取当前专家的输入数据 [-1, hidden_dim]
            current_weight = router_weights[bs_idx, topk_idx, None]  # 获取当前专家的权重 [-1,1]
            current_hidden_states = expert_layer(current_state) * current_weight  # 获取当前专家的输入数据
            final_hidden_states.index_add_(0, bs_idx, current_hidden_states.to(hidden_states.dtype))

        # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits  # shape 是 (b * s, experts_number)



class SparseMOE(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.experts_number = config.experts_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.experts_number)
            ]
        )

        self.router = MOERouter(config)

    # 是为了
    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # 合并前两个维度，因为不是 Sample 维度了，而是 token 维度
        hidden_states = x.view(-1, hidden_dim)  # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (experts_number, top_k, b * s)

        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in range(self.experts_number):  # 遍历所有专家
            topk_idx, bs_idx = torch.where(expert_mask[expert_idx])  # 获取当前专家被选中的 topk 维度索引 和 bs 维度索引
            expert_layer = self.experts[expert_idx]  # 获取当前专家层
            current_state = hidden_states[bs_idx, :]  # 获取当前专家的输入数据 [-1, hidden_dim]
            current_weight = router_weights[bs_idx, topk_idx, None]  # 获取当前专家的权重 [-1,1]
            current_hidden_states = expert_layer(current_state) * current_weight  # 获取当前专家的输入数据
            final_hidden_states.index_add_(0, bs_idx, current_hidden_states.to(hidden_states.dtype))

        # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)

        return final_hidden_states, router_logits  # shape 是 (b * s, experts_number)

def test_token_level_moe():
    x = torch.rand(2, 18, 16)
    config = MOEConfig(16, 4, 2)  # topk=2，experts_number=4
    token_level_moe = SparseMOE(config)
    out = token_level_moe(x)
    print(out[0].shape, out[1].shape)


if __name__ == '__main__':
    # test_basic_moe()
    test_token_level_moe()
