import torch
import torch.nn as nn


class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.shape
        x = x.transpose(-1, -2).contiguous().view(shape)
        return x


class SortAttn(nn.Module):
    def __init__(self, hidden_size, group=4):
        super(SortAttn, self).__init__()
        group_hidden = hidden_size * 2 // group
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Flatten(1), nn.Unflatten(1, (group, -1)),  # 重新分组
            Shuffle(),  # shuffle
            nn.Linear(group_hidden, group_hidden, bias=False),
            Shuffle(),  # shuffle
            nn.Linear(group_hidden, group_hidden, bias=False),
            nn.ReLU(),  # 非线性激活
            nn.Flatten(1),  # 合组
            nn.Linear(group_hidden * group, hidden_size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对隐藏层进行通道排序
        sorted_x, indices = torch.sort(x, dim=1, stable=True)
        # 残差链接排序通道
        se_hidden = torch.stack((x, sorted_x), dim=1)  # b, 2, d
        # 生成注意力分数
        attention_scores = self.attention(se_hidden)  # b,d
        # 叠加注意力结果
        return x * attention_scores


class SortGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SortGRUCell, self).__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.attn = SortAttn(hidden_size)

    def forward(self, x, hidden):
        # print(x.shape, hidden.shape)
        new_hidden = self.gru_cell(x, hidden)  # b,d
        new_hidden = self.attn(new_hidden)  # b,d
        return new_hidden


class SortGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SortGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru_cells = nn.ModuleList([SortGRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.init_hidden = nn.Parameter(torch.ones(num_layers, 1, hidden_size), requires_grad=False)

    def forward(self, x, hidden=torch.zeros(1)):
        batch_size, seq_len, _ = x.size()
        hidden = hidden.to(x.device) * \
                 torch.ones(1, batch_size, 1).to(x.device) * \
                 self.init_hidden.to(x.device)

        outputs = []
        for t in range(seq_len):
            inp = x[:, t]
            for l, gru_cell in enumerate(self.gru_cells):
                hidden[l] = gru_cell(inp, hidden[l])
                inp = hidden[l]
            outputs.append(inp)
        return torch.stack(outputs, dim=1), hidden


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Log(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class StreamingECG(nn.Module):
    def __init__(self, in_channel, window, seg_number=4, cls_number=4, hidden=64, num_layers=8):
        super().__init__()
        self.win = window
        # num_layers = 8
        self.gru = SortGRU(input_size=in_channel * window,
                           hidden_size=hidden,
                           num_layers=num_layers, )
        groups = 8
        self.seg_head = nn.Sequential(
            nn.Unflatten(2, (groups, -1)),  # 分组
            nn.Linear(hidden // groups, seg_number, bias=False),
            Shuffle(),
            nn.Linear(seg_number, seg_number, bias=False),
            Shuffle(),
            nn.Linear(seg_number, seg_number, bias=False),
            nn.Flatten(2), nn.Unflatten(2, (seg_number, groups)),  # 重组
            nn.Linear(groups, window, bias=False),
            Transpose(-1, -2),
            nn.Flatten(1, 2),
        )

        self.cls_head = nn.Sequential(
            Transpose(0, 1),  # l,b,d -> b,l,d
            nn.Flatten(1, ),  # b,ld
            nn.Unflatten(1, (groups, -1)),  # 分组 ,(b,g,ld//g)
            nn.Linear(hidden * num_layers // groups, cls_number, bias=False),
            Shuffle(),  # shuffle
            nn.Linear(cls_number, cls_number, bias=False),
            Shuffle(),
            nn.Linear(cls_number, cls_number, bias=False),
            nn.ReLU(),
            nn.Flatten(1),  # 合组
            nn.Linear(cls_number * groups, cls_number, bias=False),
        )

    def forward(self, x, h0=torch.zeros(1)):
        b, l, c = x.shape
        x = x.reshape(b, l // self.win, self.win * c)
        ys, h = self.gru(x, h0)  # (b,l,w*64) (l,b,64)

        seg = self.seg_head(ys)  # b,l,w,4
        cls = self.cls_head(h)  # b,l,4
        return (seg, cls), (ys, h)


if __name__ == '__main__':
    model = StreamingECG(in_channel=12,
                         window=100,
                         seg_number=4,
                         cls_number=4,
                         hidden=96,
                         num_layers=16,
                         ).cuda()  # 12个通道，每个通道有5个特征
    batch_size = 100
    x = torch.randn(batch_size, 500 * 15, 12).cuda()  # 15s的数据, 采样率 500 hz
    print(x.shape)
    (seg, cls), (ys, hidden) = model(x)
    print(seg.shape, cls.shape, ys.shape, hidden.shape)
    from ptflops import get_model_complexity_info

    res = get_model_complexity_info(model, (500 * 15, 12), as_strings=True)
    print(res)
