from typing import List
import torch
import torch.nn as nn



class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class Shuffle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = x.shape
        x = x.transpose(-1, -2).contiguous().view(shape).contiguous()
        return x


class SortAttn(nn.Module):
    def __init__(self, hidden_size, groups=4):
        super(SortAttn, self).__init__()
        group_hidden = hidden_size * 2 // groups
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Flatten(1), nn.Unflatten(1, (groups, -1)),  # 重新分组
            Shuffle(),  # shuffle
            nn.Linear(group_hidden, group_hidden, bias=False),
            Shuffle(),  # shuffle
            nn.Linear(group_hidden, group_hidden, bias=False),
            nn.ReLU(),  # 非线性激活
            nn.Flatten(1),  # 合组
            nn.Linear(group_hidden * groups, hidden_size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 对隐藏层进行通道排序
        sorted_x, indices = torch.sort(x, dim=1, stable=True)
        # 链接排序通道
        se_hidden = torch.stack((x, sorted_x), dim=1)  # b, 2, d
        # 生成注意力分数
        attention_scores = self.attention(se_hidden)  # b,d
        # 叠加注意力结果
        return x * attention_scores

 

class SortGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SortGRUCell, self).__init__() 
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.sort_attn = SortAttn(hidden_size)

    def forward(self, x, hidden: torch.Tensor):
        new_hidden = self.gru_cell(x, hidden)  # b,d
        new_hidden = self.sort_attn(new_hidden)  # b,d
        return new_hidden


class SortGRUEncoder(nn.Module):
    def __init__(self, windows, in_channel, out_channel, num_layers):
        super(SortGRUEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = out_channel
        self.win = windows

        self.gru_cells = nn.ModuleList([SortGRUCell(in_channel*windows if i == 0 else out_channel,
                                                    out_channel,)
                                        for i in range(num_layers)])

        self.hidden_0 = nn.Parameter(torch.zeros(num_layers, 1, out_channel), requires_grad=True)
        self._reset_mem()

    def _reset_mem(self):
        self.hidden_mem = torch.zeros(1)

    def patch_window(self, x):
        b, l, c = x.shape
        return x.reshape(b, l // self.win, self.win * c)  # b,l,w*c

    def forward(self, x):
        x = self.patch_window(x)
        batch_size, seq_len, _ = x.size()
        hiddens: List[torch.Tensor] = []
        if self.hidden_mem.ndim == 1:  # 初始化隐藏层
            self.hidden_mem = self.hidden_0.to(x.device) * torch.ones(1, batch_size, 1).to(x.device)
        hiddens.append(self.hidden_mem)
        outputs = []
        for t in range(seq_len):  # 遍历长度
            inp = x[:, t]
            hidden_t: List[torch.Tensor] = [torch.zeros(0) for _ in range(self.num_layers)]
            for layer, gru_cell in enumerate(self.gru_cells):  # 遍历层
                hidden_t[layer] = gru_cell(inp, hiddens[-1][layer])
                inp = hidden_t[layer]
            hidden_t = torch.stack(hidden_t, dim=0)
            hiddens.append(hidden_t)
            outputs.append(inp)
        self.hidden_mem = hiddens[-1].detach()
        return torch.stack(outputs, dim=1)


class SeqDecoder(nn.Sequential):
    def __init__(self, window, in_channel=64, out_channel=4, groups=8, POOL=False):
        super().__init__(
            nn.Unflatten(2, (groups, -1)),  # 分组
            nn.Linear(in_channel // groups, out_channel),  # 分组线性全连接
            Shuffle(),  # shuffle
            nn.Linear(out_channel, out_channel, ),
            Shuffle(),
            nn.Linear(out_channel, out_channel, ),
            nn.Flatten(2), nn.Unflatten(2, (out_channel, groups)),  # 重组
            nn.Linear(groups, window,),
            Transpose(-1, -2),
            nn.Flatten(1, 2),
        )
        self.POOL = POOL
        self._reset_mem()

    def _reset_mem(self):
        self.mem = torch.zeros(1)

    def forward(self, x):
        result = super().forward(x)
        if self.mem.ndim == 1:
            self.mem = torch.zeros_like(result).to(result.device)
        if self.POOL:  # 平滑seg结果
            pool_result = torch.cat((self.mem[:, -10:], result), dim=1)
            pool_result = torch.avg_pool1d(pool_result.transpose(-1, -2), 11, 1).transpose(-1, -2)
        else:
            pool_result = result
        self.mem = result.detach()
        return pool_result


class OneDecoder(nn.Sequential):
    def __init__(self, channel=4, hidden=64, groups=8):
        super().__init__(
            nn.Unflatten(1, (groups, -1)),  # 分组 ,b,d -> b,g,d//g
            nn.Linear(hidden // groups, channel),
            Shuffle(),  # shuffle
            nn.Linear(channel, channel,),
            Shuffle(),
            nn.Linear(channel, channel,),
            nn.ReLU(),
            nn.Flatten(1),  # 合组
            nn.Linear(channel * groups, channel,),
            nn.Sigmoid(),
        )
        self._reset_mem()

    def _reset_mem(self):
        self.mem = torch.zeros(1)

    def forward(self, x):
        b, l, _ = x.shape
        result = super().forward(x.flatten(0, 1))

        if self.mem.ndim == 1:
            self.mem = torch.zeros(b, 0, result.shape[-1]).to(result.device)

        result = result.unflatten(0, (b, l))
        result = torch.cat([result, self.mem], dim=1)[:, -10:]
        self.mem = result.detach()
        return result.mean(dim=1)


class NoneDecoder(nn.Identity):
    def _reset_mem(self):
        self.mem = torch.zeros(1)

    def forward(self,x, *args):
        return torch.zeros(0).to(x.device)

# SortGRU 
class SortGRU(nn.Module):
    def __init__(self,
                 window,
                 in_ch,
                 enc_ch,
                 out_ch_seq=4,
                 out_ch_one=4,
                 dec_groups=8,
                 n_layers=8,):
        super().__init__()
        self.win = window
        self.encoder = SortGRUEncoder(window, in_ch, enc_ch, n_layers)
        self.seq_decoder = SeqDecoder(window, enc_ch,  out_ch_seq, dec_groups) if out_ch_seq > 0 else NoneDecoder()
        self.one_decoder = OneDecoder(out_ch_one, enc_ch, dec_groups) if out_ch_one > 0 else NoneDecoder()

        self._reset_mem()
        self.use_mem(False)

    def use_mem(self, state=True):
        self.USEMEM = state

    def _reset_mem(self):
        self.encoder._reset_mem()
        self.seq_decoder._reset_mem()
        self.one_decoder._reset_mem()

    def forward(self, x):
        enc_out = self.encoder(x)  # b,l,w*64
        y_seq = self.seq_decoder(enc_out)    # b,l,w,4
        y_one = self.one_decoder(enc_out)    # b,l,4

        if not self.USEMEM:  # 如果不用缓存，则清除缓存内容
            self._reset_mem()

        return enc_out, y_seq, y_one


if __name__ == '__main__':
    model = SortGRU(window=24,  # Pach的窗口大小（长度/window)
                    in_ch=4,  # 输入通道数
                    enc_ch=512,  # 中间特征数（backbone的输出通道数）
                    out_ch_seq=20,  # 输出通道数（序列的，长度和输入特征相等）
                    out_ch_one=102,  # 输出通道数（单个的，无长度信息）
                    dec_groups=16,  # 输出特征分组数
                    n_layers=8,  # backbone的层数
                    ).cuda()
    model.use_mem(True)  # 启用缓存机制

    print(model)
    for i in range(2):
        x = torch.randn(1, 24 * 10, 4).cuda()
        enc_out, y_seq, y_one = model(x)
        print(x.shape, enc_out.shape, y_seq.shape, y_one.shape)
