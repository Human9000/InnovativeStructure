'''
Author       : Hao Liu 33023091+Human9000@users.noreply.github.com
Date         : 2025-10-20 09:31:38
LastEditors  : Hao Liu 33023091+Human9000@users.noreply.github.com
LastEditTime : 2025-10-24 16:28:36
FilePath     : \InnovativeStructure\RFS\rfs_v1.py
Description  : 
Copyright (c) 2025 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import SqueezeExcite
from timm.models.registry import register_model


class RFS(nn.Module):
    def __init__(self, channels, iterations=6, gamma=0.99 , q=0.95, min_source_strength=1e-6):
        super().__init__()
        self.iterations = iterations
        self.base_gamma = gamma 
        self.q = q
        self.min_source_strength = min_source_strength
        self.max_pool0 = nn.MaxPool2d(3, 1, 1)
        self.max_pool = nn.MaxPool2d(5, 2, 2)
        mid_channels = int(channels ** 0.5)
        self.receptor_modulator = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, channels, 1),
            nn.Sigmoid()
        )

        # 暂存用于论文可视化的特征图
        self.paper_graph_fetures = {}

    def _get_activation_mask(self, x):
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)   
        threshold = torch.quantile(flat, self.q, dim=-1) .view(B, C, 1, 1) 
        return ((x >= threshold) * 1.0 ).detach()

    def _propagate_field(self, x, size):
        out = x.clone()
        feat = x
        for i in range(self.iterations):
            gamma_i = self.base_gamma ** (2 ** i) 
            if i == 0:
                feat = self.max_pool0(feat) 
                feat = feat * gamma_i 
                out = torch.max(out, feat) 
            else:
                feat = self.max_pool(feat) 
                feat = feat * gamma_i 
                feat = F.interpolate(feat, size=size, mode='nearest')
                out = torch.max(out, feat) 
        return out

    def forward(self, x: torch.Tensor):
        M = self._get_activation_mask(x)
        L = self._propagate_field(x * M, x.shape[2:])
        R: torch.Tensor = self.receptor_modulator(L)
        y = x * R
        # 下面的代码仅用于论文可视化，对模型训练无用
        self.paper_graph_fetures["x"] = x.data.cpu()
        self.paper_graph_fetures["M"] = M.data.cpu()
        self.paper_graph_fetures["L"] = L.data.cpu()
        self.paper_graph_fetures["R"] = R.data.cpu()
        self.paper_graph_fetures["y"] = y.data.cpu()
        return x


def replace_module(parent, name, new_module):
    attrs = name.split('.')
    for attr in attrs[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, attrs[-1], new_module)


class Log(nn.Module):
    def __init__(self, origin_module):
        super().__init__()
        self.origin_module = origin_module

    def forward(self, x):
        print(x.max(), x.min())
        return self.origin_module(x)


class Res(nn.Module):
    def __init__(self, origin_module, res_module):
        super().__init__()
        self.origin_module = origin_module
        self.res_module = res_module

    def forward(self, x):
        # print(x.max(), x.min())
        self.res_module(x)
        return self.origin_module(x)


@register_model
def rfs_resnet18_test(pretrained=True, **kwargs):
    model = timm.create_model('legacy_seresnet18.in1k', pretrained=pretrained, **kwargs)

    channels = [512, 256, 128, 64]
    iters = [1, 2, 3, 4]
    gamas = [0.9, 0.95, 0.98, 0.99]

    for name, module in model.named_modules():
        if isinstance(module, timm.models.senet.SEModule):
            cin = module.fc1.in_channels
            index = channels.index(cin)
            replace_module(
                model,
                name,
                Res(
                    module,
                    RFS(cin,  iters[index], gamas[index])
                )
            )
    return model


@register_model
def rfs_resnet18(pretrained=True, **kwargs):
    model = timm.create_model('legacy_seresnet18.in1k', pretrained=pretrained, **kwargs)

    channels = [512, 256, 128, 64]
    iters = [3, 4, 5, 6]
    gamas = [0.9, 0.95, 0.98, 0.99]

    for name, module in model.named_modules():
        if isinstance(module, timm.models.senet.SEModule):
            cin = module.fc1.in_channels
            if cin == 64:
                index = channels.index(cin)
                replace_module(
                    model,
                    name,
                    RFS(cin,  iters[index], gamas[index])
                )
    return model

def plot_papper(model): 
    rfs_models = []
    for name, module in model.named_modules():
        if isinstance(module, RFS):
            rfs_models.append(module)

    from matplotlib import pyplot as plt
    for layer_index, rfs in enumerate(rfs_models):
        print(layer_index)
        rfs: RFS
        fets = rfs.paper_graph_fetures 
        x = fets['x']
        y = fets['y']
        M = fets['M']
        L = fets['L']
        R = fets['R']

        print(R.shape)

        plt.figure(figsize=(16, 4))
        j = 0
        for data in [x, M, L, R]: 
            for i in range(16):
                j += 1
                plt.subplot(4, 16, j)
                plt.imshow(data[0, i])
                plt.axis('off')
            plt.tight_layout()
        plt.show()
        break

if __name__ == "__main__": 
    m = timm.create_model("rfs_resnet18", pretrained=True)
    print(m)
    
    import cv2 as cv
    x = cv.imread("RFS\image.png")
    x = cv.resize(x, (224, 224))
    x = torch.from_numpy(x).transpose(0, 2)[None].float() / 255.0
    y = m(x) 
    rfs_models = []
    for name, module in m.named_modules():
        if isinstance(module, RFS):
            rfs_models.append(module)

    from matplotlib import pyplot as plt
    for layer_index, rfs in enumerate(rfs_models):
        print(layer_index)
        rfs: RFS
        fets = rfs.paper_graph_fetures
        
        x = fets['x']
        y = fets['y']
        M = fets['M']
        L = fets['L']
        R = fets['R']

        print(R.shape)

        for data in [x, M, R]:
            plt.figure(figsize=(10, 10))
            for i in range(64):
                plt.subplot(8, 8, i+1)
                plt.imshow(data[0, i])
                plt.axis('off')
            plt.tight_layout()
            plt.show()
