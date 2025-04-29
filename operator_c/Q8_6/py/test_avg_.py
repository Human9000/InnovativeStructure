import time

from model import LBitTanh,LBit,Binarize
from model import ECGSegMCULBit
import torch
from torch.nn import functional as F
import numpy as np

state = torch.load("model.pth")

def print_mat(name, mat):
    print("=>",name, mat.shape,":")
    for i in mat:
        for d in i:
            print("%.7f"%d,end='\t')
        print()
    print()

def print_mat3(name, mat):
    print("=>",name, mat.shape,":")
    for i in mat:
        for j in i:
            for d in j:
                print("%.7f"%d,end='\t')
            print()
        print()
    print()


net = ECGSegMCULBit().float()
net.load_state_dict(state)

x = torch.arange(0, 12*1).reshape((1, 1, 12)).float()/12
x0 = x.transpose(1, 2)

x = F.pad(x0,(180, 180)) / 8

# print(x.shape)
x = net.down(x)
# x = net.down[0](x)
# x = net.down[1](x)
xt = x
# x = net.down[2](x)
# x = net.down[3](x)
# x = net.down[4](x)
# x = net.down[5](x) # softmax
# x = net.down[6](x)
# x = net.down[7](x)
# x = net.down[8](x) #
# x = net.down[9](x) #
# x = net.down[10](x) # softmax
# x = net.down[11](x)
# x = net.down[12](x)
# x = net.down[13](x)
# x = net.down[14](x) # avg
# x = net.down[15](x) # softmax
x = net.up(x)

x = LBit.apply(x)
y = x.transpose(1,2).detach()[0]
y = y.numpy()
print(y.shape)
with open("../data.bin", 'rb') as f:
    cy = np.frombuffer(f.read(), dtype='float32')
cy = cy.reshape(y.shape)
print(np.abs(cy-y).max())
print(y.shape)
print(cy.shape)
w = LBitTanh.apply(net.down[2].weight.data)
print_mat("p/c in", xt[0])
print_mat("c out", cy)
print_mat("p out", y)
print_mat("p-c out", y-cy)
# w = w.reshape(4, -1)
# ppy = w @ xt[0, :, 0:0+3].reshape(-1)[:,None]
# ppy = LBit.apply(ppy)
# print_mat("ppy", ppy*64)