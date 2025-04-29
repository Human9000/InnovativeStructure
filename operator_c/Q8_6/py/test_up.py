import numpy as np
import torch
from torch import nn
from model import Upsample
from bnn_ops import LBit
from  torch.nn import  functional as F
def print_mat(name, mat):
    print("=>",name, mat.shape,":")
    for i in mat:
        for d in i:
            print("%.7f"%d ,end='\t')
        print()
    print()

x = torch.arange(0,12).reshape(6,2).transpose(0,1).float() / 12
x = LBit.apply(x)
up  = Upsample(5)
y = up(x[None])[0].transpose(0,1)
y = LBit.apply(y)
y = y.numpy()

print_mat("X",x)
print_mat("Y",y.T)
with open("../data.bin", 'rb') as f:
    cy = np.frombuffer(f.read(), dtype='float32')
cy = cy.reshape(y.shape)
print_mat("CY",cy.T)
print_mat("CY-y",(cy-y).T)
print(np.abs(cy-y) >0)

left = .1562500
d = .3281250 - .1562500
print([left + LBit.apply(torch.tensor(d*i/5)) for i in range(5)])