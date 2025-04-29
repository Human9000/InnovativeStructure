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


net = ECGSegMCULBit().float()
net.load_state_dict(state)
w = net.down[2].weight.data.transpose(1,2)
w = LBitTanh.apply(w)
w = w.numpy()
print(w.shape)
with open("../data.bin", 'rb') as f:
    cy = np.frombuffer(f.read(), dtype='float32')
cy = cy.reshape(w.shape)
print(np.abs(cy-w).max())
# print(np.abs(cy-y).T)

print(w.shape)
print(cy.shape)

print(w)
print()
print(cy)
print()
print(w-cy)

