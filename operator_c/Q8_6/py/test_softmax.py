from bnn_ops import lbit
import torch


def expi32(x):
    x = (x * 64.).to(torch.int64)
    x0 = torch.tensor(64, dtype=torch.int64)
    x1 = x  # 6
    x2 = torch.div(x1 * x1, 2, rounding_mode='trunc' )  # 12
    x3 = torch.div(x1 * x2, 3, rounding_mode='trunc')  # 18
    x4 = torch.div(x1 * x3, 4, rounding_mode='trunc')  # 24

    temp = x0*64**3 + x1 * 64**3 + x2 * 64**2 + x3 * 64**1 + x4
    o = torch.div(64*x0*64**3 , temp, rounding_mode='trunc')/ 64.
    # return o
    return temp

def expf32(x):
    x1 = lbit(x, 2**6)
    x2 = lbit(x1 * x1 / 2, 2**12)
    x3 = lbit(x2 * x1 / 3, 2**18)
    x4 = lbit(x3 * x1 / 4, 2**24)
    temp = 1 + x1 + x2 + x3 + x4
    o = lbit(1 / temp, 2**6 )
    return temp


for i in range(0, 128 ):
    x = torch.tensor(i / 64., dtype=torch.float32)
    i32 = expi32(x).item()
    f32 = expf32(x).item()
    x = x.item()
    print("%.6f, %.6f, %.6f, %.6f, "%(x,  f32, i32, f32 - i32))
