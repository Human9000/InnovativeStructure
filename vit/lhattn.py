import torch


def lh_exp(x):
    return x + (1+x**2)**0.5


def lh_softmax(x):
    return lh_exp(x) / lh_exp(x).sum(dim=-1, keepdim=True)


def lh_softmax(x):
    return lh_exp(x) / lh_exp(x).sum(dim=-1, keepdim=True)


def lh_softmax_qkv_d1(q, k, v):
    return lh_softmax(q@k.transpose(-1, -2)) @ v


def lh_softmax_qkv_d2(q, k, v):
    fenzi = lh_exp(q@k.transpose(-1, -2)) @ v
    fenmu = lh_exp(q@k.transpose(-1, -2)).sum(dim=-1, keepdim=True)
    return fenzi / fenmu


def lh_softmax_qkv_d3(q, k, v):
    # 用Lx1的全1矩阵ic，利用矩阵乘法代替求和公式
    c_shape = list(q.shape[:-1])
    c_shape.append(1)

    r = torch.ones(c_shape).to(k.device)  # L,1
    

    # 展开lh_exp
    fenzi = (q@k.transpose(-1, -2) + (1+(q@k.transpose(-1, -2))**2)**0.5) @ v

 
    # 展开lh_exp
    fenmu = (q@k.transpose(-1, -2) + (1+(q@k.transpose(-1, -2))**2)**0.5) @ r
    return fenzi / fenmu

# 使用结合律加速矩阵乘法的计算， 将c和v分别和每一项相乘，最后再相加
def lh_softmax_qkv_d4(q, k, v):
    r_shape = list(q.shape[:-1])
    c_shape = list(q.shape[:-1])
    r_shape.append( 1)
    c_shape.insert(-1, 1)
    r = torch.ones(r_shape).to(k.device)  # L,1
    c = torch.ones(c_shape).to(k.device)  # 1,L

    signv = torch.sign(v)

    # 展开lh_exp
    # fenzi = q@(k.transpose(-1, -2) @ v) + (r@(c@v)**2+(q@(k.transpose(-1, -2) @ v))**2)**0.5*signv
    fenzi = (q@k.transpose(-1, -2) + (1+(q@k.transpose(-1, -2))**2)**0.5) @ v

    qk = q@k.transpose(-1, -2)
    # 展开lh_exp
    # fenmu = q@(k.transpose(-1, -2) @ r) + (r@(c@r)**2+(q@(k.transpose(-1, -2) @ r))**2)**0.5
    # fenmu = q@(k.transpose(-1, -2) @ r) + (r@c@(r**2)+(qk)**2@r**2)**0.5
    fenmu = q@(k.transpose(-1, -2) @ r) +  (1+qk**2)**0.5  @ (r**2) **0.5
    fenmu = q@(k.transpose(-1, -2) @ r) + ((1+qk**2) @ (r**2))**0.5
    return fenzi / fenmu


def lh_softmax_qkv_d5(q, k, v):

    #
    r_shape = list(q.shape[:-1])
    c_shape = list(q.shape[:-1])
    r_shape.insert(-1, 1)
    c_shape.insert(-2, 1)
    r = torch.ones(r_shape).to(k.device)  # L,1
    c = torch.ones(c_shape).to(k.device)  # 1,L

    qkv = q @ (k.transpose(-1, -2) @ v)
    qkr = q @ (k.transpose(-1, -2) @ r)
    rcv = r @ (c @ v)
    rcr = r @ (c @ r)

    signv = torch.sign(v)

    # 使用中间变量计算fenzi, 当v放到根号里面的时候默认去掉符号了，最后要把符号补回来
    fenzi = qkv + ((rcv**2 + qkv**2)**0.5)*signv

    # 使用中间变量计算fenmu
    fenmu = qkr + (rcr**2 + qkr**2)**0.5

    return fenzi / fenmu

 
if __name__ == '__main__':
    pass
    q = torch.randn(1, 4, 2) # B L c
    k = torch.randn(1, 4, 2) # B L c
    v = torch.randn(1, 4, 2) # B L c

    # y2 = lh_softmax_qkv_d2(q, k, v)
    # y3 = lh_softmax_qkv_d3(q, k, v)
    # y4 = lh_softmax_qkv_d4(q, k, v)
    # print(abs(y2 - y3).mean())
    # print(abs(y3 - y4).mean())
    # print(y3)
    # print(y4)

    print(q**2 @ k.transpose(-1,-2)**2)
    print((q @ k.transpose(-1,-2))**2)
    print((q**2)**0.5)
    print(q)
