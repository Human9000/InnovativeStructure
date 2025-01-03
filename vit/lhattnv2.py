import torch


def lh_exp(q, k):
    ql = -torch.clip(q, max=0)
    qr = torch.clip(q, min=0)
    kl = -torch.clip(k, max=0)
    kr = torch.clip(k, min=0) 
    O_1 = 1
    return (ql@kl + qr@kr + O_1)/(ql@kr + qr@kl+ O_1)

def lh_softmax(q, k):
    ql = -torch.clip(q, max=0)
    qr = torch.clip(q, min=0)
    kl = -torch.clip(k, max=0)
    kr = torch.clip(k, min=0) 
    O_1 = 1e-9

    fenzi = (ql@kl + qr@kr + O_1)/(ql@kr + qr@kl+ O_1)
    fenmu = ((ql@kl + qr@kr + O_1)/(ql@kr + qr@kl+ O_1)).sum(dim=1, keepdim=True)

    return fenzi/fenmu


def lh_softmax2(q, k):
    ql = -torch.clip(q, max=0)
    qr = torch.clip(q, min=0)
    kl = -torch.clip(k, max=0)
    kr = torch.clip(k, min=0) 
    O_1 = 1e-9

    fenzi = (ql@kl + qr@kr + O_1) 
    fenmu = (ql@kl + qr@kr + O_1).sum(dim=1, keepdim=True)

    return fenzi/fenmu


def lh_attn(q, k, v): 
    s = torch.ones((k.shape[1], 1))
    x = torch.exp(q@k) 
    return (x @ v)/( x @ s)


def lh_attnv2(q, k, v):
    ql = -torch.clip(q, max=0)
    qr = torch.clip(q, min=0)
    kl = -torch.clip(k, max=0)
    kr = torch.clip(k, min=0) 
    O_1 = 1e-9

    s = torch.ones((k.shape[1], 1))
    m = s/k.shape[1]
    klv = (kl@v)
    krv = (kr@v)
    kls = (kl@s)
    krs = (kr@s)
    klm = (kl@m)
    krm = (kr@m)

    # fenzi = (ql@kl + qr@kr + O_1) @ v
    # fenmu = (ql@kl + qr@kr + O_1) @ s

    # fenzi = (ql@kl + qr@kr + O_1)/(ql@kr + qr@kl+ O_1)
    # fenmu = ((ql@kl + qr@kr + O_1)/(ql@kr + qr@kl+ O_1)).sum(dim=1, keepdim=True)
    # fenzi = (ql@klv + qr@krv + O_1) / (ql@krm + qr@klm + qr@krm + ql@klm  + O_1)
    # fenmu = (ql@kls + qr@krs + O_1) / (ql@krm + qr@klm + qr@krm + ql@klm  + O_1)
    fenzi = (ql@klv + qr@krv + O_1) / (ql@kr + qr@kl + qr@kr + ql@kl+ O_1)
    fenmu = (ql@kls + qr@krs + O_1) / (ql@kr + qr@kl + qr@kr + ql@kl+ O_1)

    return fenzi/fenmu


def lh_attnv3(q, k, v):
    ql = -torch.clip(q, max=0)
    qr = torch.clip(q, min=0)
    kl = -torch.clip(k, max=0)
    kr = torch.clip(k, min=0) 
    O_1 = 1e-9 

    s = torch.ones((k.shape[1], 1)) 

    klv = (kl@v)
    krv = (kr@v)
    kls = (kl@s)
    krs = (kr@s) 

    attn_fenzi =  (ql@kl + qr@kr + O_1) # 同号
    y_fenzi = (ql@klv + qr@krv + O_1) # 同号
    fenmu = (ql@kls + qr@krs + O_1) # 同号
    return attn_fenzi/fenmu, y_fenzi/fenmu

def lh_attnv4(q, k, v, alpha=1):
    ql = torch.nn.functional.softplus(-q)
    qr = torch.nn.functional.softplus(q)
    kl = torch.nn.functional.softplus(-k)
    kr = torch.nn.functional.softplus(k)
    O_1 = 1e-9
    scale = q.shape[1] ** -0.5 
    s = torch.ones((k.shape[1], 1)) * scale # 这里是求和符号的等价矩阵
    v= v * scale
    klv = (kl@v)
    krv = (kr@v)
    kls = (kl@s)
    krs = (kr@s) 
 
    
    fenzi = (ql@klv + qr@krv + O_1) # 同号
    fenmu = (ql@kls + qr@krs + O_1) # 同号
    tong = fenzi/fenmu
    fenzi = (ql@krv + qr@klv + O_1) # 异号
    fenmu = (ql@krs + qr@kls + O_1) # 异号
    yi = fenzi/fenmu

    beta = 1 - alpha
    return tong*alpha  - yi * beta

if __name__ == '__main__':
    q = torch.randn(3, 3)
    k = torch.randn(3, 3)
    v = torch.randn(3, 3)


    print(torch.softmax(q@ k, dim=1)@v)
    print(lh_attn(q, k, v))
    print(lh_attnv2(q, k, v))
    print(lh_attnv3(q, k, v))
    print(lh_attnv4(q, k, v))

