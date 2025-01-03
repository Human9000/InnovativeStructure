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

    klv = kl@v
    krv = kr@v
    kls = kl@s
    krs = kr@s 

    fenzi = ql@klv + qr@krv - (ql@krv + qr@klv) + O_1
    fenmu = ql@kls + qr@krs - (ql@krs + qr@kls) + O_1


    return fenzi / fenmu 

if __name__ == '__main__':
    q = torch.randn(3, 3)
    k = torch.randn(3, 3)
    v = torch.randn(3, 3)

    # print(q@k)
    # print(torch.exp(q@k)) 
    # print(lh_exp(q, k))

    # print(torch.softmax(q@ k, dim=1))
    # print(lh_softmax(q, k))
    # print(lh_softmax2(q, k))

    print(torch.softmax(q@ k, dim=1)@v)
    # print( q@ k@v)
    # print()
    print(lh_attn(q, k, v)) 
    print(lh_attnv2(q, k, v))
    # print()
    # print(v)

