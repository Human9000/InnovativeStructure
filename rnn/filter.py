import torch
import torch.nn as nn
from torch.nn import functional as F

class SimpleIIRFilter(nn.Module):
    def __init__(self, order, features):
        super().__init__()
        self.order = order
        self.b = nn.Parameter(torch.randn((1, features, order+1))  * 0.1)
        self.a = nn.Parameter(torch.randn((1, features, order)) * 0.1)

    def forward(self, x):
        # x.shape = (batch_size, features, seq_len)
        y = torch.zeros_like(x)
        order = self.order
        for r in range(order, x.shape[1]):
            l = r - order
            y[r] = (x[..., l:r+1] * self.b).sum(dim=-1)
            y[r] -= (y[..., l:r] * self.a).sum(dim=-1) 
        return y
 

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt 

    x = np.sin(np.linspace(0, 300, 1000))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = np.sin(np.linspace(30, 330, 1000))
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    model = SimpleIIRFilter(6, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 1000
    # 训练
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    p = model(x)
    plt.plot(x[0, 0], label='x')
    plt.plot(y[0, 0], label='y')
    plt.plot(p[0, 0], label='p')
    plt.legend()
    plt.show()