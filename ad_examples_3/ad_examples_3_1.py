import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(1, 100), nn.ReLU(),
                      nn.Linear(100, 100), nn.ReLU(),
                      nn.Linear(100, 100), nn.ReLU(),
                      nn.Linear(100, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()
x = torch.randn(100, 1)
for i in range(100):
    loss = criterion(model(x), x)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("{}th loss is:{:.6f}".format(i+1, loss.item()))

plt.plot(np.arange(-3, 3, 0.01), model(torch.arange(-3, 3, 0.01)[:, None]).detach().numpy())
plt.xlabel("input x")
plt.ylabel("output y")
plt.show()

