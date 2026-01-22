import torch
import torch.nn as nn
from model import S4DToy
from data import make_delay_data

# hyperparams
B = 32
L = 200
D = 8
delay = 20
epochs = 500

model = S4DToy(d_model=D)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

x, y = make_delay_data(B, L, delay, D)

for epoch in range(epochs): 
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | loss = {loss.item():.4f}")

print("Done.")