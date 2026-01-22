import torch
import torch.nn as nn
from s4.s4d import S4D

class S4DToy(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.s4 = S4D(
            d_model = d_model,
            d_state = 64,        # toy 推荐 32~64
            dropout = 0.0,
            transposed = False  # 重要：我们用 (B, L, D)
        )

        #self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        x_in = self.in_proj(x) # 残差
        y, _ = self.s4(x_in)     # S4D 返回 (output, state)
        #y = self.norm(y)
        y = self.out_proj(y)
        return y + x
