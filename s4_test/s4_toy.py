import torch
import torch.nn as nn
from s4.s4 import S4

class S4Toy(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.s4 = S4(
            d_model=d_model,
            d_state=d_state,
            transposed=False,  # IMPORTANT
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        y, _ = self.s4(x)     # S4 returns (output, state)
        y = self.norm(y)
        return y
