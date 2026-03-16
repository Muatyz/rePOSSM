# 尝试替换原 GRU 为 S4D 模块

import torch
import torch.nn as nn
from s4.s4d import S4D

class POSSM_Backbone_S4D(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        d_state=64,
        dropout=0.0,
    ):
        """
        S4D Backbone, 用于替换 POSSM 中的 GRU.
        
        input_dim: num_latents * embed_dim
        hidden_dim: 时间建模后的 latent 维度（对应 GRU hidden_dim）
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # (1) 输入投影：对齐 GRU 的 input_size → hidden_size
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # (2) S4D 时间模块
        self.s4d = S4D(
            d_model=hidden_dim,
            d_state=d_state,
            transposed=False,  # (B, T, D)
        )

        # (3) 输出投影（保持维度一致，增强表达）
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # (4) 残差 + 归一化（补偿 GRU 的 gating 行为）
        self.norm = nn.LayerNorm(hidden_dim)
        
        # (5) Dropout（可选）
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_t, bin_mask):
        """
        Args:
            z_t: (B, T, input_dim)
            bin_mask: (B, T), True=valid, False=padding
        Returns:
            output: (B, T, hidden_dim)
        """
        mask = bin_mask.unsqueeze(-1).float()

        mask = bin_mask.unsqueeze(-1).float()

        z_t = self.in_proj(z_t)
        z_t = z_t * mask

        y, _ = self.s4d(z_t)

        y = self.out_proj(y)
        y = self.dropout(y)

        y = z_t + y
        y = self.norm(y)

        y = y * mask

        return y
