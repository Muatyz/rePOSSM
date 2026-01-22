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

        # === (1) 输入投影：对齐 GRU 的 input_size → hidden_size ===
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # === (2) S4D 时间模块 ===
        self.s4d = S4D(
            d_model=hidden_dim,
            d_state=d_state,
            dropout=dropout,
            transposed=False,  # (B, T, D)
        )

        # === (3) 输出投影（保持维度一致，增强表达） ===
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # === (4) 残差 + 归一化（补偿 GRU 的 gating 行为） ===
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z_t, bin_mask):
        """
        Args:
            z_t: (B, T, num_latents, embed_dim)
            bin_mask: (B, T), True=valid, False=padding
        Returns:
            output: (B, T, hidden_dim)
        """
        B, T = z_t.shape[0], z_t.shape[1]

        # === 与 GRU 版本完全一致的 flatten ===
        z_t = z_t.view(B, T, -1)  # (B, T, input_dim)

        # === 输入投影 ===
        x = self.in_proj(z_t)     # (B, T, hidden_dim)

        # === mask padding（关键补偿点 1） ===
        x = x * bin_mask.unsqueeze(-1)

        # === S4D 时间建模 ===
        y, _ = self.s4d(x)

        # === 输出投影 ===
        y = self.out_proj(y)

        # === 残差 + norm（关键补偿点 2） ===
        y = self.norm(y + x)

        # === 再次 mask，确保 padding 为 0 ===
        y = y * bin_mask.unsqueeze(-1)

        return y
