# models/rnn.py

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, config, num_channel):
        super().__init__()

        self.num_channel = num_channel

        self.rnn = nn.GRU(
            input_size=num_channel,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )

        self.head = nn.Linear(config.hidden_size, config.output_dim)

    def spike_to_dense(self, spike):
        """
        spike: (B, T, max_token, 2)
        -> (B, T, num_channel)
        """
        B, T, K, _ = spike.shape
        device = spike.device

        channels = spike[..., 0].long()

        x = torch.zeros(B, T, self.num_channel, device=device)

        for b in range(B):
            for t in range(T):
                valid = channels[b, t] >= 0
                ch = channels[b, t][valid]
                x[b, t].scatter_add_(0, ch, torch.ones_like(ch, dtype=torch.float))

        return x
    
    def align(self, outputs, target, vel_lens):
        """
        对齐 outputs 和 target 的时间长度
        """
    
        # 1. 找最短长度
        min_len = min(outputs.shape[1], target.shape[1])
    
        # 2. 裁剪
        outputs = outputs[:, :min_len, :]
        target  = target[:, :min_len, :]
    
        # 3. 修正有效长度
        vel_lens = torch.clamp(vel_lens, max=min_len)
    
        return outputs, target, vel_lens

    def forward(self, spike, bin_mask=None, spike_mask=None):
        x = self.spike_to_dense(spike)

        h, _ = self.rnn(x)
        out = self.head(h)   # (B, T, 2)

        # ===== 🔥关键：对齐 POSSM 输出 =====
        bin_size = 50  # ⚠️ 你 config.bin_size，建议从 config 传进来

        out = out.unsqueeze(2)                  # (B, T, 1, 2)
        out = out.repeat(1, 1, bin_size, 1)     # (B, T, bin_size, 2)
        out = out.reshape(out.shape[0], -1, 2)  # (B, T*bin_size, 2)

        return out