import torch
import torch.nn as nn
from models.possm.cross_attention import POSSMCrossAttention
from models.possm.RoPE import RotaryEmbedding
  
from models.possm.backbones.GRU import POSSM_Backbone_GRU
from models.possm.backbones.S4D import POSSM_Backbone_S4D
from models.possm.Output_Decoder import POSSMOutputDecoder


class POSSM_Model(nn.Module):
    def __init__(self, config):
        '''
        初始化 POSSM_Model 模型
        '''
        super().__init__()
        self.config = config
        # ===== consistency check =====
        if config.backbone == "gru":
            assert config.hidden_size == config.gru_hidden_size

        elif config.backbone == "s4d":
            assert config.hidden_size == config.s4d_hidden_size

        print("Backbone selected:", config.backbone)
        print("Number of channels:", config.num_channel)

        print("Model dimension summary:")
        print("embed_dim:", config.embed_dim)
        print("num_latents:", config.num_latents)
        print("input_dim:", config.num_latents * config.embed_dim)
        print("hidden_size:", config.hidden_size)

        # 通过 UnitEmb(i) 将 各个 channel 映射为一个 embed_dim 维的矢量
        # 总共有 num_embeddings 个 channel
        self.emb = nn.Embedding(
            num_embeddings = config.num_channel, 
            embedding_dim = config.embed_dim
            )
        
        self.cross_attention = POSSMCrossAttention(config)
        
        input_dim = config.num_latents * config.embed_dim
        # 根据需求选取不同的 Backbone: GRU/S4D/Mamba
        if config.backbone == "gru":
            self.backbone = POSSM_Backbone_GRU(
                input_dim, 
                config.gru_hidden_size, 
                config.gru_num_layers, 
                config.dropout)
        elif config.backbone == "s4d":
            self.backbone = POSSM_Backbone_S4D(
                input_dim=input_dim, 
                hidden_dim=config.s4d_hidden_size, 
                d_state=config.s4d_state_dim, 
                dropout=config.s4d_dropout)
        elif config.backbone == "mamba":
            print("等待完善")
            raise NotImplementedError("Mamba backbone not implemented yet")
        
        head_dim = config.hidden_size // config.num_attention_heads
        
        freqs_cos, freqs_sin = RotaryEmbedding.precompute_freqs_cis(
            head_dim, 
            config.bin_size, 
            config.rope_theta
            )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        self.output_decoder = POSSMOutputDecoder(config)
        
    def align(self, outputs, target, vel_lens):
        """
        POSSM 时间对齐：
        pred[t] 对应 vel[t + shift]
        """
        # ======================
        # 1. 计算 shift
        # ======================
        shift = (self.config.k_history - 1) * self.config.bin_size

        # ======================
        # 2. 裁剪 target（去掉前面的未来窗口）
        # ======================
        target = target[:, shift:, :]

        # ======================
        # 3. 对齐长度（防止 mismatch）
        # ======================
        min_len = min(outputs.shape[1], target.shape[1])

        outputs = outputs[:, :min_len, :]
        target  = target[:, :min_len, :]

        # ======================
        # 4. 修正有效长度
        # ======================
        vel_lens = vel_lens - shift
        vel_lens = torch.clamp(vel_lens, min=0, max=min_len)

        return outputs, target, vel_lens

    def forward(self, spike, bin_mask, spike_mask):
        '''
        前向传播
        Args: 
            self:
            spike: 
            bin_mask:
            spike_mask: 
            
        Returns:
            vel_pred: 
        '''
        # spike: (batch, max_bin, max_token, 2)
        # bin_mask: (batch, max_bin)
        # spike_mask: (batch, max_bin, max_token)
        channels, offsets = spike[..., 0], spike[..., 1]
        emb = self.emb(channels) # (batch, max_bin, max_token, embed_dim)
        z = self.cross_attention(emb, offsets, spike_mask, self.freqs_cos, self.freqs_sin) # (batch_size, max_bin, num_latents, embed_dim)
        
        # 根据 config 内设定的 backbone 参数进行隐藏层的计算
        # 为了确保 S4D 和 GRU 兼容使用同一个函数接口, 需要将 z 从 (B, T, L, D) reshape 成 (B, T, L*D)
        batch_size, max_bin, num_latents, embed_dim = z.shape
        z_flattened = z.view(batch_size, max_bin, -1)
        print("z:", z.shape)
        print("bin_mask:", bin_mask.shape)
        
        h = self.backbone(z_flattened, bin_mask) # h: (batch_size, max_bin, hidden_dim)
        vel_pred = self.output_decoder(h, self.freqs_cos, self.freqs_sin) # (batch_size, (max_bin+k-1) * bin_size, 2)
        
        # 打印诊断信息
        print("="*60)
        print("Dimension info:")
        print("spike:", spike.shape)
        print("emb:", emb.shape)
        print("z:", z.shape)
        print("h:", h.shape)
        print("vel_pred:", vel_pred.shape)
        return vel_pred