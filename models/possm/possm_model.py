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
        B, T, L, D = z.shape
        z = z.reshape(B, T, L * D)
        
        h = self.backbone(z, bin_mask) # h: (batch_size, max_bin, hidden_dim)
        vel_pred = self.output_decoder(h, self.freqs_cos, self.freqs_sin) # (batch_size, (max_bin+k-1) * bin_size, 2)
        return vel_pred