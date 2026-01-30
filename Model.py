import torch
import torch.nn as nn
from Cross_Attention import POSSMCrossAttention
from RoPE import RotaryEmbedding

from Config import my_POSSMConfig
config = my_POSSMConfig()

import json
meta_data = json.load(open("processed_data/meta_data.json", "r"))
max_time_length = meta_data["max_time_length"]
  
from GRU import POSSM_Backbone_GRU
from S4D import POSSM_Backbone_S4D
from Output_Decoder import POSSMOutputDecoder


class my_POSSM(nn.Module):
    def __init__(self, config):
        '''
        初始化 my_POSSM 模型
        '''
        super().__init__()

        # 通过 UnitEmb(i) 将 各个 channel 映射为一个 embed_dim 维的矢量
        # 总共有 num_embeddings 个 channel
        self.emb = nn.Embedding(
            num_embeddings = meta_data["num_channel"], 
            embedding_dim = config.embed_dim
            )
        
        self.Cross_Attention = POSSMCrossAttention(config)
        
        # 根据需求选取不同的 Backbone: GRU/S4D/Mamba
        if config.backbone == "gru":
            self.backbone = POSSM_Backbone_GRU(
                config.num_latents * config.embed_dim, 
                config.gru_hidden_size, 
                config.gru_num_layers, 
                config.dropout)
        elif config.backbone == "s4d":
            self.backbone = POSSM_Backbone_S4D(
                config.num_latents * config.embed_dim, 
                config.s4d_hidden_size, 
                config.s4d_num_layers, 
                config.s4d_dropout)
        elif config.backbone == "mamba":
            print("等待完善")
            return 0
        
        freqs_cos, freqs_sin = RotaryEmbedding.precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, 
            config.bin_size, config.rope_theta
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
        z = self.Cross_Attention(emb, offsets, spike_mask, self.freqs_cos, self.freqs_sin) # (batch_size, max_bin, num_latents, embed_dim)
        
        # 根据 config 内设定的 backbone 参数进行隐藏层的计算
        h = self.backbone(z, bin_mask) # h: (batch_size, max_bin, hidden_dim)
        vel_pred = self.output_decoder(h, self.freqs_cos, self.freqs_sin) # (batch_size, (max_bin+k-1) * bin_size, 2)
        return vel_pred