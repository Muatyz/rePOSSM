# configs/rnn_config.py

class RNNConfig:
    def __init__(self):
        # ===== 基本标识 =====
        self.model = "rnn"
        self.backbone = "rnn"  # 为了和 possm 接口统一保留

        # ===== 模型结构 =====
        self.input_size = 128
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.1

        # ===== 输出 =====
        self.output_dim = 2
        
        self.time_lag = 80, # kinematics data lag 80ms relative to neural data
        self.bin_size = 50,