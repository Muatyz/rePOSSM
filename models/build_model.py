# possm/models/build_model.py
# 负责替代 train.py 中的模型构建部分, run_experiment() 应忽略模型内部结构的细节

from models.possm.possm_model import POSSM_Model

def build_model(config):
    """
    Unified model builder.

    Args
    ----
    config : experiment config

    Returns
    -------
    torch.nn.Module
    """

    model_type = config.model

    if model_type == "possm":
        backbone = config.backbone

        if backbone in ["gru", "s4d"]:
            return POSSM_Model(config)

        raise ValueError(f"Unsupported POSSM backbone: {backbone}")

    # ======================
    # RNN (future)
    # ======================
    if model_type == "rnn":
        raise NotImplementedError("RNN model not implemented yet")

    # ======================
    # Transformer (future)
    # ======================
    if model_type == "transformer":
        raise NotImplementedError("Transformer model not implemented yet")

    raise ValueError(f"Unknown model type: {model_type}")
