# 加载存储的模型参数. 

import torch

def load_model_from_checkpoint(
    model,
    ckpt_path,
    device="cpu",
    strict=True,
):
    """
    Load model weights from a checkpoint.
    Compatible with:
      1) pure state_dict
      2) experiment-style checkpoint with 'model_state'
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt  # pure state_dict

    model.load_state_dict(state_dict, strict=strict)
    return ckpt
