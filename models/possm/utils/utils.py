# utils.py
# 负责模型的存取
import torch
import numpy as np
import random
import os

def set_seed(seed):
    '''
    设定模型随机数
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model_path, model, config ,train_loss, val_loss):
    torch.save(
        {
        "model_state": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "backbone": config.backbone, 
    }, 
        model_path)
    
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