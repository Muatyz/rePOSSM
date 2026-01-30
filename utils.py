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

def save_checkpoint(path, model, config ,train_loss, val_loss):
    torch.save({
        "model_state": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "backbone": config.backbone, 
    }, path)
    
def load_checkpoint(path, model, device="cuda" if torch.cuda.is_available() else "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt