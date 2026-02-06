# train.py
# 单独负责模型训练

import torch
from torch.utils.tensorboard import SummaryWriter
from possm.data.Dataloader import get_dataloader
from possm.models.Model import my_POSSM
from train.engine import train_one_epoch, validate, masked_mse_loss
from possm.utils.utils import set_seed, save_checkpoint
from possm.config.Config import my_POSSMConfig
import json

def run_experiment(config, hyperparam, data_dir, meta_data_path):
    '''
    运行训练实验. 
    Args:
        config: 模型配置参数
        hyperparam: 超参数字典
        data_dir: 预处理后的数据路径
        meta_data_path: 对应 meta_data 的路径
    '''
    set_seed(hyperparam["seed"])
    
    model_path = hyperparam["model_path"]
    meta_data = json.load(open(meta_data_path, "r"))
    num_channel = meta_data["num_channel"]
    
    writer = SummaryWriter(log_dir = f"{hyperparam['log_dir']}/{config.backbone}")
    
    train_loader, valid_loader = get_dataloader(
        batch_size = hyperparam["batch_size"],
        data_dir = data_dir
    )

    model = my_POSSM(
        config, 
        num_channel = num_channel
        ).to(hyperparam["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparam["learning_rate"],
        weight_decay=hyperparam["weight_decay"]
    )
    
    criterion = masked_mse_loss
    
    train_losses, val_losses = [], []
    
    # 设定训练停止条件
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(hyperparam["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, hyperparam["device"], writer, epoch, meta_data
        )

        val_loss = validate(
            model, valid_loader, criterion, hyperparam["device"], writer, epoch, meta_data
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{hyperparam["num_epochs"]}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 1. 检查是否是最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 重置计数器
            
            # 保存最佳模型权重
            save_checkpoint(model_path, model, config,
                train_losses,
                val_losses
            )
            print(f"--> Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"--> No improvement. Patience: {early_stopping_counter}/{hyperparam['patience']}")
        
        # 2. 检查是否触发早停
        if early_stopping_counter >= hyperparam['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break
    return train_losses, val_losses

def build_model_and_dataloader(config, ckpt_path, device, batch_size, data_dir):
    '''
    工具函数, 用于快速使用 dataloader
    '''
    model = my_POSSM(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_loader = get_dataloader(batch_size, data_dir)

    return model, val_loader