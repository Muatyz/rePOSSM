# engine.py
# 纯粹的计算定义

from tqdm import tqdm
import torch

def align_predictions_and_targets(outputs, vel, vel_lens, config):
    """
    根据模型类型，对 pred 和 target 做时间对齐
    """
    if config.model == "possm":
        shift = (config.k_history - 1) * config.bin_size
        
        vel_lens = vel_lens - shift
        max_time_length = vel_lens.max()
        
        outputs = outputs[:, :max_time_length, :]
        target = vel[:, shift:shift + max_time_length, :]
        
    elif config.model == "rnn":
        # RNN 不需要 shift
        max_time_length = vel_lens.max()
        
        outputs = outputs[:, :max_time_length, :]
        target = vel[:, :max_time_length, :]
        
    else:
        raise ValueError(f"Unknown model: {config.model}")
    
    return outputs, target, vel_lens

def masked_mse_loss(output, target, lengths):
    """
    计算 masked MSE loss. 
    Args:
        output: (batch_size, max_time_length, 2)
        target: (batch_size, max_time_length, 2)
        lengths: (batch_size) - 存储每个样本的有效长度
    """
    # 1. 生成基础掩码 (batch_size, max_time_length)
    batch_size, max_time, dim = output.shape
    device = output.device
    
    # torch.arange(max_time) 生成 [0, 1, 2, ..., max_time-1]
    # 利用广播机制与 lengths 比较
    mask = torch.arange(max_time, device=device).expand(batch_size, max_time) < lengths.unsqueeze(1)
    
    # 2. 将掩码扩展到特征维度 (batch_size, max_time_length, 2)
    # 增加最后一个维度并复制
    mask = mask.unsqueeze(-1).expand_as(output)
    
    # 3. 计算平方损失
    squared_diff = (output - target) ** 2
    
    # 4. 应用掩码并求平均
    # 只计算 mask 为 True 的部分的均值
    masked_squared_diff = squared_diff * mask.float()
    num_valid_elements = mask.sum()
    
    loss = masked_squared_diff.sum() / num_valid_elements
    return loss

def train_one_epoch(model, loader, optimizer, criterion, device, writer, epoch, meta_data):
    model.train()
    
    VEL_MEAN = torch.tensor(meta_data["vel_mean"], dtype=torch.float32)
    VEL_STD = torch.tensor(meta_data["vel_std"], dtype=torch.float32)
    
    mean_tensor = VEL_MEAN.to(device)
    std_tensor = VEL_STD.to(device)

    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)
    for spike, bin_mask, spike_mask, vel, vel_lens in pbar:
        
        spike = spike.to(device)
        bin_mask = bin_mask.to(device)
        spike_mask = spike_mask.to(device)
        vel = vel.to(device)
        vel_lens = vel_lens.to(device)

        optimizer.zero_grad()

        # ======================
        # 1. forward
        # ======================
        outputs = model(spike, bin_mask, spike_mask)

        # ======================
        # 2. normalize target
        # ======================
        normalized_vel = (vel - mean_tensor) / std_tensor

        # ======================
        # 3. 关键：对齐（统一入口）
        # ======================
        outputs, target, vel_lens = model.align(
            outputs, normalized_vel, vel_lens
        )

        # ======================
        # 4. loss
        # ======================
        loss = criterion(outputs, target, vel_lens)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(loader)
    writer.add_scalar('Loss/Train', epoch_loss, epoch)

    return epoch_loss

@torch.no_grad()
def validate(model, loader, criterion, device, writer, epoch, meta_data):
    model.eval()
    
    VEL_MEAN = torch.tensor(meta_data["vel_mean"], dtype=torch.float32)
    VEL_STD = torch.tensor(meta_data["vel_std"], dtype=torch.float32)
    
    mean_tensor = VEL_MEAN.to(device)
    std_tensor = VEL_STD.to(device)

    running_loss = 0.0
    
    for spike, bin_mask, spike_mask, vel, vel_lens in tqdm(loader, desc="Validating", leave=True):
        
        spike = spike.to(device)
        bin_mask = bin_mask.to(device)
        spike_mask = spike_mask.to(device)
        vel = vel.to(device)
        vel_lens = vel_lens.to(device)

        # forward
        outputs = model(spike, bin_mask, spike_mask)

        # normalize
        normalized_vel = (vel - mean_tensor) / std_tensor

        # 对齐（同一接口）
        outputs, target, vel_lens = model.align(
            outputs, normalized_vel, vel_lens
        )

        loss = criterion(outputs, target, vel_lens)
        
        running_loss += loss.item()
        
    val_loss = running_loss / len(loader)
    writer.add_scalar('Loss/Valid', val_loss, epoch)

    return val_loss