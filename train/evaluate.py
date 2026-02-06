# evaluate.py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from possm.utils.metrics import calculate_r2

@torch.no_grad()
def evaluate_session(model, loader, device, config, train_mean, train_std):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # Validation loop
    for spike, bin_mask, spike_mask, vel, vel_lens in loader:
        spike = spike.to(device)
        bin_mask = bin_mask.to(device)
        spike_mask = spike_mask.to(device)
        vel = vel.to(device)
        vel_lens = vel_lens.to(device)
        
        # Determine valid output length based on k_history lag
        # Logic matches train_one_epoch in long_term_main.py
        effective_lens = vel_lens - (config.k_history - 1) * config.bin_size
        max_valid_time = effective_lens.max()
        
        # 1. Forward Pass
        # outputs shape: (batch, ~max_time, 2) - Normalized Space
        outputs = model(spike, bin_mask, spike_mask)
        
        # Clip to valid time length
        outputs = outputs[:, :max_valid_time, :]
        
        # 2. Process Batch
        for i in range(len(vel)):
            curr_len = effective_lens[i].item()
            if curr_len <= 0:
                continue
            
            # --- Extract Prediction ---
            # The model predicts normalized velocity based on Session 0 stats
            pred_norm = outputs[i, :curr_len, :] 
            
            # Denormalize using TRAINING (Session 0) stats
            # We want to see if the model's logic holds up in physical space
            pred_phys = (pred_norm * train_std) + train_mean
            
            # --- Extract Ground Truth ---
            # Ground truth needs to be aligned. 
            # In training, we compared outputs against: vel[:, (k-1)*bin_size:, :]
            start_idx = (config.k_history - 1) * config.bin_size
            target_phys = vel[i, start_idx : start_idx + curr_len, :]
            
            all_preds.append(pred_phys.cpu().numpy())
            all_targets.append(target_phys.cpu().numpy())

    if not all_preds:
        return None

    # Concatenate all trials to calculate global R2 for this session
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)
    
    # Calculate Metrics
    mse = np.mean((all_preds_np - all_targets_np) ** 2)
    avg_r2, r2_x, r2_y = calculate_r2(all_targets_np, all_preds_np)
    
    return {
        "mse": mse,
        "avg_r2": avg_r2,
        "r2_x": r2_x,
        "r2_y": r2_y
    }