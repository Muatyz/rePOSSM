# evaluate.py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from models.possm.utils.metrics import calculate_r2

@torch.no_grad()
def evaluate_session(model, loader, device, config, train_mean, train_std):

    model.eval()
    
    all_preds = []
    all_targets = []

    for spike, bin_mask, spike_mask, vel, vel_lens in loader:

        spike = spike.to(device)
        bin_mask = bin_mask.to(device)
        spike_mask = spike_mask.to(device)
        vel = vel.to(device)
        vel_lens = vel_lens.to(device)

        # ======================
        # 1. forward
        # ======================
        outputs = model(spike, bin_mask, spike_mask)

        # ======================
        # 2. normalize target（和 train 一致）
        # ======================
        normalized_vel = (vel - train_mean) / train_std

        # ======================
        # 3. 🔥关键：统一对齐
        # ======================
        outputs, target, vel_lens = model.align(
            outputs, normalized_vel, vel_lens
        )

        # ======================
        # 4. 反归一化 prediction
        # ======================
        preds = outputs * train_std + train_mean
        targets = target * train_std + train_mean

        # ======================
        # 5. mask 有效时间步
        # ======================
        for i in range(preds.shape[0]):
            valid_len = vel_lens[i]

            all_preds.append(preds[i, :valid_len].cpu().numpy())
            all_targets.append(targets[i, :valid_len].cpu().numpy())

    # ======================
    # 6. concat & metric
    # ======================
    if len(all_preds) == 0:
        return None

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # compute metrics（你原来的逻辑）
    mse = np.mean((all_preds - all_targets) ** 2)

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
        return 1 - ss_res / ss_tot

    r2 = r2_score(all_targets, all_preds)

    return {
        "avg_r2": float(np.mean(r2)),
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "mse": float(mse)
    }