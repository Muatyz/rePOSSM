# evaluate.py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from metrics import mse_loss, r2_score


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    save_dir,
    split: str = "val",
    max_batches: int = None,
):
    """
    Args:
        model: trained my_POSSM model
        dataloader: validation dataloader
        device: cuda / cpu
        save_dir: directory to save eval results
        split: "val" or "test"
        max_batches: for quick debug (None = full eval)
    """

    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vel_preds = []
    vel_trues = []

    for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating [{split}]")):

        spike, spike_lengths, vel_true, spike_mask, *rest = batch

        spike = spike.to(device)
        spike_lengths = spike_lengths.to(device)
        spike_mask = spike_mask.to(device)
        vel_true = vel_true.to(device)

        vel_pred = model(spike, spike_lengths, spike_mask)

        vel_preds.append(vel_pred.cpu())
        vel_trues.append(vel_true.cpu())

        if max_batches is not None and (i + 1) >= max_batches:
            break


    # --------------------------------------------------
    # concat over batch dimension
    # --------------------------------------------------
    vel_pred_all = torch.cat(vel_preds, dim=0)   # (N, T, 2)
    vel_true_all = torch.cat(vel_trues, dim=0)   # (N, T, 2)

    # --------------------------------------------------
    # flatten for metrics (ignore mask for now)
    # --------------------------------------------------
    vel_pred_flat = vel_pred_all.reshape(-1, 2).numpy()
    vel_true_flat = vel_true_all.reshape(-1, 2).numpy()

    metrics = {
        "mse": float(mse_loss(vel_pred_flat, vel_true_flat)),
        "r2": float(r2_score(vel_pred_flat, vel_true_flat)),
        "num_sequences": int(vel_pred_all.shape[0]),
        "time_bins": int(vel_pred_all.shape[1]),
    }

    # --------------------------------------------------
    # save
    # --------------------------------------------------
    torch.save(
        {
            "vel_pred": vel_pred_all,
            "vel_true": vel_true_all,
            "metrics": metrics,
        },
        save_dir / f"eval_{split}.pt"
    )

    with open(save_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
