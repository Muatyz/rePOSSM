# scripts/visualize.py

import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def collect_predictions(model, dataloader, device, train_mean, train_std, max_batches=1):
    model.eval()

    all_preds = []
    all_targets = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        spikes = batch["spikes"].to(device)
        target = batch["vel"].to(device)

        pred = model(spikes)

        # ===== 反归一化（关键！）=====
        pred = pred * train_std + train_mean
        target = target * train_std + train_mean

        # flatten
        pred = pred.reshape(-1, pred.shape[-1]).cpu()
        target = target.reshape(-1, target.shape[-1]).cpu()

        all_preds.append(pred)
        all_targets.append(target)

    return torch.cat(all_preds), torch.cat(all_targets)

def plot_prediction_vs_target(pred, target, session_id, save_path=None):
    t = range(len(pred))

    plt.figure(figsize=(12, 5))

    # X
    plt.subplot(1, 2, 1)
    plt.plot(t, target[:, 0], label="target_x")
    plt.plot(t, pred[:, 0], label="pred_x", alpha=0.7)
    plt.title(f"Session {session_id} - X")
    plt.legend()

    # Y
    plt.subplot(1, 2, 2)
    plt.plot(t, target[:, 1], label="target_y")
    plt.plot(t, pred[:, 1], label="pred_y", alpha=0.7)
    plt.title(f"Session {session_id} - Y")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()