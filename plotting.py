# plotting.py
# 单独负责绘图

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "",
    save_path: str = None
):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(alpha=0.3)

    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_loss_comparison(
    ckpt_paths: Dict[str, str],
    save_path: str = None
):
    """
    ckpt_paths:
        {
            "gru": "checkpoints/possm_gru.pt",
            "s4d": "checkpoints/possm_s4d.pt",
            "mamba": "checkpoints/possm_mamba.pt",
        }
    """
    plt.figure(figsize=(7, 4))

    for name, path in ckpt_paths.items():
        ckpt = torch.load(path, map_location="cpu")
        plt.plot(ckpt["val_loss"], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Backbone Comparison")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_velocity_timecourse(
    vel_pred: torch.Tensor,
    vel_true: torch.Tensor,
    dims=(0, 1),
    title: str = "",
    save_path: str = None
):
    """
    vel_pred, vel_true: (T, 2)
    """
    vel_pred = vel_pred.cpu().numpy()
    vel_true = vel_true.cpu().numpy()

    T = vel_true.shape[0]
    t = range(T)

    fig, axes = plt.subplots(len(dims), 1, figsize=(8, 3 * len(dims)), sharex=True)

    if len(dims) == 1:
        axes = [axes]

    for ax, d in zip(axes, dims):
        ax.plot(t, vel_true[:, d], label="True", linewidth=2)
        ax.plot(t, vel_pred[:, d], label="Pred", linestyle="--")
        ax.set_ylabel(f"Vel dim {d}")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (bins)")

    if title:
        fig.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_velocity_comparison(
    eval_paths: Dict[str, str],
    dim: int = 0,
    T: int = 1000,
    save_path: str = None
):
    """
    eval_paths:
        {
            "gru": "eval_gru.pt",
            "s4d": "eval_s4d.pt",
        }
    """
    plt.figure(figsize=(8, 4))

    vel_true = None

    for name, path in eval_paths.items():
        data = torch.load(path, map_location="cpu")
        plt.plot(data["vel_pred"][:T, dim], label=name)
        if vel_true is None:
            vel_true = data["vel_true"]

    plt.plot(vel_true[:T, dim], label="True", linewidth=2, color="black")

    plt.plot(data["vel_true"][:T, dim], label="True", linewidth=2, color="black")

    plt.xlabel("Time")
    plt.ylabel(f"Velocity dim {dim}")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
