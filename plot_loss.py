# plot_loss.py
# 用法: 
# python plot_loss.py --ckpts checkpoints/possm_gru_seed42.pt checkpoints/possm_s4d_seed42.pt --out graphs/loss_compare.png

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt


def plot_losses(ckpt_paths, out_path):
    plt.figure(figsize=(8, 5))

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        train_loss = ckpt.get("train_loss", None)
        val_loss = ckpt.get("val_loss", None)
        backbone = ckpt.get("backbone", Path(ckpt_path).stem)

        if train_loss is None:
            print(f"[Warning] No train_loss in {ckpt_path}")
            continue

        epochs_train = range(1, len(train_loss) + 1)
        epochs_val = range(1, len(val_loss) + 1) if val_loss is not None else None

        # train: solid line
        line_train, = plt.plot(
            epochs_train,
            train_loss,
            linestyle="-",
            label=f"{backbone}-train",
        )

        # val: dashed line, SAME COLOR
        if val_loss is not None:
            plt.plot(
                epochs_val,
                val_loss,
                linestyle="--",
                color=line_train.get_color(),
                label=f"{backbone}-val",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Checkpoint paths (.pt)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="graphs/loss_comparison.png",
        help="Output image path",
    )

    args = parser.parse_args()
    plot_losses(args.ckpts, args.out)
