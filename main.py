# main.py
# 省略实验细节, 只进行宏观函数使用

import os
import torch
import argparse
from pathlib import Path

from Config import my_POSSMConfig
from train import run_experiment
from evaluate import evaluate
from plotting import (
    plot_loss_curve,
    plot_velocity_timecourse,
)


def main():
    # 等待进一步传参
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--eval", action = "store_true")
    parser.add_argument("--backbone", type=str, default = "s4d")
    parser.add_argument("--seed", type=int, default = 42)
    args = parser.parse_args()
    
    # 1. 引入 config 参数
    config = my_POSSMConfig()
    config.backbone = args.backbone # 选取 backbone 参数, 目前支持 gru/s4d
    
    # 2. 设定模型存储路径
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok = True)

    ckpt_path = ckpt_dir / f"possm_{config.backbone}_seed{args.seed}.pt"
    eval_dir = Path("eval") / f"{config.backbone}_seed{args.seed}"
    eval_dir.mkdir(parents = True, exist_ok = True)
    
    # 3. 根据传参, 决定训练或者评估模型
    print("=" * 60)
    print(f"Running POSSM with backbone = {config.backbone}")
    print(f"Checkpoint path: {ckpt_path}")
    print("=" * 60)
    
    # 全局超参数
    hyperparam = {
            "seed": args.seed,
            "batch_size": 32,  # 原设定为 256, 但是会显存不足, 正在排查原因中
            "num_epochs": 20,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "log_dir": f"./log/{config.backbone}",
        }
    # ============ Train ==============
    if args.train:
        run_experiment(
            config = config,
            hyperparam = hyperparam,
            ckpt_path = str(ckpt_path)
        )
        
    
    # ========= Evaluate / Plot ===========
    if args.eval:
        from train import build_model_and_dataloader
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, val_loader = build_model_and_dataloader(
            config=config,
            ckpt_path=str(ckpt_path),
            device=device,
            batch_size = hyperparam["batch_size"],
        )
        
        metrics = evaluate(
            model = model,
            dataloader = val_loader,
            device = device,
            save_dir = eval_dir,
            split = "val",
        )
    
        print("Evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # -------------------------------------------------
        # plotting (from saved files)
        # -------------------------------------------------
        ckpt = torch.load(ckpt_path, map_location="cpu")
        plot_loss_curve(
            ckpt["train_loss"],
            ckpt["val_loss"],
            title=f"{config.backbone.upper()} Loss",
            save_path=eval_dir / "loss_curve.png",
        )

        data = torch.load(eval_dir / "eval_val.pt")
        plot_velocity_timecourse(
            data["vel_pred"][0],
            data["vel_true"][0],
            title=f"{config.backbone.upper()} Velocity Decoding",
            save_path=eval_dir / "velocity_example.png",
        )
    

if __name__ == "__main__":
    main()