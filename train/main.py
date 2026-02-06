# main.py
# 省略实验细节, 只进行宏观函数使用
# 用法示例: python main.py --train --backbone gru

import os
import torch
import argparse
from pathlib import Path

from possm.config.Config import my_POSSMConfig
from train.train import run_experiment
from scripts.plotting import (
    plot_loss_curve,
    plot_velocity_timecourse,
)
from train.long_term_inference import run_long_term_evaluation


def main():
    # 等待进一步传参
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action = "store_true")
    parser.add_argument("--eval", action = "store_true")
    parser.add_argument("--backbone", type=str, default = "gru")
    parser.add_argument("--seed", type=int, default = 42)
    args = parser.parse_args()
    
    #  引入 config 参数
    config = my_POSSMConfig()
    config.backbone = args.backbone # 选取 backbone 参数, 目前支持 gru/s4d
    
    #  设定模型存储路径
    # ckpt_dir = Path("checkpoints")
    # ckpt_dir.mkdir(exist_ok = True)

    # ckpt_path = ckpt_dir / f"possm_{config.backbone}_seed{args.seed}.pt"
    # eval_dir = Path("eval") / f"{config.backbone}_seed{args.seed}"
    # eval_dir.mkdir(parents = True, exist_ok = True)
    
    # 全局超参数
    hyperparam = {
        "seed": args.seed,
        "batch_size": 64,  # 原设定为 256, 但是会显存不足, 正在排查原因中
        "num_epochs": 40,
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "patience": 20, # 早停法耐心值
        "log_dir": f"./long_term_log/{config.backbone}",
        "model_path": f"./checkpoints/long_term_model_{config.backbone}.pt",
        }
    
    #  根据传参, 决定训练或者评估模型
    print("=" * 60)
    print(f"Running POSSM with backbone = {config.backbone}")
    print(f"Model is saved at: {hyperparam['model_path']}")
    print("=" * 60)
    

    # ============ Train ==============
    if args.train:
        run_experiment(
            config = config,
            hyperparam = hyperparam,
            data_dir = "data/long_term_data/Chewie_processed/session_0/sliced_trials.pt", # 预处理后数据的路径
            meta_data_path = "data/long_term_data/Chewie_processed/session_0/meta_data.json", # meta_data 的路径
        )
        
    
    # ========= Evaluate ===========
    if args.eval:
        run_long_term_evaluation(
            config = config,
            hyperparam = hyperparam,
            processed_root = "data/long_term_data/Chewie_processed" # 处理后数据的根目录, 下面应该有 session_0 到 session_11 的子目录
        )
        
    

if __name__ == "__main__":
    main()