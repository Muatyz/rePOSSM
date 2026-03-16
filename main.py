# main.py
# 省略实验细节, 只进行宏观函数使用

# 等待完成: 
# 未来命令设计: python main.py --train --model possm --backbone gru
# 以及作为 baseline 的命令: python main.py --train --model rnn

import torch
import argparse # 命令行参数
from pathlib import Path # 路径管理
import hashlib
import time

from configs.possm_config import POSSMConfig
from train.train import run_experiment
from train.long_term_inference import run_long_term_evaluation

from scripts.plotting import (
    plot_loss_curve,
    plot_velocity_timecourse,
)

# ======================
# Project root
# ======================
ROOT = Path(__file__).resolve().parent

DATA_ROOT = ROOT / "data/dataset/long_term_data/Chewie_processed"
CHECKPOINT_DIR = ROOT / "checkpoints" # 模型权重保存路径
LOG_DIR = ROOT / "long_term_log"

# 自动创建目录
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ======================
# Build config
# ======================
def build_config(args):
    '''
    根据命令行参数选择 config 配置
    '''
    if args.model == "possm":
        config = POSSMConfig()
        config.model = "possm"
        config.backbone = args.backbone
        return config

    if args.model == "rnn":
        return None # 等待完善

    if args.model == "transformer":
        return None # 等待完善

    raise ValueError(f"Unknown model: {args.model}")

def generate_run_id(length=6):
    """
    根据当前时间生成一个默认长度为 6 的随机码用于区分实验
    """
    raw = str(time.time()).encode()
    run_id = hashlib.sha1(raw).hexdigest()[:length]
    return run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    
    # 选择模型: possm, rnn, transformer
    parser.add_argument(
        "--model",
        type=str,
        default="possm",
        choices=["possm", "rnn", "transformer"]
    )
    
    # 选择 possm 的 backbone: gru, s4d
    parser.add_argument(
        "--backbone",
        type=str,
        default="gru",
        choices=["gru", "s4d"]
    )
    
    parser.add_argument("--seed", type=int, default=42)
    
    # 选择 eval 时的模型, 通过 6 位时间码指定
    parser.add_argument(
        "--ckpt", 
        type=str,
        default=None,
        help="checkpoint path for evaluation"
    )
    
    args = parser.parse_args()
    if not args.train and not args.eval:
        raise ValueError("Please specify --train or --eval") # 只能指定 train 或 eval 中的一个
    
    # ======================
    # Build config
    # ======================
    config = build_config(args)
    backbone_name = args.backbone if args.model == "possm" else "none"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================
    # Train
    # ======================
    if args.train:
        run_id = generate_run_id()
        model_path = CHECKPOINT_DIR / f"{args.model}_{backbone_name}_seed{args.seed}_{run_id}.pt"
        
        hyperparam = {
            "seed": args.seed,
            "batch_size": 64,
            "num_epochs": 40,
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "device": device,
            "patience": 20,
            "log_dir": LOG_DIR / args.model,
            "model_path": model_path,
        }
        
        # 打印训练实验的信息
        print("=" * 60)
        print(f"Model type: {args.model}")

        if args.model == "possm":
            print(f"Backbone: {args.backbone}")

        print(f"Device: {device}")
        print(f"Run ID: {run_id}")
        print(f"Checkpoint: {model_path}")
        print("=" * 60)
        
        run_experiment(
            config = config,
            hyperparam = hyperparam,
            data_dir = DATA_ROOT / "session_0/sliced_trials.pt", # 预处理后数据的路径
            meta_data_path = DATA_ROOT / "session_0/meta_data.json", # meta_data 的路径
        )
    
    # ======================
    # Evaluate
    # ======================
    if args.eval:
        # 核查是否指定模型
        if args.ckpt is None:
            raise ValueError("Evaluation requires --ckpt")
        
        ckpt_arg = args.ckpt
        # 如果用户直接给完整路径
        if ckpt_arg.endswith(".pt"):
            model_path = Path(ckpt_arg)
            
        # 如果只给 run_id
        else:
            backbone_name = args.backbone if args.model == "possm" else "none"
            model_name = f"{args.model}_{backbone_name}_seed{args.seed}_{ckpt_arg}.pt"
            model_path = CHECKPOINT_DIR / model_name
            
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        hyperparam = {
            "seed": args.seed,
            "device": device,
            "model_path": model_path,
        }
        
        # 打印评估实验的信息
        print("=" * 60)
        print(f"Model type: {args.model}")

        if args.model == "possm":
            print(f"Backbone: {args.backbone}")

        print(f"Device: {device}")
        print(f"Loading checkpoint: {model_path}")
        print("=" * 60)
        
        run_long_term_evaluation(
            config = config,
            hyperparam = hyperparam,
            processed_root = DATA_ROOT # 处理后数据的根目录, 下面应该有 session_0 到 session_11 的子目录
        )

if __name__ == "__main__":
    main()