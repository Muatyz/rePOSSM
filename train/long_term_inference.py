import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from tqdm import tqdm

# Import your existing modules
from models.build_model import build_model
from data.Dataloader import my_dataset, pad_collate_fn, get_inference_dataloader
from train.evaluate import evaluate_session
from models.possm.utils.checkpoint import load_model_from_checkpoint
from scripts.visualize import collect_predictions, plot_prediction_vs_target


def run_long_term_evaluation(
    config,
    hyperparam,
    processed_root, # Session 0 的 meta_data 路径, 用于获取训练分布的均值和标准差
):
    assert hyperparam is not None
    # --- 1. Setup ---
    device = hyperparam["device"]
    model_path = hyperparam["model_path"]
    
    print(f"Using device: {device}")
    
    # --- 2. Load Session 0 Metadata (Training Distribution) ---
    # We must use Session 0's Mean/Std to denormalize predictions 
    # because the model weights are fixed to that output distribution.
    meta_path_0 = os.path.join(processed_root, "session_0/meta_data.json")
    with open(meta_path_0, "r") as f:
        meta_0 = json.load(f)
        
    config.num_channel = meta_0["num_channel"]
     
    # --- 3. Load Model ---
    print(f"Loading model from {model_path}...")
    model = build_model(config).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    # These tensors are used to denormalize the model output
    train_mean = torch.tensor(meta_0["vel_mean"], device=device, dtype=torch.float32)
    train_std = torch.tensor(meta_0["vel_std"], device=device, dtype=torch.float32)
    
    print(f"Training Baseline (Session 0) Loaded.")
    print("-" * 50)

    # --- 4. Inference Loop (Session 1 to 11) ---
    # We can also include Session 0 to verify training performance
    results_summary = []
    
    for session_id in range(12): # 0 to 11
        data_path = os.path.join(processed_root, f"session_{session_id}/sliced_trials.pt")
        meta_data_path = os.path.join(processed_root, f"session_{session_id}/meta_data.json")
        num_channel = json.load(open(meta_data_path, "r"))["num_channel"]
        if num_channel != 96:
            continue
        loader = get_inference_dataloader(data_path)
        
        if loader is None:
            continue
            
        print(f"Evaluating Session {session_id}...", end=" ")
        
        metrics = evaluate_session(model, loader, device, config, train_mean, train_std)
        
        # # ===== 可视化（只挑几个 session）=====
        # if session_id in [0, 3, 11]:   # ⭐ 你可以改
        #     pred, target = collect_predictions(
        #         model,
        #         loader,
        #         device,
        #         train_mean,
        #         train_std,
        #         max_batches=1
        #     )

        #     fig_dir = os.path.join(os.getcwd(), "figures")
        #     os.makedirs(fig_dir, exist_ok=True)

        #     fig_path = os.path.join(fig_dir, f"session_{session_id}.png")

        #     plot_prediction_vs_target(pred, target, session_id, fig_path)

        #     print(f"[Saved plot] {fig_path}")
        
        if metrics:
            print(f"R2: {metrics['avg_r2']:.4f} (MSE: {metrics['mse']:.4f})")
            results_summary.append({
                "session": session_id,
                **metrics
            })
        else:
            print("Failed (No valid data)")

    # --- 5. Final Report ---
    print("\n" + "="*50)
    print(f"{'Session':<10} | {'R2 (Avg)':<10} | {'R2 (X)':<10} | {'R2 (Y)':<10} | {'MSE':<10}")
    print("-" * 50)
    for res in results_summary:
        print(f"{res['session']:<10} | {res['avg_r2']:<10.4f} | {res['r2_x']:<10.4f} | {res['r2_y']:<10.4f} | {res['mse']:<10.4f}")
    print("="*50)
    
    # --- 6. Save Results ---
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 从 config / hyperparam 解析信息
    model_name = getattr(config, "model_type", "model")
    backbone = getattr(config, "backbone", "unknown")
    seed = getattr(config, "seed", "unknown")

    # 兼容你现在 ckpt 命名（例如 possm_gru_seed42_xxx.pt）
    if "model_path" in hyperparam:
        ckpt_name = os.path.basename(hyperparam["model_path"]).replace(".pt", "")
    else:
        ckpt_name = f"{model_name}_{backbone}_seed{seed}"

    save_path = os.path.join(results_dir, f"{ckpt_name}.json")

    with open(save_path, "w") as f:
        json.dump(results_summary, f, indent=4)

    print(f"\nResults saved to: {save_path}")
    

if __name__ == "__main__":
    run_long_term_evaluation()