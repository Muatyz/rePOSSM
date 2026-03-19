import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_results(result_dir):
    """
    自动读取 results 目录下所有 json
    文件命名规则：
    possm_{model}_seed{seed}.json
    """
    data = defaultdict(lambda: defaultdict(list))
    
    for file in os.listdir(result_dir):
        if not file.endswith(".json"):
            continue
        
        path = os.path.join(result_dir, file)
        
        # 解析文件名
        # possm_gru_seed42.json
        parts = file.replace(".json", "").split("_")
        model = parts[1]
        seed = parts[-1]
        
        results = json.load(open(path))
        
        # 按 session 排序
        results = sorted(results, key=lambda x: x["session"])
        
        r2 = [x["avg_r2"] for x in results]
        sessions = [x["session"] for x in results]
        
        data[model]["sessions"] = sessions
        data[model]["avg_r2"].append(r2)
    
    return data


def compute_mean_std(data):
    stats = {}
    
    for model, d in data.items():
        r2_array = np.array(d["avg_r2"])  # (num_seeds, num_sessions)
        
        stats[model] = {
            "sessions": d["sessions"],
            "mean": r2_array.mean(axis=0),
            "std": r2_array.std(axis=0),
            "overall_mean": r2_array.mean()
        }
    
    return stats


def plot_r2(stats, save_path=None):
    plt.figure(figsize=(8, 5))
    
    for model, d in stats.items():
        sessions = d["sessions"]
        mean = d["mean"]
        std = d["std"]
        
        plt.plot(sessions, mean, marker='o', label=model)
        plt.fill_between(sessions, mean - std, mean + std, alpha=0.2)
    
    plt.axhline(0, linestyle='--')
    plt.xlabel("Session")
    plt.ylabel("R2")
    plt.title("Model Comparison (R2)")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def print_summary(stats):
    print("\n=== Summary ===")
    for model, d in stats.items():
        print(f"{model:12s} | Avg R2: {d['overall_mean']:.4f}")


def main():
    result_dir = "./results"
    save_path = "./figures/r2_comparison.png"
    
    os.makedirs("./figures", exist_ok=True)
    
    data = load_results(result_dir)
    stats = compute_mean_std(data)
    
    print_summary(stats)
    plot_r2(stats, save_path)


if __name__ == "__main__":
    main()