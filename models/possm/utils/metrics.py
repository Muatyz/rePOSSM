# metrics.py
# 用于计算模型性能指标
import numpy as np
import torch
from sklearn.metrics import r2_score

def calculate_r2(y_true, y_pred):
    """
    Calculate R2 score for 2D velocity.
    Returns the average R2 across X and Y dimensions.
    """
    # y_true, y_pred shape: (N_samples, 2)
    r2_x = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_y = r2_score(y_true[:, 1], y_pred[:, 1])
    return (r2_x + r2_y) / 2, r2_x, r2_y