# metrics.py
import numpy as np
import torch


def _to_numpy(x):
    """
    Utility: convert torch.Tensor or np.ndarray to np.ndarray
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


def mse_loss(pred, target):
    """
    Mean Squared Error

    Args:
        pred, target: shape (..., D)
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    assert pred.shape == target.shape, \
        f"Shape mismatch: {pred.shape} vs {target.shape}"

    return np.mean((pred - target) ** 2)


def rmse_loss(pred, target):
    """
    Root Mean Squared Error
    """
    return np.sqrt(mse_loss(pred, target))


def r2_score(pred, target, eps=1e-8):
    """
    R^2 (coefficient of determination)

    Computed over all dimensions jointly.
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target, axis=0)) ** 2)

    return 1.0 - ss_res / (ss_tot + eps)


def pearson_corr(pred, target, eps=1e-8):
    """
    Pearson correlation coefficient per dimension

    Returns:
        corr: np.ndarray of shape (D,)
    """
    pred = _to_numpy(pred)
    target = _to_numpy(target)

    assert pred.shape == target.shape

    D = pred.shape[-1]
    corrs = []

    for d in range(D):
        x = pred[..., d].reshape(-1)
        y = target[..., d].reshape(-1)

        x = x - x.mean()
        y = y - y.mean()

        corr = np.sum(x * y) / (np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2)) + eps)
        corrs.append(corr)

    return np.array(corrs)


def evaluate_all(pred, target):
    """
    Convenience wrapper to compute all common metrics.

    Returns:
        dict
    """
    return {
        "mse": float(mse_loss(pred, target)),
        "rmse": float(rmse_loss(pred, target)),
        "r2": float(r2_score(pred, target)),
        "corr": pearson_corr(pred, target).tolist(),
    }
