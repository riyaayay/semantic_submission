"""
GPU device handling. Everything that can run on the GPU must run there: no
silent CPU fallback. If CUDA is unavailable these helpers raise instead.
"""

import torch

_checked = False


def require_cuda() -> torch.device:
    """Returns the CUDA device, or raises if torch or XGBoost lack CUDA."""
    global _checked
    if not _checked:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required, but torch.cuda.is_available() is False.")
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA"):
            raise RuntimeError("CUDA GPU required, but this XGBoost build has no CUDA support.")
        _checked = True
    return torch.device("cuda")


def xgb_gpu_params(params: dict) -> dict:
    require_cuda()
    return {**params, "device": "cuda", "tree_method": "hist"}


def xgb_predict_gpu(booster, X) -> "torch.Tensor":
    """Predicts on the GPU from a CUDA tensor (or array moved to CUDA); returns a CUDA tensor.

    Uses inplace_predict on device memory, so no host copy of the features is made.
    """
    dev = require_cuda()
    booster.set_param({"device": "cuda"})
    Xg = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    Xg = Xg.to(dev, dtype=torch.float32).contiguous()
    best = getattr(booster, "best_iteration", None)
    kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
    out = booster.inplace_predict(Xg, **kwargs)
    return torch.as_tensor(out, device=dev)
