# src/utils/seedUtils.py
import os
import random
import numpy as np

def setGlobalSeed(seed: int) -> None:
    """
    Set seeds for reproducibility (CPU deterministic-ish).
    Note: Full determinism in PyTorch may reduce performance; set torch flags in training script if needed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not installed or not needed in some contexts
        pass