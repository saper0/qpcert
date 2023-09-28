from typing import Any, Dict

import numpy as np
import torch
from jaxtyping import Float, Integer

from src.attacks.base_attack import GlobalAttack
from src.attacks.simple_attacks import Noise, Random
from src.attacks.prbcd import PRBCD


def create_attack(target_idx: int, X: Float[torch.Tensor, "n n"], 
                  A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"],
                  idx_labeled: Integer[np.ndarray, "m"],
                  idx_unlabeled: Integer[np.ndarray, "u"],
                  hyperparams: Dict[str, Any], 
                  seed: int=42,
                  model: torch.nn.Module=None,
                  *args,
                  **kwargs) -> GlobalAttack:
    if hyperparams["attack"] == "random":
        return Random(target_idx, A, y, seed)
    elif hyperparams["attack"] == "noise":
        return Noise(target_idx, A, seed)
    elif hyperparams["attack"] == "prbcd":
        return PRBCD(A=A, X=X, y=y, idx_attack=target_idx, 
                     idx_labeled=idx_labeled, idx_unlabeled=idx_unlabeled, 
                     model=model, *args, **kwargs, **hyperparams)
    else:
        return NotImplementedError("Requested Attack not implemented.")

__all__ = [Random, create_attack]
