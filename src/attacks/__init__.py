from typing import Any, Dict

import numpy as np
import torch
from jaxtyping import Float, Integer

from src.attacks.base_attack import GlobalAttack
from src.attacks.simple_attacks import Noise, Random
from src.attacks.prbcd import PRBCD


def create_attack(idx_target: Integer[torch.Tensor, "t"],
                  X: Float[torch.Tensor, "n n"], 
                  A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"],
                  idx_labeled: Integer[np.ndarray, "m"],
                  idx_unlabeled: Integer[np.ndarray, "u"],
                  attack_params: Dict[str, Any], 
                  seed: int=42,
                  model: torch.nn.Module=None,
                  *args,
                  **kwargs) -> GlobalAttack:
    if attack_params["attack"] == "random":
        return Random(idx_target, A, y, seed)
    elif attack_params["attack"] == "noise":
        return Noise(idx_target, A, seed)
    elif attack_params["attack"] == "prbcd":
        return PRBCD(A=A, X=X, y=y, idx_attack=idx_target, 
                     idx_labeled=idx_labeled, idx_unlabeled=idx_unlabeled, 
                     model=model, *args, **kwargs, **attack_params)
    else:
        return NotImplementedError("Requested Attack not implemented.")

__all__ = [Random, create_attack]
