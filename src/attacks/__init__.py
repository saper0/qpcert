from typing import Any, Dict

import torch
from jaxtyping import Float, Integer

from src.attacks.base_attack import GlobalAttack
from src.attacks.simple_attacks import Noise, Random
#from src.attacks.prbcd import PRBCD


def create_attack(target_idx: int, X: Float[torch.Tensor, "n n"], 
                  A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"],
                  hyperparams: Dict[str, Any], 
                  seed: int=42,
                  model: torch.nn.Module=None) -> GlobalAttack:
    if hyperparams["attack"] == "random":
        return Random(target_idx, A, y, seed)
    elif hyperparams["attack"] == "noise":
        return Noise(target_idx, A, seed)
    else:
        return NotImplementedError("Requested Attack not implemented.")

__all__ = [Random, create_attack]
