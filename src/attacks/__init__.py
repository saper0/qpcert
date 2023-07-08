from typing import Any, Dict

import torch
from jaxtyping import Float, Integer

from base_attack import GlobalAttack
from simple_attacks import Random #, L2weak, L2strong


def create_attack(target_idx: int, X: Float[torch.Tensor, "n n"], 
                  A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"],
                  hyperparams: Dict[str, Any], 
                  model: torch.nn.Module) -> GlobalAttack:
    if hyperparams["attack"] == "Random":
        return Random(target_idx, A, y, hyperparams)
    else:
        return NotImplementedError("Requested Attack not implemented.")

__all__ = [Random, create_attack]
