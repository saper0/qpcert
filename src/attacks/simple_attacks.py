from typing import Dict, Any

from jaxtyping import Float, Integer
import torch

from base_attack import GlobalAttack

class Random(GlobalAttack):
    """Randomly insert inter-class edges or delete intra-class edges."""

    def __init__(self, target_idx: int, A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"]) -> None:
        self.target_idx = target_idx
        self.A = A
        self.y = y