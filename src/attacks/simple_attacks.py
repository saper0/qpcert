from typing import Dict, Any

from jaxtyping import Float, Integer
import numpy as np
import torch

from src.attacks.base_attack import GlobalAttack

class Random(GlobalAttack):
    """Randomly insert inter-class edges or delete intra-class edges."""

    def __init__(self, target_idx: int, A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"]) -> None:
        self.target_idx = target_idx
        self.A = A.clone()
        self.y = y

    def attack(self, n_perturbations: int) -> Float[torch.Tensor, "n n"]:
        pass


class Noise(GlobalAttack):
    """Randomly perturb adjacency matrix. 
    
    Allows for different probabilty of inserting or deleting an edge.
    
    Args:
        p_insert: Optional. Has to sum to 1 with p_delete. 
        p_delete: Optional.
    """
    def __init__(self, target_idx: np.ndarray, A: Float[torch.Tensor, "n n"],
                 seed: int=42, p_insert=None, p_delete=None) -> None:
        if p_insert is not None or p_delete is not None:
            assert False, "Setting p_insert or p_delete not supported now."
            assert p_insert + p_delete == 1
        self.p_insert = p_insert
        self.p_delete = p_delete
        self.target_idx = target_idx
        self.A = A
        self.rng = np.random.Generator(np.random.PCG64(seed)) # currently default numpy RNG

    def attack(self, n_perturbations: int) -> Float[torch.Tensor, "n n"]:
        if n_perturbations == 0:
            return self.A.detach().clone()
        assert len(self.target_idx) == self.A.shape[0]
        # Assume every node can be attacked, assume no self-edges
        row, col = torch.triu_indices(self.A.shape[0], self.A.shape[1], offset=1)
        n = row.shape[0]
        pert_idx = self.rng.permutation(np.arange(n))[:n_perturbations]
        row_pert = row[pert_idx]
        col_pert = col[pert_idx]
        A_pert = self.A.detach().clone()
        A_pert[row_pert, col_pert] *= -1
        A_pert[row_pert, col_pert] += 1
        A_pert = A_pert + A_pert.T
        return A_pert