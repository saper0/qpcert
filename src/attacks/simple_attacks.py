from typing import Dict, Any

from jaxtyping import Float, Integer
import numpy as np
import torch

from src.attacks.base_attack import GlobalAttack

class Random(GlobalAttack):
    """Randomly insert inter-class edges or TODO: delete intra-class edges.
    
    Assumes two classes and A sorted after class without self-loops.
    """

    def __init__(self, target_idx: int, A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"], seed: int=42) -> None:
        self.target_idx = target_idx
        self.A = A.clone()
        self.y = y
        self.n_class0 = (y==0).sum()
        self.rng = np.random.Generator(np.random.PCG64(seed)) # currently default numpy RNG

    def attack(self, n_perturbations: int) -> Float[torch.Tensor, "n n"]:
        # Inverse A mask
        A_mask = self.A.detach().clone() * -1
        A_mask += 1
        A_mask = A_mask.to(torch.bool)
        n = self.A.shape[0]
        M = torch.zeros((n,n), dtype=torch.bool, device=self.A.device)
        M[self.target_idx, :] = True
        M[:, self.target_idx] = True
        # Exclude edges connecting class 0 nodes to class 1 nodes
        M[:self.n_class0, self.n_class0:] = \
            M[:self.n_class0, self.n_class0:].logical_and(A_mask[:self.n_class0, self.n_class0:])
        M[:self.n_class0, :self.n_class0] = False
        M[self.n_class0:, self.n_class0:] = False
        triu_idx = torch.triu_indices(n,n,offset=1,device=M.device)
        M_triu = M[triu_idx[0, :], triu_idx[1,:]]
        row_idx = triu_idx[0,:][M_triu]
        col_idx = triu_idx[1,:][M_triu]
        m = row_idx.shape[0]
        pert_idx = self.rng.permutation(np.arange(m))[:n_perturbations]
        row_pert = row_idx[pert_idx]
        col_pert = col_idx[pert_idx]
        A_pert = self.A.detach().clone()
        A_pert[row_pert, col_pert] *= -1
        A_pert[row_pert, col_pert] += 1
        # A is symmetric
        A_pert[col_pert, row_pert] *= -1 
        A_pert[col_pert, row_pert] += 1
        #A_pert = A_pert + A_pert.T
        return A_pert


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
        """
        for att idx
        n = self.A.shape[0]
        M = torch.zeros((n,n), dtype=torch.bool)
        M[self.target_idx, :] = True
        M[:, self.target_idx] = True
        triu_idx = torch.triu_indices(n,n,offset=1)
        M_triu = M[triu_idx[0, :], triu_idx[1,:]]
        row_idx = triu_idx[0,:][M_triu]
        col_idx = triu_idx[1,:][M_triu]
        m = row_idx.shape[0]
        pert_idx = self.rng.permutation(np.arange(m))[:n_perturbations]
        row_pert = row_idx[pert_idx]
        col_pert = col_idx[pert_idx]
        A_pert = self.A.detach().clone()
        A_pert[row_pert, col_pert] *= -1
        A_pert[row_pert, col_pert] += 1
        A_pert = A_pert + A_pert.T
        """
        return A_pert