from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from jaxtyping import Float, Integer


class NTK_OLD(torch.nn.Module):
    r"""NTK
    """
    def __init__(self, X: Float[torch.Tensor, "n d"], S: Float[torch.Tensor, "n n"], 
                 model_dict: Dict[str, Any], dtype=torch.float64):
        super().__init__()
        self.model_dict = model_dict
        self.dtype = dtype
        self.device = X.device
        self.ntk = self.calc_ntk(X, S)

    def kappa_0(self, u):
        z = torch.zeros((u.shape), dtype=self.dtype).to(self.device)
        pi = torch.acos(z)*2
        r = (pi - torch.acos(u)) / pi
        r[r!=r] = 1.0
        return r

    def kappa_1(self, u):
        z = torch.zeros((u.shape), dtype=self.dtype).to(self.device)
        pi = torch.acos(z) * 2
        r = (u*(pi - torch.acos(u)) + torch.sqrt(1-u*u))/pi
        r[r!=r] = 1.0
        return r

    def calc_ntk(self, X: Float[torch.Tensor, "n d"], 
                 S: Float[torch.Tensor, "n n"]):
        """Calculate and return ntk matrix."""
        assert self.model_dict["model"] == "GCN" or self.model_dict["model"] == "SoftMedoid"
        csigma = 1 
        S_norm = torch.norm(S)
        XXT = X.matmul(X.T)
        Sig = S.matmul(XXT.matmul(S.T))
        kernel = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        # ReLu GCN
        depth = self.model_dict["depth"]
        kernel_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                 dtype=self.dtype).to(self.device)
        for i in range(depth):
            p = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
            Diag_Sig = torch.diagonal(Sig) 
            Sig_i = p + Diag_Sig.reshape(1, -1)
            Sig_j = p + Diag_Sig.reshape(-1, 1)
            q = torch.sqrt(Sig_i * Sig_j)
            u = Sig/q # why normalization?
            E = (q * self.kappa_1(u)) * csigma
            E_der = (self.kappa_0(u)) * csigma
            kernel_der = (S.matmul(S.T)) * E_der
            kernel_sub[i] += Sig * kernel_der
            E = E.double()
            Sig = S.matmul(E.matmul(S.T))
            for j in range(i):
                kernel_sub[j] *= kernel_der
        kernel += torch.sum(kernel_sub, dim=0)
        kernel += Sig
        return kernel
    
    def get_ntk(self):
        """Return (precomputed) ntk matrix."""
        return self.ntk

    def forward(self) -> Tensor:
        """Perform kernel regression

        Returns:
            Tensor: Resulting (soft) labeling from label propagation.
        """
        raise NotImplementedError("So far NTK can't be used for prediction")
