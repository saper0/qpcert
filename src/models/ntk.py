# Label Propagation implementation with Code & Comments mainly taken from 
# PyTorch Geometric implementation of the Correct and Smooth Framework:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/correct_and_smooth.html #noqa
from typing import Any, Dict, Optional, Union, Tuple

from jaxtyping import Float, Integer
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from src.models.common import row_normalize, tbn_normalize, degree_scaling


class NTK(torch.nn.Module):
    r"""NTK for Message Passing NNs as derived in  "Analysis of Convolutions, 
        Non-linearity and Depth in Graph Neural Networks using Neural Tangent 
        Kernel".
    """
    def __init__(self, X: Float[torch.Tensor, "n d"], A: Float[torch.Tensor, "n n"], 
                 model_dict: Dict[str, Any], dtype=torch.float64):
        super().__init__()
        self.model_dict = model_dict
        assert self.model_dict["model"] == "GCN" \
            or self.model_dict["model"] == "SoftMedoid"
        self.dtype = dtype
        self.device = X.device
        self.ntk = self.calc_ntk(X, A)

    def calc_diffusion(self, X: torch.Tensor, A: torch.Tensor):
        if self.model_dict["model"] == "GCN":
            if self.model_dict["normalization"] == "row_normalization":
                return row_normalize(A)
            elif self.model_dict["normalization"] == "trust_biggest_neighbor":
                return tbn_normalize(A)
            elif self.model_dict["normalization"] == "degree_scaling":
                return degree_scaling(A, self.model_dict["gamma"], 
                                      self.model_dict["delta"])
            else:
                raise NotImplementedError("Normalization not supported")
        elif self.model_dict["model"] == "SoftMedoid":
            # Row normalized implementation
            n = X.shape[0]
            d = X.shape[1]
            X_view = X.view((1, n, d))
            dist = torch.cdist(X_view, X_view, p=2).view(n, n)
            A_self = A + torch.eye(n)
            S = torch.exp(- (1 / self.model_dict["T"]) * (A_self @ dist))
            normalization = torch.einsum("ij,ij->i", A_self, S)
            S = (S*A_self) / normalization[:, None]
            return S
        else:
            raise NotImplementedError("Only GCN/SoftMedoid architecture implemented")

    def kappa_0(self, u):
        z = torch.zeros((u.shape), dtype=self.dtype).to(self.device)
        pi = torch.acos(z)*2
        r = (pi - torch.acos(u)) / pi
        r[r!=r] = 1.0 # Always False?
        return r

    def kappa_1(self, u):
        z = torch.zeros((u.shape), dtype=self.dtype).to(self.device)
        pi = torch.acos(z) * 2
        r = (u*(pi - torch.acos(u)) + torch.sqrt(1-u*u))/pi
        r[r!=r] = 1.0 # Always False?
        return r

    def calc_ntk(self, X: Float[torch.Tensor, "n d"], 
                 A: Float[torch.Tensor, "n n"]):
        """Calculate and return ntk matrix."""
        if "normalize" in self.model_dict:
            if self.model_dict["normalize"]:
                S = self.calc_diffusion(X, A)
            else:
                S = A
        else:
            S = self.calc_diffusion(X, A)
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

    def forward(self, 
                X: Float[torch.Tensor, "n d"],
                A: Union[SparseTensor,
                     Tuple[Integer[torch.Tensor, "2 nnz"], Float[torch.Tensor, "nnz"]],
                     Float[torch.Tensor, "n_nodes n_nodes"]],
                y: Integer[torch.Tensor, "n"],
                idx_labeled: Integer[torch.Tensor, "m"],
                idx_unlabeled: Integer[torch.Tensor, "u"]
                ) -> Tensor:
        """Perform kernel regression.

        Note: n = m + u

        Returns:
            Tensor: Resulting (soft) labeling from label propagation.
        """
         # handle different adj representations
        assert isinstance(A, tuple)
        if isinstance(A, SparseTensor):
            A = A.to_dense() # is differentiable
        elif isinstance(A, tuple):
            n, _ = X.shape
            A = torch.sparse_coo_tensor(*A, 2 * [n]).to_dense() # is differentiable
        
        ntk = self.calc_ntk(X, A)
        ntk_labeled = ntk[idx_labeled,:][:,idx_labeled]
        ntk_unlabeled = ntk[idx_unlabeled,:][:,idx_labeled]
        M = torch.linalg.solve
        return ntk
