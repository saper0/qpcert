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

from src.models.common import row_normalize, tbn_normalize, degree_scaling, sym_normalize, APPNP_propogation


class NTK(torch.nn.Module):
    r"""NTK for Message Passing NNs as derived in  "Analysis of Convolutions, 
        Non-linearity and Depth in Graph Neural Networks using Neural Tangent 
        Kernel".
    """
    def __init__(self, model_dict: Dict[str, Any], 
                 X_trn: Float[torch.Tensor, "n d"], 
                 A_trn: Float[torch.Tensor, "n n"],
                 learning_setting: str = "inductive", 
                 device: Union[torch.device, str] = None,
                 dtype: torch.dtype = torch.float64):
        """ In the forward pass, the here calculated NTK will be considered
            as calculated from the training graph. 

        Parameters
        ----------
        X_trn : Float[torch.Tensor, "n d"]
            Node features available during training (i.e., of the training
            graph)-
        A_trn : Float[torch.Tensor, "n n"]
            Graph adjacency matrix available during training (i.e., of the
            training graph).
        learning_setting : str 
            Considered learning setting for inference. Can be "inductive" 
            (default) or "transductive".
        device : Union[torch.device, str]
            Device to use for calculating the kernel and doing inference
            with it. If not set, will be set to device of X_trn.
        dtype : torch.dtype
            Precision of the kernel matrix. Default: torch.float64
        """
        super().__init__()
        self.model_dict = model_dict
        assert self.model_dict["model"] == "GCN" \
            or self.model_dict["model"] == "SoftMedoid" \
            or self.model_dict["model"] == "PPNP" \
            or self.model_dict["model"] == "APPNP"
        self.dtype = dtype
        if device is not None:
            self.device = device
        else:
            self.device = X_trn.device
        self.ntk = self.calc_ntk(X_trn, A_trn)
        self.learning_setting = learning_setting

    def calc_diffusion(self, X: torch.Tensor, A: torch.Tensor):
        if self.model_dict["model"] == "GCN":
            if self.model_dict["normalization"] == "row_normalization":
                return row_normalize(A)
            elif self.model_dict["normalization"] == "sym_normalization":
                return sym_normalize(A)
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
            A_self = A + torch.eye(n).to(self.device)
            S = torch.exp(- (1 / self.model_dict["T"]) * (A_self @ dist))
            normalization = torch.einsum("ij,ij->i", A_self, S)
            S = (S*A_self) / normalization[:, None]
            return S
        elif self.model_dict["model"] == "PPNP" or self.model_dict["model"] == "APPNP":
            exact = True if self.model_dict["model"]=="PPNP" else False
            return APPNP_propogation(A, alpha=self.model_dict["alpha"], 
                                     iteration=self.model_dict["iteration"], 
                                     exact=exact)
        else:
            raise NotImplementedError("Only GCN/SoftMedoid/(A)PPNP architecture implemented")

    def kappa_0(self, u):
        z = torch.zeros((u.shape), dtype=self.dtype).to(self.device)
        pi = torch.acos(z)*2
        r = (pi - torch.acos(u-1e-7)) / pi
        #r[r!=r] = 1.0 # Always False?
        return r

    def kappa_1(self, u):
        z = torch.zeros((u.shape), dtype=self.dtype).to(self.device)
        pi = torch.acos(z) * 2
        r = (u*(pi - torch.acos(u-1e-7)) + torch.sqrt(1-u*u+1e-7))/pi
        #r[r!=r] = 1.0 # Always False?
        return r

    def calc_ntk(self, X: Float[torch.Tensor, "n d"], 
                 A: Float[torch.Tensor, "n n"]):
        """Calculate and return ntk matrix."""
        if isinstance(A, SparseTensor):
            A = A.to_dense() 
        elif isinstance(A, tuple):
            n, _ = X.shape
            A = torch.sparse_coo_tensor(*A, 2 * [n]).to_dense() 
        
        if "normalize" in self.model_dict:
            if self.model_dict["normalize"]:
                S = self.calc_diffusion(X, A)
            else:
                S = A
        else:
            S = self.calc_diffusion(X, A)
        if self.model_dict["model"] == "GCN" or self.model_dict["model"] == "SoftMedoid":
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
                E = E.double()
                E_der = E_der.double()

                kernel_sub[i] = S.matmul((Sig * E_der).matmul(S.T))
            
                Sig = S.matmul(E.matmul(S.T))
                for j in range(i):
                    kernel_sub[j] = S.matmul((kernel_sub[j].float() * E_der).matmul(S.T))

            kernel += torch.sum(kernel_sub, dim=0)
            kernel += Sig
            return kernel
        
        elif self.model_dict["model"] == "PPNP" or self.model_dict["model"] == "APPNP":
            # NTK for PPNP with one hidden layer fcn for features
            # (A)PPNP = S ( ReLU(XW_1 + b_1) W_2 + b_2)
            kernel = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
            XXT = X.matmul(X.T)
            B = torch.ones((S.shape), dtype=self.dtype).to(self.device)
            Sig = XXT+B
            p = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
            Diag_Sig = torch.diagonal(Sig) 
            Sig_i = p + Diag_Sig.reshape(1, -1)
            Sig_j = p + Diag_Sig.reshape(-1, 1)
            q = torch.sqrt(Sig_i * Sig_j)
            u = Sig/q 
            E = (q * self.kappa_1(u)) 
            E_der = (self.kappa_0(u))
            E = E.double()
            E_der = E_der.double()
            kernel += S.matmul(Sig * E_der).matmul(S.T)
            kernel += S.matmul(E+B).matmul(S.T)
            return kernel
    
    def get_ntk(self):
        """Return (precomputed) ntk matrix."""
        return self.ntk

    def forward(self, 
                idx_labeled: Integer[np.ndarray, "m"],
                idx_unlabeled: Integer[np.ndarray, "u"],
                y_test: Integer[torch.Tensor, "n"],
                X_test: Float[torch.Tensor, "n d"] = None,
                A_test: Union[SparseTensor,
                     Tuple[Integer[torch.Tensor, "2 nnz"], Float[torch.Tensor, "nnz"]],
                     Float[torch.Tensor, "n_nodes n_nodes"]] = None,
                learning_setting: Optional[str] = None,
                return_ntk: bool = False,
                solution_method: str = "LU",
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Perform kernel regression using the NTK.

        The NTK of the test-graph is calculated using X_test & A_test, except
        if they are None and the learning setting set to transductive. Then,
        uses initialized A & X (corresponding to training graph) for prediction.
        
        Parameters
        ----------
        idx_labeled : Integer[np.ndarray, "m"]
            Indices of labeled nodes in X_test / A_test.
            Note: Assumes training graph is fully labeled!
        idx_unlabeled : Integer[np.ndarray, "u"]
            Indices of unlabeled nodes in X_test / A_test.
        y_test : Integer[torch.Tensor, "n"]
            Labels of the test graph.
        X_test : Float[torch.Tensor, "n d"]
            Node features available during testing (i.e., of the test graph). 
        A_test : Float[torch.Tensor, "n n"]
            Graph adjacency matrix available during testing (i.e., of the test
            graph).
        learning_setting : Optional[str] 
            Optional, per default uses the learning setting set when initializing
            the NTK object. However, if set, inference will be done with the
            here set learning_setting instead. Options: "inductive" (default) 
            or "transductive".
        return_ntk : Optional[bool]
            If true, return the NTK of the test-graph calculated using X_test
            and A_test. Defaul: False

        Note: n = m + u 

        Returns: 
            Logits of unlabeled nodes, defines as in
            https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        """
         # handle different adj representations
        #assert isinstance(A, tuple)
        if learning_setting is None:
            learning_setting = self.learning_setting

        if X_test is None or A_test is None:
            assert X_test is None and A_test is None
            assert learning_setting == "transductive", "No test graph given, " \
                + " thus learning setting must be transductive for inference."
            ntk_test = self.ntk
        else:
            if isinstance(A_test, SparseTensor):
                A_test = A_test.to_dense() # is differentiable
            elif isinstance(A_test, tuple):
                n, _ = X_test.shape
                A_test = torch.sparse_coo_tensor(*A_test, 2 * [n]).to_dense() # is differentiable
            ntk_test = self.calc_ntk(X_test, A_test)
        
        if learning_setting == "inductive":
            ntk_labeled = self.ntk # given fully labeled
            assert self.ntk.shape[0] == len(idx_labeled), "Semi-supervised " \
                + "setting not implemented by NTK class for inductive inference" \
                + "so far."
            ntk_unlabeled = ntk_test[idx_unlabeled,:][:,idx_labeled]
        if learning_setting == "transductive": 
            ntk_labeled = self.ntk[idx_labeled,:][:,idx_labeled]
            ntk_unlabeled = ntk_test[idx_unlabeled,:][:,idx_labeled]
        # ensure non-singularity
        ntk_labeled += (torch.rand(ntk_labeled.shape) / 1e+4).to(self.device)
        cond = torch.linalg.cond(ntk_labeled)

        # Previous
        M = torch.linalg.solve(ntk_labeled, ntk_unlabeled, left=False) # ntk_unlabeled * ntk_labeled^-1
        y_pred = torch.matmul(M,(y_test[idx_labeled] * 2 - 1).to(dtype=torch.float64))

        #M = 

        if return_ntk:
            return y_pred, ntk_test, cond, M, ntk_labeled, ntk_unlabeled
        return y_pred
