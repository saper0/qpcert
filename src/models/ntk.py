# Label Propagation implementation with Code & Comments mainly taken from 
# PyTorch Geometric implementation of the Correct and Smooth Framework:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/correct_and_smooth.html #noqa
from typing import Any, Dict, Optional, Union, Tuple

from jaxtyping import Float, Integer
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
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
                 n_classes: float,
                 idx_trn_labeled: Optional[Integer[torch.Tensor, "l"]] = None,
                 y_trn: Optional[Integer[torch.Tensor, "l"]] = None,
                 learning_setting: str = "inductive", 
                 pred_method: str = "svm",
                 regularizer: float = 1e-8,
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
        n_classes : float
            Number of classes in given training graph. Used to decide if
            binary or multi-class classification setting.
        idx_trn_labeled: Integer[torch.Tensor, "l"]
            Used only in semi-supervised learning setting, i.e. if not all nodes
            in G_trn = (X_trn, A_trn) have labeles. Represents the indices of 
            labeled nodes in X_trn, A_trn.
        y_trn : Integer[torch.Tensor, "n"]
            Only necessary for pred_method="svm", ignored for KRR. The training
            labels to fit the SVM dual variables. 
        learning_setting : str 
            Considered learning setting for inference. Can be "inductive" 
            (default) or "transductive".
        pred_method : str
            If "svm" uses SVM for predictions. If "krr" uses kernel ridge
            regression instead. Default: "svm".
        regularizer : float
            KRR: Conditioner to add to the diagonal of the NTK to invert. Higher
                 is higher regularization.
            SVM: How much weight is given to correct classification. Default: 1e-8 
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
        self.regularizer = regularizer
        self.pred_method = pred_method
        self.idx_trn_labeled = idx_trn_labeled
        self.n_classes = n_classes
        if pred_method == "svm":
            self.svm = self.fit_svm(X_trn, A_trn, y_trn, idx_trn_labeled)
        else:
            assert pred_method == "krr"
            self.solution_method = "LU"
            if "solution_method" in model_dict:
                self.solution_method = model_dict["solution_method"]


    def fit_svm(self,
                X: Float[torch.Tensor, "n d"], 
                A: Float[torch.Tensor, "n n"],
                y: Integer[torch.Tensor, "l"],
                idx_trn_labeled: Integer[torch.Tensor, "l"]):    
        cache_size = 1000
        if "cache_size" in self.model_dict:
            cache_size = self.model_dict["cache_size"]
        f = svm.SVC(C=self.regularizer, kernel="precomputed", 
                    cache_size=cache_size)
        gram_matrix = self.ntk.detach().cpu().numpy()
        if idx_trn_labeled is not None:
            gram_matrix = gram_matrix[idx_trn_labeled, :]
            gram_matrix = gram_matrix[:, idx_trn_labeled]
        if self.n_classes == 2:
            f.fit(gram_matrix, y.detach().cpu().numpy())
        else:
            f = OneVsRestClassifier(f, n_jobs=-1).fit(gram_matrix, y.detach().cpu().numpy())
        return f

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
                idx_test: Integer[np.ndarray, "u"],
                y_test: Integer[torch.Tensor, "n"],
                X_test: Float[torch.Tensor, "n d"] = None,
                A_test: Union[SparseTensor,
                     Tuple[Integer[torch.Tensor, "2 nnz"], Float[torch.Tensor, "nnz"]],
                     Float[torch.Tensor, "n_nodes n_nodes"]] = None,
                learning_setting: Optional[str] = None,
                return_ntk: bool = False,
                ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Perform kernel regression using the NTK.

        The NTK of the test-graph is calculated using X_test & A_test, except
        if they are None and the learning setting set to transductive. Then,
        uses initialized A & X (corresponding to training graph) for prediction.

        Note: KRR predictions are differentiable w.r.t. the graph, SVM
              predictions due to using scikit-learn implementation are not (yet).
        
        Parameters
        ----------
        idx_labeled : Integer[np.ndarray, "m"]
            Indices of labeled nodes in X_test / A_test.
        idx_test : Integer[np.ndarray, "u"]
            Indices of unlabeled test nodes in X_test / A_test.
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
        if learning_setting is None:
            learning_setting = self.learning_setting

        if X_test is None or A_test is None:
            assert X_test is None and A_test is None
            assert learning_setting == "transductive", "No test graph given, " \
                + " thus learning setting must be transductive for inference."
            ntk_test = self.ntk
        else:
            # handle different adj representations
            if isinstance(A_test, SparseTensor):
                A_test = A_test.to_dense() # is differentiable
            elif isinstance(A_test, tuple):
                n, _ = X_test.shape
                A_test = torch.sparse_coo_tensor(*A_test, 2 * [n]).to_dense() # is differentiable
            ntk_test = self.calc_ntk(X_test, A_test)
        
        if learning_setting == "inductive":
            ntk_labeled = self.ntk 
            if self.idx_trn_labeled is not None: # semi-supervised setting
                ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
                ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
            ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]
        if learning_setting == "transductive": 
            ntk_labeled = self.ntk[idx_labeled,:][:,idx_labeled]
            ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]

        if self.pred_method == "krr":
            # ensure non-singularity
            ntk_labeled += torch.eye(ntk_labeled.shape[0], dtype=self.dtype).to(self.device) \
                        * self.regularizer
            if self.solution_method == "QR":
                assert self.n_classes == 2, "Legacy QR-Method - no multiclass support"
                # Fascinatingly has the exact same behaviour/result as (P)LU
                Q, R = torch.linalg.qr(ntk_labeled)
                Qy = torch.matmul(Q.T, (y_test[idx_labeled] * 2 - 1).to(dtype=torch.float64))
                v = torch.linalg.solve_triangular(R, Qy.view(-1, 1), upper=True)
                v = v.view(-1)
                y_pred = torch.matmul(ntk_unlabeled, v)
            else:
                # Uses (P)LU factorization
                if self.n_classes == 2:
                    v = torch.linalg.solve(ntk_labeled, 
                                        (y_test[idx_labeled] * 2 - 1).to(dtype=torch.float64))
                else:
                    y_onehot = torch.nn.functional.one_hot(y_test[idx_labeled]).to(dtype=torch.float64)
                    v = torch.linalg.solve(ntk_labeled, y_onehot)
                y_pred = torch.matmul(ntk_unlabeled, v)
        else:
            if self.n_classes == 2:
                y_pred = self.svm.predict(ntk_unlabeled.detach().cpu().numpy())
                y_pred = torch.Tensor(y_pred).to(self.device)
            else:
                y_pred = self.svm.predict(ntk_unlabeled.detach().cpu().numpy())
                y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)
                y_pred = torch.nn.functional.one_hot(y_pred)

        if return_ntk:
            return y_pred, ntk_test 
        return y_pred
