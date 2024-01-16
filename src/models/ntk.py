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

from src.models.common import row_normalize, tbn_normalize, degree_scaling, \
                              sym_normalize, APPNP_propogation, make_dense
from src import utils


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
        self.calculated_lb_ub = False
        self.empty_gpu_memory()
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

    def empty_gpu_memory(self):
        utils.empty_gpu_memory(self.device)

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
            f = OneVsRestClassifier(f, n_jobs=1).fit(gram_matrix, y.detach().cpu().numpy())
        return f

    def _calc_diffusion(self, X: torch.Tensor, A: torch.Tensor):
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

    def calc_diffusion(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if "normalize" in self.model_dict:
            if self.model_dict["normalize"]:
                return self._calc_diffusion(X, A)
            else:
                return A
        else:
            return self._calc_diffusion(X, A)

    def kappa_0(self, u):
        pi = torch.acos(torch.tensor(0, dtype=self.dtype).to(self.device)) * 2
        return (pi - torch.acos(u-1e-7)) / pi

    def kappa_1(self, u):
        pi = torch.acos(torch.tensor(0, dtype=self.dtype).to(self.device)) * 2
        return (u*(pi - torch.acos(u-1e-7)) + torch.sqrt(1-u*u+1e-7))/pi

    def kappa_0_lb(self, u_lb):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        t_one = torch.tensor(1, dtype=self.dtype).to(self.device)
        u_lb_ = torch.maximum(u_lb-1e-7, -t_one+1e-7)
        return (pi - torch.acos(u_lb_)) / pi
    
    def kappa_0_ub(self, u_ub):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        t_one = torch.tensor(1, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        u_ub_ = torch.minimum(u_ub-1e-7, t_one)
        return (pi - torch.acos(u_ub_)) / pi

    def kappa_1_lb(self, u_lb, u_ub):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        t_one = torch.tensor(1, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        a = torch.maximum(1-u_ub*u_ub+1e-7, t_zero)
        u_lb_ = torch.maximum(u_lb-1e-7, -t_one+1e-7)
        return (u_lb*(pi - torch.acos(u_lb_)) + torch.sqrt(a))/pi
    
    def kappa_1_ub(self, u_lb, u_ub):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        t_one = torch.tensor(1, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        u_ub_ = torch.minimum(u_ub, t_one)-1e-7
        a = torch.maximum(1-u_lb*u_lb, t_zero)+1e-7
        return (u_ub*(pi - torch.acos(u_ub_)) + torch.sqrt(a))/pi
    
    def _calc_ntk_gcn(self, X: Float[torch.Tensor, "n d"], 
                      S: Float[torch.Tensor, "n n"]) -> torch.Tensor:
        csigma = 1 
        XXT = X.matmul(X.T)
        Sig = S.matmul(XXT.matmul(S.T))
        print(f"Sig.mean(): {Sig.mean()}")
        print(f"Sig.min(): {Sig.min()}")
        print(f"Sig.max(): {Sig.max()}")
        del XXT
        self.empty_gpu_memory()
        kernel = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        # ReLu GCN
        depth = self.model_dict["depth"]
        kernel_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                dtype=self.dtype).to(self.device)
        for i in range(depth):
            print(f"Depth {i}")
            p = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
            Diag_Sig = torch.diagonal(Sig) 
            Sig_i = p + Diag_Sig.reshape(1, -1)
            Sig_j = p + Diag_Sig.reshape(-1, 1)
            del p
            del Diag_Sig
            self.empty_gpu_memory()
            q = torch.sqrt(Sig_i * Sig_j)
            u = Sig/q 
            E = (q * self.kappa_1(u)) * csigma
            del q
            self.empty_gpu_memory()
            E_der = (self.kappa_0(u)) * csigma
            print(f"E_der.min(): {E_der.min()}")
            print(f"E_der.max(): {E_der.max()}")
            self.empty_gpu_memory()
            kernel_sub[i] = S.matmul((Sig * E_der).matmul(S.T))
            Sig = S.matmul(E.matmul(S.T))
            print(f"Sig.mean(): {Sig.mean()}")
            print(f"Sig.min(): {Sig.min()}")
            print(f"Sig.max(): {Sig.max()}")
            for j in range(i):
                kernel_sub[j] = S.matmul((kernel_sub[j].float() * E_der).matmul(S.T))
            del E_der
            del E
            self.empty_gpu_memory()
        kernel += torch.sum(kernel_sub, dim=0)
        kernel += Sig
        return kernel

    def _calc_ntk_appnp(self, X: Float[torch.Tensor, "n d"], 
                        S: Float[torch.Tensor, "n n"]) -> torch.Tensor:
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

    def calc_ntk(self, X: Float[torch.Tensor, "n d"], 
                 A: Float[torch.Tensor, "n n"]):
        """Calculate and return ntk matrix."""
        A = make_dense(A)
        S = self.calc_diffusion(X, A)
        self.empty_gpu_memory()
        
        if self.model_dict["model"] == "GCN" or self.model_dict["model"] == "SoftMedoid":
            kernel = self._calc_ntk_gcn(X, S)
            self.empty_gpu_memory()
            return kernel
        
        elif self.model_dict["model"] == "PPNP" or self.model_dict["model"] == "APPNP":
            # NTK for PPNP with one hidden layer fcn for features
            # (A)PPNP = S ( ReLU(XW_1 + b_1) W_2 + b_2)
            kernel = self._calc_ntk_appnp(X, S)
            self.empty_gpu_memory()
            return kernel
    
    def calc_XXT_lb_ub(self, X: Float[torch.Tensor, "n d"],
                       idx_adv: Integer[np.ndarray, "r"],
                       delta: Union[float, int],
                       perturbation_model: str
    ) -> Tuple[Float[torch.Tensor, "n n"], Float[torch.Tensor, "n n"]]:
        """Return XXT_lb and XXT_ub in that order.
        
        Parameters
        ----------
        delta : Union[float, int]
            Depends on perturbation model:
            - l0: if < 1 interpreted as % of number of features, otherwise 
                  assumed to be integer and directly interpreted as discrete 
                  local budget.
        """
        if perturbation_model == "l0":
            """TODO: Implement Sparse Computation."""
            if delta < 1:
                delta = int(delta * X.shape[1])
            delta = torch.tensor(delta, dtype=self.dtype, device=self.device)
            XXT = X.matmul(X.T)
            print(XXT.mean())
            print(XXT.min())
            print(XXT.max())
            #print(XXT)
            Delta_lb = torch.zeros(XXT.shape, dtype=self.dtype, device=self.device)
            Delta_ub = torch.zeros(XXT.shape, dtype=self.dtype, device=self.device)
            # Delta^T@Delta Terms
            #DD_lb = Delta_lb[idx_adv, :]
            #DD_lb[:, idx_adv] = -delta
            #Delta_lb[idx_adv, :] = DD_lb
            #DD_ub = Delta_ub[idx_adv, :]
            #DD_ub[:, idx_adv] = delta
            #Delta_ub[idx_adv, :] = DD_ub
            # Delta@X^T Terms
            Delta_lb[idx_adv, :] -= delta
            delta_ub = torch.minimum(delta, X.sum(dim=1))
            Delta_ub[idx_adv, :] += delta_ub.reshape(1, -1)
            # X@Delta^T Terms
            Delta_lb[:, idx_adv] -= delta
            Delta_ub[:, idx_adv] += delta_ub.reshape(-1, 1)
            # Correct for double counting
            #mask_1 = torch.zeros(XXT.shape, dtype=torch.bool, device=self.device)
            #mask_1[:, idx_adv] = True
            #mask_2 = torch.zeros(XXT.shape, dtype=torch.bool, device=self.device)
            #mask_2[idx_adv, :] = True
            #mask_3 = torch.logical_and(mask_1, mask_2)
            #Delta_lb[mask_3] += delta
            #Delta_ub[mask_3] -= delta_ub[idx_adv].repeat(len(idx_adv))
            #Delta_lb[idx_adv, idx_adv] = -delta
            #Delta_ub[idx_adv, idx_adv] = delta_ub[idx_adv]
            XXT_lb = torch.maximum(XXT+Delta_lb, 
                                   torch.tensor(0, dtype=self.dtype).to(self.device))
            print(XXT_lb.mean())
            print(XXT_lb.min())
            print(XXT_lb.max())
            XXT_ub = XXT+Delta_ub
            #print(XXT_lb)
            print(XXT_ub.mean())
            print(XXT_ub.min())
            print(XXT_ub.max())
            # percentage increase
            print("Adversarial Nodes: ")
            print((XXT[idx_adv, :].mean() + XXT[:, idx_adv].mean())/2)
            print((XXT_lb[idx_adv, :].mean() + XXT_lb[:, idx_adv].mean())/2)
            print((XXT_ub[idx_adv, :].mean() + XXT_ub[:, idx_adv].mean())/2)
            #print(XXT_ub)
            return XXT_lb, XXT_ub
        else:
            assert False, f"Perturbation model {perturbation_model} not supported"

    def calc_ntk_lb_ub(self, X: Float[torch.Tensor, "n d"], 
                       A: Float[torch.Tensor, "n n"],
                       idx_adv: Integer[np.ndarray, "r"],
                       delta: float,
                       perturbation_model: str):
        if self.calculated_lb_ub:
            return self.ntk_lb, self.ntk_ub
        csigma = 1 
        A = make_dense(A)
        S = self.calc_diffusion(X, A)
        XXT_lb, XXT_ub = self.calc_XXT_lb_ub(X, idx_adv, delta, perturbation_model)
        self.empty_gpu_memory()

        if self.model_dict["model"] == "GCN":
            Sig_lb = S.matmul(XXT_lb.matmul(S.T))
            Sig_ub = S.matmul(XXT_ub.matmul(S.T))
            print(f"Sig_lb.mean(): {Sig_lb.mean()}")
            print(f"Sig_lb.min(): {Sig_lb.min()}")
            print(f"Sig_lb.max(): {Sig_lb.max()}")
            print(f"Sig_ub.mean(): {Sig_ub.mean()}")
            print(f"Sig_ub.min(): {Sig_ub.min()}")
            print(f"Sig_ub.max(): {Sig_ub.max()}")

            ntk_lb = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
            ntk_ub = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
            # ReLu GCN
            depth = self.model_dict["depth"]
            ntk_lb_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                    dtype=self.dtype).to(self.device)
            ntk_ub_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                    dtype=self.dtype).to(self.device)
            for i in range(depth):
                print(f"Depth {i}")
                p = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
                Diag_Sig_lb = torch.diagonal(Sig_lb) 
                Diag_Sig_ub = torch.diagonal(Sig_ub) 
                Sig_i_lb = p + Diag_Sig_lb.reshape(1, -1)
                Sig_j_lb = p + Diag_Sig_lb.reshape(-1, 1)
                Sig_i_ub = p + Diag_Sig_ub.reshape(1, -1)
                Sig_j_ub = p + Diag_Sig_ub.reshape(-1, 1)
                q_lb = torch.sqrt(Sig_i_lb * Sig_j_lb) + 1e-7
                q_ub = torch.sqrt(Sig_i_ub * Sig_j_ub)
                u_lb = Sig_lb/q_ub 
                u_ub = Sig_ub/q_lb
                E_lb = (q_lb * self.kappa_1_lb(u_lb, u_ub)) * csigma
                E_ub = (q_ub * self.kappa_1_ub(u_lb, u_ub)) * csigma
                self.empty_gpu_memory()
                E_der_lb = (self.kappa_0_lb(u_lb)) * csigma
                E_der_ub = (self.kappa_0_ub(u_ub)) * csigma
                print(f"E_der_lb.min(): {E_der_lb.min()}")
                print(f"E_der_lb.max(): {E_der_lb.max()}")
                print(f"E_der_ub.min(): {E_der_ub.min()}")
                print(f"E_der_ub.max(): {E_der_ub.max()}")
                self.empty_gpu_memory()
                ntk_lb_sub[i] = S.matmul((Sig_lb * E_der_lb).matmul(S.T))
                ntk_ub_sub[i] = S.matmul((Sig_ub * E_der_ub).matmul(S.T))
                Sig_lb = S.matmul(E_lb.matmul(S.T))
                Sig_ub = S.matmul(E_ub.matmul(S.T))
                print(f"Sig_lb.mean(): {Sig_lb.mean()}")
                print(f"Sig_lb.min(): {Sig_lb.min()}")
                print(f"Sig_lb.max(): {Sig_lb.max()}")
                print(f"Sig_ub.mean(): {Sig_ub.mean()}")
                print(f"Sig_ub.min(): {Sig_ub.min()}")
                print(f"Sig_ub.max(): {Sig_ub.max()}")
                for j in range(i):
                    ntk_lb_sub[j] = S.matmul((ntk_lb_sub[j].float() * E_der_lb).matmul(S.T))
                    ntk_ub_sub[j] = S.matmul((ntk_ub_sub[j].float() * E_der_ub).matmul(S.T))
            self.empty_gpu_memory()
            ntk_lb += torch.sum(ntk_lb_sub, dim=0)
            ntk_ub += torch.sum(ntk_ub_sub, dim=0)
            ntk_lb += Sig_lb
            ntk_ub += Sig_ub
            self.calculated_lb_ub = True
            self.ntk_lb = ntk_lb
            self.ntk_ub = ntk_ub
            return self.ntk_lb, self.ntk_ub
        else:
            assert False, "Other models than GCN not implemented so far."

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
        """Perform kernel regression or SVM prediction using the NTK.

        The NTK of the test-graph is calculated using X_test & A_test, except
        if they are None and the learning setting set to transductive. Then,
        uses initialized A & X (corresponding to training graph) for prediction.
        
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
            A_test = make_dense(A_test) # is differentiable
            ntk_test = self.calc_ntk(X_test, A_test)
            if torch.cuda.is_available() and self.device != "cpu":
                torch.cuda.empty_cache()
        
        if learning_setting == "inductive":
            ntk_labeled = self.ntk 
            if self.idx_trn_labeled is not None: # semi-supervised setting
                ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
                ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
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
                # Implementation of self.svm.decision_function(ntk_unlabeled) in PyTorch
                alpha = torch.tensor(self.svm.dual_coef_, dtype=self.dtype, device=self.device)
                b = torch.tensor(self.svm.intercept_, dtype=self.dtype, device=self.device)
                idx_sup = self.svm.support_
                y_pred = (alpha * ntk_unlabeled[:,idx_sup]).sum(dim=1) + b
            else:
                y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                for i, svm in enumerate(self.svm.estimators_):
                    # Implementation of the following scikit-learn function in PyTorch:
                    # - pred = svm.decision_function(ntk_u_cpu) 
                    alpha = torch.tensor(svm.dual_coef_, dtype=self.dtype, device=self.device)
                    b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                    idx_sup = svm.support_
                    if i == 0:
                        print((alpha.reshape(1,-1) * ntk_unlabeled[:,idx_sup]).sum(dim=1))
                        print((alpha.reshape(1,-1) * ntk_unlabeled[:,idx_sup]).sum(dim=1).shape)
                    pred = (alpha * ntk_unlabeled[:,idx_sup]).sum(dim=1) + b
                    y_pred[:, i] = pred

        if return_ntk:
            return y_pred, ntk_test 
        return y_pred

    def forward_upperbound(self, 
                           idx_labeled: Integer[np.ndarray, "m"],
                           idx_test: Integer[np.ndarray, "u"],
                           idx_adv: Integer[np.ndarray, "r"],
                           y_test: Integer[torch.Tensor, "n"],
                           X_test: Float[torch.Tensor, "n d"],
                           A_test: Union[SparseTensor,
                                         Tuple[Integer[torch.Tensor, "2 nnz"], 
                                         Float[torch.Tensor, "nnz"]],
                                         Float[torch.Tensor, "n_nodes n_nodes"]],
                           delta: float = 0.01,
                           perturbation_model: str = "l0",
                           learning_setting: Optional[str] = None,
                           force_recalculation: bool = False,
                           return_ntk: bool = False,
                        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Perform kernel regression or SVM prediction using the upper-bounded
        NTK matrix.

        The NTK of the test-graph is calculated using X_test & A_test, except
        if they are None and the learning setting set to transductive. Then,
        uses initialized A & X (corresponding to training graph) for prediction.
        
        Parameters
        ----------
        idx_labeled : Integer[np.ndarray, "m"]
            Indices of labeled nodes in X_test / A_test.
        idx_test : Integer[np.ndarray, "u"]
            Indices of unlabeled test nodes in X_test / A_test.
        idx_adv : Integer[np.ndarray, "r"]
            Indices of adversarily controlled nodes in X_test / A_test.
        y_test : Integer[torch.Tensor, "n"]
            Labels of the test graph.
        X_test : Float[torch.Tensor, "n d"]
            Node features available during testing (i.e., of the test graph). 
        A_test : Float[torch.Tensor, "n n"]
            Graph adjacency matrix available during testing (i.e., of the test
            graph).
        delta : float
            Local budget, interpretation depending on chosen perurbation model:
            - l0: local budget = delta * feature dimension
        perturbation_model : str
            Currently, only l0 (default) supported.
        learning_setting : Optional[str] 
            Optional, per default uses the learning setting set when initializing
            the NTK object. However, if set, inference will be done with the
            here set learning_setting instead. Options: "inductive" (default) 
            or "transductive".
        force_recalculation : Optional[bool]
            Calculate new NTK lower and upper bounds. You want to set this to 
            true, if you change X_test, A_test or idx_adv compared to your 
            previous function call.
        return_ntk : Optional[bool]
            If true, return the NTK of the test-graph calculated using X_test
            and A_test. Defaul: False
        Returns: 
            Logits of unlabeled nodes
        """
        if learning_setting is None:
            learning_setting = self.learning_setting
        # handle different adj representations
        A_test = make_dense(A_test)

        if force_recalculation:
            self.calculated_lb_ub = False
        ntk_test_lb, ntk_test_ub = self.calc_ntk_lb_ub(X_test, A_test, idx_adv,
                                                       delta, perturbation_model)
        self.empty_gpu_memory()
        if learning_setting == "inductive":
            ntk_labeled = self.ntk 
            if self.idx_trn_labeled is not None: # semi-supervised setting
                ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
                ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
        if learning_setting == "transductive": 
            ntk_labeled = self.ntk[idx_labeled,:][:,idx_labeled]
        ntk_unlabeled_ub = ntk_test_ub[idx_test,:][:,idx_labeled]
        ntk_unlabeled_lb = ntk_test_lb[idx_test,:][:,idx_labeled]

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
                # Implementation of self.svm.decision_function(ntk_unlabeled) in PyTorch
                alpha = torch.tensor(self.svm.dual_coef_, dtype=self.dtype, device=self.device)
                b = torch.tensor(self.svm.intercept_, dtype=self.dtype, device=self.device)
                idx_sup = self.svm.support_
                y_pred = (alpha * ntk_unlabeled[:,idx_sup]).sum(dim=1) + b
            else:
                y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                for i, svm in enumerate(self.svm.estimators_):
                    # Implementation of the following scikit-learn function in PyTorch:
                    # - pred = svm.decision_function(ntk_u_cpu) 
                    alpha_pos_mask = svm.dual_coef_ > 0
                    alpha_pos = torch.tensor(svm.dual_coef_[alpha_pos_mask], 
                                         dtype=self.dtype, device=self.device)
                    alpha_neg = torch.tensor(svm.dual_coef_[~alpha_pos_mask], 
                                         dtype=self.dtype, device=self.device)
                    b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                    idx_sup = svm.support_
                    alpha_pos_mask = alpha_pos_mask.reshape(-1)
                    idx_sup_pos = svm.support_[alpha_pos_mask]
                    idx_sup_neg = svm.support_[~alpha_pos_mask]
                    if i == 0:
                        print((alpha_pos.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_pos]).sum(dim=1) \
                        + (alpha_neg.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_neg]).sum(dim=1))
                    pred = (alpha_pos.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_pos]).sum(dim=1) \
                        + (alpha_neg.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_neg]).sum(dim=1) \
                        + b
                    y_pred[:, i] = pred

        if return_ntk:
            return y_pred, ntk_test_ub 
        return y_pred
    
    def forward_lowerbound(self, 
                           idx_labeled: Integer[np.ndarray, "m"],
                           idx_test: Integer[np.ndarray, "u"],
                           idx_adv: Integer[np.ndarray, "r"],
                           y_test: Integer[torch.Tensor, "n"],
                           X_test: Float[torch.Tensor, "n d"],
                           A_test: Union[SparseTensor,
                                         Tuple[Integer[torch.Tensor, "2 nnz"], 
                                         Float[torch.Tensor, "nnz"]],
                                         Float[torch.Tensor, "n_nodes n_nodes"]],
                           delta: float = 0.01,
                           perturbation_model: str = "l0",
                           learning_setting: Optional[str] = None,
                           force_recalculation = False,
                           return_ntk: bool = False,
                        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Perform kernel regression or SVM prediction using the upper-bounded
        NTK matrix.

        The NTK of the test-graph is calculated using X_test & A_test, except
        if they are None and the learning setting set to transductive. Then,
        uses initialized A & X (corresponding to training graph) for prediction.
        
        Parameters
        ----------
        idx_labeled : Integer[np.ndarray, "m"]
            Indices of labeled nodes in X_test / A_test.
        idx_test : Integer[np.ndarray, "u"]
            Indices of unlabeled test nodes in X_test / A_test.
        idx_adv : Integer[np.ndarray, "r"]
            Indices of adversarily controlled nodes in X_test / A_test.
        y_test : Integer[torch.Tensor, "n"]
            Labels of the test graph.
        X_test : Float[torch.Tensor, "n d"]
            Node features available during testing (i.e., of the test graph). 
        A_test : Float[torch.Tensor, "n n"]
            Graph adjacency matrix available during testing (i.e., of the test
            graph).
        delta : float
            Local budget, interpretation depending on chosen perurbation model:
            - l0: local budget = delta * feature dimension
        perturbation_model : str
            Currently, only l0 (default) supported.
        force_recalculation : Optional[bool]
            Calculate new NTK lower and upper bounds. You want to set this to 
            true, if you change X_test, A_test or idx_adv compared to your 
            previous function call.
        learning_setting : Optional[str] 
            Optional, per default uses the learning setting set when initializing
            the NTK object. However, if set, inference will be done with the
            here set learning_setting instead. Options: "inductive" (default) 
            or "transductive".
        return_ntk : Optional[bool]
            If true, return the NTK of the test-graph calculated using X_test
            and A_test. Defaul: False
        Returns: 
            Logits of unlabeled nodes
        """
        if learning_setting is None:
            learning_setting = self.learning_setting
        # handle different adj representations
        A_test = make_dense(A_test)

        if force_recalculation:
            self.calculated_lb_ub = False
        ntk_test_lb, ntk_test_ub = self.calc_ntk_lb_ub(X_test, A_test, idx_adv,
                                                       delta, perturbation_model)
        self.empty_gpu_memory()
        if learning_setting == "inductive":
            ntk_labeled = self.ntk 
            if self.idx_trn_labeled is not None: # semi-supervised setting
                ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
                ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
        if learning_setting == "transductive": 
            ntk_labeled = self.ntk[idx_labeled,:][:,idx_labeled]
        ntk_unlabeled_ub = ntk_test_ub[idx_test,:][:,idx_labeled]
        ntk_unlabeled_lb = ntk_test_lb[idx_test,:][:,idx_labeled]

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
                # Implementation of self.svm.decision_function(ntk_unlabeled) in PyTorch
                alpha = torch.tensor(self.svm.dual_coef_, dtype=self.dtype, device=self.device)
                b = torch.tensor(self.svm.intercept_, dtype=self.dtype, device=self.device)
                idx_sup = self.svm.support_
                y_pred = (alpha * ntk_unlabeled[:,idx_sup]).sum(dim=1) + b
            else:
                y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                for i, svm in enumerate(self.svm.estimators_):
                    # Implementation of the following scikit-learn function in PyTorch:
                    # - pred = svm.decision_function(ntk_u_cpu) 
                    alpha_pos_mask = svm.dual_coef_ > 0
                    alpha_pos = torch.tensor(svm.dual_coef_[alpha_pos_mask], 
                                         dtype=self.dtype, device=self.device)
                    alpha_neg = torch.tensor(svm.dual_coef_[~alpha_pos_mask], 
                                         dtype=self.dtype, device=self.device)
                    b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                    idx_sup = svm.support_
                    alpha_pos_mask = alpha_pos_mask.reshape(-1)
                    idx_sup_pos = svm.support_[alpha_pos_mask]
                    idx_sup_neg = svm.support_[~alpha_pos_mask]
                    if i == 0:
                        print((alpha_pos.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_pos]).sum(dim=1) \
                        + (alpha_neg.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_neg]).sum(dim=1))
                    pred = (alpha_pos.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_pos]).sum(dim=1) \
                        + (alpha_neg.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_neg]).sum(dim=1) \
                        + b
                    y_pred[:, i] = pred

        if return_ntk:
            return y_pred, ntk_test_lb
        return y_pred