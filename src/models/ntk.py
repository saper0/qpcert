import logging
from typing import Any, Dict, Optional, Union, Tuple

import cvxopt
from jaxtyping import Float, Integer
import numpy as np
from proxsuite.torch.qplayer import QPFunction
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from src.models.common import row_normalize, tbn_normalize, degree_scaling, \
                              sym_normalize, APPNP_propogation, make_dense, \
                              add_self_loop
from src import utils, globals


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
                 bias: bool = True,
                 append_dimension: bool = False,
                 solver: str = "sklearn",
                 alpha_tol: float = 1e-4,
                 device: Union[torch.device, str] = None,
                 dtype: torch.dtype = torch.float64,
                 print_alphas: bool = True):
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
        bias : bool
            Only if "svm". Whether to include a bias term. Default: True.
        append_dimension : bool
            Only used if bias == False. If true, adds an extra feature dimension
            to compensate for no bias term. TODO: implement
        solver : string
            If pred_method = "svm": 
                Can be "cvxopt" or "sklearn". Solving the SVM-problem wihout
                bias is only possible using "cvxopt". Currenlty, multiclass
                classicification is only implemented for "sklearn".
            If pred_method = "krr":
                Currently has no effect. TODO: Can implement LU etc. factor-
                ization with this property.
        alpha_tol : float
            Only relevant if using cvxopt-solver. Numerical theshold for 
            choosing support vectors. Default 1e-4.
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
            or self.model_dict["model"] == "APPNP" \
            or self.model_dict["model"] == "GIN"
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
        self.solver = solver
        self.alpha_tol = alpha_tol
        self.bias = bias
        self.print_alphas = print_alphas
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
        gram_matrix = self.ntk.detach().cpu().numpy()
        if type(idx_trn_labeled) == torch.Tensor:
            idx_trn_labeled = idx_trn_labeled.numpy(force=True)
        if idx_trn_labeled is not None:
            gram_matrix = gram_matrix[idx_trn_labeled, :]
            gram_matrix = gram_matrix[:, idx_trn_labeled]
        if self.solver == "sklearn":
            f = svm.SVC(C=self.regularizer, kernel="precomputed", 
                        cache_size=cache_size)
            if self.n_classes == 2:
                f.fit(gram_matrix, y.detach().cpu().numpy())
                if self.print_alphas:
                    alphas_str = [f"{alpha:.04f}" for alpha in f.dual_coef_[0]]
                    print(f"{len(alphas_str)} alphas found: {alphas_str}")
            else:
                f = OneVsRestClassifier(f, n_jobs=1).fit(gram_matrix, y.detach().cpu().numpy())
            return f
        elif self.solver == "cvxopt":
            assert self.n_classes == 2, "\"cvxopt\" only implemented for " \
                + "binary classification"
            y = y.detach().cpu().numpy()
            y = y * 2 - 1
            l = y.shape[0]
            Y = np.outer(y, y)
            # Make the gram matrix symmetric (removes numerical inconsistencies)
            # gram_matrix = np.triu(gram_matrix, k=0) + np.triu(gram_matrix, k=-1).T 
            P = cvxopt.matrix(Y*gram_matrix)
            q = cvxopt.matrix(-np.ones((l,), dtype=np.float64))
            I = np.identity(n=l, dtype=np.float64)
            I_neg = -np.identity(n=l, dtype=np.float64)
            G = cvxopt.matrix(np.concatenate((I, I_neg), axis=0))
            h = cvxopt.matrix(np.zeros((2*l,), dtype=np.float64))
            h[:l] = self.regularizer
            cvxopt.solvers.options["show_progress"] = True
            if self.bias:
                A = cvxopt.matrix(y.astype(np.float64).reshape((1,-1)))
                b = cvxopt.matrix(np.zeros(1))
                solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            else:
                solution = cvxopt.solvers.qp(P, q, G, h)
            #print(solution)
            alphas = np.array(solution["x"]).reshape(-1,)
            if self.print_alphas:
                alphas_str = [f"{alpha:.04f}" for alpha in alphas]
                print(f"{len(alphas_str)} alphas found: {alphas_str}")
            #y = (y + 1) / 2
            #f_ = svm.SVC(C=self.regularizer, kernel="precomputed", 
            #            cache_size=cache_size)
            #f_.fit(gram_matrix, y)
            #self.f_ = f_
            #print(f_.dual_coef_)
            #assert False
            #print(f_.intercept_)
            return alphas
        elif self.solver == "qplayer":
            if idx_trn_labeled is not None:
                gram_matrix = self.ntk[idx_trn_labeled, :]
                gram_matrix = gram_matrix[:, idx_trn_labeled]
            else:
                gram_matrix = self.ntk
            if self.n_classes == 2:
                y_ = y * 2 - 1
                l = y.shape[0]
                Y = torch.outer(y_, y_)
                Q = Y*gram_matrix
                p = -torch.ones((l,), dtype=self.dtype)
                I = torch.eye(n=l, dtype=self.dtype)
                I_neg = -torch.eye(n=l, dtype=self.dtype)
                G = torch.eye(n=l, dtype=self.dtype)
                h = torch.zeros((l,), dtype=self.dtype)
                h[:l] = self.regularizer
                l = torch.zeros((l,), dtype=self.dtype)
                if self.bias:
                    A = y_.reshape((1, -1))
                    b = torch.zeros(1, dtype=self.dtype)
                    alphas, _, _ = QPFunction()(Q, p, A, b, G, l, h)
                else:
                    A = torch.tensor([], device=self.device)
                    b = torch.tensor([], device=self.device)
                    alphas, _, _ = QPFunction()(Q, p, A, b, G, l, h)
                alphas_str = [f"{alpha:.04f}" for alpha in alphas[0]]
                if self.print_alphas:
                    alphas_non_zero = alphas[0] > self.alpha_tol
                    print(f"{alphas_non_zero.sum()} alphas found: {alphas_str}")
                return alphas[0]
            else:
                assert False, "qplayer only for binary class"
        elif self.solver == "MSVM":
            assert self.n_classes > 2
            if idx_trn_labeled is not None:
                gram_matrix = self.ntk[idx_trn_labeled, :]
                gram_matrix = gram_matrix[:, idx_trn_labeled]
            else:
                gram_matrix = self.ntk
            K = self.n_classes
            l_ = K*y.shape[0]
            I = torch.eye(n=K, dtype=self.dtype).to(self.device)
            Q = torch.kron(I, gram_matrix)
            p = -torch.nn.functional.one_hot(y, num_classes=K).reshape(-1)
            b = torch.zeros(y.shape[0], dtype=self.dtype).to(self.device)
            I_nn = torch.eye(n=y.shape[0], dtype=self.dtype).to(self.device) 
            I_1k = torch.ones((1,K), dtype=self.dtype).to(self.device)
            A = torch.kron(I_nn,I_1k)
            G = torch.eye(n=l_, dtype=self.dtype)
            h = -p*self.regularizer
            l = torch.ones((l_), dtype=self.dtype).to(self.device)*float("-inf")
            alphas, _, _ = QPFunction(maxIter=1000)(Q, p, A, b, G, l, h)
            if self.print_alphas:
                alphas_str = [f"{alpha:.04f}" for alpha in alphas[0]]
                print(f"alphas found: {alphas_str}")
            alpha_matrix = alphas[0].reshape((y.shape[0],K))
            return alpha_matrix
        elif self.solver == "simMSVM":
            assert self.n_classes > 2
            if idx_trn_labeled is not None:
                gram_matrix = self.ntk[idx_trn_labeled, :]
                gram_matrix = gram_matrix[:, idx_trn_labeled]
            else:
                gram_matrix = self.ntk
            K = self.n_classes
            l_ = y.shape[0]
            Q = gram_matrix
            y_onehot = torch.nn.functional.one_hot(y, num_classes=K).type(self.dtype)
            Kernel = (y_onehot.matmul(y_onehot.t()))*(K-1)
            Kernel[Kernel==0] = -1
            Q = Q*Kernel
            p = -torch.ones(l_, dtype=self.dtype).to(self.device)
            G = torch.eye(n=l_, dtype=self.dtype)
            h = -(p*self.regularizer*K)/((K-1)*(K-1))
            l = torch.zeros(l_, dtype=self.dtype).to(self.device)
            A = torch.tensor([], device=self.device)
            b = torch.tensor([], device=self.device)
            alphas, _, _ = QPFunction()(Q, p, A, b, G, l, h)
            alphas_str = [f"{alpha:.04f}" for alpha in alphas[0]]
            alpha_matrix = alphas[0]
            if self.print_alphas:
                print(f"alphas found: {alphas_str}")
                print("alpha shape ", alpha_matrix.shape)
            return alpha_matrix
        elif self.solver == "qplayer_one_vs_all":
            assert self.n_classes > 2
            if idx_trn_labeled is not None:
                gram_matrix = self.ntk[idx_trn_labeled, :]
                gram_matrix = gram_matrix[:, idx_trn_labeled]
            else:
                gram_matrix = self.ntk
            alphas = []
            for k in range(self.n_classes):
                y_mask = y == k
                y_ = y.clone()
                y_[y_mask] = 1
                y_[~y_mask] = -1
                l = y.shape[0]
                Y = torch.outer(y_, y_)
                Q = Y*gram_matrix
                p = -torch.ones((l,), dtype=self.dtype)
                I = torch.eye(n=l, dtype=self.dtype)
                I_neg = -torch.eye(n=l, dtype=self.dtype)
                G = torch.eye(n=l, dtype=self.dtype)
                h = torch.zeros((l,), dtype=self.dtype)
                h[:l] = self.regularizer
                l = torch.zeros((l,), dtype=self.dtype)
                if self.bias:
                    A = y_.reshape((1, -1))
                    b = torch.zeros(1, dtype=self.dtype)
                    alphas_, _, _ = QPFunction()(Q, p, A, b, G, l, h)
                else:
                    A = torch.tensor([], device=self.device)
                    b = torch.tensor([], device=self.device)
                    alphas_, _, _ = QPFunction()(Q, p, A, b, G, l, h)
                alphas.append(alphas_[0])
                alphas_str = [f"{alpha:.04f}" for alpha in alphas_[0]]
                if self.print_alphas:
                    alphas_non_zero = alphas[0] > self.alpha_tol
                    print(f"Class {k}: {alphas_non_zero.sum()} alphas found: {alphas_str}")
            return alphas
        else:
            assert False, "Solver not found"

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
            elif self.model_dict["normalization"] == "graph_sage_normalization":
                S = row_normalize(A, self_loop=False)
                return add_self_loop(S)
            elif self.model_dict["normalization"] == "none":
                add_self_loop(A)
                return A
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
        elif self.model_dict["model"] == "GIN":
            if self.model_dict["normalization"] == "graph_size_normalization":
                return add_self_loop(A)/A.shape[0]
            else:
                return add_self_loop(A)
        else:
            raise NotImplementedError("Only GCN/SoftMedoid/(A)PPNP architecture implemented")

    def calc_diffusion(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if "weigh_adjacency" in self.model_dict:
            A = self.model_dict["weigh_adjacency"] * A
        if "normalize" in self.model_dict:
            if self.model_dict["normalize"]:
                return self._calc_diffusion(X, A)
            else:
                return A
        else:
            return self._calc_diffusion(X, A)

    def kappa_0(self, u):
        pi = torch.acos(torch.tensor(0, dtype=self.dtype).to(self.device)) * 2
        assert (u > 1).sum() == 0
        assert (u < -1).sum() == 0
        if u.requires_grad:
            u = torch.clamp(u, -1+globals.grad_tol, 1-globals.grad_tol)
        return (pi - torch.acos(u)) / pi

    def kappa_1(self, u):
        pi = torch.acos(torch.tensor(0, dtype=self.dtype).to(self.device)) * 2
        assert (u > 1).sum() == 0 
        assert (u < -1).sum() == 0
        if u.requires_grad:
            u = torch.clamp(u, -1+globals.grad_tol, 1-globals.grad_tol)
        return (u*(pi - torch.acos(u)) + torch.sqrt(1-u*u))/pi

    def kappa_0_lb(self, u_lb):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        return (pi - torch.acos(u_lb)) / pi
    
    def kappa_0_ub(self, u_ub):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        return (pi - torch.acos(u_ub)) / pi

    def kappa_1_lb(self, u_lb, u_ub_sq):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        return (u_lb*(pi - torch.acos(u_lb)) + torch.sqrt(1-u_ub_sq))/pi
    
    def kappa_1_ub(self, u_ub, u_lb_sq):
        t_zero = torch.tensor(0, dtype=self.dtype).to(self.device)
        pi = torch.acos(t_zero) * 2
        return (u_ub*(pi - torch.acos(u_ub)) + torch.sqrt(1-u_lb_sq))/pi
    
    def calc_relu_expectations(self, M: Float[torch.Tensor, "n n"]):
        csigma = 1
        p = torch.zeros((M.shape), dtype=self.dtype).to(self.device)
        Diag_Sig = torch.diagonal(M) 
        Sig_i = p + Diag_Sig.reshape(1, -1)
        Sig_j = p + Diag_Sig.reshape(-1, 1)
        # ensure covariance_ij <= min{var_i, var_j} : could happen otherwise because of numerical precision issues
        cov_gr_var = torch.logical_or((M>Sig_i), (M>Sig_j))
        M[cov_gr_var] = torch.min(Sig_i[cov_gr_var], Sig_j[cov_gr_var])
        del p
        del Diag_Sig
        self.empty_gpu_memory()
        q = torch.sqrt(Sig_i * Sig_j)
        u = M/q 
        u = u.fill_diagonal_(1)
        E = (q * self.kappa_1(u)) * csigma
        del q
        self.empty_gpu_memory()
        E_der = (self.kappa_0(u)) * csigma
        return E, E_der, u

    def _calc_ntk_gcn(self, X: Float[torch.Tensor, "n d"], 
                      S: Float[torch.Tensor, "n n"]) -> torch.Tensor:
        csigma = 1 
        XXT = NTK._calc_XXT(X)
        Sig = S.matmul(XXT.matmul(S.T))
        if globals.debug:
            print(f"Sig.mean(): {Sig.mean()}")
            print(f"Sig.min(): {Sig.min()}")
            print(f"Sig.max(): {Sig.max()}")
        # del XXT
        self.empty_gpu_memory()
        kernel = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        depth = self.model_dict["depth"]
        kernel_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                dtype=self.dtype).to(self.device)
        if "skip_connection" in self.model_dict:
            # get skip_alpha, if not given then set it to 0.2
            if self.model_dict["skip_connection"] == "skip_alpha_linear" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                self.skip_alpha = self.model_dict["skip_alpha"] if "skip_alpha" in self.model_dict \
                    else 0.2
            # compute Sig and Sig_skip for the residual component
            if self.model_dict["skip_connection"] == "skip_pc_relu" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                E_skip, _, _ = self.calc_relu_expectations(XXT)
                Sig = S.matmul((E_skip).matmul(S.T))
                if self.model_dict["skip_connection"] == "skip_pc_relu":
                    Sig_skip = Sig
                else:
                    Sig = ((1-self.skip_alpha)**2 * Sig + \
                        (1-self.skip_alpha)*self.skip_alpha*(E_skip.matmul(S.T) + S.matmul(E_skip)) + \
                        self.skip_alpha**2 * E_skip) * csigma
                    Sig_skip = E_skip
            elif self.model_dict["skip_connection"] == "skip_pc_linear": 
                Sig_skip = Sig
            elif self.model_dict["skip_connection"] == "skip_alpha_linear":
                Sig = ((1-self.skip_alpha)**2 * Sig + \
                        (1-self.skip_alpha)*self.skip_alpha*(XXT.matmul(S.T) + S.matmul(XXT)) + \
                        self.skip_alpha**2 * XXT) * csigma
                Sig_skip = XXT
            else:
                assert False, f"'skip_pc_linear', 'skip_pc_relu', 'skip_alpha_linear' and 'skip_alpha_relu' are only supported. "
        del XXT
        for i in range(depth):
            if globals.debug:
                print(f"Depth {i}")
            if self.model_dict["activation"] == 'relu':
                E, E_der, _ = self.calc_relu_expectations(Sig)
            elif self.model_dict["activation"] == 'linear':
                E = Sig
                E_der = torch.ones((S.shape), dtype=self.dtype).to(self.device)
            else:
                assert False, f"'linear' and 'relu' GCNs are supported. Please update activation in the model_dict."
            if globals.debug:
                print(f"E_der.min(): {E_der.min()}")
                print(f"E_der.max(): {E_der.max()}")
            self.empty_gpu_memory()
            kernel_sub[i] = S.matmul((Sig * E_der).matmul(S.T))
            Sig = S.matmul(E.matmul(S.T))
            if "skip_connection" in self.model_dict:
                if self.model_dict["skip_connection"] == "skip_pc_linear" or \
                self.model_dict["skip_connection"] == "skip_pc_relu"  :
                    Sig += Sig_skip
                elif self.model_dict["skip_connection"] == "skip_alpha_linear" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                    Sig = (1-self.skip_alpha)**2 * Sig + self.skip_alpha**2 * Sig_skip 
                else:
                    assert False, f"'skip_pc_linear', 'skip_pc_relu', 'skip_alpha_linear' and 'skip_alpha_relu' are only supported. "
            if globals.debug:
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
        XXT = NTK._calc_XXT(X)
        B = torch.ones((S.shape), dtype=self.dtype).to(self.device)
        Sig = XXT+B
        E, E_der, _ = self.calc_relu_expectations(Sig)
        kernel += S.matmul(Sig * E_der).matmul(S.T)
        kernel += S.matmul(E+B).matmul(S.T)
        return kernel
    
    def _calc_ntk_gin(self, X: Float[torch.Tensor, "n d"], 
                        S: Float[torch.Tensor, "n n"]) -> torch.Tensor:
        kernel = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        XXT = NTK._calc_XXT(X)
        Sig = S.matmul(XXT.matmul(S.T))
        E, E_der, _ = self.calc_relu_expectations(Sig)
        kernel += (Sig * E_der) + E
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
        elif self.model_dict["model"] == "GIN":
            # NTK for GIN with one hidden layer MLP
            # GIN = ReLU((A+I)XW_1)W_2
            kernel = self._calc_ntk_gin(X, S)
            self.empty_gpu_memory()
            return kernel
        else:
            raise NotImplementedError("Only GCN/SoftMedoid/(A)PPNP/GIN architecture implemented")
 
    
    @staticmethod
    def _calc_XXT(X: Float[torch.Tensor, "n d"]):
        """Uniform method to calculate XXT. Ensures symmetry.
        
        Choosing lower triangular part as "numerical truth". Could also use
        upper triangular part.
        """
        XXT = X.matmul(X.T)
        return torch.tril(XXT) + torch.tril(XXT, diagonal=-1).T

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
            XXT = NTK._calc_XXT(X)
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
        elif perturbation_model == "linf":
            XXT = NTK._calc_XXT(X)
            X_1norm = torch.linalg.vector_norm(X,ord=1,dim=1)
            Delta_l = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            Delta_u = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            # D^TD Interaction Term
            DD_l = Delta_l[idx_adv, :]
            DD_l[:, idx_adv] = -X.shape[1]*delta*delta
            Delta_l[idx_adv, :] = DD_l
            assert (Delta_l != Delta_l.T).sum() == 0
            DD_u = Delta_u[idx_adv, :]
            DD_u[:, idx_adv] = X.shape[1]*delta*delta
            Delta_u[idx_adv, :] = DD_u
            assert (Delta_u != Delta_u.T).sum() == 0
            # D^TX and X^TD Terms
            delta_times_X_1norm = delta*X_1norm
            Delta_l[idx_adv, :] -= delta_times_X_1norm.view(1,-1)
            Delta_l[:, idx_adv] -= delta_times_X_1norm.view(-1,1)
            Delta_l.fill_diagonal_(0.)
            Delta_l = torch.tril(Delta_l) + torch.tril(Delta_l, diagonal=-1).T
            Delta_u[idx_adv, :] += delta_times_X_1norm.view(1,-1)
            Delta_u[:, idx_adv] += delta_times_X_1norm.view(-1,1)
            Delta_u = torch.tril(Delta_u) + torch.tril(Delta_u, diagonal=-1).T
            assert (Delta_l != Delta_l.T).sum() == 0
            assert (Delta_u != Delta_u.T).sum() == 0
            assert (XXT != XXT.T).sum() == 0
            # Symmetrice (due to numerical issues)
            XXT_lb = XXT+Delta_l
            XXT_ub = XXT+Delta_u
            return XXT_lb, XXT_ub
        elif perturbation_model == "l2":
            XXT = NTK._calc_XXT(X)
            X_2norm = torch.linalg.vector_norm(X,ord=2,dim=1)
            Delta_l = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            Delta_u = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            # D^TD Interaction Term
            DD_l = Delta_l[idx_adv, :]
            DD_l[:, idx_adv] = -delta*delta
            Delta_l[idx_adv, :] = DD_l
            assert (Delta_l != Delta_l.T).sum() == 0
            DD_u = Delta_u[idx_adv, :]
            DD_u[:, idx_adv] = delta*delta
            Delta_u[idx_adv, :] = DD_u
            assert (Delta_u != Delta_u.T).sum() == 0
            # D^TX and X^TD Terms
            delta_times_X_2norm = delta*X_2norm
            Delta_l[idx_adv, :] -= delta_times_X_2norm.view(1,-1)
            Delta_l[:, idx_adv] -= delta_times_X_2norm.view(-1,1)
            Delta_l.fill_diagonal_(0.)
            Delta_l = torch.tril(Delta_l) + torch.tril(Delta_l, diagonal=-1).T
            Delta_u[idx_adv, :] += delta_times_X_2norm.view(1,-1)
            Delta_u[:, idx_adv] += delta_times_X_2norm.view(-1,1)
            Delta_u = torch.tril(Delta_u) + torch.tril(Delta_u, diagonal=-1).T
            assert (Delta_l != Delta_l.T).sum() == 0
            assert (Delta_u != Delta_u.T).sum() == 0
            assert (XXT != XXT.T).sum() == 0
            # Symmetrice (due to numerical issues)
            XXT_lb = XXT+Delta_l
            XXT_ub = XXT+Delta_u
            return XXT_lb, XXT_ub
        elif perturbation_model == "l2_binary_feature":
            # Assumes original feature X is binary {0,1}^d
            # Perturbation linf results in X \in [0,1] domain
            XXT = NTK._calc_XXT(X)
            Delta_l = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            Delta_u = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            X_adv = torch.zeros(size=X.shape, dtype=self.dtype, device=self.device)
            X_adv[idx_adv, :] = X[idx_adv, :]
            # D^TD Interaction Term
            DD_l = Delta_l[idx_adv, :]
            DD_l[:, idx_adv] = -delta*delta
            Delta_l[idx_adv, :] = DD_l
            assert (Delta_l != Delta_l.T).sum() == 0
            DD_u = Delta_u[idx_adv, :]
            DD_u[:, idx_adv] = delta*delta
            Delta_u[idx_adv, :] = DD_u
            assert (Delta_u != Delta_u.T).sum() == 0
            # D^TX and X^TD Terms
            for idx in idx_adv:
                x_adv = (X_adv[idx]==1).type(self.dtype)
                X_2norm = torch.linalg.vector_norm(X*x_adv,ord=2,dim=1) 
                delta_times_X_2norm = delta*X_2norm
                Delta_l[idx, :] -= delta_times_X_2norm
                Delta_l[:, idx] -= delta_times_X_2norm
                x_adv = (X_adv[idx]==0).type(self.dtype)
                X_2norm = torch.linalg.vector_norm(X*x_adv,ord=2,dim=1) 
                delta_times_X_2norm = delta*X_2norm
                Delta_u[idx, :] += delta_times_X_2norm
                Delta_u[:, idx] += delta_times_X_2norm
            Delta_l.fill_diagonal_(0.)
            Delta_l = torch.tril(Delta_l) + torch.tril(Delta_l, diagonal=-1).T
            Delta_u = torch.tril(Delta_u) + torch.tril(Delta_u, diagonal=-1).T
            assert (Delta_l != Delta_l.T).sum() == 0
            assert (Delta_u != Delta_u.T).sum() == 0
            assert (XXT != XXT.T).sum() == 0
            # Symmetrice (due to numerical issues)
            XXT_lb = XXT+Delta_l
            XXT_ub = XXT+Delta_u
            return XXT_lb, XXT_ub
        elif perturbation_model == "linf_binary_feature":
            # Assumes original feature X is binary {0,1}^d
            # Perturbation linf results in X \in [0,1] domain
            XXT = NTK._calc_XXT(X)
            Delta_l = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            Delta_u = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            X_adv = torch.zeros(size=X.shape, dtype=self.dtype, device=self.device)
            X_adv[idx_adv, :] = X[idx_adv, :]
            # D^TD Interaction Term
            for idx_i in idx_adv:
                for idx_j in idx_adv:
                    x_adv_i = X_adv[idx_i]==1
                    x_adv_j = X_adv[idx_j]==1
                    Delta_l[idx_i, idx_j] += -torch.logical_and(~x_adv_i, x_adv_j).sum()*delta*delta
                    Delta_l[idx_i, idx_j] += -torch.logical_and(x_adv_i, ~x_adv_j).sum()*delta*delta
                    Delta_l[idx_j, idx_i] = Delta_l[idx_i, idx_j]
                    Delta_u[idx_i, idx_j] += torch.logical_and(~x_adv_i, ~x_adv_j).sum()*delta*delta
                    Delta_u[idx_i, idx_j] += torch.logical_and(x_adv_i, x_adv_j).sum()*delta*delta
                    Delta_u[idx_j, idx_i] = Delta_u[idx_i, idx_j]
            assert (Delta_l != Delta_l.T).sum() == 0
            assert (Delta_u != Delta_u.T).sum() == 0
            # D^TX and X^TD Terms
            for idx in idx_adv:
                x_adv = (X_adv[idx]==1).type(self.dtype)
                X_1norm = torch.linalg.vector_norm(X*x_adv,ord=1,dim=1) 
                delta_times_X_1norm = delta*X_1norm
                Delta_l[idx, :] -= delta_times_X_1norm
                Delta_l[:, idx] -= delta_times_X_1norm
                x_adv = (X_adv[idx]==0).type(self.dtype)
                X_1norm = torch.linalg.vector_norm(X*x_adv,ord=1,dim=1) 
                delta_times_X_1norm = delta*X_1norm
                Delta_u[idx, :] += delta_times_X_1norm
                Delta_u[:, idx] += delta_times_X_1norm
            Delta_l.fill_diagonal_(0.)
            Delta_l = torch.tril(Delta_l) + torch.tril(Delta_l, diagonal=-1).T
            Delta_u = torch.tril(Delta_u) + torch.tril(Delta_u, diagonal=-1).T
            assert (Delta_l != Delta_l.T).sum() == 0
            assert (Delta_u != Delta_u.T).sum() == 0
            assert (XXT != XXT.T).sum() == 0
            # Symmetrice (due to numerical issues)
            XXT_lb = XXT+Delta_l
            XXT_ub = XXT+Delta_u
            return XXT_lb, XXT_ub
        elif perturbation_model == "l1":
            XXT = NTK._calc_XXT(X)
            X_inf_norm = torch.linalg.vector_norm(X,ord=torch.inf,dim=1)
            Delta_l = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            Delta_u = torch.zeros(size=XXT.shape, dtype=self.dtype, device=self.device)
            # D^TD Interaction Term
            DD_l = Delta_l[idx_adv, :]
            DD_l[:, idx_adv] = -delta*delta
            Delta_l[idx_adv, :] = DD_l
            assert (Delta_l != Delta_l.T).sum() == 0
            DD_u = Delta_u[idx_adv, :]
            DD_u[:, idx_adv] = delta*delta
            Delta_u[idx_adv, :] = DD_u
            assert (Delta_u != Delta_u.T).sum() == 0
            # D^TX and X^TD Terms
            delta_times_X_inf_norm = delta*X_inf_norm
            Delta_l[idx_adv, :] -= delta_times_X_inf_norm.view(1,-1)
            Delta_l[:, idx_adv] -= delta_times_X_inf_norm.view(-1,1)
            Delta_l.fill_diagonal_(0.)
            Delta_l = torch.tril(Delta_l) + torch.tril(Delta_l, diagonal=-1).T
            Delta_u[idx_adv, :] += delta_times_X_inf_norm.view(1,-1)
            Delta_u[:, idx_adv] += delta_times_X_inf_norm.view(-1,1)
            Delta_u = torch.tril(Delta_u) + torch.tril(Delta_u, diagonal=-1).T
            assert (Delta_l != Delta_l.T).sum() == 0
            assert (Delta_u != Delta_u.T).sum() == 0
            assert (XXT != XXT.T).sum() == 0
            # Symmetrice (due to numerical issues)
            XXT_lb = XXT+Delta_l
            XXT_ub = XXT+Delta_u
            return XXT_lb, XXT_ub
        else:
            assert False, f"Perturbation model {perturbation_model} not supported"

    def calc_SXXTS_ub(self, X: Float[torch.Tensor, "n d"],
                      S: Float[torch.Tensor, "n n"],
                      idx_adv: Integer[np.ndarray, "r"],
                      delta: Union[float, int],
                      perturbation_model: str):
        if perturbation_model == "l0":
            C = S.matmul(X)
            assert len(idx_adv) == 1, "Only implemented single node attack."
            S_idxadv = S[idx_adv, :].view(-1)
            SS = S_idxadv.outer(S_idxadv)
            SXXTS_ub = torch.zeros((S.shape[0], S.shape[0]), device=self.device, dtype=self.dtype)
            # Calculates SXXTS elements in [:,:spit_size] juncks
            split_size = 100
            for i in range(0, len(S_idxadv), split_size):
                utils.empty_gpu_memory(self.device)
                if False:
                    print(f"Entering {i}")
                split_end = i + split_size
                if split_end > len(S_idxadv):
                    split_end = len(S_idxadv)
                    split_size = split_end - i
                assert len(S_idxadv) == C.shape[0]
                cS = torch.einsum("i,jk->ijk",S_idxadv,C[i:split_end,:])
                cST = torch.einsum("i,jk->jik",S_idxadv[i:split_end],C)
                W = cS
                W += cST
                del cST
                utils.empty_gpu_memory(self.device)
                W += SS[:,i:split_end].view((SS.shape[0], split_size, 1))
                W = torch.topk(W,k=delta,dim=2)[0].sum(dim=2)
                SXXTS_ub[:,i:split_end] = W
                del W
                utils.empty_gpu_memory(self.device)
            XXT = NTK._calc_XXT(X)
            Sig = S.matmul(XXT.matmul(S.T))
            return Sig + SXXTS_ub
        elif perturbation_model == "l0_del":
            C = S.matmul(X)
            assert len(idx_adv) == 1, "Only implemented single node attack."
            S_idxadv = S[idx_adv, :].view(-1)
            SS = S_idxadv.outer(S_idxadv)
            SXXTS_ub = torch.zeros((S.shape[0], S.shape[0]), device=self.device, dtype=self.dtype)
            # Calculates SXXTS elements in [:,:spit_size] juncks
            split_size = 100
            for i in range(0, len(S_idxadv), split_size):
                utils.empty_gpu_memory(self.device)
                if False:
                    print(f"Entering {i}")
                split_end = i + split_size
                if split_end > len(S_idxadv):
                    split_end = len(S_idxadv)
                    split_size = split_end - i
                assert len(S_idxadv) == C.shape[0]
                cS = torch.einsum("i,jk->ijk",S_idxadv,C[i:split_end,:])
                cST = torch.einsum("i,jk->jik",S_idxadv[i:split_end],C)
                W = -cS
                W -= cST
                del cST
                utils.empty_gpu_memory(self.device)
                W += SS[:,i:split_end].view((SS.shape[0], split_size, 1))
                W = torch.topk(W,k=delta,dim=2)[0].sum(dim=2)
                W[W<0] = 0 # then, always better to choose 0 instead of -1
                SXXTS_ub[:,i:split_end] = W
                del W
                utils.empty_gpu_memory(self.device)
            XXT = NTK._calc_XXT(X)
            Sig = S.matmul(XXT.matmul(S.T))
            return Sig + SXXTS_ub
        else:
            assert False, "Only l0 perturbation model implemented."

    def calc_SXXTS_lb(self, X: Float[torch.Tensor, "n d"],
                      S: Float[torch.Tensor, "n n"],
                      idx_adv: Integer[np.ndarray, "r"],
                      delta: Union[float, int],
                      perturbation_model: str):
        if perturbation_model in ["l0", "l0_del"]:
            C = S.matmul(X)
            assert len(idx_adv) == 1, "Only implemented single node attack."
            S_idxadv = S[idx_adv, :].view(-1)
            SS = S_idxadv.outer(S_idxadv)
            SXXTS_lb = torch.zeros((S.shape[0], S.shape[0]), device=self.device, dtype=self.dtype)
            # Calculates SXXTS elements in [:,:split_size] juncks to save memory
            split_size = 100
            for i in range(0, len(S_idxadv), split_size):
                utils.empty_gpu_memory(self.device)
                if False:
                    print(f"Entering {i}")
                split_end = i + split_size
                if split_end > len(S_idxadv):
                    split_end = len(S_idxadv)
                    split_size = split_end - i
                assert len(S_idxadv) == C.shape[0]
                cS = torch.einsum("i,jk->ijk",S_idxadv,C[i:split_end,:])
                cST = torch.einsum("i,jk->jik",S_idxadv[i:split_end],C)
                W = cS
                W += cST
                del cST
                utils.empty_gpu_memory(self.device)
                W -= SS[:,i:split_end].view((SS.shape[0], split_size, 1))
                W = torch.topk(W,k=delta,dim=2)[0].sum(dim=2)
                W[W<0] = 0 # then, always better to choose 0 instead of -1
                SXXTS_lb[:,i:split_end] = W
                del W
                utils.empty_gpu_memory(self.device)
            XXT = NTK._calc_XXT(X)
            Sig = S.matmul(XXT.matmul(S.T))
            SXXTS_lb = Sig - SXXTS_lb
            SXXTS_lb[SXXTS_lb<0] = 0
            return SXXTS_lb
        else:
            assert False, "Only l0 perturbation model implemented."

    def _calc_relu_expectations_lb_ub(self, 
                       Sig_lb: Float[torch.Tensor, "n n"],
                       Sig_ub: Float[torch.Tensor, "n n"],
                       E: Float[torch.Tensor, "n n"],
                       E_der: Float[torch.Tensor, "n n"],
                       u: float,
                       idx_adv: Integer[np.ndarray, "r"],
                       perturbation_model: str):
        csigma = 1
        p = torch.zeros((E.shape), dtype=self.dtype).to(self.device)
        Diag_Sig_lb = torch.diagonal(Sig_lb) 
        assert (Diag_Sig_lb < 0).sum() == 0
        Diag_Sig_ub = torch.diagonal(Sig_ub) 
        assert (Diag_Sig_ub == 0).sum() == 0
        Sig_i_lb = p + Diag_Sig_lb.reshape(1, -1)
        Sig_j_lb = p + Diag_Sig_lb.reshape(-1, 1)
        Sig_i_ub = p + Diag_Sig_ub.reshape(1, -1)
        Sig_j_ub = p + Diag_Sig_ub.reshape(-1, 1)
        # ensure covariance_ij <= max{var_i, var_j} : could happen otherwise because of numerical precision issues
        cov_gr_var = torch.logical_or((Sig_lb>Sig_i_lb), (Sig_lb>Sig_j_lb))
        Sig_lb[cov_gr_var] = torch.min(Sig_i_lb[cov_gr_var], Sig_j_lb[cov_gr_var])
        cov_gr_var = torch.logical_or((Sig_ub>Sig_i_ub), (Sig_ub>Sig_j_ub))
        Sig_ub[cov_gr_var] = torch.min(Sig_i_ub[cov_gr_var], Sig_j_ub[cov_gr_var])
        q_lb = torch.sqrt(Sig_i_lb * Sig_j_lb) #+ 1e-7 q_lb_{ij} = Sigma_ii * Sigma_jj
        q_ub = torch.sqrt(Sig_i_ub * Sig_j_ub)
        assert (q_lb < 0).sum() == 0
        assert (q_ub <= 0).sum() == 0
        mask_pos = Sig_lb >= 0
        mask_neg = ~mask_pos
        u_lb = torch.zeros(Sig_lb.shape, device=self.device, dtype=self.dtype)
        u_lb[mask_pos] = Sig_lb[mask_pos]/q_ub[mask_pos]
        u_lb[mask_neg] = Sig_lb[mask_neg]/q_lb[mask_neg]
        u_lb[u_lb < -1] = -1
        u_lb[u_lb > 1] = 1
        assert (u_lb > u).sum() == 0
        assert torch.isnan(u_lb).sum() == 0
        u_ub = torch.zeros(Sig_lb.shape, device=self.device, dtype=self.dtype)
        mask_pos = Sig_ub >= 0
        mask_neg = ~mask_pos
        u_ub[mask_pos] = Sig_ub[mask_pos]/q_lb[mask_pos] 
        u_ub[mask_neg] = Sig_ub[mask_neg]/q_ub[mask_neg]
        u_ub[u_ub > 1] = 1
        u_ub[u_ub < -1] = -1
        u_ub[torch.isnan(u_ub)] = 0
        assert (u_ub < u).sum() == 0
        mask_abs_u = torch.abs(Sig_ub) >= torch.abs(Sig_lb)
        mask_abs_l = ~mask_abs_u
        #For u_ub_sq use same calculation scheme as for NTK
        #Sligthly better (numerically on a scale of 1e-16 & faster) would 
        #be the following, however - then NTK would need more storage!
        #u_ub_sq[mask_abs_u] = Sig_ub[mask_abs_u] * Sig_ub[mask_abs_u] \
        #                        / q_lb[mask_abs_u] * q_lb[mask_abs_u])
        #u_ub_sq[mask_abs_l] = Sig_lb[mask_abs_l] * Sig_lb[mask_abs_l] \
        #                        / (q_lb[mask_abs_l] * q_lb[mask_abs_l])
        #u_lb_sq[mask_abs_u] = Sig_lb[mask_abs_u] * Sig_lb[mask_abs_u] \
        #                    / (q_ub[mask_abs_u] * q_ub[mask_abs_u])
        #u_lb_sq[mask_abs_l] = Sig_ub[mask_abs_l] * Sig_ub[mask_abs_l] \
        #                    / (q_ub[mask_abs_l] * q_ub[mask_abs_l])
        u_ub_sq = torch.zeros(Sig_lb.shape, device=self.device, dtype=self.dtype)
        u_ub_sq[mask_abs_u] = Sig_ub[mask_abs_u] / q_lb[mask_abs_u]
        u_ub_sq[mask_abs_u] *= u_ub_sq[mask_abs_u]
        u_ub_sq[mask_abs_l] = Sig_lb[mask_abs_l] / q_lb[mask_abs_l]
        u_ub_sq[mask_abs_l] *= u_ub_sq[mask_abs_l]
        u_ub_sq[u_ub_sq > 1] = 1
        u_ub_sq[u_ub_sq < 0] = 0
        u_ub_sq[torch.isnan(u_ub_sq)] = 0
        u_lb_sq = torch.zeros(Sig_lb.shape, device=self.device, dtype=self.dtype)
        u_lb_sq[mask_abs_u] = Sig_lb[mask_abs_u] / q_ub[mask_abs_u]
        u_lb_sq[mask_abs_u] *= u_lb_sq[mask_abs_u]
        u_lb_sq[mask_abs_l] = Sig_ub[mask_abs_l] / q_ub[mask_abs_l]
        u_lb_sq[mask_abs_l] *= u_lb_sq[mask_abs_l]
        u_lb_sq[u_lb_sq > 1] = 1
        u_lb_sq[u_lb_sq < 0] = 0
        u_lb_sq[torch.isnan(u_lb_sq)] = 0
        assert (u_ub_sq < u_lb_sq).sum() == 0
        if perturbation_model == "l0":
            assert (Sig_lb < 0).sum() == 0
            assert (Sig_ub < 0).sum() == 0
        E_lb = (q_lb * self.kappa_1_lb(u_lb, u_ub_sq)) * csigma
        E_lb[E_lb < 0] = 0
        E_ub = (q_ub * self.kappa_1_ub(u_ub, u_lb_sq)) * csigma
        assert (E_ub < E_lb).sum() == 0
        assert (E_lb > E).sum() == 0
        assert (E_ub < E).sum() == 0
        self.empty_gpu_memory()
        E_der_lb = self.kappa_0_lb(u_lb)
        assert (E_der_lb < 0).sum() == 0
        assert (E_der_lb > 1).sum() == 0
        E_der_lb = E_der_lb * csigma
        E_der_ub = self.kappa_0_ub(u_ub) 
        assert (E_der_ub < 0).sum() == 0
        assert (E_der_ub > 1).sum() == 0
        E_der_ub = E_der_ub * csigma
        assert (E_der_ub < E_der).sum() == 0
        assert (E_der_lb > E_der).sum() == 0
        assert (E_der_ub < E_der_lb).sum() == 0
        if globals.debug:
            print(f"E_der.mean(): {E_der.mean()}")
            print(f"E_der.min(): {E_der.min()}")
            print(f"E_der.max(): {E_der.max()}")
            print(f"E_der[:,idx_adv]: {E_der[:,idx_adv].mean()}")
            print(f"E_der_lb.mean(): {E_der_lb.mean()}")
            print(f"E_der_lb.min(): {E_der_lb.min()}")
            print(f"E_der_lb.max(): {E_der_lb.max()}")
            print(f"E_der_lb[:,idx_adv]: {E_der_lb[:,idx_adv].mean()}")
            print(f"E_der_ub.mean(): {E_der_ub.mean()}")
            print(f"E_der_ub.min(): {E_der_ub.min()}")
            print(f"E_der_ub.max(): {E_der_ub.max()}")
            print(f"E_der_ub[:,idx_adv]: {E_der_ub[:,idx_adv].mean()}")
        self.empty_gpu_memory()
        assert (Sig_lb > Sig_ub).sum() == 0
        mask_pos = Sig_ub >= 0
        mask_neg = ~mask_pos
        sig_dot_E_der_ub = torch.zeros(Sig_lb.shape, device=self.device, dtype=self.dtype)
        sig_dot_E_der_ub[mask_pos] = (Sig_ub * E_der_ub)[mask_pos]
        sig_dot_E_der_ub[mask_neg] = (Sig_ub * E_der_lb)[mask_neg]
        mask_pos = Sig_lb >= 0
        mask_neg = ~mask_pos
        sig_dot_E_der_lb = torch.zeros(Sig_lb.shape, device=self.device, dtype=self.dtype)
        sig_dot_E_der_lb[mask_pos] = (Sig_lb * E_der_lb)[mask_pos]
        sig_dot_E_der_lb[mask_neg] = (Sig_lb * E_der_ub)[mask_neg]
        return E_lb, E_ub, E_der_lb, E_der_ub, sig_dot_E_der_lb, sig_dot_E_der_ub
    
    def calc_gcn_lb_ub(self, X: Float[torch.Tensor, "n d"], 
                       A: Float[torch.Tensor, "n n"],
                       idx_adv: Integer[np.ndarray, "r"],
                       delta: float,
                       perturbation_model: str,
                       method: str="SXXTS"):
        csigma = 1 
        A = make_dense(A)
        S = self.calc_diffusion(X, A)
        XXT = NTK._calc_XXT(X)
        if method == "XXT":
            XXT_lb, XXT_ub = self.calc_XXT_lb_ub(X, idx_adv, delta, perturbation_model)
            assert (XXT_lb != XXT_lb.T).sum() == 0
            assert (XXT_ub != XXT_ub.T).sum() == 0
            assert (XXT_lb > XXT_ub).sum() == 0
            assert (XXT_lb > XXT).sum() == 0
            assert (XXT_ub < XXT).sum() == 0
        self.empty_gpu_memory()
        Sig = S.matmul(XXT.matmul(S.T))
        if method == "XXT":
            Sig_lb = S.matmul(XXT_lb.matmul(S.T))
            diag = Sig_lb.diag()
            diag[diag < 0] = 0
            mask = torch.eye(diag.shape[0], dtype=bool, device=self.device)
            Sig_lb[mask] = diag
            Sig_ub = S.matmul(XXT_ub.matmul(S.T))
        else:
            Sig_ub = self.calc_SXXTS_ub(X, S, idx_adv, delta, perturbation_model)
            Sig_lb = self.calc_SXXTS_lb(X, S, idx_adv, delta, perturbation_model)
        assert (Sig_lb > Sig_ub).sum() == 0
        assert (Sig_lb > Sig).sum() == 0
        assert (Sig_ub < Sig).sum() == 0
        if globals.debug:
            print(f"Sig.mean(): {Sig.mean()}")
            print(f"Sig.min(): {Sig.min()}")
            print(f"Sig.max(): {Sig.max()}")
            print(f"Sig[:,idx_adv]: {Sig[:,idx_adv].mean()}")
            print(f"Sig_ub.mean(): {Sig_ub.mean()}")
            print(f"Sig_ub.min(): {Sig_ub.min()}")
            print(f"Sig_ub.max(): {Sig_ub.max()}")
            print(f"Sig_ub[:,idx_adv]: {Sig_ub[:,idx_adv].mean()}")
            print(f"Sig_lb.mean(): {Sig_lb.mean()}")
            print(f"Sig_lb.min(): {Sig_lb.min()}")
            print(f"Sig_lb.max(): {Sig_lb.max()}")
            print(f"Sig_lb[:,idx_adv]: {Sig_lb[:,idx_adv].mean()}")
        self.empty_gpu_memory()
        ntk_lb = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        ntk_ub = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        ntk = torch.zeros((S.shape), dtype=self.dtype).to(self.device)

        depth = self.model_dict["depth"]
        ntk_lb_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                dtype=self.dtype).to(self.device)
        ntk_ub_sub = torch.zeros((depth, S.shape[0], S.shape[1]), 
                                dtype=self.dtype).to(self.device)
        ntk_sub = torch.zeros((depth, S.shape[0], S.shape[1]), # only for debug
                                dtype=self.dtype).to(self.device)
        if "skip_connection" in self.model_dict:
            # skip_alpha should be set, otherwise do
            # get skip_alpha, if not given then set it to 0.2
            if self.model_dict["skip_connection"] == "skip_alpha_linear" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                self.skip_alpha = self.model_dict["skip_alpha"] if "skip_alpha" in self.model_dict \
                    else 0.2
            # compute Sig and Sig_skip for the residual component
            if self.model_dict["skip_connection"] == "skip_pc_relu" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                E_skip, E_der_skip, u_skip = self.calc_relu_expectations(XXT)
                Sig = S.matmul((E_skip).matmul(S.T))
                E_skip_lb, E_skip_ub, _, _, _, _ = \
                    self._calc_relu_expectations_lb_ub(
                        XXT_lb, XXT_ub, E_skip, E_der_skip, u_skip, idx_adv, perturbation_model
                    )
                if method == "XXT":
                    Sig_lb = S.matmul(E_skip_lb.matmul(S.T))
                    diag = Sig_lb.diag()
                    diag[diag < 0] = 0
                    mask = torch.eye(diag.shape[0], dtype=bool, device=self.device)
                    Sig_lb[mask] = diag
                    Sig_ub = S.matmul(E_skip_ub.matmul(S.T))
                else:
                    assert False, "Perturbation method evaluation verified completely only for XXT."
                if self.model_dict["skip_connection"] == "skip_pc_relu":
                    Sig_skip = Sig
                    Sig_skip_lb = Sig_lb
                    Sig_skip_ub = Sig_ub
                elif self.model_dict["skip_connection"] == "skip_alpha_relu":
                    Sig = ((1-self.skip_alpha)**2 * Sig + \
                        (1-self.skip_alpha)*self.skip_alpha*(E_skip.matmul(S.T) + S.matmul(E_skip)) + \
                        self.skip_alpha**2 * E_skip) * csigma
                    Sig_lb = ((1-self.skip_alpha)**2 * Sig_lb + \
                        (1-self.skip_alpha)*self.skip_alpha*(E_skip_lb.matmul(S.T) + S.matmul(E_skip_lb)) + \
                        self.skip_alpha**2 * E_skip_lb) * csigma
                    diag = Sig_lb.diag()
                    diag[diag < 0] = 0
                    mask = torch.eye(diag.shape[0], dtype=bool, device=self.device)
                    Sig_lb[mask] = diag
                    Sig_ub = ((1-self.skip_alpha)**2 * Sig_ub + \
                        (1-self.skip_alpha)*self.skip_alpha*(E_skip_ub.matmul(S.T) + S.matmul(E_skip_ub)) + \
                        self.skip_alpha**2 * E_skip_ub) * csigma
                    Sig_skip = E_skip
                    Sig_skip_lb = E_skip_lb
                    Sig_skip_ub = E_skip_ub
            elif self.model_dict["skip_connection"] == "skip_pc_linear": 
                Sig_skip = Sig
                Sig_skip_lb = Sig_lb
                Sig_skip_ub = Sig_ub
            elif self.model_dict["skip_connection"] == "skip_alpha_linear":
                Sig = ((1-self.skip_alpha)**2 * Sig + \
                        (1-self.skip_alpha)*self.skip_alpha*(XXT.matmul(S.T) + S.matmul(XXT)) + \
                        self.skip_alpha**2 * XXT) * csigma
                Sig_lb = ((1-self.skip_alpha)**2 * Sig_lb + \
                        (1-self.skip_alpha)*self.skip_alpha*(XXT_lb.matmul(S.T) + S.matmul(XXT_lb)) + \
                        self.skip_alpha**2 * XXT_lb) * csigma
                diag = Sig_lb.diag()
                diag[diag < 0] = 0
                mask = torch.eye(diag.shape[0], dtype=bool, device=self.device)
                Sig_lb[mask] = diag
                Sig_ub = ((1-self.skip_alpha)**2 * Sig_ub + \
                        (1-self.skip_alpha)*self.skip_alpha*(XXT_ub.matmul(S.T) + S.matmul(XXT_ub)) + \
                        self.skip_alpha**2 * XXT_ub) * csigma
                Sig_skip = XXT
                Sig_skip_lb = XXT_lb
                Sig_skip_ub = XXT_ub
            else:
                assert False, f"'skip_pc_linear', 'skip_pc_relu', 'skip_alpha_linear' and 'skip_alpha_relu' are only supported. "
            assert (Sig_skip_lb > Sig_skip_ub).sum() == 0
            assert (Sig_skip_lb > Sig_skip).sum() == 0
            assert (Sig_skip_ub < Sig_skip).sum() == 0
        for i in range(depth):
            if globals.debug:
                print(f"Depth {i}")
            # only for debug
            if self.model_dict["activation"] == 'relu':
                E, E_der, u = self.calc_relu_expectations(Sig)
            elif self.model_dict["activation"] == 'linear':
                E = Sig
                self.empty_gpu_memory()
                E_der = torch.ones((S.shape), dtype=self.dtype).to(self.device)
                self.empty_gpu_memory()
            else:
                assert False, f"'linear' and 'relu' GCNs are supported. Please update activation in the model_dict."
            ntk_sub[i] = S.matmul((Sig * E_der).matmul(S.T))
            Sig = S.matmul(E.matmul(S.T))
            if "skip_connection" in self.model_dict:
                if self.model_dict["skip_connection"] == "skip_pc_linear" or \
                self.model_dict["skip_connection"] == "skip_pc_relu"  :
                    Sig += Sig_skip
                elif self.model_dict["skip_connection"] == "skip_alpha_linear" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                    Sig = (1-self.skip_alpha)**2 * Sig + self.skip_alpha**2 * Sig_skip 
                else:
                    assert False, f"'skip_pc_linear', 'skip_pc_relu', 'skip_alpha_linear' and 'skip_alpha_relu' are only supported. "
            for j in range(i):
                ntk_sub[j] = S.matmul((ntk_sub[j].float() * E_der).matmul(S.T))
            ######################
            if self.model_dict["activation"] == 'relu':
                E_lb, E_ub, E_der_lb, E_der_ub, sig_dot_E_der_lb, sig_dot_E_der_ub = \
                    self._calc_relu_expectations_lb_ub(
                        Sig_lb, Sig_ub, E, E_der, u, idx_adv, perturbation_model
                    )
            elif self.model_dict["activation"] == 'linear': 
                E_lb = Sig_lb
                E_ub = Sig_ub
                E_der_lb = torch.ones((S.shape), dtype=self.dtype).to(self.device)
                E_der_ub = torch.ones((S.shape), dtype=self.dtype).to(self.device)
                sig_dot_E_der_lb = Sig_lb * E_der_lb
                sig_dot_E_der_ub = Sig_ub * E_der_ub
            else:
                assert False, f"'linear' and 'relu' GCNs are supported. Please update activation in the model_dict."
            ntk_lb_sub[i] = S.matmul(sig_dot_E_der_lb.matmul(S.T))
            ntk_ub_sub[i] = S.matmul(sig_dot_E_der_ub.matmul(S.T))
            assert (ntk_lb_sub[i] > ntk_ub_sub[i]).sum() == 0
            Sig_lb = S.matmul(E_lb.matmul(S.T))
            Sig_ub = S.matmul(E_ub.matmul(S.T))
            assert (Sig_lb > Sig_ub).sum() == 0
            if "skip_connection" in self.model_dict:
                if self.model_dict["skip_connection"] == "skip_pc_linear" or \
                self.model_dict["skip_connection"] == "skip_pc_relu"  :
                    Sig_lb += Sig_skip_lb
                    Sig_ub += Sig_skip_ub
                elif self.model_dict["skip_connection"] == "skip_alpha_linear" or \
                self.model_dict["skip_connection"] == "skip_alpha_relu":
                    Sig_lb = (1-self.skip_alpha)**2 * Sig_lb + self.skip_alpha**2 * Sig_skip_lb 
                    Sig_ub = (1-self.skip_alpha)**2 * Sig_ub + self.skip_alpha**2 * Sig_skip_ub 
                else:
                    assert False, f"'skip_pc_linear', 'skip_pc_relu', 'skip_alpha_linear' and 'skip_alpha_relu' are only supported. "
            assert (Sig_lb > Sig_ub).sum() == 0
            if globals.debug:
                print(f"Sig.mean(): {Sig.mean()}")
                print(f"Sig.min(): {Sig.min()}")
                print(f"Sig.max(): {Sig.max()}")
                print(f"Sig[:,idx_adv]: {Sig[:,idx_adv].mean()}")
                print(f"Sig_lb.mean(): {Sig_lb.mean()}")
                print(f"Sig_lb.min(): {Sig_lb.min()}")
                print(f"Sig_lb.max(): {Sig_lb.max()}")
                print(f"Sig_lb[:,idx_adv]: {Sig_lb[:,idx_adv].mean()}")
                print(f"Sig_ub.mean(): {Sig_ub.mean()}")
                print(f"Sig_ub.min(): {Sig_ub.min()}")
                print(f"Sig_ub.max(): {Sig_ub.max()}")
                print(f"Sig_ub[:,idx_adv]: {Sig_ub[:,idx_adv].mean()}")
            for j in range(i):
                ntk_lb_sub[j] = S.matmul((ntk_lb_sub[j].float() * E_der_lb).matmul(S.T))
                ntk_ub_sub[j] = S.matmul((ntk_ub_sub[j].float() * E_der_ub).matmul(S.T))
        self.empty_gpu_memory()
        ntk += torch.sum(ntk_sub, dim=0)
        ntk_lb += torch.sum(ntk_lb_sub, dim=0)
        ntk_ub += torch.sum(ntk_ub_sub, dim=0)
        ntk += Sig
        ntk_lb += Sig_lb
        ntk_ub += Sig_ub
        self.calculated_lb_ub = True
        self.ntk_lb = ntk_lb
        self.ntk_ub = ntk_ub
        assert (ntk_lb > ntk_ub).sum() == 0
        if globals.debug:
            print(f"ntk.mean(): {ntk.mean()}")
            print(f"ntk.min(): {ntk.min()}")
            print(f"ntk.max(): {ntk.max()}")
            print(f"ntk[:,idx_adv]: {ntk[:,idx_adv].mean()}")
            print(f"ntk_lb.mean(): {ntk_lb.mean()}")
            print(f"ntk_lb.min(): {ntk_lb.min()}")
            print(f"ntk_lb.max(): {ntk_lb.max()}")
            print(f"ntk_lb[:,idx_adv]: {ntk_lb[:,idx_adv].mean()}")
            print(f"ntk_ub.mean(): {ntk_ub.mean()}")
            print(f"ntk_ub.min(): {ntk_ub.min()}")
            print(f"ntk_ub.max(): {ntk_ub.max()}")
            print(f"ntk_ub[:,idx_adv]: {ntk_ub[:,idx_adv].mean()}")
        return self.ntk_lb, self.ntk_ub

    def calc_appnp_lb_ub(self, X: Float[torch.Tensor, "n d"], 
                       A: Float[torch.Tensor, "n n"],
                       idx_adv: Integer[np.ndarray, "r"],
                       delta: float,
                       perturbation_model: str,
                       method: str="SXXTS"):
        A = make_dense(A)
        S = self.calc_diffusion(X, A)
        XXT = NTK._calc_XXT(X)
        if method == "XXT":
            XXT_lb, XXT_ub = self.calc_XXT_lb_ub(X, idx_adv, delta, perturbation_model)
            assert (XXT_lb != XXT_lb.T).sum() == 0
            assert (XXT_ub != XXT_ub.T).sum() == 0
            assert (XXT_lb > XXT_ub).sum() == 0
            assert (XXT_lb > XXT).sum() == 0
            assert (XXT_ub < XXT).sum() == 0
        self.empty_gpu_memory()

        ntk_lb = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        ntk_ub = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        ntk = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        B = torch.ones((S.shape), dtype=self.dtype).to(self.device)
        Sig = XXT+B
        Sig_lb = XXT_lb+B
        Sig_ub = XXT_ub+B
        assert (Sig_lb > Sig_ub).sum() == 0
        assert (Sig_lb > Sig).sum() == 0
        assert (Sig_ub < Sig).sum() == 0
        
        E, E_der, u = self.calc_relu_expectations(Sig)
        ntk += S.matmul(Sig * E_der).matmul(S.T)
        ntk += S.matmul(E+B).matmul(S.T)

        E_lb, E_ub, E_der_lb, E_der_ub, sig_dot_E_der_lb, sig_dot_E_der_ub = \
                    self._calc_relu_expectations_lb_ub(Sig_lb, Sig_ub, E, E_der, u, idx_adv, perturbation_model)
        ntk_lb += S.matmul(sig_dot_E_der_lb).matmul(S.T)
        ntk_lb += S.matmul(E_lb+B).matmul(S.T)
        ntk_ub += S.matmul(sig_dot_E_der_ub).matmul(S.T)
        ntk_ub += S.matmul(E_ub+B).matmul(S.T)    

        self.calculated_lb_ub = True
        self.ntk_lb = ntk_lb
        self.ntk_ub = ntk_ub
        assert (ntk_lb > ntk_ub).sum() == 0
        assert (ntk_lb > ntk).sum() == 0
        assert (ntk_ub < ntk).sum() == 0
        if globals.debug:
            print(f"ntk.mean(): {ntk.mean()}")
            print(f"ntk.min(): {ntk.min()}")
            print(f"ntk.max(): {ntk.max()}")
            print(f"ntk[:,idx_adv]: {ntk[:,idx_adv].mean()}")
            print(f"ntk_lb.mean(): {ntk_lb.mean()}")
            print(f"ntk_lb.min(): {ntk_lb.min()}")
            print(f"ntk_lb.max(): {ntk_lb.max()}")
            print(f"ntk_lb[:,idx_adv]: {ntk_lb[:,idx_adv].mean()}")
            print(f"ntk_ub.mean(): {ntk_ub.mean()}")
            print(f"ntk_ub.min(): {ntk_ub.min()}")
            print(f"ntk_ub.max(): {ntk_ub.max()}")
            print(f"ntk_ub[:,idx_adv]: {ntk_ub[:,idx_adv].mean()}")
        return self.ntk_lb, self.ntk_ub

    def calc_gin_lb_ub(self, X: Float[torch.Tensor, "n d"], 
                       A: Float[torch.Tensor, "n n"],
                       idx_adv: Integer[np.ndarray, "r"],
                       delta: float,
                       perturbation_model: str,
                       method: str="SXXTS"):
        A = make_dense(A)
        S = self.calc_diffusion(X, A)
        XXT = NTK._calc_XXT(X)
        if method == "XXT":
            XXT_lb, XXT_ub = self.calc_XXT_lb_ub(X, idx_adv, delta, perturbation_model)
            assert (XXT_lb != XXT_lb.T).sum() == 0
            assert (XXT_ub != XXT_ub.T).sum() == 0
            assert (XXT_lb > XXT_ub).sum() == 0
            assert (XXT_lb > XXT).sum() == 0
            assert (XXT_ub < XXT).sum() == 0
        self.empty_gpu_memory()

        ntk_lb = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        ntk_ub = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        ntk = torch.zeros((S.shape), dtype=self.dtype).to(self.device)
        Sig = S.matmul(XXT.matmul(S.T))
        Sig_lb = S.matmul(XXT_lb.matmul(S.T))
        Diag_Sig_lb = torch.diagonal(Sig_lb) 
        Diag_Sig_lb_neg_idx = (Diag_Sig_lb<0).nonzero(as_tuple=True)[0]
        Sig_lb[Diag_Sig_lb_neg_idx,Diag_Sig_lb_neg_idx] = 0
        Sig_ub = S.matmul(XXT_ub.matmul(S.T))
        assert (Sig_lb > Sig_ub).sum() == 0
        assert (Sig_lb > Sig).sum() == 0
        assert (Sig_ub < Sig).sum() == 0
        
        E, E_der, u = self.calc_relu_expectations(Sig)
        ntk += (Sig * E_der) + E

        E_lb, E_ub, E_der_lb, E_der_ub, sig_dot_E_der_lb, sig_dot_E_der_ub = \
                    self._calc_relu_expectations_lb_ub(Sig_lb, Sig_ub, E, E_der, u, idx_adv, perturbation_model)
        ntk_lb += sig_dot_E_der_lb + E_lb
        ntk_ub += sig_dot_E_der_ub + E_ub

        self.calculated_lb_ub = True
        self.ntk_lb = ntk_lb
        self.ntk_ub = ntk_ub
        assert (ntk_lb > ntk_ub).sum() == 0
        assert (ntk_lb > ntk).sum() == 0
        assert (ntk_ub < ntk).sum() == 0
        if globals.debug:
            print(f"ntk.mean(): {ntk.mean()}")
            print(f"ntk.min(): {ntk.min()}")
            print(f"ntk.max(): {ntk.max()}")
            print(f"ntk[:,idx_adv]: {ntk[:,idx_adv].mean()}")
            print(f"ntk_lb.mean(): {ntk_lb.mean()}")
            print(f"ntk_lb.min(): {ntk_lb.min()}")
            print(f"ntk_lb.max(): {ntk_lb.max()}")
            print(f"ntk_lb[:,idx_adv]: {ntk_lb[:,idx_adv].mean()}")
            print(f"ntk_ub.mean(): {ntk_ub.mean()}")
            print(f"ntk_ub.min(): {ntk_ub.min()}")
            print(f"ntk_ub.max(): {ntk_ub.max()}")
            print(f"ntk_ub[:,idx_adv]: {ntk_ub[:,idx_adv].mean()}")
        return self.ntk_lb, self.ntk_ub

    def calc_ntk_lb_ub(self, X: Float[torch.Tensor, "n d"], 
                       A: Float[torch.Tensor, "n n"],
                       idx_adv: Integer[np.ndarray, "r"],
                       delta: float,
                       perturbation_model: str,
                       method: str="SXXTS"):
        if self.calculated_lb_ub:
            return self.ntk_lb, self.ntk_ub

        if self.model_dict["model"] == "GCN":
            return self.calc_gcn_lb_ub(X, A, idx_adv, delta, perturbation_model, method)
        elif self.model_dict["model"] == "PPNP" or self.model_dict["model"] == "APPNP":
            return self.calc_appnp_lb_ub(X, A, idx_adv, delta, perturbation_model, method)
        elif self.model_dict["model"] == "GIN":
            return self.calc_gin_lb_ub(X, A, idx_adv, delta, perturbation_model, method)
        else:
            assert False, "Models other than GCN, (A)PPNP, GIN are not implemented so far."

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
        elif learning_setting == "inductive":
            # handle different adj representations
            A_test = make_dense(A_test) # is differentiable
            ntk_test = self.calc_ntk(X_test, A_test)
            if torch.cuda.is_available() and self.device != "cpu":
                torch.cuda.empty_cache()
        else:
            ntk_test = self.ntk

        ntk_labeled = self.ntk 
        if self.idx_trn_labeled is not None: # semi-supervised setting
            ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
            ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
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
                if self.solver == "cvxopt" or self.solver == "qplayer":
                    if self.solver == "cvxopt":
                        alpha = torch.tensor(self.svm, dtype=self.dtype, 
                                            device=self.device)
                    else:
                        alpha = self.svm
                    idx_sup = (alpha > self.alpha_tol)
                    y_sup = y_test[idx_labeled][idx_sup] * 2 - 1
                    alpha_sup = alpha[idx_sup]
                    w = y_sup * alpha_sup
                    bias_tol = max(self.alpha_tol, 1e-10)
                    bias_mask = alpha_sup < (self.regularizer - bias_tol)
                    ntk_sup = ntk_labeled[idx_sup, :]
                    ntk_sup = ntk_sup[:, idx_sup]
                    y_pred = (y_sup * alpha_sup * ntk_unlabeled[:,idx_sup]).sum(dim=1) 
                    if self.bias:
                        if bias_mask.sum() == 0:
                            raise NotImplementedError("All alpha = C, calculating"
                                " bias not supported by cvxopt implementaiton")
                        b = y_sup[bias_mask] - (y_sup * alpha_sup * ntk_sup[bias_mask, :]).sum(dim=1)
                        b = b.mean()
                        y_pred += b
                elif self.solver == "sklearn":
                    # Implementation of self.svm.decision_function(ntk_unlabeled) in PyTorch
                    alpha = torch.tensor(self.svm.dual_coef_, dtype=self.dtype, device=self.device)
                    b = torch.tensor(self.svm.intercept_, dtype=self.dtype, device=self.device)
                    idx_sup = self.svm.support_
                    y_pred = (alpha * ntk_unlabeled[:,idx_sup]).sum(dim=1) + b
                else:
                    assert False
            else:
                if self.solver == "qplayer":
                    alpha = self.svm
                    if self.multiclass_svm_method == "simMSVM":
                        alpha_ = alpha.reshape(-1,1) @ torch.ones((1,self.n_classes), dtype=self.dtype).to(device=self.device)
                        y_labeled_onehot = torch.nn.functional.one_hot(y_test[idx_labeled], num_classes=self.n_classes)
                        alpha = alpha_ * y_labeled_onehot
                    y_pred = ntk_unlabeled @ alpha
                elif self.solver == "sklearn":
                    y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                    idx_sup_set = set()
                    alpha_mean = 0
                    for i, svm in enumerate(self.svm.estimators_):
                        # Implementation of the following scikit-learn function in PyTorch:
                        # - pred = svm.decision_function(ntk_u_cpu) 
                        alpha = torch.tensor(svm.dual_coef_, dtype=self.dtype, device=self.device)
                        alpha_mean += alpha.mean()
                        b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                        idx_sup = svm.support_
                        idx_sup_set.update(idx_sup)
                        #if i == 0:
                        #    print((alpha.reshape(1,-1) * ntk_unlabeled[:,idx_sup]).sum(dim=1))
                        #    print((alpha.reshape(1,-1) * ntk_unlabeled[:,idx_sup]).sum(dim=1).shape)
                        pred = (alpha * ntk_unlabeled[:,idx_sup]).sum(dim=1) + b
                        y_pred[:, i] = pred
                    print("#Number of Support Vectors")
                    print(len(idx_sup_set))
                    print(f"alpha_mean: {alpha_mean / len(self.svm.estimators_)}")
                elif self.solver == "qplayer_one_vs_all":
                    y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                    alpha_l = self.svm
                    for k, alpha in enumerate(alpha_l):
                        idx_sup = (alpha > self.alpha_tol)
                        y_sup = y_test[idx_labeled][idx_sup]
                        y_mask = y_sup == k
                        y_sup[y_mask] = 1
                        y_sup[~y_mask] = -1
                        alpha_sup = alpha[idx_sup]
                        w = y_sup * alpha_sup
                        bias_tol = max(self.alpha_tol, 1e-10)
                        bias_mask = alpha_sup < (self.regularizer - bias_tol)
                        ntk_sup = ntk_labeled[idx_sup, :]
                        ntk_sup = ntk_sup[:, idx_sup]
                        pred = (y_sup * alpha_sup * ntk_unlabeled[:,idx_sup]).sum(dim=1) 
                        if self.bias:
                            if bias_mask.sum() == 0:
                                raise NotImplementedError("All alpha = C, calculating"
                                    " bias not supported by cvxopt implementaiton")
                            b = y_sup[bias_mask] - (y_sup * alpha_sup * ntk_sup[bias_mask, :]).sum(dim=1)
                            b = b.mean()
                            pred += b
                        y_pred[:, k] = pred
                elif self.solver == "MSVM" or self.solver == "simMSVM":
                    y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                    alpha = self.svm
                    if self.solver == "simMSVM":
                        alpha_ = alpha.reshape(-1,1) @ torch.ones((1,self.n_classes), dtype=self.dtype).to(device=self.device)
                        y_labeled_onehot = torch.nn.functional.one_hot(y_test[idx_labeled], num_classes=self.n_classes)
                        alpha = alpha_ * y_labeled_onehot
                    y_pred = ntk_unlabeled @ alpha
                else:
                    assert False
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
                           method: str="SXXTS",
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
                                                       delta, perturbation_model,
                                                       method)
        self.empty_gpu_memory()
        ntk_labeled = self.ntk 
        if self.idx_trn_labeled is not None: # semi-supervised setting
            ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
            ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
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
                if self.solver == "cvxopt" or self.solver == "qplayer":
                    if self.solver == "cvxopt":
                        alpha = torch.tensor(self.svm, dtype=self.dtype, 
                                            device=self.device)
                    else:
                        alpha = self.svm
                    idx_sup = (alpha > self.alpha_tol)
                    y_sup = y_test[idx_labeled][idx_sup] * 2 - 1
                    y_sup_pos_mask = y_sup > 0
                    y_sup_pos = y_sup[y_sup_pos_mask]
                    y_sup_neg = y_sup[~y_sup_pos_mask]
                    alpha_sup = alpha[idx_sup]
                    alpha_sup_pos = alpha_sup[y_sup_pos_mask]
                    alpha_sup_neg = alpha_sup[~y_sup_pos_mask]
                    bias_tol = max(self.alpha_tol, 1e-10)
                    idx_bias = alpha_sup < (self.regularizer - bias_tol)
                    ntk_sup = ntk_labeled[idx_sup, :]
                    ntk_sup = ntk_sup[:, idx_sup]
                    ntk_unlabeled_sup_ub = ntk_unlabeled_ub[:,idx_sup]
                    ntk_unlabeled_sup_lb = ntk_unlabeled_lb[:,idx_sup]
                     
                    y_pred = (y_sup_pos * alpha_sup_pos * ntk_unlabeled_sup_ub[:,y_sup_pos_mask]).sum(dim=1) \
                        + (y_sup_neg * alpha_sup_neg * ntk_unlabeled_sup_lb[:,~y_sup_pos_mask]).sum(dim=1)
                    if self.bias:
                        b = y_sup[idx_bias] - (y_sup * alpha_sup * ntk_sup[idx_bias, :]).sum(dim=1)
                        b = b.mean()
                        y_pred += b
                elif self.solver == 'sklearn':
                    # Implementation of self.svm.decision_function(ntk_unlabeled) in PyTorch
                    svm = self.svm
                    alpha_pos_mask = svm.dual_coef_ > 0
                    alpha_pos = torch.tensor(svm.dual_coef_[alpha_pos_mask], 
                                            dtype=self.dtype, device=self.device)
                    alpha_neg = torch.tensor(svm.dual_coef_[~alpha_pos_mask], 
                                            dtype=self.dtype, device=self.device)
                    b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                    alpha_pos_mask = alpha_pos_mask.reshape(-1)
                    idx_sup_pos = svm.support_[alpha_pos_mask]
                    idx_sup_neg = svm.support_[~alpha_pos_mask]
                    #if i == 0:
                    #    print((alpha_pos.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_pos]).sum(dim=1) \
                    #    + (alpha_neg.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_neg]).sum(dim=1))
                    y_pred = (alpha_pos.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_pos]).sum(dim=1) \
                        + (alpha_neg.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_neg]).sum(dim=1) \
                        + b
                else:
                    assert False
            else:
                if self.solver == "sklearn":
                    y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                    alpha_pos_mean = 0
                    alpha_neg_mean = 0
                    n_alpha_pos_mean = 0
                    n_alpha_neg_mean = 0
                    for i, svm in enumerate(self.svm.estimators_):
                        # Implementation of the following scikit-learn function in PyTorch:
                        # - pred = svm.decision_function(ntk_u_cpu) 
                        alpha_pos_mask = svm.dual_coef_ > 0
                        alpha_pos = torch.tensor(svm.dual_coef_[alpha_pos_mask], 
                                            dtype=self.dtype, device=self.device)
                        alpha_neg = torch.tensor(svm.dual_coef_[~alpha_pos_mask], 
                                            dtype=self.dtype, device=self.device)
                        b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                        alpha_pos_mask = alpha_pos_mask.reshape(-1)
                        idx_sup_pos = svm.support_[alpha_pos_mask]
                        idx_sup_neg = svm.support_[~alpha_pos_mask]
                        n_alpha_pos_mean += len(idx_sup_pos)
                        n_alpha_neg_mean += len(idx_sup_neg)
                        alpha_pos_mean += alpha_pos.mean()
                        alpha_neg_mean += alpha_neg.mean()
                        #if i == 0:
                        #    print((alpha_pos.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_pos]).sum(dim=1) \
                        #    + (alpha_neg.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_neg]).sum(dim=1))
                        pred = (alpha_pos.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_pos]).sum(dim=1) \
                            + (alpha_neg.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_neg]).sum(dim=1) \
                            + b
                        y_pred[:, i] = pred
                    print(f"n_alpha_pos_mean: {n_alpha_pos_mean / len(self.svm.estimators_)}")
                    print(f"n_alpha_neg_mean: {n_alpha_neg_mean / len(self.svm.estimators_)}")
                    print(f"alpha_pos_mean: {alpha_pos_mean / len(self.svm.estimators_)}")
                    print(f"alpha_neg_mean: {alpha_neg_mean / len(self.svm.estimators_)}")
                elif self.solver == "qplayer_one_vs_all":
                    y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                    alpha_l = self.svm
                    for k, alpha in enumerate(alpha_l):
                        idx_sup = (alpha > self.alpha_tol)
                        y_sup = y_test[idx_labeled][idx_sup]
                        y_mask = y_sup == k
                        y_sup[y_mask] = 1
                        y_sup[~y_mask] = -1
                        y_sup_pos_mask = y_sup > 0
                        y_sup_pos = y_sup[y_sup_pos_mask]
                        y_sup_neg = y_sup[~y_sup_pos_mask]
                        alpha_sup = alpha[idx_sup]
                        alpha_sup_pos = alpha_sup[y_sup_pos_mask]
                        alpha_sup_neg = alpha_sup[~y_sup_pos_mask]
                        bias_tol = max(self.alpha_tol, 1e-10)
                        idx_bias = alpha_sup < (self.regularizer - bias_tol)
                        ntk_sup = ntk_labeled[idx_sup, :]
                        ntk_sup = ntk_sup[:, idx_sup]
                        ntk_unlabeled_sup_ub = ntk_unlabeled_ub[:,idx_sup]
                        ntk_unlabeled_sup_lb = ntk_unlabeled_lb[:,idx_sup]
                        
                        pred = (y_sup_pos * alpha_sup_pos * ntk_unlabeled_sup_ub[:,y_sup_pos_mask]).sum(dim=1) \
                            + (y_sup_neg * alpha_sup_neg * ntk_unlabeled_sup_lb[:,~y_sup_pos_mask]).sum(dim=1)
                        if self.bias:
                            b = y_sup[idx_bias] - (y_sup * alpha_sup * ntk_sup[idx_bias, :]).sum(dim=1)
                            b = b.mean()
                            pred += b
                        y_pred[:, k] = pred
                else:
                    assert False
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
                           method: str="SXXTS",
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
                                                       delta, perturbation_model,
                                                       method)
        self.empty_gpu_memory()
        ntk_labeled = self.ntk 
        if self.idx_trn_labeled is not None: # semi-supervised setting
            ntk_labeled = ntk_labeled[self.idx_trn_labeled, :]
            ntk_labeled = ntk_labeled[:, self.idx_trn_labeled]
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
                if self.solver == "cvxopt" or self.solver == "qplayer":
                    if self.solver == "cvxopt":
                        alpha = torch.tensor(self.svm, dtype=self.dtype, 
                                            device=self.device)
                    else:
                        alpha = self.svm
                    idx_sup = (alpha > self.alpha_tol)
                    y_sup = y_test[idx_labeled][idx_sup] * 2 - 1
                    y_sup_pos_mask = y_sup > 0
                    y_sup_pos = y_sup[y_sup_pos_mask]
                    y_sup_neg = y_sup[~y_sup_pos_mask]
                    alpha_sup = alpha[idx_sup]
                    alpha_sup_pos = alpha_sup[y_sup_pos_mask]
                    alpha_sup_neg = alpha_sup[~y_sup_pos_mask]
                    bias_tol = max(self.alpha_tol, 1e-10)
                    idx_bias = alpha_sup < (self.regularizer - bias_tol)
                    ntk_sup = ntk_labeled[idx_sup, :]
                    ntk_sup = ntk_sup[:, idx_sup]
                    ntk_unlabeled_sup_ub = ntk_unlabeled_ub[:,idx_sup]
                    ntk_unlabeled_sup_lb = ntk_unlabeled_lb[:,idx_sup]
                        
                    y_pred = (y_sup_pos * alpha_sup_pos * ntk_unlabeled_sup_lb[:,y_sup_pos_mask]).sum(dim=1) \
                        + (y_sup_neg * alpha_sup_neg * ntk_unlabeled_sup_ub[:,~y_sup_pos_mask]).sum(dim=1)
                    if self.bias:
                        b = y_sup[idx_bias] - (y_sup * alpha_sup * ntk_sup[idx_bias, :]).sum(dim=1)
                        b = b.mean()
                        y_pred += b
                elif self.solver == 'sklearn':
                    # Implementation of self.svm.decision_function(ntk_unlabeled) in PyTorch
                    svm = self.svm
                    alpha_pos_mask = svm.dual_coef_ > 0
                    alpha_pos = torch.tensor(svm.dual_coef_[alpha_pos_mask], 
                                            dtype=self.dtype, device=self.device)
                    alpha_neg = torch.tensor(svm.dual_coef_[~alpha_pos_mask], 
                                            dtype=self.dtype, device=self.device)
                    b = torch.tensor(svm.intercept_, dtype=self.dtype, device=self.device)
                    alpha_pos_mask = alpha_pos_mask.reshape(-1)
                    idx_sup_pos = svm.support_[alpha_pos_mask]
                    idx_sup_neg = svm.support_[~alpha_pos_mask]
                    #if i == 0:
                    #    print((alpha_pos.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_pos]).sum(dim=1) \
                    #    + (alpha_neg.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_neg]).sum(dim=1))
                    y_pred = (alpha_pos.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_pos]).sum(dim=1) \
                        + (alpha_neg.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_neg]).sum(dim=1) \
                        + b
            else:
                if self.solver == "sklearn":
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
                        alpha_pos_mask = alpha_pos_mask.reshape(-1)
                        idx_sup_pos = svm.support_[alpha_pos_mask]
                        idx_sup_neg = svm.support_[~alpha_pos_mask]
                        #if i == 0:
                        #    print((alpha_pos.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_pos]).sum(dim=1) \
                        #    + (alpha_neg.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_neg]).sum(dim=1))
                        pred = (alpha_pos.reshape(1,-1) * ntk_unlabeled_lb[:,idx_sup_pos]).sum(dim=1) \
                            + (alpha_neg.reshape(1,-1) * ntk_unlabeled_ub[:,idx_sup_neg]).sum(dim=1) \
                            + b
                        y_pred[:, i] = pred
                elif self.solver == "qplayer_one_vs_all":
                    y_pred = torch.zeros((len(idx_test), self.n_classes), device=self.device)
                    alpha_l = self.svm
                    for k, alpha in enumerate(alpha_l):
                        idx_sup = (alpha > self.alpha_tol)
                        y_sup = y_test[idx_labeled][idx_sup]
                        y_mask = y_sup == k
                        y_sup[y_mask] = 1
                        y_sup[~y_mask] = -1
                        y_sup_pos_mask = y_sup > 0
                        y_sup_pos = y_sup[y_sup_pos_mask]
                        y_sup_neg = y_sup[~y_sup_pos_mask]
                        alpha_sup = alpha[idx_sup]
                        alpha_sup_pos = alpha_sup[y_sup_pos_mask]
                        alpha_sup_neg = alpha_sup[~y_sup_pos_mask]
                        bias_tol = max(self.alpha_tol, 1e-10)
                        idx_bias = alpha_sup < (self.regularizer - bias_tol)
                        ntk_sup = ntk_labeled[idx_sup, :]
                        ntk_sup = ntk_sup[:, idx_sup]
                        ntk_unlabeled_sup_ub = ntk_unlabeled_ub[:,idx_sup]
                        ntk_unlabeled_sup_lb = ntk_unlabeled_lb[:,idx_sup]
                            
                        pred = (y_sup_pos * alpha_sup_pos * ntk_unlabeled_sup_lb[:,y_sup_pos_mask]).sum(dim=1) \
                            + (y_sup_neg * alpha_sup_neg * ntk_unlabeled_sup_ub[:,~y_sup_pos_mask]).sum(dim=1)
                        if self.bias:
                            b = y_sup[idx_bias] - (y_sup * alpha_sup * ntk_sup[idx_bias, :]).sum(dim=1)
                            b = b.mean()
                            pred += b
                        y_pred[:, k] = pred
                else:
                    assert False

        if return_ntk:
            return y_pred, ntk_test_lb
        return y_pred