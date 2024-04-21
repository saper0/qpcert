from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from jaxtyping import Float, Integer
import numpy as np
import torch

from src.attacks.base_attack import Attack 
from src.attacks.utils import Projection
from src.models.ntk import NTK

class PGD(Attack):
    """
    Projected gradient descent.
    """
    def __init__(self,
                 delta: float,
                 perturbation_model: str,
                 X: Float[torch.Tensor, "n n"], 
                 A: Float[torch.Tensor, "n n"], 
                 y: Integer[torch.Tensor, "n"],
                 idx_labeled: Integer[np.ndarray, "l"],
                 idx_adv: Integer[np.ndarray, "u"],
                 model_params: Dict[str, Any],
                 eta: float=None,
                 max_iter: int=1000,
                 n_restarts: int=1, #not implemented
                 dtype: torch.dtype=torch.float64,
                 **kwarg):
        self.delta = delta
        self.X = X
        self.project = Projection(delta, perturbation_model, self.X)
        self.A = A
        self.y = y
        self.n_classes = int(y.max() + 1)
        assert self.n_classes == 2, "Multi-class attack not implemented."
        self.idx_labeled = idx_labeled
        self.idx_adv = idx_adv
        self.model_params = model_params
        if eta is None:
            self.eta = 2*delta
        else:
            self.eta = eta
        self.max_iter = max_iter
        self.dtype = dtype
        # Prepare graph reordering
        cln_mask = torch.ones((X.shape[0],), dtype=torch.bool, device=X.device)
        cln_mask[self.idx_adv] = False
        self.idx_all = torch.arange(0, X.shape[0], device=X.device)
        idx_not_adv = self.idx_all[cln_mask]
        self.idx_r = torch.cat((self.idx_adv, idx_not_adv))
        
    def _convert_idx(self, idx: Integer[torch.Tensor, "n"]) -> Integer[torch.Tensor, "n"]:
        """Convert idx to index tensor indexing the reordered graph used by APGD.
        
        Convert index, indexing the original graph, to an index tensor, indexing 
        a reordered graph that is composed first of the adversarial nodes and then 
        of the clean nodes. A new index tensor is returned.
        """
        mask = torch.zeros((self.X.shape[0],), dtype=torch.bool, device=self.X.device)
        mask[idx] = True
        mask = mask[self.idx_r]
        return self.idx_all[mask]

    def _get_logits(self, idx_target, X):
        ntk = NTK(self.model_params, X_trn=X, A_trn=self.A, n_classes=self.n_classes, 
                   idx_trn_labeled=self.idx_labeled, y_trn=self.y[self.idx_labeled],
                   learning_setting="transductive",
                   pred_method=self.model_params["pred_method"],
                   regularizer=self.model_params["regularizer"],
                   bias=bool(self.model_params["bias"]),
                   solver=self.model_params["solver"],
                   alpha_tol=self.model_params["alpha_tol"],
                   dtype=self.dtype)
        return ntk(idx_labeled=self.idx_labeled, idx_test=idx_target,
                   y_test=self.y, X_test=X, A_test=self.A)

    def _gradient(self, idx_target: Integer[torch.Tensor, "1 1"], X):
        """Perform prediction for idx_target and return the gradient w.r.t. X.
        
        Returns: Gradient w.r.t. X, Prediction for idx_target
        """
        X_adv = X[self.idx_adv, :]
        X_adv.requires_grad = True
        cln_mask = torch.ones((X.shape[0],), dtype=torch.bool, device=X.device)
        cln_mask[self.idx_adv] = False
        X_cln = X[cln_mask, :]
        X_r = torch.cat((X_adv, X_cln), dim=0)
        # Reorder back 
        idx_sort = torch.argsort(self.idx_r)
        X = X_r[idx_sort, :]
        y_pred = self._get_logits(idx_target, X)
        y_pred.backward()
        return X_adv.grad, y_pred

    def attack(self, idx_target: Integer[torch.Tensor, "1 1"], do_logging=True) \
        -> Tuple[torch.Tensor]:
        # Create reordered node features with differentiable adversarial nodes
        X_pert = torch.clone(self.X)
        for i in range(self.max_iter):
            gradient, y_pred = self._gradient(idx_target, X_pert)
            if i == 0:
                sgn = (torch.sign(y_pred[0]) * (-1)).detach()
            #todo: min/max depending on sign
            X_pert[self.idx_adv, :] += sgn * self.eta * gradient
            self.project(X_pert)
        return X_pert, None