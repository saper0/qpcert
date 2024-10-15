from abc import ABC, abstractmethod
import logging
from math import ceil
from typing import Any, Dict, List, Tuple

from jaxtyping import Float, Integer
import numpy as np
import torch

from src.attacks.base_attack import Attack 
from src.attacks.utils import Projection
from src.models.ntk import NTK

class APGD(Attack):
    """Adaptation of APGD [1] for graphs. 
    
    [1] https://arxiv.org/pdf/2003.01690.pdf
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
                 alpha_momentum: float=0.75,
                 rho: float=0.75,
                 max_iter: int=1000,
                 n_restarts: int=0,
                 dtype: torch.dtype=torch.float64,
                 normalize_grad: bool=False,
                 **kwarg):
        self.delta = delta
        self.X = X
        #self.project = Projection(delta, perturbation_model, self.X)
        self.A = A
        self.y = y
        self.n_classes = int(y.max() + 1)
        assert self.n_classes == 2, "Multi-class attack not implemented."
        self.idx_labeled = idx_labeled
        self.idx_adv = idx_adv
        self.model_params = model_params
        self.normalize_grad = bool(normalize_grad)
        print(f" normalize grad: {self.normalize_grad}")
        if eta is None:
            self.eta = 2*delta
        else:
            self.eta = eta*delta
        self.W = self._get_checkpoints(max_iter)
        print(f"W: {self.W}")
        self.alpha = alpha_momentum
        self.rho = rho
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.perturbation_model = perturbation_model
        self.project_method = "new"
        if self.n_restarts > 0 and perturbation_model != "linf":
            assert False, "Restarts currently only implemented with linf."
        self.dtype = dtype
        # Prepare graph reordering
        cln_mask = torch.ones((X.shape[0],), dtype=torch.bool, device=X.device)
        cln_mask[self.idx_adv] = False
        self.idx_all = torch.arange(0, X.shape[0], device=X.device)
        idx_not_adv = self.idx_all[cln_mask]
        self.idx_r = torch.cat((self.idx_adv, idx_not_adv))

    def _get_checkpoints(self, max_iter: int) -> List[int]:
        """Checkpoints as defined on page 3 (right column) in APGD paper."""
        W = []
        p_old = None
        p = 0
        i = 1
        while p <= 1:
            W.append(ceil(max_iter * p))
            if i == 1:
                p_old = p
                p = 0.22
            else:
                p_new = p + max(p-p_old-0.03, 0.06)
                p_old = p
                p = p_new
            i += 1
        return W

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
                   dtype=self.dtype,
                   print_alphas=False)
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
        gradient = X_adv.grad
        if self.normalize_grad:
            if self.perturbation_model == "linf":
                gradient = torch.sign(gradient)
            elif self.perturbation_model == "l2":
                grad_norm = torch.linalg.vector_norm(gradient, ord=2, dim=1)
                gradient = gradient / grad_norm.view(-1,1)
            else:
                    assert False, f"Perturbation model {self.perturbation_model} not supported."
        return gradient, y_pred[0].detach().cpu().item()
    
    def _loss_has_improved(self, loss_new, loss_old, sgn):
        """If sgn=-1: loss is minimized, if sgn=1: loss is maximized."""
        if sgn == -1:
            return loss_new < loss_old
        if sgn == 1:
            return loss_new > loss_old
        assert False
    
    def _project(self, X_pert):
        """Calculates P(X_pert) where P is a projection onto the feasible domain."""
        if self.perturbation_model == "linf":
            X_lb = self.X - self.delta
            X_ub = self.X + self.delta
            mask = X_pert < X_lb
            X_pert[mask] = X_lb[mask]
            mask = X_pert > X_ub
            X_pert[mask] = X_ub[mask]
        elif self.perturbation_model == "l2":
            diff = X_pert[self.idx_adv, :] - self.X[self.idx_adv, :]
            diff_norm = torch.linalg.vector_norm(diff, ord=2, dim=1)
            mask = diff_norm > self.delta
            idx_violated = self.idx_adv[mask]
            if len(idx_violated) > 0:
                X_pert[idx_violated, :] = self.X[idx_violated, :] + \
                    diff[mask, :] / diff_norm[mask].view(-1, 1) * self.delta
        else:
            assert False

    def _update_with_momentum(self, sgn, eta, gradient, Z, X_pert, X_pert_prev):
        """Perform gradient update with momentum (lines 7-8) of APGD pseudocode."""
        Z[self.idx_adv, :] = X_pert[self.idx_adv, :] + sgn * eta * gradient
        self._project(Z)
        Z[self.idx_adv, :] = X_pert[self.idx_adv, :] \
            + self.alpha * (Z[self.idx_adv, :] - X_pert[self.idx_adv, :]) \
            + (1 - self.alpha) * (X_pert[self.idx_adv, :] - X_pert_prev[self.idx_adv, :])
        self._project(Z)
        X_pert_prev[self.idx_adv, :] = X_pert[self.idx_adv, :]
        X_pert[self.idx_adv, :] = Z[self.idx_adv, :]

    def _attack(self, idx_target, X_start, sgn = None, do_logging=True) \
        -> Tuple[torch.Tensor, List[float]]:
        # Create reordered node features with differentiable adversarial nodes
        X_pert = torch.clone(X_start)
        X_pert_prev = torch.clone(X_pert) 
        y_pred_l = []
        gradient, y_pred = self._gradient(idx_target, X_pert)
        if sgn is None:
            # First prediction assumed to be unperturbed
            sgn = int(y_pred > 0) * (-1) + int(y_pred <= 0) 
        X_pert[self.idx_adv, :] += sgn * self.eta * gradient #x1
        self._project(X_pert)
        y_pred_l.append(y_pred)
        X_pert_worst = torch.clone(X_pert_prev)
        y_pred_worst = y_pred_l[0]
        Z = torch.clone(X_pert)
        Z_old = torch.clone(X_pert)
        idx_checkpoint = 1
        n_improved = 0
        eta = self.eta
        eta_old = None
        y_pred_worst_old = None
        for i in range(1, self.max_iter):
            gradient, y_pred = self._gradient(idx_target, X_pert)
            if self._loss_has_improved(y_pred, y_pred_l[-1], sgn):
                n_improved += 1
            y_pred_l.append(y_pred)
            if self._loss_has_improved(y_pred, y_pred_worst, sgn):
                y_pred_worst = y_pred_l[-1]
                X_pert_worst = torch.clone(X_pert)
            if (i-1) % self.W[idx_checkpoint] == 0 and i-1 > 0:
                cond1 = n_improved < self.rho * (self.W[idx_checkpoint] - self.W[idx_checkpoint-1])
                #cond2 = eta == eta_old and y_pred_worst == y_pred_worst_old
                cond2 = eta == eta_old and abs(y_pred_worst - y_pred_worst_old) < 1e-4
                #cond3 = False
                eta_old = eta
                y_pred_worst_old = y_pred_worst
                if cond1 or cond2:
                    eta = eta / 2
                    X_pert[self.idx_adv, :] = X_pert_worst[self.idx_adv, :]
                print(do_logging)
                if do_logging:
                    logging.info(f"k {i-1}, W: {self.W[idx_checkpoint]}, "
                                f"y_pred_worst: {y_pred_worst}, eta: {eta}")
                if idx_checkpoint < len(self.W) - 1:
                    idx_checkpoint += 1
            # Apply momentum
            self._update_with_momentum(sgn, eta, gradient, Z, X_pert, X_pert_prev)
        # Check value of last change
        gradient, y_pred = self._gradient(idx_target, X_pert)
        y_pred_l.append(y_pred)
        if y_pred_l[-1] * sgn > y_pred_worst * sgn:
            y_pred_worst = y_pred_l[-1]
            X_pert_worst = torch.clone(X_pert)
        return X_pert_worst, y_pred_l, sgn, y_pred_worst

    def attack(self, idx_target: Integer[torch.Tensor, "1 1"], do_logging=True) \
        -> Tuple[torch.Tensor, List[float], float]:
        """ Attack idx_target. 
        
        do_logging=True enables occasionally outputting progress of attack.
        
        Returns:
            - Feature matrix of most successfull attack
            - List of evolution of logits along attack iterations
            - Logits of most successfull attack
        """
        X_pert, y_pert_l, sgn, y_pert = self._attack(idx_target, self.X, 
                                                     do_logging=do_logging)
        if self.n_restarts > 0:
            y_best = y_pert
            X_best = torch.clone(X_pert)
            sgn_orig = sgn
            for restart in range(self.n_restarts):
                X_pert = torch.clone(self.X)
                rand_start = torch.rand(X_pert[self.idx_adv,:].shape, 
                                        dtype=self.dtype, device=self.X.device)
                rand_start *= self.delta
                X_pert[self.idx_adv,:] = self.X[self.idx_adv,:] + rand_start
                X_pert, y_pert_new, sgn, y_pert = self._attack(idx_target, self.X, 
                                                               sgn_orig, do_logging)
                y_pert_l.extend(y_pert_new)
                if self._loss_has_improved(y_pert, y_best, sgn_orig):
                    X_best[self.idx_adv,:] = X_pert[self.idx_adv,:]
                    y_best = y_pert
            return X_best, y_pert_l, y_best
        else:
            return X_pert, y_pert_l, y_pert
