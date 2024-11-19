from abc import ABC, abstractmethod
import logging
from math import ceil
from typing import Any, Dict, List, Tuple

from jaxtyping import Float, Integer
import numpy as np
import torch

from src.attacks.base_attack import Attack 
from src.attacks.utils import Projection
from src.models import create_model
from src.models.ntk import NTK

class CGBA(Attack):
    """Implementation of the clean-label graph backdoor attack (CGBA) from [1]
    
    [1] https://arxiv.org/pdf/2401.00163
    """
    def __init__(self,
                 delta: float,
                 perturbation_model: str,
                 X: Float[torch.Tensor, "n n"], 
                 A: Float[torch.Tensor, "n n"], 
                 y: Integer[torch.Tensor, "n"],
                 idx_trn: Integer[torch.Tensor, "m"],
                 idx_labeled: Integer[np.ndarray, "l"],
                 idx_adv: Integer[np.ndarray, "u"],
                 model_params: Dict[str, Any],
                 dtype: torch.dtype=torch.float64,
                 trigger_size: float=0.1,
                 evasion_attack=False,
                 **kwarg):
        self.delta = delta
        self.X = X
        self.A = A
        self.y = y
        self.n_classes = int(y.max() + 1)
        assert self.n_classes == 2, "Multi-class attack not implemented."
        self.idx_trn = idx_trn
        self.idx_labeled = idx_labeled
        self.idx_adv = idx_adv
        self.model_params = model_params
        self.perturbation_model = perturbation_model
        self.evasion_attack = evasion_attack
        self.dtype = dtype
        self.trigger_size = trigger_size

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

    def _get_logits(self, idx_target, X):
        """Get logits for idx_target in a differentiable way.
        
        If evasion_attack=True, only NTK forward pass is differentiable, not the
        NTK creation and SVM training.
        """
        if self.evasion_attack:
            if not hasattr(self, "ntk"):
                self.ntk = NTK(self.model_params, X_trn=self.X, A_trn=self.A, 
                               n_classes=self.n_classes, 
                               idx_trn_labeled=self.idx_labeled, 
                               y_trn=self.y[self.idx_labeled],
                               learning_setting="transductive",
                               pred_method=self.model_params["pred_method"],
                               regularizer=self.model_params["regularizer"],
                               bias=bool(self.model_params["bias"]),
                               solver=self.model_params["solver"],
                               alpha_tol=self.model_params["alpha_tol"],
                               dtype=self.dtype,
                               print_alphas=False)
            ntk = self.ntk
        else:
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

    def attack(self, idx_target: Integer[torch.Tensor, "1 1"], do_logging=False) \
        -> Tuple[torch.Tensor, List[float], float]:
        """ Attack idx_target. 
        
        Returns:
            - Feature matrix of most successfull attack
            - List of evolution of logits along attack iterations
            - Logits of most successfull attack
        """
        #print(self.delta)
        X_pert = self.X.clone()
        d = X_pert.shape[1]
        # Get largest degree node
        y_bin = self.y * 2 - 1
        y_target = self.y[idx_target]
        mask_other_cls = self.y != y_target
        idx_max_deg = self.A.sum(dim=1)[mask_other_cls].argmax()
        # Get largest elements of largest degree node
        trigger_size = ceil(self.trigger_size * d)
        X_pert_cls = X_pert[mask_other_cls,:]
        max_node_features = X_pert_cls[idx_max_deg,:]
        if y_bin[mask_other_cls][idx_max_deg] == 1:
            topk_values, topk_indices = torch.topk(max_node_features, trigger_size)
        else:
            topk_values, topk_indices = torch.topk(max_node_features, trigger_size, largest=False)
        # Apply triggers
        if not self.evasion_attack:
            # Poison dataset
            mask_idx_adv = torch.zeros(X_pert.shape[0], dtype=torch.bool)
            mask_idx_adv.scatter_(0, self.idx_adv, True)
            mask_adv_and_other_cls = torch.logical_and(mask_idx_adv, mask_other_cls)
            X_pert_ = X_pert[mask_adv_and_other_cls, :]
            X_pert_[:, topk_indices] = topk_values
            X_pert[mask_adv_and_other_cls, :] = X_pert_
        else:
            # Evasion attack
            X_pert[idx_target, topk_indices] = topk_values
        self._project(X_pert)
        #if self.evasion_attack:
        #    print(X_pert[idx_target,:])
        #    assert False
        #else:
        #    print(topk_values)
        # Evaluate attack
        y_pred = self._get_logits(idx_target, X_pert).detach().cpu().item()
        return X_pert, [y_pred], y_pred
