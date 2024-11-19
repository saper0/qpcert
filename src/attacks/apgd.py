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
                 idx_trn: Integer[torch.Tensor, "m"],
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
                 gradient_method: str="exact", #or approximate,
                 surrogate_model_dict: Dict[str, Any]={},
                 lmbda=0.5, # weigh labeled vs self-trned loss
                 evasion_attack=False,
                 verbose=True,
                 **kwarg):
        """Initialization of APGD.
        
        Args: 
            delta (float): 
                Perturbation budget.
            perturbation_model (str): 
                Type of perturbation model.
            X (Float[torch.Tensor, "n n"]): 
                Input feature matrix.
            A (Float[torch.Tensor, "n n"]): 
                Adjacency matrix.
            y (Integer[torch.Tensor, "n"]): 
                Labels.
            idx_trn (Integer[torch.Tensor, "m"]): 
                Indices of training samples.
            idx_labeled (Integer[np.ndarray, "l"]): 
                Indices of labeled samples.
            idx_adv (Integer[np.ndarray, "u"]): 
                Indices nodes w.r.t. to which the gradient to generate X_pert
                is computed.
            model_params (Dict[str, Any]): 
             Model parameters.
            eta (float, optional): 
                Step size. 
            alpha_momentum (float, optional): 
                Momentum factor. Defaults to 0.75.
            rho (float, optional): 
                Factor for updating perturbation. Defaults to 0.75. (From APPGD)
            max_iter (int, optional): 
                Maximum number of iterations for APGD attack. Defaults to 1000.
            n_restarts (int, optional): 
                Number of restarts. Defaults to 0.
            dtype (torch.dtype, optional): 
                Data type for tensors. Defaults to torch.float64.
            normalize_grad (bool, optional): 
                Whether to normalize gradients. Defaults to False.
            gradient_method (str, optional): 
                Method for computing gradients. Defaults to "exact", meaning 
                differentiate through the quadratic program. Setting to 
                "approximate" uses a surrogate model (see MetaAttack) to 
                approximate the gradient.
            surrogate_model_dict (Dict[str, Any], optional): 
                Dictionary for surrogate model parameters used in meta-attack. 
                Defaults to {}. If "model_params" not specified in surrogate 
                model parameters. Surrogate model is a two-layer SGC model 
                with 8 filters and no dropout.
            lmbda (int, optional): 
                Weight for labeled vs self-trained loss if using (surrogate)
                MetaAttack. Defaults to 0.5.
            evasion_attack (bool, optional): 
                Whether the attack should be performed as an 
                evasion attack. Defaults to False.
            **kwarg: Additional keyword arguments.
        """
        self.delta = delta
        self.X = X
        #self.project = Projection(delta, perturbation_model, self.X)
        self.A = A
        self.y = y
        self.n_classes = int(y.max() + 1)
        assert self.n_classes == 2, "Multi-class attack not implemented."
        self.idx_trn = idx_trn
        self.idx_labeled = idx_labeled
        self.idx_adv = idx_adv
        self.model_params = model_params
        self.normalize_grad = bool(normalize_grad)
        if eta is None:
            self.eta = 2*delta
        else:
            self.eta = eta*delta
        self.W = self._get_checkpoints(max_iter)
        self.alpha = alpha_momentum
        self.rho = rho
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.perturbation_model = perturbation_model
        self.project_method = "new"
        self.gradient_method = gradient_method
        self.surrogate_model_dict = surrogate_model_dict
        self.lmbda = lmbda
        self.evasion_attack = evasion_attack
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
            return self.ntk(idx_labeled=self.idx_labeled, idx_test=idx_target,
                            y_test=self.y, X_test=X, A_test=self.A, 
                            learning_setting="inductive")
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

    def _get_meta_loss(self, idx_target, y_bin, logits):
        """Get meta loss as defined in MetaAttack Algorithm 2.
        
        y_bin: Labels in {-1, 1}
        """
        C = self.model_params["regularizer"]
        trn_term = y_bin[self.idx_labeled] * logits[self.idx_labeled]
        self_trn_term = (y_bin[idx_target] * logits[idx_target])[0]
        hinge_loss = self.lmbda * torch.mean(torch.clamp(1 - trn_term, min=0))
        hinge_loss += (1-self.lmbda) * torch.clamp(1 - self_trn_term, min=0)
        hinge_loss = C * hinge_loss
        return hinge_loss
    
    def _eval_model_wo_parameter_grad(self, model, X):
        """Evaluate model without computing gradients w.r.t. model parameters.
        
        Ensures gradient computation w.r.t. model parameters is enabled after
        evaluation.

        Return logits of model.
        """
        model.train()
        for params in model.parameters():
            params.requires_grad_(False)
        logits = model(X, self.A).reshape(-1)
        for params in model.parameters():
            params.requires_grad_(True)
        return logits
    
    def _get_meta_gradient(self, model, idx_target, X, X_adv, y_bin):
        """Calculate meta gradient as defined in MetaAttack Algorithm 2.
        
        Return meta-gradient and associated loss.
        """
        logits = self._eval_model_wo_parameter_grad(model, X)
        meta_loss = self._get_meta_loss(idx_target, y_bin, logits)
        meta_loss.backward()
        gradient = X_adv.grad
        X_adv.grad = None
        return gradient, meta_loss

    def _get_approximate_gradient(self, idx_target: Integer[torch.Tensor, "1 1"], 
                                  X, X_adv):
        """ Get approx. (meta) gradient as defined in MetaAttack Algorithm 2.
        
        X_adv is part of X but has requires_grad=True.

        Returns approx. gradient and associated loss.
        """
        # init surrogate model
        model_params = {}
        if "model_params" in self.surrogate_model_dict:
            model_params = self.surrogate_model_dict["n_filter"]
        else:
            model_params["model"] = "SGC"
            model_params["n_filter"] = 8
            model_params["activation"] = "linear"
        model_params["n_features"] = X.shape[1]
        model_params["n_classes"] = len(np.unique(self.y))
        model_params["bias"] = False
        model = create_model(model_params)
        model = model.to(X.device)
        # train model / get meta gradient
        X = X.float() # better for performance
        y_bin = self.y * 2 - 1
        X_nograd = X.clone().detach().requires_grad_(False)
        n_epochs = 100 # default from MetaAttack
        C = self.model_params["regularizer"]
        if "n_epochs" in self.surrogate_model_dict:
            n_epochs = self.surrogate_model_dict["n_epochs"]
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, 
                                     weight_decay=5e-3)
        meta_gradient, _ = self._get_meta_gradient(model, idx_target, X, X_adv, 
                                                y_bin)
        for epoch in range(n_epochs):
            # Update Surrogate Model
            optimizer.zero_grad()
            logits = model(X_nograd, self.A).reshape(-1)
            trn_term = y_bin[self.idx_labeled] * logits[self.idx_labeled]
            hinge_loss = C * torch.mean(torch.clamp(1 - trn_term, min=0))
            #for params in model.conv2.parameters():
            #    hinge_loss += torch.sum(params**2)
            hinge_loss.backward()
            optimizer.step()
            # eval acc
            logits = torch.sgn(logits)
            acc = torch.mean((logits[self.idx_labeled] == y_bin[self.idx_labeled]).float())
            
            # Update Meta Gradient
            X, X_adv = self._make_differentiable_X(X)
            _meta_gradient, meta_loss = self._get_meta_gradient(model, 
                                                                idx_target, X, 
                                                                X_adv, y_bin)
            meta_gradient += _meta_gradient
        return meta_gradient, meta_loss
    
    def _make_differentiable_X(self, X, idx_diff=None):
        """ Make X differentiable w.r.t. nodes given in idx_diff.
        
        If idx_diff is not provided, X will be made differentiable w.r.t. 
        adversarial nodes (idx_adv). If X is already differentiable, creates a 
        new differentiable X.

        Returns differentiable X and subtensor X_adv corresponding to the
        differentiable (adversarial) nodes (being leafs in the computation graph).
        """
        if idx_diff is None:
            idx_diff = self.idx_adv
        if X.requires_grad == True:
            X = X.detach()
        X_adv = X[idx_diff, :]
        X_adv.requires_grad = True
        cln_mask = torch.ones((X.shape[0],), dtype=torch.bool, device=X.device)
        cln_mask[idx_diff] = False
        X_cln = X[cln_mask, :]
        X_r = torch.cat((X_adv, X_cln), dim=0)
        # Reorder back 
        idx_sort = torch.argsort(self.idx_r)
        X = X_r[idx_sort, :]
        return X, X_adv

    def _gradient(self, idx_target: Integer[torch.Tensor, "1 1"], X):
        """Return the gradient w.r.t. X and prediction w.r.t. X
        
        Returns: Gradient w.r.t. X, Prediction for idx_target
        """
        X, X_adv = self._make_differentiable_X(X)
        y_pred = self._get_logits(idx_target, X)
        
        if self.gradient_method == "exact":
            y_pred.backward()
            gradient = X_adv.grad
        elif self.gradient_method == "approximate":
            assert self.evasion_attack is False, \
                "Approx. gradient not implemented for evasion attack."
            gradient, _ = self._get_approximate_gradient(idx_target, X, X_adv)
        else:
            assert False, f"Gradient method {self.gradient_method} not supported"
    
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
        X_pert = torch.clone(X_start) # x0
        X_pert_prev = torch.clone(X_pert) # x0
        y_pred_l = []
        # Calculate x1 (Line 3)
        gradient, y_pred = self._gradient(idx_target, X_pert)
        y_pred_l.append(y_pred)
        if sgn is None:
            # First prediction assumed to be unperturbed
            sgn = int(y_pred > 0) * (-1) + int(y_pred <= 0) 
        X_pert[self.idx_adv, :] += sgn * self.eta * gradient #x1
        self._project(X_pert)
        # Set f_max & x_max (Line 4/5)
        gradient, y_pred = self._gradient(idx_target, X_pert)
        y_pred_l.append(y_pred)
        if self._loss_has_improved(y_pred_l[1], y_pred_l[0], sgn):
            n_improved = 1
            y_pred_worst = y_pred_l[1]
            X_pert_worst = torch.clone(X_pert)
        else:
            n_improved = 0
            y_pred_worst = y_pred_l[0]
            X_pert_worst = torch.clone(X_pert_prev)
        Z = torch.clone(X_pert)
        idx_checkpoint = 1
        eta = self.eta
        eta_old = None
        y_pred_worst_old = y_pred_l[0] # First checkpoint is 0
        if do_logging:
            logging.info(f"W: {self.W}")
        for i in range(1, self.max_iter):
            # Calculate x_i+1
            self._update_with_momentum(sgn, eta, gradient, Z, X_pert, X_pert_prev)
            # Calculate gradient and prediction at x_i+1
            gradient, y_pred = self._gradient(idx_target, X_pert)
            if self._loss_has_improved(y_pred, y_pred_l[-1], sgn):
                n_improved += 1
            y_pred_l.append(y_pred)
            if self._loss_has_improved(y_pred, y_pred_worst, sgn):
                y_pred_worst = y_pred_l[-1]
                X_pert_worst = torch.clone(X_pert)
            # Update learning rate 
            if (i-1) % self.W[idx_checkpoint] == 0 and i-1 > 0:
                cond1 = n_improved < self.rho * (self.W[idx_checkpoint] - self.W[idx_checkpoint-1])
                cond2 = eta == eta_old and abs(y_pred_worst - y_pred_worst_old) < 1e-4
                eta_old = eta
                y_pred_worst_old = y_pred_worst
                if cond1 or cond2:
                    eta = eta / 2
                    X_pert[self.idx_adv, :] = X_pert_worst[self.idx_adv, :]
                if do_logging:
                    logging.info(f"k {i-1}, W: {self.W[idx_checkpoint]}, "
                                f"y_pred_worst: {y_pred_worst}, eta: {eta}")
                if idx_checkpoint < len(self.W) - 1:
                    idx_checkpoint += 1
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
