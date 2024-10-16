import logging
from typing import Tuple, Union, Sequence
import os
import time

from jaxtyping import Float, Integer
import numpy as np
import torch
from torch_sparse import coalesce
from src import globals

import gurobipy as gp
from gurobipy import GRB

MILP_INT_FEAS_TOL = 1e-4
MILP_OPTIMALITY_TOL = 1e-4
MILP_FEASIBILITY_TOL = 1e-4 
MILP_NODELIMIT = 5*1e6


def empty_gpu_memory(device: Union[str, torch.device]):
    if torch.cuda.is_available() and device.type != "cpu":
        torch.cuda.empty_cache()


def certify_robust(
        y_pred: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_ub: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_lb: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]]
    ) -> float:
    if len(y_pred.shape) > 1:
        n = y_pred.shape[0]
        pred_orig = y_pred.argmax(1)
        pred_orig_lb = y_lb[range(n), pred_orig]
        mask = torch.ones(y_ub.shape, dtype=torch.bool).to(y_ub.device)
        mask[range(n), pred_orig] = False
        pred_other_ub = y_ub[mask].reshape((n, y_ub.shape[1]-1))
        count_beaten = (pred_other_ub > pred_orig_lb.reshape(-1, 1)).sum(dim=1)
        return ((count_beaten == 0).sum() / n).cpu().item()
    else:
        mask_neg = y_pred < 0
        n_cert = (y_ub[mask_neg] < 0).sum()
        mask_pos = y_pred >= 0
        n_cert += (y_lb[mask_pos] >= 0).sum()
        return (n_cert / y_pred.shape[0]).cpu().item()


def certify_unrobust(
        y_pred: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_ub: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_lb: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]]
    ) -> float:
    if len(y_pred.shape) > 1:
        n = y_pred.shape[0]
        pred_orig = y_pred.argmax(1)
        pred_orig_ub = y_ub[range(n), pred_orig]
        mask = torch.ones(y_lb.shape, dtype=torch.bool).to(y_lb.device)
        mask[range(n), pred_orig] = False
        pred_other_lb = y_lb[mask].reshape((n, y_lb.shape[1]-1))
        count_beaten = (pred_other_lb > pred_orig_ub.reshape(-1, 1)).sum(dim=1)
        return ((count_beaten > 0).sum() / n).cpu().item()
    else:
        mask_neg = y_pred < 0
        n_cert = (y_lb[mask_neg] >= 0).sum()
        mask_pos = y_pred >= 0
        n_cert += (y_ub[mask_pos] < 0).sum()
        return (n_cert / y_pred.shape[0]).cpu().item()


def accuracy(logits: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
             labels: Integer[torch.Tensor, "n"], 
             idx_labels: np.ndarray = None) -> float:
    """Returns the accuracy for a tensor of logits, a list of lables and and a split indices.

    Works for binary and multi-class classification.

    Returns
    -------
    float
        the Accuracy
    """
    if len(logits.shape) > 1:
        if idx_labels is not None:
            return (logits.argmax(1) == labels[idx_labels]).float().mean().cpu().item()
        else:
            return (logits.argmax(1) == labels).float().mean().cpu().item()
    else:
        logits = logits.reshape(-1,)
        p_cls1 = torch.sigmoid(logits)
        y_pred = (p_cls1 > 0.5).to(dtype=torch.long)
        if idx_labels is not None:
            return ((y_pred == labels[idx_labels]).sum() / len(idx_labels)).cpu().item()
        else:
            return ((y_pred == labels).sum() / len(labels)).cpu().item()


def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    """Duplicates indices in edge_index but with flipped row/col indices. 
    Furthermore, edge_weights are also duplicated without change.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: symmetric_edge_index, symmetric_edge_weight
    """
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    # Remove duplicate values in symmetric_edge_index
    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def grad_with_checkpoint(outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)

    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()

    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def _set_big_M(M_u, M_v, y_mask, C, ntk_labeled_lb, ntk_labeled_ub) -> None:
    # Set big M for nodes i with y_i = 1 (or -1 if ~y_mask given)
    y_pos_rows_mask = np.zeros(ntk_labeled_lb.shape, dtype=bool)
    y_pos_rows_mask[y_mask] = True
    y_pos_cols_mask = np.transpose(y_pos_rows_mask)
    # y=1 nodes connected to y=1 nodes (or y=-1 nodes connected to y=-1 if ~y_mask)
    y_pos_mask = y_pos_rows_mask & y_pos_cols_mask
    # y=1 nodes connected to y=-1 nodes (or y=-1 nodes connected to y=1 if ~y_mask)
    y_neg_mask = y_pos_mask.copy()
    y_neg_mask[y_mask] = ~y_neg_mask[y_mask]
    ub_pos_mask = ntk_labeled_ub > 0
    lb_neg_mask = ntk_labeled_lb < 0
    # Zero negative Q_ij^U for y_j=1 (or y_j=-1 if ~y_mask)
    mask = y_pos_mask & ub_pos_mask
    ntk_labeled_ub_pos = np.copy(ntk_labeled_ub)
    ntk_labeled_ub_pos[~mask] = 0
    # Zero positive Q_ij^L for y_j=-1 (or y_j=1 if ~y_mask)
    mask = (~y_pos_mask) & lb_neg_mask
    ntk_labeled_lb_neg = np.copy(ntk_labeled_lb)
    ntk_labeled_lb_neg[~mask] = 0
    #assert (ntk_labeled_ub_pos < 0).sum() == 0
    #assert (ntk_labeled_lb_neg < 0).sum() == 0
    M_u[y_mask] = ntk_labeled_ub_pos[y_mask].sum(axis=1) * C \
                - ntk_labeled_lb_neg[y_mask].sum(axis=1) * C \
                - 1
    M_u[M_u < 0] = 0
    # Zero negative Q_ij^U for y_j=-1 (or y_j=1 if ~y_mask)
    mask = (~y_pos_mask) & ub_pos_mask
    ntk_labeled_ub_pos_yneg = np.copy(ntk_labeled_ub)
    ntk_labeled_ub_pos_yneg[~mask] = 0
    # Zero positive Q_ij^L for y_j=1 (or y_j=-1 if ~y_mask)
    mask = y_pos_mask & lb_neg_mask
    ntk_labeled_lb_neg_ypos = np.copy(ntk_labeled_lb)
    ntk_labeled_lb_neg_ypos[~mask] = 0
    M_v[y_mask] = -ntk_labeled_lb_neg_ypos[y_mask].sum(axis=1)*C \
                + ntk_labeled_ub_pos_yneg[y_mask].sum(axis=1)*C \
                + 1
    

def certify_one_vs_all_milp(idx_labeled, idx_test, ntk, ntk_lb, ntk_ub, y, 
                            y_pred, svm_alpha, certificate_params,
                            C=1, M=1e4, Mprime=1e4, 
                            milp=True):
    assert (ntk_lb > ntk_ub).sum() == 0
    assert (ntk_lb > ntk).sum() == 0
    assert (ntk_ub < ntk).sum() == 0

    n_labeled = idx_labeled.shape[0]
    ntk = ntk.detach().cpu().numpy()
    ntk_labeled = ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    ntk_ub = ntk_ub.detach().cpu().numpy()
    ntk_labeled_ub = ntk_ub[idx_labeled, :]
    ntk_labeled_ub = ntk_labeled_ub[:, idx_labeled]
    ntk_lb = ntk_lb.detach().cpu().numpy()
    ntk_labeled_lb = ntk_lb[idx_labeled, :]
    ntk_labeled_lb = ntk_labeled_lb[:, idx_labeled]
    y_labeled = y[idx_labeled].detach().cpu().numpy()
    if len(idx_test) == 1:
        ntk_unlabeled = np.reshape(ntk[idx_test,:][idx_labeled], (1,-1))
        ntk_unlabeled_ub = np.reshape(ntk_ub[idx_test,:][idx_labeled], (1,-1))
        ntk_unlabeled_lb = np.reshape(ntk_lb[idx_test,:][idx_labeled], (1,-1))
    else:
        ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]
        ntk_unlabeled_ub = ntk_ub[idx_test,:][:,idx_labeled]
        ntk_unlabeled_lb = ntk_lb[idx_test,:][:,idx_labeled]

    # Find the initial start feasible solution
    u_start_d = {}
    v_start_d = {}
    s_start_d = {}
    t_start_d = {}
    alpha_l = []
    for k, alpha in enumerate(svm_alpha):
        y_labeled_ = np.copy(y_labeled)
        y_mask = y_labeled_ == k
        y_labeled_[y_mask] = 1
        y_labeled_[~y_mask] = -1
        alpha = alpha.numpy(force=True)
        alpha_mask = alpha < globals.zero_tol
        alpha[alpha_mask] = 0
        alpha_nz_mask = alpha>0 
        eq_constraint = y_labeled_*((ntk_labeled * alpha)@y_labeled_) - 1
        eq_constraint[np.abs(eq_constraint) < MILP_INT_FEAS_TOL] = 0
        u_start = np.zeros(alpha.shape[0], dtype=np.float64)
        v_start = np.zeros(alpha.shape[0], dtype=np.float64)
        u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
        v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
        #u_start[u_start<globals.zero_tol] = 0
        #v_start[v_start<globals.zero_tol] = 0
        s_start = np.zeros(alpha.shape[0], dtype=np.int64)
        u_nz_mask = u_start > globals.zero_tol 
        s_start[u_nz_mask] = 1
        t_start = np.zeros(alpha.shape[0], dtype=np.int64)
        v_nz_mask = v_start > globals.zero_tol 
        t_start[v_nz_mask] = 1
        u_start_d[k] = u_start
        v_start_d[k] = v_start
        s_start_d[k] = s_start
        t_start_d[k] = t_start
        alpha_l.append(alpha)
        assert (alpha<0).sum() == 0
        assert (u_start<-globals.zero_tol).sum() == 0
        assert (v_start<-globals.zero_tol).sum() == 0
        assert ((y_labeled_*((ntk_labeled * alpha)@y_labeled_) - 1 - u_start + v_start) > MILP_FEASIBILITY_TOL).sum() == 0
        assert (u_start-M*s_start > MILP_FEASIBILITY_TOL).sum() == 0
        assert (alpha-C*(1-s_start) > MILP_FEASIBILITY_TOL).sum() == 0
        assert (v_start-Mprime*t_start > MILP_FEASIBILITY_TOL).sum() == 0
        assert (-alpha+C*t_start > MILP_FEASIBILITY_TOL).sum() == 0

    
    is_robust_l = []
    robust_count = 0
    n_classes = y_pred.shape[1]
    for i in range(idx_test.shape[0]):
        idx = idx_test[i]
        # Start with correct prediction & gradually work through others
        pred_ordered = y_pred[i,:].topk(n_classes).indices
        is_robust = True
        stop_obj = None
        for j, k in enumerate(pred_ordered):
            k = k.cpu().item()
            y_labeled_ = np.copy(y_labeled)
            y_mask = y_labeled_ == k
            y_labeled_[y_mask] = 1
            y_labeled_[~y_mask] = -1
            if j == 0:
                obj_min = True
            else:
                obj_min = False
            try:
                # Create a new model
                m = gp.Model("milp_provable_robustness")

                # Create variables
                z_bound = np.minimum(0.0, C*ntk_lb.min())
                alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
                u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
                v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
                # todo: why no ub?
                z = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z")
                z_test = m.addMVar(shape=(1, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z_test")
                if milp:
                    s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
                    t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

                # Add constraints
                m.addConstr(z <= ntk_labeled_ub * alpha, "z_ub")
                m.addConstr(z >= ntk_labeled_lb * alpha, "z_lb")
                m.addConstr(z_test <= ntk_unlabeled_ub[i,:].reshape(1,-1) * alpha, "z_test_ub")
                m.addConstr(z_test >= ntk_unlabeled_lb[i,:].reshape(1,-1) * alpha, "z_test_lb")
                m.addConstr(y_labeled_*(z@y_labeled_) - u + v == 1, "eq_constraint")
                if milp:    
                    M_u = np.zeros((y_mask.shape[0],))
                    M_v = np.zeros((y_mask.shape[0],))
                    # Set big M for nodes i with y_i = 1
                    _set_big_M(M_u, M_v, y_mask, C, ntk_labeled_lb, ntk_labeled_ub)
                    # Set big M for nodes i with y_i = -1
                    _set_big_M(M_u, M_v, ~y_mask, C, ntk_labeled_lb, ntk_labeled_ub)
                    assert (u_start_d[k] > M_u).sum() == 0
                    assert (v_start_d[k] > M_v).sum() == 0
                    m.addConstr(u <= M_u*s, "u_mil1")
                    m.addConstr(alpha <= C*(1-s), "u_mil2")
                    m.addConstr(v <= M_v*t, "v_mil1")
                    m.addConstr(C-alpha <= C*(1-t), "v_mil2")
                else:
                    m.addConstr(u*alpha == 0, "u_comp_slack")
                    m.addConstr(v*(C-alpha) == 0, "v_comp_slack")

                # Set the initial values for the parameters
                alpha.Start = alpha_l[k]
                z.Start = ntk_labeled * alpha_l[k]
                z_test.Start = ntk_unlabeled[i,:].reshape(1,-1) * alpha_l[k]
                u.Start = u_start_d[k]
                v.Start = v_start_d[k]
                if milp:
                    s.Start = s_start_d[k]
                    t.Start = t_start_d[k]

                # Set objective
                if obj_min:
                    m.setObjective(z_test @ y_labeled_, GRB.MINIMIZE)
                else:
                    m.setObjective(z_test @ y_labeled_, GRB.MAXIMIZE)

                if stop_obj is not None:
                    m.Params.BestObjStop = stop_obj # terminate when the objective reaches 0, implies node not robust
                    m.Params.BestBdStop = stop_obj # if the bound falls below the stop_obj, node definitely can't change prediction
                else:
                    assert obj_min == True
                    secnd_best_ypred = torch.topk(y_pred, 2).values[0][-1]
                    m.Params.BestObjStop = secnd_best_ypred.detach().cpu().item()
                m.Params.IntegralityFocus = 1 # to stabilize big-M constraint (must)
                m.Params.IntFeasTol = MILP_INT_FEAS_TOL # to stabilize big-M constraint (helps, works without this also) 
                if "LogToConsole" in certificate_params:
                    m.Params.LogToConsole = certificate_params["LogToConsole"]
                else:
                    m.Params.LogToConsole = 0 # to suppress the logging in console - for better readability
                if "OutputFlag" in certificate_params:
                    m.Params.LogToConsole = certificate_params["OutputFlag"]
                else:
                    m.params.OutputFlag= 0 # to suppress branch bound search tree outputs
                m.Params.DualReductions = 0 # to know whether the model is infeasible or unbounded                
            
                # Played around with the following flags to escape infeasibility solutions
                m.Params.FeasibilityTol = MILP_FEASIBILITY_TOL
                if stop_obj is None and "MIPGap" in certificate_params:
                    m.Params.MIPGap = certificate_params["MIPGap"]
                m.Params.OptimalityTol = MILP_OPTIMALITY_TOL
                if "NumericFocus" in certificate_params:
                    m.Params.NumericFocus = certificate_params["NumericFocus"]
                else:
                    m.Params.NumericFocus = 3
                if "TimeLimit" in certificate_params:
                    m.Params.TimeLimit = certificate_params["TimeLimit"]
                # m.Params.MIPGap = 1e-4
                # m.Params.MIPGapAbs = 1e-4
                # m.Params.Presolve = 0
                # m.Params.Aggregate = 0 #aggregation level in presolve
                if "MIPFocus" in certificate_params:
                    m.Params.MIPFocus = certificate_params["MIPFocus"]
                elif "MIPFocus_first" in certificate_params:
                    if stop_obj is None:
                        m.Params.MIPFocus = certificate_params["MIPFocus_first"]
                    else:
                        m.Params.MIPFocus = certificate_params["MIPFocus_other"]
                else:
                    if stop_obj is None:
                        m.Params.MIPFocus = 2
                    else:
                        m.Params.MIPFocus = 3
                if "Cuts" in certificate_params:
                    m.Params.Cuts = certificate_params["Cuts"]
                if "Heuristics" in certificate_params:
                    m.Params.Heuristics = certificate_params["Heuristics"]
                if "Presolve" in certificate_params:
                    m.Params.Presolve = certificate_params["Presolve"]
                if "Threads" in certificate_params:
                    m.Params.Threads = certificate_params["Threads"]
                # m.Params.InfProofCuts = 0

                def callback(model, where):
                    if where == gp.GRB.Callback.MESSAGE:
                        msg = model.cbGet(gp.GRB.Callback.MSG_STRING)
                        if "MIP start" in msg:
                            print(msg)

                if globals.debug:
                    m.write('milp_optimization.lp') # helps in checking if the implemented model is correct
                    m.optimize(callback)
                    print('Optimization status ', m.Status)
                else:
                    m.optimize(callback)
                    logging.info(f"Optimization status: {m.Status}")

                if m.Status == GRB.INFEASIBLE:
                    # WARNING: not a good sign to be here as our model will have feasible region for sure.
                    # Good to debug by closely looking at the violated constraints using the below code
                    # Would relaxing the constraint help as a last resort? try m.feasRelaxS(2, True, False, True) and then m.optimize()
                    # Check the arguments in feasRelaxS
                    assert False, "Time to debug the gurobi optimization!!"

                    # do IIS if the model is infeasible
                    m.computeIIS()

                    # Print out the IIS constraints and variables
                    print('\nThe following constraints and variables are in the IIS:')
                    for c in m.getConstrs():
                        if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

                    for v in m.getVars():
                        if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                        if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
                elif (m.Status == GRB.OPTIMAL or m.Status == GRB.USER_OBJ_LIMIT) and globals.debug:
                    print(f'Original {y_pred[idx].item()}, Opt objective {m.ObjVal}')
                    # Debugging 
                    # for var in m.getVars():
                    #     print('%s %g' % (var.VarName, var.X))
                    #
                    # print('z ', z.X)
                    # print('v ', v.X)
                    # print('u ', u.X)
                    # print('equality constraint ', y_labeled*(z.X@y_labeled) - u.X + v.X)

                # log results
                logging.info(f'Original {y_pred[i, k].item():.5f}, Opt objective '
                            f'{m.ObjVal:.5f}, Opt bound: {m.ObjBound:.5f}, stop_obj: {stop_obj}')
                # analyse result
                if stop_obj is None:
                    if m.ObjVal < secnd_best_ypred:
                        is_robust = False
                        m.dispose()
                        break
                    if m.Status == GRB.OPTIMAL:
                        stop_obj = m.ObjVal
                    else:
                        stop_obj = m.ObjBound
                else:
                    if m.Status == GRB.OPTIMAL:
                        if m.ObjVal > stop_obj:
                            is_robust = False
                            m.dispose()
                            break
                    else:
                        if m.ObjBound > stop_obj:
                            is_robust = False
                            m.dispose()
                            break
                m.dispose()

            except gp.GurobiError as e:
                logging.error(f"Error code {e.errno}: {e}")
                return
            except AttributeError:
                logging.error("Encountered an attribute error")
                return
        is_robust_l.append(is_robust)
        if is_robust:
            robust_count += 1
        logging.info(f'Robust count {robust_count} out of {i+1}')
    return is_robust_l


def certify_robust_bilevel_svm(idx_labeled, idx_test, ntk, ntk_lb, ntk_ub, y, 
                               y_pred, svm_alpha, certificate_params,
                               C=1, M=1e4, Mprime=1e4, milp=True):
    """TODO: Create documentation 
    """
    if isinstance(svm_alpha, torch.Tensor):
        svm_alpha = svm_alpha.numpy(force=True)
    
    assert (ntk_lb > ntk_ub).sum() == 0
    assert (ntk_lb > ntk).sum() == 0
    assert (ntk_ub < ntk).sum() == 0

    n_labeled = idx_labeled.shape[0]
    ntk = ntk.detach().cpu().numpy()
    ntk_labeled = ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    ntk_ub = ntk_ub.detach().cpu().numpy()
    ntk_labeled_ub = ntk_ub[idx_labeled, :]
    ntk_labeled_ub = ntk_labeled_ub[:, idx_labeled]
    ntk_lb = ntk_lb.detach().cpu().numpy()
    ntk_labeled_lb = ntk_lb[idx_labeled, :]
    ntk_labeled_lb = ntk_labeled_lb[:, idx_labeled]
    y_labeled = y[idx_labeled].detach().cpu().numpy()
    if len(idx_test) == 1:
        ntk_unlabeled = np.reshape(ntk[idx_test,:][idx_labeled], (1,-1))
        ntk_unlabeled_ub = np.reshape(ntk_ub[idx_test,:][idx_labeled], (1,-1))
        ntk_unlabeled_lb = np.reshape(ntk_lb[idx_test,:][idx_labeled], (1,-1))
    else:
        ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]
        ntk_unlabeled_ub = ntk_ub[idx_test,:][:,idx_labeled]
        ntk_unlabeled_lb = ntk_lb[idx_test,:][:,idx_labeled]

    # Labels are learned as -1 or 1, but loaded as 0 or 1
    y_labeled = y_labeled*2 -1 
    y_mask = y_labeled == 1
    
    # Find the initial start feasible solution
    alpha_mask = svm_alpha < globals.zero_tol
    svm_alpha[alpha_mask] = 0
    alpha_nz_mask = svm_alpha>0 
    eq_constraint = y_labeled*((ntk_labeled * svm_alpha)@y_labeled) - 1
    eq_constraint[np.abs(eq_constraint) < MILP_FEASIBILITY_TOL] = 0 # Added bec. nec. for real data, remove if results in problems
    u_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    v_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
    v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
    #u_start[u_start<globals.zero_tol] = 0
    #v_start[v_start<globals.zero_tol] = 0
    s_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    u_nz_mask = u_start > globals.zero_tol
    s_start[u_nz_mask] = 1
    t_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    v_nz_mask = v_start > 0
    t_start[v_nz_mask] = 1
    assert (svm_alpha<0).sum() == 0
    assert (u_start<-globals.zero_tol).sum() == 0
    assert (v_start<-globals.zero_tol).sum() == 0
    assert ((y_labeled*((ntk_labeled * svm_alpha)@y_labeled) - 1 - u_start + v_start).sum() < MILP_FEASIBILITY_TOL)
    assert (u_start-M*s_start > MILP_FEASIBILITY_TOL).sum() == 0
    assert (svm_alpha-C*(1-s_start) > MILP_FEASIBILITY_TOL).sum() == 0
    assert (v_start-Mprime*t_start > MILP_FEASIBILITY_TOL).sum() == 0
    assert (-svm_alpha+C*t_start > MILP_FEASIBILITY_TOL).sum() == 0

    obj_l = []
    obj_bd_l = []
    is_robust_l = []
    opt_status_l = []
    obj_min = None
    robust_count = 0
    for idx in range(y_pred.shape[0]):
        if y_pred[idx] < 0:
            obj_min = False
        else:
            obj_min = True
        try:
            # Create a new model
            m = gp.Model("milp_provable_robustness")

            # Create variables
            z_bound = np.minimum(0.0, C*ntk_lb.min())
            alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
            u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
            v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
            z = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z")
            z_test = m.addMVar(shape=(1, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z_test")
            if milp:
                s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
                t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

            # Add constraints
            m.addConstr(z <= ntk_labeled_ub * alpha, "z_ub")
            m.addConstr(z >= ntk_labeled_lb * alpha, "z_lb")
            m.addConstr(z_test <= ntk_unlabeled_ub[idx,:].reshape(1,-1) * alpha, "z_test_ub")
            m.addConstr(z_test >= ntk_unlabeled_lb[idx,:].reshape(1,-1) * alpha, "z_test_lb")
            m.addConstr(y_labeled*(z@y_labeled) - u + v == 1, "eq_constraint")
            if milp:
                M_u = np.zeros((y_mask.shape[0],))
                M_v = np.zeros((y_mask.shape[0],))
                # Set big M for nodes i with y_i = 1
                _set_big_M(M_u, M_v, y_mask, C, ntk_labeled_lb, ntk_labeled_ub)
                # Set big M for nodes i with y_i = -1
                _set_big_M(M_u, M_v, ~y_mask, C, ntk_labeled_lb, ntk_labeled_ub)
                assert (u_start > M_u).sum() == 0
                assert (v_start > M_v).sum() == 0
                #m.addConstr(u <= M*s, "u_mil1")
                m.addConstr(u <= M_u*s, "u_mil1")
                m.addConstr(alpha <= C*(1-s), "u_mil2")
                #m.addConstr(v <= Mprime*t, "v_mil1")
                m.addConstr(v <= M_v*t, "v_mil1")
                m.addConstr(C-alpha <= C*(1-t), "v_mil2")
            else:
                m.addConstr(u*alpha == 0, "u_comp_slack")
                m.addConstr(v*(C-alpha) == 0, "v_comp_slack")

            # Set the initial values for the parameters
            alpha.Start = svm_alpha
            z.Start = ntk_labeled * svm_alpha
            z_test.Start = ntk_unlabeled[idx,:].reshape(1,-1) * svm_alpha
            u.Start = u_start
            v.Start = v_start
            if milp:
                s.Start = s_start
                t.Start = t_start

            # Set objective
            if obj_min:
                m.setObjective(z_test @ y_labeled, GRB.MINIMIZE)
            else:
                m.setObjective(z_test @ y_labeled, GRB.MAXIMIZE)


            if "MIPGap" in certificate_params:
                m.Params.MIPGap = certificate_params["MIPGap"]
            if "NumericFocus" in certificate_params:
                m.Params.NumericFocus = certificate_params["NumericFocus"]
            else:
                m.Params.NumericFocus = 0
            if "TimeLimit" in certificate_params:
                m.Params.TimeLimit = certificate_params["TimeLimit"]
            # m.Params.MIPGap = 1e-4
            # m.Params.MIPGapAbs = 1e-4
            # m.Params.Presolve = 0
            # m.Params.Aggregate = 0 #aggregation level in presolve
            if "MIPFocus" in certificate_params:
                m.Params.MIPFocus = certificate_params["MIPFocus"]
            else:
                m.Params.MIPFocus = 0
            if "Cuts" in certificate_params:
                m.Params.Cuts = certificate_params["Cuts"]
            if "Heuristics" in certificate_params:
                m.Params.Heuristics = certificate_params["Heuristics"]
            if "Threads" in certificate_params:
                m.Params.Threads = certificate_params["Threads"]

            if obj_min:
                m.Params.BestBdStop = MILP_OPTIMALITY_TOL + 1e-16
                m.Params.BestObjStop = -MILP_OPTIMALITY_TOL 
            else:
                m.Params.BestBdStop = -MILP_OPTIMALITY_TOL - 1e-16
                m.Params.BestObjStop = MILP_OPTIMALITY_TOL

            m.Params.IntegralityFocus = 1 # to stabilize big-M constraint (must)
            m.Params.IntFeasTol = MILP_INT_FEAS_TOL # to stabilize big-M constraint (helps, works without this also) 
            if "LogToConsole" in certificate_params:
                m.Params.LogToConsole = certificate_params["LogToConsole"]
            else:
                m.Params.LogToConsole = 0 # to suppress the logging in console - for better readability
            if "OutputFlag" in certificate_params:
                m.Params.LogToConsole = certificate_params["OutputFlag"]
            else:
                m.params.OutputFlag= 0 # to suppress branch bound search tree outputs
            m.Params.DualReductions = 0 # to know whether the model is infeasible or unbounded                

            # Played around with the following flags to escape infeasibility solutions
            m.Params.FeasibilityTol = MILP_FEASIBILITY_TOL
            m.Params.OptimalityTol = MILP_OPTIMALITY_TOL
            # m.Params.MIPGap = 1e-4
            # m.Params.MIPGapAbs = 1e-4
            # m.Params.Presolve = 0
            # m.Params.Aggregate = 0 #aggregation level in presolve
            # m.Params.MIPFocus = 1
            # m.Params.InfProofCuts = 0

            def callback(model, where):
                if where == gp.GRB.Callback.MESSAGE:
                    msg = model.cbGet(gp.GRB.Callback.MSG_STRING)
                    if "MIP start" in msg:
                        print(msg)

            if globals.debug:
                m.write('milp_optimization.lp') # helps in checking if the implemented model is correct
                m.optimize(callback)
                print('Optimization status ', m.Status)
            else:
                m.optimize(callback)
                logging.info(f"Optimization status: {m.Status}")

            if m.Status == GRB.INFEASIBLE:
                # WARNING: not a good sign to be here as our model will have feasible region for sure.
                # Good to debug by closely looking at the violated constraints using the below code
                # Would relaxing the constraint help as a last resort? try m.feasRelaxS(2, True, False, True) and then m.optimize()
                # Check the arguments in feasRelaxS
                assert False, "Time to debug the gurobi optimization!!"

                # do IIS if the model is infeasible
                m.computeIIS()

                # Print out the IIS constraints and variables
                print('\nThe following constraints and variables are in the IIS:')
                for c in m.getConstrs():
                    if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

                for v in m.getVars():
                    if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                    if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')
            elif (m.Status == GRB.OPTIMAL or m.Status == GRB.USER_OBJ_LIMIT) and globals.debug:
                print(f'Original {y_pred[idx].item()}, Opt objective {m.ObjVal}')
                # Debugging 
                # for var in m.getVars():
                #     print('%s %g' % (var.VarName, var.X))
                #
                # print('z ', z.X)
                # print('v ', v.X)
                # print('u ', u.X)
                # print('equality constraint ', y_labeled*(z.X@y_labeled) - u.X + v.X)

            # log results
            logging.info(f'Original {y_pred[idx].item():.5f}, Opt objective '
                         f'{m.ObjVal:.5f}')
            if m.Status == GRB.TIME_LIMIT:
                is_robust_l.append(False)
            elif obj_min and m.ObjVal > MILP_OPTIMALITY_TOL:
                robust_count += 1
                is_robust_l.append(True)
            elif not obj_min and m.ObjVal < -MILP_OPTIMALITY_TOL:
                robust_count += 1
                is_robust_l.append(True)
            else:
                is_robust_l.append(False)
            obj_l.append(m.ObjVal)
            obj_bd_l.append(m.ObjBound)
            opt_status_l.append(m.Status)
            
            m.dispose()
            logging.info(f'Robust count {robust_count} out of {idx+1}')
        except gp.GurobiError as e:
            m.dispose()
            logging.error(f"Error code {e.errno}: {e}")
            return
        except AttributeError:
            m.dispose()
            logging.error("Encountered an attribute error")
            return
    return is_robust_l, obj_l, obj_bd_l, opt_status_l

