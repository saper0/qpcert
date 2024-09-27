import logging
from typing import Tuple, Union, Sequence
import os

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
    

def _set_big_M_label(C, ntk_labeled) -> Tuple[np.ndarray, np.ndarray]:
    """ Return big-M constraints for label poisoining."""
    ntk_abs_row_sum = np.abs(ntk_labeled).sum(axis=1)
    M_u = ntk_abs_row_sum * C - 1
    M_u[M_u < 0] = 0
    M_v = ntk_abs_row_sum * C + 1
    return M_u, M_v


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
        u_start[u_start<globals.zero_tol] = 0
        v_start[v_start<globals.zero_tol] = 0
        s_start = np.zeros(alpha.shape[0], dtype=np.int64)
        u_nz_mask = u_start > 0
        s_start[u_nz_mask] = 1
        t_start = np.zeros(alpha.shape[0], dtype=np.int64)
        v_nz_mask = v_start > 0
        t_start[v_nz_mask] = 1
        u_start_d[k] = u_start
        v_start_d[k] = v_start
        s_start_d[k] = s_start
        t_start_d[k] = t_start
        alpha_l.append(alpha)
        assert (alpha<0).sum() == 0
        assert (u_start<0).sum() == 0
        assert (v_start<0).sum() == 0
        assert ((y_labeled_*((ntk_labeled * alpha)@y_labeled_) - 1 - u_start + v_start) > globals.zero_tol).sum() == 0
        assert (u_start-M*s_start > globals.zero_tol).sum() == 0
        assert (alpha-C*(1-s_start) > globals.zero_tol).sum() == 0
        assert (v_start-Mprime*t_start > globals.zero_tol).sum() == 0
        assert (-alpha+C*t_start > globals.zero_tol).sum() == 0

    
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


def certify_collective_bilevel_svm(idx_labeled, idx_test, ntk, ntk_lb, ntk_ub, y, 
                               y_pred, svm_alpha, C=1, M=1e4, Mprime=1e4, 
                               milp=True):
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
    ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]
    ntk_unlabeled_ub = ntk_ub[idx_test,:][:,idx_labeled]
    ntk_unlabeled_lb = ntk_lb[idx_test,:][:,idx_labeled]

    # Labels are learned as -1 or 1, but loaded as 0 or 1
    y_labeled = y_labeled*2 -1 
    
    # Find the initial start feasible solution
    alpha_mask = svm_alpha < globals.zero_tol
    svm_alpha[alpha_mask] = 0
    alpha_nz_mask = svm_alpha>0 
    eq_constraint = y_labeled*((ntk_labeled * svm_alpha)@y_labeled) - 1
    u_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    v_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
    v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
    u_start[u_start<globals.zero_tol] = 0
    v_start[v_start<globals.zero_tol] = 0
    s_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    u_nz_mask = u_start > 0
    s_start[u_nz_mask] = 1
    t_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    v_nz_mask = v_start > 0
    t_start[v_nz_mask] = 1
    assert (svm_alpha<0).sum() == 0
    assert (u_start<0).sum() == 0
    assert (v_start<0).sum() == 0
    assert ((y_labeled*((ntk_labeled * svm_alpha)@y_labeled) - 1 - u_start + v_start).sum() < globals.zero_tol)
    assert (u_start-M*s_start > globals.zero_tol).sum() == 0
    assert (svm_alpha-C*(1-s_start) > globals.zero_tol).sum() == 0
    assert (v_start-Mprime*t_start > globals.zero_tol).sum() == 0
    assert (-svm_alpha+C*t_start > globals.zero_tol).sum() == 0

    obj = None
    is_robust = None
    opt_status = None
    y_worst_obj = None
    y_pred = y_pred.detach().cpu().numpy()
    n_unlabeled = y_pred.shape[0]
    y_pred_pos = (y_pred>=0)
    y_labeled_pos = (y_labeled>0)

    try:
        # Create a new model
        m = gp.Model("milp_provable_robustness")

        # Create variables
        count = m.addMVar(shape=n_unlabeled, vtype=GRB.BINARY, name="count")
        z_bound = np.minimum(0.0, C*ntk_lb.min())
        alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
        u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
        v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
        z = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z")
        z_test = m.addMVar(shape=(n_unlabeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z_test")
        if milp:
            s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
            t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

        # Add constraints
        m.addConstr(z <= ntk_labeled_ub * alpha, "z_ub")
        m.addConstr(z >= ntk_labeled_lb * alpha, "z_lb")
        m.addConstr(z_test <= ntk_unlabeled_ub * alpha, "z_test_ub")
        m.addConstr(z_test >= ntk_unlabeled_lb * alpha, "z_test_lb")
        p_test_lb = np.zeros(n_unlabeled, dtype=np.float64)
        p_test_ub = np.zeros(n_unlabeled, dtype=np.float64)
        if y_labeled_pos.sum() > 0:
            ntk_ub_pos = ntk_unlabeled_ub[:,y_labeled_pos]
            ntk_ub_pos[ntk_ub_pos<0] = 0
            p_test_ub += C* (ntk_ub_pos @ np.ones(y_labeled_pos.sum()))
            ntk_lb_pos = ntk_unlabeled_lb[:,y_labeled_pos]
            ntk_lb_pos[ntk_lb_pos>0] = 0
            p_test_lb += C* (ntk_lb_pos @ np.ones(y_labeled_pos.sum()))
        if (~y_labeled_pos).sum() > 0:
            ntk_lb_neg = ntk_unlabeled_lb[:,~y_labeled_pos]
            ntk_lb_neg[ntk_lb_neg>0] = 0
            p_test_ub += -C* (ntk_lb_neg @ np.ones((~y_labeled_pos).sum()))
            ntk_ub_neg = ntk_unlabeled_ub[:,~y_labeled_pos]
            ntk_ub_neg[ntk_ub_neg<0] = 0
            p_test_lb += -C* (ntk_ub_neg @ np.ones((~y_labeled_pos).sum()))
        if y_pred_pos.sum() >0:
            m.addConstr((z_test @ y_labeled)[y_pred_pos] <= p_test_ub[y_pred_pos]*(1-count[y_pred_pos]), "p_test_pos_ub")
            m.addConstr((z_test @ y_labeled)[y_pred_pos] >= p_test_lb[y_pred_pos]*(count[y_pred_pos]), "p_test_pos_ub")
        if (~y_pred_pos).sum() >0:
            m.addConstr((z_test @ y_labeled)[~y_pred_pos] <= p_test_ub[~y_pred_pos]*(count[~y_pred_pos]), "p_test_pos_ub")
            m.addConstr((z_test @ y_labeled)[~y_pred_pos] >= p_test_lb[~y_pred_pos]*(1-count[~y_pred_pos]), "p_test_pos_ub")
        m.addConstr(y_labeled*(z@y_labeled) - u + v == 1, "eq_constraint")
        if milp:
            m.addConstr(u <= M*s, "u_mil1")
            m.addConstr(alpha <= C*(1-s), "u_mil2")
            m.addConstr(v <= Mprime*t, "v_mil1")
            m.addConstr(C-alpha <= C*(1-t), "v_mil2")
        else:
            m.addConstr(u*alpha == 0, "u_comp_slack")
            m.addConstr(v*(C-alpha) == 0, "v_comp_slack")

        # Set the initial values for the parameters
        alpha.Start = svm_alpha
        z.Start = ntk_labeled * svm_alpha
        z_test.Start = ntk_unlabeled * svm_alpha
        u.Start = u_start
        v.Start = v_start
        if milp:
            s.Start = s_start
            t.Start = t_start
        count.Start = np.zeros(n_unlabeled)

        # Set objective
        m.setObjective(count @ np.ones(n_unlabeled), GRB.MAXIMIZE)

        # m.Params.BestObjStop = 0 # terminate when the objective reaches 0, implies node not robust
        m.Params.IntegralityFocus = 1 # to stabilize big-M constraint (must)
        m.Params.IntFeasTol = 1e-4 # to stabilize big-M constraint (helps, works without this also) 
        m.Params.LogToConsole = 0 # to suppress the logging in console - for better readability
        m.params.OutputFlag=0 # to suppress branch bound search tree outputs
        m.Params.DualReductions = 0 # to know whether the model is infeasible or unbounded                
    
        # Played around with the following flags to escape infeasibility solutions
        m.Params.FeasibilityTol = 1e-4
        m.Params.OptimalityTol = 1e-4
        m.Params.NumericFocus = 3
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
            print(f"Objective: #sign flips {m.ObjVal} out of {y_pred.shape[0]}")
            
            # Debugging 
            # for var in m.getVars():
            #     print('%s %g' % (var.VarName, var.X))
            #
            # print('z ', z.X)
            # print('v ', v.X)
            # print('u ', u.X)
            # print('equality constraint ', y_labeled*(z.X@y_labeled) - u.X + v.X)
            # print('z_test ', z_test.X)
            # print('Count ', count.X)
        is_robust = count.X
        obj = m.ObjVal
        opt_status = m.Status
        y_worst_obj = z_test.X @ y_labeled
        # log results
        print(f'Percentage of nodes certified {(y_pred.shape[0]-m.ObjVal)/y_pred.shape[0]}')
        
        logging.info(f"Objective: #sign flips {m.ObjVal} out of {y_pred.shape[0]}")
        logging.info(f'Percentage of nodes certified {(y_pred.shape[0]-m.ObjVal)/y_pred.shape[0]}')
        m.dispose()

    except gp.GurobiError as e:
        logging.error(f"Error code {e.errno}: {e}")
        return
    except AttributeError:
            logging.error("Encountered an attribute error")
            return
    return obj, is_robust, y_worst_obj, opt_status


def certify_robust_label(idx_labeled, idx_test, ntk, y, 
                         y_pred, svm_alpha, certificate_params,
                         l_flip=0.2, C=1, M=1e4, Mprime=1e4, 
                         milp=True):
    """TODO: Create documentation 
    """
    if isinstance(svm_alpha, torch.Tensor):
        svm_alpha = svm_alpha.numpy(force=True)
    
    n_labeled = idx_labeled.shape[0]
    ntk = ntk.detach().cpu().numpy()
    ntk_labeled = ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    y_labeled = y[idx_labeled].detach().cpu().numpy()
    ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]
    
    # Labels are learned as -1 or 1, but loaded as 0 or 1
    y_labeled_ = y_labeled*2 -1 

    # Find the initial start feasible solution
    alpha_mask = svm_alpha < globals.zero_tol
    svm_alpha[alpha_mask] = 0
    alpha_nz_mask = svm_alpha>0 
    eq_constraint = y_labeled_*((ntk_labeled * svm_alpha)@y_labeled_) - 1
    eq_constraint[np.abs(eq_constraint) < MILP_INT_FEAS_TOL] = 0 # Added bec. nec. for real data, remove if results in problems
    u_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    v_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
    v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
    u_start[u_start<globals.zero_tol] = 0
    v_start[v_start<globals.zero_tol] = 0
    s_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    u_nz_mask = u_start > 0
    s_start[u_nz_mask] = 1
    t_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    v_nz_mask = v_start > 0
    t_start[v_nz_mask] = 1
    assert (svm_alpha<0).sum() == 0
    assert (u_start<0).sum() == 0
    assert (v_start<0).sum() == 0
    assert ((y_labeled_*((ntk_labeled * svm_alpha)@y_labeled_) - 1 - u_start + v_start).sum() < globals.zero_tol)
    assert (u_start-M*s_start > globals.zero_tol).sum() == 0
    assert (svm_alpha-C*(1-s_start) > globals.zero_tol).sum() == 0
    assert (v_start-Mprime*t_start > globals.zero_tol).sum() == 0
    assert (-svm_alpha+C*t_start > globals.zero_tol).sum() == 0

    obj_l = []
    is_robust_l = []
    opt_status_l = []
    y_opt_l = []
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
            z_lb = np.minimum(0.0, -C)
            z_ub = np.maximum(0.0, C)
            alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
            u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
            v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
            z = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name="z")
            z_ = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name="z'")
            y_b = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name="y_binary")
            y = m.addMVar(shape=(n_labeled), lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="y")
            if milp:
                s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
                t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

            # Add constraints
            m.addConstr(y == 2*y_b - 1)
            m.addConstr(z <= alpha)
            m.addConstr(z >= -alpha)
            m.addConstr(z >= alpha - (1-y)*C)
            m.addConstr(z <= -alpha + (1+y)*C)
            M_ones = np.ones(shape=(n_labeled, n_labeled))
            y_T = y.reshape((n_labeled, 1))
            m.addConstr(z_ <= M_ones*z + (1-M_ones*y_T)*C) #
            m.addConstr(z_ >= M_ones*z - (1-M_ones*y_T)*C) #
            m.addConstr(z_ <= -M_ones*z + (1+M_ones*y_T)*C) #
            m.addConstr(z_ >= -M_ones*z - (1+M_ones*y_T)*C) #
            v_ones = np.ones(shape=(n_labeled))
            m.addConstr((z_*ntk_labeled)@v_ones - u + v == 1, "eq_constraint")
            n_flips = int(l_flip*n_labeled)
            m.addConstr(-((y*y_labeled_)-1)@v_ones <= 2*n_flips, "num_flips")
            if milp:
                if "use_tight_big_M" in certificate_params:
                    if certificate_params["use_tight_big_M"]:
                        M_u, M_v = _set_big_M_label(C, ntk_labeled)
                        assert (u_start > M_u).sum() == 0
                        assert (v_start > M_v).sum() == 0
                        m.addConstr(u <= M_u*s, "u_mil1")
                        m.addConstr(v <= M_v*t, "v_mil1")
                    else:
                        m.addConstr(u <= M*s, "u_mil1")
                        m.addConstr(v <= Mprime*t, "v_mil1")
                else:
                    m.addConstr(u <= M*s, "u_mil1")
                    m.addConstr(v <= Mprime*t, "v_mil1")
                m.addConstr(alpha <= C*(1-s), "u_mil2")
                m.addConstr(C-alpha <= C*(1-t), "v_mil2")
            else:
                m.addConstr(u*alpha == 0, "u_comp_slack")
                m.addConstr(v*(C-alpha) == 0, "v_comp_slack")

            # Set the initial values for the parameters
            alpha.Start = svm_alpha
            y_b.Start = y_labeled
            y.Start = y_labeled_
            z_start = y_labeled_ * svm_alpha
            z.Start = z_start
            z_.Start = np.outer(y_labeled_, z_start)
            u.Start = u_start
            v.Start = v_start
            if milp:
                s.Start = s_start
                t.Start = t_start

            # Set objective
            if obj_min:
                m.setObjective(ntk_unlabeled[idx,:] @ z, GRB.MINIMIZE)
            else:
                m.setObjective(ntk_unlabeled[idx,:] @ z, GRB.MAXIMIZE)

            m.Params.BestObjStop = 0 # terminate when the objective reaches 0, implies node not robust
            m.Params.BestBdStop = 0 # terminate when the best bound reaches 0, implies node robust
            if "IntegralityFocus" in certificate_params:
                m.Params.IntegralityFocus = certificate_params["IntegralityFocus"]
            else:
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
            if "DualReductions" in certificate_params:
                m.Params.DualReductions = certificate_params["DualReductions"]
            else:
                m.params.DualReductions = 0 
            if "Presolve" in certificate_params:
                m.Params.Presolve = certificate_params["Presolve"]
            if "Cuts" in certificate_params:
                m.Params.Cuts = certificate_params["Cuts"]
            if "Aggregate" in certificate_params:
                m.Params.Aggregate = certificate_params["Aggregate"]
            if "Threads" in certificate_params:
                m.Params.Threads = certificate_params["Threads"]
            # Played around with the following flags to escape infeasibility solutions
            m.Params.FeasibilityTol = MILP_FEASIBILITY_TOL
            m.Params.OptimalityTol = MILP_OPTIMALITY_TOL
            m.Params.NumericFocus = 0
            if "NodeLimit" in certificate_params:
                m.Params.NodeLimit = certificate_params["NodeLimit"] # Explored node limit to stop 
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
            if obj_min and m.ObjVal > 0:
                robust_count += 1
                is_robust_l.append(True)
            elif not obj_min and m.ObjVal < 0:
                robust_count += 1
                is_robust_l.append(True)
            else:
                is_robust_l.append(False)
            obj_l.append(m.ObjVal)
            opt_status_l.append(m.Status)
            y_opt_l.append(y_b.X.tolist())
            # print('y initial ', y_labeled_)
            # print('y opt ', y.X)
            # print('actual flips ', -((y.X*y_labeled_)-1)@v_ones)
            m.dispose()
            logging.info(f'Robust count {robust_count} out of {idx+1}')

        except gp.GurobiError as e:
            logging.error(f"Error code {e.errno}: {e}")
            return
        except AttributeError as a:
            logging.error(f"Encountered an attribute error {a.errno}: {a}")
            return
    return is_robust_l, obj_l, opt_status_l, y_opt_l


def certify_robust_label_one_vs_all_inexact(idx_labeled, idx_test, ntk, y, 
                                    y_pred, svm_alpha, certificate_params,
                                    l_flip=0.2, C=1, M=1e4, Mprime=1e4, 
                                    milp=True):
    """TODO: Create documentation 
    """
    if isinstance(svm_alpha, torch.Tensor):
        svm_alpha = svm_alpha.numpy(force=True)
    
    n_labeled = idx_labeled.shape[0]
    ntk = ntk.detach().cpu().numpy()
    ntk_labeled = ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    y_labeled = y[idx_labeled].detach().cpu().numpy()
    ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]
    
    # Labels are learned as -1 or 1, but loaded as 0 or 1
    #y_labeled_ = y_labeled*2 -1 
    
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
        #alpha_mask = alpha < certificate_params["alpha_tol"]
        alpha[alpha_mask] = 0
        alpha_nz_mask = alpha>0 
        eq_constraint = y_labeled_*((ntk_labeled * alpha)@y_labeled_) - 1
        eq_constraint[np.abs(eq_constraint) < MILP_FEASIBILITY_TOL] = 0 # Added bec. nec. for real data, remove if results in problems
        u_start = np.zeros(alpha.shape[0], dtype=np.float64)
        v_start = np.zeros(alpha.shape[0], dtype=np.float64)
        u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
        v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
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
        assert (-alpha-1+t_start > MILP_FEASIBILITY_TOL).sum() == 0

    obj_l = []
    is_robust_l = []
    opt_status_l = []
    y_opt_l = []
    obj_min = None
    robust_count = 0
    n_classes = y_pred.shape[1]
    for idx in range(y_pred.shape[0]):
        # Start with correct prediction & gradually work through others
        pred_ordered = y_pred[idx,:].topk(n_classes).indices
        is_robust = True
        stop_obj = None
        for j, k in enumerate(pred_ordered):
            k = k.cpu().item()
            y_labeled_ = np.copy(y_labeled)
            y_mask = y_labeled_ == k
            y_labeled_[y_mask] = 1
            y_labeled_[~y_mask] = -1
            y_labeled_b = (y_labeled_ + 1)/2

            if j == 0:
                obj_min = True
            else:
                obj_min = False
            try:
                # Create a new model
                m = gp.Model("milp_provable_robustness")

                # Create variables
                z_lb = np.minimum(0.0, -C)
                z_ub = np.maximum(0.0, C)
                alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
                u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
                v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
                z = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name="z")
                z_ = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name="z'")
                y_b = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name="y_binary")
                y = m.addMVar(shape=(n_labeled), lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="y")
                if milp:
                    s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
                    t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

                # Add constraints
                m.addConstr(y == 2*y_b - 1)
                m.addConstr(z <= alpha)
                m.addConstr(z >= -alpha)
                m.addConstr(z >= alpha - (1-y)*C)
                m.addConstr(z <= -alpha + (1+y)*C)
                M_ones = np.ones(shape=(n_labeled, n_labeled))
                y_T = y.reshape((n_labeled, 1))
                m.addConstr(z_ <= M_ones*z + (1-M_ones*y_T)*C) #
                m.addConstr(z_ >= M_ones*z - (1-M_ones*y_T)*C) #
                m.addConstr(z_ <= -M_ones*z + (1+M_ones*y_T)*C) #
                m.addConstr(z_ >= -M_ones*z - (1+M_ones*y_T)*C) #
                v_ones = np.ones(shape=(n_labeled))
                m.addConstr((z_*ntk_labeled)@v_ones - u + v == 1, "eq_constraint")
                n_flips = int(l_flip*n_labeled)
                m.addConstr(-((y*y_labeled_)-1)@v_ones <= 2*n_flips, "num_flips")
                if milp:
                    if "use_tight_big_M" in certificate_params:
                        if certificate_params["use_tight_big_M"]:
                            M_u, M_v = _set_big_M_label(C, ntk_labeled)
                            assert (u_start_d[k] > M_u).sum() == 0
                            assert (v_start_d[k] > M_v).sum() == 0
                            m.addConstr(u <= M_u*s, "u_mil1")
                            m.addConstr(v <= M_v*t, "v_mil1")
                        else:
                            m.addConstr(u <= M*s, "u_mil1")
                            m.addConstr(v <= Mprime*t, "v_mil1")
                    else:
                        m.addConstr(u <= M*s, "u_mil1")
                        m.addConstr(v <= Mprime*t, "v_mil1")
                    m.addConstr(alpha <= C*(1-s), "u_mil2")
                    m.addConstr(C-alpha <= C*(1-t), "v_mil2")
                else:
                    m.addConstr(u*alpha == 0, "u_comp_slack")
                    m.addConstr(v*(C-alpha) == 0, "v_comp_slack")

                # Set the initial values for the parameters
                alpha.Start = alpha_l[k]
                y_b.Start = y_labeled_b
                y.Start = y_labeled_
                z_start = y_labeled_ * alpha_l[k]
                z.Start = z_start
                z_.Start = np.outer(y_labeled_, z_start)
                u.Start = u_start_d[k]
                v.Start = v_start_d[k]
                if milp:
                    s.Start = s_start_d[k]
                    t.Start = t_start_d[k]

                # Set objective
                if obj_min:
                    m.setObjective(ntk_unlabeled[idx,:] @ z, GRB.MINIMIZE)
                else:
                    m.setObjective(ntk_unlabeled[idx,:] @ z, GRB.MAXIMIZE)

                if stop_obj is not None:
                    m.Params.BestObjStop = stop_obj # terminate when the objective reaches 0, implies node not robust
                    m.Params.BestBdStop = stop_obj # if the bound falls below the stop_obj, node definitely can't change prediction
                else:
                    assert obj_min == True
                    secnd_best_ypred = torch.topk(y_pred, 2).values[0][-1]
                    m.Params.BestObjStop = secnd_best_ypred.detach().cpu().item()
                if "IntegralityFocus" in certificate_params:
                    m.Params.IntegralityFocus = certificate_params["IntegralityFocus"]
                else:
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
                if "Threads" in certificate_params:
                    m.Params.Threads = certificate_params["Threads"]
                if "Presolve" in certificate_params:
                    m.Params.Presolve = certificate_params["Presolve"]
                # Played around with the following flags to escape infeasibility solutions
                m.Params.FeasibilityTol = MILP_FEASIBILITY_TOL
                m.Params.OptimalityTol = MILP_OPTIMALITY_TOL
                m.Params.NumericFocus = 0
                if "TimeLimit" in certificate_params:
                    m.Params.TimeLimit = certificate_params["TimeLimit"]
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
                logging.info(f'Original {y_pred[idx, k].item():.5f}, Opt objective '
                            f'{m.ObjVal:.5f}, Opt bound: {m.ObjBound:.5f}, stop_obj: {stop_obj}')
                # analyse result
                if stop_obj is None:
                    if m.ObjVal < secnd_best_ypred:
                        is_robust = False
                        y_opt_l.append(y_b.X.tolist())
                        m.dispose()
                        break
                    if m.Status == GRB.OPTIMAL:
                        stop_obj = m.ObjVal
                    else:
                        stop_obj = m.ObjBound
                else:
                    if m.Status == GRB.OPTIMAL:
                        if m.ObjVal >= stop_obj:
                            is_robust = False
                            y_opt_l.append(y_b.X.tolist())
                            m.dispose()
                            break
                    else:
                        if m.ObjBound >= stop_obj:
                            is_robust = False
                            y_opt_l.append(y_b.X.tolist())
                            m.dispose()
                            break
                #obj_l.append(m.ObjVal)
                #opt_status_l.append(m.Status)
                # print('y initial ', y_labeled_)
                # print('y opt ', y.X)
                # print('actual flips ', -((y.X*y_labeled_)-1)@v_ones)
                m.dispose()

            except gp.GurobiError as e:
                logging.error(f"Error code {e.errno}: {e}")
                return
            except AttributeError as a:
                logging.error(f"Encountered an attribute error {a.errno}: {a}")
                return
        is_robust_l.append(is_robust)
        if is_robust:
            y_opt_l.append([])
    return is_robust_l, y_opt_l


def certify_robust_label_one_vs_all(idx_labeled, idx_test, ntk, y, 
                                    y_pred, svm_alpha, certificate_params,
                                    l_flip=0.2, C=1, M=1e4, Mprime=1e4,
                                    milp=True):
    """TODO: Create documentation 
    """
    if isinstance(svm_alpha, torch.Tensor):
        svm_alpha = svm_alpha.numpy(force=True)
    
    n_labeled = idx_labeled.shape[0]
    ntk = ntk.detach().cpu().numpy()
    ntk_labeled = ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    y_labeled = y[idx_labeled].detach().cpu().numpy()
    ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]
    
    # Labels are learned as -1 or 1, but loaded as 0 or 1
    #y_labeled_ = y_labeled*2 -1 
    
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
        #alpha_mask = alpha < certificate_params["alpha_tol"]
        alpha[alpha_mask] = 0
        alpha_nz_mask = alpha>0 
        eq_constraint = y_labeled_*((ntk_labeled * alpha)@y_labeled_) - 1
        eq_constraint[np.abs(eq_constraint) < MILP_FEASIBILITY_TOL] = 0 # Added bec. nec. for real data, remove if results in problems
        u_start = np.zeros(alpha.shape[0], dtype=np.float64)
        v_start = np.zeros(alpha.shape[0], dtype=np.float64)
        u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
        v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
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
        assert (-alpha-1+t_start > MILP_FEASIBILITY_TOL).sum() == 0

    obj_l = []
    is_robust_l = []
    opt_status_l = []
    y_opt_l = []
    obj_b_l = []
    obj_min = None
    robust_count = 0
    n_classes = y_pred.shape[1]
    if "epsilon" in certificate_params:
        eps = certificate_params["epsilon"]
    else:
        eps = 1e-3
    for idx in range(y_pred.shape[0]):
        # Start with correct prediction & gradually work through others
        pred_ordered = y_pred[idx,:].topk(n_classes).indices
        is_robust = True
        

        try:
            # Create a new model
            m = gp.Model("milp_provable_robustness")
            z_lb = np.minimum(0.0, -C)
            z_ub = np.maximum(0.0, C)
            p_ub = np.abs(ntk_unlabeled[idx,:]).sum() * C
            p_lb = -p_ub

            # Create variables independent of k
            alpha = {}
            u = {}
            v = {}
            z = {}
            z_ = {}        
            s = {}
            t = {}        
            p = {}
            y = {}
            y_b = {}
            y_b_ = {}
            y_int = m.addMVar(shape=(n_labeled), lb=0, ub=n_classes-1, vtype=GRB.INTEGER, name="y_int")
            delta = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name="y_int_changed")
            delta_ = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name="y_int_changed_sign")
            p_max = m.addVar(p_lb, p_ub, vtype=GRB.CONTINUOUS, name="max_logit")
            b = m.addMVar(shape=n_classes-1, vtype=GRB.BINARY, name="b")
            m.update()

            # Create variables for each index k 
            for j, k in enumerate(pred_ordered):
                k = k.cpu().item()
                y_labeled_ = np.copy(y_labeled)
                # -> start solution
                y_mask = y_labeled_ == k
                y_labeled_[y_mask] = 1
                y_labeled_[~y_mask] = -1
                y_labeled_b = (y_labeled_ + 1)/2
                y_smaller = np.array(y_labeled < k, dtype=np.float32)

                # Create variables
                y_b[k] = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name=f"y_b{k}")
                y_b_[k] = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name=f"y_b_{k}")
                y[k] = m.addMVar(shape=(n_labeled), lb=-1, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{k}")
                y_T = y[k].reshape((n_labeled, 1))
                alpha[k] = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name=f"alpha_{k}")
                u[k] = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name=f"u_{k}")
                v[k] = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name=f"v_{k}")
                z[k] = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name=f"z_{k}")
                z_[k] = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name=f"z'_{k}")
                p[k] = m.addVar(p_lb, p_ub, vtype=GRB.CONTINUOUS, name=f"logit_{k}")
                if milp:
                    s[k] = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name=f"s_{k}")
                    t[k] = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name=f"t_{k}")
                m.update()

                # Add constraints
                m.addConstr(y[k] == 2*y_b[k] - 1)
                M_ = n_classes - 1
                m.addConstr(y_int - k <= M_*(1-y_b[k]), f"y_int_b1_{k}")
                m.addConstr(y_int - k >= -M_*(1-y_b[k]), f"y_int_b2_{k}")
                m.addConstr(y_int - k <= -eps*(1-y_b[k]) + (M_+eps)*(1-y_b_[k]), f"y_int_b_1_{k}")
                m.addConstr(y_int - k >= eps*(1-y_b[k]) - (M_+eps)*y_b_[k], f"y_int_b_2_{k}")
                m.addConstr(p[k] == ntk_unlabeled[idx,:] @ z[k], f"logit_constraint_{k}")
                if j > 0:
                    m.addConstr(p_max >= p[k], f"logit_max_constraint1_{k}")
                    m.addConstr(p_max <= p[k] + (1-b[j-1])*(p_ub - p_lb), f"logit_max_constraint2_{k}")
                m.addConstr(z[k] <= alpha[k], f"z[{k}]_alpha1")
                m.addConstr(z[k] >= -alpha[k], f"z[{k}]_alpha2")
                m.addConstr(z[k] >= alpha[k] - (1-y[k])*C, f"z[{k}]_alphaC1")
                m.addConstr(z[k] <= -alpha[k] + (1+y[k])*C, f"z[{k}]_alphaC2")
                M_ones = np.ones(shape=(n_labeled, n_labeled))
                m.addConstr(z_[k] <= M_ones*z[k] + (1-M_ones*y_T)*C, f"z_[{k}]_1") #
                m.addConstr(z_[k] >= M_ones*z[k] - (1-M_ones*y_T)*C, f"z_[{k}]_2") #
                m.addConstr(z_[k] <= -M_ones*z[k] + (1+M_ones*y_T)*C, f"z_[{k}]_3") #
                m.addConstr(z_[k] >= -M_ones*z[k] - (1+M_ones*y_T)*C, f"z_[{k}]_4") #
                v_ones = np.ones(shape=(n_labeled))
                m.addConstr((z_[k]*ntk_labeled)@v_ones - u[k] + v[k] == 1, f"eq_constraint_{k}")

                if milp:
                    if "use_tight_big_M" in certificate_params:
                        if certificate_params["use_tight_big_M"]:
                            M_u, M_v = _set_big_M_label(C, ntk_labeled)
                            assert (u_start_d[k] > M_u).sum() == 0
                            assert (v_start_d[k] > M_v).sum() == 0
                            m.addConstr(u[k] <= M_u*s[k], f"u_mil1_{k}")
                            m.addConstr(v[k] <= M_v*t[k], f"v_mil1_{k}")
                        else:
                            m.addConstr(u[k] <= M*s[k], f"u_mil1_{k}")
                            m.addConstr(v[k] <= Mprime*t[k], f"v_mil1_{k}")
                    else:
                        m.addConstr(u[k] <= M*s[k], f"u_mil1_{k}")
                        m.addConstr(v[k] <= Mprime*t[k], f"v_mil1_{k}")
                    m.addConstr(alpha[k] <= C*(1-s[k]), f"u_mil2_{k}")
                    m.addConstr(C-alpha[k] <= C*(1-t[k]), f"v_mil2_{k}")
                else:
                    m.addConstr(u[k]*alpha[k] == 0, f"u_comp_slack_{k}")
                    m.addConstr(v[k]*(C-alpha[k]) == 0, f"v_comp_slack_{k}")
                m.update()
                # Set the initial values for the parameters
                y_b[k].Start = y_labeled_b
                y_b_[k].Start = y_smaller
                y[k].Start = y_labeled_
                alpha[k].Start = alpha_l[k]
                z_start = y_labeled_ * alpha_l[k]
                z[k].Start = z_start
                z_[k].Start = np.outer(y_labeled_, z_start)
                u[k].Start = u_start_d[k]
                v[k].Start = v_start_d[k]
                if milp:
                    s[k].Start = s_start_d[k]
                    t[k].Start = t_start_d[k]
                p[k].Start = ntk_unlabeled[idx,:] @ z_start
                if j == 1:
                    p_max.Start = ntk_unlabeled[idx,:] @ z_start
                    b[j-1].Start = 1
                elif j > 1:
                    b[j-1].Start = 0
                m.update()
            print(p_max.Start)
            k_best = pred_ordered[0].cpu().item()
            print(p[k_best].Start)
            print(p[k_best].Start - p_max.Start)
            # Index independent constraints
            M_ = n_classes - 1
            m.addConstr(y_labeled - y_int <= M_*delta, f"y_int_delta1")
            m.addConstr(y_labeled - y_int >= -M_*delta, f"y_int_delta2")
            m.addConstr(y_labeled - y_int <= -eps*delta + (M_+eps)*(1-delta_), f"y_int_delta_1")
            m.addConstr(y_labeled - y_int >= eps*delta - (M_+eps)*delta_, f"y_int_delta_2")
            m.addConstr(b.sum() == 1, "b_sum")
            n_flips = int(l_flip*n_labeled)
            m.addConstr(delta.sum() <= n_flips, "budget_constraint")
            m.update()

            # Index independent starts
            y_int.Start = y_labeled
            delta.Start = np.zeros(n_labeled)
            delta_.Start = np.zeros(n_labeled)
            m.update()

            # Set objective
            m.setObjective(p[k_best] - p_max, GRB.MINIMIZE)
            m.Params.BestObjStop = 0 # terminate when the objective reaches 0, implies node not robust
            m.Params.BestBdStop = 0 # if the bound falls below the stop_obj, node definitely can't change prediction
            m.update()

            m.write("./cache/model.lp")
            print(m.getConstrByName("R481"))

            if "IntegralityFocus" in certificate_params:
                m.Params.IntegralityFocus = certificate_params["IntegralityFocus"]
            else:
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
            if "Threads" in certificate_params:
                m.Params.Threads = certificate_params["Threads"]
            if "Presolve" in certificate_params:
                m.Params.Presolve = certificate_params["Presolve"]
            # Played around with the following flags to escape infeasibility solutions
            m.Params.FeasibilityTol = MILP_FEASIBILITY_TOL
            m.Params.OptimalityTol = MILP_OPTIMALITY_TOL
            m.Params.NumericFocus = 0
            if "TimeLimit" in certificate_params:
                m.Params.TimeLimit = certificate_params["TimeLimit"]
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
            logging.info(f'Original {y_pred[idx, k].item():.5f}, Opt objective '
                        f'{m.ObjVal:.5f}, Opt bound: {m.ObjBound:.5f}')
            obj_l.append(m.ObjVal)
            obj_b_l.append(m.ObjBound)
            opt_status_l.append(m.Status)
            y_opt_l.append(y_int.X.tolist())
            if m.Status == GRB.OPTIMAL:
                if m.ObjVal < MILP_OPTIMALITY_TOL:
                    is_robust = False
            else:
                if m.ObjBound < MILP_OPTIMALITY_TOL:
                    is_robust = False
            is_robust_l.append(is_robust)
            m.dispose()

        except gp.GurobiError as e:
            logging.error(f"Error code {e.errno}: {e}")
            return
        except AttributeError as a:
            logging.error(f"Encountered an attribute error {a.errno}: {a}")
            return
    return is_robust_l, y_opt_l, obj_l, obj_b_l, opt_status_l


def certify_collective_robust_label(idx_labeled, idx_test, ntk, y, 
                         y_pred, svm_alpha, certificate_params, l_flip=0.2, C=1, M=1e4, Mprime=1e4, 
                         milp=True, model_params=None, save_params=None):
    """TODO: Create documentation 
    """
    if isinstance(svm_alpha, torch.Tensor):
        svm_alpha = svm_alpha.numpy(force=True)
    
    n_labeled = idx_labeled.shape[0]
    ntk = ntk.detach().cpu().numpy()
    ntk_labeled = ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    y_labeled = y[idx_labeled].detach().cpu().numpy()
    ntk_unlabeled = ntk[idx_test,:][:,idx_labeled]

    if "MILP_INT_FEAS_TOL" in certificate_params:
        _MILP_INT_FEAS_TOL = certificate_params["MILP_INT_FEAS_TOL"]
    else:
        _MILP_INT_FEAS_TOL = MILP_INT_FEAS_TOL
    if "MILP_FEASIBILITY_TOL" in certificate_params:
        _MILP_FEASIBILITY_TOL = certificate_params["MILP_FEASIBILITY_TOL"]
    else:
        _MILP_FEASIBILITY_TOL = MILP_FEASIBILITY_TOL
    if "MILP_OPTIMALITY_TOL" in certificate_params:
        _MILP_OPTIMALITY_TOL = certificate_params["MILP_OPTIMALITY_TOL"]
    else:
        _MILP_OPTIMALITY_TOL = MILP_OPTIMALITY_TOL
    
    # Labels are learned as -1 or 1, but loaded as 0 or 1
    y_labeled_ = y_labeled*2 -1 
    
    # Find the initial start feasible solution
    alpha_mask = svm_alpha < globals.zero_tol
    svm_alpha[alpha_mask] = 0
    alpha_nz_mask = svm_alpha>0 
    eq_constraint = y_labeled_*((ntk_labeled * svm_alpha)@y_labeled_) - 1
    eq_constraint[np.abs(eq_constraint) < _MILP_INT_FEAS_TOL] = 0 # Added bec. nec. for real data, remove if results in problems
    u_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    v_start = np.zeros(svm_alpha.shape[0], dtype=np.float64)
    u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
    v_start[alpha_nz_mask] = -eq_constraint[alpha_nz_mask]
    u_start[u_start<globals.zero_tol] = 0
    v_start[v_start<globals.zero_tol] = 0
    s_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    u_nz_mask = u_start > 0
    s_start[u_nz_mask] = 1
    t_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    v_nz_mask = v_start > 0
    t_start[v_nz_mask] = 1
    assert (svm_alpha<0).sum() == 0
    assert (u_start<0).sum() == 0
    assert (v_start<0).sum() == 0
    assert ((y_labeled_*((ntk_labeled * svm_alpha)@y_labeled_) - 1 - u_start + v_start).sum() < globals.zero_tol)
    assert (u_start-M*s_start > globals.zero_tol).sum() == 0
    assert (svm_alpha-C*(1-s_start) > globals.zero_tol).sum() == 0
    assert (v_start-Mprime*t_start > globals.zero_tol).sum() == 0
    assert (-svm_alpha+C*t_start > globals.zero_tol).sum() == 0

    is_robust_l = []
    y_opt_l = []
    n_unlabeled = y_pred.shape[0]
    y_pred_pos = (y_pred>=0)
    try:
        # Create a new model
        m = gp.Model("milp_provable_robustness")

        # Create variables
        z_lb = np.minimum(0.0, -C)
        z_ub = np.maximum(0.0, C)
        count = m.addMVar(shape=n_unlabeled, vtype=GRB.BINARY, name="count")
        alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
        u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
        v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
        z = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name="z")
        z_ = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_lb, ub=z_ub, name="z'")
        y_b = m.addMVar(shape=(n_labeled), vtype=GRB.BINARY, name="y_binary")
        y = m.addMVar(shape=(n_labeled), lb=-1, ub=1, vtype=GRB.INTEGER, name="y")
        if milp:
            s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
            t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

        # Add constraints
        m.addConstr(y == 2*y_b - 1)
        m.addConstr(z <= alpha)
        m.addConstr(z >= -alpha)
        m.addConstr(z >= alpha - (1-y)*C)
        m.addConstr(z <= -alpha + (1+y)*C)
        M_ones = np.ones(shape=(n_labeled, n_labeled))
        y_T = y.reshape((n_labeled, 1))
        m.addConstr(z_ <= M_ones*z + (1-M_ones*y_T)*C) #
        m.addConstr(z_ >= M_ones*z - (1-M_ones*y_T)*C) #
        m.addConstr(z_ <= -M_ones*z + (1+M_ones*y_T)*C) #
        m.addConstr(z_ >= -M_ones*z - (1+M_ones*y_T)*C) #
        v_ones = np.ones(shape=(n_labeled))
        m.addConstr((z_*ntk_labeled)@v_ones - u + v == 1, "eq_constraint")
        n_flips = int(l_flip*n_labeled)
        m.addConstr(-((y*y_labeled_)-1)@v_ones <= 2*n_flips, "num_flips")
        if milp:
            if "use_tight_big_M" in certificate_params:
                if certificate_params["use_tight_big_M"]:
                    logging.info("Using tight big-Ms.")
                    M_u, M_v = _set_big_M_label(C, ntk_labeled)
                    assert (u_start > M_u).sum() == 0
                    assert (v_start > M_v).sum() == 0
                    m.addConstr(u <= M_u*s, "u_mil1")
                    m.addConstr(v <= M_v*t, "v_mil1")
                else:
                    logging.info("Using default big-Ms.")
                    m.addConstr(u <= M*s, "u_mil1")
                    m.addConstr(v <= Mprime*t, "v_mil1")
            else:
                logging.info("Using default big-Ms.")
                m.addConstr(u <= M*s, "u_mil1")
                m.addConstr(v <= Mprime*t, "v_mil1")
            m.addConstr(alpha <= C*(1-s), "u_mil2")
            m.addConstr(C-alpha <= C*(1-t), "v_mil2")
        else:
            m.addConstr(u*alpha == 0, "u_comp_slack")
            m.addConstr(v*(C-alpha) == 0, "v_comp_slack")
        p_test = C * np.abs(ntk_unlabeled)@v_ones
        p_test_lb = - p_test
        p_test_ub = + p_test
        if y_pred_pos.sum() >0:
            m.addConstr((ntk_unlabeled @ z)[y_pred_pos] <= p_test_ub[y_pred_pos]*(1-count[y_pred_pos]), "p_test_pos_ub")
            m.addConstr((ntk_unlabeled @ z)[y_pred_pos] >= p_test_lb[y_pred_pos]*(count[y_pred_pos]), "p_test_pos_ub")
        if (~y_pred_pos).sum() >0:
            m.addConstr((ntk_unlabeled @ z)[~y_pred_pos] <= p_test_ub[~y_pred_pos]*(count[~y_pred_pos]), "p_test_neg_ub")
            m.addConstr((ntk_unlabeled @ z)[~y_pred_pos] >= p_test_lb[~y_pred_pos]*(1-count[~y_pred_pos]), "p_test_neg_ub")

        # Set the initial values for the parameters
        alpha.Start = svm_alpha
        y_b.Start = y_labeled
        y.Start = y_labeled_
        z_start = y_labeled_ * svm_alpha
        z.Start = z_start
        z_.Start = np.outer(y_labeled_, z_start)
        u.Start = u_start
        v.Start = v_start
        count.Start = 0
        if milp:
            s.Start = s_start
            t.Start = t_start
        # Set objective
        m.setObjective(count @ np.ones(n_unlabeled), GRB.MAXIMIZE)
        # m.Params.BestObjStop = 0 # terminate when the objective reaches 0, implies node not robust
        if "IntegralityFocus" in certificate_params:
            m.Params.IntegralityFocus = certificate_params["IntegralityFocus"]
        else:
            m.Params.IntegralityFocus = 1 # to stabilize big-M constraint (must)
        m.Params.IntFeasTol = _MILP_INT_FEAS_TOL # to stabilize big-M constraint (helps, works without this also) 
        if "LogToConsole" in certificate_params:
            m.Params.LogToConsole = certificate_params["LogToConsole"]
        else:
            m.Params.LogToConsole = 0 # to suppress the logging in console - for better readability
        if "OutputFlag" in certificate_params:
            m.Params.OutputFlag = certificate_params["OutputFlag"]
        else:
            m.params.OutputFlag = 0 # to suppress branch bound search tree outputs
        if "DualReductions" in certificate_params:
            m.Params.DualReductions = certificate_params["DualReductions"]
        else:
            m.params.DualReductions = 0 
        if "Presolve" in certificate_params:
            m.Params.Presolve = certificate_params["Presolve"]
        if "Cuts" in certificate_params:
            m.Params.Cuts = certificate_params["Cuts"]
        if "Aggregate" in certificate_params:
            m.Params.Aggregate = certificate_params["Aggregate"]
        if "Threads" in certificate_params:
            m.Params.Threads = certificate_params["Threads"]
        # Played around with the following flags to escape infeasibility solutions
        m.Params.FeasibilityTol = _MILP_FEASIBILITY_TOL
        m.Params.OptimalityTol = _MILP_OPTIMALITY_TOL
        m.Params.NumericFocus = 0
        if "SoftMemLimit" in certificate_params:
            m.Params.SoftMemLimit = certificate_params["SoftMemLimit"]
        if "NodeLimit" in certificate_params:
            m.Params.NodeLimit = certificate_params["NodeLimit"] # Explored node limit to stop 
        if "TimeLimit" in certificate_params:
            m.Params.TimeLimit = certificate_params["TimeLimit"]
        # m.Params.MIPGap = 1e-4
        m.Params.MIPGapAbs = 0.99
        # m.Params.Presolve = 0
        # m.Params.Aggregate = 0 #aggregation level in presolve
        if "MIPFocus" in certificate_params:
            m.Params.MIPFocus = certificate_params["MIPFocus"]
        else:
            m.Params.MIPFocus = 0
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
            print(f"Objective: #sign flips {m.ObjVal} out of {y_pred.shape[0]}")
            print(f'Percentage of nodes certified {(y_pred.shape[0]-m.ObjVal)/y_pred.shape[0]}')

            # Debugging 
            # for var in m.getVars():
            #     print('%s %g' % (var.VarName, var.X))
            #
            # print('z ', z.X)
            # print('v ', v.X)
            # print('u ', u.X)
            # print('equality constraint ', y_labeled*(z.X@y_labeled) - u.X + v.X)

        # log results
        logging.info(f"Objective: #sign flips {m.ObjVal} out of {y_pred.shape[0]}")
        logging.info(f'Percentage of nodes certified {(y_pred.shape[0]-m.ObjVal)/y_pred.shape[0]}')
        
        is_robust_l = (1-count.X).tolist()
        obj = m.ObjVal
        obj_bound = m.ObjBound
        opt_status = m.Status
        y_test_worst_obj = ntk_unlabeled @ z.X
        y_opt_l = y_b.X.tolist()
        if save_params is not None:
            if opt_status == 9 or opt_status == 17:
                base_filename = save_params["dataset"] + "_" \
                                + save_params["label"] + "_" \
                                + str(save_params["seed"])
                solution_file = os.path.join(save_params["save_dir"], base_filename + ".sol")
                logging.info(f"Opt Status {opt_status}, saving solution to {solution_file}.")
                m.write(solution_file)
        # print('y initial ', y_labeled_)
        # print('y opt ', y.X)
        # print('actual flips ', -((y.X*y_labeled_)-1)@v_ones)
        m.dispose()

    except gp.GurobiError as e:
        logging.error(f"Error code {e.errno}: {e}")
        return
    except AttributeError as a:
        logging.error(f"Encountered an attribute error {a.errno}: {a}")
        return
    return is_robust_l, obj, obj_bound, opt_status, y_opt_l, y_test_worst_obj