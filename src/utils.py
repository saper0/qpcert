from typing import Tuple, Union, Sequence

from jaxtyping import Float, Integer
import numpy as np
import torch
from torch_sparse import coalesce

import gurobipy as gp
from gurobipy import GRB


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


def certify_robust_bilevel_svm(idx_labeled, idx_test, ntk, ntk_lb, ntk_ub, y, y_pred, svm_alpha, C=1, M=1e4, Mprime=1e4, n_adv=1):
    print('n_labeled ', idx_labeled.shape[0])
    print('ntk shapes ', ntk.shape, ntk_lb.shape, ntk_ub.shape)
    print('y shapes ', y.shape, y_pred.shape, y, y_pred)
    print('alpha ', svm_alpha.shape, svm_alpha)
    assert ntk_lb.min() <= ntk_ub.min()
    assert ntk_lb.max() <= ntk_ub.max()
    print(torch.all(torch.less(ntk_lb, ntk_ub))) # Todo: it doesn't hold element wise! Debug it!

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

    print(ntk_labeled.shape, ntk_labeled_ub.shape, ntk_labeled_lb.shape)
    print(ntk_unlabeled.shape, ntk_unlabeled_ub.shape, ntk_unlabeled_ub.shape)
    y_labeled = y_labeled*2 -1 

    svm_alpha = np.round(svm_alpha, 4) #rounding off helps to start with a feasible solution sometimes
    alpha_nz_mask = svm_alpha>1e-5 #svm_alpha>0 #
    alpha_c_mask = svm_alpha>(C-1e-5) #svm_alpha==C #
    eq_constraint = y_labeled*((ntk_labeled * svm_alpha)@y_labeled) - 1
    u_start = np.copy(svm_alpha)
    v_start = np.copy(svm_alpha)
    u_start[alpha_nz_mask] = 0
    v_start[~alpha_c_mask] = 0
    u_start[~alpha_nz_mask] = eq_constraint[~alpha_nz_mask]
    v_start[alpha_c_mask] = -eq_constraint[alpha_c_mask]
    s_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    u_nz_mask = u_start > 0
    s_start[u_nz_mask] = 1
    t_start = np.zeros(svm_alpha.shape[0], dtype=np.int64)
    v_nz_mask = v_start > 0
    t_start[v_nz_mask] = 1
    # print('ustart ', u_start)
    # print('sstart ', s_start)
    # print('vstart ', v_start)
    # print('tstart ', t_start)
    # print('svm alpha ', svm_alpha)
    # print('eq cons ', eq_constraint)
    # print('full constraint ', y_labeled*((ntk_labeled * svm_alpha)@y_labeled) - 1 - u_start + v_start)
    # print('u*alp ', u_start*svm_alpha)
    # print('v*(C-alp) ', v_start*(C-svm_alpha))

    obj_min = None
    robust_count = 0
    for idx in range(y_pred.shape[0]):
        if y_pred[idx] < 0:
            obj_min = False
        else:
            obj_min = True
        print(y_pred[idx], obj_min)
        try:
            # Create a new model
            m = gp.Model("poison_cert")

            # Create variables
            z_bound = np.minimum(0.0, C*ntk_lb.min())
            alpha = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, ub=C, name="alpha")
            u = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="u")
            v = m.addMVar(shape=n_labeled, vtype=GRB.CONTINUOUS, name="v")
            z = m.addMVar(shape=(n_labeled, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z")
            z_test = m.addMVar(shape=(1, n_labeled), vtype=GRB.CONTINUOUS, lb=z_bound, name="z_test")
            s = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="s")
            t = m.addMVar(shape=n_labeled, vtype=GRB.BINARY, name="t")

            # Add constraints
            m.addConstr(z <= ntk_labeled_ub * alpha, "c1")
            m.addConstr(z >= ntk_labeled_lb * alpha, "c2")
            m.addConstr(z_test <= ntk_unlabeled_ub[idx,:].reshape(1,-1) * alpha, "c11")
            m.addConstr(z_test >= ntk_unlabeled_lb[idx,:].reshape(1,-1) * alpha, "c22")
            m.addConstr(y_labeled*(z@y_labeled) - u + v == 1, "c3")
            m.addConstr(u <= M*s, "c4")
            m.addConstr(alpha <= C*(1-s), "c5")
            m.addConstr(v <= Mprime*t, "c6")
            m.addConstr(C-alpha <= C*(1-t), "c7")
            # m.addConstr(u*alpha == 0, "c8")
            # m.addConstr(v*(C-alpha) == 0, "c9")

            # Set the initial value of alpha and z
            alpha.Start = svm_alpha
            z.Start = ntk_labeled * svm_alpha
            z_test.Start = ntk_unlabeled[idx,:].reshape(1,-1) * svm_alpha
            u.Start = u_start
            v.Start = v_start
            s.Start = s_start
            t.Start = t_start

            # Set objective
            if obj_min:
                m.setObjective(z_test @ y_labeled, GRB.MINIMIZE)
                # m.Params.BestBdStop = -1-1e-8
            else:
                m.setObjective(z_test @ y_labeled, GRB.MAXIMIZE)
                # m.Params.BestObjStop = -1e-8
                # m.Params.BestBdStop = -1-1e-8

            m.Params.IntegralityFocus = 1 # to stabilize big-M constraint (must)
            m.Params.IntFeasTol = 1e-4 # to stabilize big-M constraint (helps, works without this also) 
            m.Params.LogToConsole = 0 # to suppress the logging in console - for better readability
            # Debugging
            m.Params.DualReductions = 0 # to know whether the model is infeasible or unbounded
            # m.write('poison_cert.lp') # helps in checking if the implemented model is correct
        
            # Played around with the following flags to escape infeasibility solutions
            m.Params.FeasibilityTol = 1e-4
            # m.Params.MIPGap = 1e-4
            # m.Params.MIPGapAbs = 1e-4
            m.Params.OptimalityTol = 1e-4
            # m.Params.Presolve = 0
            # m.Params.Aggregate = 0 #aggregation level in presolve
            m.Params.NumericFocus = 3
            # m.Params.MIPFocus = 1
            # m.Params.InfProofCuts = 0

            #Stopping criteria
            m.Params.SolutionLimit = 1 #stops when a solution is found
            # m.Params.NodeLimit = 0 #stops when the root is found 

            def callback(model, where):
                if where == gp.GRB.Callback.MESSAGE:
                    msg = model.cbGet(gp.GRB.Callback.MSG_STRING)
                    if "MIP start" in msg:
                        print("in callback ", msg)

            m.params.OutputFlag=0

            # Optimize model
            m.optimize()
            # Debugging -  use callback
            # m.optimize(callback)

            print('optimization status ', m.Status)

            # do IIS if the model is infeasible
            if m.Status == GRB.INFEASIBLE:
                m.computeIIS()

                # Print out the IIS constraints and variables
                print('\nThe following constraints and variables are in the IIS:')
                for c in m.getConstrs():
                    if c.IISConstr: print(f'\t{c.constrname}: {m.getRow(c)} {c.Sense} {c.RHS}')

                for v in m.getVars():
                    if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
                    if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')

                # m.feasRelaxS(2, True, False, True) #relaxes the constraints depending on the argument
                # m.optimize()
            elif m.Status == GRB.OPTIMAL or m.Status == GRB.NODE_LIMIT or m.Status == GRB.SOLUTION_LIMIT:
                # # Debugging values 
                print('Original ', y_pred[idx])
                print('Obj: %g' % m.ObjVal)

                # Debugging 
                # for var in m.getVars():
                #     print('%s %g' % (var.VarName, var.X))
                #
                # print('z ', z.X)
                # print('v ', v.X)
                # print('u ', u.X)
                # print('const check ', y_labeled*(z.X@y_labeled) - u.X + v.X)

            #check robustness using the objective value
            if obj_min and m.ObjVal > 0:
                robust_count += 1
            elif not obj_min and m.ObjVal < 0:
                robust_count += 1
            
            m.dispose()
            print('robust_count ', robust_count)

        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")
            return
        except AttributeError:
            print("Encountered an attribute error")
            return
    return (robust_count / y_pred.shape[0])