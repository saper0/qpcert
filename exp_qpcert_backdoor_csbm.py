import logging
from typing import Any, Dict, Union, Tuple
import os
import socket

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import seml
import torch

from src import utils, globals
from src.attacks import create_structure_attack
from src.data import get_graph, split
from src.graph_models.csbm import CSBM
from src.models.ntk import NTK
from common import count_edges_for_idx


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, 
                                                          overwrite=overwrite))
    
    seed = 0

    data_params = dict(
        dataset = "csbm",
        learning_setting = "inductive", # or "transdructive" 
        specification = dict(
            classes = 2,
            n_trn_labeled = 600,
            n_trn_unlabeled = 0,
            n_val = 200,
            n_test = 200,
            sigma = 1,
            avg_within_class_degree = 1.58 * 2,
            avg_between_class_degree = 0.37 * 2,
            K = 1.5,
            seed = 0 # used to generate the dataset & data split
        ),
        #specification = dict(
        #    n_per_class = 20,
        #    fraction_test = 0.1,
        #    data_dir = "./data",
        #    make_undirected = True,
        #    binary_attr = False,
        #    balance_test = True,
        #)
    )
    
    model_params = dict(
        label = "GCN",
        model = "GCN",
        normalization = "row_normalization",
        depth = 1,
        regularizer = 0.1,
        pred_method = "svm",
        solver = "qplayer",
        alpha_tol = 1e-4,
        bias = False,
    )

    certificate_params = dict(
        n_adversarial = 10, # number adversarial nodes
        method = "XXT",
        perturbation_model = "linf",
        delta = 0.01, # l0: local budget = delta * feature_dim
        delta_absolute = True, # if false interpreted as % of 2*mu
        attack_nodes = "test", # "train", "all"
    )

    verbosity_params = dict(
        debug_lvl = "info"
    )  

    other_params = dict(
        device = "gpu",
        dtype = "float64",
        allow_tf32 = False,
        enable_gradient = False,
        path_gurobi_license = "", #default path if empty
    )


def set_debug_lvl(debug_lvl: str):
    if debug_lvl is not None and isinstance(debug_lvl, str):
        logger = logging.getLogger()
        if debug_lvl.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_lvl.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_lvl.lower() == "warning":
            logger.setLevel(logging.WARNING)
        if debug_lvl.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_lvl.lower() == "error":
            logger.setLevel(logging.ERROR)


def log_configuration(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                      certificate_params: Dict[str, Any], 
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
    logging.info(f"certification_params: {certificate_params}")
    logging.info(f"verbosity_params: {verbosity_params}")
    logging.info(f"other_params: {other_params}")
    logging.info(f"seed: {seed}")


def choose_gurobi_license(other_params: Dict[str, Any]):
    if other_params["path_gurobi_license"] != "":
        os.environ["GRB_LICENSE_FILE"] = other_params["path_gurobi_license"] 
    elif other_params["path_gurobi_license"] == "":
        logging.info("No known gurobi license provided. Trying default.")
    else:
        assert False

def configure_hardware(
    other_params: Dict[str, Any], seed: int
) -> Union[torch.device, str]:
    """Configure seed and computational hardware. Return calc. device."""
    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # dtype
    dtype = other_params["dtype"]
    if other_params["dtype"] == "float32":
        dtype = torch.float32
    elif other_params["dtype"] == "float64":
        dtype = torch.float64
    elif type(other_params["dtype"]) is not torch.dtype:
        assert False, "Given dtype not supported."

    # Gurobi
    choose_gurobi_license(other_params)

    # Hardware
    torch.backends.cuda.matmul.allow_tf32 = bool(other_params["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(other_params["allow_tf32"])

    device = other_params["device"]
    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}")
        logging.info(f"Currently on gpu device {device}")

    return device, dtype


def setup_experiment(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                     certificate_params: Dict[str, Any], 
                     verbosity_params: Dict[str, Any], 
                     other_params: Dict[str, Any], seed: int
) -> Tuple[Union[torch.device, str], np.random.Generator]:
    """Set general configuration for the seml experiment and configure hardware.
    
    Returns the device and a random number generator.
    """
    set_debug_lvl(verbosity_params["debug_lvl"])
    log_configuration(data_params, model_params, certificate_params,
                      verbosity_params, other_params, seed)
    globals.init(other_params)
    device, dtype = configure_hardware(other_params, seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    return device, dtype, rng


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        certificate_params: Dict[str, Any],
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device, dtype, rng = setup_experiment(data_params, model_params, 
                                          certificate_params, verbosity_params, 
                                          other_params, seed)
    
    X, A, y, csbm = get_graph(data_params, sort=True, return_csbm=True)
    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()
    idx_trn, idx_unlabeled, idx_val, idx_test = split(data_params, y)
    if len(idx_unlabeled) != 0:
        idx_test = np.concatenate((idx_unlabeled, idx_test))
    X = torch.tensor(X, dtype=dtype, device=device)
    A = torch.tensor(A, dtype=dtype, device=device)
    y = torch.tensor(y, device=device)
    n_classes = int(y.max() + 1)

    idx_labeled = np.concatenate((idx_trn, idx_val)) 
    # idx of labeled nodes in nodes known during training (for semi-supervised)
    if not data_params["learning_setting"] == "transductive":
        assert False, "Only transductive setting supported"

    with torch.no_grad():
        ntk = NTK(model_params, X_trn=X, A_trn=A, n_classes=n_classes, 
                idx_trn_labeled=idx_labeled, y_trn=y[idx_labeled],
                learning_setting=data_params["learning_setting"],
                pred_method=model_params["pred_method"],
                regularizer=model_params["regularizer"],
                bias=bool(model_params["bias"]),
                solver=model_params["solver"],
                alpha_tol=model_params["alpha_tol"],
                dtype=dtype)
        
        ntk_test = ntk.ntk
        y_pred_trn, _ = ntk(idx_labeled=idx_labeled, idx_test=idx_labeled,
                               y_test=y, X_test=X, A_test=A, return_ntk=True)
        if certificate_params["attack_nodes"] == "train_val_backdoor":
            idx_adv = rng.choice(idx_labeled, 
                                 size=certificate_params["n_adversarial"],
                                 replace=False)
        elif certificate_params["attack_nodes"] == "test_backdoor":
            idx_adv = rng.choice(idx_test, 
                                 size=certificate_params["n_adversarial"],
                                 replace=False)
        else:
            assert False, "Use train_val_backdoor or test_backdoor as attack_nodes for evaluating inductive backdoor"

        delta = certificate_params["delta"]
        if not bool(certificate_params["delta_absolute"]):
            delta = round(delta * 2 * csbm.mu[0].item(), 4)
        logging.info(f"Delta: {delta}")
        y_ub, ntk_ub = ntk.forward_upperbound(idx_labeled, idx_test, idx_adv,
                                              y, X, A, delta,
                                              certificate_params["perturbation_model"],
                                              return_ntk=True,
                                              method=certificate_params["method"])

        y_ub_trn, _ = ntk.forward_upperbound(idx_labeled, idx_labeled, idx_adv,
                                              y, X, A, delta,
                                              certificate_params["perturbation_model"],
                                              return_ntk=True,
                                              method=certificate_params["method"])
        y_lb, ntk_lb = ntk.forward_lowerbound(idx_labeled, idx_test, idx_adv,
                                              y, X, A, delta,
                                              certificate_params["perturbation_model"],
                                              return_ntk=True,
                                              method=certificate_params["method"])
        y_lb_trn, ntk_lb = ntk.forward_lowerbound(idx_labeled, idx_labeled, idx_adv,
                                              y, X, A, delta,
                                              certificate_params["perturbation_model"],
                                              return_ntk=True,
                                              method=certificate_params["method"])
        acc_trn = utils.accuracy(y_pred_trn, y[idx_labeled])
        acc_ub_trn = utils.accuracy(y_ub_trn, y[idx_labeled])
        acc_lb_trn = utils.accuracy(y_lb_trn, y[idx_labeled])
    # Trivial (1-Layer) Evasion Certifcation:
    mask_no_adv_in_n = (A[:, idx_adv] > 0).sum(dim=1) == 0
    A2 = A.matmul(A)
    mask_no_adv_in_n2 = (A2[:, idx_adv] > 0).sum(dim=1) == 0
    mask_no_adv_in_receptive_field = torch.logical_and(mask_no_adv_in_n, mask_no_adv_in_n2)
    acc_cert_trivial = mask_no_adv_in_receptive_field[idx_test].sum().cpu().item() / len(idx_test)
    logging.info(f"Train accuracy: {acc_trn}")
    logging.info(f"Accuracy_lb_trn: {acc_lb_trn}")
    logging.info(f"Accuracy_ub_trn: {acc_ub_trn}")
    logging.info(f"Certified accuracy (evasion, trivial): {acc_cert_trivial}")

    # Poisoning Certificate
    svm_alpha = ntk.svm
    n_original = A.shape[0]
    n_unlabeled = idx_test.shape[0]
    X_original = X.cpu().detach().numpy()
    A_original = A.cpu().detach().numpy()
    y_original = y.cpu().detach().numpy()
    idx_adv_original = idx_adv
    acc_backdoor_l = []
    acc_backdoor_ub_l = []
    acc_backdoor_lb_l = []
    y_pred_backdoor_l = []
    y_true_cls_backdoor_l = []
    is_robust_l = []
    obj_l = [] 
    obj_bd_l = [] 
    opt_status_l = []
    for i in range(n_unlabeled):
        X, A, y = csbm.sample_conditional(n=1, X=X_original,
                                            A=A_original, 
                                            y=y_original)
        X = torch.tensor(X, dtype=dtype, device=device)
        A = torch.tensor(A, dtype=dtype, device=device)
        y = torch.tensor(y, device=device)
        n_inductive = A.shape[0]
        idx_new = np.array([A.shape[0]-1])
        idx_adv = np.concatenate((idx_adv_original, idx_new))
        idx_backdoor = idx_new[0] 
        idx_backdoor = torch.Tensor([idx_backdoor], device=device).to(torch.long)
        with torch.no_grad():
            ntk_bd = NTK(model_params, X_trn=X, A_trn=A, n_classes=n_classes, 
                        idx_trn_labeled=idx_labeled, y_trn=y[idx_labeled],
                        learning_setting=data_params["learning_setting"],
                        pred_method=model_params["pred_method"],
                        regularizer=model_params["regularizer"],
                        bias=bool(model_params["bias"]),
                        solver=model_params["solver"],
                        alpha_tol=model_params["alpha_tol"],
                        dtype=dtype)
            
            y_pred_bd, ntk_test_bd = ntk(idx_labeled=idx_labeled, 
                                         idx_test=idx_backdoor,
                                         y_test=y, X_test=X, A_test=A, 
                                         learning_setting="inductive",
                                         return_ntk=True)
                
            y_bd_ub, ntk_bd_ub = ntk_bd.forward_upperbound(idx_labeled, idx_backdoor, idx_adv,
                                                    y, X, A, delta,
                                                    certificate_params["perturbation_model"],
                                                    return_ntk=True,
                                                    method=certificate_params["method"])

            y_bd_lb, ntk_bd_lb = ntk_bd.forward_lowerbound(idx_labeled, idx_backdoor, idx_adv,
                                                    y, X, A, delta,
                                                    certificate_params["perturbation_model"],
                                                    return_ntk=True,
                                                    method=certificate_params["method"])
                
            acc_bd = utils.accuracy(y_pred_bd, y[idx_backdoor])
            acc_bd_ub = utils.accuracy(y_bd_ub, y[idx_backdoor])
            acc_bd_lb = utils.accuracy(y_bd_lb, y[idx_backdoor])
            acc_backdoor_l.append(acc_bd)
            acc_backdoor_ub_l.append(acc_bd_ub)
            acc_backdoor_lb_l.append(acc_bd_lb)
            y_pred_backdoor_l.append(y_pred_bd.numpy(force=True).tolist()[0])
            y_true_cls_backdoor_l.append((y[idx_backdoor] * 2 - 1).numpy(force=True).tolist()[0])
            
            ntk_inductive = torch.zeros((n_inductive, n_inductive), device=device)
            ntk_inductive[:n_original, :n_original] = ntk.ntk
            ntk_inductive[n_original,:] = ntk_bd.ntk[n_original, :]
            ntk_inductive[:, n_original] = ntk_bd.ntk[:, n_original]

            ntk_inductive_lb = torch.zeros((n_inductive, n_inductive), device=device)
            ntk_inductive_lb[:n_original, :n_original] = ntk_lb
            ntk_inductive_lb[n_original,:] = ntk_bd_lb[n_original, :]
            ntk_inductive_lb[:, n_original] = ntk_bd_lb[:, n_original]

            ntk_inductive_ub = torch.zeros((n_inductive, n_inductive), device=device)
            ntk_inductive_ub[:n_original, :n_original] = ntk_ub
            ntk_inductive_ub[n_original,:] = ntk_bd_ub[n_original, :]
            ntk_inductive_ub[:, n_original] = ntk_bd_ub[:, n_original]

            is_robust, obj, obj_bd, opt_status = \
                utils.certify_robust_bilevel_svm(
                    idx_labeled, idx_backdoor, ntk_inductive, ntk_inductive_lb, ntk_inductive_ub, y, y_pred_bd,
                    svm_alpha, certificate_params, C=model_params["regularizer"], 
                    M=1e3, Mprime=1e3
            )
            is_robust_l.append(is_robust[0])
            obj_l.append(obj[0])
            obj_bd_l.append(obj_bd[0])
            opt_status_l.append(opt_status[0])
    acc_cert = sum(is_robust_l) / n_unlabeled
    acc_cert_u = 0 #not implemented
    logging.info(f"Certified accuracy (poisoning): {acc_cert}")
    acc_backdoor = sum(acc_backdoor_l) / n_unlabeled
    acc_backdoor_ub = sum(acc_backdoor_ub_l) / n_unlabeled
    acc_backdoor_lb = sum(acc_backdoor_lb_l) / n_unlabeled

    # Some Debugging Info
    ntk_labeled = ntk.ntk[idx_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_labeled]
    ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]
    cond = torch.linalg.cond(ntk_labeled)
    ntk_labeled += torch.eye(ntk_labeled.shape[0], dtype=torch.float64).to(device) \
                    * model_params["regularizer"]
    cond_regularized = torch.linalg.cond(ntk_labeled)
    min_ypred = torch.min(torch.tensor(y_pred_backdoor_l)).cpu().item()
    max_ypred = torch.max(torch.tensor(y_pred_backdoor_l)).cpu().item()
    min_ylb = torch.min(y_lb).cpu().item()
    max_ylb = torch.max(y_lb).cpu().item()
    min_yub = torch.min(y_ub).cpu().item()
    max_yub = torch.max(y_ub).cpu().item()
    min_ntklabeled = torch.min(ntk_labeled).cpu().item()
    max_ntklabeled = torch.max(ntk_labeled).cpu().item()
    avg_ntkunlabeled = torch.mean(ntk_unlabeled).cpu().item()
    min_ntkunlabeled = torch.min(ntk_unlabeled).cpu().item()
    max_ntkunlabeled = torch.max(ntk_unlabeled).cpu().item()
    avg_ntklb = torch.mean(ntk_lb).cpu().item()
    min_ntklb = torch.min(ntk_lb).cpu().item()
    max_ntklb = torch.max(ntk_lb).cpu().item()
    avg_ntkub = torch.mean(ntk_ub).cpu().item()
    min_ntkub = torch.min(ntk_ub).cpu().item()
    max_ntkub = torch.max(ntk_ub).cpu().item()

    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()

    return dict(
        # general statistics
        accuracy_test = acc_backdoor,
        accuracy_trn = acc_trn,
        accuracy_ub_test = acc_backdoor_ub,
        accuracy_lb_test = acc_backdoor_lb,
        accuracy_ub_trn = acc_ub_trn,
        accuracy_lb_trn = acc_lb_trn,
        accuracy_backdoor = acc_backdoor,
        accuracy_cert_evasion_trivial = acc_cert_trivial,
        accuracy_cert_pois_robust = acc_cert,
        accuracy_cert_pois_unrobust = acc_cert_u,
        delta_absolute = delta,
        # node-wise pois. robustness statistics
        y_true_cls = y_true_cls_backdoor_l, #(y[idx_test] * 2 - 1).numpy(force=True).tolist(),
        y_pred_logit = y_pred_backdoor_l, # y_pred.numpy(force=True).tolist(),
        y_worst_obj = obj_l,
        y_worst_obj_bound = obj_bd_l,
        y_is_robust = is_robust_l,
        y_opt_status = opt_status_l,
        # split statistics
        idx_train = idx_trn.tolist(),
        idx_val = idx_val.tolist(),
        idx_labeled = idx_labeled.tolist(),
        idx_test = idx_test.tolist(),
        idx_adv = idx_adv.tolist(),
        # data statistics
        csbm_mu = csbm.mu[0].item(),
        csbm_p = csbm.p,
        csbm_q = csbm.q,
        data_dim = X.shape[1],
        # other statistics ntk / pred
        min_ypred = min_ypred,
        max_ypred = max_ypred,
        min_ylb = min_ylb,
        max_ylb = max_ylb,
        min_yub = min_yub,
        max_yub = max_yub,
        avg_ntklb = avg_ntklb,
        min_ntklb = min_ntklb,
        max_ntklb = max_ntklb,
        avg_ntkub = avg_ntkub,
        min_ntkub = min_ntkub,
        max_ntkub = max_ntkub,
        min_ntklabeled = min_ntklabeled,
        max_ntklabeled = max_ntklabeled,
        avg_ntkunlabeled = avg_ntkunlabeled,
        min_ntkunlabeled = min_ntkunlabeled,
        max_ntkunlabeled = max_ntkunlabeled,
        cond = cond.cpu().item(),
        cond_regularized = cond_regularized.cpu().item()
    )