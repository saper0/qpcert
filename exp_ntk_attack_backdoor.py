import copy
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
from src.attacks import create_attack
from src.data import get_graph, split
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

    attack_params = dict(
        n_adversarial = 10, # number adversarial nodes
        n_attack = -1, #if set to > 0, attack n_attack randomly chosen test_nodes
        method = "XXT",
        perturbation_model = "linf",
        delta = 0.01, # l0: local budget = delta * feature_dim
        delta_absolute = True, # if false interpreted as % of 2*mu
        attack_nodes = "test", # "train", "all"
        normalize_grad = False, # if gradient computation in attack should be normalized, can help escape bad initial local minima
        evasion_attack = False, # if True, additional to poisoning attack attacks evaluated node with an evasion attack
    )

    verbosity_params = dict(
        debug_lvl = "info"
    )  

    other_params = dict(
        device = "gpu",
        dtype = "float64",
        allow_tf32 = False,
        enable_gradient = False,
        max_logging_iters = 10, # For outputting progress of attack to console
        store_attack_curve = True, # If the "learning curve" of the attack should be saved to mongodb
        store_first_iter = 10, # Always save first 100 iterations of attack a particular node
        store_every_X_iter = 50, # Afterwards, save every 20th iteration of attack
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


def prepare_data(data_params: Dict[str, Any], device: torch.device,
                 dtype: torch.dtype):
    if not data_params["learning_setting"] == "transductive":
        assert False, "Only transductive setting supported"
    X, A, y, mu, p, q = get_graph(data_params, sort=True)
    if torch.cuda.is_available() and device.type != "cpu":
        torch.cuda.empty_cache()
    idx_trn, idx_unlabeled, idx_val, idx_test = split(data_params, y)
    if len(idx_unlabeled) != 0:
        idx_test = np.concatenate((idx_unlabeled, idx_test))
    idx_trn = torch.tensor(idx_trn, dtype=torch.long, device=device)
    idx_val = torch.tensor(idx_val, dtype=torch.long, device=device)
    idx_test = torch.tensor(idx_test, dtype=torch.long, device=device)
    X = torch.tensor(X, dtype=dtype, device=device)
    A = torch.tensor(A, dtype=dtype, device=device)
    y = torch.tensor(y, device=device)
    n_classes = int(y.max() + 1)
    return X, A, y, mu, p, q, idx_trn, idx_val, idx_test, n_classes


def get_clean_acc(X, A, n_classes, y, idx_labeled, idx_test, model_params,
                    data_params, dtype):
    """Return clean accuracy of NTK on given idx_test. """
    with torch.no_grad():
        ntk = NTK(model_params, X_trn=X, A_trn=A, n_classes=n_classes, 
                idx_trn_labeled=idx_labeled, y_trn=y[idx_labeled],
                learning_setting=data_params["learning_setting"],
                pred_method=model_params["pred_method"],
                regularizer=model_params["regularizer"],
                bias=bool(model_params["bias"]),
                solver=model_params["solver"],
                alpha_tol=model_params["alpha_tol"],
                dtype=dtype,
                print_alphas=False)
        
        y_pred_clean, _ = ntk(idx_labeled=idx_labeled, idx_test=idx_test,
                            y_test=y, X_test=X, A_test=A, return_ntk=True)
    return y_pred_clean


def prepare_attack(attack_params: Dict[str, Any], idx_test, idx_trn, idx_labeled,
                   rng, mu, device):
    """Return attack target nodes (idx_adv) and attack strength (delta)."""
    # Prepare adversarial nodes
    if attack_params["attack_nodes"] == "test":
        idx_adv = rng.choice(idx_test.numpy(force=True), 
                                size=attack_params["n_adversarial"],
                                replace=False)
    elif attack_params["attack_nodes"] == "train":
        idx_adv = rng.choice(idx_trn.numpy(force=True), 
                                size=attack_params["n_adversarial"],
                                replace=False)
    elif attack_params["attack_nodes"] == "train_val":
        idx_adv = rng.choice(idx_labeled, 
                                size=attack_params["n_adversarial"],
                                replace=False)
    elif attack_params["attack_nodes"] == "all":
        idx_known = np.concatenate((idx_labeled, idx_test)) 
        idx_adv = rng.choice(idx_known.numpy(force=True), 
                                size=attack_params["n_adversarial"],
                                replace=False)
    else:
        assert False, "Choose set of nodes to be attacked!"
    idx_adv = torch.tensor(idx_adv, dtype=torch.long, device=device)
    # Prepeare attack strength
    delta = attack_params["delta"]
    if not bool(attack_params["delta_absolute"]):
        delta = round(delta * 2 * mu[0].item(), 4)
    logging.info(f"Delta: {delta}")
    # Prepare targets
    idx_targets = idx_test
    if "n_attack" in attack_params and attack_params["n_attack"] > -1:
        n = idx_targets.shape[0]
        idx_targets = rng.choice(range(n), 
                                 size=attack_params["n_attack"],
                                 replace=False)
        idx_targets = np.sort(idx_targets)
        idx_targets = idx_test[idx_targets]
    return idx_targets, idx_adv, delta


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        attack_params: Dict[str, Any],
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device, dtype, rng = setup_experiment(data_params, model_params, 
                                          attack_params, verbosity_params, 
                                          other_params, seed)

    X, A, y, mu, p, q, idx_trn, idx_val, idx_test, n_classes = \
        prepare_data(data_params, device, dtype)
    #idx_labeled = np.concatenate((idx_trn, idx_val)) 
    idx_labeled = torch.cat((idx_trn, idx_val)) 

    y_pred_clean = get_clean_acc(X, A, n_classes, y, idx_labeled, idx_test,
                                 model_params, data_params, dtype)
    # idx_targets is idx_test!
    idx_targets, idx_adv, delta = prepare_attack(attack_params, idx_test, 
                                                 idx_trn, idx_labeled,
                                                 rng, mu, device)
    attack_pois = create_attack(delta, attack_params, model_params,
                                X, A, y, idx_trn, idx_labeled, idx_adv)
    
    n_corr = 0
    n_corr_clean = 0
    n_pert = 0
    clean_y_pred_l = []
    pert_y_pred_l = []
    clean_acc_l = []
    robust_acc_l = []
    pert_success_l = []
    y_pert_ll = []
    do_logging = True
    for i, idx_target in enumerate(idx_targets):
        if i > other_params["max_logging_iters"]:
            do_logging = False
        # Perform Poisoning Attack
        idx_target = torch.tensor([idx_target], dtype=torch.long, device=X.device)
        X_pert, y_pert_l, y_pert = attack_pois.attack(idx_target, do_logging)
        if torch.sgn(torch.tensor(y_pert)) == torch.sgn(y_pred_clean[i]):
            # Perform Evasion Attack
            attack_params_new = copy.deepcopy(attack_params)
            attack_params_new["evasion_attack"] = True
            attack_evasion = create_attack(delta, attack_params_new, model_params,
                                        X_pert, A, y, idx_trn, idx_labeled, 
                                        idx_target)
            _, y_pert_l, y_pert = attack_evasion.attack(idx_target, do_logging)
        # Statistics
        y_pert_t = torch.tensor(y_pert, dtype=dtype, device=device)
        acc = utils.accuracy(y_pert_t, y[idx_target])
        acc_clean = utils.accuracy(y_pred_clean[i], y[idx_target])
        pert_success = torch.sign(y_pert_t) != torch.sign(y_pred_clean[i])
        clean_y_pred_l.append(y_pred_clean[i].detach().cpu().item())
        clean_acc_l.append(acc_clean)
        pert_y_pred_l.append(y_pert)
        if bool(acc_clean):
            robust_acc_l.append(acc)
        else:
            robust_acc_l.append(acc_clean)
        pert_success_l.append(int(pert_success.detach().cpu().item()))
        if pert_success:
            n_pert += 1
        n_corr += acc
        n_corr_clean += acc_clean
        logging.info(f"y_pert: {y_pert:.8f}; "
              f"y_clean: {clean_y_pred_l[-1]:.8f}; "
              f"Pert: {pert_success}; Correct Clean: {acc_clean > 0}; Total Pert: {n_pert} "
              f"Total Clean: {n_corr_clean}; Total: {i+1}")
        if do_logging:
            y_pert_ll.append(y_pert_l)
        elif bool(other_params["store_attack_curve"]):
            # store less variables to save space in MongoDB
            y_pert_l_trimmed = []
            for counter, logit in enumerate(y_pert_l):
                if counter < other_params["store_first_iter"]:
                    y_pert_l_trimmed.append(logit)
                elif counter % other_params["store_every_X_iter"] == 0: 
                    y_pert_l_trimmed.append(logit)
            y_pert_ll.append(y_pert_l_trimmed)
    acc = sum(clean_acc_l) / idx_targets.shape[0]
    acc_rob = sum(robust_acc_l) / idx_targets.shape[0]
    pert_success_ratio = sum(pert_success_l) / idx_targets.shape[0]
    logging.info(f"Clean Accuracy: {acc:.2f}")
    logging.info(f"Robust Accuracy: {acc_rob:.2f}")
    logging.info(f"% of Successfull Perturbations: {pert_success_ratio:.2f}")
    
    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()
    
    if mu is None:
        mu = np.array([0])
        p = 0
        q = 0
    return dict(
        # general statistics
        accuracy_test = acc,
        robust_accuracy_test = acc_rob,
        pert_success_ratio = pert_success_ratio,
        delta_absolute = delta,
        # node-wise pois. robustness statistics
        y_true_cls = (y[idx_test] * 2 - 1).numpy(force=True).tolist(),
        y_pred_logit = clean_y_pred_l,
        y_worst_obj = pert_y_pred_l,
        y_is_pert = pert_success_l,
        y_pert_ll = y_pert_ll,
        # split statistics
        idx_train = idx_trn.tolist(),
        idx_val = idx_val.tolist(),
        idx_labeled = idx_labeled.tolist(),
        idx_test = idx_targets.tolist(),
        idx_adv = idx_adv.tolist(),
        # data statistics
        csbm_mu = mu[0].item(),
        csbm_p = p,
        csbm_q = q,
        data_dim = X.shape[1]
    )