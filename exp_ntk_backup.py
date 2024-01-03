import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import seml
import torch

from src import utils
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
        learning_setting = "inductive", # or "transdructive"
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
    )
    
    model_params = dict(
        label = "GCN",
        model = "GCN",
        normalization = "row_normalization",
        depth = 1,
        regularizer = 1e-8,
        pred_method = "krr",
        solver = "LU"
    )

    attack_params = dict(
        attack = "random",
        epsilon_list = [0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1, 2.5, 5, 10],
        attack_setting = "evasion" # or "poisioning"
    )

    verbosity_params = dict(
        debug_lvl = "info"
    )  

    other_params = dict(
        device = "gpu",
        dtype = torch.float64,
        allow_tf32 = False
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
                      attack_params: Dict[str, Any],
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
    logging.info(f"attack_params: {attack_params}")
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

    # Hardware
    torch.backends.cuda.matmul.allow_tf32 = other_params["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = other_params["allow_tf32"]

    device = other_params["device"]
    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}")
        logging.info(f"Currently on gpu device {device}")

    return device


def setup_experiment(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                     train_params: Dict[str, Any],
                     verbosity_params: Dict[str, Any], 
                     other_params: Dict[str, Any], seed: int
) -> Union[torch.device, str]:
    """Set general configuration for the seml experiment and configure hardware.
    
    Returns the device.
    """
    set_debug_lvl(verbosity_params["debug_lvl"])
    log_configuration(data_params, model_params, train_params, verbosity_params,
                      other_params, seed)
    return configure_hardware(other_params, seed)


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        attack_params : Dict[str, Any],
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device = setup_experiment(data_params, model_params, attack_params, 
                              verbosity_params, other_params, seed)

    X, A, y = get_graph(data_params, sort=True)
    idx_trn, idx_val, idx_test = split(data_params, y)
    X = torch.tensor(X, dtype=other_params["dtype"], device=device)
    A = torch.tensor(A, dtype=other_params["dtype"], device=device)
    y = torch.tensor(y, device=device)

    idx_labeled = np.concatenate((idx_trn, idx_val)) 
    if data_params["learning_setting"] == "transductive":
        ntk = NTK(model_params, X_trn=X, A_trn=A, y_trn=y[idx_labeled],
                  learning_setting=data_params["learning_setting"],
                  pred_method=model_params["pred_method"],
                  regularizer=model_params["regularizer"],
                  dtype=other_params["dtype"])
    else:
        A_trn = A[idx_labeled, :]
        A_trn = A_trn[:, idx_labeled]
        ntk = NTK(model_params, X_trn=X[idx_labeled, :], A_trn=A_trn, 
                  y_trn=y[idx_labeled],
                  learning_setting=data_params["learning_setting"],
                  pred_method=model_params["pred_method"],
                  regularizer=model_params["regularizer"],
                  dtype=other_params["dtype"])

    if attack_params["attack_setting"] == "poisioning":
        assert data_params["learning_setting"] == "transductive", "Currently "\
             + "only considering poisioing for the transductive setting, need"\
             + " to think more about how to set idx_target otherwise."
        # Thought: Should be idx_val and/or idx_test in idx_target set? What if we even do inductive poisioning? 
        idx_target = np.concatenate((idx_trn, idx_val, idx_test)) 
    else:
        idx_target = idx_test
    attack = create_attack(idx_target, X, A, y, 
                           idx_labeled=idx_labeled, 
                           idx_unlabeled=idx_test, 
                           attack_params=attack_params, 
                           seed=seed)
    if data_params["learning_setting"] == "inductive":
        ntk_clean = NTK(model_params, X_trn=X[idx_labeled, :], A_trn=A_trn, 
                        y_trn=y[idx_labeled],
                        dtype=other_params["dtype"],
                        learning_setting=data_params["learning_setting"],
                        pred_method=model_params["pred_method"],
                        regularizer=model_params["regularizer"])
    else:
        ntk_clean = ntk
    acc_l = []
    min_ypred = []
    max_ypred = []
    min_M = []
    max_M = []
    min_ntklabeled = []
    max_ntklabeled = []
    min_ntkunlabeled = []
    max_ntkunlabeled = []
    for eps in attack_params["epsilon_list"]:
        n_pert = int(round(eps * count_edges_for_idx(A.cpu(), idx_test)))
        A_pert = attack.attack(n_pert).detach().clone()
        y_pred, ntk_test, cond, M, ntk_labeled, ntk_unlabeled = ntk(
            idx_labeled=idx_labeled, idx_unlabeled=idx_test,
                                y_test=y, X_test=X, A_test=A_pert, 
                                return_ntk=True, solution_method=model_params["solver"]
        )
        acc = utils.accuracy(y_pred, y[idx_target]).cpu().item()
        acc_l.append(acc)
        min_ypred.append(torch.min(y_pred).cpu().item())
        max_ypred.append(torch.max(y_pred).cpu().item())
        min_M.append(torch.min(M).cpu().item())
        max_M.append(torch.max(M).cpu().item())
        min_ntklabeled.append(torch.min(ntk_labeled).cpu().item())
        max_ntklabeled.append(torch.max(ntk_labeled).cpu().item())
        min_ntkunlabeled.append(torch.min(ntk_unlabeled).cpu().item())
        max_ntkunlabeled.append(torch.max(ntk_unlabeled).cpu().item())
        logging.info(f'Accuracy {acc} for epsilon {eps}')

        if torch.cuda.is_available() and other_params["device"] != "cpu":
                torch.cuda.empty_cache()
                #torch.cuda.synchronize() # Maybe necessary for custom cuda kernel
    assert len(acc_l) > 0

    return dict(
        accuracy_l = acc_l,
        epsilon_l = attack_params["epsilon_list"],
        min_ypred = min_ypred,
        max_ypred = max_ypred,
        min_M = min_M,
        max_M = max_M,
        min_ntklabeled = min_ntklabeled,
        max_ntklabeled = max_ntklabeled,
        min_ntkunlabeled = min_ntkunlabeled,
        max_ntkunlabeled = max_ntkunlabeled,
        cond = cond.cpu().item()
    )