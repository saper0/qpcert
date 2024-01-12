# Experiment file to perform classification using the NTK.

from collections import Counter
import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
from sklearn.model_selection import StratifiedKFold
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
        dataset = "csbm",
        learning_setting = "inductive", # or "transdructive"
        cv_folds = 4,
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
        regularizer = 1e-8,
        pred_method = "krr",
        solver = "LU"
    )

    verbosity_params = dict(
        debug_lvl = "info"
    )  

    other_params = dict(
        device = "gpu",
        dtype = torch.float64,
        allow_tf32 = False,
        enable_gradient = False
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
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
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
                     verbosity_params: Dict[str, Any], 
                     other_params: Dict[str, Any], seed: int
) -> Union[torch.device, str]:
    """Set general configuration for the seml experiment and configure hardware.
    
    Returns the device.
    """
    set_debug_lvl(verbosity_params["debug_lvl"])
    log_configuration(data_params, model_params, verbosity_params,
                      other_params, seed)
    return configure_hardware(other_params, seed)


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device = setup_experiment(data_params, model_params, verbosity_params, 
                              other_params, seed)
    
    X, A, y = get_graph(data_params, sort=True)
    utils.empty_gpu_memory(device)
    idx_trn, idx_unlabeled, idx_val, idx_test = split(data_params, y)
    X = torch.tensor(X, dtype=other_params["dtype"], device=device)
    A = torch.tensor(A, dtype=other_params["dtype"], device=device)
    y = torch.tensor(y, device=device)
    n_classes = int(y.max() + 1)

    idx_labeled = np.concatenate((idx_trn, idx_val)) 
    split_seed = seed
    if "seed" in data_params["specification"]:
        split_seed = data_params["specification"]["seed"]
    skf = StratifiedKFold(n_splits=data_params["cv_folds"], 
                          shuffle=True, random_state=split_seed)
    with torch.no_grad():
        val_acc_l = []
        cond_l = []
        min_ypred_l = []
        max_ypred_l = []
        min_ntklabeled_l = []
        max_ntklabeled_l = []
        min_ntkunlabeled_l = []
        max_ntkunlabeled_l = []
        # k-fold cross validation
        for trn_split, val_split in skf.split(np.zeros(len(idx_labeled)), 
                                              y=y[idx_labeled].cpu().numpy()):
            idx_trn_split = idx_labeled[trn_split]
            idx_val_split = idx_labeled[val_split]
            idx_known = np.concatenate((idx_trn_split, idx_unlabeled))
            idx_known_labeled = np.array([i for i in range(len(idx_trn_split))]) #actually is just 0 to len(idx_labeled)
            if data_params["learning_setting"] == "transductive":
                A_trn = A
                X_trn = X
            else:
                A_trn = A[idx_known, :]
                A_trn = A_trn[:, idx_known]
                X_trn = X[idx_known, :]
            ntk = NTK(model_params, X_trn=X_trn, A_trn=A_trn, n_classes=n_classes, 
                    idx_trn_labeled=idx_known_labeled, y_trn=y[idx_trn_split],
                    learning_setting=data_params["learning_setting"],
                    pred_method=model_params["pred_method"],
                    regularizer=model_params["regularizer"],
                    dtype=other_params["dtype"])
            
            y_pred, ntk_test = ntk(idx_labeled=idx_trn_split, 
                                   idx_test=idx_val_split,
                                   y_test=y, X_test=X, A_test=A, 
                                   return_ntk=True)
            val_acc = utils.accuracy(y_pred, y[idx_val_split])
            val_acc_l.append(val_acc)
            # Some Statistics
            ntk_labeled = ntk.ntk[idx_known_labeled, :]
            ntk_labeled = ntk_labeled[:, idx_known_labeled]
            ntk_labeled += torch.eye(ntk_labeled.shape[0], dtype=torch.float64).to(device) \
                            * model_params["regularizer"]
            ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]
            cond_l.append(torch.linalg.cond(ntk_labeled).cpu().item())
            min_ypred_l.append(torch.min(y_pred).cpu().item())
            max_ypred_l.append(torch.max(y_pred).cpu().item())
            min_ntklabeled_l.append(torch.min(ntk_labeled).cpu().item())
            max_ntklabeled_l.append(torch.max(ntk_labeled).cpu().item())
            min_ntkunlabeled_l.append(torch.min(ntk_unlabeled).cpu().item())
            max_ntkunlabeled_l.append(torch.max(ntk_unlabeled).cpu().item())
            # free memory
            del ntk
            del ntk_labeled
            del ntk_test
            del y_pred
            utils.empty_gpu_memory(device)
        # Test
        idx_known = np.concatenate((idx_trn, idx_unlabeled))
        idx_known_labeled = np.array([i for i in range(len(idx_trn))]) #actually is just 0 to len(idx_labeled)
        if data_params["learning_setting"] == "transductive":
            A_trn = A
            X_trn = X
        else:
            A_trn = A[idx_known, :]
            A_trn = A_trn[:, idx_known]
            X_trn = X[idx_known, :]
        ntk = NTK(model_params, X_trn=X_trn, A_trn=A_trn, n_classes=n_classes, 
                idx_trn_labeled=idx_known_labeled, y_trn=y[idx_trn],
                learning_setting=data_params["learning_setting"],
                pred_method=model_params["pred_method"],
                regularizer=model_params["regularizer"],
                dtype=other_params["dtype"])
        
        y_pred, ntk_test = ntk(idx_labeled=idx_trn, idx_test=idx_test,
                               y_test=y, X_test=X, A_test=A, return_ntk=True)
        test_acc = utils.accuracy(y_pred, y[idx_test])
        # Some Statistics
        ntk_labeled = ntk.ntk[idx_known_labeled, :]
        ntk_labeled = ntk_labeled[:, idx_known_labeled]
        ntk_labeled += torch.eye(ntk_labeled.shape[0], dtype=torch.float64).to(device) \
                        * model_params["regularizer"]
        ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]
        cond = torch.linalg.cond(ntk_labeled).cpu().item()
        min_ypred = torch.min(y_pred).cpu().item()
        max_ypred = torch.max(y_pred).cpu().item()
        min_ntklabeled = torch.min(ntk_labeled).cpu().item()
        max_ntklabeled = torch.max(ntk_labeled).cpu().item()
        min_ntkunlabeled = torch.min(ntk_unlabeled).cpu().item()
        max_ntkunlabeled = torch.max(ntk_unlabeled).cpu().item()
    return dict(
        val_acc_l = val_acc_l,
        test_acc = test_acc,
        val_cond = cond_l,
        val_min_ypred = min_ypred_l,
        val_max_ypred = max_ypred_l,
        val_min_ntklabeled = min_ntklabeled_l,
        val_max_ntklabeled = max_ntklabeled_l,
        val_min_ntkunlabeled = min_ntkunlabeled_l,
        val_max_ntkunlabeled = max_ntkunlabeled_l,
        test_min_ypred = min_ypred,
        test_max_ypred = max_ypred,
        test_min_ntklabeled = min_ntklabeled,
        test_max_ntklabeled = max_ntklabeled,
        test_min_ntkunlabeled = min_ntkunlabeled,
        test_max_ntkunlabeled = max_ntkunlabeled,
        test_cond = cond
    )