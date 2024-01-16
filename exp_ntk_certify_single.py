# Collective feature perturbation certificate.
# Experiment chooses r nodes at random from the test nodes and certifies the
# rest.

import logging
from typing import Any, Dict, Union, Tuple

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import seml
import torch

from src import utils
from src.data import get_graph, split
from src.models.ntk import NTK


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
        regularizer = 1e-8,
        pred_method = "krr",
        solver = "LU"
    )

    certificate_params = dict(
        perturbation_model = "l0",
        delta = 5 # l0: local budget = delta * feature_dim
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
    if other_params["dtype"] == "float32":
        other_params["dtype"] = torch.float32
    elif other_params["dtype"] == "float64":
        other_params["dtype"] = torch.float64
    elif type(other_params["dtype"]) is not torch.dtype:
        assert False, "Given dtype not supported."

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
    device = configure_hardware(other_params, seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    return device, rng


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        certificate_params: Dict[str, Any],
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device, rng = setup_experiment(data_params, model_params, certificate_params,
                                   verbosity_params, other_params, seed)
    
    X, A, y = get_graph(data_params, sort=True)
    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()
    idx_trn, idx_unlabeled, idx_val, idx_test = split(data_params, y)
    X = torch.tensor(X, dtype=other_params["dtype"], device=device)
    A = torch.tensor(A, dtype=other_params["dtype"], device=device)
    y = torch.tensor(y, device=device)
    n_classes = int(y.max() + 1)

    idx_labeled = np.concatenate((idx_trn, idx_val)) 
    idx_known = np.concatenate((idx_labeled, idx_unlabeled))
    # idx of labeled nodes in nodes known during training (for semi-supervised)
    idx_known_labeled = np.nonzero(np.isin(idx_known, idx_labeled))[0] #actually is just 0 to len(idx_labeled)
    if data_params["learning_setting"] == "transductive":
        A_trn = A
        X_trn = X
    else:
        A_trn = A[idx_known, :]
        A_trn = A_trn[:, idx_known]
        X_trn = X[idx_known, :]
    with torch.no_grad():
        ntk = NTK(model_params, X_trn=X_trn, A_trn=A_trn, n_classes=n_classes, 
                idx_trn_labeled=idx_known_labeled, y_trn=y[idx_labeled],
                learning_setting=data_params["learning_setting"],
                pred_method=model_params["pred_method"],
                regularizer=model_params["regularizer"],
                dtype=other_params["dtype"])
        
        y_pred, ntk_test = ntk(idx_labeled=idx_labeled, idx_test=idx_test,
                               y_test=y, X_test=X, A_test=A, return_ntk=True)
        acc = utils.accuracy(y_pred, y[idx_test])
        logging.info(f'Accuracy {acc}')
        acc_ub_l = []
        acc_lb_l = []
        acc_cert_all_r_l = []
        acc_cert_all_u_l = []
        cert_target_r_l = []
        cert_target_u_l = []
        for i, idx in enumerate(idx_test):
            idx_adv = [idx]
            y_ub = ntk.forward_upperbound(idx_labeled, idx_test, idx_adv,
                                          y, X, A, certificate_params["delta"],
                                          certificate_params["perturbation_model"],
                                          force_recalculation=True,
                                          return_ntk=False)
            y_lb = ntk.forward_lowerbound(idx_labeled, idx_test, idx_adv,
                                          y, X, A, certificate_params["delta"],
                                          certificate_params["perturbation_model"],
                                          force_recalculation=False,
                                          return_ntk=False)
            acc_ub = utils.accuracy(y_ub, y[idx_test])
            acc_lb = utils.accuracy(y_lb, y[idx_test])
            acc_cert_all_r = utils.certify_robust(y_pred, y_ub, y_lb)
            acc_cert_all_u = utils.certify_unrobust(y_pred, y_ub, y_lb)
            acc_cert_all_r_l.append(acc_cert_all_r)
            acc_cert_all_u_l.append(acc_cert_all_u)
            y_pred_i = y_pred[i, :].reshape(1, -1)
            y_ub_i = y_ub[i, :].reshape(1, -1)
            y_lb_i = y_lb[i, :].reshape(1, -1)
            cert_target_r = utils.certify_robust(y_pred_i, y_ub_i, y_lb_i)
            cert_target_u = utils.certify_unrobust(y_pred_i, y_ub_i, y_lb_i)
            cert_target_r_l.append(cert_target_r)
            cert_target_u_l.append(cert_target_u)
            if i % 10 == 0:
                print(f"Calc {i}")
                print(y_pred_i)
                print(y_ub_i)
                print(y_lb_i)
                print(np.mean(cert_target_r_l))
                print(np.mean(cert_target_u_l))
            utils.empty_gpu_memory(device)

    # Some Debugging Info
    ntk_labeled = ntk.ntk[idx_known_labeled, :]
    ntk_labeled = ntk_labeled[:, idx_known_labeled]
    ntk_labeled += torch.eye(ntk_labeled.shape[0], dtype=torch.float64).to(device) \
                    * model_params["regularizer"]
    ntk_unlabeled = ntk_test[idx_test,:][:,idx_labeled]
    cond = torch.linalg.cond(ntk_labeled)
    min_ypred = torch.min(y_pred).cpu().item()
    max_ypred = torch.max(y_pred).cpu().item()
    min_ntklabeled = torch.min(ntk_labeled).cpu().item()
    max_ntklabeled = torch.max(ntk_labeled).cpu().item()
    min_ntkunlabeled = torch.min(ntk_unlabeled).cpu().item()
    max_ntkunlabeled = torch.max(ntk_unlabeled).cpu().item()

    if torch.cuda.is_available() and other_params["device"] != "cpu":
        torch.cuda.empty_cache()

    return dict(
        accuracy = acc,
        accuracy_ub = acc_ub,
        accuracy_lb = acc_lb,
        accuracy_cert = acc_cert,
        accuracy_cert_unrobust = acc_cert_u,
        min_ypred = min_ypred,
        max_ypred = max_ypred,
        min_ntklabeled = min_ntklabeled,
        max_ntklabeled = max_ntklabeled,
        min_ntkunlabeled = min_ntkunlabeled,
        max_ntkunlabeled = max_ntkunlabeled,
        cond = cond.cpu().item()
    )