import logging
from typing import Any, Dict, Union

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import seml
import torch

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
        learning_setting = "inductive"
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
    )

    train_params = dict(
        lr=1e-2,
        weight_decay=1e-3,
        patience=300,
        max_epochs=3000,
        inductive=True
    )

    attack_params = dict(
        attack = "random"
    )

    verbosity_params = dict(
        display_steps = 100,
        debug_lvl = "info"
    )  

    other_params = dict(
        device = "0",
        dtype = torch.float32
    )


def set_debug_lvl(debug_lvl: str):
    if debug_lvl is not None and isinstance(debug_lvl, str):
        logger = logging.getLogger()
        if debug_lvl.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_lvl.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_lvl.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_lvl.lower() == "error":
            logger.setLevel(logging.ERROR)


def log_configuration(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                      train_params: Dict[str, Any],
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
    logging.info(f"train_params: {train_params}")
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
        train_params: Dict[str, Any], 
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        _run: Run):
    device = setup_experiment(data_params, model_params, train_params, 
                              verbosity_params, other_params, seed)

    X, A, y = get_graph(data_params, sort=True)
    idx_trn, idx_val, idx_test = split(data_params, y)
    X = torch.tensor(X, dtype=other_params["dtype"], device=device)
    A = torch.tensor(A, dtype=other_params["dtype"], device=device)
    y = torch.tensor(y, device=device)

    ntk = NTK(model_params, X, A)
    with torch.no_grad():