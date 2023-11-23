from typing import Any, Dict, Tuple

from jaxtyping import Float, Integer
import numpy as np
from numpy import ndarray
import torch

from src.graph_models.csbm import CSBM


def get_graph(
        data_params: Dict[str, Any], sort: bool=True
) -> Tuple[Float[ndarray, "n n"], Integer[ndarray, "n n"], Integer[ndarray, "n"]]:
    """Return graph sampled from a CSBM.

    If sort is true, X, A and y are sorted for class.
    
    Returns X, A, y."""
    n = data_params["n_trn_labeled"] + data_params["n_trn_unlabeled"] \
        + data_params["n_val"] + data_params["n_test"]
    csbm = CSBM(n=n, **data_params)

    X, A, y = csbm.sample(n, data_params["seed"])
    if sort:
        idx = np.argsort(y)
        y = y[idx]
        X = X[idx, :]
        A = A[idx, :]
        A = A[:, idx]
    return X, A, y


def split(
        data_params: Dict[str, Any], y: Integer[ndarray, "n"]
) -> Tuple[ndarray, ndarray, ndarray]:
    """ Split nodes into training, validation and test indices.

    Nodes are split in a class balanced fashion for training and validation
    set. The remaining nodes constitute the test set. 

    TODO: Implement a semi-supervised (compared to fully labeled) setting.

    Returns:
        A tuple (idx_trn, idx_val, idx_test).
    """
    assert data_params["n_trn_unlabeled"] == 0, \
        "Only fully labeled setting implemented so far."
    rng = np.random.Generator(np.random.PCG64(data_params["seed"]))
    n_cls0 = sum(y == 0)
    n = len(y)
    idx_cls0 = rng.permutation(np.arange(n_cls0))
    idx_cls1 = rng.permutation(np.arange(n_cls0, n))
    n_cls = data_params["classes"]
    assert data_params["n_trn_labeled"] % n_cls == 0 \
           and data_params["n_val"] % n_cls == 0, \
           "Unable to create class balanced training and validation split."
    n_labeled = data_params["n_trn_labeled"] / n_cls
    n_val = data_params["n_val"] / n_cls
    idx_trn = np.concatenate((idx_cls0[:n_labeled], idx_cls1[:n_labeled]))
    start_test_id = n_labeled + n_val
    idx_val = np.concatenate((idx_cls0[n_labeled:start_test_id], 
                              idx_cls1[n_labeled:start_test_id]))
    idx_test = np.concatenate((idx_cls0[start_test_id:], 
                               idx_cls1[start_test_id:]))
    return idx_trn, idx_val, idx_test