import logging
from pathlib import Path
from typing import Any, Dict, Union, Tuple

from jaxtyping import Float, Integer
import networkx as nx
import numpy as np
from numpy import ndarray
import torch
from torch_geometric.datasets.planetoid import Planetoid
from torch_geometric.datasets.wikics import WikiCS
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import scipy.sparse as sp

from src.graph_models.csbm import CSBM


def get_csbm(
        specification: Dict[str, Any]
) -> Tuple[Float[ndarray, "n n"], Integer[ndarray, "n n"], Integer[ndarray, "n"]]:
    n = specification["n_trn_labeled"] + specification["n_trn_unlabeled"] \
        + specification["n_val"] + specification["n_test"]
    csbm = CSBM(n=n, **specification)
    logging.info(f"CSBM(p={csbm.p}, q={csbm.q})")
    X, A, y = csbm.sample(n, specification["seed"])
    return X, A, y


def get_planetoid(dataset: str, specification: Dict[str, Any]):
    '''Loads Planetoid datasets from 
    https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid
    '''
    make_undirected = specification["make_undirected"]
    dataset_root = specification["data_dir"]
    assert make_undirected == True , "undirected not implemented for cora"
    data = Planetoid(root = dataset_root, name=dataset)
    X = data.x.numpy()
    y = data.y.numpy()
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.shape[1])
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, 
                     sparse_sizes=(edge_index.max()+1, edge_index.max()+1))
    A = A.to_dense().numpy()
    return X, A, y


def get_wikics(specification: Dict[str, Any]):
    """Loads WikiCS dataset from 
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/wikics.html#WikiCS
    """
    wikics = WikiCS(root = specification["data_dir"])
    X = wikics.x.numpy()
    y = wikics.y.numpy()
    edge_index = wikics.edge_index
    edge_weight = torch.ones(edge_index.shape[1])
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, 
                     sparse_sizes=(edge_index.max()+1, edge_index.max()+1))
    A = A.to_dense().numpy()
    return X, A, y


def get_cora_ml(specification: Dict[str, Any]):
    """Loads cora_ml and makes it undirected."""
    directory = specification["data_dir"]
    if isinstance(directory, str):
        directory = Path(directory)
    path_to_file = directory / ("cora_ml.npz")
    with np.load(path_to_file, allow_pickle=True) as loader:
        loader = dict(loader)
        del_entries = []
        # Construct sparse matrices
        for key in loader.keys():
            if key.endswith('.data'):
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = matrix_name + ".indices"
                mat_indptr = matrix_name + ".indptr"
                mat_shape = matrix_name + ".shape"
                M = sp.csr_matrix((loader[mat_data], loader[mat_indices],
                                loader[mat_indptr]), shape=loader[mat_shape])
                if matrix_name == "adj_matrix":
                    A = M.toarray()
                elif matrix_name == "attr_matrix":
                    X = (M.toarray() > 0).astype("float32")
                else:
                    assert False
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])
        # Delete sparse matrix entries
        for del_entry in del_entries:
            del loader[del_entry]
        y = np.array(loader["labels"])

    # make undirected
    lt = np.tril(A) == 1
    ut = np.triu(A) == 1
    lt = np.logical_or(lt, lt.T)
    ut = np.logical_or(ut, ut.T)
    A = np.logical_or(lt, ut).astype(np.int64)
    return X, A, y


def get_graph(
        data_params: Dict[str, Any], sort: bool=True
) -> Tuple[Float[ndarray, "n n"], Integer[ndarray, "n n"], Integer[ndarray, "n"]]:
    """Return graph sampled from a CSBM.

    If sort is true, X, A and y are sorted for class.
    
    Returns X, A, y."""

    if data_params["dataset"] == "csbm":
        X, A, y = get_csbm(data_params["specification"])
    elif data_params["dataset"] in ["cora", "citeseer", "pubmed"]:
        X, A, y = get_planetoid(data_params["dataset"], data_params["specification"])
    elif data_params["dataset"] == "wikics":
        X, A, y = get_wikics(data_params["specification"])
    elif data_params["dataset"] == "cora_ml":
        X, A, y = get_cora_ml(data_params["specification"])
    if data_params["dataset"] in ["citeseer", "wikics", "cora_ml"]:
        G = nx.from_numpy_array(A)
        idx_lcc = list(max(nx.connected_components(G), key=len))
        X = X[idx_lcc, :]
        A = A[idx_lcc, :]
        A = A[:, idx_lcc]
        y = y[idx_lcc]
    if sort:
        idx = np.argsort(y)
        y = y[idx]
        X = X[idx, :]
        A = A[idx, :]
        A = A[:, idx]
    return X, A, y


def split_csbm(
        data_params: Dict[str, Any], y: Integer[ndarray, "n"]
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Split nodes into training, validation and test indices.

    Nodes are split in a class balanced fashion for training and validation
    set. The remaining nodes constitute the test set. 

    TODO: Implement a semi-supervised (compared to fully labeled) setting.

    Returns:
        A tuple (idx_trn, idx_unlabeled, idx_val, idx_test).
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
    n_labeled = int(data_params["n_trn_labeled"] / n_cls)
    n_val = int(data_params["n_val"] / n_cls)
    idx_trn = np.concatenate((idx_cls0[:n_labeled], idx_cls1[:n_labeled]))
    start_test_id = n_labeled + n_val
    idx_val = np.concatenate((idx_cls0[n_labeled:start_test_id], 
                              idx_cls1[n_labeled:start_test_id]))
    idx_test = np.concatenate((idx_cls0[start_test_id:], 
                               idx_cls1[start_test_id:]))
    return idx_trn, np.array([]), idx_val, idx_test


def split_inductive(labels, n_per_class=20, fraction_test=0.1, seed=None, 
                    balance_test = True):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    fraction_test : float
        How much % of nodes are test nodes.
    balance_test: bool
        wether to balance the classes in the test set; if true, take 10% of all nodes as test set
    seed: int
        Seed

    Returns
    -------
    split_labeled: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test: array-like [n_per_class * nc]
        The indices of the test nodes
    split_unlabeled: array-like [num_nodes - 3*n_per_class * nc]
        The indices of the unlabeled nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1
    if balance_test:
    # compute n_per_class
        bins = np.bincount(labels)
        n_test_per_class = np.ceil(fraction_test*bins)
    else: 
        n_test_per_class = np.ones(nc)*n_per_class

    split_labeled, split_val, split_test = [], [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_labeled.append(perm[:n_per_class])
        split_val.append(perm[n_per_class: 2 * n_per_class])
        split_test.append(perm[2*n_per_class: 2 * n_per_class + n_test_per_class[label].astype(int)])

    split_labeled = np.random.permutation(np.concatenate(split_labeled))
    split_val = np.random.permutation(np.concatenate(split_val))
    split_test = np.random.permutation(np.concatenate(split_test))
    

    assert split_labeled.shape[0] == split_val.shape[0] == n_per_class * nc

    split_unlabeled = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_labeled, split_val, split_test)))

    logging.info(f'number of samples\n - labeled: {n_per_class * nc} \n - val: {n_per_class * nc} \n - test: {split_test.shape[0]} \n - unlabeled: {split_unlabeled.shape[0]}')

    return split_labeled, split_unlabeled, split_val, split_test


def split(
        data_params: Dict[str, Any], y: Integer[ndarray, "n"]
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """ Split nodes into training, validation and test indices.

    Returns:
        A tuple (idx_trn, idx_unlabeled, idx_val, idx_test).
    """
    if data_params["dataset"] == "csbm":
        idx_trn, idx_unlabeled, idx_val, idx_test = split_csbm(
            data_params["specification"], y
        )
    else:
        spec = data_params["specification"]
        seed = 0
        if "seed" in spec:
            seed = spec["seed"]
        idx_trn, idx_unlabeled, idx_val, idx_test = split_inductive(
            y, n_per_class=spec["n_per_class"], balance_test=spec["balance_test"], 
            seed=seed
        )

    return idx_trn, idx_unlabeled, idx_val, idx_test
