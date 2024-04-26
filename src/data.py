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
try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    pass

from src.graph_models.csbm import CSBM
from src import globals


def get_csbm(
        specification: Dict[str, Any]
) -> Tuple[Float[ndarray, "n n"], Integer[ndarray, "n n"], Integer[ndarray, "n"]]:
    n = specification["n_trn_labeled"] + specification["n_trn_unlabeled"] \
        + specification["n_val"] + specification["n_test"]
    csbm = CSBM(n=n, **specification)
    logging.info(f"CSBM(p={csbm.p:.05f}, q={csbm.q:.05f})")
    X, A, y = csbm.sample(n, specification["seed"])
    return X, A, y, csbm.mu, csbm.p, csbm.q


def get_planetoid(dataset: str, specification: Dict[str, Any]):
    '''Loads Planetoid datasets from 
    https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid
    '''
    dataset_root = specification["data_dir"]
    data = Planetoid(root = dataset_root, name=dataset)
    X = data.x.numpy()
    y = data.y.numpy()
    idx_features = (X.sum(axis=1) != 0)
    X = X[idx_features, :]
    y = y[idx_features]
    edge_index = data.edge_index
    edge_weight = torch.ones(edge_index.shape[1])
    A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, 
                     sparse_sizes=(edge_index.max()+1, edge_index.max()+1))
    A = A.to_dense().numpy()
    A = A[idx_features, :]
    A = A[:, idx_features]
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
        adj_matrix = sp.csr_matrix((loader['adj_matrix.data'], 
                                    loader['adj_matrix.indices'],
                                    loader['adj_matrix.indptr']), 
                                    shape=loader['adj_matrix.shape'])
        attr_matrix = sp.csr_matrix((loader['attr_matrix.data'], 
                                     loader['attr_matrix.indices'],
                                     loader['attr_matrix.indptr']), 
                                     shape=loader['attr_matrix.shape'])
        A = adj_matrix.toarray()
        X = (attr_matrix.toarray() > 0).astype("float32")
        y = np.array(loader["labels"])
        del loader

    # make undirected
    lt = np.tril(A) == 1
    ut = np.triu(A) == 1
    lt = np.logical_or(lt, lt.T)
    ut = np.logical_or(ut, ut.T)
    A = np.logical_or(lt, ut).astype(np.int64)
    return X, A, y


def get_cora_ml_cont(dataset: str, specification: Dict[str, Any], load_binary_feature: bool = False):
    """Loads cora_ml and makes it undirected."""
    directory = specification["data_dir"]
    if isinstance(directory, str):
        directory = Path(directory)
    path_to_file = directory / (dataset + ".npz")
    with np.load(path_to_file, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], 
                                    loader['adj_indices'],
                                    loader['adj_indptr']), 
                                    shape=loader['adj_shape'])
        if load_binary_feature:
            attr_matrix = sp.csr_matrix((loader['attr_data'], 
                                        loader['attr_indices'],
                                        loader['attr_indptr']), 
                                        shape=loader['attr_shape'])
        else:
            attr_matrix = loader["attr_bert_embedding"]
        A = adj_matrix.toarray()
        if load_binary_feature:
            X = (attr_matrix.toarray() > 0).astype("float32")
        else:
            X = attr_matrix
        y = np.array(loader["labels"])
        del loader

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

    If sort is true, X, A and y are sorted for class (only applied for CSBM!)
    
    Returns X, A, y."""
    if data_params["dataset"] == "csbm":
        X, A, y, mu, p, q = get_csbm(data_params["specification"])
        if sort:
            idx = np.argsort(y)
            y = y[idx]
            X = X[idx, :]
            A = A[idx, :]
            A = A[:, idx]
    elif data_params["dataset"] in ["cora", "citeseer", "pubmed"]:
        X, A, y = get_planetoid(data_params["dataset"], data_params["specification"])
    elif data_params["dataset"] in ["cora_inv"]:
        X, A, y = get_planetoid("cora", data_params["specification"])
        X = (X - 1) * (-1)
    elif data_params["dataset"] == "wikics":
        X, A, y = get_wikics(data_params["specification"])
    elif data_params["dataset"] == "cora_ml":
        X, A, y = get_cora_ml(data_params["specification"])
    elif data_params["dataset"] in ["cora_ml_cont", "dblp", "cora_cont", "cora_full", "cora_ml_cont_binary"]:
        dataset = data_params["dataset"]
        if dataset.endswith("_binary"):
            dataset = dataset[:-7]
            X, A, y = get_cora_ml_cont(dataset, data_params["specification"], load_binary_feature = True)
        else:
            X, A, y = get_cora_ml_cont(dataset, data_params["specification"], load_binary_feature = False)
    if data_params["dataset"] in ["citeseer", "wikics", "cora_ml"]:
        G = nx.from_numpy_array(A)
        idx_lcc = list(max(nx.connected_components(G), key=len))
        X = X[idx_lcc, :]
        A = A[idx_lcc, :]
        A = A[:, idx_lcc]
        y = y[idx_lcc]
    # Log statistics, could remove
    if globals.debug:
        X_rowsum = X.sum(axis=1)
        logging.info(f"X.min(): {X.min()}")
        logging.info(f"X.max(): {X.max()}")
        logging.info(f"X_rowsum.mean(): {X_rowsum.mean()}")
        logging.info(f"X_rowsum.median(): {np.median(X_rowsum)}")
        logging.info(f"X_rowsum.min(): {np.min(X_rowsum)}")
        logging.info(f"X_rowsum.max(): {np.max(X_rowsum)}")
    if data_params["dataset"] == "csbm":
        return X, A, y, mu, p, q
    return X, A, y, None, None, None


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
    if seed is None:
        seed = 0
    rng = np.random.Generator(np.random.PCG64(seed))
    nc = labels.max() + 1
    if balance_test:
    # compute n_per_class
        bins = np.bincount(labels)
        n_test_per_class = np.ceil(fraction_test*bins)
    else: 
        n_test_per_class = np.ones(nc)*n_per_class

    split_labeled, split_val, split_test = [], [], []
    for label in range(nc):
        perm = rng.permutation((labels == label).nonzero()[0])
        split_labeled.append(perm[:n_per_class])
        split_val.append(perm[n_per_class: 2 * n_per_class])
        split_test.append(perm[2*n_per_class: 2 * n_per_class + n_test_per_class[label].astype(int)])

    split_labeled = rng.permutation(np.concatenate(split_labeled))
    split_val = rng.permutation(np.concatenate(split_val))
    split_test = rng.permutation(np.concatenate(split_test))
    

    assert split_labeled.shape[0] == split_val.shape[0] == n_per_class * nc

    split_unlabeled = np.setdiff1d(np.arange(len(labels)), 
                                   np.concatenate((split_labeled, split_val, 
                                                   split_test)))

    logging.info(f'number of samples\n - labeled: {n_per_class * nc} \n' 
                 f' - val: {n_per_class * nc} \n'
                 f' - test: {split_test.shape[0]} \n'
                 f' - unlabeled: {split_unlabeled.shape[0]}')

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

def get_bert_embeddings(text):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Tokenize and encode text using batch_encode_plus
    # The function returns a dictionary containing the token IDs and attention masks
    encoding = tokenizer.batch_encode_plus(
        text,				        # List of input texts
        padding=True,			    # Pad to the maximum sequence length
        truncation=True,		    # Truncate to the maximum sequence length if necessary
        return_tensors='pt',	    # Return PyTorch tensors
        add_special_tokens=True     # Add special tokens CLS and SEP
    )

    input_ids = encoding['input_ids'] # Token IDs
    attention_mask = encoding['attention_mask'] # Attention mask

    # Generate embeddings using BERT model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state # This contains the embeddings
    print(f"Shape of Word Embeddings: {word_embeddings.shape}")

    # Compute the average of word embeddings to get the sentence embedding
    sentence_embedding = word_embeddings.mean(dim=1) # Average pooling along the sequence length dimension
    print(f"Shape of Sentence Embedding: {sentence_embedding.shape}")
    return sentence_embedding
    
