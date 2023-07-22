from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import torch
from jaxtyping import Float, Integer
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

from src.graph_models.csbm import CSBM
from src.models.ntk import NTK
from src.attacks import create_attack

def configure_hardware(device, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Hardware
    #torch.backends.cuda.matmul.allow_tf32 = other_params["allow_tf32"]
    #torch.backends.cudnn.allow_tf32 = other_params["allow_tf32"]
    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}")

    return device


def get_graph(data_dict: Dict[str, Any], seed: int,  sort: bool=True):
    """Return graph sampled from a CSBM.

    If sort is true, X, A and y are sorted for class.
    
    Returns X, A, y."""
    csbm = CSBM(**data_dict)
    X, A, y = csbm.sample(data_dict["n"], seed)
    if sort:
        idx = np.argsort(y)
        y = y[idx]
        X = X[idx, :]
        A = A[idx, :]
        A = A[:, idx]
    return X, A, y


def row_normalize(A):
    # Row normalize
    S = torch.triu(A, diagonal=1) + torch.triu(A, diagonal=1).T
    S.data[torch.arange(S.shape[0]), torch.arange(S.shape[0])] = 1
    Deg_inv = torch.diag(torch.pow(S.sum(axis=1), - 1))
    return Deg_inv @ S


def tbn_normalize(A: Float[torch.Tensor, "n n"]):
    #S = torch.triu(A, diagonal=1) + torch.triu(A, diagonal=1).T
    #S.data[torch.arange(S.shape[0]), torch.arange(S.shape[0])] = 1
    A_ = A + torch.eye(A.shape[0])
    deg = A_.sum(axis=0)
    A_x_deg = torch.einsum("ij,j -> ij", A_, deg)
    norm = 1 / torch.sum(A_x_deg, axis=1) #torch.einsum("ij->i", A_x_deg)
    S = torch.einsum("ij, i -> ij", A_x_deg, norm)
    return S


def degree_scaling(A: Float[torch.Tensor, "n n"], gamma: float=3, delta: float=0):
    A_ = A + torch.eye(A.shape[0])
    deg = A_.sum(axis=0).view(A.shape[0], 1)
    deg_num = deg - delta
    deg_den = deg + gamma
    weight = (deg_num / deg_den).view(-1)
    A_x_weight = torch.einsum("ij,j -> ij", A_, weight)
    norm = 1 / torch.sum(A_x_weight, axis=1)
    S = torch.einsum("ij, i -> ij", A_x_weight, norm)
    return S


def get_diffusion(X: torch.Tensor, A: torch.Tensor, model_dict):
    if model_dict["model"] == "GCN":
        if model_dict["normalization"] == "row_normalization":
            return row_normalize(A)
        elif model_dict["normalization"] == "trust_biggest_neighbor":
            return tbn_normalize(A)
        elif model_dict["normalization"] == "degree_scaling":
            return degree_scaling(A, model_dict["gamma"], model_dict["delta"])
        else:
            raise NotImplementedError("Only row normalization for GCN implemented")
    elif model_dict["model"] == "SoftMedoid":
        # Row normalized implementation
        n = X.shape[0]
        d = X.shape[1]
        X_view = X.view((1, n, d))
        dist = torch.cdist(X_view, X_view, p=2).view(n, n)
        A_self = A + torch.eye(n)
        S = torch.exp(- (1 / model_dict["T"]) * (A_self @ dist))
        normalization = torch.einsum("ij,ij->i", A_self, S)
        S = (S*A_self) / normalization[:, None]
        return S
    else:
        raise NotImplementedError("Only GCN architecture implemented")
    

def count_edges_for_idx(A: Float[torch.Tensor, "n n"], idx: np.ndarray):
    '''count edges connected to nodes in idx'''
    row, col = A.triu().to_sparse().indices()

    mapping = torch.zeros(A.size(dim=0)).bool()
    mapping[idx]=True # True if node in idx

    mask_col = mapping[col] # True if col in idx
    mask_row = mapping[row] # True if row in idx
    mask_row_col = torch.logical_or(mask_col, mask_row) # True if either row or col in idx -> edges connected to idx
    return mask_row_col.sum().cpu().item()

