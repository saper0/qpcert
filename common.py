from typing import Any, Dict, List

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
    

def count_edges_for_idx(A: Float[torch.Tensor, "n n"], idx: np.ndarray):
    '''count edges connected to nodes in idx'''
    row, col = A.triu().to_sparse().indices()

    mapping = torch.zeros(A.size(dim=0)).bool()
    mapping[idx]=True # True if node in idx

    mask_col = mapping[col] # True if col in idx
    mask_row = mapping[row] # True if row in idx
    mask_row_col = torch.logical_or(mask_col, mask_row) # True if either row or col in idx -> edges connected to idx
    return mask_row_col.sum().cpu().item()


def calc_kernel_means(ntk: NTK, y: Integer[torch.Tensor, "n"],
                      idx_use: np.ndarray = None):
    ntk = ntk.get_ntk()
    idx_sorted = np.sort(idx_use)
    ntk = ntk[idx_sorted, :][:, idx_sorted]
    n = y[idx_sorted].shape[0]
    n_class0 = (y[idx_sorted]==0).sum()
    mask_class0 = torch.zeros((n,n), dtype=torch.bool)
    mask_class0[:n_class0, :n_class0] = True
    mask_class0 = mask_class0.triu(diagonal=1)
    mask_interclass = torch.zeros((n,n), dtype=torch.bool)
    mask_interclass[:n_class0, n_class0:] = True
    mask_class1 = torch.zeros((n,n), dtype=torch.bool)
    mask_class1[n_class0:, n_class0:] = True
    mask_class1 = mask_class1.triu(diagonal=1)
    avg_class0 = ntk[mask_class0].mean().detach().cpu().item()
    std_class0 = ntk[mask_class0].std().detach().cpu().item()
    avg_class1 = ntk[mask_class1].mean().detach().cpu().item()
    std_class1 = ntk[mask_class1].std().detach().cpu().item()
    avg_interclass = ntk[mask_interclass].mean().detach().cpu().item()
    std_interclass = ntk[mask_interclass].std().detach().cpu().item()
    mask_inclass = mask_class0.logical_or(mask_class1)
    avg_inclass = ntk[mask_inclass].mean().detach().cpu().item()
    std_inclass = ntk[mask_inclass].std().detach().cpu().item()
    m = mask_interclass.sum()
    k = mask_inclass.sum()
    # todo: recheck if std_diff calc correct
    std_diff = torch.sqrt(1 / m * std_interclass**2 + 1/k * std_inclass**2) # upper bound to true std due to ignoriance of (negative) correlation term
    std_diff = std_diff.detach().cpu().item()
    return avg_class0, std_class0, avg_class1, std_class1, \
        avg_interclass, std_interclass, avg_inclass, std_inclass, std_diff


def plot_ntk_model_diff(ntk_dict: Dict[str, Any], y: Float[torch.Tensor, "n 1"],
                        eps_l: List[float], idx_use: np.ndarray = None, 
                        plot_title: str="Attack"):
    """ Plot Class Difference for attack strengths eps_l for NTKs collected
        in ntk_dict.
    """
    if idx_use is None:
        n = len(y)
        idx_use = np.array([idx for idx in range(n)])
    # 1 Layer
    color_list = ['r', 
                'tab:green', 
                'b', 
                'lime', 
                'slategrey', 
                'k', 
                "lightsteelblue",
                "antiquewhite",
                ]
    linestyle_list = ['-', '--', ':', '-.']
    fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', color_list))
    x = np.arange(len(eps_l))
    for model_label, ntk_l in ntk_dict.items():
        if model_label.endswith("_acc"):
            continue
        y_val = []
        y_std = []
        for ntk in ntk_l:
            _, _, _, _, avg_interclass, std_interclass, avg_inclass, std_inclass, \
                diff_std = calc_kernel_means(ntk, y, idx_use)
            y_val.append(avg_inclass - avg_interclass)
            y_std.append(diff_std)
        ax.errorbar(x, y_val, yerr=y_std, marker="o", linestyle="-",
                    label=f"{model_label}", capsize=5, linewidth=2.5, markersize=8)
    ax.xaxis.set_ticks(x, minor=False)
    xticks = [f"{eps*100:.0f}%" for eps in eps_l]
    ax.xaxis.set_ticklabels(xticks, fontsize=10, fontweight="normal")
    ax.yaxis.grid()
    ax.xaxis.grid()
    ax.set_title("Attack: " + plot_title, fontweight="normal", fontsize=15)        
    ax.legend()

def plot_ntk_model_acc(ntk_dict: Dict[str, Any], y: Float[torch.Tensor, "n 1"],
                        eps_l: List[float], plot_title: str="Attack"):
    """ Plot Acc for attack strengths eps_l for NTKs collected
        in ntk_dict.
    """
    # 1 Layer
    n_class0 = (y==0).sum()
    color_list = ['r', 
                'tab:green', 
                'b', 
                'lime', 
                'slategrey', 
                'k', 
                "lightsteelblue",
                "antiquewhite",
                ]
    linestyle_list = ['-', '--', ':', '-.']
    fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', color_list))
    x = np.arange(len(eps_l))
    for model_label, acc_l in ntk_dict.items():
        if not model_label.endswith("_acc"):
            continue
        y_val = []
        y_std = []
        for acc in acc_l:
            y_val.append(acc)
            y_std.append(0)
        ax.errorbar(x, y_val, yerr=y_std, marker="o", linestyle="-",
                    label=f"{model_label}", capsize=5, linewidth=2.5, markersize=8)
    ax.xaxis.set_ticks(x, minor=False)
    xticks = [f"{eps*100:.0f}%" for eps in eps_l]
    ax.xaxis.set_ticklabels(xticks, fontsize=10, fontweight="normal")
    ax.yaxis.grid()
    ax.xaxis.grid()
    ax.set_title("Attack: " + plot_title, fontweight="normal", fontsize=15)        
    ax.legend()

