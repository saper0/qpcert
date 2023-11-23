from typing import Union

from jaxtyping import Float
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from common import process_adj

class GCN(torch.nn.Module):
    """ Two layer GCN for sparse computation.
    """
    def __init__(self, n_input, n_classes, n_filter=64, dropout=0, ntk_norm=False):
        super().__init__()
        self.conv1 = GCNConv(n_input, n_filter, normalize=True)
        self.conv2 = GCNConv(n_filter, n_classes, normalize=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.ntk_norm1 = torch.sqrt()

    def forward(self, 
                X: Float[torch.Tensor, "n d"] = None, 
                A: Union[Float[torch.Tensor, "n n"], SparseTensor] = None):
        edge_idx, edge_weight = process_adj(A)

        x = self.conv1(X, edge_idx, edge_weight)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_idx, edge_weight)

        return x
