from typing import Union

from jaxtyping import Float
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from src.models.common import process_adj

class GCN(torch.nn.Module):
    """ Two layer GCN for sparse computation.
    """
    def __init__(self, n_features, n_classes, n_filter=64, dropout=0, bias=True,
                 activation="relu", **kwargs):
        super().__init__()
        self.conv1 = GCNConv(n_features, n_filter, normalize=True, bias=bias)
        if n_classes == 2:
            n_classes = 1
            self.pred_layer = nn.Sigmoid()
        else:
            self.pred_layer = nn.Softmax(dim=1)
        self.conv2 = GCNConv(n_filter, n_classes, normalize=True, bias=bias)
        if activation == "relu":
            self.non_linearity = nn.ReLU()
        elif activation == "linear":
            self.non_linearity = nn.Identity()
        else:
            raise ValueError(f"Activation {activation} not supported.")
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, 
                X: Float[torch.Tensor, "n d"] = None, 
                A: Union[Float[torch.Tensor, "n n"], SparseTensor] = None):
        edge_idx, edge_weight = process_adj(A)

        x = self.conv1(X, edge_idx, edge_weight)
        x = self.non_linearity(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_idx, edge_weight)
        #x = self.pred_layer(x)
        
        return x
