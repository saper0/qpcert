from typing import Tuple, Union

from jaxtyping import Float, Num, Integer
import torch
from torch_sparse import SparseTensor


def process_adj(adj: Union[SparseTensor, 
                           Tuple[Integer[torch.Tensor, "2 nnz"], 
                                 Float[torch.Tensor, "nnz"]],
                           Float[torch.Tensor, "n n"]]) \
                        -> Union[Integer[torch.Tensor, "2 nnz"],
                                 Float[torch.Tensor, "nnz"]]:
    """Process (divers) adjacency matrix formats into sparse format.
    
    Adapted from https://github.com/saper0/revisiting_robustness/.

    Returns:
        edge_index ... edge indices (2, |E|)
        edge_weights ... edge weights, tensor with |E| elements.
    """
    edge_weight = None

    if isinstance(adj, tuple):
        edge_index, edge_weight = adj[0], adj[1]
    elif isinstance(adj, SparseTensor):
        edge_idx_rows, edge_idx_cols, edge_weight = adj.coo()
        edge_index = torch.stack([edge_idx_rows, edge_idx_cols], dim=0)
    else:
        if not adj.is_sparse:
            adj = adj.to_sparse()
        edge_index, edge_weight = adj.indices(), adj.values()

    if edge_weight is None:
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float32)

    if edge_weight.dtype != torch.float32:
        edge_weight = edge_weight.float()

    return edge_index, edge_weight


def make_dense(A: Union[Float[torch.Tensor, "n n"],
                        SparseTensor,
                        Tuple[Integer[torch.Tensor, "2 nnz"], 
                                 Float[torch.Tensor, "nnz"]]]
               ) -> Float[torch.Tensor, "n n"]:
    """Return a dense version of a potentially sparse adjacency matrix."""
    if isinstance(A, SparseTensor):
            A = A.to_dense() 
    elif isinstance(A, tuple):
        n, _ = A.shape
        A = torch.sparse_coo_tensor(*A, 2 * [n]).to_dense() 
    return A

def add_self_loop(A):
    A.data[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 1
    return A


def row_normalize(A):
    # Row normalize
    S = torch.triu(A, diagonal=1) + torch.triu(A, diagonal=1).T
    S.data[torch.arange(S.shape[0]), torch.arange(S.shape[0])] = 1
    Deg_inv = torch.diag(torch.pow(S.sum(axis=1), - 1))
    return Deg_inv @ S


def sym_normalize(A):
    # Symmetric normalize
    S = torch.triu(A, diagonal=1) + torch.triu(A, diagonal=1).T
    S.data[torch.arange(S.shape[0]), torch.arange(S.shape[0])] = 1
    Deg_inv = torch.diag(torch.pow(S.sum(axis=1), - 0.5))
    return Deg_inv @ S @ Deg_inv


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


def APPNP_propogation(A: Float[torch.Tensor, "n n"], alpha: float=0.1, iteration: float=10, exact: bool=True):
    S = sym_normalize(A)
    I = torch.eye(A.shape[0], dtype=A.dtype)
    if exact:
        S_upd = alpha* torch.inverse(I- (1-alpha)*S)
    else:
        S_upd = I 
        for i in range(iteration):
            S_upd = (1-alpha)*(S@S_upd) + alpha*I
    return S_upd


def get_diffusion(X: torch.Tensor, A: torch.Tensor, model_dict):
    if model_dict["model"] == "GCN":
        if model_dict["normalization"] == "row_normalization":
            return row_normalize(A)
        elif model_dict["normalization"] == "trust_biggest_neighbor":
            return tbn_normalize(A)
        elif model_dict["normalization"] == "degree_scaling":
            return degree_scaling(A, model_dict["gamma"], model_dict["delta"])
        else:
            raise NotImplementedError("Normalization not supported")
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
        raise NotImplementedError("Only GCN/SoftMedoid architecture implemented")