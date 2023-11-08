from jaxtyping import Float
import torch

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