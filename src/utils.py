from typing import Tuple, Union, Sequence

from jaxtyping import Float, Integer
import numpy as np
import torch
from torch_sparse import coalesce
from sklearn.kernel_ridge import KernelRidge 
from scipy.stats import wasserstein_distance

def accuracy(logits: Float[torch.Tensor, "m"], 
             labels: Integer[torch.Tensor, "n"], 
             split_idx: np.ndarray = None) -> float:
    """Returns the accuracy for a tensor of logits, a list of lables and and a split indices.

    Note: logit is defines as in 
    https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    Returns
    -------
    float
        the Accuracy
    """
    p_cls1 = torch.sigmoid(logits)
    y_pred = (p_cls1 > 0.5).to(dtype=torch.long)
    if split_idx is not None:
        return (y_pred == labels[split_idx]).sum() / len(split_idx)
    else:
        return (y_pred == labels).sum() / len(labels)


def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    """Duplicates indices in edge_index but with flipped row/col indices.
    Furthermore, edge_weights are also duplicated without change.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: symmetric_edge_index, symmetric_edge_weight
    """
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    # Remove duplicate values in symmetric_edge_index
    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def grad_with_checkpoint(outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)

    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()

    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs

def kernel_mmd(k, center=False, norm=False):
    """Computes MMD of kernel as per https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions
    where P and Q are the two blocks of the kernel
    assumes the nodes are sorted according to the class

    center = True does mean zero kernel
    norm = True does normalize the kernel by its Frobenius norm

    Returns:
        Scalar value : MMD of kernel k
    """
    n = k.shape[0]
    m = int(n/2)

    if center:
        k = k-np.mean(k)
    if norm:   
        k = k/np.linalg.norm(k) 

    # Term 1
    c1 = 1 / ( m * (m - 1))
    Kxx = k[:int(n/2),:int(n/2)]
    A = np.sum(Kxx - np.diag(np.diagonal(Kxx)))

    # Term II
    c2 = 1 / (m * (m - 1))
    Kyy = k[int(n/2):,int(n/2):]
    B = np.sum(Kyy - np.diag(np.diagonal(Kyy)))

    # Term III
    c3 = 1 / (m * m)
    C = np.sum(k[:int(n/2),int(n/2):]) + np.sum(k[int(n/2):,:int(n/2)])

    # estimate MMD
    mmd_est = c1 * A + c2 * B - 2 * c3 * C
    return mmd_est

def kernel_w2(k, center=False, norm=False):
    """Computes wasserstein distance between P and Q 
    where P and Q are the two blocks of the kernel
    assumes the nodes are sorted according to the class

    center = True does mean zero kernel
    norm = True does normalize the kernel by its Frobenius norm

    Returns:
        Scalar value : Wasserstein distance of kernel k
    """
    n = k.shape[0]
    if center:
        k = k-np.mean(k)
    if norm:   
        k = k/np.linalg.norm(k) 

    u1 = k[:int(n/2),:int(n/2)].reshape(-1)
    u2 = k[int(n/2):,int(n/2):].reshape(-1)
    u = np.concatenate((u1,u2), axis=0)
    
    v1 = k[:int(n/2),int(n/2):].reshape(-1)
    v2 = k[int(n/2):,:int(n/2)].reshape(-1)
    v = np.concatenate((v1,v2), axis=0)
    
    return wasserstein_distance(u, v)

def block_diff(k, center=False, norm=False):
    """Computes block difference of the kernel using the average of P - Q 
    where P and Q are the two blocks of the kernel
    assumes the nodes are sorted according to the class

    center = True does mean zero kernel
    norm = True does normalize the kernel by its Frobenius norm

    Returns:
        Scalar value : block difference of kernel k
    """
    n = k.shape[0]
    if center:
        k = k-np.mean(k)
    if norm:   
        k = k/np.linalg.norm(k) 
        
    diff = np.mean(k[:int(n/2),:int(n/2)]-k[:int(n/2),int(n/2):] + 
                                     k[int(n/2):,int(n/2):]-k[int(n/2):,:int(n/2)])
    return diff

def class_separability(k, center=False, norm=False):
    """Computes the statistical class separability of the kernel 
    using the (average of P - Q)/(average of P + Q)
    where P and Q are the two blocks of the kernel
    assumes the nodes are sorted according to the class

    center = True does mean zero kernel
    norm = True does normalize the kernel by its Frobenius norm

    Returns:
        Scalar value : class separability of kernel k
    """
    n = k.shape[0]
    if center:
        k = k-np.mean(k)
    if norm:   
        k = k/np.linalg.norm(k)
        
    diff = np.mean(k[:int(n/2),:int(n/2)]-k[:int(n/2),int(n/2):] + 
                                     k[int(n/2):,int(n/2):]-k[int(n/2):,:int(n/2)])
    sum = np.mean(k[:int(n/2),:int(n/2)]+k[:int(n/2),int(n/2):] + 
                                     k[int(n/2):,int(n/2):]+k[int(n/2):,:int(n/2)])
    return diff/sum

def kernel_alignment(k1, k2):
    """Computes the alignment between kernels k1 and k2 
    tr(k1 k2^T)/\sqrt{tr(k1 k1^T) . tr(k2 k2^T)}

    Returns:
        Scalar value : kernel alignment between k1 and k2
    """
    k1 = k1.float()
    k2 = k2.float()
    return torch.trace(k1 @ k2.T)/torch.sqrt((torch.trace(k1 @ k1.T) * torch.trace(k2 @ k2.T)))


def kernel_cosine_sim(k1, k2):
    """Sanity check for kernel_alignment -- kernel_cosine_sim should effectively do the same
    
    Returns:
        Scalar value : kernel cosine sim between k1 and k2
    """
    k1 = k1.float()
    k2 = k2.float()
    k11 = torch.flatten(k1).reshape(1,-1)
    k22 = torch.flatten(k2).reshape(1,-1)
    cos = torch.nn.CosineSimilarity()
    return cos(k11, k22)

def sym_norm_kernel(S):
    """ Symmetric normalize any matrix S
    D^{-0.5} S D^{-0.5} where D is similar to the degree matrix 

    Returns:
        Scalar value : symmetric normalized matrix of S
    """
    Deg_inv = torch.diag(torch.pow(S.sum(axis=1), - 0.5))
    return Deg_inv @ S @ Deg_inv