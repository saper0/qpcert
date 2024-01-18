from typing import Tuple, Union, Sequence

from jaxtyping import Float, Integer
import numpy as np
import torch
from torch_sparse import coalesce


def empty_gpu_memory(device: Union[str, torch.device]):
    if torch.cuda.is_available() and device.type != "cpu":
        torch.cuda.empty_cache()


def certify_robust(
        y_pred: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_ub: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_lb: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]]
    ) -> float:
    if len(y_pred.shape) > 1:
        n = y_pred.shape[0]
        pred_orig = y_pred.argmax(1)
        pred_orig_lb = y_lb[range(n), pred_orig]
        mask = torch.ones(y_ub.shape, dtype=torch.bool).to(y_ub.device)
        mask[range(n), pred_orig] = False
        pred_other_ub = y_ub[mask].reshape((n, y_ub.shape[1]-1))
        count_beaten = (pred_other_ub > pred_orig_lb.reshape(-1, 1)).sum(dim=1)
        return ((count_beaten == 0).sum() / n).cpu().item()
    else:
        mask_neg = y_pred < 0
        n_cert = (y_ub[mask_neg] < 0).sum()
        mask_pos = y_pred >= 0
        n_cert += (y_lb[mask_pos] >= 0).sum()
        return (n_cert / y_pred.shape[0]).cpu().item()


def certify_unrobust(
        y_pred: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_ub: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
        y_lb: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]]
    ) -> float:
    if len(y_pred.shape) > 1:
        n = y_pred.shape[0]
        pred_orig = y_pred.argmax(1)
        pred_orig_ub = y_ub[range(n), pred_orig]
        mask = torch.ones(y_lb.shape, dtype=torch.bool).to(y_lb.device)
        mask[range(n), pred_orig] = False
        pred_other_lb = y_lb[mask].reshape((n, y_lb.shape[1]-1))
        count_beaten = (pred_other_lb > pred_orig_ub.reshape(-1, 1)).sum(dim=1)
        return ((count_beaten > 0).sum() / n).cpu().item()
    else:
        mask_neg = y_pred < 0
        n_cert = (y_lb[mask_neg] >= 0).sum()
        mask_pos = y_pred >= 0
        n_cert += (y_ub[mask_pos] < 0).sum()
        return (n_cert / y_pred.shape[0]).cpu().item()


def accuracy(logits: Union[Float[torch.Tensor, "n"], Float[torch.Tensor, "n c"]],
             labels: Integer[torch.Tensor, "n"], 
             idx_labels: np.ndarray = None) -> float:
    """Returns the accuracy for a tensor of logits, a list of lables and and a split indices.

    Works for binary and multi-class classification.

    Returns
    -------
    float
        the Accuracy
    """
    if len(logits.shape) > 1:
        if idx_labels is not None:
            return (logits.argmax(1) == labels[idx_labels]).float().mean().cpu().item()
        else:
            return (logits.argmax(1) == labels).float().mean().cpu().item()
    else:
        logits = logits.reshape(-1,)
        p_cls1 = torch.sigmoid(logits)
        y_pred = (p_cls1 > 0.5).to(dtype=torch.long)
        if idx_labels is not None:
            return ((y_pred == labels[idx_labels]).sum() / len(idx_labels)).cpu().item()
        else:
            return ((y_pred == labels).sum() / len(labels)).cpu().item()


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
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, ...]:
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