from typing import Tuple, Union, Sequence

from jaxtyping import Float, Integer
import numpy as np
import torch
from torch_sparse import coalesce

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