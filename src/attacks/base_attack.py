from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, List, Dict
from jaxtyping import Float, Integer
from typeguard import typechecked

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F

#from torch_sparse import SparseTensor


class GlobalAttack(ABC):
    """Provides possibility to attack a set of nodes."""
    @abstractmethod
    def attack(self, n_perturbations: int, **kwargs) -> Float[torch.Tensor, "n n"]:
        """Create n_perturbations in the graph.
        
        Returns:
            A: n x n perturbed adjacancy matrix.
        """
        pass


# @typechecked
# class Attack(GlobalAttack):
#     """
#     Base class for all attacks providing a uniform interface for all attacks
#     as well as the implementation for all losses proposed or mentioned in our paper.

#     Parameters
#     ----------
#     A : SparseTensor or torch.Tensor
#         [n, n] (sparse) adjacency matrix.
#     X : torch.Tensor
#         [n, d] feature/attribute matrix.
#     y : torch.Tensor
#         Labels vector of shape [n].
#     idx_attack : np.ndarray
#         Indices of the nodes which are to be attacked.
#     model : 
#         Model to be attacked.
#     device : Union[str, int, torch.device]
#         The cuda device to use for the attack
#     data_device : Union[str, int, torch.device]
#         The cuda device to use for storing the dataset.
#         For batched models (like PPRGo) this may differ from the device parameter.
#         Other models require the dataset and model to be on the same device.
#     make_undirected: bool
#         Wether the perturbed adjacency matrix should be made undirected (symmetric degree normalization)
#     binary_attr: bool
#         If true the perturbed attributes are binarized (!=0)
#     loss_type: str
#         The loss to be used by a gradient based attack, can be one of the following loss types:
#             - CW: Carlini-Wagner
#             - LCW: Leaky Carlini-Wagner
#             - Margin: Negative classification margin
#             - tanhMargin: Negative TanH of classification margin
#             - eluMargin: Negative Exponential Linear Unit (ELU) of classification margin
#             - CE: Cross Entropy
#             - MCE: Masked Cross Entropy
#             - NCE: Negative Cross Entropy
#     """

#     def __init__(self,
#                  A: Union[SparseTensor, Float[torch.Tensor, "n n"]],
#                  X: Float[torch.Tensor, "n d"],
#                  y: Integer[torch.Tensor, "n"],
#                  idx_attack: np.ndarray,
#                  model: None,
#                  device: Union[str, int, torch.device],
#                  data_device: Union[str, int, torch.device],
#                  make_undirected: bool,
#                  binary_attr: bool,
#                  loss_type: str = 'CE',  # 'CW', 'LeakyCW'  # 'CE', 'MCE', 'Margin'
#                  **kwargs):
#         assert model is None 

#         self.device = device
#         self.data_device = data_device
#         self.idx_attack = idx_attack
#         self.loss_type = loss_type

#         self.make_undirected = make_undirected
#         self.binary_attr = binary_attr

#         if model is not None:
#             # ToDo implement NTK as model
#             self.attacked_model = deepcopy(model).to(self.device)
#             self.attacked_model.eval()
#             for p in self.attacked_model.parameters():
#                 p.requires_grad = False
#             self.eval_model = self.attacked_model

#         self.y = y.to(torch.long).to(self.device)
#         self.y_attack = self.y[self.idx_attack]
#         self.X = X.to(self.data_device)
#         self.A = A.to(self.data_device)

#         self.X_pert = self.X
#         self.A_pert = self.A

#     @abstractmethod
#     def _attack(self, n_perturbations: int, **kwargs):
#         pass

#     def attack(self, n_perturbations: int, **kwargs):
#         """
#         Executes the attack on the model updating the attributes
#         self.A_pert and self.X_pert accordingly.

#         Parameters
#         ----------
#         n_perturbations : int
#             number of perturbations (attack budget in terms of node additions/deletions) that constrain the atack
#         """
#         if n_perturbations > 0:
#             return self._attack(n_perturbations, **kwargs)
#         else:
#             self.X_pert = self.X
#             self.A_pert = self.A

#     def get_pertubations(self):
#         A_pert, X_pert = self.A_pert, self.X_pert

#         if isinstance(self.A_pert, torch.Tensor):
#             A_pert = SparseTensor.from_dense(self.A_pert)

#         if isinstance(self.X_pert, SparseTensor):
#             X_pert = self.X_pert.to_dense()

#         return A_pert, X_pert

#     @staticmethod
#     @torch.no_grad()
#     def evaluate_global(model,
#                         X: Float[torch.Tensor, "n d"],
#                         A: Union[SparseTensor, Float[torch.Tensor, "n, n"]],
#                         labels: Integer[torch.Tensor, "n"],
#                         eval_idx: Union[List[int], np.ndarray]):
#         """
#         Evaluates any model w.r.t. accuracy for a given (perturbed) adjacency and attribute matrix.
#         """
#         model.eval()
#         if hasattr(model, 'release_cache'):
#             model.release_cache()

#         if type(model) in BATCHED_PPR_MODELS.__args__:
#             pred_logits_target = model.forward(X, A, ppr_idx=np.array(eval_idx))
#         else:
#             pred_logits_target = model(X, A)[eval_idx]

#         acc_test_target = accuracy(pred_logits_target.cpu(), labels.cpu()[eval_idx],
#                                    np.arange(pred_logits_target.shape[0]))

#         return pred_logits_target, acc_test_target

#     def calculate_loss(self, logits, labels):
#         """
#         TODO: maybe add formal definition for all losses? or maybe don't
#         """
#         if self.loss_type == 'CW':
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )
#             loss = -torch.clamp(margin, min=0).mean()
#         elif self.loss_type == 'LCW':
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )
#             loss = -F.leaky_relu(margin, negative_slope=0.1).mean()
#         elif self.loss_type == 'tanhMargin':
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )
#             loss = torch.tanh(-margin).mean()
#         elif self.loss_type == 'Margin':
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )
#             loss = -margin.mean()
#         elif self.loss_type.startswith('tanhMarginCW-'):
#             alpha = float(self.loss_type.split('-')[-1])
#             assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
#             assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )
#             loss = (alpha * torch.tanh(-margin) - (1 - alpha) * torch.clamp(margin, min=0)).mean()
#         elif self.loss_type.startswith('tanhMarginMCE-'):
#             alpha = float(self.loss_type.split('-')[-1])
#             assert alpha >= 0, f'Alpha {alpha} must be greater or equal 0'
#             assert alpha <= 1, f'Alpha {alpha} must be less or equal 1'

#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )

#             not_flipped = logits.argmax(-1) == labels

#             loss = alpha * torch.tanh(-margin).mean() + (1 - alpha) * \
#                 F.cross_entropy(logits[not_flipped], labels[not_flipped])
#         elif self.loss_type == 'eluMargin':
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             margin = (
#                 logits[np.arange(logits.size(0)), labels]
#                 - logits[np.arange(logits.size(0)), best_non_target_class]
#             )
#             loss = -F.elu(margin).mean()
#         elif self.loss_type == 'MCE':
#             not_flipped = logits.argmax(-1) == labels
#             loss = F.cross_entropy(logits[not_flipped], labels[not_flipped])
#         elif self.loss_type == 'NCE':
#             sorted = logits.argsort(-1)
#             best_non_target_class = sorted[sorted != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
#             loss = -F.cross_entropy(logits, best_non_target_class)
#         else:
#             loss = F.cross_entropy(logits, labels)
#         return loss

#     @staticmethod
#     def project(n_perturbations: int, values: torch.Tensor, eps: float = 0, inplace: bool = False):
#         if not inplace:
#             values = values.clone()

#         if torch.clamp(values, 0, 1).sum() > n_perturbations:
#             left = (values - 1).min()
#             right = values.max()
#             miu = Attack.bisection(values, left, right, n_perturbations)
#             values.data.copy_(torch.clamp(
#                 values - miu, min=eps, max=1 - eps
#             ))
#         else:
#             values.data.copy_(torch.clamp(
#                 values, min=eps, max=1 - eps
#             ))
#         return values

#     @staticmethod
#     def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
#         def func(x):
#             return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

#         miu = a
#         for i in range(int(iter_max)):
#             miu = (a + b) / 2
#             # Check if middle point is root
#             if (func(miu) == 0.0):
#                 break
#             # Decide the side to repeat the steps
#             if (func(miu) * func(a) < 0):
#                 b = miu
#             else:
#                 a = miu
#             if ((b - a) <= epsilon):
#                 break
#         return miu


# @typechecked
# class SparseAttack(Attack):
#     """
#     Base class for all sparse attacks.
#     Just like the base attack class but automatically casting the adjacency to sparse format.
#     """

#     def __init__(self,
#                  adj: Union[SparseTensor, Float[torch.Tensor, "n n"], sp.csr_matrix],
#                  **kwargs):

#         if isinstance(adj, torch.Tensor):
#             adj = SparseTensor.from_dense(adj)
#         elif isinstance(adj, sp.csr_matrix):
#             adj = SparseTensor.from_scipy(adj)

#         super().__init__(adj, **kwargs)

#         edge_index_rows, edge_index_cols, edge_weight = adj.coo()
#         self.edge_index = torch.stack([edge_index_rows, edge_index_cols], dim=0).to(self.data_device)
#         self.edge_weight = edge_weight.to(self.data_device)
#         self.n = adj.size(0)
#         self.d = self.X.shape[1]