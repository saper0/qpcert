import copy
from typing import Any, Dict

import numpy as np
import torch
from jaxtyping import Float, Integer

from src.attacks.apgd import APGD
from src.attacks.pgd import PGD
from src.attacks.base_attack import Attack
from src.attacks.base_structure_attack import GlobalStructureAttack
from src.attacks.simple_structure_attacks import Noise, Random
from src.attacks.prbcd import PRBCD


def create_attack(delta: float,
                  attack_params: Dict[str, Any],
                  model_params: Dict[str, Any],
                  X: Float[torch.Tensor, "n n"], 
                  A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"],
                  idx_labeled: Integer[torch.Tensor, "l"],
                  idx_adv: Integer[torch.Tensor, "u"]) -> Attack:
    if attack_params["attack"] == "apgd":
        attack_params = copy.deepcopy(attack_params)
        pert_model = attack_params["perturbation_model"]
        del attack_params["delta"]
        del attack_params["perturbation_model"]
        return APGD(delta, pert_model, X, A, y, idx_labeled, idx_adv, model_params, 
                    **attack_params)
    elif attack_params["attack"] == "pgd":
        attack_params = copy.deepcopy(attack_params)
        pert_model = attack_params["perturbation_model"]
        del attack_params["delta"]
        del attack_params["perturbation_model"]
        return PGD(delta, pert_model, X, A, y, idx_labeled, idx_adv, model_params, 
                    **attack_params)
    else:
        return NotImplementedError("Requested Attack not implemented.")


def create_structure_attack(idx_target: Integer[torch.Tensor, "t"],
                  X: Float[torch.Tensor, "n n"], 
                  A: Float[torch.Tensor, "n n"], 
                  y: Integer[torch.Tensor, "n"],
                  idx_labeled: Integer[np.ndarray, "m"],
                  idx_unlabeled: Integer[np.ndarray, "u"],
                  attack_params: Dict[str, Any], 
                  seed: int=42,
                  model: torch.nn.Module=None,
                  *args,
                  **kwargs) -> GlobalStructureAttack:
    if attack_params["attack"] == "random":
        return Random(idx_target, A, y, seed)
    elif attack_params["attack"] == "noise":
        return Noise(idx_target, A, seed)
    elif attack_params["attack"] == "prbcd":
        return PRBCD(A=A, X=X, y=y, idx_attack=idx_target, 
                     idx_labeled=idx_labeled, idx_unlabeled=idx_unlabeled, 
                     model=model, *args, **kwargs, **attack_params)
    else:
        return NotImplementedError("Requested Attack not implemented.")


__all__ = [create_attack, create_structure_attack, Attack, GlobalStructureAttack]
