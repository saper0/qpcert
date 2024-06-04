import copy
from typing import Any, Dict

import numpy as np
import torch
from jaxtyping import Float, Integer

from src.attacks.apgd import APGD
from src.attacks.base_attack import Attack


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
    else:
        return NotImplementedError("Requested Attack not implemented.")


__all__ = [create_attack, Attack]
