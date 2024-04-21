from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from jaxtyping import Float, Integer
import torch

class Attack(ABC):
    """Base class for node feature attacks."""

    @abstractmethod
    def attack(self, idx_target: int, do_logging=True) \
        -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs targeted attack against node with idx idx_target."""
        pass


