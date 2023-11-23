from typing import Any, Dict, Union

from src.models.ntk import NTK
from src.models.gcn import GCN


MODEL_TYPE = Union[NTK, GCN, None]

        
def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Args:
        hyperparams (Dict[str, Any]): Containing the hyperparameters.

    Raises:
        ValueError: If a not implemented model is requested.

    Returns:
        MODEL_TYPE: The created model instance.
    """
    if hyperparams["model"] == "NTK":
        return NTK(**hyperparams)
    if hyperparams["model"] == "GCN":
        return GCN(**hyperparams)
    raise ValueError("Specified model not found.")


__all__ = [NTK,
           create_model,
           MODEL_TYPE]
