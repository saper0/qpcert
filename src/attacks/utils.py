from jaxtyping import Float
import torch

class Projection(torch.nn.Module):
    """
    Helper class to project node features back to feasible domain (i.e.
    perturbation ball).

    ToDo: Implement project only for idx_adv
    """
    def __init__(self, 
                 delta: float,
                 pert_model: str,
                 X: Float[torch.Tensor, "n d"]):
        super().__init__()
        self.pert_model = pert_model
        if pert_model == "linf":
            self.X_lb = X - delta
            self.X_ub = X + delta
        else:
            raise NotImplementedError("Perturbation model not implemented.")
        
    def forward(self, X_pert: Float[torch.Tensor, "n d"]):
        if self.pert_model == "linf":
            mask = X_pert < self.X_lb
            X_pert[mask] = self.X_lb[mask]
            mask = X_pert > self.X_ub
            X_pert[mask] = self.X_ub[mask]