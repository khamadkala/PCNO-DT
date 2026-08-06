from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from pcno_dt.models.operator import OperatorOutput
from pcno_dt.physics.residuals import CoupledResiduals, ResidualOutput


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    data: Tensor
    segmentation: Tensor
    reaction_diffusion: Tensor
    mechanics: Tensor
    coupling: Tensor


class CompositePhysicsLoss(nn.Module):
    def __init__(
        self,
        residuals: CoupledResiduals,
        residual_weights: tuple[float, float, float],
        segmentation_weight: float,
        density_threshold: float,
    ) -> None:
        super().__init__()
        self.residuals = residuals
        self.register_buffer("weights", torch.tensor(residual_weights, dtype=torch.float32))
        self.segmentation_weight = segmentation_weight
        self.density_threshold = density_threshold

    def forward(
        self,
        output: OperatorOutput,
        target_density: Tensor,
        target_mask: Tensor,
        coordinates: Tensor,
    ) -> tuple[LossOutput, ResidualOutput]:
        target = target_density.view_as(output.density)
        data_loss = functional.mse_loss(output.density, target)
        probabilities = torch.sigmoid((output.density - self.density_threshold) * 12.0)
        mask = target_mask.view_as(probabilities).float()
        intersection = (probabilities * mask).sum()
        denominator = probabilities.sum() + mask.sum()
        segmentation_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        residual = self.residuals(
            output.density,
            output.displacement,
            output.effective_proliferation,
            output.parameters,
            coordinates,
        )
        rd_loss = residual.reaction_diffusion.square().mean()
        mechanics_loss = residual.mechanics.square().mean()
        coupling_loss = residual.coupling.square().mean()
        total = data_loss + self.segmentation_weight * segmentation_loss
        total = total + self.weights[0] * rd_loss
        total = total + self.weights[1] * mechanics_loss
        total = total + self.weights[2] * coupling_loss
        return (
            LossOutput(
                total=total,
                data=data_loss,
                segmentation=segmentation_loss,
                reaction_diffusion=rd_loss,
                mechanics=mechanics_loss,
                coupling=coupling_loss,
            ),
            residual,
        )

    @torch.no_grad()
    def update_weights(self, traces: Tensor) -> None:
        if traces.shape != (4,):
            raise ValueError("NTK traces must contain data and three residual terms")
        safe = traces.clamp_min(1e-12)
        target = safe.mean()
        updated = target / safe[1:]
        updated = updated / updated.mean()
        self.weights.copy_(updated.clamp(0.01, 100.0))


def estimate_gradient_trace(loss: Tensor, parameters: list[nn.Parameter]) -> Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    terms = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return torch.zeros((), device=loss.device)
    return torch.stack(terms).sum()
