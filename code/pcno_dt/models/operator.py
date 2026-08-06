from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pcno_dt.config import ModelConfig
from pcno_dt.models.branch import HemodynamicBranch
from pcno_dt.models.trunk import SpatiotemporalTrunk


@dataclass(frozen=True)
class OperatorOutput:
    density: Tensor
    displacement: Tensor
    effective_proliferation: Tensor
    parameters: Tensor
    coefficients: Tensor
    basis: Tensor


class PCNODigitalTwin(nn.Module):
    parameter_order = (
        "diffusion",
        "proliferation",
        "young_modulus",
        "poisson_ratio",
        "arterial_fraction",
        "portal_fraction",
    )

    def __init__(self, config: ModelConfig, input_channels: int = 7) -> None:
        super().__init__()
        bounds = [config.parameter_bounds[name] for name in self.parameter_order]
        self.branch = HemodynamicBranch(
            input_channels=input_channels,
            stage_channels=config.branch_channels,
            stage_blocks=config.branch_blocks,
            rank=config.rank,
            parameter_bounds=bounds,
        )
        self.trunk = SpatiotemporalTrunk(
            width=config.trunk_width,
            depth=config.trunk_depth,
            rank=config.rank,
            frequencies=config.frequencies,
            harmonics=config.harmonics,
        )
        self.output_bias = nn.Parameter(torch.zeros(()))

    def forward(self, images: Tensor, coordinates: Tensor) -> OperatorOutput:
        coefficients, parameters = self.branch(images)
        basis, displacement, effective_proliferation = self.trunk(coordinates)
        if coordinates.ndim == 2:
            raw = torch.einsum("bp,np->bn", coefficients, basis)
            displacement = displacement.unsqueeze(0).expand(images.shape[0], -1, -1)
            effective_proliferation = effective_proliferation.unsqueeze(0).expand(
                images.shape[0], -1, -1
            )
        elif coordinates.ndim == 3:
            raw = torch.einsum("bp,bnp->bn", coefficients, basis)
        else:
            raise ValueError("coordinates must have shape [N,4] or [B,N,4]")
        density = torch.sigmoid(raw + self.output_bias)
        return OperatorOutput(
            density=density,
            displacement=displacement,
            effective_proliferation=effective_proliferation,
            parameters=parameters,
            coefficients=coefficients,
            basis=basis,
        )

    def freeze_for_clinical_stage(self) -> None:
        self.branch.freeze_early_layers()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
