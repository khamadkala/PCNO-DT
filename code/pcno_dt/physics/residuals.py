from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pcno_dt.physics.differential import (
    cauchy_stress,
    laplacian,
    scalar_spatial_gradient,
    stress_divergence,
    symmetric_strain,
    time_derivative,
    von_mises,
)


@dataclass(frozen=True)
class ResidualOutput:
    reaction_diffusion: Tensor
    mechanics: Tensor
    coupling: Tensor
    stress: Tensor
    von_mises_stress: Tensor


class CoupledResiduals:
    def __init__(self, critical_stress: float, body_force: float) -> None:
        self.critical_stress = critical_stress
        self.body_force = body_force

    def __call__(
        self,
        density: Tensor,
        displacement: Tensor,
        effective_proliferation: Tensor,
        parameters: Tensor,
        coordinates: Tensor,
    ) -> ResidualOutput:
        diffusion = parameters[:, 0:1]
        proliferation = parameters[:, 1:2]
        young_modulus = parameters[:, 2:3]
        poisson_ratio = parameters[:, 3:4]
        while diffusion.ndim < density.ndim:
            diffusion = diffusion.unsqueeze(1)
            proliferation = proliferation.unsqueeze(1)
            young_modulus = young_modulus.unsqueeze(1)
            poisson_ratio = poisson_ratio.unsqueeze(1)
        density_field = density.unsqueeze(-1) if density.ndim == 2 else density
        reaction = effective_proliferation * density_field * (1.0 - density_field)
        rd = time_derivative(density_field, coordinates) - diffusion * laplacian(
            density_field, coordinates
        ) - reaction
        strain = symmetric_strain(displacement, coordinates)
        stress = cauchy_stress(strain, young_modulus, poisson_ratio)
        mechanics = stress_divergence(stress, coordinates)
        mechanics = mechanics + self.body_force * scalar_spatial_gradient(density_field, coordinates)
        equivalent = von_mises(stress).unsqueeze(-1)
        target_proliferation = proliferation * (1.0 - equivalent / self.critical_stress)
        coupling = effective_proliferation - target_proliferation
        return ResidualOutput(
            reaction_diffusion=rd,
            mechanics=mechanics,
            coupling=coupling,
            stress=stress,
            von_mises_stress=equivalent,
        )


def normalized_residual(residual: Tensor, reference: Tensor) -> Tensor:
    numerator = torch.sqrt(torch.mean(residual.square()))
    denominator = torch.sqrt(torch.mean(reference.square())).clamp_min(1e-8)
    return numerator / denominator


def residual_statistics(residual: Tensor) -> dict[str, Tensor]:
    absolute = residual.detach().abs().flatten()
    return {
        "mean": absolute.mean(),
        "median": absolute.median(),
        "p95": torch.quantile(absolute, 0.95),
        "maximum": absolute.max(),
    }

