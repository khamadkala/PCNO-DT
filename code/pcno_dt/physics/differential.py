from __future__ import annotations

import torch
from torch import Tensor


def gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    derivative = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return derivative


def scalar_spatial_gradient(outputs: Tensor, coordinates: Tensor) -> Tensor:
    return gradient(outputs, coordinates)[..., :3]


def time_derivative(outputs: Tensor, coordinates: Tensor) -> Tensor:
    return gradient(outputs, coordinates)[..., 3:4]


def divergence(vector: Tensor, coordinates: Tensor) -> Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("divergence expects a three-dimensional vector")
    terms = []
    for axis in range(3):
        component_gradient = gradient(vector[..., axis], coordinates)
        terms.append(component_gradient[..., axis : axis + 1])
    return torch.stack(terms, dim=0).sum(dim=0)


def laplacian(outputs: Tensor, coordinates: Tensor) -> Tensor:
    first = scalar_spatial_gradient(outputs, coordinates)
    terms = []
    for axis in range(3):
        second = gradient(first[..., axis], coordinates)[..., axis : axis + 1]
        terms.append(second)
    return torch.stack(terms, dim=0).sum(dim=0)


def displacement_jacobian(displacement: Tensor, coordinates: Tensor) -> Tensor:
    rows = []
    for component in range(3):
        rows.append(gradient(displacement[..., component], coordinates)[..., :3])
    return torch.stack(rows, dim=-2)


def symmetric_strain(displacement: Tensor, coordinates: Tensor) -> Tensor:
    jacobian = displacement_jacobian(displacement, coordinates)
    return 0.5 * (jacobian + jacobian.transpose(-1, -2))


def lame_parameters(young_modulus: Tensor, poisson_ratio: Tensor) -> tuple[Tensor, Tensor]:
    first = young_modulus * poisson_ratio
    second = (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    lame_lambda = first / second.clamp_min(1e-8)
    lame_mu = young_modulus / (2.0 * (1.0 + poisson_ratio)).clamp_min(1e-8)
    return lame_lambda, lame_mu


def cauchy_stress(strain: Tensor, young_modulus: Tensor, poisson_ratio: Tensor) -> Tensor:
    lame_lambda, lame_mu = lame_parameters(young_modulus, poisson_ratio)
    trace = torch.diagonal(strain, dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, device=strain.device, dtype=strain.dtype)
    while lame_lambda.ndim < trace.ndim:
        lame_lambda = lame_lambda.unsqueeze(-1)
        lame_mu = lame_mu.unsqueeze(-1)
    volumetric = lame_lambda * trace
    return volumetric[..., None, None] * identity + 2.0 * lame_mu[..., None, None] * strain


def stress_divergence(stress: Tensor, coordinates: Tensor) -> Tensor:
    components = []
    for row in range(3):
        value = torch.zeros_like(coordinates[..., :1])
        for column in range(3):
            derivative = gradient(stress[..., row, column], coordinates)
            value = value + derivative[..., column : column + 1]
        components.append(value)
    return torch.cat(components, dim=-1)


def von_mises(stress: Tensor) -> Tensor:
    trace = torch.diagonal(stress, dim1=-2, dim2=-1).sum(dim=-1)
    identity = torch.eye(3, device=stress.device, dtype=stress.dtype)
    deviatoric = stress - trace[..., None, None] * identity / 3.0
    contraction = (deviatoric * deviatoric).sum(dim=(-2, -1))
    return torch.sqrt(1.5 * contraction.clamp_min(0.0))

