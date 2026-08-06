from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def dice_score(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    predicted = prediction >= threshold
    observed = target >= threshold
    dimensions = tuple(range(1, prediction.ndim))
    intersection = (predicted & observed).sum(dim=dimensions).float()
    denominator = predicted.sum(dim=dimensions).float() + observed.sum(dim=dimensions).float()
    return torch.where(denominator > 0, 2.0 * intersection / denominator, torch.ones_like(denominator))


def physical_volume(mask: Tensor, spacing: tuple[float, float, float]) -> Tensor:
    voxel_volume_ml = float(np.prod(spacing)) / 1000.0
    dimensions = tuple(range(1, mask.ndim))
    return mask.sum(dim=dimensions).float() * voxel_volume_ml


def volume_mae(
    prediction: Tensor,
    target: Tensor,
    spacing: tuple[float, float, float],
    threshold: float = 0.5,
) -> Tensor:
    predicted_volume = physical_volume(prediction >= threshold, spacing)
    observed_volume = physical_volume(target >= threshold, spacing)
    return torch.mean(torch.abs(predicted_volume - observed_volume))


def time_to_progression_mae(prediction: Tensor, target: Tensor, event: Tensor) -> Tensor:
    observed = event.bool()
    if not torch.any(observed):
        return torch.full((), float("nan"), device=prediction.device)
    return torch.mean(torch.abs(prediction[observed] - target[observed]))


def concordance_index(times: Tensor, risks: Tensor, events: Tensor) -> Tensor:
    times = times.flatten()
    risks = risks.flatten()
    events = events.flatten().bool()
    earlier = times[:, None] < times[None, :]
    comparable = earlier & events[:, None]
    pairs = comparable.sum()
    if pairs == 0:
        return torch.full((), float("nan"), device=times.device)
    ordered = risks[:, None] > risks[None, :]
    tied = risks[:, None] == risks[None, :]
    concordant = (ordered & comparable).sum().float()
    concordant += 0.5 * (tied & comparable).sum().float()
    return concordant / pairs


def normalized_pde_residual(residual: Tensor, scale: Tensor | float = 1.0) -> Tensor:
    denominator = torch.as_tensor(scale, device=residual.device, dtype=residual.dtype).clamp_min(1e-12)
    return torch.sqrt(torch.mean(residual.square())) / denominator


def calibration_error(
    lower: Tensor,
    upper: Tensor,
    target: Tensor,
    nominal_coverage: float,
) -> Tensor:
    covered = ((target >= lower) & (target <= upper)).float().mean()
    return torch.abs(covered - nominal_coverage)


def trajectory_rmse_by_horizon(
    prediction: Tensor,
    target: Tensor,
    horizons: Tensor,
    requested: Tensor,
) -> dict[float, float]:
    result = {}
    for horizon in requested:
        selected = torch.isclose(horizons, horizon)
        if torch.any(selected):
            error = torch.sqrt(torch.mean((prediction[selected] - target[selected]).square()))
            result[float(horizon)] = float(error)
    return result

