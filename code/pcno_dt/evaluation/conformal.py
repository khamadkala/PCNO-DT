from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConformalInterval:
    lower: Tensor
    estimate: Tensor
    upper: Tensor
    half_width: Tensor


class SplitConformalCalibrator:
    def __init__(self, coverage: float = 0.9) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie between zero and one")
        self.coverage = coverage
        self.quantile: Tensor | None = None

    def fit(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes differ")
        scores = torch.abs(target - prediction).flatten()
        count = scores.numel()
        level = min(1.0, torch.ceil(torch.tensor((count + 1) * self.coverage)).item() / count)
        self.quantile = torch.quantile(scores, level, interpolation="higher")

    def interval(self, prediction: Tensor) -> ConformalInterval:
        if self.quantile is None:
            raise RuntimeError("calibrator has not been fitted")
        width = self.quantile.to(prediction.device)
        return ConformalInterval(prediction - width, prediction, prediction + width, width)

    def empirical_coverage(self, prediction: Tensor, target: Tensor) -> float:
        interval = self.interval(prediction)
        covered = (target >= interval.lower) & (target <= interval.upper)
        return float(covered.float().mean())
