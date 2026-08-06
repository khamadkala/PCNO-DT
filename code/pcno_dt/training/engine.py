from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from pcno_dt.physics.loss import CompositePhysicsLoss
from pcno_dt.training.state import TrainingState, atomic_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        objective: CompositePhysicsLoss,
        optimizer: Optimizer,
        scheduler: Any,
        device: torch.device,
        gradient_clip: float,
        output_dir: str | Path,
        precision: str = "fp32",
    ) -> None:
        self.model = model.to(device)
        self.objective = objective.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip
        self.output_dir = Path(output_dir)
        self.precision = precision
        self.logger = logging.getLogger(__name__)
        self.scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16")

    def _autocast(self) -> Any:
        if self.device.type != "cuda" or self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }

    def train_epoch(self, loader: Iterable[dict[str, Any]], state: TrainingState) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        batches = 0
        for raw in loader:
            batch = self._move(raw)
            coordinates = batch["coordinates"].requires_grad_(True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.model(batch["images"], coordinates)
                losses, _ = self.objective(
                    output,
                    batch["density"],
                    batch["density"] >= self.objective.density_threshold,
                    coordinates,
                )
            self.scaler.scale(losses.total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            state.step += 1
            batches += 1
            for name, value in losses.__dict__.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
        return {name: value / max(1, batches) for name, value in totals.items()}

    @torch.no_grad()
    def validate(self, loader: Iterable[dict[str, Any]]) -> dict[str, float]:
        self.model.eval()
        absolute_error = 0.0
        squared_error = 0.0
        count = 0
        for raw in loader:
            batch = self._move(raw)
            output = self.model(batch["images"], batch["coordinates"])
            difference = output.density - batch["density"].view_as(output.density)
            absolute_error += float(difference.abs().sum())
            squared_error += float(difference.square().sum())
            count += difference.numel()
        return {
            "mae": absolute_error / max(1, count),
            "rmse": (squared_error / max(1, count)) ** 0.5,
        }

    def fit(
        self,
        train_loader: Iterable[dict[str, Any]],
        validation_loader: Iterable[dict[str, Any]],
        epochs: int,
        state: TrainingState,
        patience: int | None,
    ) -> TrainingState:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for epoch in range(state.epoch, epochs):
            state.epoch = epoch
            training = self.train_epoch(train_loader, state)
            validation = self.validate(validation_loader)
            score = validation["mae"]
            improved = score < state.best_validation
            if improved:
                state.best_validation = score
                state.patience_count = 0
                atomic_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    state,
                )
            else:
                state.patience_count += 1
            self.logger.info(
                "epoch=%d train=%.6f validation=%.6f best=%.6f",
                epoch,
                training["total"],
                score,
                state.best_validation,
            )
            atomic_checkpoint(
                self.output_dir / "latest.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                state,
            )
            if patience is not None and state.patience_count >= patience:
                break
        return state


def distributed_model(model: nn.Module, device: torch.device) -> nn.Module:
    if not torch.distributed.is_initialized():
        return model
    return DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
