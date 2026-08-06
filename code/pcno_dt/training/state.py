from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainingState:
    epoch: int = 0
    step: int = 0
    best_validation: float = float("inf")
    patience_count: int = 0
    seed: int = 0


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    state: TrainingState,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training": asdict(state),
        "random": capture_random_state(),
    }
    torch.save(payload, temporary)
    temporary.replace(target)


def restore_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
) -> TrainingState:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_random_state(payload["random"])
    return TrainingState(**payload["training"])

