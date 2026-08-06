from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: str
    manifest: str
    crop_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    phases: tuple[str, ...]
    workers: int
    validation_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    branch_channels: tuple[int, ...]
    branch_blocks: tuple[int, ...]
    rank: int
    trunk_width: int
    trunk_depth: int
    harmonics: int
    frequencies: tuple[float, ...]
    parameter_bounds: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class PhysicsConfig:
    collocation_points: int
    residual_weights: tuple[float, float, float]
    rebalance_interval: int
    critical_stress: float
    body_force: float
    segmentation_weight: float
    density_threshold: float


@dataclass(frozen=True)
class StageConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    world_size: int
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    patience: int | None = None
    trajectories: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    folds: int
    seeds: tuple[int, ...]
    bootstrap_samples: int
    alpha: float
    conformal_coverage: float
    deferral_threshold: float


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    physics: PhysicsConfig
    synthetic: StageConfig
    clinical: StageConfig
    evaluation: EvaluationConfig


def _tuple3(values: list[Any]) -> tuple[Any, Any, Any]:
    if len(values) != 3:
        raise ValueError("expected three values")
    return values[0], values[1], values[2]


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    data_raw = raw["data"]
    model_raw = raw["model"]
    physics_raw = raw["physics"]
    eval_raw = raw["evaluation"]
    experiment_raw = raw["experiment"]
    bounds = {key: tuple(value) for key, value in model_raw["parameter_bounds"].items()}
    data = DataConfig(
        root=data_raw["root"],
        manifest=data_raw["manifest"],
        crop_shape=_tuple3(data_raw["crop_shape"]),
        spacing=_tuple3(data_raw["spacing"]),
        phases=tuple(data_raw["phases"]),
        workers=int(data_raw["workers"]),
        validation_fraction=float(data_raw["validation_fraction"]),
    )
    model = ModelConfig(
        branch_channels=tuple(model_raw["branch_channels"]),
        branch_blocks=tuple(model_raw["branch_blocks"]),
        rank=int(model_raw["rank"]),
        trunk_width=int(model_raw["trunk_width"]),
        trunk_depth=int(model_raw["trunk_depth"]),
        harmonics=int(model_raw["harmonics"]),
        frequencies=tuple(model_raw["frequencies"]),
        parameter_bounds=bounds,
    )
    physics = PhysicsConfig(
        collocation_points=int(physics_raw["collocation_points"]),
        residual_weights=tuple(physics_raw["residual_weights"]),
        rebalance_interval=int(physics_raw["rebalance_interval"]),
        critical_stress=float(physics_raw["critical_stress"]),
        body_force=float(physics_raw["body_force"]),
        segmentation_weight=float(physics_raw["segmentation_weight"]),
        density_threshold=float(physics_raw["density_threshold"]),
    )
    synthetic = StageConfig(**raw["synthetic"])
    clinical = StageConfig(**raw["clinical"])
    evaluation = EvaluationConfig(
        folds=int(eval_raw["folds"]),
        seeds=tuple(eval_raw["seeds"]),
        bootstrap_samples=int(eval_raw["bootstrap_samples"]),
        alpha=float(eval_raw["alpha"]),
        conformal_coverage=float(eval_raw["conformal_coverage"]),
        deferral_threshold=float(eval_raw["deferral_threshold"]),
    )
    return ExperimentConfig(
        name=experiment_raw["name"],
        output_dir=experiment_raw["output_dir"],
        data=data,
        model=model,
        physics=physics,
        synthetic=synthetic,
        clinical=clinical,
        evaluation=evaluation,
    )
