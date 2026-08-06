from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from pcno_dt.config import load_config
from pcno_dt.data.dataset import ClinicalTrajectoryDataset, SyntheticTrajectoryDataset
from pcno_dt.models.operator import PCNODigitalTwin
from pcno_dt.physics.loss import CompositePhysicsLoss
from pcno_dt.physics.residuals import CoupledResiduals
from pcno_dt.training.engine import Trainer
from pcno_dt.training.optim import build_optimizer, cosine_with_warmup, set_seed
from pcno_dt.training.state import TrainingState


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pcno-train")
    result.add_argument("--config", default="configs/main.yaml")
    result.add_argument("--stage", choices=("synthetic", "clinical"), required=True)
    result.add_argument("--synthetic-root", default="data/synthetic")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return result


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)
    set_seed(arguments.seed)
    if arguments.stage == "synthetic":
        dataset = SyntheticTrajectoryDataset(arguments.synthetic_root)
        stage = config.synthetic
    else:
        dataset = ClinicalTrajectoryDataset(
            config.data.manifest,
            config.data.crop_shape,
            config.physics.collocation_points,
        )
        stage = config.clinical
    validation_size = max(1, int(len(dataset) * config.data.validation_fraction))
    training_size = len(dataset) - validation_size
    training, validation = random_split(
        dataset,
        (training_size, validation_size),
        generator=torch.Generator().manual_seed(arguments.seed),
    )
    training_loader = DataLoader(
        training,
        batch_size=stage.batch_size,
        shuffle=True,
        num_workers=config.data.workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=stage.batch_size,
        num_workers=config.data.workers,
        pin_memory=True,
    )
    model = PCNODigitalTwin(config.model)
    if arguments.stage == "clinical":
        model.freeze_for_clinical_stage()
    residuals = CoupledResiduals(config.physics.critical_stress, config.physics.body_force)
    objective = CompositePhysicsLoss(
        residuals,
        config.physics.residual_weights,
        config.physics.segmentation_weight,
        config.physics.density_threshold,
    )
    optimizer = build_optimizer(model, stage.learning_rate, stage.weight_decay)
    total_steps = stage.epochs * max(1, len(training_loader))
    scheduler = cosine_with_warmup(optimizer, stage.warmup_steps, total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(config.output_dir) / arguments.stage / f"seed_{arguments.seed}"
    trainer = Trainer(
        model,
        objective,
        optimizer,
        scheduler,
        device,
        stage.gradient_clip,
        output,
        arguments.precision,
    )
    trainer.fit(
        training_loader,
        validation_loader,
        stage.epochs,
        TrainingState(seed=arguments.seed),
        stage.patience,
    )


if __name__ == "__main__":
    main()

