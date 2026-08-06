from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float) -> Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)


def cosine_with_warmup(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, scale)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

