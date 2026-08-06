from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, input_channels: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        output_channels = planes * self.expansion
        self.conv1 = nn.Conv3d(input_channels, planes, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, planes), planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, planes), planes)
        self.conv3 = nn.Conv3d(planes, output_channels, 1, bias=False)
        self.norm3 = nn.GroupNorm(min(32, output_channels), output_channels)
        self.activation = nn.GELU()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(min(32, output_channels), output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.activation(self.norm2(self.conv2(features)))
        features = self.norm3(self.conv3(features))
        return self.activation(features + residual)


class ResidualStage(nn.Module):
    def __init__(self, input_channels: int, planes: int, blocks: int, stride: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [Bottleneck3D(input_channels, planes, stride)]
        output_channels = planes * Bottleneck3D.expansion
        layers.extend(Bottleneck3D(output_channels, planes) for _ in range(1, blocks))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class BoundedParameterHead(nn.Module):
    def __init__(self, input_features: int, bounds: Tensor) -> None:
        super().__init__()
        self.projection = nn.Linear(input_features, bounds.shape[0])
        self.register_buffer("lower", bounds[:, 0])
        self.register_buffer("upper", bounds[:, 1])

    def forward(self, features: Tensor) -> Tensor:
        unit = torch.sigmoid(self.projection(features))
        parameters = self.lower + unit * (self.upper - self.lower)
        arterial = parameters[:, 4:5]
        portal = 1.0 - arterial
        return torch.cat((parameters[:, :5], portal), dim=1)


class HemodynamicBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        stage_channels: Sequence[int],
        stage_blocks: Sequence[int],
        rank: int,
        parameter_bounds: Sequence[tuple[float, float]],
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4 or len(stage_blocks) != 4:
            raise ValueError("branch requires four residual stages")
        stem_width = int(stage_channels[0])
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, stem_width, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(32, stem_width), stem_width),
            nn.GELU(),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        current = stem_width
        for index, (planes, blocks) in enumerate(zip(stage_channels, stage_blocks, strict=True)):
            stride = 1 if index == 0 else 2
            stages.append(ResidualStage(current, int(planes), int(blocks), stride))
            current = int(planes) * Bottleneck3D.expansion
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.coefficients = nn.Sequential(
            nn.Linear(current, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, rank),
        )
        bounds = torch.tensor(parameter_bounds, dtype=torch.float32)
        self.parameters_head = BoundedParameterHead(current, bounds)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.stem(images)
        for stage in self.stages:
            features = stage(features)
        pooled = self.pool(features).flatten(1)
        return self.coefficients(pooled), self.parameters_head(pooled)

    def freeze_early_layers(self) -> None:
        for parameter in self.stem.parameters():
            parameter.requires_grad = False
        for stage in self.stages[:2]:
            for parameter in stage.parameters():
                parameter.requires_grad = False

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

