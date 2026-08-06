from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class BiologicalFourierFeatures(nn.Module):
    def __init__(self, frequencies: Sequence[float], harmonics: int) -> None:
        super().__init__()
        bases = torch.tensor(frequencies, dtype=torch.float32)
        orders = torch.arange(1, harmonics + 1, dtype=torch.float32)
        self.register_buffer("angular", 2.0 * torch.pi * (bases[:, None] * orders[None, :]).flatten())

    @property
    def output_features(self) -> int:
        return int(self.angular.numel() * 2)

    def forward(self, time: Tensor) -> Tensor:
        angles = time * self.angular
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class GatedLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.value = nn.Linear(width, width)
        self.gate = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, inputs: Tensor) -> Tensor:
        transformed = self.value(inputs) * torch.sigmoid(self.gate(inputs))
        return self.norm(inputs + transformed)


class SpatiotemporalTrunk(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        rank: int,
        frequencies: Sequence[float],
        harmonics: int,
    ) -> None:
        super().__init__()
        self.fourier = BiologicalFourierFeatures(frequencies, harmonics)
        input_width = 3 + self.fourier.output_features
        self.input = nn.Sequential(nn.Linear(input_width, width), nn.GELU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList(GatedLayer(width) for _ in range(depth - 2))
        self.basis = nn.Sequential(nn.GELU(), nn.Linear(width, rank))
        self.displacement = nn.Sequential(nn.GELU(), nn.Linear(width, 3))
        self.proliferation = nn.Sequential(nn.GELU(), nn.Linear(width, 1), nn.Softplus())

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        spatial = coordinates[..., :3]
        time = coordinates[..., 3:4]
        encoded = torch.cat((spatial, self.fourier(time)), dim=-1)
        features = self.input(encoded)
        for layer in self.layers:
            features = layer(features)
        return self.basis(features), self.displacement(features), self.proliferation(features)

