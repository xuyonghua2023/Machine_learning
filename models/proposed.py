from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .transformer import SinusoidalPositionalEncoding


class MultiScaleTemporalBlock(nn.Module):
    """Extracts local usage patterns with several temporal receptive fields."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        branches = []
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branches.append(
                nn.Sequential(
                    nn.Conv1d(input_dim, d_model, kernel_size, padding=padding),
                    nn.GELU(),
                    nn.BatchNorm1d(d_model),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model * len(kernel_sizes), d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual = nn.Conv1d(input_dim, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channels_first = x.transpose(1, 2)
        local_features = torch.cat(
            [branch(x_channels_first) for branch in self.branches],
            dim=1,
        )
        fused = self.fusion(local_features)
        gated = fused * self.gate(fused)
        return (gated + self.residual(x_channels_first)).transpose(1, 2)


class AttentionPooling(nn.Module):
    """Learns which historical days are most useful for the forecast."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class ProposedModel(nn.Module):
    """CNN-Transformer hybrid proposed model.

    Local convolution branches capture weekly-scale consumption fluctuations,
    while the Transformer encoder models longer temporal dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        kernel_sizes: Sequence[int] = (3, 5, 7),
    ) -> None:
        super().__init__()
        self.temporal_block = MultiScaleTemporalBlock(
            input_dim=input_dim,
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        self.position = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_block(x)
        x = self.position(x)
        encoded = self.encoder(x)
        pooled = self.pool(encoded)
        return self.head(pooled)

