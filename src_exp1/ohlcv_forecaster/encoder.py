from __future__ import annotations

import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 5,
        hidden_size: int = 128,
        decoder_hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers != 2:
            raise ValueError("This architecture requires exactly two recurrent layers")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        projected_input = hidden_size * 2
        self.hidden_projections = nn.ModuleList(
            nn.Linear(projected_input, decoder_hidden_size) for _ in range(num_layers)
        )
        self.cell_projections = nn.ModuleList(
            nn.Linear(projected_input, decoder_hidden_size) for _ in range(num_layers)
        )

    def _project_state(
        self, state: torch.Tensor, projections: nn.ModuleList
    ) -> torch.Tensor:
        batch = state.shape[1]
        state = state.reshape(self.num_layers, 2, batch, self.hidden_size)
        directions = torch.cat((state[:, 0], state[:, 1]), dim=-1)
        return torch.stack(
            [torch.tanh(projections[layer](directions[layer])) for layer in range(2)],
            dim=0,
        )

    def forward(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if context.ndim != 3 or context.shape[-1] != self.lstm.input_size:
            raise ValueError("Encoder context must have shape [batch, time, 5]")
        outputs, (hidden, cell) = self.lstm(context)
        decoder_hidden = self._project_state(hidden, self.hidden_projections)
        decoder_cell = self._project_state(cell, self.cell_projections)
        return outputs, (decoder_hidden, decoder_cell)
