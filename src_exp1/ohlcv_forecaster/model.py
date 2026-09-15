from __future__ import annotations

import torch
from torch import nn

from .config import ExperimentConfig
from .decoder import AttentiveLSTMDecoder
from .encoder import BiLSTMEncoder


class Seq2SeqOHLCVForecaster(nn.Module):
    def __init__(self, config: ExperimentConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExperimentConfig()
        cfg = self.config
        self.encoder = BiLSTMEncoder(
            input_size=cfg.input_size,
            hidden_size=cfg.encoder_hidden_size,
            decoder_hidden_size=cfg.decoder_hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        self.decoder = AttentiveLSTMDecoder(
            token_size=cfg.input_size,
            encoder_output_size=cfg.encoder_hidden_size * 2,
            hidden_size=cfg.decoder_hidden_size,
            num_layers=cfg.num_layers,
            attention_dim=cfg.attention_dim,
            head_hidden_size=cfg.head_hidden_size,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        context: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        teacher_forcing_probability: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if context.ndim != 3 or context.shape[1:] != (
            cfg.context_length,
            cfg.input_size,
        ):
            raise ValueError(
                f"Context must have shape [B,{cfg.context_length},{cfg.input_size}]"
            )
        if targets is not None and targets.shape != (
            context.shape[0],
            cfg.horizon,
            cfg.input_size,
        ):
            raise ValueError(
                f"Targets must have shape [B,{cfg.horizon},{cfg.input_size}]"
            )
        if targets is None and teacher_forcing_probability > 0:
            raise ValueError("Teacher forcing requires targets")
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("Teacher-forcing probability must be in [0, 1]")

        encoder_outputs, state = self.encoder(context)
        query = state[0][-1]
        token = context[:, -1]
        predictions: list[torch.Tensor] = []
        attention_steps: list[torch.Tensor] = []

        for step in range(cfg.horizon):
            prediction, weights, state, query = self.decoder.forward_step(
                token, query, encoder_outputs, state
            )
            predictions.append(prediction)
            attention_steps.append(weights)
            if step == cfg.horizon - 1:
                continue
            token = prediction
            if targets is not None and teacher_forcing_probability > 0:
                use_actual = torch.rand(
                    context.shape[0],
                    1,
                    device=context.device,
                    generator=generator,
                ) < teacher_forcing_probability
                token = torch.where(use_actual, targets[:, step], prediction)

        return torch.stack(predictions, dim=1), torch.stack(attention_steps, dim=1)
