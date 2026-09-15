from __future__ import annotations

import torch
from torch import nn


class BahdanauAttention(nn.Module):
    def __init__(
        self,
        *,
        query_size: int = 128,
        value_size: int = 256,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        self.query_projection = nn.Linear(query_size, attention_dim, bias=False)
        self.value_projection = nn.Linear(value_size, attention_dim, bias=False)
        self.energy_projection = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self, query: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2 or values.ndim != 3:
            raise ValueError("Attention requires query [B,Q] and values [B,T,V]")
        energy = self.energy_projection(
            torch.tanh(
                self.query_projection(query).unsqueeze(1)
                + self.value_projection(values)
            )
        ).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights


class AttentiveLSTMDecoder(nn.Module):
    def __init__(
        self,
        *,
        token_size: int = 5,
        encoder_output_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        attention_dim: int = 128,
        head_hidden_size: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.attention = BahdanauAttention(
            query_size=hidden_size,
            value_size=encoder_output_size,
            attention_dim=attention_dim,
        )
        self.lstm = nn.LSTM(
            input_size=token_size + encoder_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size + encoder_output_size, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, token_size),
        )

    def forward_step(
        self,
        token: torch.Tensor,
        query: torch.Tensor,
        encoder_outputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        context, attention_weights = self.attention(query, encoder_outputs)
        recurrent_input = torch.cat((token, context), dim=-1).unsqueeze(1)
        recurrent_output, next_state = self.lstm(recurrent_input, state)
        recurrent_output = recurrent_output.squeeze(1)
        prediction = self.output_head(torch.cat((recurrent_output, context), dim=-1))
        next_query = next_state[0][-1]
        return prediction, attention_weights, next_state, next_query
