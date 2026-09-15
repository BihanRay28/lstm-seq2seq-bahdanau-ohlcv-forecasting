from __future__ import annotations

import torch
from torch import nn

from ohlcv_forecaster.config import ExperimentConfig
from ohlcv_forecaster.decoder import BahdanauAttention
from ohlcv_forecaster.model import Seq2SeqOHLCVForecaster


def test_attention_shapes_and_normalization() -> None:
    attention = BahdanauAttention()
    context, weights = attention(torch.randn(3, 128), torch.randn(3, 48, 256))
    assert context.shape == (3, 256)
    assert weights.shape == (3, 48)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_model_shapes_and_state_projection() -> None:
    config = ExperimentConfig()
    model = Seq2SeqOHLCVForecaster(config)
    context = torch.randn(2, 48, 5)
    encoder_outputs, (hidden, cell) = model.encoder(context)
    assert encoder_outputs.shape == (2, 48, 256)
    assert hidden.shape == cell.shape == (2, 2, 128)
    prediction, attention = model(context)
    assert prediction.shape == (2, 24, 5)
    assert attention.shape == (2, 24, 48)


class _DummyEncoder(nn.Module):
    def forward(self, context: torch.Tensor):
        batch = len(context)
        outputs = torch.zeros(batch, 48, 256)
        state = torch.zeros(2, batch, 128)
        return outputs, (state, state.clone())


class _RecordingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[torch.Tensor] = []
        self.step = 0

    def forward_step(self, token, query, encoder_outputs, state):
        self.tokens.append(token.detach().clone())
        self.step += 1
        prediction = torch.full_like(token, float(self.step))
        weights = torch.full((len(token), 48), 1 / 48)
        return prediction, weights, state, query


def test_step_one_and_whole_token_teacher_forcing() -> None:
    model = Seq2SeqOHLCVForecaster()
    model.encoder = _DummyEncoder()
    recorder = _RecordingDecoder()
    model.decoder = recorder
    context = torch.randn(2, 48, 5)
    targets = torch.randn(2, 24, 5)
    model(context, targets, teacher_forcing_probability=1.0)
    torch.testing.assert_close(recorder.tokens[0], context[:, -1])
    torch.testing.assert_close(recorder.tokens[1], targets[:, 0])
    torch.testing.assert_close(recorder.tokens[-1], targets[:, -2])


def test_autoregressive_feedback_uses_prediction() -> None:
    model = Seq2SeqOHLCVForecaster()
    model.encoder = _DummyEncoder()
    recorder = _RecordingDecoder()
    model.decoder = recorder
    context = torch.randn(1, 48, 5)
    model(context)
    torch.testing.assert_close(recorder.tokens[1], torch.ones(1, 5))
    torch.testing.assert_close(recorder.tokens[2], torch.full((1, 5), 2.0))


def test_finite_loss_and_gradients() -> None:
    model = Seq2SeqOHLCVForecaster()
    prediction, _ = model(torch.randn(2, 48, 5))
    loss = nn.SmoothL1Loss(beta=1.0)(prediction, torch.randn(2, 24, 5))
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
