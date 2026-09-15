from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .data import ChannelStandardizer, VALUE_COLUMNS, reconstruct_ohlcv
from .model import Seq2SeqOHLCVForecaster


DOJI_BODY_TO_RANGE_THRESHOLD = 0.10


def classify_candle_directions(candles: np.ndarray) -> np.ndarray:
    """Return 0=bearish, 1=doji, 2=bullish for raw OHLCV candles."""
    values = np.asarray(candles, dtype=np.float64)
    if values.shape[-1] != 5:
        raise ValueError("Candles must have five OHLCV fields")
    body = np.abs(values[..., 3] - values[..., 0])
    candle_range = np.maximum(values[..., 1] - values[..., 2], 0.0)
    doji = body <= DOJI_BODY_TO_RANGE_THRESHOLD * candle_range
    classes = np.zeros(body.shape, dtype=np.int8)
    classes[doji] = 1
    classes[(~doji) & (values[..., 3] > values[..., 0])] = 2
    return classes


def calculate_raw_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Predictions and targets must share shape [N, H, 5]")
    errors = predictions - targets
    absolute = np.abs(errors)
    squared = errors**2
    by_field: dict[str, Any] = {}
    for field_index, field in enumerate(VALUE_COLUMNS):
        horizon_mae = absolute[:, :, field_index].mean(axis=0)
        horizon_rmse = np.sqrt(squared[:, :, field_index].mean(axis=0))
        by_field[field] = {
            "mae_by_horizon": horizon_mae.tolist(),
            "rmse_by_horizon": horizon_rmse.tolist(),
            "average_mae": float(absolute[:, :, field_index].mean()),
            "average_rmse": float(np.sqrt(squared[:, :, field_index].mean())),
        }
    return {"sample_count": int(len(predictions)), "by_field": by_field}


def calculate_value_accuracy(
    predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Predictions and targets must share shape [N, H, 5]")
    denominator = np.maximum(np.abs(targets), epsilon)
    scores = np.clip(1.0 - np.abs(predictions - targets) / denominator, 0.0, 1.0)
    by_field: dict[str, Any] = {}
    for field_index, field in enumerate(VALUE_COLUMNS):
        field_scores = scores[:, :, field_index]
        by_field[field] = {
            "accuracy": float(field_scores.mean()),
            "accuracy_by_horizon": field_scores.mean(axis=0).tolist(),
        }
    return {
        "definition": (
            "mean clipped relative closeness: "
            "max(0, 1 - abs(prediction-actual)/max(abs(actual), epsilon))"
        ),
        "macro_accuracy": float(scores.mean()),
        "macro_accuracy_by_horizon": scores.mean(axis=(0, 2)).tolist(),
        "by_field": by_field,
    }


def make_persistence_predictions(
    reference_ohlcv: np.ndarray, horizon: int
) -> np.ndarray:
    reference = np.asarray(reference_ohlcv, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 5:
        raise ValueError("Persistence references must have shape [N, 5]")
    return np.repeat(reference[:, None, :], horizon, axis=1)


def calculate_trend_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    reference_ohlcv: np.ndarray,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    references = np.asarray(reference_ohlcv, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("Predictions and targets must share shape [N, H, 5]")
    if references.shape != (len(predictions), 5):
        raise ValueError("Reference candles must have shape [N, 5]")

    predicted_classes = classify_candle_directions(predictions)
    actual_classes = classify_candle_directions(targets)
    candle_correct = predicted_classes == actual_classes

    predicted_endpoint_bullish = predictions[:, -1, 3] > references[:, 3]
    actual_endpoint_bullish = targets[:, -1, 3] > references[:, 3]
    endpoint_correct = predicted_endpoint_bullish == actual_endpoint_bullish

    return {
        "definition": {
            "candle": (
                "doji when abs(close-open) <= 10% of (high-low); otherwise "
                "bullish when close > open and bearish when close < open"
            ),
            "twelve_hour_endpoint": (
                "bullish when horizon-24 close > final observed close; otherwise bearish"
            ),
        },
        "candle_accuracy": float(candle_correct.mean()),
        "candle_accuracy_by_horizon": candle_correct.mean(axis=0).tolist(),
        "twelve_hour_endpoint_accuracy": float(endpoint_correct.mean()),
        "actual_bullish_candles": int((actual_classes == 2).sum()),
        "actual_doji_candles": int((actual_classes == 1).sum()),
        "actual_bearish_candles": int((actual_classes == 0).sum()),
        "predicted_bullish_candles": int((predicted_classes == 2).sum()),
        "predicted_doji_candles": int((predicted_classes == 1).sum()),
        "predicted_bearish_candles": int((predicted_classes == 0).sum()),
        "actual_bullish_endpoints": int(actual_endpoint_bullish.sum()),
        "actual_bearish_endpoints": int(
            actual_endpoint_bullish.size - actual_endpoint_bullish.sum()
        ),
        "predicted_bullish_endpoints": int(predicted_endpoint_bullish.sum()),
        "predicted_bearish_endpoints": int(
            predicted_endpoint_bullish.size - predicted_endpoint_bullish.sum()
        ),
    }


@torch.no_grad()
def predict_dataset(
    model: Seq2SeqOHLCVForecaster,
    loader: Iterable[dict[str, torch.Tensor]],
    scaler: ChannelStandardizer,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    standardized_predictions: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    attention_weights: list[np.ndarray] = []
    references: list[np.ndarray] = []
    for batch in loader:
        context = batch["context"].to(device)
        predictions, attention = model(context)
        standardized = predictions.cpu().numpy()
        reference_close = batch["reference_close"].numpy()
        standardized_predictions.append(standardized)
        raw_predictions.append(
            reconstruct_ohlcv(standardized, scaler, reference_close)
        )
        raw_targets.append(batch["raw_target"].numpy())
        attention_weights.append(attention.cpu().numpy())
        references.append(batch["reference_ohlcv"].numpy())
    if not standardized_predictions:
        raise ValueError("Evaluation dataset produced no batches")
    raw_prediction_array = np.concatenate(raw_predictions)
    raw_target_array = np.concatenate(raw_targets)
    reference_array = np.concatenate(references)
    persistence = make_persistence_predictions(
        reference_array, raw_prediction_array.shape[1]
    )
    return {
        "standardized_predictions": np.concatenate(standardized_predictions),
        "raw_predictions": raw_prediction_array,
        "raw_targets": raw_target_array,
        "attention_weights": np.concatenate(attention_weights),
        "persistence_predictions": persistence,
        "reference_ohlcv": reference_array,
    }
