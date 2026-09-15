from __future__ import annotations

import numpy as np
import pytest

from ohlcv_forecaster.evaluation import (
    calculate_raw_metrics,
    calculate_trend_metrics,
    calculate_value_accuracy,
    classify_candle_directions,
    make_persistence_predictions,
)
from ohlcv_forecaster.training import teacher_forcing_probability


def test_teacher_forcing_schedule_boundaries() -> None:
    assert teacher_forcing_probability(1) == pytest.approx(1.0)
    assert teacher_forcing_probability(80) == pytest.approx(0.2)
    assert teacher_forcing_probability(100) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        teacher_forcing_probability(0)


def test_metrics_are_per_field_and_horizon() -> None:
    targets = np.ones((3, 24, 5))
    predictions = targets + 2.0
    metrics = calculate_raw_metrics(predictions, targets)
    assert set(metrics) == {"sample_count", "by_field"}
    assert "overall" not in metrics
    assert len(metrics["by_field"]["open"]["mae_by_horizon"]) == 24
    assert metrics["by_field"]["volume"]["average_mae"] == pytest.approx(2.0)
    assert metrics["by_field"]["close"]["average_rmse"] == pytest.approx(2.0)


def test_persistence_repeats_complete_vector() -> None:
    reference = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=float)
    prediction = make_persistence_predictions(reference, horizon=24)
    assert prediction.shape == (2, 24, 5)
    np.testing.assert_array_equal(prediction[:, 0], reference)
    np.testing.assert_array_equal(prediction[:, -1], reference)


def test_value_accuracy_is_bounded_and_exact() -> None:
    targets = np.full((2, 24, 5), 100.0)
    predictions = targets.copy()
    exact = calculate_value_accuracy(predictions, targets)
    assert exact["macro_accuracy"] == pytest.approx(1.0)
    predictions[:, :, 0] = 50.0
    scored = calculate_value_accuracy(predictions, targets)
    assert scored["by_field"]["open"]["accuracy"] == pytest.approx(0.5)
    assert 0.0 <= scored["macro_accuracy"] <= 1.0


def test_directional_and_endpoint_accuracy() -> None:
    reference = np.array([[100, 101, 99, 100, 10]], dtype=float)
    targets = np.repeat(reference[:, None, :], 24, axis=1)
    targets[:, :, 0] = 100
    targets[:, :, 3] = 101
    predictions = targets.copy()
    trends = calculate_trend_metrics(predictions, targets, reference)
    assert trends["candle_accuracy"] == pytest.approx(1.0)
    assert trends["twelve_hour_endpoint_accuracy"] == pytest.approx(1.0)
    assert trends["actual_bullish_candles"] == 24


def test_doji_classification_uses_body_to_range_ratio() -> None:
    candles = np.array(
        [
            [100.0, 105.0, 95.0, 100.5, 10.0],  # 5% body-to-range: doji
            [100.0, 105.0, 95.0, 103.0, 10.0],  # bullish
            [100.0, 105.0, 95.0, 97.0, 10.0],   # bearish
        ]
    )
    np.testing.assert_array_equal(
        classify_candle_directions(candles), np.array([1, 2, 0])
    )
    targets = np.repeat(candles[:1][None, :, :], 24, axis=1)
    predictions = targets.copy()
    reference = np.array([[100.0, 101.0, 99.0, 100.0, 10.0]])
    trends = calculate_trend_metrics(predictions, targets, reference)
    assert trends["actual_doji_candles"] == 24
    assert trends["predicted_doji_candles"] == 24
