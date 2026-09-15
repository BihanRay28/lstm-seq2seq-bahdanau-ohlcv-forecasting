from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ohlcv_forecaster.config import DATASET_PATH, ExperimentConfig
from ohlcv_forecaster.data import (
    ChannelStandardizer,
    build_fold_datasets,
    build_window_dataset,
    fit_partition_scaler,
    load_and_validate_dataset,
    make_chronological_split,
    prepare_inference_context,
    reconstruct_ohlcv,
    transform_partition,
)


def test_cemented_dataset_exists() -> None:
    assert DATASET_PATH.is_file()
    assert DATASET_PATH.name == "BTC_USDT_30m_Binance_20260913_184821.csv"


def test_missing_path_fails(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_validate_dataset(tmp_path / "missing.csv")


def test_exact_schema_is_required(tmp_path, valid_frame: pd.DataFrame) -> None:
    path = tmp_path / "wrong.csv"
    valid_frame.rename(columns={"volume": "vol"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exact columns"):
        load_and_validate_dataset(path)


@pytest.mark.parametrize("failure", ["order", "duplicate", "gap", "missing", "ohlc"])
def test_dataset_validation_failures(
    tmp_path, valid_frame: pd.DataFrame, failure: str
) -> None:
    frame = valid_frame.copy()
    if failure == "order":
        frame = frame.iloc[::-1]
    elif failure == "duplicate":
        frame.loc[2, "datetime"] = frame.loc[1, "datetime"]
    elif failure == "gap":
        frame.loc[2:, "datetime"] += pd.Timedelta(minutes=30)
    elif failure == "missing":
        frame.loc[3, "volume"] = np.nan
    else:
        frame.loc[3, "high"] = frame.loc[3, "open"] - 1
    path = tmp_path / f"{failure}.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_and_validate_dataset(path)


def test_transform_scaler_and_reconstruction_round_trip(valid_frame: pd.DataFrame) -> None:
    transformed, offset = transform_partition(valid_frame)
    scaler = ChannelStandardizer().fit(transformed)
    reconstructed = reconstruct_ohlcv(
        scaler.transform(transformed), scaler, valid_frame.iloc[0]["close"]
    )
    expected = valid_frame.iloc[1:][["open", "high", "low", "close", "volume"]]
    assert offset == 1
    np.testing.assert_allclose(reconstructed, expected.to_numpy(), rtol=1e-10)


def test_sample_formula_and_shapes(valid_frame: pd.DataFrame) -> None:
    config = ExperimentConfig()
    scaler = fit_partition_scaler(valid_frame)
    dataset = build_window_dataset(valid_frame, scaler, config, stride=1)
    # 160 raw rows -> 159 transformable rows -> 159 - 48 - 24 + 1.
    assert len(dataset) == 88
    sample = dataset[0]
    assert sample["context"].shape == (48, 5)
    assert sample["target"].shape == (24, 5)
    assert sample["raw_target"].shape == (24, 5)


def test_stride_behavior(valid_frame: pd.DataFrame) -> None:
    config = ExperimentConfig()
    scaler = fit_partition_scaler(valid_frame)
    dataset = build_window_dataset(
        valid_frame,
        scaler,
        config,
        stride=24,
        previous_close=99.0,
    )
    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset.starts, np.array([0, 24, 48, 72]))


def test_real_split_counts_and_partition_isolation() -> None:
    config = ExperimentConfig()
    frame = load_and_validate_dataset(config.dataset_path)
    split = make_chronological_split(frame, config)
    assert len(split.development) == 47_304
    assert len(split.test) == 5_256
    assert [fold.train_end for fold in split.folds] == [
        23_664,
        28_392,
        33_120,
        37_848,
        42_576,
    ]
    expected_train_samples = [23_592, 28_320, 33_048, 37_776, 42_504]
    for fold, expected in zip(split.folds, expected_train_samples, strict=True):
        train, validation, _ = build_fold_datasets(split, fold, config)
        assert len(train) == expected
        assert len(validation) == 195
        assert fold.train_end == fold.validation_start
    assert split.folds[-1].validation_end == len(split.development)


def test_config_rejects_missing_cemented_path(tmp_path) -> None:
    config = replace(ExperimentConfig(), dataset_path=tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="Cemented"):
        config.validate()


def test_reconstruction_constraints(valid_frame: pd.DataFrame) -> None:
    transformed, _ = transform_partition(valid_frame)
    scaler = ChannelStandardizer().fit(transformed)
    extreme = np.zeros((2, 24, 5), dtype=np.float64)
    extreme[:, :, 2:4] = -1_000
    extreme[:, :, 4] = -1_000
    candles = reconstruct_ohlcv(extreme, scaler, np.array([100.0, 200.0]))
    assert np.isfinite(candles).all()
    assert np.all(candles[:, :, 1] >= np.maximum(candles[:, :, 0], candles[:, :, 3]))
    assert np.all(candles[:, :, 2] <= np.minimum(candles[:, :, 0], candles[:, :, 3]))
    assert np.all(candles[:, :, :4] > 0)
    assert np.all(candles[:, :, 4] >= 0)


def test_inference_requires_anchor_plus_48_candles(valid_frame: pd.DataFrame) -> None:
    config = ExperimentConfig()
    scaler = fit_partition_scaler(valid_frame)
    context, last_close = prepare_inference_context(
        valid_frame.iloc[:49].reset_index(drop=True), scaler, config
    )
    assert context.shape == (48, 5)
    assert last_close == pytest.approx(valid_frame.iloc[48]["close"])
    with pytest.raises(ValueError, match="exactly 49"):
        prepare_inference_context(valid_frame.iloc[:48], scaler, config)
