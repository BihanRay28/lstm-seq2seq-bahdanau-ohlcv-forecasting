from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig


RAW_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")
VALUE_COLUMNS = ("open", "high", "low", "close", "volume")
FEATURE_COLUMNS = (
    "log_open_gap",
    "log_close_return",
    "log_upper_excursion",
    "log_lower_excursion",
    "log1p_volume",
)


@dataclass(frozen=True)
class FoldSpec:
    index: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int


@dataclass(frozen=True)
class ChronologicalSplit:
    development: pd.DataFrame
    test: pd.DataFrame
    test_anchor_close: float
    folds: tuple[FoldSpec, ...]


@dataclass
class ChannelStandardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ChannelStandardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 5 or len(array) == 0:
            raise ValueError("Scaler requires a non-empty [N, 5] array")
        if not np.isfinite(array).all():
            raise ValueError("Scaler input contains non-finite values")
        scale = array.std(axis=0, ddof=0)
        if np.any(scale <= np.finfo(np.float64).eps):
            raise ValueError("Every transformed channel must have non-zero variance")
        self.mean = array.mean(axis=0)
        self.scale = scale
        return self

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted")
        return self.mean, self.scale

    def transform(self, values: np.ndarray) -> np.ndarray:
        mean, scale = self._require_fitted()
        return (np.asarray(values, dtype=np.float64) - mean) / scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        mean, scale = self._require_fitted()
        return np.asarray(values, dtype=np.float64) * scale + mean

    def to_dict(self) -> dict[str, list[float]]:
        mean, scale = self._require_fitted()
        return {"mean": mean.tolist(), "scale": scale.tolist()}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, values: dict[str, Sequence[float]]) -> "ChannelStandardizer":
        scaler = cls(
            mean=np.asarray(values["mean"], dtype=np.float64),
            scale=np.asarray(values["scale"], dtype=np.float64),
        )
        scaler._require_fitted()
        return scaler


class OHLCVWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        transformed: np.ndarray,
        raw_rows: np.ndarray,
        *,
        raw_offset: int,
        context_length: int,
        horizon: int,
        stride: int,
    ) -> None:
        self.transformed = torch.as_tensor(transformed, dtype=torch.float32)
        self.raw_rows = np.array(raw_rows, dtype=np.float64, copy=True)
        self.raw_offset = raw_offset
        self.context_length = context_length
        self.horizon = horizon
        self.stride = stride
        available = len(self.transformed) - context_length - horizon
        self.sample_count = max(0, 1 + available // stride)
        self.starts = np.arange(self.sample_count, dtype=np.int64) * stride

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        target_start = start + self.context_length
        target_end = target_start + self.horizon
        raw_context_end = self.raw_offset + target_start - 1
        raw_target_start = raw_context_end + 1
        raw_target_end = raw_target_start + self.horizon
        return {
            "context": self.transformed[start:target_start],
            "target": self.transformed[target_start:target_end],
            "reference_close": torch.tensor(
                self.raw_rows[raw_context_end, 3], dtype=torch.float64
            ),
            "reference_ohlcv": torch.as_tensor(
                self.raw_rows[raw_context_end].copy(), dtype=torch.float64
            ),
            "raw_target": torch.as_tensor(
                self.raw_rows[raw_target_start:raw_target_end], dtype=torch.float64
            ),
            "start": torch.tensor(start, dtype=torch.int64),
        }


def load_and_validate_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    frame = pd.read_csv(path)
    if tuple(frame.columns) != RAW_COLUMNS:
        raise ValueError(
            f"Expected exact columns {RAW_COLUMNS}, received {tuple(frame.columns)}"
        )
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    for column in VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not frame["datetime"].is_monotonic_increasing:
        raise ValueError("Datetimes must be strictly chronological")
    if frame["datetime"].duplicated().any():
        raise ValueError("Dataset contains duplicate datetimes")
    deltas = frame["datetime"].diff().dropna()
    if not (deltas == pd.Timedelta(minutes=30)).all():
        raise ValueError("Dataset must have an uninterrupted 30-minute cadence")
    values = frame.loc[:, VALUE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains non-finite OHLCV values")
    o, h, l, c, v = values.T
    if np.any(o <= 0) or np.any(h <= 0) or np.any(l <= 0) or np.any(c <= 0):
        raise ValueError("OHLC prices must be positive")
    if np.any(v < 0):
        raise ValueError("Volume must be nonnegative")
    if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)):
        raise ValueError("Dataset violates OHLC high/low consistency")
    return frame.reset_index(drop=True)


def transform_partition(
    frame: pd.DataFrame, previous_close: float | None = None
) -> tuple[np.ndarray, int]:
    raw = frame.loc[:, VALUE_COLUMNS].to_numpy(dtype=np.float64)
    if len(raw) == 0:
        return np.empty((0, 5), dtype=np.float64), 0
    if previous_close is None:
        if len(raw) < 2:
            return np.empty((0, 5), dtype=np.float64), 1
        current = raw[1:]
        previous = raw[:-1, 3]
        raw_offset = 1
    else:
        if not np.isfinite(previous_close) or previous_close <= 0:
            raise ValueError("The partition anchor close must be finite and positive")
        current = raw
        previous = np.concatenate(([previous_close], raw[:-1, 3]))
        raw_offset = 0
    o, h, l, c, v = current.T
    features = np.column_stack(
        (
            np.log(o / previous),
            np.log(c / o),
            np.log(h / np.maximum(o, c)),
            np.log(np.minimum(o, c) / l),
            np.log1p(v),
        )
    )
    if not np.isfinite(features).all():
        raise ValueError("Transformation produced non-finite values")
    if np.any(features[:, 2:4] < -1e-10) or np.any(features[:, 4] < 0):
        raise ValueError("Excursion and log-volume features must be nonnegative")
    return features, raw_offset


def reconstruct_ohlcv(
    standardized_predictions: np.ndarray,
    scaler: ChannelStandardizer,
    initial_previous_close: float | np.ndarray,
) -> np.ndarray:
    standardized = np.asarray(standardized_predictions, dtype=np.float64)
    squeeze = standardized.ndim == 2
    if squeeze:
        standardized = standardized[None, ...]
    if standardized.ndim != 3 or standardized.shape[-1] != 5:
        raise ValueError("Predictions must have shape [H, 5] or [B, H, 5]")
    transformed = scaler.inverse_transform(standardized)
    batch, horizon, _ = transformed.shape
    previous = np.broadcast_to(
        np.asarray(initial_previous_close, dtype=np.float64), (batch,)
    ).copy()
    if np.any(previous <= 0) or not np.isfinite(previous).all():
        raise ValueError("Initial previous closes must be finite and positive")
    output = np.empty((batch, horizon, 5), dtype=np.float64)
    for step in range(horizon):
        token = transformed[:, step]
        open_price = previous * np.exp(np.clip(token[:, 0], -20.0, 20.0))
        close_price = open_price * np.exp(np.clip(token[:, 1], -20.0, 20.0))
        upper = np.clip(token[:, 2], 0.0, 20.0)
        lower = np.clip(token[:, 3], 0.0, 20.0)
        high_price = np.maximum(open_price, close_price) * np.exp(upper)
        low_price = np.minimum(open_price, close_price) / np.exp(lower)
        volume = np.expm1(np.clip(token[:, 4], 0.0, 30.0))
        output[:, step] = np.column_stack(
            (open_price, high_price, low_price, close_price, volume)
        )
        previous = close_price
    return output[0] if squeeze else output


def make_chronological_split(
    frame: pd.DataFrame, config: ExperimentConfig
) -> ChronologicalSplit:
    raw_test_rows = int(math.floor(len(frame) * config.test_fraction))
    test_rows = (raw_test_rows // config.horizon) * config.horizon
    if test_rows < config.context_length + config.horizon:
        raise ValueError("Test partition is too small for one evaluation window")
    development_end = len(frame) - test_rows
    development = frame.iloc[:development_end].reset_index(drop=True)
    test = frame.iloc[development_end:].reset_index(drop=True)
    development_blocks = len(development) // config.horizon
    validation_blocks = (development_blocks // 2) // config.n_folds
    validation_rows = validation_blocks * config.horizon
    initial_train_rows = len(development) - config.n_folds * validation_rows
    if validation_rows < config.context_length + config.horizon:
        raise ValueError("Validation blocks are too small for evaluation windows")
    folds = tuple(
        FoldSpec(
            index=i + 1,
            train_start=0,
            train_end=initial_train_rows + i * validation_rows,
            validation_start=initial_train_rows + i * validation_rows,
            validation_end=initial_train_rows + (i + 1) * validation_rows,
        )
        for i in range(config.n_folds)
    )
    return ChronologicalSplit(
        development=development,
        test=test,
        test_anchor_close=float(development.iloc[-1]["close"]),
        folds=folds,
    )


def build_window_dataset(
    frame: pd.DataFrame,
    scaler: ChannelStandardizer,
    config: ExperimentConfig,
    *,
    stride: int,
    previous_close: float | None = None,
) -> OHLCVWindowDataset:
    transformed, raw_offset = transform_partition(frame, previous_close)
    standardized = scaler.transform(transformed)
    raw = frame.loc[:, VALUE_COLUMNS].to_numpy(dtype=np.float64)
    return OHLCVWindowDataset(
        standardized,
        raw,
        raw_offset=raw_offset,
        context_length=config.context_length,
        horizon=config.horizon,
        stride=stride,
    )


def fit_partition_scaler(frame: pd.DataFrame) -> ChannelStandardizer:
    transformed, _ = transform_partition(frame)
    return ChannelStandardizer().fit(transformed)


def prepare_inference_context(
    raw_candles: pd.DataFrame,
    scaler: ChannelStandardizer,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, float]:
    required_rows = config.context_length + 1
    if len(raw_candles) != required_rows:
        raise ValueError(
            f"Inference requires exactly {required_rows} chronological raw candles: "
            "one transform anchor and the observed context"
        )
    transformed, offset = transform_partition(raw_candles)
    if offset != 1 or transformed.shape != (config.context_length, config.input_size):
        raise RuntimeError("Inference transformation did not produce the expected context")
    standardized = scaler.transform(transformed)
    return (
        torch.as_tensor(standardized, dtype=torch.float32),
        float(raw_candles.iloc[-1]["close"]),
    )


def build_fold_datasets(
    split: ChronologicalSplit, fold: FoldSpec, config: ExperimentConfig
) -> tuple[OHLCVWindowDataset, OHLCVWindowDataset, ChannelStandardizer]:
    train = split.development.iloc[fold.train_start : fold.train_end].reset_index(drop=True)
    validation = split.development.iloc[
        fold.validation_start : fold.validation_end
    ].reset_index(drop=True)
    scaler = fit_partition_scaler(train)
    train_dataset = build_window_dataset(
        train, scaler, config, stride=config.train_stride
    )
    validation_dataset = build_window_dataset(
        validation,
        scaler,
        config,
        stride=config.eval_stride,
        previous_close=float(train.iloc[-1]["close"]),
    )
    return train_dataset, validation_dataset, scaler


def build_final_datasets(
    split: ChronologicalSplit, config: ExperimentConfig
) -> tuple[OHLCVWindowDataset, OHLCVWindowDataset, ChannelStandardizer]:
    scaler = fit_partition_scaler(split.development)
    development_dataset = build_window_dataset(
        split.development, scaler, config, stride=config.train_stride
    )
    test_dataset = build_window_dataset(
        split.test,
        scaler,
        config,
        stride=config.eval_stride,
        previous_close=split.test_anchor_close,
    )
    return development_dataset, test_dataset, scaler
