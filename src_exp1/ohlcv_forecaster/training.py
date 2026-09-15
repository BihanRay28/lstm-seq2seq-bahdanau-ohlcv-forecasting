from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import (
    ChannelStandardizer,
    build_final_datasets,
    build_fold_datasets,
    load_and_validate_dataset,
    make_chronological_split,
    reconstruct_ohlcv,
)
from .evaluation import (
    calculate_raw_metrics,
    calculate_trend_metrics,
    calculate_value_accuracy,
    classify_candle_directions,
    predict_dataset,
)
from .model import Seq2SeqOHLCVForecaster
from .reporting import plot_horizon_metrics, plot_training_curves


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float | int]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_forcing_probability(epoch: int) -> float:
    if epoch < 1:
        raise ValueError("Epochs use one-based indexing")
    return 1.0 - 0.8 * min((epoch - 1) / 79.0, 1.0)


def select_device(preference: str = "auto") -> torch.device:
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device preference must be auto, cpu, or cuda")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    dataset: Dataset[Any],
    config: ExperimentConfig,
    *,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        # Disabled deliberately: on the current Windows/Python 3.14 runtime,
        # the pin-memory worker can terminate the otherwise-successful CUDA
        # process with 0xC0000409 during interpreter shutdown.
        pin_memory=False,
        generator=generator,
    )


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["context"].to(device), batch["target"].to(device)


def _batch_accuracy_statistics(
    prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
    scaler: ChannelStandardizer,
) -> tuple[int, int, float, int]:
    raw_prediction = reconstruct_ohlcv(
        prediction.detach().cpu().numpy(),
        scaler,
        batch["reference_close"].numpy(),
    )
    raw_target = batch["raw_target"].numpy()
    predicted_classes = classify_candle_directions(raw_prediction)
    actual_classes = classify_candle_directions(raw_target)
    direction_correct = int((predicted_classes == actual_classes).sum())
    direction_count = int(actual_classes.size)
    denominator = np.maximum(np.abs(raw_target), 1e-8)
    scores = np.clip(1.0 - np.abs(raw_prediction - raw_target) / denominator, 0.0, 1.0)
    return direction_correct, direction_count, float(scores.sum()), int(scores.size)


def train_one_epoch(
    model: Seq2SeqOHLCVForecaster,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    epoch: int,
    scaler: ChannelStandardizer,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    trend_correct = 0
    trend_total = 0
    value_score_sum = 0.0
    value_score_count = 0
    probability = teacher_forcing_probability(epoch)
    for batch in loader:
        context, target = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(
            context,
            target,
            teacher_forcing_probability=probability,
        )
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(context)
        total_samples += len(context)
        (
            batch_direction_correct,
            batch_direction_count,
            batch_value_sum,
            batch_value_count,
        ) = _batch_accuracy_statistics(
            prediction, batch, scaler
        )
        trend_correct += batch_direction_correct
        trend_total += batch_direction_count
        value_score_sum += batch_value_sum
        value_score_count += batch_value_count
    return (
        total_loss / total_samples,
        trend_correct / trend_total,
        value_score_sum / value_score_count,
    )


@torch.no_grad()
def validation_loss(
    model: Seq2SeqOHLCVForecaster,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    scaler: ChannelStandardizer,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    trend_correct = 0
    trend_total = 0
    value_score_sum = 0.0
    value_score_count = 0
    for batch in loader:
        context, target = _move(batch, device)
        prediction, _ = model(context)
        loss = criterion(prediction, target)
        total_loss += float(loss.item()) * len(context)
        total_samples += len(context)
        (
            batch_direction_correct,
            batch_direction_count,
            batch_value_sum,
            batch_value_count,
        ) = _batch_accuracy_statistics(
            prediction, batch, scaler
        )
        trend_correct += batch_direction_correct
        trend_total += batch_direction_count
        value_score_sum += batch_value_sum
        value_score_count += batch_value_count
    if total_samples == 0:
        raise ValueError("Validation dataset contains no samples")
    return (
        total_loss / total_samples,
        trend_correct / trend_total,
        value_score_sum / value_score_count,
    )


def fit_with_early_stopping(
    model: Seq2SeqOHLCVForecaster,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_path: Path,
    scaler: ChannelStandardizer,
    progress_label: str,
) -> FitResult:
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    progress = tqdm(
        range(1, config.max_epochs + 1),
        desc=progress_label,
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        train_loss, train_accuracy, train_value_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch, scaler
        )
        valid_loss, valid_accuracy, valid_value_accuracy = validation_loss(
            model, validation_loader, criterion, device, scaler
        )
        history.append(
            {
                "epoch": epoch,
                "teacher_forcing_probability": teacher_forcing_probability(epoch),
                "train_loss": train_loss,
                "validation_loss": valid_loss,
                "train_trend_accuracy": train_accuracy,
                "validation_trend_accuracy": valid_accuracy,
                "train_value_accuracy": train_value_accuracy,
                "validation_value_accuracy": valid_value_accuracy,
            }
        )
        progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{valid_loss:.4f}",
            train_acc=f"{100 * train_accuracy:.1f}%",
            val_acc=f"{100 * valid_accuracy:.1f}%",
            train_value=f"{100 * train_value_accuracy:.1f}%",
            val_value=f"{100 * valid_value_accuracy:.1f}%",
            tf=f"{teacher_forcing_probability(epoch):.2f}",
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return FitResult(best_epoch, best_loss, history)


def fit_fixed_epochs(
    model: Seq2SeqOHLCVForecaster,
    train_loader: DataLoader[Any],
    config: ExperimentConfig,
    device: torch.device,
    epochs: int,
    scaler: ChannelStandardizer,
    progress_label: str,
) -> list[dict[str, float | int]]:
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float | int]] = []
    progress = tqdm(
        range(1, epochs + 1),
        desc=progress_label,
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        loss, accuracy, value_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, epoch, scaler
        )
        history.append(
            {
                "epoch": epoch,
                "teacher_forcing_probability": teacher_forcing_probability(epoch),
                "train_loss": loss,
                "train_trend_accuracy": accuracy,
                "train_value_accuracy": value_accuracy,
            }
        )
        progress.set_postfix(
            train_loss=f"{loss:.4f}",
            train_acc=f"{100 * accuracy:.1f}%",
            value_acc=f"{100 * value_accuracy:.1f}%",
            tf=f"{teacher_forcing_probability(epoch):.2f}",
        )
    return history


def create_run_directory(config: ExperimentConfig, label: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = config.output_root / f"{timestamp}_{label}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "configuration.json").write_text(
        json.dumps(config.as_serializable_dict(), indent=2), encoding="utf-8"
    )
    return path


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_smoke(config: ExperimentConfig, device_preference: str = "cpu") -> Path:
    config.validate()
    seed_everything(config.seed)
    device = select_device(device_preference)
    frame = load_and_validate_dataset(config.dataset_path)
    split = make_chronological_split(frame, config)
    train_dataset, validation_dataset, scaler = build_fold_datasets(
        split, split.folds[0], config
    )
    train_subset = Subset(train_dataset, range(min(128, len(train_dataset))))
    validation_subset = Subset(
        validation_dataset, range(min(32, len(validation_dataset)))
    )
    train_loader = make_loader(train_subset, config, shuffle=True, batch_size=32)
    validation_loader = make_loader(
        validation_subset, config, shuffle=False, batch_size=32
    )
    model = Seq2SeqOHLCVForecaster(config).to(device)
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    progress = tqdm(total=1, desc="Smoke", unit="epoch", dynamic_ncols=True)
    train_loss, train_accuracy, train_value_accuracy = train_one_epoch(
        model, train_loader, optimizer, criterion, device, config, epoch=1, scaler=scaler
    )
    valid_loss, valid_accuracy, valid_value_accuracy = validation_loss(
        model, validation_loader, criterion, device, scaler
    )
    progress.set_postfix(
        train_loss=f"{train_loss:.4f}",
        val_loss=f"{valid_loss:.4f}",
        direction=f"{100 * valid_accuracy:.1f}%",
        value=f"{100 * valid_value_accuracy:.1f}%",
    )
    progress.update(1)
    progress.close()
    run_dir = create_run_directory(config, "smoke")
    scaler.save(run_dir / "scaler.json")
    torch.save(model.state_dict(), run_dir / "smoke_checkpoint.pt")
    _write_json(
        run_dir / "smoke_metrics.json",
        {
            "device": str(device),
            "training_samples": len(train_subset),
            "validation_samples": len(validation_subset),
            "train_loss": train_loss,
            "validation_loss": valid_loss,
            "train_trend_accuracy": train_accuracy,
            "validation_trend_accuracy": valid_accuracy,
            "train_value_accuracy": train_value_accuracy,
            "validation_value_accuracy": valid_value_accuracy,
        },
    )
    plot_training_curves(
        [
            {
                "epoch": 1,
                "train_loss": train_loss,
                "validation_loss": valid_loss,
                "train_trend_accuracy": train_accuracy,
                "validation_trend_accuracy": valid_accuracy,
                "train_value_accuracy": train_value_accuracy,
                "validation_value_accuracy": valid_value_accuracy,
            }
        ],
        run_dir / "training_curves.png",
        title="Smoke-run loss, directional accuracy, and value accuracy",
    )
    return run_dir


def run_full_training(config: ExperimentConfig) -> Path:
    config.validate()
    seed_everything(config.seed)
    device = select_device()
    frame = load_and_validate_dataset(config.dataset_path)
    split = make_chronological_split(frame, config)
    run_dir = create_run_directory(config, "full")
    fold_summaries: list[dict[str, float | int]] = []
    best_epochs: list[int] = []

    for fold in split.folds:
        fold_dir = run_dir / f"fold_{fold.index}"
        fold_dir.mkdir()
        train_dataset, valid_dataset, scaler = build_fold_datasets(
            split, fold, config
        )
        scaler.save(fold_dir / "scaler.json")
        seed_everything(config.seed + fold.index)
        model = Seq2SeqOHLCVForecaster(config).to(device)
        result = fit_with_early_stopping(
            model,
            make_loader(train_dataset, config, shuffle=True),
            make_loader(valid_dataset, config, shuffle=False),
            config,
            device,
            fold_dir / "best_checkpoint.pt",
            scaler,
            f"Fold {fold.index}/{config.n_folds}",
        )
        _write_json(fold_dir / "history.json", result.history)
        plot_training_curves(
            result.history,
            fold_dir / "training_curves.png",
            title=f"Walk-forward fold {fold.index}",
        )
        best_epochs.append(result.best_epoch)
        fold_summaries.append(
            {
                "fold": fold.index,
                "training_samples": len(train_dataset),
                "validation_samples": len(valid_dataset),
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
            }
        )

    selected_epochs = max(1, int(median(best_epochs)))
    development_dataset, test_dataset, final_scaler = build_final_datasets(
        split, config
    )
    final_scaler.save(run_dir / "final_scaler.json")
    seed_everything(config.seed)
    final_model = Seq2SeqOHLCVForecaster(config).to(device)
    final_history = fit_fixed_epochs(
        final_model,
        make_loader(development_dataset, config, shuffle=True),
        config,
        device,
        selected_epochs,
        final_scaler,
        "Final development fit",
    )
    torch.save(final_model.state_dict(), run_dir / "final_checkpoint.pt")
    _write_json(run_dir / "final_history.json", final_history)
    plot_training_curves(
        final_history,
        run_dir / "final_training_curves.png",
        title="Final training on complete development period",
    )

    test_loader = make_loader(test_dataset, config, shuffle=False)
    test_outputs = predict_dataset(final_model, test_loader, final_scaler, device)
    model_metrics = calculate_raw_metrics(
        test_outputs["raw_predictions"], test_outputs["raw_targets"]
    )
    persistence_metrics = calculate_raw_metrics(
        test_outputs["persistence_predictions"], test_outputs["raw_targets"]
    )
    trend_metrics = calculate_trend_metrics(
        test_outputs["raw_predictions"],
        test_outputs["raw_targets"],
        test_outputs["reference_ohlcv"],
    )
    value_accuracy = calculate_value_accuracy(
        test_outputs["raw_predictions"], test_outputs["raw_targets"]
    )
    np.savez_compressed(run_dir / "test_predictions_and_attention.npz", **test_outputs)
    _write_json(
        run_dir / "metrics.json",
        {
            "selected_final_epochs": selected_epochs,
            "folds": fold_summaries,
            "model": model_metrics,
            "persistence": persistence_metrics,
            "trend": trend_metrics,
            "value_accuracy": value_accuracy,
        },
    )
    plot_horizon_metrics(
        model_metrics,
        persistence_metrics,
        trend_metrics,
        value_accuracy,
        run_dir / "test_horizon_curves.png",
    )
    return run_dir
