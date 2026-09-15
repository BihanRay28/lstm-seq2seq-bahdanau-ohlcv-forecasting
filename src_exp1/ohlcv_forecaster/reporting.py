from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(
    history: Sequence[dict[str, Any]], path: Path, *, title: str
) -> None:
    if not history:
        raise ValueError("Cannot plot an empty training history")
    epochs = [int(row["epoch"]) for row in history]
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.8))

    loss_axis.plot(
        epochs, [row["train_loss"] for row in history], marker="o", label="Training"
    )
    if "validation_loss" in history[0]:
        loss_axis.plot(
            epochs,
            [row["validation_loss"] for row in history],
            marker="o",
            label="Validation",
        )
    loss_axis.set_title("Smooth L1 loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    accuracy_axis.plot(
        epochs,
        [100.0 * row["train_trend_accuracy"] for row in history],
        marker="o",
        label="Training direction",
    )
    accuracy_axis.plot(
        epochs,
        [100.0 * row["train_value_accuracy"] for row in history],
        marker="o",
        linestyle="--",
        label="Training value",
    )
    if "validation_trend_accuracy" in history[0]:
        accuracy_axis.plot(
            epochs,
            [100.0 * row["validation_trend_accuracy"] for row in history],
            marker="o",
            label="Validation direction",
        )
        accuracy_axis.plot(
            epochs,
            [100.0 * row["validation_value_accuracy"] for row in history],
            marker="o",
            linestyle="--",
            label="Validation value",
        )
    accuracy_axis.set_title("Bullish/bearish/doji and value accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_horizon_metrics(
    model_metrics: dict[str, Any],
    persistence_metrics: dict[str, Any],
    trend_metrics: dict[str, Any],
    value_accuracy: dict[str, Any],
    path: Path,
) -> None:
    horizons = range(1, len(trend_metrics["candle_accuracy_by_horizon"]) + 1)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for field in ("open", "high", "low", "close"):
        axes[0, 0].plot(
            horizons,
            model_metrics["by_field"][field]["mae_by_horizon"],
            label=field.upper(),
        )
        axes[0, 1].plot(
            horizons,
            model_metrics["by_field"][field]["rmse_by_horizon"],
            label=field.upper(),
        )
    axes[0, 0].set_title("Model price MAE by forecast horizon")
    axes[0, 1].set_title("Model price RMSE by forecast horizon")
    for axis in axes[0]:
        axis.set_xlabel("30-minute forecast step")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)

    axes[1, 0].plot(
        horizons,
        model_metrics["by_field"]["close"]["mae_by_horizon"],
        label="Model",
    )
    axes[1, 0].plot(
        horizons,
        persistence_metrics["by_field"]["close"]["mae_by_horizon"],
        label="Persistence",
    )
    axes[1, 0].set_title("Close MAE: model vs persistence")
    axes[1, 0].set_xlabel("30-minute forecast step")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(
        horizons,
        [100.0 * value for value in trend_metrics["candle_accuracy_by_horizon"]],
        color="tab:green",
        label="Directional",
    )
    axes[1, 1].plot(
        horizons,
        [100.0 * value for value in value_accuracy["macro_accuracy_by_horizon"]],
        color="tab:blue",
        linestyle="--",
        label="Value",
    )
    axes[1, 1].axhline(50.0, color="grey", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Accuracy by horizon")
    axes[1, 1].set_xlabel("30-minute forecast step")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    figure.suptitle("Final test forecasting performance")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
