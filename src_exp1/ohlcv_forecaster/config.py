from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PACKAGE_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = PACKAGE_FILE.parents[2]
DATASET_PATH = (
    EXPERIMENT_ROOT
    / "data"
    / "BTC_USDT_30m_Binance_20260913_184821.csv"
)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_path: Path = DATASET_PATH
    output_root: Path = EXPERIMENT_ROOT / "outputs"
    context_length: int = 48
    horizon: int = 24
    train_stride: int = 1
    eval_stride: int = 24
    test_fraction: float = 0.10
    n_folds: int = 5
    input_size: int = 5
    encoder_hidden_size: int = 128
    decoder_hidden_size: int = 128
    num_layers: int = 2
    attention_dim: int = 128
    head_hidden_size: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    seed: int = 42

    def as_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["dataset_path"] = str(self.dataset_path)
        values["output_root"] = str(self.output_root)
        return values

    def validate(self) -> None:
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Cemented dataset path does not exist: {self.dataset_path}"
            )
        if self.context_length <= 0 or self.horizon <= 0:
            raise ValueError("Context length and horizon must be positive")
        if self.n_folds <= 0:
            raise ValueError("At least one walk-forward fold is required")
