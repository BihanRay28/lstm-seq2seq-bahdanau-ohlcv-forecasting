from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .data import (
    build_final_datasets,
    load_and_validate_dataset,
    make_chronological_split,
)
from .training import run_full_training, run_smoke


def validate_data(config: ExperimentConfig) -> None:
    config.validate()
    frame = load_and_validate_dataset(config.dataset_path)
    split = make_chronological_split(frame, config)
    development, test, _ = build_final_datasets(split, config)
    print(f"Dataset: {config.dataset_path}")
    print(f"Raw rows: {len(frame):,}")
    print(f"Range: {frame.iloc[0]['datetime']} to {frame.iloc[-1]['datetime']}")
    print(f"Development rows: {len(split.development):,}")
    print(f"Test rows: {len(split.test):,}")
    print(f"Final development samples: {len(development):,}")
    print(f"Test samples: {len(test):,}")
    for fold in split.folds:
        validation_rows = fold.validation_end - fold.validation_start
        print(
            f"Fold {fold.index}: train rows={fold.train_end:,}, "
            f"validation rows={validation_rows:,}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BiLSTM Seq2Seq Bahdanau OHLCV forecasting experiment"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("validate-data", help="Validate data and display folds")
    smoke_parser = subparsers.add_parser(
        "smoke", help="Run one short end-to-end training pass"
    )
    smoke_parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="cpu"
    )
    train_parser = subparsers.add_parser("train", help="Run explicit training")
    train_parser.add_argument("--mode", choices=("full",), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig()
    if args.command == "validate-data":
        validate_data(config)
    elif args.command == "train":
        output = run_full_training(config)
        print(f"Full training artifacts: {output}")
    else:
        device = getattr(args, "device", "cpu")
        output = run_smoke(config, device_preference=device)
        print(f"Smoke-run artifacts: {output}")


if __name__ == "__main__":
    main()
