from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; experiments still run without it.
    tqdm = None

from train import TrainConfig, run_one_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all assignment experiments.")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "proposed"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable tqdm progress bars.",
    )
    parser.set_defaults(progress=True)
    return parser.parse_args()


def make_progress_bar(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, dynamic_ncols=True, **kwargs)
    return iterable


def summarize(results: List[dict], output_dir: Path) -> pd.DataFrame:
    runs = pd.DataFrame(results)
    runs_path = output_dir / "experiment_runs.csv"
    runs.to_csv(runs_path, index=False)

    summary = (
        runs.groupby(["model", "horizon"], as_index=False)
        .agg(
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", "std"),
            best_epoch_mean=("best_epoch", "mean"),
            runs=("seed", "count"),
        )
        .fillna(0.0)
    )
    summary_path = output_dir / "experiment_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        device=args.device,
        progress=args.progress,
    )
    if args.train_path is not None:
        base_config.train_path = args.train_path
    if args.test_path is not None:
        base_config.test_path = args.test_path

    jobs = [
        (horizon, model, seed)
        for horizon in args.horizons
        for model in args.models
        for seed in args.seeds
    ]
    results: List[dict] = []
    iterator = make_progress_bar(
        jobs,
        enabled=args.progress,
        desc="All experiments",
        leave=True,
    )
    for horizon, model, seed in iterator:
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(model=model, horizon=horizon, seed=seed)
        print(f"Running model={model}, horizon={horizon}, seed={seed}")
        config = replace(base_config, model=model, horizon=horizon, seed=seed)
        metrics = run_one_experiment(config)
        results.append(metrics)

    summary = summarize(results, output_dir)
    print(json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

