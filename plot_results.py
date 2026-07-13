from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_prediction_curve(
    prediction_path: Path,
    output_path: Path,
    sample_id: int = 0,
) -> None:
    df = pd.read_csv(prediction_path)
    available_samples = sorted(df["sample_id"].unique())
    if sample_id not in available_samples:
        sample_id = available_samples[0]
    sample = df[df["sample_id"] == sample_id].copy()
    sample["date"] = pd.to_datetime(sample["date"])

    plt.figure(figsize=(12, 5))
    plt.plot(sample["date"], sample["y_true"], label="Ground Truth", linewidth=2)
    plt.plot(sample["date"], sample["y_pred"], label="Prediction", linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Global active power")
    plt.title(f"Power Forecast vs Ground Truth (sample {sample_id})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_summary(summary_path: Path, output_path: Path, metric: str = "mse") -> None:
    summary = pd.read_csv(summary_path)
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in summary.columns or std_col not in summary.columns:
        raise ValueError(f"Summary file does not contain {mean_col!r} and {std_col!r}.")

    labels = [f"{row.model}\nH={row.horizon}" for row in summary.itertuples()]
    x = range(len(summary))
    plt.figure(figsize=(max(9, len(summary) * 1.4), 5))
    plt.bar(x, summary[mean_col], yerr=summary[std_col], capsize=5)
    plt.xticks(list(x), labels)
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} mean and std over repeated runs")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot assignment result figures.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="proposed")
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--metric", choices=["mse", "mae"], default="mse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prediction_path = (
        output_dir
        / "predictions"
        / f"{args.model}_h{args.horizon}_seed{args.seed}.csv"
    )
    prediction_output = (
        output_dir
        / "figures"
        / f"{args.model}_h{args.horizon}_seed{args.seed}_sample{args.sample_id}.png"
    )
    plot_prediction_curve(
        prediction_path=prediction_path,
        output_path=prediction_output,
        sample_id=args.sample_id,
    )

    summary_path = output_dir / "experiment_summary.csv"
    if summary_path.exists():
        plot_metric_summary(
            summary_path=summary_path,
            output_path=output_dir / "figures" / f"summary_{args.metric}.png",
            metric=args.metric,
        )


if __name__ == "__main__":
    main()

