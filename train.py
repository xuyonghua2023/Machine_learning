from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from data_utils import (
    DEFAULT_TARGET_COL,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    create_dataloaders,
    prediction_frame_from_arrays,
    set_seed,
)
from models.lstm import LSTMRegressor
from models.proposed import ProposedModel
from models.transformer import TransformerRegressor

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; training still works without it.
    tqdm = None


@dataclass
class TrainConfig:
    model: str = "lstm"
    horizon: int = 90
    input_window: int = 90
    train_path: str = str(DEFAULT_TRAIN_PATH)
    test_path: str = str(DEFAULT_TEST_PATH)
    output_dir: str = "outputs"
    target_col: str = DEFAULT_TARGET_COL
    seed: int = 2026
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    grad_clip: float = 1.0
    val_ratio: float = 0.15
    train_stride: int = 1
    test_stride: int = 1
    num_workers: int = 0
    hidden_dim: int = 128
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    device: str = "auto"
    log_interval: int = 10
    progress: bool = True


def build_model(config: TrainConfig, input_dim: int) -> nn.Module:
    model_name = config.model.lower()
    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            horizon=config.horizon,
            hidden_dim=config.hidden_dim,
            num_layers=max(1, config.num_layers),
            dropout=config.dropout,
        )
    if model_name == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            horizon=config.horizon,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
    if model_name == "proposed":
        return ProposedModel(
            input_dim=input_dim,
            horizon=config.horizon,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown model: {config.model}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    progress: bool = True,
    description: str = "Training",
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    iterator = make_progress_bar(loader, enabled=progress, desc=description, leave=False)
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_scaler,
    progress: bool = True,
    description: str = "Evaluating",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    losses: List[float] = []
    y_true_norm: List[np.ndarray] = []
    y_pred_norm: List[np.ndarray] = []
    iterator = make_progress_bar(loader, enabled=progress, desc=description, leave=False)
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        losses.append(loss.item() * x.size(0))
        y_true_norm.append(y.detach().cpu().numpy())
        y_pred_norm.append(pred.detach().cpu().numpy())
        if tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{loss.item():.4f}")

    true_norm = np.concatenate(y_true_norm, axis=0)
    pred_norm = np.concatenate(y_pred_norm, axis=0)
    true = inverse_target(true_norm, target_scaler)
    pred = inverse_target(pred_norm, target_scaler)
    metrics = {
        "loss": float(np.sum(losses) / max(len(loader.dataset), 1)),
        "mse": float(np.mean((pred - true) ** 2)),
        "mae": float(np.mean(np.abs(pred - true))),
    }
    return metrics, true, pred


def inverse_target(values: np.ndarray, target_scaler) -> np.ndarray:
    original_shape = values.shape
    restored = target_scaler.inverse_transform(values.reshape(-1, 1))
    return restored.reshape(original_shape)


def make_progress_bar(iterable, enabled: bool, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, dynamic_ncols=True, **kwargs)
    return iterable


def run_one_experiment(config: TrainConfig) -> Dict[str, object]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    history_dir = output_dir / "history"
    for directory in [checkpoints_dir, metrics_dir, predictions_dir, history_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        train_path=config.train_path,
        test_path=config.test_path,
        input_window=config.input_window,
        horizon=config.horizon,
        target_col=config.target_col,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        train_stride=config.train_stride,
        test_stride=config.test_stride,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    model = build_model(config, input_dim=int(metadata["feature_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, config.patience // 3),
    )
    criterion = nn.MSELoss()
    target_scaler = metadata["target_scaler"]

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    stale_epochs = 0
    history: List[Dict[str, float]] = []

    epoch_iterator = make_progress_bar(
        range(1, config.epochs + 1),
        enabled=config.progress,
        desc=f"{config.model} H={config.horizon} seed={config.seed}",
        leave=True,
    )
    for epoch in epoch_iterator:
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip=config.grad_clip,
            progress=config.progress,
            description=f"epoch {epoch}/{config.epochs} train",
        )
        val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            target_scaler=target_scaler,
            progress=config.progress,
            description=f"epoch {epoch}/{config.epochs} val",
        )
        scheduler.step(val_metrics["loss"])
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_mse": float(val_metrics["mse"]),
            "val_mae": float(val_metrics["mae"]),
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if tqdm is not None and hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_metrics['loss']:.4f}",
                best=f"{best_val_loss:.4f}",
                stale=stale_epochs,
            )

        if config.log_interval > 0 and epoch % config.log_interval == 0:
            print(
                f"[{config.model} h={config.horizon} seed={config.seed}] "
                f"epoch={epoch:03d} train_loss={train_loss:.5f} "
                f"val_loss={val_metrics['loss']:.5f}"
            )
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    test_metrics, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        target_scaler=target_scaler,
        progress=config.progress,
        description=f"{config.model} H={config.horizon} test",
    )

    run_name = f"{config.model}_h{config.horizon}_seed{config.seed}"
    checkpoint_path = checkpoints_dir / f"{run_name}.pt"
    metrics_path = metrics_dir / f"{run_name}.json"
    prediction_path = predictions_dir / f"{run_name}.csv"
    history_path = history_dir / f"{run_name}.csv"

    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "feature_columns": metadata["feature_columns"],
            "target_scaler": target_scaler.to_dict(),
            "feature_scaler": metadata["feature_scaler"].to_dict(),
        },
        checkpoint_path,
    )
    prediction_frame_from_arrays(test_loader.dataset, y_true, y_pred).to_csv(
        prediction_path,
        index=False,
    )
    pd.DataFrame(history).to_csv(history_path, index=False)

    metrics: Dict[str, object] = {
        "model": config.model,
        "horizon": config.horizon,
        "seed": config.seed,
        "input_window": config.input_window,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_loss": test_metrics["loss"],
        "num_train_samples": metadata["num_train_samples"],
        "num_val_samples": metadata["num_val_samples"],
        "num_test_samples": metadata["num_test_samples"],
        "feature_dim": metadata["feature_dim"],
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(prediction_path),
        "history_path": str(history_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train one power forecasting model.")
    parser.add_argument("--model", choices=["lstm", "transformer", "proposed"], default="lstm")
    parser.add_argument("--horizon", type=int, choices=[90, 365], default=90)
    parser.add_argument("--input-window", type=int, default=90)
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="progress",
        help="Disable tqdm progress bars.",
    )
    parser.set_defaults(progress=True)
    return TrainConfig(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()
    metrics = run_one_experiment(config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

