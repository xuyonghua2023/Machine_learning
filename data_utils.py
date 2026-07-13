from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "dataset" / "processed" / "train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "dataset" / "processed" / "test.csv"
DEFAULT_TARGET_COL = "global_active_power"
DEFAULT_DATE_COL = "date"


@dataclass
class StandardScaler:
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        values = _as_2d_float(values)
        self.mean_ = np.nanmean(values, axis=0)
        self.scale_ = np.nanstd(values, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = _as_2d_float(values)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return values * self.scale_ + self.mean_

    def to_dict(self) -> Dict[str, List[float]]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return {
            "mean": self.mean_.astype(float).tolist(),
            "scale": self.scale_.astype(float).tolist(),
        }


class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        dates: Sequence[str],
        samples: Sequence[Tuple[int, int]],
        input_window: int,
        horizon: int,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32).reshape(-1)
        self.dates = np.asarray(dates)
        self.samples = list(samples)
        self.input_window = input_window
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_start, target_start = self.samples[index]
        input_end = input_start + self.input_window
        target_end = target_start + self.horizon
        x = self.features[input_start:input_end]
        y = self.target[target_start:target_end]
        return torch.from_numpy(x), torch.from_numpy(y)

    def target_dates_for_sample(self, index: int) -> np.ndarray:
        _, target_start = self.samples[index]
        return self.dates[target_start : target_start + self.horizon]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_power_frame(path: Union[Path, str], date_col: str = DEFAULT_DATE_COL) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    numeric_cols = [col for col in df.columns if col != date_col]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if numeric_cols:
        df[numeric_cols] = (
            df[numeric_cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    return add_calendar_features(df, date_col=date_col)


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = DEFAULT_DATE_COL,
) -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    day_of_year = dates.dt.dayofyear.astype(float)
    day_of_week = dates.dt.dayofweek.astype(float)
    month = dates.dt.month.astype(float)
    df["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    df["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    date_col: str = DEFAULT_DATE_COL,
) -> List[str]:
    columns = [
        col
        for col in df.columns
        if col != date_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if target_col not in columns:
        raise ValueError(f"Target column {target_col!r} was not found in data.")
    return columns


def make_window_samples(
    num_rows: int,
    input_window: int,
    horizon: int,
    stride: int = 1,
    min_target_start: int = 0,
    max_target_end: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if max_target_end is None:
        max_target_end = num_rows
    samples: List[Tuple[int, int]] = []
    max_input_start = num_rows - input_window - horizon
    for input_start in range(0, max_input_start + 1, stride):
        target_start = input_start + input_window
        target_end = target_start + horizon
        if target_start < min_target_start:
            continue
        if target_end > max_target_end:
            continue
        samples.append((input_start, target_start))
    return samples


def create_dataloaders(
    train_path: Union[Path, str] = DEFAULT_TRAIN_PATH,
    test_path: Union[Path, str] = DEFAULT_TEST_PATH,
    input_window: int = 90,
    horizon: int = 90,
    target_col: str = DEFAULT_TARGET_COL,
    date_col: str = DEFAULT_DATE_COL,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    train_stride: int = 1,
    test_stride: int = 1,
    num_workers: int = 0,
    seed: int = 2026,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    train_df = load_power_frame(train_path, date_col=date_col)
    test_df = load_power_frame(test_path, date_col=date_col)
    feature_cols = infer_feature_columns(train_df, target_col=target_col, date_col=date_col)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    feature_scaler = StandardScaler().fit(train_df[feature_cols].to_numpy())
    target_scaler = StandardScaler().fit(train_df[[target_col]].to_numpy())

    train_features = feature_scaler.transform(train_df[feature_cols].to_numpy())
    train_target = target_scaler.transform(train_df[[target_col]].to_numpy()).reshape(-1)
    combined_features = feature_scaler.transform(combined_df[feature_cols].to_numpy())
    combined_target = target_scaler.transform(combined_df[[target_col]].to_numpy()).reshape(-1)

    train_dates = _date_strings(train_df, date_col)
    combined_dates = _date_strings(combined_df, date_col)

    train_samples = make_window_samples(
        num_rows=len(train_df),
        input_window=input_window,
        horizon=horizon,
        stride=train_stride,
    )
    test_samples = make_window_samples(
        num_rows=len(combined_df),
        input_window=input_window,
        horizon=horizon,
        stride=test_stride,
        min_target_start=len(train_df),
        max_target_end=len(combined_df),
    )
    if not train_samples:
        raise ValueError("No training samples were created. Reduce horizon or input_window.")
    if not test_samples:
        raise ValueError("No test samples were created. Reduce horizon or input_window.")

    full_train_dataset = TimeSeriesWindowDataset(
        features=train_features,
        target=train_target,
        dates=train_dates,
        samples=train_samples,
        input_window=input_window,
        horizon=horizon,
    )
    test_dataset = TimeSeriesWindowDataset(
        features=combined_features,
        target=combined_target,
        dates=combined_dates,
        samples=test_samples,
        input_window=input_window,
        horizon=horizon,
    )

    train_indices, val_indices = temporal_train_val_split(
        len(full_train_dataset),
        val_ratio=val_ratio,
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metadata: Dict[str, object] = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "input_window": input_window,
        "horizon": horizon,
        "target_col": target_col,
        "date_col": date_col,
        "feature_columns": feature_cols,
        "feature_dim": len(feature_cols),
        "target_scaler": target_scaler,
        "feature_scaler": feature_scaler,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_test_samples": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, metadata


def temporal_train_val_split(
    dataset_size: int,
    val_ratio: float = 0.15,
) -> Tuple[List[int], List[int]]:
    if dataset_size < 2:
        raise ValueError("At least two samples are needed for a train/validation split.")
    val_size = max(1, int(round(dataset_size * val_ratio)))
    val_size = min(val_size, dataset_size - 1)
    split = dataset_size - val_size
    return list(range(split)), list(range(split, dataset_size))


def prediction_frame_from_arrays(
    dataset: TimeSeriesWindowDataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for sample_id in range(len(dataset)):
        dates = dataset.target_dates_for_sample(sample_id)
        for step in range(dataset.horizon):
            rows.append(
                {
                    "sample_id": sample_id,
                    "horizon_step": step + 1,
                    "date": dates[step],
                    "y_true": float(y_true[sample_id, step]),
                    "y_pred": float(y_pred[sample_id, step]),
                }
            )
    return pd.DataFrame(rows)


def _as_2d_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _date_strings(df: pd.DataFrame, date_col: str) -> np.ndarray:
    if date_col in df.columns:
        return pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d").to_numpy()
    return np.arange(len(df)).astype(str)


