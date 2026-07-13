#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 UCI 家庭用电分钟级 TXT 与法国 Météo-France 月度气象 CSV 合并为日级数据。

运行示例：
python merge_power_weather.py \
    --power household_power_consumption.txt \
    --weather MENSQ_92_previous-1950-2024.csv \
    --output household_power_weather_daily.csv

如需按时间切分训练集和测试集：
python merge_power_weather.py \
    --power household_power_consumption.txt \
    --weather MENSQ_92_previous-1950-2024.csv \
    --output household_power_weather_daily.csv \
    --test-start 2010-01-01

说明：
1. 两个原始文件都使用分号 ; 分隔。
2. 用电数据按作业要求聚合：
   - global_active_power、global_reactive_power、sub_metering_1/2/3：每日求和
   - voltage、global_intensity：每日求平均
3. 气象数据是月度数据，按“年月”重复合并到该月的每一天。
4. 默认不将 RR 除以 10。当前官方 CSV 中 RR=67.8、83.5 等已符合毫米量级。
   只有确认你的 RR 是“整数形式的十分之一毫米”时，才添加 --rr-divide-by-10。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


POWER_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "voltage",
    "global_intensity",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]

WEATHER_COLUMNS = [
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]

# 用电采集地点 Sceaux 的近似中心坐标，仅用于站点覆盖相同时的距离排序。
SCEAUX_LAT = 48.778
SCEAUX_LON = 2.290


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两个经纬度点之间的大圆距离，单位 km。"""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(a))


def read_power_txt(path: Path) -> pd.DataFrame:
    """读取并清洗 UCI 分钟级用电数据。"""
    print(f"[1/6] 读取用电文件：{path}")

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?", "", "NA", "NaN"],
        low_memory=False,
    )
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"date", "time", *POWER_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"用电文件缺少字段：{missing}")

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip()
        + " "
        + df["time"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).copy()

    for col in POWER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[["datetime", *POWER_COLUMNS]]
        .sort_values("datetime")
        .drop_duplicates(subset="datetime", keep="first")
        .set_index("datetime")
    )

    # 记录插值前 global_active_power 的有效分钟数，便于检查数据质量。
    raw_valid = df["global_active_power"].notna().astype("int16")

    # 构造完整分钟索引。UCI 数据量约 200 万行，普通内存一般可以处理。
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="min")
    df = df.reindex(full_index)
    df.index.name = "datetime"
    raw_valid = raw_valid.reindex(full_index, fill_value=0)

    # 对少量分钟缺失做时间插值；首尾残留值使用前后值补齐。
    # 原始 UCI 数据缺失比例较低，此方法足够用于课程项目。
    df[POWER_COLUMNS] = (
        df[POWER_COLUMNS]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )

    df["raw_valid_minute"] = raw_valid
    return df


def aggregate_power_daily(power_minute: pd.DataFrame) -> pd.DataFrame:
    """按照作业规定，将分钟级用电数据聚合为日级数据。"""
    print("[2/6] 将分钟级用电数据聚合为日级数据")

    aggregation = {
        "global_active_power": "sum",
        "global_reactive_power": "sum",
        "voltage": "mean",
        "global_intensity": "mean",
        "sub_metering_1": "sum",
        "sub_metering_2": "sum",
        "sub_metering_3": "sum",
        "raw_valid_minute": "sum",
    }

    daily = power_minute.resample("D").agg(aggregation)
    daily = daily.rename(columns={"raw_valid_minute": "power_valid_minutes"})

    # 删除原始数据开头和结尾可能存在的“不完整自然日”。
    # 中间日期即使有少量缺失，也已在分钟级插值。
    first_day = power_minute.index.min().normalize()
    last_day = power_minute.index.max().normalize()
    if power_minute.index.min() != first_day:
        daily = daily[daily.index != first_day]
    if power_minute.index.max() != last_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1):
        daily = daily[daily.index != last_day]

    # 其他未被三个分表覆盖的每日用电量，单位为 Wh。
    daily["sub_metering_remainder"] = (
        daily["global_active_power"] * 1000.0 / 60.0
        - daily["sub_metering_1"]
        - daily["sub_metering_2"]
        - daily["sub_metering_3"]
    )

    # 因插值或测量误差产生极小负数时，截断为 0。
    daily["sub_metering_remainder"] = daily["sub_metering_remainder"].clip(lower=0)

    return daily


def read_weather_csv(path: Path) -> pd.DataFrame:
    """读取 Météo-France 月度气象文件。"""
    print(f"[3/6] 读取气象文件：{path}")

    weather = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    weather.columns = [str(c).strip() for c in weather.columns]

    required = {
        "NUM_POSTE",
        "NOM_USUEL",
        "LAT",
        "LON",
        "AAAAMM",
        *WEATHER_COLUMNS,
    }
    missing = sorted(required.difference(weather.columns))
    if missing:
        raise ValueError(f"气象文件缺少字段：{missing}")

    numeric_columns = [
        "NUM_POSTE",
        "LAT",
        "LON",
        "AAAAMM",
        *WEATHER_COLUMNS,
    ]
    for col in numeric_columns:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather.dropna(subset=["NUM_POSTE", "AAAAMM"]).copy()
    weather["NUM_POSTE"] = weather["NUM_POSTE"].astype("int64")
    weather["AAAAMM"] = weather["AAAAMM"].astype("int64")
    weather["month"] = weather["AAAAMM"] % 100

    return weather


def choose_station(
    weather: pd.DataFrame,
    start_yyyymm: int,
    end_yyyymm: int,
    forced_station_id: int | None = None,
) -> int:
    """
    选择气象站：
    1. 优先保证 RR、NBJRR1、NBJRR5、NBJRR10 的月份覆盖完整；
    2. 覆盖相同时，选择更靠近 Sceaux 的站点。
    NBJBROU 在 92 省的目标年份中普遍严重缺失，因此不用于主排序。
    """
    period = weather[
        weather["AAAAMM"].between(start_yyyymm, end_yyyymm)
    ].copy()

    if forced_station_id is not None:
        if forced_station_id not in set(period["NUM_POSTE"]):
            raise ValueError(
                f"指定站点 {forced_station_id} 在 {start_yyyymm}—{end_yyyymm} 无数据"
            )
        return forced_station_id

    rain_cols = ["RR", "NBJRR1", "NBJRR5", "NBJRR10"]
    rows = []

    for station_id, group in period.groupby("NUM_POSTE"):
        station_name = str(group["NOM_USUEL"].dropna().iloc[0])
        lat = float(group["LAT"].dropna().iloc[0])
        lon = float(group["LON"].dropna().iloc[0])

        month_count = int(group["AAAAMM"].nunique())
        complete_rain_months = int(group[rain_cols].notna().all(axis=1).sum())
        fog_months = int(group["NBJBROU"].notna().sum())
        distance = haversine_km(SCEAUX_LAT, SCEAUX_LON, lat, lon)

        rows.append(
            {
                "station_id": int(station_id),
                "station_name": station_name,
                "month_count": month_count,
                "complete_rain_months": complete_rain_months,
                "fog_months": fog_months,
                "distance_km": distance,
            }
        )

    if not rows:
        raise ValueError("目标日期范围内没有可用气象站")

    ranking = pd.DataFrame(rows).sort_values(
        by=["complete_rain_months", "month_count", "distance_km"],
        ascending=[False, False, True],
    )

    print("\n气象站候选排序（前 10 个）：")
    print(ranking.head(10).to_string(index=False))

    selected = ranking.iloc[0]
    print(
        "\n自动选择气象站："
        f"{selected['station_name']} "
        f"(NUM_POSTE={int(selected['station_id'])}, "
        f"距离约 {selected['distance_km']:.1f} km)"
    )
    return int(selected["station_id"])


def prepare_monthly_weather(
    weather: pd.DataFrame,
    station_id: int,
    start_month: pd.Timestamp,
    end_month: pd.Timestamp,
    rr_divide_by_10: bool,
) -> pd.DataFrame:
    """
    提取单一站点月度数据，并填补缺失月份/字段。

    缺失填补规则：
    - 先使用该站点在目标期开始之前的同月历史均值；
    - 若该站点历史仍不可用，则使用 92 省全部站点在目标期开始之前的同月历史均值；
    - 最后再使用目标期开始之前的总体均值。
    这样不会使用 2011—2024 年的未来信息填补 2006—2010 年数据。
    """
    print("[4/6] 整理月度气象数据并处理缺失值")

    start_yyyymm = int(start_month.strftime("%Y%m"))
    end_yyyymm = int(end_month.strftime("%Y%m"))

    station_all = weather[weather["NUM_POSTE"] == station_id].copy()
    station_name = str(station_all["NOM_USUEL"].dropna().iloc[0])

    target = station_all[
        station_all["AAAAMM"].between(start_yyyymm, end_yyyymm)
    ][["AAAAMM", *WEATHER_COLUMNS]].copy()

    target["month_start"] = pd.to_datetime(
        target["AAAAMM"].astype(str), format="%Y%m"
    )
    target = (
        target.sort_values("month_start")
        .drop_duplicates("month_start", keep="first")
        .set_index("month_start")
    )

    full_months = pd.date_range(start_month, end_month, freq="MS")
    target = target.reindex(full_months)
    target.index.name = "month_start"
    target["AAAAMM"] = target.index.strftime("%Y%m").astype(int)
    target["month"] = target.index.month

    # 仅使用目标期开始之前的数据构造历史气候均值，避免未来信息泄漏。
    station_history = station_all[station_all["AAAAMM"] < start_yyyymm].copy()
    department_history = weather[weather["AAAAMM"] < start_yyyymm].copy()

    station_climatology = station_history.groupby("month")[WEATHER_COLUMNS].mean()
    department_climatology = department_history.groupby("month")[WEATHER_COLUMNS].mean()
    department_overall = department_history[WEATHER_COLUMNS].mean()

    for col in WEATHER_COLUMNS:
        flag_col = f"{col}_imputed"
        target[flag_col] = target[col].isna().astype("int8")

        # 同一站点、同一月份的历史均值
        target[col] = target[col].fillna(
            target["month"].map(station_climatology[col])
        )
        # 92 省全部站点、同一月份的历史均值
        target[col] = target[col].fillna(
            target["month"].map(department_climatology[col])
        )
        # 最后使用历史总体均值
        target[col] = target[col].fillna(department_overall[col])

    # 天数类字段允许保持浮点数，因为历史均值填补后可能不是整数；
    # 作为机器学习输入无需强行取整。
    if rr_divide_by_10:
        target["RR"] = target["RR"] / 10.0
        print("已按命令参数将 RR 除以 10。")
    else:
        print("RR 保持官方 CSV 原值，不除以 10。")

    print(f"气象站名称：{station_name}")
    print("目标时期气象字段的填补月份数：")
    for col in WEATHER_COLUMNS:
        print(f"  {col}: {int(target[f'{col}_imputed'].sum())} 个月")

    output_cols = [
        "AAAAMM",
        *WEATHER_COLUMNS,
        *[f"{c}_imputed" for c in WEATHER_COLUMNS],
    ]
    return target[output_cols].reset_index()


def merge_daily_power_weather(
    daily_power: pd.DataFrame,
    monthly_weather: pd.DataFrame,
) -> pd.DataFrame:
    """按年月将月度天气重复合并到每日用电数据。"""
    print("[5/6] 按年月合并每日用电与月度天气")

    daily = daily_power.reset_index().rename(columns={"datetime": "date"})
    if "date" not in daily.columns:
        # resample 后索引名一般仍为 datetime；兼容其他情况。
        daily = daily.rename(columns={daily.columns[0]: "date"})

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month_start"] = daily["date"].dt.to_period("M").dt.to_timestamp()

    merged = daily.merge(monthly_weather, on="month_start", how="left")

    missing_weather_rows = merged[WEATHER_COLUMNS].isna().any(axis=1).sum()
    if missing_weather_rows:
        raise ValueError(
            f"合并后仍有 {missing_weather_rows} 行天气字段缺失，请检查时间范围"
        )

    # 可直接作为模型输入的日历特征。
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["day_of_week"] = merged["date"].dt.dayofweek
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype("int8")

    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12.0)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12.0)
    merged["dow_sin"] = np.sin(2 * np.pi * merged["day_of_week"] / 7.0)
    merged["dow_cos"] = np.cos(2 * np.pi * merged["day_of_week"] / 7.0)
    merged["doy_sin"] = np.sin(2 * np.pi * merged["day_of_year"] / 365.25)
    merged["doy_cos"] = np.cos(2 * np.pi * merged["day_of_year"] / 365.25)

    merged = merged.drop(columns=["month_start"])

    ordered_columns = [
        "date",
        "global_active_power",
        "global_reactive_power",
        "voltage",
        "global_intensity",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
        "sub_metering_remainder",
        "RR",
        "NBJRR1",
        "NBJRR5",
        "NBJRR10",
        "NBJBROU",
        # "power_valid_minutes",
        # *[f"{c}_imputed" for c in WEATHER_COLUMNS],
        # "year",
        # "month",
        # "day_of_week",
        # "day_of_year",
        # "is_weekend",
        # "month_sin",
        # "month_cos",
        # "dow_sin",
        # "dow_cos",
        # "doy_sin",
        # "doy_cos",
        # "AAAAMM",
    ]
    return merged[ordered_columns].sort_values("date").reset_index(drop=True)


def save_outputs(
    merged: pd.DataFrame,
    output_path: Path,
    test_start: str | None,
) -> None:
    """保存完整日级 CSV；可选按日期切分 train/test。"""
    print("[6/6] 保存结果")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"完整数据已保存：{output_path}")
    print(f"数据形状：{merged.shape}")
    print(f"日期范围：{merged['date'].min().date()} 至 {merged['date'].max().date()}")

    if test_start is not None:
        split_date = pd.Timestamp(test_start)
        train = merged[merged["date"] < split_date].copy()
        test = merged[merged["date"] >= split_date].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"--test-start={test_start} 导致训练集或测试集为空"
            )

        train_path = output_path.with_name("train.csv")
        test_path = output_path.with_name("test.csv")
        train.to_csv(train_path, index=False, encoding="utf-8-sig")
        test.to_csv(test_path, index=False, encoding="utf-8-sig")
        print(f"训练集已保存：{train_path}，共 {len(train)} 天")
        print(f"测试集已保存：{test_path}，共 {len(test)} 天")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并 UCI 家庭用电 TXT 与 Météo-France 月度气象 CSV"
    )
    parser.add_argument("--power", type=Path, 
        default=r"D:\__Projects\__A_Important\MachineLearning\dataset\MENSQ_92_previous-1950-2024.csv\household_power_consumption.txt",
        help="用电 TXT 文件路径"
    )
    parser.add_argument("--weather", type=Path,
        default=r"D:\__Projects\__A_Important\MachineLearning\dataset\MENSQ_92_previous-1950-2024.csv\MENSQ_92_previous-1950-2024.csv",
        help="月度气象 CSV 路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"D:\__Projects\__A_Important\MachineLearning\dataset\processed\household_power_weather_daily.csv"),
        help="输出完整日级 CSV 路径",
    )
    parser.add_argument(
        "--station-id",
        type=int,
        default=None,
        help="可选：指定 NUM_POSTE；默认按覆盖率和距离自动选择",
    )
    parser.add_argument(
        "--rr-divide-by-10",
        type=bool,
        default=True,
        help="仅在确认 RR 为十分之一毫米的整数值时使用",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default="2009-11-26",
        help="可选：按日期切分 train.csv/test.csv，例如 2010-01-01",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.power.exists():
        raise FileNotFoundError(f"找不到用电文件：{args.power}")
    if not args.weather.exists():
        raise FileNotFoundError(f"找不到气象文件：{args.weather}")

    power_minute = read_power_txt(args.power)
    daily_power = aggregate_power_daily(power_minute)
    weather = read_weather_csv(args.weather)

    start_month = daily_power.index.min().to_period("M").to_timestamp()
    end_month = daily_power.index.max().to_period("M").to_timestamp()
    start_yyyymm = int(start_month.strftime("%Y%m"))
    end_yyyymm = int(end_month.strftime("%Y%m"))

    station_id = choose_station(
        weather,
        start_yyyymm=start_yyyymm,
        end_yyyymm=end_yyyymm,
        forced_station_id=args.station_id,
    )

    monthly_weather = prepare_monthly_weather(
        weather,
        station_id=station_id,
        start_month=start_month,
        end_month=end_month,
        rr_divide_by_10=args.rr_divide_by_10,
    )

    merged = merge_daily_power_weather(daily_power, monthly_weather)
    save_outputs(merged, args.output, args.test_start)


if __name__ == "__main__":
    main()
