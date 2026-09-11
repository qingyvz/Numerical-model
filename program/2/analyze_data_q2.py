"""C 题问题二的原始数据规律与信息边界分析。

本脚本只做描述性统计：读取附件 1、附件 2 和附件 5 模板，不修改源文件，
也不在本步骤生成预测或场景。问题二的滚动预测和场景优化由
``q2_causal_core.py`` 单独实现；两者共同遵守“第 d 日计划只能使用 d 日以前
附件数据”的时间边界。输出的 CSV、JSON 和图片均写入 result/2/tmp。
"""

from __future__ import annotations

import json
import math
from datetime import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ATTACHMENT_DIR = ROOT / "CUMCM2026Problems" / "C题" / "附件"
ATTACHMENT_1 = ATTACHMENT_DIR / "附件1.xlsx"
ATTACHMENT_2 = ATTACHMENT_DIR / "附件2.xlsx"
RESULT_TEMPLATE = ATTACHMENT_DIR / "附件5" / "result2.xlsx"
OUTPUT_DIR = ROOT / "result" / "2" / "tmp"
FIGURE_DIR = OUTPUT_DIR / "figures"

DT_HOURS = 1.0 / 6.0
SLOTS_PER_DAY = 144
OUTPUT_START = pd.Timestamp("2025-02-01")
BATTERY_CAPACITY_KWH = 12_000.0
BATTERY_SOC_MIN_KWH = 1_200.0
BATTERY_SOC_MAX_KWH = 10_800.0
BATTERY_POWER_MAX_KW = 5_000.0
BATTERY_EFFICIENCY = 0.90
KEY_DATES = (
    pd.Timestamp("2025-03-20"),
    pd.Timestamp("2025-06-21"),
    pd.Timestamp("2025-09-23"),
    pd.Timestamp("2025-12-21"),
)
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if CHINESE_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def slot_to_minutes(value: Any) -> int:
    if isinstance(value, time):
        minute = value.hour * 60 + value.minute
        return 1440 if minute == 0 else minute
    if isinstance(value, pd.Timestamp):
        minute = value.hour * 60 + value.minute
        return 1440 if minute == 0 else minute
    text = str(value).strip()
    next_day = "+1" in text
    text = text.replace("+1", "")
    hour_text, minute_text = text.split(":")[:2]
    minute = int(hour_text) * 60 + int(minute_text)
    return minute + (1440 if next_day else 0)


def minute_label(minute: int) -> str:
    if minute == 1440:
        return "24:00"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def load_inputs() -> dict[str, Any]:
    price_raw = pd.read_excel(ATTACHMENT_1, sheet_name=0)
    load_raw = pd.read_excel(ATTACHMENT_2, sheet_name="小区负载")
    pv_raw = pd.read_excel(ATTACHMENT_2, sheet_name="光伏发电实际功率")
    template_headers = list(
        pd.read_excel(RESULT_TEMPLATE, sheet_name="计划购电量", nrows=0).columns
    )

    dates = pd.DatetimeIndex(pd.to_datetime(load_raw.iloc[:, 0])).normalize()
    pv_dates = pd.DatetimeIndex(pd.to_datetime(pv_raw.iloc[:, 0])).normalize()
    load_slots = np.array([slot_to_minutes(value) for value in load_raw.columns[1:]], dtype=int)
    pv_slots = np.array([slot_to_minutes(value) for value in pv_raw.columns[1:]], dtype=int)
    price_slots = np.array([slot_to_minutes(value) for value in price_raw.iloc[:, 0]], dtype=int)
    load_kw = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    pv_kw = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    price = pd.to_numeric(price_raw.iloc[:, 1], errors="raise").to_numpy(float)

    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    expected_slots = np.arange(10, 1441, 10)
    if not dates.equals(pv_dates) or not dates.equals(expected_dates):
        raise ValueError("附件 2 日期不连续，或负载与光伏日期不一致。")
    if not (
        np.array_equal(load_slots, pv_slots)
        and np.array_equal(load_slots, price_slots)
        and np.array_equal(load_slots, expected_slots)
    ):
        raise ValueError("附件 1、附件 2 的 10 分钟时点未完全对齐。")
    if load_kw.shape != (365, SLOTS_PER_DAY) or pv_kw.shape != load_kw.shape:
        raise ValueError(f"附件 2 维度异常：{load_kw.shape}，{pv_kw.shape}")
    if price.shape != (SLOTS_PER_DAY,):
        raise ValueError(f"附件 1 电价维度异常：{price.shape}")
    if len(template_headers) != SLOTS_PER_DAY + 3:
        raise ValueError("结果模板的计划购电量列数异常。")
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all() or not np.isfinite(price).all():
        raise ValueError("附件中存在缺失值或非有限值。")
    if (load_kw < 0).any() or (pv_kw < 0).any() or (price <= 0).any():
        raise ValueError("附件中的负载、光伏或电价超出题设范围。")
    return {
        "dates": dates,
        "slot_minutes": load_slots,
        "slot_labels": [minute_label(int(value)) for value in load_slots],
        "load_kw": load_kw,
        "pv_kw": pv_kw,
        "price": price,
        "template_headers": template_headers,
    }


def lag_profile_corr(values: np.ndarray, lag: int) -> float:
    return float(np.corrcoef(values[lag:].ravel(), values[:-lag].ravel())[0, 1])


def lag_series_corr(values: np.ndarray, lag: int) -> float:
    return float(np.corrcoef(values[lag:], values[:-lag])[0, 1])


def ramp_statistics(values: np.ndarray, dates: pd.DatetimeIndex, labels: list[str]) -> dict[str, Any]:
    ramps = np.diff(values, axis=1)
    absolute = np.abs(ramps)
    flat_index = int(np.argmax(absolute))
    day, transition = np.unravel_index(flat_index, absolute.shape)
    return {
        "绝对变化中位数（kW）": float(np.median(absolute)),
        "绝对变化95%分位（kW）": float(np.percentile(absolute, 95)),
        "绝对变化99%分位（kW）": float(np.percentile(absolute, 99)),
        "最大绝对变化（kW）": float(absolute[day, transition]),
        "最大变化日期": dates[day].strftime("%Y-%m-%d"),
        "最大变化时间段": f"{labels[transition]}-{labels[transition + 1]}",
    }


def extreme(values: np.ndarray, dates: pd.DatetimeIndex, labels: list[str], mode: str) -> dict[str, Any]:
    flat_index = int(np.argmax(values) if mode == "max" else np.argmin(values))
    day, slot = np.unravel_index(flat_index, values.shape)
    return {
        "数值（kW）": float(values[day, slot]),
        "日期": dates[day].strftime("%Y-%m-%d"),
        "时刻": labels[slot],
    }


def one_hour_window(price: np.ndarray, labels: list[str], mode: str) -> dict[str, Any]:
    sums = np.convolve(price, np.ones(6), mode="valid")
    start = int(np.argmin(sums) if mode == "min" else np.argmax(sums))
    return {
        "时间段": f"{labels[start]}-{labels[start + 5]}",
        "平均电价（元/kWh）": float(sums[start] / 6.0),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def create_figures(
    dates: pd.DatetimeIndex,
    slot_minutes: np.ndarray,
    price: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    net_kw: np.ndarray,
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"load": "#1f4e79", "pv": "#e69f00", "net": "#2a9d8f", "price": "#b23a48"}

    # 全年 52,560 个原始点逐点绘制；每周一标日期，每日保留短刻度。
    timestamps = np.concatenate(
        [(date + pd.to_timedelta(slot_minutes, unit="m")).to_numpy() for date in dates]
    )
    figure, axes = plt.subplots(2, 1, figsize=(24, 8), sharex=True, constrained_layout=True)
    for axis, values, label, color in (
        (axes[0], load_kw.ravel(), "小区负载原始值", colors["load"]),
        (axes[1], pv_kw.ravel(), "光伏实际功率原始值", colors["pv"]),
    ):
        axis.plot(timestamps, values, color=color, linewidth=0.42, label=label)
        axis.set_ylabel("功率（kW）")
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        axis.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        axis.tick_params(axis="x", which="major", labelrotation=90, labelsize=6, length=5)
        axis.tick_params(axis="x", which="minor", length=2)
        axis.grid(axis="x", which="major", linewidth=0.55, alpha=0.28)
        axis.grid(axis="x", which="minor", linewidth=0.3, alpha=0.12)
        axis.legend(loc="upper right", frameon=False)
    axes[-1].set_xlim(dates.min(), dates.max() + pd.Timedelta(days=1))
    axes[-1].set_xlabel("日期（每周一标注月-日，每日一条短刻度）")
    figure.suptitle("附件2全年原始10分钟功率折线（2025年，无采样截断）")
    figure.savefig(FIGURE_DIR / "00_attachment2_raw_lines.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    axis.plot(dates, daily["load_mwh"], label="负载", color=colors["load"], linewidth=1.2)
    axis.plot(dates, daily["pv_mwh"], label="光伏", color=colors["pv"], linewidth=1.2)
    axis.plot(
        dates,
        daily["positive_net_mwh"],
        label="储能介入前的外网需求",
        color=colors["net"],
        linewidth=1.2,
    )
    for key_date in KEY_DATES:
        axis.axvline(key_date, color="#777777", linewidth=0.8, alpha=0.45)
    axis.set_title("2025年逐日能量及季节变化")
    axis.set_ylabel("电量（MWh/日）")
    axis.set_xlabel("日期（主刻度为月，短刻度为周）")
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    axis.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    axis.legend(ncol=3, frameon=False)
    figure.savefig(FIGURE_DIR / "01_daily_energy.png", dpi=180)
    plt.close(figure)

    hours = slot_minutes / 60.0
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for values, label, color in (
        (load_kw, "负载", colors["load"]),
        (pv_kw, "光伏", colors["pv"]),
        (net_kw, "净负荷", colors["net"]),
    ):
        mean = np.mean(values, axis=0)
        p10, p90 = np.percentile(values, [10, 90], axis=0)
        axes[0].plot(hours, mean, label=f"{label}均值", color=color, linewidth=1.8)
        axes[0].fill_between(hours, p10, p90, color=color, alpha=0.12)
    axes[0].axhline(0, color="#333333", linewidth=0.7)
    axes[0].set_title("日内功率曲线：均值及10%—90%分位区间")
    axes[0].set_ylabel("功率（kW）")
    axes[0].legend(ncol=3, frameon=False)
    axes[1].plot(hours, price, color=colors["price"], linewidth=1.5)
    axes[1].set_ylabel("电价\n（元/kWh）")
    axes[1].set_xlabel("时刻")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(np.arange(0, 25, 2))
    figure.savefig(FIGURE_DIR / "02_intraday_profiles_and_price.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    x = np.arange(len(monthly))
    width = 0.25
    axes[0].bar(x - width, monthly["load_mwh_per_day"], width, label="负载", color=colors["load"])
    axes[0].bar(x, monthly["pv_mwh_per_day"], width, label="光伏", color=colors["pv"])
    axes[0].bar(
        x + width,
        monthly["positive_net_mwh_per_day"],
        width,
        label="储能介入前的外网需求",
        color=colors["net"],
    )
    axes[0].set_xticks(x, [f"{int(value.split('-')[1])}月" for value in monthly["month"]])
    axes[0].set_ylabel("平均日电量（MWh/日）")
    axes[0].set_title("月度能量结构")
    axes[0].legend(ncol=3, frameon=False)
    weekday_plot = (
        daily.assign(weekday=daily.index.day_name())
        .groupby("weekday")[["load_mwh", "pv_mwh", "positive_net_mwh"]]
        .mean()
        .reindex(WEEKDAY_ORDER)
    )
    x = np.arange(len(weekday_plot))
    axes[1].bar(x - width, weekday_plot["load_mwh"], width, label="负载", color=colors["load"])
    axes[1].bar(x, weekday_plot["pv_mwh"], width, label="光伏", color=colors["pv"])
    axes[1].bar(
        x + width,
        weekday_plot["positive_net_mwh"],
        width,
        label="储能介入前的外网需求",
        color=colors["net"],
    )
    axes[1].set_xticks(x, WEEKDAY_CN)
    axes[1].set_ylabel("平均日电量（MWh/日）")
    axes[1].set_title("星期周期")
    axes[1].legend(ncol=1, frameon=False)
    figure.savefig(FIGURE_DIR / "03_monthly_energy.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, key_date in zip(axes.ravel(), KEY_DATES):
        index = int(np.flatnonzero(dates == key_date)[0])
        axis.plot(hours, load_kw[index], label="负载", color=colors["load"], linewidth=1.3)
        axis.plot(hours, pv_kw[index], label="光伏", color=colors["pv"], linewidth=1.3)
        axis.plot(hours, net_kw[index], label="净负荷", color=colors["net"], linewidth=1.3)
        axis.axhline(0, color="#333333", linewidth=0.6)
        axis.set_title(f"{key_date.year}年{key_date.month}月{key_date.day}日")
        axis.set_xlim(0, 24)
        axis.set_xticks(np.arange(0, 25, 4))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.supxlabel("时刻")
    figure.supylabel("功率（kW）")
    figure.savefig(FIGURE_DIR / "05_key_date_profiles.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    dates = inputs["dates"]
    slot_minutes = inputs["slot_minutes"]
    slot_labels = inputs["slot_labels"]
    load_kw = inputs["load_kw"]
    pv_kw = inputs["pv_kw"]
    price = inputs["price"]
    net_kw = load_kw - pv_kw
    positive_net_kw = np.maximum(net_kw, 0.0)
    surplus_kw = np.maximum(-net_kw, 0.0)

    daily = pd.DataFrame(index=dates)
    daily.index.name = "date"
    daily["load_mwh"] = load_kw.sum(axis=1) * DT_HOURS / 1000.0
    daily["pv_mwh"] = pv_kw.sum(axis=1) * DT_HOURS / 1000.0
    daily["positive_net_mwh"] = positive_net_kw.sum(axis=1) * DT_HOURS / 1000.0
    daily["surplus_pv_mwh"] = surplus_kw.sum(axis=1) * DT_HOURS / 1000.0
    daily["pv_used_directly_mwh"] = np.minimum(load_kw, pv_kw).sum(axis=1) * DT_HOURS / 1000.0
    daily["direct_coverage_pct"] = daily["pv_used_directly_mwh"] / daily["load_mwh"] * 100.0
    daily["pv_self_consumption_pct"] = daily["pv_used_directly_mwh"] / daily["pv_mwh"] * 100.0

    daily_export = daily.reset_index().rename(
        columns={
            "date": "日期",
            "load_mwh": "负荷电量（MWh）",
            "pv_mwh": "光伏电量（MWh）",
            "positive_net_mwh": "正净负荷电量（MWh）",
            "surplus_pv_mwh": "光伏富余电量（MWh）",
            "pv_used_directly_mwh": "光伏直接利用电量（MWh）",
            "direct_coverage_pct": "光伏直接覆盖率（%）",
            "pv_self_consumption_pct": "光伏自消纳率（%）",
        }
    )
    daily_export["日期"] = pd.to_datetime(daily_export["日期"]).dt.strftime("%Y-%m-%d")
    daily_export.to_csv(OUTPUT_DIR / "daily_energy_summary.csv", index=False, encoding="utf-8-sig")

    monthly = (
        daily.assign(month=dates.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            days=("load_mwh", "size"),
            load_mwh_per_day=("load_mwh", "mean"),
            pv_mwh_per_day=("pv_mwh", "mean"),
            positive_net_mwh_per_day=("positive_net_mwh", "mean"),
            surplus_pv_mwh_per_day=("surplus_pv_mwh", "mean"),
            direct_coverage_pct=("direct_coverage_pct", "mean"),
            pv_self_consumption_pct=("pv_self_consumption_pct", "mean"),
        )
    )
    monthly.rename(
        columns={
            "month": "月份",
            "days": "天数",
            "load_mwh_per_day": "平均日负荷（MWh）",
            "pv_mwh_per_day": "平均日光伏（MWh）",
            "positive_net_mwh_per_day": "平均日正净负荷（MWh）",
            "surplus_pv_mwh_per_day": "平均日光伏富余（MWh）",
            "direct_coverage_pct": "光伏直接覆盖率（%）",
            "pv_self_consumption_pct": "光伏自消纳率（%）",
        }
    ).to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False, encoding="utf-8-sig")

    weekday = (
        daily.assign(weekday=dates.day_name())
        .groupby("weekday", as_index=False)
        .agg(
            days=("load_mwh", "size"),
            load_mwh_per_day=("load_mwh", "mean"),
            pv_mwh_per_day=("pv_mwh", "mean"),
            positive_net_mwh_per_day=("positive_net_mwh", "mean"),
        )
        .set_index("weekday")
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )
    weekday["weekday"] = WEEKDAY_CN
    weekday.rename(
        columns={
            "weekday": "星期",
            "days": "天数",
            "load_mwh_per_day": "平均日负荷（MWh）",
            "pv_mwh_per_day": "平均日光伏（MWh）",
            "positive_net_mwh_per_day": "平均日正净负荷（MWh）",
        }
    ).to_csv(OUTPUT_DIR / "weekday_summary.csv", index=False, encoding="utf-8-sig")

    key_rows: list[dict[str, Any]] = []
    for key_date in KEY_DATES:
        index = int(np.flatnonzero(dates == key_date)[0])
        peak = int(np.argmax(net_kw[index]))
        trough = int(np.argmin(net_kw[index]))
        key_rows.append(
            {
                "日期": key_date.strftime("%Y-%m-%d"),
                "星期": WEEKDAY_CN[key_date.weekday()],
                "负荷电量（MWh）": float(daily.iloc[index]["load_mwh"]),
                "光伏电量（MWh）": float(daily.iloc[index]["pv_mwh"]),
                "正净负荷电量（MWh）": float(daily.iloc[index]["positive_net_mwh"]),
                "光伏富余电量（MWh）": float(daily.iloc[index]["surplus_pv_mwh"]),
                "净负荷峰值（kW）": float(net_kw[index, peak]),
                "净负荷峰值时刻": slot_labels[peak],
                "净负荷最小值（kW）": float(net_kw[index, trough]),
                "净负荷最小时刻": slot_labels[trough],
                "光伏富余时段数": int((net_kw[index] < 0).sum()),
            }
        )
    pd.DataFrame(key_rows).to_csv(
        OUTPUT_DIR / "key_date_summary.csv", index=False, encoding="utf-8-sig"
    )

    output_mask = dates >= OUTPUT_START
    low_days = dates.day_name().isin(["Friday", "Saturday"])
    direct_pv = float(daily["pv_used_directly_mwh"].sum())
    mean_load = load_kw.mean(axis=0)
    mean_pv = pv_kw.mean(axis=0)
    mean_net = net_kw.mean(axis=0)
    daylight_first: list[int] = []
    daylight_last: list[int] = []
    for row in pv_kw:
        daylight = np.flatnonzero(row > 10.0)
        if daylight.size:
            daylight_first.append(int(slot_minutes[daylight[0]]))
            daylight_last.append(int(slot_minutes[daylight[-1]]))
    tiny_pv = (pv_kw > 0) & (pv_kw < 1)
    metrics = {
        "数据口径": (
            "附件2是全年实际回测数据；制定第d日计划时只允许使用d日前历史，"
            "第d日实际值仅在计划锁定后用于运行仿真和费用评价"
        ),
        "数据质量": {
            "日期范围": [dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")],
            "负荷维度": list(load_kw.shape),
            "光伏维度": list(pv_kw.shape),
            "电价点数": int(price.size),
            "缺失或非有限值个数": 0,
            "负值个数": 0,
            "重复日期个数": int(dates.duplicated().sum()),
            "负荷完全重复日曲线个数": int(len(load_kw) - np.unique(load_kw, axis=0).shape[0]),
            "光伏完全重复日曲线个数": int(len(pv_kw) - np.unique(pv_kw, axis=0).shape[0]),
            "光伏精确为0的比例（%）": float(np.mean(pv_kw == 0) * 100.0),
            "0至1kW微小光伏点数": int(tiny_pv.sum()),
            "0至1kW微小光伏年电量（kWh）": float(pv_kw[tiny_pv].sum() * DT_HOURS),
        },
        "功率统计（kW）": {
            "负荷": {"均值": float(load_kw.mean()), "标准差": float(load_kw.std()), "最小值": float(load_kw.min()), "中位数": float(np.median(load_kw)), "95%分位": float(np.percentile(load_kw, 95)), "最大值": float(load_kw.max())},
            "光伏": {"均值": float(pv_kw.mean()), "标准差": float(pv_kw.std()), "最小值": float(pv_kw.min()), "中位数": float(np.median(pv_kw)), "95%分位": float(np.percentile(pv_kw, 95)), "最大值": float(pv_kw.max())},
            "净负荷": {"均值": float(net_kw.mean()), "标准差": float(net_kw.std()), "最小值": float(net_kw.min()), "中位数": float(np.median(net_kw)), "95%分位": float(np.percentile(net_kw, 95)), "最大值": float(net_kw.max())},
        },
        "全局极值": {
            "负荷最大值": extreme(load_kw, dates, slot_labels, "max"),
            "光伏最大值": extreme(pv_kw, dates, slot_labels, "max"),
            "净负荷最大值": extreme(net_kw, dates, slot_labels, "max"),
            "净负荷最小值": extreme(net_kw, dates, slot_labels, "min"),
        },
        "全年能量（MWh）": {
            "负荷": float(daily["load_mwh"].sum()),
            "光伏": float(daily["pv_mwh"].sum()),
            "正净负荷": float(daily["positive_net_mwh"].sum()),
            "光伏富余": float(daily["surplus_pv_mwh"].sum()),
            "光伏直接覆盖率（%）": direct_pv / float(daily["load_mwh"].sum()) * 100.0,
            "光伏自消纳率（%）": direct_pv / float(daily["pv_mwh"].sum()) * 100.0,
            "光伏超过负荷的时段比例（%）": float(np.mean(pv_kw > load_kw) * 100.0),
            "出现光伏富余的天数": int((pv_kw > load_kw).any(axis=1).sum()),
        },
        "输出期能量（MWh）": {
            "天数": int(output_mask.sum()),
            "负荷": float(daily.loc[output_mask, "load_mwh"].sum()),
            "光伏": float(daily.loc[output_mask, "pv_mwh"].sum()),
            "正净负荷": float(daily.loc[output_mask, "positive_net_mwh"].sum()),
            "光伏富余": float(daily.loc[output_mask, "surplus_pv_mwh"].sum()),
        },
        "日内规律": {
            "平均负荷峰值": {"数值（kW）": float(mean_load.max()), "时刻": slot_labels[int(np.argmax(mean_load))]},
            "平均负荷谷值": {"数值（kW）": float(mean_load.min()), "时刻": slot_labels[int(np.argmin(mean_load))]},
            "平均光伏峰值": {"数值（kW）": float(mean_pv.max()), "时刻": slot_labels[int(np.argmax(mean_pv))]},
            "平均净负荷峰值": {"数值（kW）": float(mean_net.max()), "时刻": slot_labels[int(np.argmax(mean_net))]},
            "平均净负荷谷值": {"数值（kW）": float(mean_net.min()), "时刻": slot_labels[int(np.argmin(mean_net))]},
            "平均净负荷与电价相关系数": float(np.corrcoef(mean_net, price)[0, 1]),
            "光伏大于10kW的中位起始时刻": minute_label(int(np.median(daylight_first))),
            "光伏大于10kW的中位结束时刻": minute_label(int(np.median(daylight_last))),
        },
        "星期与相邻日期规律": {
            "周五周六平均日负荷（MWh）": float(daily.loc[low_days, "load_mwh"].mean()),
            "其余日期平均日负荷（MWh）": float(daily.loc[~low_days, "load_mwh"].mean()),
            "负荷日曲线相隔1日相关系数": lag_profile_corr(load_kw, 1),
            "负荷日曲线相隔7日相关系数": lag_profile_corr(load_kw, 7),
            "光伏日曲线相隔1日相关系数": lag_profile_corr(pv_kw, 1),
            "光伏日曲线相隔7日相关系数": lag_profile_corr(pv_kw, 7),
            "净负荷日曲线相隔1日相关系数": lag_profile_corr(net_kw, 1),
            "净负荷日曲线相隔7日相关系数": lag_profile_corr(net_kw, 7),
            "负荷日电量相隔1日相关系数": lag_series_corr(daily["load_mwh"].to_numpy(), 1),
            "负荷日电量相隔7日相关系数": lag_series_corr(daily["load_mwh"].to_numpy(), 7),
            "光伏日电量相隔1日相关系数": lag_series_corr(daily["pv_mwh"].to_numpy(), 1),
            "光伏日电量相隔7日相关系数": lag_series_corr(daily["pv_mwh"].to_numpy(), 7),
            "正净负荷日电量相隔1日相关系数": lag_series_corr(daily["positive_net_mwh"].to_numpy(), 1),
            "正净负荷日电量相隔7日相关系数": lag_series_corr(daily["positive_net_mwh"].to_numpy(), 7),
            "净负荷10分钟变化": ramp_statistics(net_kw, dates, slot_labels),
        },
        "电价": {
            "平均值（元/kWh）": float(price.mean()),
            "中位数（元/kWh）": float(np.median(price)),
            "最低价（元/kWh）": float(price.min()),
            "最低价时刻": slot_labels[int(np.argmin(price))],
            "最高价（元/kWh）": float(price.max()),
            "最高价时刻": slot_labels[int(np.argmax(price))],
            "峰谷比": float(price.max() / price.min()),
            "最低连续1小时窗口": one_hour_window(price, slot_labels, "min"),
            "最高连续1小时窗口": one_hour_window(price, slot_labels, "max"),
            "双程效率套利阈值": float(1.0 / BATTERY_EFFICIENCY**2),
        },
        "储能尺度": {
            "额定容量（kWh）": BATTERY_CAPACITY_KWH,
            "可用SOC摆幅（kWh）": BATTERY_SOC_MAX_KWH - BATTERY_SOC_MIN_KWH,
            "每10分钟最大充放电量（kWh）": BATTERY_POWER_MAX_KW * DT_HOURS,
            "最大单日光伏富余（MWh）": float(daily["surplus_pv_mwh"].max()),
            "最大单日光伏富余日期": daily["surplus_pv_mwh"].idxmax().strftime("%Y-%m-%d"),
            "日富余量超过SOC摆幅的天数": int((daily["surplus_pv_mwh"] * 1000 > BATTERY_SOC_MAX_KWH - BATTERY_SOC_MIN_KWH).sum()),
            "光伏富余功率超过5MW的时段数": int((surplus_kw > BATTERY_POWER_MAX_KW).sum()),
            "正净负荷功率超过5MW的时段数": int((positive_net_kw > BATTERY_POWER_MAX_KW).sum()),
        },
        "指定日期": key_rows,
    }
    (OUTPUT_DIR / "analysis_metrics.json").write_text(
        json.dumps(json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_figures(dates, slot_minutes, price, load_kw, pv_kw, net_kw, daily, monthly)
    print("问题二原始数据规律与信息边界分析完成。")
    print(f"数据规模：负荷={load_kw.shape}，光伏={pv_kw.shape}，电价={price.shape}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
