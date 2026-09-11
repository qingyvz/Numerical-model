"""C 题问题二：因果滚动预测、两阶段场景优化与实际运行回测核心。

信息边界严格设为：制定第 d 天计划时，只允许使用 d 日 0:00 之前的附件 2
历史值；第 d 天的实际负载与光伏仅在计划锁定后用于运行仿真和费用结算。
所有功率先乘 1/6 小时转换为每 10 分钟电量。
"""

from __future__ import annotations

import json
import math
import os
import time as timer
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
ATTACHMENT_DIR = ROOT / "CUMCM2026Problems" / "C题" / "附件"
ATTACHMENT_1 = ATTACHMENT_DIR / "附件1.xlsx"
ATTACHMENT_2 = ATTACHMENT_DIR / "附件2.xlsx"
RESULT_TEMPLATE = ATTACHMENT_DIR / "附件5" / "result2.xlsx"
OUTPUT_DIR = ROOT / "result" / "2" / "tmp"
FIGURE_DIR = OUTPUT_DIR / "figures"
COMPARISON_DIR = OUTPUT_DIR / "causal_compare"

MODEL_VERSION = "q2-causal-two-stage-v1"
DT_HOURS = 1.0 / 6.0
SLOTS_PER_DAY = 144
OUTPUT_START = pd.Timestamp("2025-02-01")
EMERGENCY_PRICE_MULTIPLIER = 5.0
KEY_DATES = (
    pd.Timestamp("2025-03-20"),
    pd.Timestamp("2025-06-21"),
    pd.Timestamp("2025-09-23"),
    pd.Timestamp("2025-12-21"),
)
KEY_PURCHASE_INTERVALS = (
    "10:00-10:10",
    "12:00-12:10",
    "14:00-14:10",
    "16:00-16:10",
    "18:00-18:10",
    "20:00-20:10",
)
FOUR_HOUR_BLOCKS = (
    "0:00-4:00",
    "4:00-8:00",
    "8:00-12:00",
    "12:00-16:00",
    "16:00-20:00",
    "20:00-24:00",
)

BATTERY_CAPACITY_KWH = 12_000.0
BATTERY_SOC_MIN_KWH = 1_200.0
BATTERY_SOC_MAX_KWH = 10_800.0
BATTERY_INITIAL_SOC_KWH = 6_000.0
BATTERY_POWER_MAX_KW = 5_000.0
BATTERY_INTERVAL_MAX_KWH = BATTERY_POWER_MAX_KW * DT_HOURS
BATTERY_CHARGE_EFFICIENCY = 0.90
BATTERY_DISCHARGE_EFFICIENCY = 0.90
NUMERICAL_TOLERANCE = 1.0e-6
LEXICOGRAPHIC_TIE_BREAKER = 1.0e-8
FORECAST_SCALE_KW = 10_000.0

CHINESE_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if CHINESE_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(CHINESE_FONT_PATH))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class InputData:
    dates: pd.DatetimeIndex
    slot_minutes: np.ndarray
    source_slot_labels: list[str]
    result_interval_headers: list[str]
    load_kw: np.ndarray
    pv_kw: np.ndarray
    price_yuan_per_kwh: np.ndarray

    @property
    def net_kw(self) -> np.ndarray:
        return self.load_kw - self.pv_kw


@dataclass
class ForecastBundle:
    method_id: str
    name: str
    load_kw: np.ndarray
    pv_kw: np.ndarray
    fit_seconds: float
    note: str
    hyperparameters: dict[str, Any]


@dataclass
class DayModelResult:
    planned_purchase_kwh: np.ndarray
    emergency_kwh: np.ndarray
    grid_charge_kwh: np.ndarray
    pv_charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_end_kwh: np.ndarray
    grid_spill_kwh: np.ndarray
    pv_spill_kwh: np.ndarray
    monetary_objective_yuan: float
    solver_objective: float
    solve_seconds: float


@dataclass
class StrategyResult:
    method_id: str
    name: str
    planned_purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_start_kwh: np.ndarray
    soc_end_kwh: np.ndarray
    emergency_kwh: np.ndarray
    surplus_or_curtailed_kwh: np.ndarray
    solve_seconds: float
    algorithm_note: str
    forecast_id: str | None = None


def slot_to_minutes(value: Any) -> int:
    """把附件时刻转换为日内累计分钟，次日 0:00 记为 1440。"""
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


def load_inputs() -> InputData:
    """读取附件并验证日期、时点、单位及结果模板结构。"""
    price_raw = pd.read_excel(ATTACHMENT_1, sheet_name=0)
    load_raw = pd.read_excel(ATTACHMENT_2, sheet_name="小区负载")
    pv_raw = pd.read_excel(ATTACHMENT_2, sheet_name="光伏发电实际功率")
    template_headers = list(
        pd.read_excel(RESULT_TEMPLATE, sheet_name="计划购电量", nrows=0).columns
    )

    dates_load = pd.DatetimeIndex(pd.to_datetime(load_raw.iloc[:, 0])).normalize()
    dates_pv = pd.DatetimeIndex(pd.to_datetime(pv_raw.iloc[:, 0])).normalize()
    if not dates_load.equals(dates_pv) or dates_load.has_duplicates:
        raise ValueError("附件 2 的负载与光伏日期不一致或存在重复日期。")

    load_minutes = np.array([slot_to_minutes(v) for v in load_raw.columns[1:]], dtype=int)
    pv_minutes = np.array([slot_to_minutes(v) for v in pv_raw.columns[1:]], dtype=int)
    price_minutes = np.array([slot_to_minutes(v) for v in price_raw.iloc[:, 0]], dtype=int)
    if not (np.array_equal(load_minutes, pv_minutes) and np.array_equal(load_minutes, price_minutes)):
        raise ValueError("附件 1 与附件 2 的 10 分钟时点未对齐。")

    load_kw = load_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    pv_kw = pv_raw.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    price = pd.to_numeric(price_raw.iloc[:, 1], errors="raise").to_numpy(float)
    if load_kw.shape != (365, SLOTS_PER_DAY) or pv_kw.shape != load_kw.shape:
        raise ValueError(f"附件 2 规模异常：load={load_kw.shape}, pv={pv_kw.shape}")
    if price.shape != (SLOTS_PER_DAY,) or len(template_headers) != SLOTS_PER_DAY + 3:
        raise ValueError("电价或 result2.xlsx 模板的时段数量异常。")
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        raise ValueError("附件 2 含缺失值或非有限值。")
    if (load_kw < 0).any() or (pv_kw < 0).any() or (price <= 0).any():
        raise ValueError("负载、光伏或电价超出题目允许范围。")

    return InputData(
        dates=dates_load,
        slot_minutes=load_minutes,
        source_slot_labels=[minute_label(int(v)) for v in load_minutes],
        result_interval_headers=[str(v) for v in template_headers[1 : 1 + SLOTS_PER_DAY]],
        load_kw=load_kw,
        pv_kw=pv_kw,
        price_yuan_per_kwh=price,
    )


def output_indices(data: InputData) -> np.ndarray:
    return np.flatnonzero(np.asarray(data.dates >= OUTPUT_START))


def _series_features(values_kw: np.ndarray, dates: pd.DatetimeIndex, day: int) -> np.ndarray:
    """为某日 144 个时点构造只依赖历史的低维时序特征。"""
    if day < 14:
        raise ValueError("特征需要至少 14 天历史。")
    slots = np.arange(SLOTS_PER_DAY, dtype=float) / SLOTS_PER_DAY
    columns: list[np.ndarray] = [np.ones(SLOTS_PER_DAY)]
    for harmonic in (1, 2, 3):
        columns.append(np.sin(2 * np.pi * harmonic * slots))
        columns.append(np.cos(2 * np.pi * harmonic * slots))

    year_position = (dates[day].dayofyear - 1) / 365.0
    for harmonic in (1, 2):
        columns.append(np.full(SLOTS_PER_DAY, np.sin(2 * np.pi * harmonic * year_position)))
        columns.append(np.full(SLOTS_PER_DAY, np.cos(2 * np.pi * harmonic * year_position)))

    weekday = dates[day].weekday()
    for index in range(7):
        columns.append(np.full(SLOTS_PER_DAY, 1.0 if index == weekday else 0.0))

    lag1 = values_kw[day - 1] / FORECAST_SCALE_KW
    lag7 = values_kw[day - 7] / FORECAST_SCALE_KW
    lag14 = values_kw[day - 14] / FORECAST_SCALE_KW
    rolling7 = values_kw[day - 7 : day].mean(axis=0) / FORECAST_SCALE_KW
    same_week_mean = (values_kw[day - 7] + values_kw[day - 14]) / (2 * FORECAST_SCALE_KW)
    columns.extend((lag1, lag7, lag14, rolling7, same_week_mean, lag1 - lag7))
    return np.column_stack(columns)


def _feature_cube(values_kw: np.ndarray, dates: pd.DatetimeIndex) -> list[np.ndarray | None]:
    return [None if day < 14 else _series_features(values_kw, dates, day) for day in range(len(dates))]


def make_seasonal_forecast(data: InputData) -> ForecastBundle:
    load = np.full_like(data.load_kw, np.nan, dtype=float)
    pv = np.full_like(data.pv_kw, np.nan, dtype=float)
    load[7:] = data.load_kw[:-7]
    pv[7:] = data.pv_kw[:-7]
    return ForecastBundle(
        method_id="seasonal_7d",
        name="前7天同一时刻季节朴素预测",
        load_kw=load,
        pv_kw=pv,
        fit_seconds=0.0,
        note="第d天预测等于d-7天同一时刻实测值，仅使用历史数据",
        hyperparameters={"lag_days": 7},
    )


def _ridge_predict_expanding(
    values_kw: np.ndarray,
    features: list[np.ndarray | None],
    alpha: float,
) -> np.ndarray:
    predictions = np.full_like(values_kw, np.nan, dtype=float)
    first_x = features[14]
    if first_x is None:
        raise RuntimeError("特征构造失败。")
    feature_count = first_x.shape[1]
    xtx = np.zeros((feature_count, feature_count), dtype=float)
    xty = np.zeros(feature_count, dtype=float)
    regularizer = np.eye(feature_count) * alpha
    regularizer[0, 0] = 0.0

    for day in range(14, len(values_kw)):
        x_day = features[day]
        if x_day is None:
            continue
        if day == 14:
            prediction = values_kw[day - 7]
        else:
            try:
                beta = np.linalg.solve(xtx + regularizer, xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(xtx + regularizer, xty, rcond=None)[0]
            prediction = (x_day @ beta) * FORECAST_SCALE_KW
        predictions[day] = np.clip(prediction, 0.0, None)
        y_day = values_kw[day] / FORECAST_SCALE_KW
        xtx += x_day.T @ x_day
        xty += x_day.T @ y_day
    return predictions


def _select_ridge_alpha(
    data: InputData,
    load_features: list[np.ndarray | None],
    pv_features: list[np.ndarray | None],
) -> float:
    """只用 1 月内部的顺序验证选择固定正则系数。"""
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    validation_days = range(24, 31)
    scores: dict[float, float] = {}
    for alpha in candidates:
        total_absolute = 0.0
        total_count = 0
        for values, features in ((data.load_kw, load_features), (data.pv_kw, pv_features)):
            first_x = features[14]
            if first_x is None:
                raise RuntimeError("特征构造失败。")
            p = first_x.shape[1]
            xtx = np.zeros((p, p), dtype=float)
            xty = np.zeros(p, dtype=float)
            reg = np.eye(p) * alpha
            reg[0, 0] = 0.0
            for day in range(14, 31):
                x_day = features[day]
                if x_day is None:
                    continue
                if day in validation_days:
                    beta = np.linalg.solve(xtx + reg, xty)
                    predicted = np.clip((x_day @ beta) * FORECAST_SCALE_KW, 0.0, None)
                    total_absolute += float(np.abs(predicted - values[day]).sum())
                    total_count += predicted.size
                y_day = values[day] / FORECAST_SCALE_KW
                xtx += x_day.T @ x_day
                xty += x_day.T @ y_day
        scores[alpha] = total_absolute / max(total_count, 1)
    return min(scores, key=scores.get)


def make_ridge_forecast(data: InputData) -> ForecastBundle:
    started = timer.perf_counter()
    load_features = _feature_cube(data.load_kw, data.dates)
    pv_features = _feature_cube(data.pv_kw, data.dates)
    alpha = _select_ridge_alpha(data, load_features, pv_features)
    load = _ridge_predict_expanding(data.load_kw, load_features, alpha)
    pv = _ridge_predict_expanding(data.pv_kw, pv_features, alpha)
    return ForecastBundle(
        method_id="ridge_causal",
        name="扩展窗口Ridge正则化时序预测",
        load_kw=load,
        pv_kw=pv,
        fit_seconds=timer.perf_counter() - started,
        note="固定正则系数仅由1月顺序验证选取；此后每日只以此前实测值更新充分统计量",
        hyperparameters={"alpha": alpha, "initial_validation": "2025-01-25至2025-01-31"},
    )


def make_tree_forecast(data: InputData, ridge: ForecastBundle) -> ForecastBundle:
    """每 28 天因果重训一次直方图梯度提升树，兼顾精度与计算开销。"""
    started = timer.perf_counter()
    load_features = _feature_cube(data.load_kw, data.dates)
    pv_features = _feature_cube(data.pv_kw, data.dates)
    predicted_load = ridge.load_kw.copy()
    predicted_pv = ridge.pv_kw.copy()
    start_day = int(output_indices(data)[0])
    models: tuple[HistGradientBoostingRegressor | None, HistGradientBoostingRegressor | None] = (None, None)

    for day in range(start_day, len(data.dates)):
        if (day - start_day) % 28 == 0 or models[0] is None:
            fitted: list[HistGradientBoostingRegressor] = []
            train_start = max(14, day - 180)
            for values, features in ((data.load_kw, load_features), (data.pv_kw, pv_features)):
                x_train = np.vstack([features[index] for index in range(train_start, day) if features[index] is not None])
                y_train = np.concatenate([values[index] for index in range(train_start, day)])
                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.07,
                    max_iter=70,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=4.0,
                    early_stopping=False,
                    random_state=2026,
                )
                model.fit(x_train, y_train)
                fitted.append(model)
            models = (fitted[0], fitted[1])
        load_x = load_features[day]
        pv_x = pv_features[day]
        if load_x is None or pv_x is None or models[0] is None or models[1] is None:
            raise RuntimeError("树模型的滚动特征或模型缺失。")
        predicted_load[day] = np.clip(models[0].predict(load_x), 0.0, None)
        predicted_pv[day] = np.clip(models[1].predict(pv_x), 0.0, None)

    return ForecastBundle(
        method_id="hist_gbdt_causal",
        name="滚动直方图梯度提升树预测",
        load_kw=predicted_load,
        pv_kw=predicted_pv,
        fit_seconds=timer.perf_counter() - started,
        note="每28天仅用此前最多180天样本重训；1月残差热启动沿用因果Ridge预测",
        hyperparameters={
            "refit_days": 28,
            "training_window_days": 180,
            "max_iter": 70,
            "max_leaf_nodes": 31,
            "l2_regularization": 4.0,
        },
    )


def forecast_metrics(data: InputData, forecast: ForecastBundle) -> dict[str, Any]:
    indices = output_indices(data)
    actual_load = data.load_kw[indices]
    actual_pv = data.pv_kw[indices]
    predicted_load = forecast.load_kw[indices]
    predicted_pv = forecast.pv_kw[indices]
    if not np.isfinite(predicted_load).all() or not np.isfinite(predicted_pv).all():
        raise ValueError(f"{forecast.name} 在输出期存在非有限预测。")

    def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        error = predicted - actual
        return {
            "mae_kw": float(np.mean(np.abs(error))),
            "rmse_kw": float(np.sqrt(np.mean(error**2))),
            "bias_kw": float(np.mean(error)),
        }

    result = {
        "model_version": MODEL_VERSION,
        "forecast_id": forecast.method_id,
        "forecast_name": forecast.name,
        "fit_seconds": float(forecast.fit_seconds),
        "note": forecast.note,
        "hyperparameters": forecast.hyperparameters,
        "load": metrics(actual_load, predicted_load),
        "pv": metrics(actual_pv, predicted_pv),
        "net_load": metrics(actual_load - actual_pv, predicted_load - predicted_pv),
    }
    return result


def solve_day_model(
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    price_yuan_per_kwh: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    initial_soc_kwh: float = BATTERY_INITIAL_SOC_KWH,
    terminal_soc_kwh: float | None = BATTERY_INITIAL_SOC_KWH,
    risk_weight: float = 0.0,
    cvar_alpha: float = 0.90,
    fixed_plan_kwh: np.ndarray | None = None,
) -> DayModelResult:
    """求解单日两阶段 LP；固定计划时退化为真实场景的补救调度。

    计划电与光伏分别拆分为供负载、充电和弃置三条流，紧急购电仅进入负载，
    因而不会出现用紧急电给电池充电的含义错误。
    """
    load = np.asarray(load_scenarios_kw, dtype=float)
    pv = np.asarray(pv_scenarios_kw, dtype=float)
    if load.ndim == 1:
        load = load[None, :]
    if pv.ndim == 1:
        pv = pv[None, :]
    if load.shape != pv.shape or load.shape[1] != SLOTS_PER_DAY:
        raise ValueError(f"场景规模异常：load={load.shape}, pv={pv.shape}")
    if (load < 0).any() or (pv < 0).any() or not np.isfinite(load).all() or not np.isfinite(pv).all():
        raise ValueError("场景负载/光伏必须为非负有限值。")
    scenario_count, periods = load.shape
    price = np.asarray(price_yuan_per_kwh, dtype=float).ravel()
    if price.shape != (periods,) or (price <= 0).any():
        raise ValueError("日内电价维度或取值异常。")
    if probabilities is None:
        probs = np.full(scenario_count, 1.0 / scenario_count)
    else:
        probs = np.asarray(probabilities, dtype=float).ravel()
        if probs.shape != (scenario_count,) or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError("场景概率异常。")
        probs = probs / probs.sum()
    if risk_weight < 0 or not 0 < cvar_alpha < 1:
        raise ValueError("CVaR 参数异常。")

    block = scenario_count * periods
    q0 = 0
    e0 = q0 + periods
    cg0 = e0 + block
    cp0 = cg0 + block
    d0 = cp0 + block
    soc0 = d0 + block
    sg0 = soc0 + block
    sp0 = sg0 + block
    variable_count = sp0 + block
    var0: int | None = None
    excess0: int | None = None
    if risk_weight > 0:
        var0 = variable_count
        excess0 = var0 + 1
        variable_count = excess0 + scenario_count

    objective = np.zeros(variable_count, dtype=float)
    objective[q0:e0] = price
    for scenario in range(scenario_count):
        lo = scenario * periods
        hi = lo + periods
        objective[e0 + lo : e0 + hi] = probs[scenario] * EMERGENCY_PRICE_MULTIPLIER * price
        tie = probs[scenario] * LEXICOGRAPHIC_TIE_BREAKER
        for start in (cg0, cp0, d0, sg0, sp0):
            objective[start + lo : start + hi] = tie
    if risk_weight > 0:
        if var0 is None or excess0 is None:
            raise RuntimeError("CVaR 变量索引未建立。")
        objective[var0] = risk_weight
        objective[excess0 : excess0 + scenario_count] = (
            risk_weight * probs / (1.0 - cvar_alpha)
        )

    # 等式：每场景逐时能量流平衡，以及 SOC 状态转移。
    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_values: list[float] = []
    b_eq = np.zeros(2 * block, dtype=float)
    for scenario in range(scenario_count):
        scenario_offset = scenario * periods
        for period in range(periods):
            flat = scenario_offset + period
            energy_row = flat
            # q + e - c_grid - c_pv + d - spill_grid - spill_pv = load - pv
            eq_rows.extend([energy_row] * 7)
            eq_cols.extend(
                [
                    q0 + period,
                    e0 + flat,
                    cg0 + flat,
                    cp0 + flat,
                    d0 + flat,
                    sg0 + flat,
                    sp0 + flat,
                ]
            )
            eq_values.extend((1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0))
            b_eq[energy_row] = (load[scenario, period] - pv[scenario, period]) * DT_HOURS

            state_row = block + flat
            eq_rows.extend([state_row] * (4 if period > 0 else 3))
            eq_cols.extend(
                [soc0 + flat, cg0 + flat, cp0 + flat, d0 + flat]
                if period > 0
                else [soc0 + flat, cg0 + flat, cp0 + flat]
            )
            eq_values.extend(
                (1.0, -BATTERY_CHARGE_EFFICIENCY, -BATTERY_CHARGE_EFFICIENCY, 1.0 / BATTERY_DISCHARGE_EFFICIENCY)
                if period > 0
                else (1.0, -BATTERY_CHARGE_EFFICIENCY, -BATTERY_CHARGE_EFFICIENCY)
            )
            if period == 0:
                # 第一个时段的放电项也必须进入状态方程。
                eq_rows.append(state_row)
                eq_cols.append(d0 + flat)
                eq_values.append(1.0 / BATTERY_DISCHARGE_EFFICIENCY)
                b_eq[state_row] = initial_soc_kwh
            else:
                eq_rows.append(state_row)
                eq_cols.append(soc0 + flat - 1)
                eq_values.append(-1.0)

    a_eq = coo_matrix(
        (eq_values, (eq_rows, eq_cols)), shape=(2 * block, variable_count)
    ).tocsr()

    # 不等式：计划电分流、光伏分流、充放电功率，以及可选 CVaR 线性化。
    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_values: list[float] = []
    ub_rhs: list[float] = []
    row_index = 0
    for scenario in range(scenario_count):
        scenario_offset = scenario * periods
        for period in range(periods):
            flat = scenario_offset + period
            # c_grid + spill_grid <= q，隐含计划电供负载量非负。
            ub_rows.extend([row_index] * 3)
            ub_cols.extend([cg0 + flat, sg0 + flat, q0 + period])
            ub_values.extend((1.0, 1.0, -1.0))
            ub_rhs.append(0.0)
            row_index += 1
            # c_pv + spill_pv <= pv，隐含光伏供负载量非负。
            ub_rows.extend([row_index] * 2)
            ub_cols.extend([cp0 + flat, sp0 + flat])
            ub_values.extend((1.0, 1.0))
            ub_rhs.append(pv[scenario, period] * DT_HOURS)
            row_index += 1
            # 充电与放电合计不超过时段功率上限；实际解中再检查是否同时发生。
            ub_rows.extend([row_index] * 3)
            ub_cols.extend([cg0 + flat, cp0 + flat, d0 + flat])
            ub_values.extend((1.0, 1.0, 1.0))
            ub_rhs.append(BATTERY_INTERVAL_MAX_KWH)
            row_index += 1

    if risk_weight > 0:
        if var0 is None or excess0 is None:
            raise RuntimeError("CVaR 变量索引未建立。")
        for scenario in range(scenario_count):
            lo = scenario * periods
            ub_rows.extend([row_index] * (periods + 2))
            ub_cols.extend(
                list(range(e0 + lo, e0 + lo + periods))
                + [var0, excess0 + scenario]
            )
            ub_values.extend(list(EMERGENCY_PRICE_MULTIPLIER * price) + [-1.0, -1.0])
            ub_rhs.append(0.0)
            row_index += 1

    a_ub = coo_matrix(
        (ub_values, (ub_rows, ub_cols)), shape=(row_index, variable_count)
    ).tocsr()
    b_ub = np.asarray(ub_rhs, dtype=float)

    bounds: list[tuple[float | None, float | None]] = []
    if fixed_plan_kwh is None:
        bounds.extend([(0.0, None)] * periods)
    else:
        fixed = np.asarray(fixed_plan_kwh, dtype=float).ravel()
        if fixed.shape != (periods,) or (fixed < -NUMERICAL_TOLERANCE).any():
            raise ValueError("固定计划购电量维度或取值异常。")
        bounds.extend([(max(float(value), 0.0), max(float(value), 0.0)) for value in fixed])
    bounds.extend([(0.0, None)] * block)  # emergency
    bounds.extend([(0.0, BATTERY_INTERVAL_MAX_KWH)] * block)  # grid charge
    bounds.extend([(0.0, BATTERY_INTERVAL_MAX_KWH)] * block)  # pv charge
    bounds.extend([(0.0, BATTERY_INTERVAL_MAX_KWH)] * block)  # discharge
    for scenario in range(scenario_count):
        for period in range(periods):
            if terminal_soc_kwh is not None and period == periods - 1:
                bounds.append((terminal_soc_kwh, terminal_soc_kwh))
            else:
                bounds.append((BATTERY_SOC_MIN_KWH, BATTERY_SOC_MAX_KWH))
    bounds.extend([(0.0, None)] * block)  # grid spill
    bounds.extend([(0.0, None)] * block)  # pv spill
    if risk_weight > 0:
        bounds.append((0.0, None))
        bounds.extend([(0.0, None)] * scenario_count)

    started = timer.perf_counter()
    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    elapsed = timer.perf_counter() - started
    if not result.success:
        raise RuntimeError(f"单日两阶段线性规划失败：{result.message}")

    x = result.x
    planned = x[q0:e0]
    emergency = x[e0:cg0].reshape(scenario_count, periods)
    grid_charge = x[cg0:cp0].reshape(scenario_count, periods)
    pv_charge = x[cp0:d0].reshape(scenario_count, periods)
    discharge = x[d0:soc0].reshape(scenario_count, periods)
    soc = x[soc0:sg0].reshape(scenario_count, periods)
    grid_spill = x[sg0:sp0].reshape(scenario_count, periods)
    pv_spill = x[sp0 : sp0 + block].reshape(scenario_count, periods)
    monetary = float(np.dot(price, planned))
    monetary += float(
        np.sum(probs[:, None] * EMERGENCY_PRICE_MULTIPLIER * price[None, :] * emergency)
    )
    return DayModelResult(
        planned_purchase_kwh=planned,
        emergency_kwh=emergency,
        grid_charge_kwh=grid_charge,
        pv_charge_kwh=pv_charge,
        discharge_kwh=discharge,
        soc_end_kwh=soc,
        grid_spill_kwh=grid_spill,
        pv_spill_kwh=pv_spill,
        monetary_objective_yuan=monetary,
        solver_objective=float(result.fun),
        solve_seconds=float(elapsed),
    )


def _empty_strategy_arrays(data: InputData) -> dict[str, np.ndarray]:
    shape = data.load_kw.shape
    return {
        "planned": np.zeros(shape, dtype=float),
        "charge": np.zeros(shape, dtype=float),
        "discharge": np.zeros(shape, dtype=float),
        "soc_start": np.full(len(data.dates), np.nan, dtype=float),
        "soc_end": np.full(shape, np.nan, dtype=float),
        "emergency": np.zeros(shape, dtype=float),
        "spill": np.zeros(shape, dtype=float),
    }


def _record_actual_day(arrays: dict[str, np.ndarray], day: int, result: DayModelResult) -> None:
    charge = result.grid_charge_kwh[0] + result.pv_charge_kwh[0]
    discharge = result.discharge_kwh[0].copy()
    spill = result.grid_spill_kwh[0] + result.pv_spill_kwh[0]
    # 货币目标的等价最优解偶尔含数值退化的同段充放电。沿保持 SOC 的方向消去
    # 循环量，并把往返损耗对应的差额并入弃电；费用、供需平衡和 SOC 均不变。
    for slot in np.flatnonzero((charge > NUMERICAL_TOLERANCE) & (discharge > NUMERICAL_TOLERANCE)):
        removable_charge = min(
            float(charge[slot]),
            float(discharge[slot])
            / (BATTERY_CHARGE_EFFICIENCY * BATTERY_DISCHARGE_EFFICIENCY),
        )
        charge[slot] -= removable_charge
        discharge[slot] -= (
            BATTERY_CHARGE_EFFICIENCY
            * BATTERY_DISCHARGE_EFFICIENCY
            * removable_charge
        )
        spill[slot] += (
            1.0
            - BATTERY_CHARGE_EFFICIENCY * BATTERY_DISCHARGE_EFFICIENCY
        ) * removable_charge
    arrays["planned"][day] = result.planned_purchase_kwh
    arrays["charge"][day] = np.where(charge > NUMERICAL_TOLERANCE, charge, 0.0)
    arrays["discharge"][day] = np.where(discharge > NUMERICAL_TOLERANCE, discharge, 0.0)
    arrays["soc_start"][day] = BATTERY_INITIAL_SOC_KWH
    arrays["soc_end"][day] = result.soc_end_kwh[0]
    arrays["emergency"][day] = np.where(
        result.emergency_kwh[0] > NUMERICAL_TOLERANCE,
        result.emergency_kwh[0],
        0.0,
    )
    arrays["spill"][day] = np.where(spill > NUMERICAL_TOLERANCE, spill, 0.0)


def _strategy_from_arrays(
    arrays: dict[str, np.ndarray],
    *,
    method_id: str,
    name: str,
    solve_seconds: float,
    note: str,
    forecast_id: str | None,
) -> StrategyResult:
    return StrategyResult(
        method_id=method_id,
        name=name,
        planned_purchase_kwh=arrays["planned"],
        charge_kwh=arrays["charge"],
        discharge_kwh=arrays["discharge"],
        soc_start_kwh=arrays["soc_start"],
        soc_end_kwh=arrays["soc_end"],
        emergency_kwh=arrays["emergency"],
        surplus_or_curtailed_kwh=arrays["spill"],
        solve_seconds=float(solve_seconds),
        algorithm_note=note,
        forecast_id=forecast_id,
    )


def _low_load_day(date: pd.Timestamp) -> bool:
    return date.weekday() in (4, 5)


def select_similar_days(data: InputData, day: int, scenario_count: int = 8) -> np.ndarray:
    """从目标日前的真实历史中选日型和近期状态最相近的日期。"""
    if day <= 0:
        raise ValueError("相似日选择需要历史数据。")
    history = np.arange(max(1, day - 140), day, dtype=int)
    target_weekday = data.dates[day].weekday()
    target_low = _low_load_day(data.dates[day])
    previous_load = float(data.load_kw[day - 1].sum())
    previous_pv = float(data.pv_kw[day - 1].sum())
    hist_load_scale = max(float(np.std(data.load_kw[:day].sum(axis=1))), 1.0)
    hist_pv_scale = max(float(np.std(data.pv_kw[:day].sum(axis=1))), 1.0)
    scored: list[tuple[float, int]] = []
    for candidate in history:
        weekday = data.dates[candidate].weekday()
        same_weekday_penalty = 0.0 if weekday == target_weekday else 2.0
        same_class_penalty = 0.0 if _low_load_day(data.dates[candidate]) == target_low else 8.0
        context = (
            abs(float(data.load_kw[candidate - 1].sum()) - previous_load) / hist_load_scale
            + abs(float(data.pv_kw[candidate - 1].sum()) - previous_pv) / hist_pv_scale
        )
        recency = (day - candidate) / 70.0
        scored.append((same_class_penalty + same_weekday_penalty + context + recency, int(candidate)))
    selected = [candidate for _, candidate in sorted(scored)[: min(scenario_count, len(scored))]]
    if not selected:
        raise RuntimeError("没有可用相似日。")
    return np.asarray(selected, dtype=int)


def _residual_scenario_days(
    data: InputData,
    forecast: ForecastBundle,
    day: int,
    scenario_count: int,
) -> np.ndarray:
    finite = np.isfinite(forecast.load_kw[:day]).all(axis=1) & np.isfinite(forecast.pv_kw[:day]).all(axis=1)
    candidates = np.flatnonzero(finite)
    candidates = candidates[candidates < day]
    if candidates.size == 0:
        raise RuntimeError("尚无可回收的历史预测残差。")
    target_weekday = data.dates[day].weekday()
    target_low = _low_load_day(data.dates[day])
    score: list[tuple[float, int]] = []
    for candidate in candidates:
        weekday = data.dates[candidate].weekday()
        weekday_penalty = 0.0 if weekday == target_weekday else 1.5
        class_penalty = 0.0 if _low_load_day(data.dates[candidate]) == target_low else 7.0
        recency = (day - candidate) / 35.0
        score.append((class_penalty + weekday_penalty + recency, int(candidate)))
    return np.asarray(
        [candidate for _, candidate in sorted(score)[: min(scenario_count, len(score))]],
        dtype=int,
    )


def _recency_probabilities(day: int, scenario_days: np.ndarray, half_life_days: float = 42.0) -> np.ndarray:
    ages = day - np.asarray(scenario_days, dtype=float)
    weights = np.exp(-np.log(2.0) * ages / half_life_days)
    return weights / weights.sum()


def make_direct_similar_scenarios(
    data: InputData,
    day: int,
    scenario_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = select_similar_days(data, day, scenario_count)
    return (
        data.load_kw[selected].copy(),
        data.pv_kw[selected].copy(),
        _recency_probabilities(day, selected),
        selected,
    )


def make_residual_scenarios(
    data: InputData,
    forecast: ForecastBundle,
    day: int,
    scenario_count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = _residual_scenario_days(data, forecast, day, scenario_count)
    load_residual = data.load_kw[selected] - forecast.load_kw[selected]
    pv_residual = data.pv_kw[selected] - forecast.pv_kw[selected]
    history_load_cap = float(data.load_kw[:day].max())
    history_pv_cap = float(data.pv_kw[:day].max())
    scenarios_load = np.clip(
        forecast.load_kw[day][None, :] + load_residual,
        0.0,
        history_load_cap,
    )
    scenarios_pv = np.clip(
        forecast.pv_kw[day][None, :] + pv_residual,
        0.0,
        history_pv_cap,
    )
    return (
        scenarios_load,
        scenarios_pv,
        _recency_probabilities(day, selected),
        selected,
    )


def _actual_recourse(data: InputData, day: int, plan_kwh: np.ndarray) -> DayModelResult:
    return solve_day_model(
        data.load_kw[day],
        data.pv_kw[day],
        data.price_yuan_per_kwh,
        np.array([1.0]),
        initial_soc_kwh=BATTERY_INITIAL_SOC_KWH,
        terminal_soc_kwh=BATTERY_INITIAL_SOC_KWH,
        fixed_plan_kwh=plan_kwh,
    )


def run_point_forecast_strategy(
    data: InputData,
    forecast: ForecastBundle,
    *,
    method_id: str,
    name: str,
    day_indices: Iterable[int] | None = None,
    progress: bool = True,
) -> StrategyResult:
    indices = np.asarray(list(day_indices) if day_indices is not None else output_indices(data), dtype=int)
    arrays = _empty_strategy_arrays(data)
    elapsed = 0.0
    for order, day in enumerate(indices, start=1):
        if not np.isfinite(forecast.load_kw[day]).all() or not np.isfinite(forecast.pv_kw[day]).all():
            raise ValueError(f"{forecast.name} 在 {data.dates[day].date()} 没有可用预测。")
        planned_model = solve_day_model(
            forecast.load_kw[day],
            forecast.pv_kw[day],
            data.price_yuan_per_kwh,
            np.array([1.0]),
            terminal_soc_kwh=BATTERY_INITIAL_SOC_KWH,
        )
        actual = _actual_recourse(data, day, planned_model.planned_purchase_kwh)
        elapsed += planned_model.solve_seconds + actual.solve_seconds
        _record_actual_day(arrays, day, actual)
        if progress and (order % 50 == 0 or order == len(indices)):
            print(f"  {name}：{order}/{len(indices)} 天", flush=True)
    return _strategy_from_arrays(
        arrays,
        method_id=method_id,
        name=name,
        solve_seconds=elapsed,
        note=(
            f"每天0:00以{forecast.name}的144点预测制定并锁定计划购电；"
            "实际值仅用于两阶段补救仿真；每日首尾SOC设为6000kWh"
        ),
        forecast_id=forecast.method_id,
    )


def run_scenario_strategy(
    data: InputData,
    *,
    method_id: str,
    name: str,
    scenario_mode: str,
    forecast: ForecastBundle | None = None,
    scenario_count: int = 8,
    risk_weight: float = 0.0,
    cvar_alpha: float = 0.90,
    day_indices: Iterable[int] | None = None,
    progress: bool = True,
) -> StrategyResult:
    indices = np.asarray(list(day_indices) if day_indices is not None else output_indices(data), dtype=int)
    arrays = _empty_strategy_arrays(data)
    elapsed = 0.0
    used_counts: list[int] = []
    for order, day in enumerate(indices, start=1):
        if scenario_mode == "direct_similar":
            load_s, pv_s, probabilities, selected = make_direct_similar_scenarios(
                data, day, scenario_count
            )
        elif scenario_mode == "residual":
            if forecast is None:
                raise ValueError("残差场景需要点预测。")
            load_s, pv_s, probabilities, selected = make_residual_scenarios(
                data, forecast, day, scenario_count
            )
        else:
            raise ValueError(f"未知场景模式：{scenario_mode}")
        used_counts.append(len(selected))
        planned_model = solve_day_model(
            load_s,
            pv_s,
            data.price_yuan_per_kwh,
            probabilities,
            terminal_soc_kwh=BATTERY_INITIAL_SOC_KWH,
            risk_weight=risk_weight,
            cvar_alpha=cvar_alpha,
        )
        actual = _actual_recourse(data, day, planned_model.planned_purchase_kwh)
        elapsed += planned_model.solve_seconds + actual.solve_seconds
        _record_actual_day(arrays, day, actual)
        if progress and (order % 25 == 0 or order == len(indices)):
            print(f"  {name}：{order}/{len(indices)} 天", flush=True)
    source_note = (
        "直接从目标日前的附件2历史相似日选取真实轨迹"
        if scenario_mode == "direct_similar"
        else f"在{forecast.name if forecast else '点预测'}上回放目标日前的成对历史残差"
    )
    return _strategy_from_arrays(
        arrays,
        method_id=method_id,
        name=name,
        solve_seconds=elapsed,
        note=(
            f"{source_note}；平均场景数{np.mean(used_counts):.2f}；"
            f"期望紧急电费+{risk_weight:g}×CVaR{cvar_alpha:.0%}；"
            "实际值仅用于补救仿真；每日首尾SOC为6000kWh"
        ),
        forecast_id=forecast.method_id if forecast is not None else None,
    )


def run_perfect_foresight_lower_bound(
    data: InputData,
    *,
    day_indices: Iterable[int] | None = None,
    progress: bool = True,
) -> StrategyResult:
    indices = np.asarray(list(day_indices) if day_indices is not None else output_indices(data), dtype=int)
    arrays = _empty_strategy_arrays(data)
    elapsed = 0.0
    for order, day in enumerate(indices, start=1):
        solved = solve_day_model(
            data.load_kw[day],
            data.pv_kw[day],
            data.price_yuan_per_kwh,
            np.array([1.0]),
            terminal_soc_kwh=BATTERY_INITIAL_SOC_KWH,
        )
        elapsed += solved.solve_seconds
        _record_actual_day(arrays, day, solved)
        if progress and (order % 75 == 0 or order == len(indices)):
            print(f"  完全预知下界：{order}/{len(indices)} 天", flush=True)
    return _strategy_from_arrays(
        arrays,
        method_id="perfect_foresight_lower_bound",
        name="完全预知未来下界（不可实施）",
        solve_seconds=elapsed,
        note="使用当天实际轨迹求解，仅作为理论成本下界，不参与可实施方法排名",
        forecast_id=None,
    )


def evaluate_strategy(
    data: InputData,
    strategy: StrategyResult,
    day_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    indices = np.asarray(
        list(day_indices) if day_indices is not None else output_indices(data), dtype=int
    )
    price = data.price_yuan_per_kwh[None, :]
    planned = strategy.planned_purchase_kwh[indices]
    charge = strategy.charge_kwh[indices]
    discharge = strategy.discharge_kwh[indices]
    emergency = strategy.emergency_kwh[indices]
    spill = strategy.surplus_or_curtailed_kwh[indices]
    soc_start = strategy.soc_start_kwh[indices]
    soc_end = strategy.soc_end_kwh[indices]
    load_energy = data.load_kw[indices] * DT_HOURS
    pv_energy = data.pv_kw[indices] * DT_HOURS
    balance = planned + pv_energy + discharge + emergency - load_energy - charge - spill
    unserved = np.maximum(-balance, 0.0)
    simultaneous = (charge > NUMERICAL_TOLERANCE) & (discharge > NUMERICAL_TOLERANCE)
    plan_cost = float(np.sum(planned * price))
    emergency_cost = float(np.sum(EMERGENCY_PRICE_MULTIPLIER * emergency * price))
    return {
        "model_version": MODEL_VERSION,
        "method_id": strategy.method_id,
        "method_name": strategy.name,
        "forecast_id": strategy.forecast_id,
        "algorithm_note": strategy.algorithm_note,
        "output_start": data.dates[indices[0]].strftime("%Y-%m-%d"),
        "output_end": data.dates[indices[-1]].strftime("%Y-%m-%d"),
        "output_days": int(len(indices)),
        "planned_energy_mwh": float(planned.sum() / 1000.0),
        "emergency_energy_mwh": float(emergency.sum() / 1000.0),
        "spill_energy_mwh": float(spill.sum() / 1000.0),
        "scheduled_cost_yuan": plan_cost,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": plan_cost + emergency_cost,
        "emergency_intervals": int((emergency > NUMERICAL_TOLERANCE).sum()),
        "emergency_days": int((emergency.sum(axis=1) > NUMERICAL_TOLERANCE).sum()),
        "supply_interruptions": int((unserved > NUMERICAL_TOLERANCE).sum()),
        "unserved_energy_kwh": float(unserved.sum()),
        "charge_energy_mwh": float(charge.sum() / 1000.0),
        "discharge_energy_mwh": float(discharge.sum() / 1000.0),
        "soc_min_kwh": float(min(np.nanmin(soc_start), np.nanmin(soc_end))),
        "soc_max_kwh": float(max(np.nanmax(soc_start), np.nanmax(soc_end))),
        "initial_soc_kwh": float(soc_start[0]),
        "final_soc_kwh": float(soc_end[-1, -1]),
        "max_daily_terminal_soc_error_kwh": float(np.max(np.abs(soc_end[:, -1] - soc_start))),
        "max_charge_power_kw": float(np.max(charge) / DT_HOURS),
        "max_discharge_power_kw": float(np.max(discharge) / DT_HOURS),
        "simultaneous_charge_discharge_intervals": int(simultaneous.sum()),
        "max_supply_balance_violation_kwh": float(np.max(np.abs(balance))),
        "solve_seconds": float(strategy.solve_seconds),
    }


def daily_summary(data: InputData, strategy: StrategyResult) -> pd.DataFrame:
    indices = output_indices(data)
    price = data.price_yuan_per_kwh[None, :]
    planned = strategy.planned_purchase_kwh[indices]
    emergency = strategy.emergency_kwh[indices]
    frame = pd.DataFrame(
        {
            "日期": data.dates[indices].strftime("%Y-%m-%d"),
            "0:00储电量（kWh）": strategy.soc_start_kwh[indices],
            "24:00储电量（kWh）": strategy.soc_end_kwh[indices, -1],
            "计划购电量（kWh）": planned.sum(axis=1),
            "紧急购电量（kWh）": emergency.sum(axis=1),
            "充电量（kWh）": strategy.charge_kwh[indices].sum(axis=1),
            "放电量（kWh）": strategy.discharge_kwh[indices].sum(axis=1),
            "弃电量（kWh）": strategy.surplus_or_curtailed_kwh[indices].sum(axis=1),
            "计划购电费（元）": np.sum(planned * price, axis=1),
            "紧急购电费（元）": np.sum(
                EMERGENCY_PRICE_MULTIPLIER * emergency * price, axis=1
            ),
        }
    )
    frame["总购电费（元）"] = frame["计划购电费（元）"] + frame["紧急购电费（元）"]
    return frame


def interval_summary(data: InputData, strategy: StrategyResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in output_indices(data):
        for slot in range(SLOTS_PER_DAY):
            rows.append(
                {
                    "日期": data.dates[day].strftime("%Y-%m-%d"),
                    "时间段": data.result_interval_headers[slot],
                    "电价（元/kWh）": data.price_yuan_per_kwh[slot],
                    "实际负载（kW）": data.load_kw[day, slot],
                    "实际光伏（kW）": data.pv_kw[day, slot],
                    "计划购电量（kWh）": strategy.planned_purchase_kwh[day, slot],
                    "充电量（kWh）": strategy.charge_kwh[day, slot],
                    "放电量（kWh）": strategy.discharge_kwh[day, slot],
                    "时段末储电量（kWh）": strategy.soc_end_kwh[day, slot],
                    "紧急购电量（kWh）": strategy.emergency_kwh[day, slot],
                    "弃电量（kWh）": strategy.surplus_or_curtailed_kwh[day, slot],
                }
            )
    return pd.DataFrame(rows)


def merge_emergency_blocks(
    emergency_kwh: np.ndarray,
    interval_headers: list[str],
) -> list[dict[str, Any]]:
    active = np.flatnonzero(np.asarray(emergency_kwh) > NUMERICAL_TOLERANCE)
    if active.size == 0:
        return []
    groups: list[list[int]] = [[int(active[0])]]
    for value in active[1:]:
        index = int(value)
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    blocks: list[dict[str, Any]] = []
    for group in groups:
        first = interval_headers[group[0]]
        last = interval_headers[group[-1]]
        start = first.split("-", 1)[0]
        end = last.split("-", 1)[1]
        blocks.append(
            {
                "interval": f"{start}-{end}",
                "purchase_kwh": float(np.sum(emergency_kwh[group])),
                "start_slot": group[0],
                "end_slot": group[-1],
            }
        )
    return blocks


def make_workbook_payload(data: InputData, strategy: StrategyResult) -> dict[str, Any]:
    plan_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    emergency_rows: list[dict[str, Any]] = []
    for day in output_indices(data):
        date_text = data.dates[day].strftime("%Y-%m-%d")
        planned = strategy.planned_purchase_kwh[day]
        emergency = strategy.emergency_kwh[day]
        plan_rows.append(
            {
                "date": date_text,
                "interval_kwh": [float(value) for value in planned],
                "daily_purchase_kwh": float(planned.sum()),
                "daily_scheduled_cost_yuan": float(np.dot(planned, data.price_yuan_per_kwh)),
            }
        )
        for block_index, label in enumerate(FOUR_HOUR_BLOCKS):
            lo, hi = block_index * 24, (block_index + 1) * 24
            storage_rows.append(
                {
                    "date": date_text if block_index == 0 else None,
                    "interval": label,
                    "charge_kwh": float(strategy.charge_kwh[day, lo:hi].sum()),
                    "discharge_kwh": float(strategy.discharge_kwh[day, lo:hi].sum()),
                    "soc_time": "0:00" if block_index == 0 else "24:00" if block_index == 1 else None,
                    "soc_kwh": (
                        float(strategy.soc_start_kwh[day])
                        if block_index == 0
                        else float(strategy.soc_end_kwh[day, -1])
                        if block_index == 1
                        else None
                    ),
                }
            )
        blocks = merge_emergency_blocks(emergency, data.result_interval_headers)
        if not blocks:
            emergency_rows.append({"date": date_text, "interval": "无", "purchase_kwh": 0.0})
        else:
            for index, block in enumerate(blocks):
                emergency_rows.append(
                    {
                        "date": date_text if index == 0 else None,
                        "interval": block["interval"],
                        "purchase_kwh": block["purchase_kwh"],
                    }
                )
    return {
        "metadata": {
            "model_version": MODEL_VERSION,
            "source_template": str(RESULT_TEMPLATE.relative_to(ROOT)),
            "information_assumption": "第d天0:00仅使用d日前附件2历史；当天实测仅事后仿真",
            "terminal_assumption": "为避免日末短视并保持逐日可比，日首日末SOC均固定6000kWh",
            "method_id": strategy.method_id,
            "method_name": strategy.name,
            "output_start": OUTPUT_START.strftime("%Y-%m-%d"),
            "output_end": data.dates[-1].strftime("%Y-%m-%d"),
            "days": int(len(output_indices(data))),
            "slots_per_day": SLOTS_PER_DAY,
            "interval_headers": data.result_interval_headers,
            "emergency_price_multiplier": EMERGENCY_PRICE_MULTIPLIER,
        },
        "planned_purchase_rows": plan_rows,
        "storage_rows": storage_rows,
        "emergency_rows": emergency_rows,
    }


def make_key_date_results(data: InputData, strategy: StrategyResult) -> list[dict[str, Any]]:
    header_to_slot = {header: slot for slot, header in enumerate(data.result_interval_headers)}
    rows: list[dict[str, Any]] = []
    for date in KEY_DATES:
        day = int(np.flatnonzero(data.dates == date)[0])
        planned = strategy.planned_purchase_kwh[day]
        emergency = strategy.emergency_kwh[day]
        plan_cost = float(np.dot(planned, data.price_yuan_per_kwh))
        emergency_cost = float(
            np.dot(EMERGENCY_PRICE_MULTIPLIER * emergency, data.price_yuan_per_kwh)
        )
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "purchase_at_specified_intervals_kwh": {
                    header: float(planned[header_to_slot[header]])
                    for header in KEY_PURCHASE_INTERVALS
                },
                "daily_planned_purchase_kwh": float(planned.sum()),
                "daily_scheduled_cost_yuan": plan_cost,
                "daily_emergency_purchase_kwh": float(emergency.sum()),
                "daily_emergency_cost_yuan": emergency_cost,
                "daily_total_cost_yuan": plan_cost + emergency_cost,
                "storage_blocks": [
                    {
                        "interval": label,
                        "charge_kwh": float(
                            strategy.charge_kwh[day, index * 24 : (index + 1) * 24].sum()
                        ),
                        "discharge_kwh": float(
                            strategy.discharge_kwh[day, index * 24 : (index + 1) * 24].sum()
                        ),
                    }
                    for index, label in enumerate(FOUR_HOUR_BLOCKS)
                ],
                "soc_start_kwh": float(strategy.soc_start_kwh[day]),
                "soc_end_kwh": float(strategy.soc_end_kwh[day, -1]),
                "emergency_blocks": merge_emergency_blocks(
                    emergency, data.result_interval_headers
                ),
            }
        )
    return rows


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_strategy(strategy: StrategyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        method_id=np.array([strategy.method_id]),
        name=np.array([strategy.name]),
        planned=strategy.planned_purchase_kwh,
        charge=strategy.charge_kwh,
        discharge=strategy.discharge_kwh,
        soc_start=strategy.soc_start_kwh,
        soc_end=strategy.soc_end_kwh,
        emergency=strategy.emergency_kwh,
        spill=strategy.surplus_or_curtailed_kwh,
        solve_seconds=np.array([strategy.solve_seconds]),
        algorithm_note=np.array([strategy.algorithm_note]),
        forecast_id=np.array([strategy.forecast_id or ""]),
        model_version=np.array([MODEL_VERSION]),
    )


def load_strategy(path: Path) -> StrategyResult:
    arrays = np.load(path, allow_pickle=False)
    version = str(arrays["model_version"][0]) if "model_version" in arrays else ""
    if version != MODEL_VERSION:
        raise ValueError(f"缓存模型版本不一致：{version!r} != {MODEL_VERSION!r}")
    forecast_id = str(arrays["forecast_id"][0]) if "forecast_id" in arrays else ""
    return StrategyResult(
        method_id=str(arrays["method_id"][0]),
        name=str(arrays["name"][0]),
        planned_purchase_kwh=arrays["planned"],
        charge_kwh=arrays["charge"],
        discharge_kwh=arrays["discharge"],
        soc_start_kwh=arrays["soc_start"],
        soc_end_kwh=arrays["soc_end"],
        emergency_kwh=arrays["emergency"],
        surplus_or_curtailed_kwh=arrays["spill"],
        solve_seconds=float(arrays["solve_seconds"][0]),
        algorithm_note=str(arrays["algorithm_note"][0]),
        forecast_id=forecast_id or None,
    )


def create_solution_figure(data: InputData, strategy: StrategyResult, path: Path) -> None:
    daily = daily_summary(data, strategy)
    dates = pd.to_datetime(daily["日期"])
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(dates, daily["计划购电费（元）"] / 10_000.0, label="计划购电费", color="#2f75b5")
    axes[0].plot(dates, daily["紧急购电费（元）"] / 10_000.0, label="紧急购电费", color="#c00000")
    axes[0].set_ylabel("每日费用（万元）")
    axes[0].set_title("因果日前计划与实际补救结果")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(dates, daily["紧急购电量（kWh）"] / 1000.0, color="#ed7d31", label="紧急购电量")
    axes[1].plot(dates, daily["弃电量（kWh）"] / 1000.0, color="#a5a5a5", label="弃电量")
    axes[1].set_ylabel("每日电量（MWh）")
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(dates, daily["计划购电量（kWh）"] / 1000.0, color="#70ad47", label="计划购电量")
    axes[2].plot(dates, daily["充电量（kWh）"] / 1000.0, color="#4472c4", alpha=0.8, label="充电量")
    axes[2].plot(dates, daily["放电量（kWh）"] / 1000.0, color="#ffc000", alpha=0.8, label="放电量")
    axes[2].set_ylabel("每日电量（MWh）")
    axes[2].set_xlabel("日期（主刻度为月，短刻度为周）")
    axes[2].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
        axis.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        axis.grid(alpha=0.2)
        axis.set_xlim(dates.min(), dates.max())
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_final_outputs(data: InputData, strategy: StrategyResult) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_strategy(data, strategy)
    (OUTPUT_DIR / "q2_solution_metrics.json").write_text(
        json.dumps(json_ready(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    daily_summary(data, strategy).to_csv(
        OUTPUT_DIR / "q2_daily_summary.csv", index=False, encoding="utf-8-sig"
    )
    interval_summary(data, strategy).to_csv(
        OUTPUT_DIR / "q2_interval_strategy.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "q2_workbook_payload.json").write_text(
        json.dumps(json_ready(make_workbook_payload(data, strategy)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "q2_key_dates.json").write_text(
        json.dumps(json_ready(make_key_date_results(data, strategy)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_strategy(strategy, OUTPUT_DIR / "q2_final_solution.npz")
    create_solution_figure(data, strategy, FIGURE_DIR / "07_q2_strategy_results.png")
    return metrics
