"""2026 国赛 C 题“解答2”统一计算核心。

本模块只实现可复核的数值部分：

* 读取附件 1--4 与题目结果模板；
* 载入问题二已经完成滚动检验的严格因果 Ridge 负载/光伏预测；
* 将附件 3 的整点光伏预报因果地插值到 10 分钟；
* 构造只取自目标日前历史残差的联合场景；
* 求解计划购电、逐次调整、储能补救与紧急购电线性规划；
* 生成符合附件 5 字段结构的工作簿、汇总表、图片与校验量。

所有功率先乘 1/6 h 转为区间电量。紧急购电只进入负载支路，不能给
储能充电；计划电和光伏分别允许供负载、充电或弃置。问题三多次调整
采用“与上一版有效计划比较、逐次结算”的明确假设。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "program" / "_deps"
PROGRAM2 = ROOT / "program" / "2"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
if PROGRAM2.exists():
    sys.path.insert(0, str(PROGRAM2))

# matplotlib 默认尝试写用户目录；改到工作区，保证受限环境也可复现。
MPL_DIR = ROOT / "result" / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xlsxwriter  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

import q2_causal_core as q2  # noqa: E402


DT_HOURS = 1.0 / 6.0
SLOTS = 144
OUTPUT_START_INDEX = 31  # 2025-02-01；一月仅作为历史热启动。
ISSUES = (0, 6, 12, 18)
ISSUE_TO_SLOT = {value: value * 6 for value in ISSUES}
ETA_C = 0.90
ETA_D = 0.90
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_TARGET = 6000.0
POWER_ENERGY_LIMIT = 5000.0 * DT_HOURS
EMERGENCY_MULTIPLIER = 5.0
UP_MULTIPLIER = 1.5
DOWN_REFUND_MULTIPLIER = 0.5
FLOW_TIE_BREAKER = 1.0e-9
TOL = 1.0e-6
SCENARIO_COUNT = 6
KEY_DATES = (
    pd.Timestamp("2025-03-20"),
    pd.Timestamp("2025-06-21"),
    pd.Timestamp("2025-09-23"),
    pd.Timestamp("2025-12-21"),
)
KEY_INTERVALS = (
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

CHINESE_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
if CHINESE_FONT.exists():
    font_manager.fontManager.addfont(str(CHINESE_FONT))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class DataBundle:
    base: q2.InputData
    ridge: q2.ForecastBundle
    pv_hourly: np.ndarray  # (365, 4, 24)
    price_actual: np.ndarray  # (365, 144)
    fixed_price: np.ndarray  # (144,)


@dataclass
class StageResult:
    purchase_kwh: np.ndarray
    emergency_kwh: np.ndarray
    grid_charge_kwh: np.ndarray
    pv_charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    soc_end_kwh: np.ndarray
    grid_spill_kwh: np.ndarray
    pv_spill_kwh: np.ndarray
    objective_without_constant: float
    solve_seconds: float


@dataclass
class RollingResult:
    name: str
    issues: tuple[int, ...]
    dates: pd.DatetimeIndex
    initial_plan_kwh: np.ndarray
    final_purchase_kwh: np.ndarray
    charge_kwh: np.ndarray
    discharge_kwh: np.ndarray
    emergency_kwh: np.ndarray
    spill_kwh: np.ndarray
    soc_start_kwh: np.ndarray
    soc_end_kwh: np.ndarray
    upward_by_issue_kwh: np.ndarray  # (D, 4, 144)
    downward_by_issue_kwh: np.ndarray
    daily_plan_cost_yuan: np.ndarray
    daily_adjustment_cost_yuan: np.ndarray
    daily_emergency_cost_yuan: np.ndarray
    daily_total_cost_yuan: np.ndarray
    solve_seconds: float
    execution_target_relaxations: int


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_data() -> DataBundle:
    """读取并交叉核验全部附件和问题二预测缓存。"""
    base = q2.load_inputs()
    cache_path = ROOT / "result" / "2" / "tmp" / "causal_compare" / "causal_forecasts.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少问题二严格因果预测缓存：{cache_path}")
    cache = np.load(cache_path, allow_pickle=False)
    ridge = q2.ForecastBundle(
        method_id="ridge_causal",
        name="扩展窗口 Ridge 严格因果预测",
        load_kw=np.asarray(cache["ridge_causal_load"], dtype=float),
        pv_kw=np.asarray(cache["ridge_causal_pv"], dtype=float),
        fit_seconds=0.0,
        note="由问题二滚动回测选出的预测；目标日真实值不进入计划阶段",
        hyperparameters={"alpha": 0.1},
    )

    attachment_dir = ROOT / "CUMCM2026Problems" / "C题" / "附件"
    forecast_frame = pd.read_excel(attachment_dir / "附件3.xlsx")
    forecast_frame.iloc[:, 0] = forecast_frame.iloc[:, 0].ffill()
    forecast_dates = pd.to_datetime(forecast_frame.iloc[:, 0]).dt.normalize()
    issue_values = (
        forecast_frame.iloc[:, 1]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
        .to_numpy()
    )
    pv_hourly = np.full((len(base.dates), 4, 24), np.nan, dtype=float)
    date_to_index = {value: index for index, value in enumerate(base.dates)}
    issue_to_index = {value: index for index, value in enumerate(ISSUES)}
    for row in range(len(forecast_frame)):
        day = date_to_index[pd.Timestamp(forecast_dates.iloc[row])]
        issue = int(issue_values[row])
        values = pd.to_numeric(forecast_frame.iloc[row, 2:], errors="raise").to_numpy(float)
        pv_hourly[day, issue_to_index[issue]] = values
    if not np.isfinite(pv_hourly).all() or (pv_hourly < 0).any():
        raise ValueError("附件3预报存在缺失、非有限值或负值。")

    price_frame = pd.read_excel(attachment_dir / "附件4.xlsx")
    price_dates = pd.DatetimeIndex(pd.to_datetime(price_frame.iloc[:, 0])).normalize()
    price_actual = price_frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not price_dates.equals(base.dates) or price_actual.shape != base.load_kw.shape:
        raise ValueError("附件4日期/时间网格与附件2不一致。")
    if not np.isfinite(price_actual).all() or (price_actual <= 0).any():
        raise ValueError("附件4含缺失、非有限或非正电价。")

    return DataBundle(
        base=base,
        ridge=ridge,
        pv_hourly=pv_hourly,
        price_actual=price_actual,
        fixed_price=base.price_yuan_per_kwh.copy(),
    )


PV_FUSION_WEIGHTS = {0: 0.50, 6: 0.75, 12: 0.95, 18: 0.75}


def attachment3_pv_forecast_curve(
    data: DataBundle, day: int, issue: int
) -> np.ndarray:
    """把附件3在发布时刻后的整点预报线性插值至十分钟网格。"""
    start = ISSUE_TO_SLOT[issue]
    if issue == 0:
        anchor = float(data.base.pv_kw[day - 1, -1]) if day > 0 else 0.0
    else:
        anchor = float(data.base.pv_kw[day, start - 1])
    hourly = data.pv_hourly[day, ISSUES.index(issue)]
    result: list[float] = []
    for slot in range(start, SLOTS):
        target_minutes = (slot + 1) * 10
        lead = (target_minutes - issue * 60) / 60.0
        lower = math.floor(lead)
        upper = math.ceil(lead)
        if lower == upper:
            value = hourly[lower - 1]
        else:
            lower_value = anchor if lower == 0 else hourly[lower - 1]
            upper_value = hourly[upper - 1]
            weight = lead - lower
            value = (1.0 - weight) * lower_value + weight * upper_value
        result.append(max(0.0, float(value)))
    return np.asarray(result, dtype=float)


def ridge_pv_forecast_curve(data: DataBundle, day: int, issue: int) -> np.ndarray:
    """问题二严格因果 Ridge 光伏预报在当前时刻之后的曲线。"""
    start = ISSUE_TO_SLOT[issue]
    values = np.asarray(data.ridge.pv_kw[day, start:], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{data.base.dates[day]:%Y-%m-%d} 的 Ridge 光伏预报不可用。")
    return np.clip(values, 0.0, None)


def pv_forecast_curve(data: DataBundle, day: int, issue: int) -> np.ndarray:
    """仅1月顺序验证定权的附件3/Ridge融合光伏预报。

    候选附件3权重为 0,0.05,...,1，以 1 月 15—31 日剩余时域
    净负荷 MAE 选择；2—12 月不再重估，避免信息泄漏。
    """
    weight = PV_FUSION_WEIGHTS[issue]
    attachment3 = attachment3_pv_forecast_curve(data, day, issue)
    ridge = ridge_pv_forecast_curve(data, day, issue)
    return weight * attachment3 + (1.0 - weight) * ridge


def load_forecast_curve(data: DataBundle, day: int, issue: int) -> np.ndarray:
    """Ridge 0:00 预测；日内只用已经发生的负荷作比例校正。"""
    start = ISSUE_TO_SLOT[issue]
    base = np.asarray(data.ridge.load_kw[day], dtype=float)
    if not np.isfinite(base).all():
        raise ValueError(f"{data.base.dates[day]:%Y-%m-%d} 的 Ridge 预测不可用。")
    scale = 1.0
    if start > 0:
        denominator = float(base[:start].sum())
        if denominator > 1.0e-9:
            scale = float(data.base.load_kw[day, :start].sum() / denominator)
            scale = float(np.clip(scale, 0.90, 1.10))
    return np.clip(base[start:] * scale, 0.0, None)


def _low_load_day(value: pd.Timestamp) -> bool:
    return value.weekday() in (4, 5)


def _scenario_days(data: DataBundle, day: int, count: int = SCENARIO_COUNT) -> np.ndarray:
    """选择目标日前日型一致且较近的完整历史残差日。"""
    candidates = np.arange(max(14, day - 120), day, dtype=int)
    candidates = candidates[
        np.isfinite(data.ridge.load_kw[candidates]).all(axis=1)
        & np.isfinite(data.ridge.pv_kw[candidates]).all(axis=1)
    ]
    target_date = data.base.dates[day]
    scored: list[tuple[float, int]] = []
    for candidate in candidates:
        candidate_date = data.base.dates[candidate]
        class_penalty = 0.0 if _low_load_day(candidate_date) == _low_load_day(target_date) else 8.0
        weekday_penalty = 0.0 if candidate_date.weekday() == target_date.weekday() else 1.5
        recency = (day - int(candidate)) / 42.0
        scored.append((class_penalty + weekday_penalty + recency, int(candidate)))
    selected = [candidate for _, candidate in sorted(scored)[: min(count, len(scored))]]
    if not selected:
        raise RuntimeError("没有可用历史残差场景。")
    return np.asarray(selected, dtype=int)


def stage_scenarios(
    data: DataBundle, day: int, issue: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用最新预测叠加目标日前成对历史残差，保留时序/负荷-光伏相关性。"""
    start = ISSUE_TO_SLOT[issue]
    current_load = load_forecast_curve(data, day, issue)
    current_pv = pv_forecast_curve(data, day, issue)
    selected = _scenario_days(data, day)
    scenario_load: list[np.ndarray] = []
    scenario_pv: list[np.ndarray] = []
    load_cap = float(np.max(data.base.load_kw[:day]))
    pv_cap = float(np.max(data.base.pv_kw[:day]))
    for historical_day in selected:
        old_load = load_forecast_curve(data, int(historical_day), issue)
        old_pv = pv_forecast_curve(data, int(historical_day), issue)
        load_residual = data.base.load_kw[historical_day, start:] - old_load
        pv_residual = data.base.pv_kw[historical_day, start:] - old_pv
        scenario_load.append(np.clip(current_load + load_residual, 0.0, load_cap))
        scenario_pv.append(np.clip(current_pv + pv_residual, 0.0, pv_cap))
    ages = day - selected.astype(float)
    probabilities = np.exp(-np.log(2.0) * ages / 42.0)
    probabilities /= probabilities.sum()
    return (
        np.asarray(scenario_load),
        np.asarray(scenario_pv),
        probabilities,
        selected,
    )


def price_forecast(
    data: DataBundle, day: int, issue: int, method: str = "causal_selected"
) -> np.ndarray:
    """严格因果的波动电价预测。

    1 月顺序验证选择的规则为：0:00 用前 7 日同刻；6:00 用“前 7 日同刻
    与近 7 日均值”的混合曲线并乘当日已观测比例；12/18 点对同一混合
    曲线作已观测均值差校正。``fixed_mean`` 与 ``oracle`` 只作对照。
    """
    start = ISSUE_TO_SLOT[issue]
    if method == "fixed_mean":
        return data.fixed_price[start:].copy()
    if method == "oracle":
        return data.price_actual[day, start:].copy()
    if day < 7:
        raise ValueError("7日价格预测需要至少一周历史。")
    lag7 = data.price_actual[day - 7].copy()
    if method == "lag7" or issue == 0:
        return lag7[start:]
    mean7 = data.price_actual[day - 7 : day].mean(axis=0)
    blend = 0.5 * lag7 + 0.5 * mean7
    observed = data.price_actual[day, :start]
    if issue == 6:
        ratio = float(observed.mean() / max(float(blend[:start].mean()), 1.0e-9))
        prediction = blend[start:] * float(np.clip(ratio, 0.60, 1.40))
    else:
        difference = float(np.mean(observed - blend[:start]))
        prediction = blend[start:] + difference
    return np.clip(prediction, 0.001, None)


def price_forecast_metrics(data: DataBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, name in (
        ("fixed_mean", "附件1固定平均日曲线"),
        ("lag7", "前7日同刻"),
        ("causal_selected", "1月验证选出的分时更新规则"),
    ):
        for issue in ISSUES:
            actuals: list[np.ndarray] = []
            predictions: list[np.ndarray] = []
            for day in range(OUTPUT_START_INDEX, len(data.base.dates)):
                actuals.append(data.price_actual[day, ISSUE_TO_SLOT[issue] :])
                predictions.append(price_forecast(data, day, issue, method))
            actual = np.concatenate(actuals)
            predicted = np.concatenate(predictions)
            error = predicted - actual
            rows.append(
                {
                    "方法": name,
                    "发布时刻": f"{issue}:00",
                    "样本点数": int(error.size),
                    "MAE（元/kWh）": float(np.mean(np.abs(error))),
                    "RMSE（元/kWh）": float(np.sqrt(np.mean(error**2))),
                    "偏差（元/kWh）": float(np.mean(error)),
                    "相关系数": float(np.corrcoef(actual, predicted)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def solve_stage_lp(
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    decision_price: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    initial_soc_kwh: float = SOC_TARGET,
    terminal_soc_kwh: float | None = SOC_TARGET,
    reference_plan_kwh: np.ndarray | None = None,
    fixed_plan_kwh: np.ndarray | None = None,
) -> StageResult:
    """求任意剩余时域的两阶段流模型或固定计划补救模型。"""
    load = np.asarray(load_scenarios_kw, dtype=float)
    pv = np.asarray(pv_scenarios_kw, dtype=float)
    if load.ndim == 1:
        load = load[None, :]
    if pv.ndim == 1:
        pv = pv[None, :]
    if load.shape != pv.shape or load.ndim != 2:
        raise ValueError(f"负荷/光伏场景维度不一致：{load.shape}, {pv.shape}")
    if not np.isfinite(load).all() or not np.isfinite(pv).all() or (load < 0).any() or (pv < 0).any():
        raise ValueError("负荷与光伏场景必须为非负有限值。")
    scenarios, periods = load.shape
    price = np.asarray(decision_price, dtype=float).ravel()
    if price.shape != (periods,) or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("决策电价维度或取值异常。")
    if probabilities is None:
        probs = np.full(scenarios, 1.0 / scenarios)
    else:
        probs = np.asarray(probabilities, dtype=float).ravel()
        if probs.shape != (scenarios,) or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError("场景概率异常。")
        probs /= probs.sum()
    if reference_plan_kwh is not None and fixed_plan_kwh is not None:
        raise ValueError("reference_plan 与 fixed_plan 不能同时给出。")

    block = scenarios * periods
    offset = 0
    q_slice = slice(offset, offset + periods)
    offset += periods
    up_slice: slice | None = None
    down_slice: slice | None = None
    reference: np.ndarray | None = None
    if reference_plan_kwh is not None:
        reference = np.asarray(reference_plan_kwh, dtype=float).ravel()
        if reference.shape != (periods,) or (reference < -TOL).any():
            raise ValueError("上一版有效计划维度或取值异常。")
        up_slice = slice(offset, offset + periods)
        offset += periods
        down_slice = slice(offset, offset + periods)
        offset += periods
    slices: dict[str, slice] = {}
    for key in ("emergency", "grid_charge", "pv_charge", "discharge", "soc", "grid_spill", "pv_spill"):
        slices[key] = slice(offset, offset + block)
        offset += block
    variable_count = offset

    objective = np.zeros(variable_count, dtype=float)
    if fixed_plan_kwh is None and reference is None:
        objective[q_slice] = price
    elif reference is not None:
        assert up_slice is not None and down_slice is not None
        objective[up_slice] = UP_MULTIPLIER * price
        objective[down_slice] = -DOWN_REFUND_MULTIPLIER * price
    for scenario in range(scenarios):
        lo = scenario * periods
        hi = lo + periods
        objective[slices["emergency"].start + lo : slices["emergency"].start + hi] = (
            probs[scenario] * EMERGENCY_MULTIPLIER * price
        )
        tie = probs[scenario] * FLOW_TIE_BREAKER
        for key in ("grid_charge", "pv_charge", "discharge", "grid_spill", "pv_spill"):
            objective[slices[key].start + lo : slices[key].start + hi] = tie

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_values: list[float] = []
    b_eq: list[float] = []
    row = 0
    for scenario in range(scenarios):
        scenario_offset = scenario * periods
        for period in range(periods):
            flat = scenario_offset + period
            # q + e - c_grid - c_pv + d - spill_grid - spill_pv = load - pv
            eq_rows.extend([row] * 7)
            eq_cols.extend(
                [
                    q_slice.start + period,
                    slices["emergency"].start + flat,
                    slices["grid_charge"].start + flat,
                    slices["pv_charge"].start + flat,
                    slices["discharge"].start + flat,
                    slices["grid_spill"].start + flat,
                    slices["pv_spill"].start + flat,
                ]
            )
            eq_values.extend((1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0))
            b_eq.append((load[scenario, period] - pv[scenario, period]) * DT_HOURS)
            row += 1

            eq_rows.extend([row] * (5 if period > 0 else 4))
            eq_cols.extend(
                [
                    slices["soc"].start + flat,
                    slices["grid_charge"].start + flat,
                    slices["pv_charge"].start + flat,
                    slices["discharge"].start + flat,
                ]
            )
            eq_values.extend((1.0, -ETA_C, -ETA_C, 1.0 / ETA_D))
            if period > 0:
                eq_cols.append(slices["soc"].start + flat - 1)
                eq_values.append(-1.0)
                b_eq.append(0.0)
            else:
                b_eq.append(float(initial_soc_kwh))
            row += 1
    if reference is not None:
        assert up_slice is not None and down_slice is not None
        for period in range(periods):
            eq_rows.extend([row] * 3)
            eq_cols.extend(
                [q_slice.start + period, up_slice.start + period, down_slice.start + period]
            )
            eq_values.extend((1.0, -1.0, 1.0))
            b_eq.append(float(reference[period]))
            row += 1
    a_eq = coo_matrix(
        (eq_values, (eq_rows, eq_cols)), shape=(row, variable_count)
    ).tocsr()

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_values: list[float] = []
    b_ub: list[float] = []
    row = 0
    for scenario in range(scenarios):
        scenario_offset = scenario * periods
        for period in range(periods):
            flat = scenario_offset + period
            # 计划电只能供负载、充电或弃置。
            ub_rows.extend([row] * 3)
            ub_cols.extend(
                [
                    slices["grid_charge"].start + flat,
                    slices["grid_spill"].start + flat,
                    q_slice.start + period,
                ]
            )
            ub_values.extend((1.0, 1.0, -1.0))
            b_ub.append(0.0)
            row += 1
            # 光伏只能供负载、充电或弃置。
            ub_rows.extend([row] * 2)
            ub_cols.extend(
                [slices["pv_charge"].start + flat, slices["pv_spill"].start + flat]
            )
            ub_values.extend((1.0, 1.0))
            b_ub.append(float(pv[scenario, period] * DT_HOURS))
            row += 1
            # 充电与放电合计功率上限同时排除同段满功率充放循环。
            ub_rows.extend([row] * 3)
            ub_cols.extend(
                [
                    slices["grid_charge"].start + flat,
                    slices["pv_charge"].start + flat,
                    slices["discharge"].start + flat,
                ]
            )
            ub_values.extend((1.0, 1.0, 1.0))
            b_ub.append(POWER_ENERGY_LIMIT)
            row += 1
    a_ub = coo_matrix(
        (ub_values, (ub_rows, ub_cols)), shape=(row, variable_count)
    ).tocsr()

    bounds: list[tuple[float | None, float | None]] = []
    if fixed_plan_kwh is None:
        bounds.extend([(0.0, None)] * periods)
    else:
        fixed = np.asarray(fixed_plan_kwh, dtype=float).ravel()
        if fixed.shape != (periods,) or (fixed < -TOL).any():
            raise ValueError("固定购电计划维度或取值异常。")
        bounds.extend([(max(0.0, float(v)), max(0.0, float(v))) for v in fixed])
    if reference is not None:
        bounds.extend([(0.0, None)] * periods)
        bounds.extend([(0.0, max(0.0, float(v))) for v in reference])
    bounds.extend([(0.0, None)] * block)  # emergency
    bounds.extend([(0.0, POWER_ENERGY_LIMIT)] * block)  # grid charge
    bounds.extend([(0.0, POWER_ENERGY_LIMIT)] * block)  # pv charge
    bounds.extend([(0.0, POWER_ENERGY_LIMIT)] * block)  # discharge
    for _scenario in range(scenarios):
        for period in range(periods):
            if terminal_soc_kwh is not None and period == periods - 1:
                target = float(terminal_soc_kwh)
                bounds.append((target, target))
            else:
                bounds.append((SOC_MIN, SOC_MAX))
    bounds.extend([(0.0, None)] * block)  # grid spill
    bounds.extend([(0.0, None)] * block)  # pv spill

    started = time.perf_counter()
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.perf_counter() - started
    if not solved.success:
        raise RuntimeError(f"阶段线性规划失败：{solved.message}")
    x = solved.x

    def matrix(key: str) -> np.ndarray:
        return x[slices[key]].reshape(scenarios, periods)

    return StageResult(
        purchase_kwh=x[q_slice].copy(),
        emergency_kwh=matrix("emergency"),
        grid_charge_kwh=matrix("grid_charge"),
        pv_charge_kwh=matrix("pv_charge"),
        discharge_kwh=matrix("discharge"),
        soc_end_kwh=matrix("soc"),
        grid_spill_kwh=matrix("grid_spill"),
        pv_spill_kwh=matrix("pv_spill"),
        objective_without_constant=float(solved.fun),
        solve_seconds=float(elapsed),
    )


def execute_block(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    actual_price: np.ndarray,
    fixed_purchase_kwh: np.ndarray,
    initial_soc_kwh: float,
    desired_terminal_soc_kwh: float,
) -> tuple[StageResult, bool]:
    """固定计划后用真实场景补救；若目标 SOC 不可达则向当前 SOC 缩放。"""
    targets = [desired_terminal_soc_kwh]
    targets.extend(
        initial_soc_kwh + fraction * (desired_terminal_soc_kwh - initial_soc_kwh)
        for fraction in (0.9, 0.75, 0.5, 0.25, 0.0)
    )
    last_error: Exception | None = None
    for index, target in enumerate(targets):
        try:
            result = solve_stage_lp(
                load_kw,
                pv_kw,
                actual_price,
                np.array([1.0]),
                initial_soc_kwh=initial_soc_kwh,
                terminal_soc_kwh=float(np.clip(target, SOC_MIN, SOC_MAX)),
                fixed_plan_kwh=fixed_purchase_kwh,
            )
            return result, index > 0
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(f"固定计划补救阶段无可行解：{last_error}")


def run_rolling_policy(
    data: DataBundle,
    allowed_issues: Sequence[int],
    *,
    variable_price: bool,
    price_method: str = "causal_selected",
    day_limit: int | None = None,
    progress_prefix: str = "",
) -> RollingResult:
    """逐日运行某个预报更新组合，并按上一版计划逐次结算。"""
    issues = tuple(sorted(set(int(value) for value in allowed_issues)))
    if not issues or issues[0] != 0 or any(value not in ISSUES for value in issues):
        raise ValueError("滚动策略必须从0:00开始，且只能使用0/6/12/18点。")
    global_days = np.arange(OUTPUT_START_INDEX, len(data.base.dates), dtype=int)
    if day_limit is not None:
        global_days = global_days[:day_limit]
    count = len(global_days)
    shape = (count, SLOTS)
    initial_plan = np.zeros(shape)
    final_purchase = np.zeros(shape)
    charge = np.zeros(shape)
    discharge = np.zeros(shape)
    emergency = np.zeros(shape)
    spill = np.zeros(shape)
    soc_start = np.full(count, SOC_TARGET)
    soc_end = np.full(shape, np.nan)
    upward = np.zeros((count, len(ISSUES), SLOTS))
    downward = np.zeros_like(upward)
    plan_cost = np.zeros(count)
    adjustment_cost = np.zeros(count)
    emergency_cost = np.zeros(count)
    total_cost = np.zeros(count)
    solve_seconds = 0.0
    relaxations = 0

    for local_day, global_day in enumerate(global_days):
        actual_price_day = (
            data.price_actual[global_day] if variable_price else data.fixed_price
        )
        effective = np.zeros(SLOTS)
        current_soc = SOC_TARGET
        for stage_index, issue in enumerate(issues):
            start = ISSUE_TO_SLOT[issue]
            end = ISSUE_TO_SLOT[issues[stage_index + 1]] if stage_index + 1 < len(issues) else SLOTS
            scenario_load, scenario_pv, probabilities, _ = stage_scenarios(
                data, global_day, issue
            )
            decision_price = (
                price_forecast(data, global_day, issue, price_method)
                if variable_price
                else data.fixed_price[start:]
            )
            previous = None if issue == 0 else effective[start:].copy()
            stage = solve_stage_lp(
                scenario_load,
                scenario_pv,
                decision_price,
                probabilities,
                initial_soc_kwh=current_soc,
                terminal_soc_kwh=SOC_TARGET,
                reference_plan_kwh=previous,
            )
            solve_seconds += stage.solve_seconds
            if issue == 0:
                initial_plan[local_day] = stage.purchase_kwh
                effective[:] = stage.purchase_kwh
            else:
                assert previous is not None
                change = stage.purchase_kwh - previous
                up = np.maximum(change, 0.0)
                down = np.maximum(-change, 0.0)
                issue_index = ISSUES.index(issue)
                upward[local_day, issue_index, start:] = up
                downward[local_day, issue_index, start:] = down
                adjustment_cost[local_day] += float(
                    np.dot(
                        actual_price_day[start:],
                        UP_MULTIPLIER * up - DOWN_REFUND_MULTIPLIER * down,
                    )
                )
                effective[start:] = stage.purchase_kwh

            block_length = end - start
            if end == SLOTS:
                target_soc = SOC_TARGET
            else:
                target_soc = float(
                    np.dot(probabilities, stage.soc_end_kwh[:, block_length - 1])
                )
            actual, relaxed = execute_block(
                data.base.load_kw[global_day, start:end],
                data.base.pv_kw[global_day, start:end],
                actual_price_day[start:end],
                effective[start:end],
                current_soc,
                target_soc,
            )
            solve_seconds += actual.solve_seconds
            relaxations += int(relaxed)
            block_charge = actual.grid_charge_kwh[0] + actual.pv_charge_kwh[0]
            charge[local_day, start:end] = block_charge
            discharge[local_day, start:end] = actual.discharge_kwh[0]
            emergency[local_day, start:end] = actual.emergency_kwh[0]
            spill[local_day, start:end] = (
                actual.grid_spill_kwh[0] + actual.pv_spill_kwh[0]
            )
            soc_end[local_day, start:end] = actual.soc_end_kwh[0]
            current_soc = float(actual.soc_end_kwh[0, -1])

        final_purchase[local_day] = effective
        plan_cost[local_day] = float(np.dot(actual_price_day, initial_plan[local_day]))
        emergency_cost[local_day] = float(
            np.dot(actual_price_day, EMERGENCY_MULTIPLIER * emergency[local_day])
        )
        total_cost[local_day] = (
            plan_cost[local_day]
            + adjustment_cost[local_day]
            + emergency_cost[local_day]
        )
        if progress_prefix and (
            local_day == 0 or (local_day + 1) % 25 == 0 or local_day + 1 == count
        ):
            print(
                f"{progress_prefix} {issues}：{local_day+1}/{count} 天，"
                f"累计费用 {total_cost[:local_day+1].sum()/10000:.2f} 万元",
                flush=True,
            )

    name = "+".join(f"{value}:00" for value in issues)
    return RollingResult(
        name=name,
        issues=issues,
        dates=data.base.dates[global_days],
        initial_plan_kwh=initial_plan,
        final_purchase_kwh=final_purchase,
        charge_kwh=charge,
        discharge_kwh=discharge,
        emergency_kwh=emergency,
        spill_kwh=spill,
        soc_start_kwh=soc_start,
        soc_end_kwh=soc_end,
        upward_by_issue_kwh=upward,
        downward_by_issue_kwh=downward,
        daily_plan_cost_yuan=plan_cost,
        daily_adjustment_cost_yuan=adjustment_cost,
        daily_emergency_cost_yuan=emergency_cost,
        daily_total_cost_yuan=total_cost,
        solve_seconds=float(solve_seconds),
        execution_target_relaxations=relaxations,
    )


def rolling_metrics(result: RollingResult) -> dict[str, Any]:
    simultaneous = (result.charge_kwh > TOL) & (result.discharge_kwh > TOL)
    soc_min = float(np.nanmin(result.soc_end_kwh))
    soc_max = float(np.nanmax(result.soc_end_kwh))
    final_error = float(np.max(np.abs(result.soc_end_kwh[:, -1] - SOC_TARGET)))
    return {
        "方案": result.name,
        "使用时刻": list(result.issues),
        "天数": len(result.dates),
        "初始计划购电量_MWh": float(result.initial_plan_kwh.sum() / 1000.0),
        "最终执行购电量_MWh": float(result.final_purchase_kwh.sum() / 1000.0),
        "向上调整量_MWh": float(result.upward_by_issue_kwh.sum() / 1000.0),
        "向下调整量_MWh": float(result.downward_by_issue_kwh.sum() / 1000.0),
        "紧急购电量_MWh": float(result.emergency_kwh.sum() / 1000.0),
        "充电量_MWh": float(result.charge_kwh.sum() / 1000.0),
        "放电量_MWh": float(result.discharge_kwh.sum() / 1000.0),
        "弃电量_MWh": float(result.spill_kwh.sum() / 1000.0),
        "计划购电费_元": float(result.daily_plan_cost_yuan.sum()),
        "调整费用净额_元": float(result.daily_adjustment_cost_yuan.sum()),
        "紧急购电费_元": float(result.daily_emergency_cost_yuan.sum()),
        "总购电费_元": float(result.daily_total_cost_yuan.sum()),
        "发生紧急购电天数": int(np.sum(result.emergency_kwh.sum(axis=1) > TOL)),
        "紧急购电十分钟时段数": int(np.sum(result.emergency_kwh > TOL)),
        "供电中断次数": 0,
        "SOC最小值_kWh": soc_min,
        "SOC最大值_kWh": soc_max,
        "日末SOC最大误差_kWh": final_error,
        "最大充电功率_kW": float(result.charge_kwh.max() / DT_HOURS),
        "最大放电功率_kW": float(result.discharge_kwh.max() / DT_HOURS),
        "同时充放电时段数": int(simultaneous.sum()),
        "执行SOC目标放宽次数": result.execution_target_relaxations,
        "求解时间_秒": result.solve_seconds,
    }


def save_rolling_result(path: Path, result: RollingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.array([result.name]),
        issues=np.asarray(result.issues),
        dates=np.asarray(result.dates.strftime("%Y-%m-%d"), dtype="U10"),
        initial_plan_kwh=result.initial_plan_kwh,
        final_purchase_kwh=result.final_purchase_kwh,
        charge_kwh=result.charge_kwh,
        discharge_kwh=result.discharge_kwh,
        emergency_kwh=result.emergency_kwh,
        spill_kwh=result.spill_kwh,
        soc_start_kwh=result.soc_start_kwh,
        soc_end_kwh=result.soc_end_kwh,
        upward_by_issue_kwh=result.upward_by_issue_kwh,
        downward_by_issue_kwh=result.downward_by_issue_kwh,
        daily_plan_cost_yuan=result.daily_plan_cost_yuan,
        daily_adjustment_cost_yuan=result.daily_adjustment_cost_yuan,
        daily_emergency_cost_yuan=result.daily_emergency_cost_yuan,
        daily_total_cost_yuan=result.daily_total_cost_yuan,
        solve_seconds=np.array([result.solve_seconds]),
        execution_target_relaxations=np.array([result.execution_target_relaxations]),
    )


def load_rolling_result(path: Path) -> RollingResult:
    z = np.load(path, allow_pickle=False)
    return RollingResult(
        name=str(z["name"][0]),
        issues=tuple(int(v) for v in z["issues"]),
        dates=pd.DatetimeIndex(pd.to_datetime(z["dates"])),
        initial_plan_kwh=z["initial_plan_kwh"],
        final_purchase_kwh=z["final_purchase_kwh"],
        charge_kwh=z["charge_kwh"],
        discharge_kwh=z["discharge_kwh"],
        emergency_kwh=z["emergency_kwh"],
        spill_kwh=z["spill_kwh"],
        soc_start_kwh=z["soc_start_kwh"],
        soc_end_kwh=z["soc_end_kwh"],
        upward_by_issue_kwh=z["upward_by_issue_kwh"],
        downward_by_issue_kwh=z["downward_by_issue_kwh"],
        daily_plan_cost_yuan=z["daily_plan_cost_yuan"],
        daily_adjustment_cost_yuan=z["daily_adjustment_cost_yuan"],
        daily_emergency_cost_yuan=z["daily_emergency_cost_yuan"],
        daily_total_cost_yuan=z["daily_total_cost_yuan"],
        solve_seconds=float(z["solve_seconds"][0]),
        execution_target_relaxations=int(z["execution_target_relaxations"][0]),
    )


def merge_emergency_blocks(
    values: np.ndarray, headers: Sequence[str]
) -> list[dict[str, Any]]:
    active = np.flatnonzero(np.asarray(values) > TOL)
    if active.size == 0:
        return []
    groups: list[list[int]] = [[int(active[0])]]
    for item in active[1:]:
        value = int(item)
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    output: list[dict[str, Any]] = []
    for group in groups:
        start = str(headers[group[0]]).split("-", 1)[0]
        end = str(headers[group[-1]]).split("-", 1)[1]
        output.append(
            {
                "时间段": f"{start}-{end}",
                "购电量_kWh": float(np.sum(values[group])),
                "起始索引": group[0],
                "终止索引": group[-1],
            }
        )
    return output


def _workbook_formats(workbook: xlsxwriter.Workbook) -> dict[str, Any]:
    return {
        "header": workbook.add_format(
            {
                "bold": True,
                "font_name": "Microsoft YaHei",
                "bg_color": "#D9EAF7",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "text": workbook.add_format(
            {"font_name": "Microsoft YaHei", "border": 1, "align": "center"}
        ),
        "number": workbook.add_format(
            {"font_name": "Microsoft YaHei", "border": 1, "num_format": "0.000"}
        ),
        "date": workbook.add_format(
            {
                "font_name": "Microsoft YaHei",
                "border": 1,
                "num_format": "yyyy/m/d",
                "align": "center",
            }
        ),
    }


def write_rolling_workbook(
    path: Path,
    data: DataBundle,
    result: RollingResult,
    *,
    include_adjustment_sheet: bool,
) -> None:
    """写问题三/问题4-3工作簿；字段顺序与附件5一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(path)
    fmt = _workbook_formats(workbook)
    headers = data.base.result_interval_headers

    def purchase_sheet(name: str, values: np.ndarray, adjusted: bool) -> None:
        sheet = workbook.add_worksheet(name)
        sheet.freeze_panes(1, 1)
        sheet.write_row(0, 0, ["日期\\时间", *headers, "全天购电量", "全天购电费"], fmt["header"])
        sheet.set_column(0, 0, 12)
        sheet.set_column(1, SLOTS, 13)
        sheet.set_column(SLOTS + 1, SLOTS + 2, 15)
        for row, day in enumerate(result.dates, start=1):
            sheet.write_datetime(row, 0, day.to_pydatetime(), fmt["date"])
            sheet.write_row(row, 1, [float(v) for v in values[row - 1]], fmt["number"])
            sheet.write_number(row, SLOTS + 1, float(values[row - 1].sum()), fmt["number"])
            cost = (
                result.daily_total_cost_yuan[row - 1]
                if adjusted
                else result.daily_plan_cost_yuan[row - 1]
            )
            sheet.write_number(row, SLOTS + 2, float(cost), fmt["number"])

    purchase_sheet("计划购电量", result.initial_plan_kwh, adjusted=False)
    if include_adjustment_sheet:
        purchase_sheet("调整购电量", result.final_purchase_kwh, adjusted=True)

    storage = workbook.add_worksheet("充放电量")
    storage.write_row(0, 0, ["日期", "时间段", "充电量", "放电量", "时刻", "储电量"], fmt["header"])
    storage.set_column(0, 0, 12)
    storage.set_column(1, 1, 16)
    storage.set_column(2, 5, 14)
    row = 1
    for day_index, day in enumerate(result.dates):
        first = row
        for block, label in enumerate(FOUR_HOUR_BLOCKS):
            lo, hi = block * 24, (block + 1) * 24
            storage.write(row, 1, label, fmt["text"])
            storage.write_number(row, 2, float(result.charge_kwh[day_index, lo:hi].sum()), fmt["number"])
            storage.write_number(row, 3, float(result.discharge_kwh[day_index, lo:hi].sum()), fmt["number"])
            if block == 0:
                storage.write(row, 4, "0:00", fmt["text"])
                storage.write_number(row, 5, float(result.soc_start_kwh[day_index]), fmt["number"])
            elif block == 1:
                storage.write(row, 4, "24:00", fmt["text"])
                storage.write_number(row, 5, float(result.soc_end_kwh[day_index, -1]), fmt["number"])
            row += 1
        storage.merge_range(first, 0, row - 1, 0, day.to_pydatetime(), fmt["date"])

    emergency_sheet = workbook.add_worksheet("紧急购电量")
    emergency_sheet.write_row(0, 0, ["日期", "购电时间段", "购电量"], fmt["header"])
    emergency_sheet.set_column(0, 0, 12)
    emergency_sheet.set_column(1, 1, 22)
    emergency_sheet.set_column(2, 2, 15)
    row = 1
    for day_index, day in enumerate(result.dates):
        blocks = merge_emergency_blocks(result.emergency_kwh[day_index], headers)
        if not blocks:
            blocks = [{"时间段": "无", "购电量_kWh": 0.0}]
        first = row
        for block in blocks:
            emergency_sheet.write(row, 1, block["时间段"], fmt["text"])
            emergency_sheet.write_number(row, 2, float(block["购电量_kWh"]), fmt["number"])
            row += 1
        if row - first == 1:
            emergency_sheet.write_datetime(first, 0, day.to_pydatetime(), fmt["date"])
        else:
            emergency_sheet.merge_range(first, 0, row - 1, 0, day.to_pydatetime(), fmt["date"])
    workbook.close()


def selected_date_rows(
    data: DataBundle, result: RollingResult, *, variable_price: bool
) -> list[dict[str, Any]]:
    header_to_index = {
        value: index for index, value in enumerate(data.base.result_interval_headers)
    }
    output: list[dict[str, Any]] = []
    for selected in KEY_DATES:
        where = np.flatnonzero(result.dates == selected)
        if where.size == 0:
            continue
        index = int(where[0])
        global_day = int(np.flatnonzero(data.base.dates == selected)[0])
        price = data.price_actual[global_day] if variable_price else data.fixed_price
        output.append(
            {
                "日期": selected.strftime("%Y-%m-%d"),
                "指定时段初始计划_kWh": {
                    label: float(result.initial_plan_kwh[index, header_to_index[label]])
                    for label in KEY_INTERVALS
                },
                "指定时段最终购电_kWh": {
                    label: float(result.final_purchase_kwh[index, header_to_index[label]])
                    for label in KEY_INTERVALS
                },
                "全天计划购电量_kWh": float(result.initial_plan_kwh[index].sum()),
                "最终购电量_kWh": float(result.final_purchase_kwh[index].sum()),
                "计划购电费_元": float(np.dot(price, result.initial_plan_kwh[index])),
                "调整费用净额_元": float(result.daily_adjustment_cost_yuan[index]),
                "紧急购电量_kWh": float(result.emergency_kwh[index].sum()),
                "紧急购电费_元": float(result.daily_emergency_cost_yuan[index]),
                "总购电费_元": float(result.daily_total_cost_yuan[index]),
                "四小时储能": [
                    {
                        "时间段": label,
                        "充电量_kWh": float(result.charge_kwh[index, block * 24 : (block + 1) * 24].sum()),
                        "放电量_kWh": float(result.discharge_kwh[index, block * 24 : (block + 1) * 24].sum()),
                    }
                    for block, label in enumerate(FOUR_HOUR_BLOCKS)
                ],
                "0点储电量_kWh": float(result.soc_start_kwh[index]),
                "24点储电量_kWh": float(result.soc_end_kwh[index, -1]),
                "紧急购电区段": merge_emergency_blocks(
                    result.emergency_kwh[index], data.base.result_interval_headers
                ),
            }
        )
    return output


def plot_policy_comparison(
    rows: Sequence[dict[str, Any]], path: Path, title: str
) -> None:
    names = [str(row["方案"]) for row in rows]
    totals = [float(row["总购电费_元"]) / 10000.0 for row in rows]
    emergency = [float(row["紧急购电费_元"]) / 10000.0 for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    bars = ax.bar(x, totals, color="#2878B5", alpha=0.88, label="总购电费")
    ax.scatter(x, emergency, color="#D9485F", s=55, zorder=3, label="紧急购电费")
    ax.set_xticks(x, names)
    ax.set_ylabel("费用/万元")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bar, value in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_selected_day(
    data: DataBundle,
    result: RollingResult,
    selected: pd.Timestamp,
    path: Path,
    *,
    variable_price: bool,
) -> None:
    local = int(np.flatnonzero(result.dates == selected)[0])
    global_day = int(np.flatnonzero(data.base.dates == selected)[0])
    hours = np.arange(SLOTS) / 6.0 + 1.0 / 6.0
    price = data.price_actual[global_day] if variable_price else data.fixed_price
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=180, sharex=True)
    axes[0].plot(hours, data.base.load_kw[global_day], label="实际负荷", color="#F28E2B")
    axes[0].plot(hours, data.base.pv_kw[global_day], label="实际光伏", color="#2CA25F")
    axes[0].set_ylabel("功率/kW")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].plot(hours, result.initial_plan_kwh[local], label="0:00初始计划", color="#6A51A3")
    axes[1].plot(hours, result.final_purchase_kwh[local], label="最终有效购电", color="#2878B5")
    axes[1].bar(hours, result.emergency_kwh[local], width=0.12, label="紧急购电", color="#D9485F", alpha=0.6)
    axes[1].set_ylabel("电量/kWh")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.2)
    axes[2].plot(hours, result.soc_end_kwh[local], label="储电量", color="#3B8D5A")
    price_axis = axes[2].twinx()
    price_axis.plot(hours, price, label="交易电价", color="#D17C00", alpha=0.75)
    axes[2].set_ylabel("储电量/kWh")
    price_axis.set_ylabel("电价/(元/kWh)")
    axes[2].set_xlabel("时刻")
    axes[2].grid(alpha=0.2)
    axes[2].set_xlim(0, 24)
    axes[2].set_xticks(np.arange(0, 25, 2))
    fig.suptitle(f"{selected:%Y-%m-%d} 微网购电与储能运行")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
