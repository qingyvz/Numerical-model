#!/usr/bin/env python3
"""求解 C 题第三问并生成 result3.xlsx 及可复核中间结果。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = ROOT / "result" / "3" / "tmp"
DEPS_DIR = TMP_DIR / "deps"
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))

import pulp  # type: ignore  # noqa: E402
import xlsxwriter  # type: ignore  # noqa: E402

from analyze_question3_data import (  # noqa: E402
    endpoint_label,
    load_inputs,
    mean,
    quantile,
)


DT_HOURS = 1.0 / 6.0
ETA_CHARGE = 0.90
ETA_DISCHARGE = 0.90
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_TARGET = 6000.0
POWER_LIMIT_KW = 5000.0
ENERGY_LIMIT_KWH = POWER_LIMIT_KW * DT_HOURS
EMERGENCY_MULTIPLIER = 5.0
UPWARD_MULTIPLIER = 1.5
DOWNWARD_REFUND_MULTIPLIER = 0.5
CYCLING_COST = 1.0e-5
HISTORY_WINDOW_DAYS = 28
SCENARIO_QUANTILES = (0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95)
ISSUE_HOURS = (0, 6, 12, 18)
STAGE_START_INDEX = {0: 0, 6: 36, 12: 72, 18: 108}
STAGE_EXECUTION_END = {0: 36, 6: 72, 12: 108, 18: 144}
OUTPUT_START = date(2025, 2, 1)
OUTPUT_END = date(2025, 12, 31)
WARMUP_DAY = date(2025, 1, 31)
SELECTED_DAYS = (
    date(2025, 3, 20),
    date(2025, 6, 21),
    date(2025, 9, 23),
    date(2025, 12, 21),
)


def date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_clock(total_minutes: int, next_day_suffix: bool = False) -> str:
    normalized = total_minutes % 1440
    hour = normalized // 60
    minute = normalized % 60
    value = f"{hour}:{minute:02d}"
    if next_day_suffix:
        value += "+1"
    return value


def interval_label(index: int) -> str:
    start = (index + 1) * 10
    end = start + 10
    start_next = start > 1440
    end_next = end >= 1440
    return (
        f"{format_clock(start, start_next)}-"
        f"{format_clock(end, end_next)}"
    )


INTERVAL_LABELS = [interval_label(index) for index in range(144)]


def actual_at_issue(pv: dict[date, list[float]], day: date, issue_hour: int) -> float:
    if issue_hour > 0:
        return pv[day][issue_hour * 6 - 1]
    previous_day = day - timedelta(days=1)
    if previous_day in pv:
        return pv[previous_day][-1]
    return 0.0


def pv_forecast_curve(
    pv: dict[date, list[float]],
    forecasts: dict[tuple[date, int], list[float]],
    day: date,
    issue_hour: int,
) -> list[float]:
    """将未来整点功率预报线性插值到附件的 10 分钟起始时刻。"""
    start_index = STAGE_START_INDEX[issue_hour]
    anchor = actual_at_issue(pv, day, issue_hour)
    hourly = forecasts[(day, issue_hour)]
    values: list[float] = []
    for index in range(start_index, 144):
        target_minutes = (index + 1) * 10
        lead_hours = (target_minutes - issue_hour * 60) / 60.0
        lower = math.floor(lead_hours)
        upper = math.ceil(lead_hours)
        if lower == upper:
            value = hourly[lower - 1]
        else:
            lower_value = anchor if lower == 0 else hourly[lower - 1]
            upper_value = hourly[upper - 1]
            fraction = lead_hours - lower
            value = lower_value * (1.0 - fraction) + upper_value * fraction
        values.append(max(0.0, float(value)))
    return values


def load_forecast_curve(
    load: dict[date, list[float]],
    exemplar_load: Sequence[float],
    day: date,
    issue_hour: int,
) -> tuple[list[float], float]:
    """同星期前一周曲线，并在日内发布时刻用已观测负荷作比例校正。"""
    start_index = STAGE_START_INDEX[issue_hour]
    reference_day = day - timedelta(days=7)
    reference = load.get(reference_day, list(exemplar_load))
    scale = 1.0
    if issue_hour > 0 and reference_day in load:
        observed_end = issue_hour * 6
        denominator = sum(reference[:observed_end])
        if denominator > 1.0e-9:
            scale = sum(load[day][:observed_end]) / denominator
            scale = min(1.10, max(0.90, scale))
    return [max(0.0, value * scale) for value in reference[start_index:144]], scale


def build_residual_curves(data: dict[str, Any]) -> dict[tuple[date, int], list[float]]:
    load: dict[date, list[float]] = data["load"]
    pv: dict[date, list[float]] = data["pv"]
    forecasts: dict[tuple[date, int], list[float]] = data["forecasts"]
    exemplar_load: list[float] = data["exemplar_load"]
    curves: dict[tuple[date, int], list[float]] = {}
    for day in sorted(load):
        if day - timedelta(days=7) not in load:
            continue
        for issue_hour in ISSUE_HOURS:
            start_index = STAGE_START_INDEX[issue_hour]
            load_prediction, _ = load_forecast_curve(
                load, exemplar_load, day, issue_hour
            )
            pv_prediction = pv_forecast_curve(pv, forecasts, day, issue_hour)
            residuals = []
            for local_index, global_index in enumerate(range(start_index, 144)):
                actual_net = (load[day][global_index] - pv[day][global_index]) * DT_HOURS
                predicted_net = (
                    load_prediction[local_index] - pv_prediction[local_index]
                ) * DT_HOURS
                residuals.append(actual_net - predicted_net)
            curves[(day, issue_hour)] = residuals
    return curves


def residual_scenarios(
    residual_curves: dict[tuple[date, int], list[float]],
    day: date,
    issue_hour: int,
) -> list[list[float]]:
    start_index = STAGE_START_INDEX[issue_hour]
    eligible_days = sorted(
        curve_day
        for curve_day, curve_issue in residual_curves
        if curve_issue == issue_hour and curve_day < day
    )[-HISTORY_WINDOW_DAYS:]
    length = 144 - start_index
    if not eligible_days:
        return [[0.0] * length for _ in SCENARIO_QUANTILES]
    scenarios: list[list[float]] = [[] for _ in SCENARIO_QUANTILES]
    for local_index in range(length):
        sample = [
            residual_curves[(history_day, issue_hour)][local_index]
            for history_day in eligible_days
        ]
        for scenario_index, probability in enumerate(SCENARIO_QUANTILES):
            scenarios[scenario_index].append(quantile(sample, probability))
    return scenarios


def clean_value(value: float | None, tolerance: float = 1.0e-7) -> float:
    if value is None:
        raise RuntimeError("求解器返回空变量值")
    if abs(value) < tolerance:
        return 0.0
    return float(value)


def solve_stage(
    *,
    day: date,
    issue_hour: int,
    initial_soc: float,
    price: Sequence[float],
    load_prediction_kw: Sequence[float],
    pv_prediction_kw: Sequence[float],
    residual_scenario_kwh: Sequence[Sequence[float]],
    planned_purchase_kwh: Sequence[float] | None,
    solver: pulp.LpSolver,
) -> dict[str, Any]:
    start_index = STAGE_START_INDEX[issue_hour]
    global_indices = list(range(start_index, 144))
    count = len(global_indices)
    if len(load_prediction_kw) != count or len(pv_prediction_kw) != count:
        raise ValueError("预报曲线长度与阶段长度不一致")
    if any(len(values) != count for values in residual_scenario_kwh):
        raise ValueError("残差情景长度与阶段长度不一致")

    model = pulp.LpProblem(f"Q3_{day.isoformat()}_{issue_hour:02d}", pulp.LpMinimize)
    purchase = [pulp.LpVariable(f"q_{i}", lowBound=0.0) for i in range(count)]
    charge = [
        pulp.LpVariable(f"c_{i}", lowBound=0.0, upBound=ENERGY_LIMIT_KWH)
        for i in range(count)
    ]
    discharge = [
        pulp.LpVariable(f"d_{i}", lowBound=0.0, upBound=ENERGY_LIMIT_KWH)
        for i in range(count)
    ]
    soc = [
        pulp.LpVariable(f"s_{i}", lowBound=SOC_MIN, upBound=SOC_MAX)
        for i in range(count)
    ]
    emergency = [
        [pulp.LpVariable(f"u_{scenario}_{i}", lowBound=0.0) for i in range(count)]
        for scenario in range(len(residual_scenario_kwh))
    ]

    upward: list[pulp.LpVariable] = []
    downward: list[pulp.LpVariable] = []
    objective_terms: list[Any] = []
    if planned_purchase_kwh is None:
        for local_index, global_index in enumerate(global_indices):
            objective_terms.append(price[global_index] * purchase[local_index])
    else:
        if len(planned_purchase_kwh) != 144:
            raise ValueError("计划购电向量必须包含 144 个区间")
        upward = [pulp.LpVariable(f"up_{i}", lowBound=0.0) for i in range(count)]
        downward = [
            pulp.LpVariable(
                f"down_{i}",
                lowBound=0.0,
                upBound=max(0.0, planned_purchase_kwh[global_index]),
            )
            for i, global_index in enumerate(global_indices)
        ]
        for local_index, global_index in enumerate(global_indices):
            model += (
                purchase[local_index]
                == planned_purchase_kwh[global_index]
                + upward[local_index]
                - downward[local_index]
            )
            objective_terms.append(
                price[global_index]
                * (
                    UPWARD_MULTIPLIER * upward[local_index]
                    - DOWNWARD_REFUND_MULTIPLIER * downward[local_index]
                )
            )

    scenario_weight = 1.0 / len(residual_scenario_kwh)
    for local_index, global_index in enumerate(global_indices):
        previous_soc: Any = initial_soc if local_index == 0 else soc[local_index - 1]
        model += (
            soc[local_index]
            == previous_soc
            + ETA_CHARGE * charge[local_index]
            - discharge[local_index] / ETA_DISCHARGE
        )
        predicted_net_kwh = (
            load_prediction_kw[local_index] - pv_prediction_kw[local_index]
        ) * DT_HOURS
        for scenario_index, residuals in enumerate(residual_scenario_kwh):
            model += (
                purchase[local_index]
                + discharge[local_index]
                - charge[local_index]
                + emergency[scenario_index][local_index]
                >= predicted_net_kwh + residuals[local_index]
            )
            objective_terms.append(
                EMERGENCY_MULTIPLIER
                * price[global_index]
                * scenario_weight
                * emergency[scenario_index][local_index]
            )
        objective_terms.append(CYCLING_COST * (charge[local_index] + discharge[local_index]))

    model += soc[-1] == SOC_TARGET
    model += pulp.lpSum(objective_terms)
    status_code = model.solve(solver)
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        raise RuntimeError(f"{day} {issue_hour:02d}:00 阶段求解状态为 {status}")

    result = {
        "day": day.isoformat(),
        "issue_hour": issue_hour,
        "start_index": start_index,
        "status": status,
        "objective_without_constants": float(pulp.value(model.objective)),
        "purchase_kwh": [clean_value(variable.value()) for variable in purchase],
        "charge_kwh": [clean_value(variable.value()) for variable in charge],
        "discharge_kwh": [clean_value(variable.value()) for variable in discharge],
        "soc_kwh": [clean_value(variable.value()) for variable in soc],
        "predicted_emergency_mean_kwh": [
            mean(
                [
                    clean_value(emergency[scenario][local_index].value())
                    for scenario in range(len(residual_scenario_kwh))
                ]
            )
            for local_index in range(count)
        ],
    }
    return result


def expand_stage(stage: dict[str, Any], key: str) -> list[float]:
    start_index = int(stage["start_index"])
    return [math.nan] * start_index + [float(value) for value in stage[key]]


def compose_schedule(
    stage0: dict[str, Any],
    stage6: dict[str, Any],
    stage12: dict[str, Any],
    stage18: dict[str, Any],
    last_issue: int,
) -> dict[str, list[float]]:
    stages = {0: stage0, 6: stage6, 12: stage12, 18: stage18}
    included = [issue for issue in ISSUE_HOURS if issue <= last_issue]
    output: dict[str, list[float]] = {}
    for key in ("purchase_kwh", "charge_kwh", "discharge_kwh"):
        arrays = {issue: expand_stage(stages[issue], key) for issue in included}
        values: list[float] = []
        for index in range(144):
            source = max(issue for issue in included if STAGE_START_INDEX[issue] <= index)
            values.append(arrays[source][index])
        output[key] = values
    return output


def actual_emergency(
    day: date,
    schedule: dict[str, list[float]],
    load: dict[date, list[float]],
    pv: dict[date, list[float]],
) -> list[float]:
    values = []
    for index in range(144):
        net_kwh = (load[day][index] - pv[day][index]) * DT_HOURS
        shortage = (
            net_kwh
            + schedule["charge_kwh"][index]
            - schedule["discharge_kwh"][index]
            - schedule["purchase_kwh"][index]
        )
        values.append(max(0.0, shortage))
    return values


def schedule_soc(schedule: dict[str, list[float]], initial_soc: float = SOC_TARGET) -> list[float]:
    state = initial_soc
    output = []
    for charge, discharge in zip(schedule["charge_kwh"], schedule["discharge_kwh"]):
        state += ETA_CHARGE * charge - discharge / ETA_DISCHARGE
        output.append(state)
    return output


def settlement_costs(
    planned: dict[str, list[float]],
    final_schedule: dict[str, list[float]],
    emergency: Sequence[float],
    price: Sequence[float],
) -> dict[str, float]:
    plan_cost = sum(
        price[index] * planned["purchase_kwh"][index] for index in range(144)
    )
    upward_kwh = sum(
        max(final_schedule["purchase_kwh"][index] - planned["purchase_kwh"][index], 0.0)
        for index in range(144)
    )
    downward_kwh = sum(
        max(planned["purchase_kwh"][index] - final_schedule["purchase_kwh"][index], 0.0)
        for index in range(144)
    )
    upward_cost = sum(
        UPWARD_MULTIPLIER
        * price[index]
        * max(final_schedule["purchase_kwh"][index] - planned["purchase_kwh"][index], 0.0)
        for index in range(144)
    )
    downward_refund = sum(
        DOWNWARD_REFUND_MULTIPLIER
        * price[index]
        * max(planned["purchase_kwh"][index] - final_schedule["purchase_kwh"][index], 0.0)
        for index in range(144)
    )
    emergency_cost = sum(
        EMERGENCY_MULTIPLIER * price[index] * emergency[index]
        for index in range(144)
    )
    total = plan_cost + upward_cost - downward_refund + emergency_cost
    return {
        "planned_purchase_kwh": sum(planned["purchase_kwh"]),
        "adjusted_purchase_kwh": sum(final_schedule["purchase_kwh"]),
        "upward_adjustment_kwh": upward_kwh,
        "downward_adjustment_kwh": downward_kwh,
        "emergency_purchase_kwh": sum(emergency),
        "plan_cost_yuan": plan_cost,
        "upward_adjustment_cost_yuan": upward_cost,
        "downward_adjustment_refund_yuan": downward_refund,
        "emergency_cost_yuan": emergency_cost,
        "total_cost_yuan": total,
    }


def emergency_periods(values: Sequence[float], tolerance: float = 1.0e-4) -> list[dict[str, Any]]:
    periods: list[dict[str, Any]] = []
    start: int | None = None
    for index in range(145):
        positive = index < 144 and values[index] > tolerance
        if positive and start is None:
            start = index
        elif not positive and start is not None:
            end = index - 1
            start_minutes = (start + 1) * 10
            end_minutes = (end + 2) * 10
            label = (
                f"{format_clock(start_minutes, start_minutes > 1440)}-"
                f"{format_clock(end_minutes, end_minutes >= 1440)}"
            )
            periods.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "period": label,
                    "energy_kwh": sum(values[start : end + 1]),
                }
            )
            start = None
    return periods


def build_solver() -> pulp.LpSolver:
    solver = pulp.HiGHS(
        msg=False,
        mip=False,
        threads=1,
        presolve="on",
        primal_feasibility_tolerance=1.0e-8,
        dual_feasibility_tolerance=1.0e-8,
    )
    return solver


def solve_days(data: dict[str, Any], days: Sequence[date]) -> dict[date, dict[str, Any]]:
    price: list[float] = data["price"]
    load: dict[date, list[float]] = data["load"]
    pv: dict[date, list[float]] = data["pv"]
    forecasts: dict[tuple[date, int], list[float]] = data["forecasts"]
    exemplar_load: list[float] = data["exemplar_load"]
    residual_curves = build_residual_curves(data)
    solver = build_solver()
    output: dict[date, dict[str, Any]] = {}

    for day_index, day in enumerate(days, start=1):
        stages: dict[int, dict[str, Any]] = {}
        stage_scales: dict[int, float] = {}
        initial_soc = SOC_TARGET
        planned_vector: list[float] | None = None
        for issue_hour in ISSUE_HOURS:
            load_prediction, load_scale = load_forecast_curve(
                load, exemplar_load, day, issue_hour
            )
            pv_prediction = pv_forecast_curve(pv, forecasts, day, issue_hour)
            scenarios = residual_scenarios(residual_curves, day, issue_hour)
            stage = solve_stage(
                day=day,
                issue_hour=issue_hour,
                initial_soc=initial_soc,
                price=price,
                load_prediction_kw=load_prediction,
                pv_prediction_kw=pv_prediction,
                residual_scenario_kwh=scenarios,
                planned_purchase_kwh=planned_vector,
                solver=solver,
            )
            stages[issue_hour] = stage
            stage_scales[issue_hour] = load_scale
            if issue_hour == 0:
                planned_vector = expand_stage(stage, "purchase_kwh")
            end_index = STAGE_EXECUTION_END[issue_hour] - 1
            if issue_hour < 18:
                stage_soc = expand_stage(stage, "soc_kwh")
                initial_soc = stage_soc[end_index]

        plan_schedule = compose_schedule(
            stages[0], stages[6], stages[12], stages[18], last_issue=0
        )
        combinations = {
            "00_only": plan_schedule,
            "00_06": compose_schedule(
                stages[0], stages[6], stages[12], stages[18], last_issue=6
            ),
            "00_06_12": compose_schedule(
                stages[0], stages[6], stages[12], stages[18], last_issue=12
            ),
            "00_06_12_18": compose_schedule(
                stages[0], stages[6], stages[12], stages[18], last_issue=18
            ),
        }
        combination_results: dict[str, Any] = {}
        for name, schedule in combinations.items():
            emergency = actual_emergency(day, schedule, load, pv)
            soc = schedule_soc(schedule)
            if min(soc) < SOC_MIN - 1.0e-3 or max(soc) > SOC_MAX + 1.0e-3:
                raise RuntimeError(
                    f"{day} {name} 的 SOC 越界：min={min(soc):.8f}, max={max(soc):.8f}"
                )
            if abs(soc[-1] - SOC_TARGET) > 1.0e-3:
                raise RuntimeError(f"{day} {name} 的终端 SOC 不为 {SOC_TARGET}")
            combination_results[name] = {
                "schedule": schedule,
                "soc_kwh": soc,
                "emergency_kwh": emergency,
                "costs": settlement_costs(plan_schedule, schedule, emergency, price),
            }

        output[day] = {
            "stages": stages,
            "load_scales": stage_scales,
            "combinations": combination_results,
        }
        print(
            f"[{day_index:03d}/{len(days):03d}] {day.isoformat()} "
            f"费用={combination_results['00_06_12_18']['costs']['total_cost_yuan']:.2f} 元",
            flush=True,
        )
    return output


def calendar_storage_summary(
    day: date,
    solutions: dict[date, dict[str, Any]],
    combination: str = "00_06_12_18",
) -> dict[str, Any]:
    current = solutions[day]["combinations"][combination]
    previous_day = day - timedelta(days=1)
    if previous_day in solutions:
        previous = solutions[previous_day]["combinations"][combination]
        first_charge = previous["schedule"]["charge_kwh"][143]
        first_discharge = previous["schedule"]["discharge_kwh"][143]
        soc_zero = previous["soc_kwh"][142]
    else:
        first_charge = 0.0
        first_discharge = 0.0
        soc_zero = SOC_TARGET
    charge_slots = [first_charge] + current["schedule"]["charge_kwh"][:143]
    discharge_slots = [first_discharge] + current["schedule"]["discharge_kwh"][:143]
    soc_24 = current["soc_kwh"][142]
    blocks = []
    for block_index, start_hour in enumerate((0, 4, 8, 12, 16, 20)):
        start = block_index * 24
        end = start + 24
        blocks.append(
            {
                "time_block": f"{start_hour}:00-{start_hour+4}:00",
                "charge_kwh": sum(charge_slots[start:end]),
                "discharge_kwh": sum(discharge_slots[start:end]),
            }
        )
    return {
        "date": day.isoformat(),
        "soc_0_kwh": soc_zero,
        "soc_24_kwh": soc_24,
        "blocks": blocks,
    }


def aggregate_results(
    solutions: dict[date, dict[str, Any]], output_days: Sequence[date]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    policy_names = ("00_only", "00_06", "00_06_12", "00_06_12_18")
    policy_totals: dict[str, dict[str, float]] = {}
    for name in policy_names:
        keys = list(
            solutions[output_days[0]]["combinations"][name]["costs"].keys()
        )
        policy_totals[name] = {
            key: sum(
                solutions[day]["combinations"][name]["costs"][key]
                for day in output_days
            )
            for key in keys
        }

    daily_rows = []
    emergency_rows = []
    for day in output_days:
        final = solutions[day]["combinations"]["00_06_12_18"]
        costs = final["costs"]
        storage = calendar_storage_summary(day, solutions)
        daily_rows.append(
            {
                "date": day.isoformat(),
                **costs,
                "charge_kwh": sum(block["charge_kwh"] for block in storage["blocks"]),
                "discharge_kwh": sum(
                    block["discharge_kwh"] for block in storage["blocks"]
                ),
                "soc_0_kwh": storage["soc_0_kwh"],
                "soc_24_kwh": storage["soc_24_kwh"],
                "load_scale_06": solutions[day]["load_scales"][6],
                "load_scale_12": solutions[day]["load_scales"][12],
                "load_scale_18": solutions[day]["load_scales"][18],
            }
        )
        periods = emergency_periods(final["emergency_kwh"])
        if not periods:
            emergency_rows.append(
                {"date": day.isoformat(), "period": "无", "energy_kwh": 0.0}
            )
        else:
            for period in periods:
                emergency_rows.append(
                    {
                        "date": day.isoformat(),
                        "period": period["period"],
                        "energy_kwh": period["energy_kwh"],
                    }
                )

    baseline_cost = policy_totals["00_only"]["total_cost_yuan"]
    comparison = []
    previous_cost: float | None = None
    for name in policy_names:
        values = policy_totals[name]
        comparison.append(
            {
                "combination": name,
                **values,
                "cost_change_vs_00_only_yuan": values["total_cost_yuan"]
                - baseline_cost,
                "cost_change_vs_00_only_pct": 100
                * (values["total_cost_yuan"] - baseline_cost)
                / baseline_cost,
                "marginal_cost_change_yuan": (
                    0.0 if previous_cost is None else values["total_cost_yuan"] - previous_cost
                ),
            }
        )
        previous_cost = values["total_cost_yuan"]

    summary = {
        "parameters": {
            "time_step_hours": DT_HOURS,
            "charge_efficiency": ETA_CHARGE,
            "discharge_efficiency": ETA_DISCHARGE,
            "soc_min_kwh": SOC_MIN,
            "soc_max_kwh": SOC_MAX,
            "soc_terminal_kwh": SOC_TARGET,
            "power_limit_kw": POWER_LIMIT_KW,
            "history_window_days": HISTORY_WINDOW_DAYS,
            "scenario_quantiles": list(SCENARIO_QUANTILES),
            "emergency_price_multiplier": EMERGENCY_MULTIPLIER,
            "upward_adjustment_multiplier": UPWARD_MULTIPLIER,
            "downward_refund_multiplier": DOWNWARD_REFUND_MULTIPLIER,
        },
        "output_start": output_days[0].isoformat(),
        "output_end": output_days[-1].isoformat(),
        "days": len(output_days),
        "policy_comparison": comparison,
        "days_with_emergency": sum(
            row["emergency_purchase_kwh"] > 1.0e-4 for row in daily_rows
        ),
        "emergency_period_count": sum(row["period"] != "无" for row in emergency_rows),
        "soc_observed_min_kwh": min(
            min(solutions[day]["combinations"]["00_06_12_18"]["soc_kwh"])
            for day in output_days
        ),
        "soc_observed_max_kwh": max(
            max(solutions[day]["combinations"]["00_06_12_18"]["soc_kwh"])
            for day in output_days
        ),
    }
    return summary, daily_rows, emergency_rows


def selected_details(
    solutions: dict[date, dict[str, Any]],
    selected_days: Iterable[date],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_interval_rows = []
    selected_storage_rows = []
    selected_emergency_rows = []
    target_start_minutes = (600, 720, 840, 960, 1080, 1200)
    for day in selected_days:
        plan = solutions[day]["combinations"]["00_only"]
        final = solutions[day]["combinations"]["00_06_12_18"]
        for start_minute in target_start_minutes:
            index = start_minute // 10 - 1
            selected_interval_rows.append(
                {
                    "date": day.isoformat(),
                    "period": interval_label(index),
                    "planned_purchase_kwh": plan["schedule"]["purchase_kwh"][index],
                    "adjusted_purchase_kwh": final["schedule"]["purchase_kwh"][index],
                    "emergency_purchase_kwh": final["emergency_kwh"][index],
                }
            )
        selected_interval_rows.append(
            {
                "date": day.isoformat(),
                "period": "全天",
                "planned_purchase_kwh": plan["costs"]["planned_purchase_kwh"],
                "adjusted_purchase_kwh": final["costs"]["adjusted_purchase_kwh"],
                "emergency_purchase_kwh": final["costs"]["emergency_purchase_kwh"],
            }
        )
        storage = calendar_storage_summary(day, solutions)
        for block in storage["blocks"]:
            selected_storage_rows.append(
                {
                    "date": day.isoformat(),
                    **block,
                    "soc_0_kwh": storage["soc_0_kwh"],
                    "soc_24_kwh": storage["soc_24_kwh"],
                }
            )
        periods = emergency_periods(final["emergency_kwh"])
        if not periods:
            selected_emergency_rows.append(
                {"date": day.isoformat(), "period": "无", "energy_kwh": 0.0}
            )
        else:
            for period in periods:
                selected_emergency_rows.append(
                    {
                        "date": day.isoformat(),
                        "period": period["period"],
                        "energy_kwh": period["energy_kwh"],
                    }
                )
    return selected_interval_rows, selected_storage_rows, selected_emergency_rows


def write_result_workbook(
    path: Path,
    solutions: dict[date, dict[str, Any]],
    output_days: Sequence[date],
    daily_rows: Sequence[dict[str, Any]],
) -> None:
    workbook = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    workbook.set_properties(
        {
            "title": "C题第三问微网购电策略",
            "subject": "计划购电、调整购电、储能与紧急购电结果",
            "author": "数学建模团队",
            "comments": "基于滚动随机线性规划生成",
        }
    )
    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "white",
            "bg_color": "#4472C4",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        }
    )
    date_format = workbook.add_format(
        {"num_format": "yyyy/m/d", "border": 1, "align": "center"}
    )
    number_format = workbook.add_format(
        {"num_format": "0.0000", "border": 1, "align": "right"}
    )
    money_format = workbook.add_format(
        {"num_format": "0.00", "border": 1, "align": "right"}
    )
    text_format = workbook.add_format({"border": 1, "align": "center"})
    zero_format = workbook.add_format(
        {"num_format": "0.0000", "border": 1, "align": "right", "font_color": "#777777"}
    )

    daily_lookup = {row["date"]: row for row in daily_rows}
    for sheet_name, combination, cost_kind in (
        ("计划购电量", "00_only", "plan"),
        ("调整购电量", "00_06_12_18", "total"),
    ):
        worksheet = workbook.add_worksheet(sheet_name)
        worksheet.freeze_panes(1, 1)
        worksheet.set_row(0, 42)
        worksheet.set_column(0, 0, 12)
        worksheet.set_column(1, 144, 12)
        worksheet.set_column(145, 146, 15)
        headers = ["日期\\时间", *INTERVAL_LABELS, "全天购电量", "全天购电费"]
        worksheet.write_row(0, 0, headers, header_format)
        worksheet.autofilter(0, 0, len(output_days), 146)
        for row_index, day in enumerate(output_days, start=1):
            result = solutions[day]["combinations"][combination]
            worksheet.write_datetime(
                row_index,
                0,
                datetime.combine(day, datetime.min.time()),
                date_format,
            )
            for column_index, value in enumerate(
                result["schedule"]["purchase_kwh"], start=1
            ):
                worksheet.write_number(
                    row_index,
                    column_index,
                    value,
                    zero_format if abs(value) < 5.0e-8 else number_format,
                )
            costs = result["costs"]
            worksheet.write_number(
                row_index, 145, costs["adjusted_purchase_kwh"], number_format
            )
            cost_value = (
                costs["plan_cost_yuan"]
                if cost_kind == "plan"
                else costs["total_cost_yuan"]
            )
            worksheet.write_number(row_index, 146, cost_value, money_format)
        if sheet_name == "调整购电量":
            worksheet.write_comment(
                0,
                146,
                "本列为最终总购电费：计划费用＋上调费用－下调返还＋紧急购电费用。",
            )

    storage_sheet = workbook.add_worksheet("充放电量")
    storage_sheet.freeze_panes(1, 0)
    storage_sheet.set_column(0, 0, 12)
    storage_sheet.set_column(1, 1, 15)
    storage_sheet.set_column(2, 5, 14)
    storage_sheet.write_row(
        0, 0, ["日期", "时间段", "充电量", "放电量", "时刻", "储电量"], header_format
    )
    row_index = 1
    for day in output_days:
        storage = calendar_storage_summary(day, solutions)
        first_row = row_index
        for block_index, block in enumerate(storage["blocks"]):
            storage_sheet.write(row_index, 1, block["time_block"], text_format)
            storage_sheet.write_number(row_index, 2, block["charge_kwh"], number_format)
            storage_sheet.write_number(row_index, 3, block["discharge_kwh"], number_format)
            if block_index == 0:
                storage_sheet.write(row_index, 4, "0:00", text_format)
                storage_sheet.write_number(row_index, 5, storage["soc_0_kwh"], number_format)
            elif block_index == 1:
                storage_sheet.write(row_index, 4, "24:00", text_format)
                storage_sheet.write_number(row_index, 5, storage["soc_24_kwh"], number_format)
            else:
                storage_sheet.write_blank(row_index, 4, None, text_format)
                storage_sheet.write_blank(row_index, 5, None, number_format)
            row_index += 1
        storage_sheet.merge_range(
            first_row,
            0,
            row_index - 1,
            0,
            datetime.combine(day, datetime.min.time()),
            date_format,
        )

    emergency_sheet = workbook.add_worksheet("紧急购电量")
    emergency_sheet.freeze_panes(1, 0)
    emergency_sheet.set_column(0, 0, 12)
    emergency_sheet.set_column(1, 1, 22)
    emergency_sheet.set_column(2, 2, 16)
    emergency_sheet.write_row(0, 0, ["日期", "购电时间段", "购电量"], header_format)
    row_index = 1
    for day in output_days:
        emergency = solutions[day]["combinations"]["00_06_12_18"]["emergency_kwh"]
        periods = emergency_periods(emergency)
        if not periods:
            periods = [{"period": "无", "energy_kwh": 0.0}]
        first_row = row_index
        for period in periods:
            emergency_sheet.write(row_index, 1, period["period"], text_format)
            emergency_sheet.write_number(row_index, 2, period["energy_kwh"], number_format)
            row_index += 1
        if row_index - first_row == 1:
            emergency_sheet.write_datetime(
                first_row,
                0,
                datetime.combine(day, datetime.min.time()),
                date_format,
            )
        else:
            emergency_sheet.merge_range(
                first_row,
                0,
                row_index - 1,
                0,
                datetime.combine(day, datetime.min.time()),
                date_format,
            )

    workbook.close()


def serialize_solutions(
    path: Path, solutions: dict[date, dict[str, Any]], output_days: Sequence[date]
) -> None:
    payload: dict[str, Any] = {}
    for day in output_days:
        day_result = solutions[day]
        payload[day.isoformat()] = {
            "load_scales": day_result["load_scales"],
            "combinations": day_result["combinations"],
        }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit-output-days",
        type=int,
        default=None,
        help="仅用于试算：限制从 2025-02-01 起的输出天数",
    )
    parser.add_argument(
        "--skip-workbook", action="store_true", help="试算时不生成最终工作簿"
    )
    args = parser.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    output_end = OUTPUT_END
    if args.limit_output_days is not None:
        output_end = OUTPUT_START + timedelta(days=args.limit_output_days - 1)
    output_days = date_range(OUTPUT_START, output_end)
    solve_days_list = [WARMUP_DAY, *output_days]
    solutions = solve_days(data, solve_days_list)
    summary, daily_rows, emergency_rows = aggregate_results(solutions, output_days)
    selected_days = [day for day in SELECTED_DAYS if day in solutions]
    selected_interval_rows, selected_storage_rows, selected_emergency_rows = selected_details(
        solutions, selected_days
    )

    write_csv(TMP_DIR / "question3_daily_solution.csv", daily_rows)
    write_csv(TMP_DIR / "question3_emergency_periods.csv", emergency_rows)
    write_csv(TMP_DIR / "question3_selected_intervals.csv", selected_interval_rows)
    write_csv(TMP_DIR / "question3_selected_storage.csv", selected_storage_rows)
    write_csv(TMP_DIR / "question3_selected_emergency.csv", selected_emergency_rows)
    write_csv(TMP_DIR / "question3_policy_comparison.csv", summary["policy_comparison"])
    (TMP_DIR / "question3_solution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    serialize_solutions(
        TMP_DIR / "question3_solution_arrays.json.gz", solutions, output_days
    )
    if not args.skip_workbook and output_end == OUTPUT_END:
        write_result_workbook(
            ROOT / "result" / "3" / "result3.xlsx",
            solutions,
            output_days,
            daily_rows,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
