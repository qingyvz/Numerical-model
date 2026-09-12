"""C题问题四：波动电价下重新计算问题二和问题三。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "program" / "_deps"))
sys.path.insert(0, str(ROOT / "program"))
_mpl_dir = ROOT / "result" / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import solution2_core as core  # noqa: E402


OUTPUT = ROOT / "result" / "4"
TMP = OUTPUT / "tmp"
Q42_METHODS = (
    ("fixed_mean", "固定平均日电价（可实施基线）", True),
    ("lag7", "前7日同刻电价（正式因果策略）", True),
    ("oracle", "完全预知当日电价（不可实施下界）", False),
)
Q43_COMBINATIONS = ((0,), (0, 6), (0, 6, 12), (0, 6, 12, 18))


def clean_actual_day(result: core.q2.DayModelResult) -> tuple[np.ndarray, ...]:
    charge = result.grid_charge_kwh[0] + result.pv_charge_kwh[0]
    discharge = result.discharge_kwh[0].copy()
    spill = result.grid_spill_kwh[0] + result.pv_spill_kwh[0]
    # 沿不改变SOC的方向消去数值退化循环。
    for slot in np.flatnonzero((charge > core.TOL) & (discharge > core.TOL)):
        removable = min(
            float(charge[slot]),
            float(discharge[slot]) / (core.ETA_C * core.ETA_D),
        )
        charge[slot] -= removable
        discharge[slot] -= core.ETA_C * core.ETA_D * removable
        spill[slot] += (1.0 - core.ETA_C * core.ETA_D) * removable
    charge = np.where(charge > core.TOL, charge, 0.0)
    discharge = np.where(discharge > core.TOL, discharge, 0.0)
    emergency = np.where(result.emergency_kwh[0] > core.TOL, result.emergency_kwh[0], 0.0)
    spill = np.where(spill > core.TOL, spill, 0.0)
    return charge, discharge, emergency, spill, result.soc_end_kwh[0].copy()


def q42_cache_path(method: str) -> Path:
    return TMP / f"question42_{method}.npz"


def run_q42_method(
    data: core.DataBundle,
    method: str,
    name: str,
    *,
    day_limit: int | None,
) -> core.RollingResult:
    global_days = np.arange(core.OUTPUT_START_INDEX, len(data.base.dates), dtype=int)
    if day_limit is not None:
        global_days = global_days[:day_limit]
    count = len(global_days)
    shape = (count, core.SLOTS)
    plan = np.zeros(shape)
    charge = np.zeros(shape)
    discharge = np.zeros(shape)
    emergency = np.zeros(shape)
    spill = np.zeros(shape)
    soc_end = np.zeros(shape)
    plan_cost = np.zeros(count)
    emergency_cost = np.zeros(count)
    solve_seconds = 0.0
    started = time.perf_counter()
    for local_day, global_day in enumerate(global_days):
        scenario_load, scenario_pv, probabilities, _ = core.q2.make_residual_scenarios(
            data.base, data.ridge, int(global_day), scenario_count=core.SCENARIO_COUNT
        )
        predicted_price = core.price_forecast(data, int(global_day), 0, method)
        planned = core.q2.solve_day_model(
            scenario_load,
            scenario_pv,
            predicted_price,
            probabilities,
            initial_soc_kwh=core.SOC_TARGET,
            terminal_soc_kwh=core.SOC_TARGET,
        )
        actual_price = data.price_actual[global_day]
        actual = core.q2.solve_day_model(
            data.base.load_kw[global_day],
            data.base.pv_kw[global_day],
            actual_price,
            np.array([1.0]),
            initial_soc_kwh=core.SOC_TARGET,
            terminal_soc_kwh=core.SOC_TARGET,
            fixed_plan_kwh=planned.planned_purchase_kwh,
        )
        solve_seconds += planned.solve_seconds + actual.solve_seconds
        plan[local_day] = planned.planned_purchase_kwh
        (
            charge[local_day],
            discharge[local_day],
            emergency[local_day],
            spill[local_day],
            soc_end[local_day],
        ) = clean_actual_day(actual)
        plan_cost[local_day] = float(np.dot(actual_price, plan[local_day]))
        emergency_cost[local_day] = float(
            np.dot(actual_price, core.EMERGENCY_MULTIPLIER * emergency[local_day])
        )
        if local_day == 0 or (local_day + 1) % 25 == 0 or local_day + 1 == count:
            print(
                f"问题4-2 {method}：{local_day+1}/{count} 天，"
                f"累计费用 {(plan_cost[:local_day+1]+emergency_cost[:local_day+1]).sum()/10000:.2f} 万元",
                flush=True,
            )
    return core.RollingResult(
        name=name,
        issues=(0,),
        dates=data.base.dates[global_days],
        initial_plan_kwh=plan,
        final_purchase_kwh=plan.copy(),
        charge_kwh=charge,
        discharge_kwh=discharge,
        emergency_kwh=emergency,
        spill_kwh=spill,
        soc_start_kwh=np.full(count, core.SOC_TARGET),
        soc_end_kwh=soc_end,
        upward_by_issue_kwh=np.zeros((count, len(core.ISSUES), core.SLOTS)),
        downward_by_issue_kwh=np.zeros((count, len(core.ISSUES), core.SLOTS)),
        daily_plan_cost_yuan=plan_cost,
        daily_adjustment_cost_yuan=np.zeros(count),
        daily_emergency_cost_yuan=emergency_cost,
        daily_total_cost_yuan=plan_cost + emergency_cost,
        solve_seconds=solve_seconds,
        execution_target_relaxations=0,
    )


def audit_result(data: core.DataBundle, result: core.RollingResult) -> dict[str, object]:
    global_indices = np.array(
        [int(np.flatnonzero(data.base.dates == value)[0]) for value in result.dates]
    )
    balance = (
        result.final_purchase_kwh
        + result.emergency_kwh
        + data.base.pv_kw[global_indices] * core.DT_HOURS
        + result.discharge_kwh
        - data.base.load_kw[global_indices] * core.DT_HOURS
        - result.charge_kwh
        - result.spill_kwh
    )
    previous = np.column_stack([result.soc_start_kwh, result.soc_end_kwh[:, :-1]])
    expected_soc = previous + core.ETA_C * result.charge_kwh - result.discharge_kwh / core.ETA_D
    actual_prices = data.price_actual[global_indices]
    plan_cost = np.sum(actual_prices * result.initial_plan_kwh, axis=1)
    adjustment_cost = np.sum(
        actual_prices[:, None, :]
        * (
            core.UP_MULTIPLIER * result.upward_by_issue_kwh
            - core.DOWN_REFUND_MULTIPLIER * result.downward_by_issue_kwh
        ),
        axis=(1, 2),
    )
    emergency_cost = np.sum(
        core.EMERGENCY_MULTIPLIER * actual_prices * result.emergency_kwh, axis=1
    )
    adjustment_identity = (
        result.initial_plan_kwh
        + result.upward_by_issue_kwh.sum(axis=1)
        - result.downward_by_issue_kwh.sum(axis=1)
    )
    checks = {
        "最大供需平衡残差_kWh": float(np.max(np.abs(balance))),
        "最大SOC递推残差_kWh": float(np.max(np.abs(expected_soc - result.soc_end_kwh))),
        "逐次调整恒等式误差_kWh": float(
            np.max(np.abs(adjustment_identity - result.final_purchase_kwh))
        ),
        "计划费用复算误差_元": float(np.max(np.abs(plan_cost - result.daily_plan_cost_yuan))),
        "调整费用复算误差_元": float(
            np.max(np.abs(adjustment_cost - result.daily_adjustment_cost_yuan))
        ),
        "紧急费用复算误差_元": float(
            np.max(np.abs(emergency_cost - result.daily_emergency_cost_yuan))
        ),
        "SOC最小值_kWh": float(result.soc_end_kwh.min()),
        "SOC最大值_kWh": float(result.soc_end_kwh.max()),
        "日末SOC最大误差_kWh": float(
            np.max(np.abs(result.soc_end_kwh[:, -1] - core.SOC_TARGET))
        ),
        "最大充电功率_kW": float(result.charge_kwh.max() / core.DT_HOURS),
        "最大放电功率_kW": float(result.discharge_kwh.max() / core.DT_HOURS),
        "同时充放电时段数": int(
            np.sum((result.charge_kwh > core.TOL) & (result.discharge_kwh > core.TOL))
        ),
        "供电中断次数": 0,
    }
    checks["校验通过"] = bool(
        checks["最大供需平衡残差_kWh"] < 1.0e-5
        and checks["最大SOC递推残差_kWh"] < 1.0e-5
        and checks["逐次调整恒等式误差_kWh"] < 1.0e-5
        and checks["计划费用复算误差_元"] < 1.0e-5
        and checks["调整费用复算误差_元"] < 1.0e-5
        and checks["紧急费用复算误差_元"] < 1.0e-5
        and checks["SOC最小值_kWh"] >= core.SOC_MIN - 1.0e-5
        and checks["SOC最大值_kWh"] <= core.SOC_MAX + 1.0e-5
        and checks["日末SOC最大误差_kWh"] < 1.0e-5
        and checks["同时充放电时段数"] == 0
    )
    return checks


def plot_price_data(data: core.DataBundle) -> None:
    timestamps = pd.date_range("2025-01-01 00:10", periods=365 * core.SLOTS, freq="10min")
    raw = data.price_actual.reshape(-1)
    daily_mean = data.price_actual.mean(axis=1)
    daily_min = data.price_actual.min(axis=1)
    daily_max = data.price_actual.max(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=180)
    axes[0].plot(timestamps, raw, color="#2878B5", linewidth=0.45)
    axes[0].set_title("附件4全年52,560个十分钟实时电价原始序列")
    axes[0].set_ylabel("电价/(元/kWh)")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    axes[0].xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[0].grid(which="major", alpha=0.25)
    axes[0].grid(which="minor", alpha=0.08)
    axes[1].fill_between(data.base.dates, daily_min, daily_max, color="#9ECAE1", alpha=0.6, label="日日峰谷区间")
    axes[1].plot(data.base.dates, daily_mean, color="#D9485F", linewidth=1.1, label="日均价")
    axes[1].set_ylabel("电价/(元/kWh)")
    axes[1].set_xlabel("日期（细网格为周）")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    axes[1].xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[1].grid(which="major", alpha=0.25)
    axes[1].grid(which="minor", alpha=0.08)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(TMP / "attachment4_full_price_series.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="仅调试前N个输出日")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    data = core.load_data()
    price_metrics = core.price_forecast_metrics(data)
    core.dataframe_to_csv(price_metrics, TMP / "price_forecast_metrics.csv")
    plot_price_data(data)

    expected_days = args.days or (len(data.base.dates) - core.OUTPUT_START_INDEX)
    q42_results: list[core.RollingResult] = []
    q42_rows: list[dict[str, object]] = []
    for method, name, eligible in Q42_METHODS:
        cache = q42_cache_path(method)
        if cache.exists() and not args.force:
            result = core.load_rolling_result(cache)
            if len(result.dates) != expected_days:
                result = run_q42_method(data, method, name, day_limit=args.days)
                core.save_rolling_result(cache, result)
        else:
            result = run_q42_method(data, method, name, day_limit=args.days)
            core.save_rolling_result(cache, result)
        metrics = core.rolling_metrics(result)
        metrics["预测方法"] = method
        metrics["可实施"] = eligible
        q42_results.append(result)
        q42_rows.append(metrics)
    q42_frame = pd.DataFrame(q42_rows)
    core.dataframe_to_csv(q42_frame, TMP / "question42_method_comparison.csv")
    core.write_json(TMP / "question42_method_comparison.json", q42_rows)
    core.plot_policy_comparison(q42_rows, TMP / "question42_method_comparison.png", "问题4-2电价信息方案比较")

    eligible_indices = [i for i, (_, _, eligible) in enumerate(Q42_METHODS) if eligible]
    q42_best_index = min(
        eligible_indices, key=lambda i: float(q42_rows[i]["总购电费_元"])
    )
    q42_best = q42_results[q42_best_index]
    core.write_rolling_workbook(
        OUTPUT / "result4-2.xlsx", data, q42_best, include_adjustment_sheet=False
    )
    q42_selected = core.selected_date_rows(data, q42_best, variable_price=True)
    q42_audit = audit_result(data, q42_best)
    core.write_json(TMP / "question42_selected_dates.json", q42_selected)
    core.write_json(TMP / "question42_verification.json", q42_audit)
    if not bool(q42_audit["校验通过"]):
        raise RuntimeError(f"问题4-2校验失败：{q42_audit}")

    q43_results: list[core.RollingResult] = []
    q43_rows: list[dict[str, object]] = []
    for issues in Q43_COMBINATIONS:
        cache = TMP / ("question43_policy_" + "_".join(f"{x:02d}" for x in issues) + ".npz")
        if cache.exists() and not args.force:
            result = core.load_rolling_result(cache)
            if len(result.dates) != expected_days:
                result = core.run_rolling_policy(
                    data,
                    issues,
                    variable_price=True,
                    price_method="causal_selected",
                    day_limit=args.days,
                    progress_prefix="问题4-3",
                )
                core.save_rolling_result(cache, result)
        else:
            result = core.run_rolling_policy(
                data,
                issues,
                variable_price=True,
                price_method="causal_selected",
                day_limit=args.days,
                progress_prefix="问题4-3",
            )
            core.save_rolling_result(cache, result)
        q43_results.append(result)
        q43_rows.append(core.rolling_metrics(result))
    base_cost = float(q43_rows[0]["总购电费_元"])
    previous_cost = base_cost
    for row in q43_rows:
        cost = float(row["总购电费_元"])
        row["相对仅0点节省_元"] = base_cost - cost
        row["相对仅0点节省_pct"] = (base_cost - cost) / base_cost * 100.0
        row["本次新增时刻边际节省_元"] = previous_cost - cost
        previous_cost = cost
    q43_frame = pd.DataFrame(q43_rows)
    core.dataframe_to_csv(q43_frame, TMP / "question43_policy_comparison.csv")
    core.write_json(TMP / "question43_policy_comparison.json", q43_rows)
    core.plot_policy_comparison(q43_rows, TMP / "question43_policy_comparison.png", "问题4-3不同滚动时刻的实际费用")
    q43_best_index = int(np.argmin([float(row["总购电费_元"]) for row in q43_rows]))
    q43_best = q43_results[q43_best_index]
    core.write_rolling_workbook(
        OUTPUT / "result4-3.xlsx", data, q43_best, include_adjustment_sheet=True
    )
    q43_selected = core.selected_date_rows(data, q43_best, variable_price=True)
    q43_audit = audit_result(data, q43_best)
    core.write_json(TMP / "question43_selected_dates.json", q43_selected)
    core.write_json(TMP / "question43_verification.json", q43_audit)
    if not bool(q43_audit["校验通过"]):
        raise RuntimeError(f"问题4-3校验失败：{q43_audit}")

    for selected_day in core.KEY_DATES:
        if selected_day in q43_best.dates:
            core.plot_selected_day(
                data,
                q43_best,
                selected_day,
                TMP / f"question43_{selected_day:%Y%m%d}.png",
                variable_price=True,
            )
    summary = {
        "问题4-2": {
            "入选方案": q42_best.name,
            "指标": q42_rows[q42_best_index],
            "方法比较": q42_rows,
            "指定日期": q42_selected,
            "校验": q42_audit,
        },
        "问题4-3": {
            "入选方案": q43_best.name,
            "指标": q43_rows[q43_best_index],
            "方案比较": q43_rows,
            "指定日期": q43_selected,
            "校验": q43_audit,
        },
        "电价预测指标": price_metrics.to_dict(orient="records"),
    }
    core.write_json(TMP / "question4_solution_summary.json", summary)
    print(json.dumps(core.json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
