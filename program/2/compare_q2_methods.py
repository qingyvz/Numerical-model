"""比较问题二的多种严格因果预测—优化方法，并选出实际总费用最低者。

运行：python -X utf8 program/2/compare_q2_methods.py
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import q2_causal_core as core


FORECAST_CACHE = core.COMPARISON_DIR / "causal_forecasts.npz"


def save_forecasts(forecasts: dict[str, core.ForecastBundle]) -> None:
    FORECAST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "model_version": np.array([core.MODEL_VERSION]),
        "ids": np.array(list(forecasts)),
    }
    metadata: dict[str, Any] = {}
    for method_id, bundle in forecasts.items():
        payload[f"{method_id}_load"] = bundle.load_kw
        payload[f"{method_id}_pv"] = bundle.pv_kw
        metadata[method_id] = {
            "name": bundle.name,
            "fit_seconds": bundle.fit_seconds,
            "note": bundle.note,
            "hyperparameters": bundle.hyperparameters,
        }
    payload["metadata_json"] = np.array(
        [json.dumps(metadata, ensure_ascii=False)], dtype=str
    )
    np.savez_compressed(FORECAST_CACHE, **payload)


def load_forecasts() -> dict[str, core.ForecastBundle] | None:
    if not FORECAST_CACHE.exists():
        return None
    arrays = np.load(FORECAST_CACHE, allow_pickle=False)
    if str(arrays["model_version"][0]) != core.MODEL_VERSION:
        return None
    metadata = json.loads(str(arrays["metadata_json"][0]))
    result: dict[str, core.ForecastBundle] = {}
    for method_id_value in arrays["ids"]:
        method_id = str(method_id_value)
        item = metadata[method_id]
        result[method_id] = core.ForecastBundle(
            method_id=method_id,
            name=item["name"],
            load_kw=arrays[f"{method_id}_load"],
            pv_kw=arrays[f"{method_id}_pv"],
            fit_seconds=float(item["fit_seconds"]),
            note=item["note"],
            hyperparameters=item["hyperparameters"],
        )
    return result


def build_forecasts(data: core.InputData) -> dict[str, core.ForecastBundle]:
    cached = load_forecasts()
    if cached is not None:
        print("读取严格因果预测缓存。", flush=True)
        return cached
    print("生成7日周期、Ridge与树模型的严格因果滚动预测……", flush=True)
    seasonal = core.make_seasonal_forecast(data)
    ridge = core.make_ridge_forecast(data)
    tree = core.make_tree_forecast(data, ridge)
    forecasts = {
        seasonal.method_id: seasonal,
        ridge.method_id: ridge,
        tree.method_id: tree,
    }
    save_forecasts(forecasts)
    return forecasts


def load_cached_strategy(method_id: str) -> core.StrategyResult | None:
    path = core.COMPARISON_DIR / f"{method_id}.npz"
    metrics_path = core.COMPARISON_DIR / f"{method_id}_metrics.json"
    if not path.exists() or not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("model_version") != core.MODEL_VERSION:
            return None
        return core.load_strategy(path)
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def save_method(data: core.InputData, strategy: core.StrategyResult) -> dict[str, Any]:
    metrics = core.evaluate_strategy(data, strategy)
    core.save_strategy(strategy, core.COMPARISON_DIR / f"{strategy.method_id}.npz")
    (core.COMPARISON_DIR / f"{strategy.method_id}_metrics.json").write_text(
        json.dumps(core.json_ready(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def create_forecast_figure(metrics: list[dict[str, Any]], path) -> None:
    labels = [item["forecast_name"].replace("预测", "") for item in metrics]
    net_mae = [item["net_load"]["mae_kw"] for item in metrics]
    net_rmse = [item["net_load"]["rmse_kw"] for item in metrics]
    x = np.arange(len(labels))
    width = 0.35
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    bars1 = axis.bar(x - width / 2, net_mae, width, label="净负荷MAE", color="#4472c4")
    bars2 = axis.bar(x + width / 2, net_rmse, width, label="净负荷RMSE", color="#ed7d31")
    axis.bar_label(bars1, fmt="%.1f", padding=3, fontsize=9)
    axis.bar_label(bars2, fmt="%.1f", padding=3, fontsize=9)
    axis.set_xticks(x, labels)
    axis.set_ylabel("误差（kW）")
    axis.set_title("严格因果滚动预测误差（2025年2月1日至12月31日）")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_key_date_forecast_figure(
    data: core.InputData,
    forecasts: dict[str, core.ForecastBundle],
    path,
) -> None:
    hours = np.arange(core.SLOTS_PER_DAY) / 6.0
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.8), sharex=True)
    colors = {
        "actual": "#222222",
        "seasonal_7d": "#a5a5a5",
        "ridge_causal": "#4472c4",
        "hist_gbdt_causal": "#ed7d31",
    }
    for axis, date in zip(axes.ravel(), core.KEY_DATES):
        day = int(np.flatnonzero(data.dates == date)[0])
        axis.plot(hours, data.net_kw[day], label="实际净负荷", color=colors["actual"], linewidth=1.7)
        for method_id, bundle in forecasts.items():
            predicted = bundle.load_kw[day] - bundle.pv_kw[day]
            axis.plot(
                hours,
                predicted,
                label=bundle.name.replace("预测", ""),
                color=colors[method_id],
                linewidth=1.0,
                alpha=0.9,
            )
        axis.axhline(0, color="#666666", linewidth=0.6)
        axis.set_title(f"{date.year}年{date.month}月{date.day}日")
        axis.set_xticks(np.arange(0, 25, 4))
        axis.grid(alpha=0.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
    )
    figure.supxlabel("时刻")
    figure.supylabel("净负荷功率（kW）")
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.90, hspace=0.20, wspace=0.10)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_method_figure(frame: pd.DataFrame, path) -> None:
    ordered = frame.sort_values("total_cost_yuan", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    plan = ordered["scheduled_cost_yuan"] / 1_000_000.0
    emergency = ordered["emergency_cost_yuan"] / 1_000_000.0
    figure, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.barh(y, plan, label="计划购电费", color="#4472c4")
    axis.barh(y, emergency, left=plan, label="紧急购电费", color="#ed7d31")
    axis.set_yticks(y, ordered["method_name"])
    axis.set_xlabel("输出期总购电费（百万元）")
    axis.set_title("候选方法与完全预知下界的实际费用比较")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="x", alpha=0.2)
    for index, total in enumerate(ordered["total_cost_yuan"] / 1_000_000.0):
        axis.text(total + 0.02, index, f"{total:.3f}", va="center", fontsize=9)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_risk_figure(frame: pd.DataFrame, path) -> None:
    risk_ids = ["tree_residual_saa", "tree_residual_cvar_010", "tree_residual_cvar_030"]
    risk = frame.set_index("method_id").loc[risk_ids].reset_index()
    labels = ["风险中性（权重0）", "CVaR权重0.10", "CVaR权重0.30"]
    figure, left = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    right = left.twinx()
    x = np.arange(len(risk))
    left.plot(x, risk["scheduled_cost_yuan"] / 1_000_000.0, marker="o", label="计划购电费", color="#4472c4")
    left.plot(x, risk["total_cost_yuan"] / 1_000_000.0, marker="o", label="实际总费用", color="#70ad47")
    right.bar(x, risk["emergency_energy_mwh"], width=0.34, alpha=0.55, label="紧急购电量", color="#ed7d31")
    left.set_xticks(x, labels)
    left.set_ylabel("费用（百万元）")
    right.set_ylabel("紧急购电量（MWh）")
    left.set_title("风险系数对计划费用与紧急购电的影响")
    handles1, labels1 = left.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    left.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=3, loc="upper center")
    left.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def chinese_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "method_id": "方法编号",
        "method_name": "方法",
        "forecast_id": "预测编号",
        "planned_energy_mwh": "计划购电量（MWh）",
        "emergency_energy_mwh": "紧急购电量（MWh）",
        "spill_energy_mwh": "弃电量（MWh）",
        "scheduled_cost_yuan": "计划购电费（元）",
        "emergency_cost_yuan": "紧急购电费（元）",
        "total_cost_yuan": "总购电费（元）",
        "cost_gap_vs_perfect_yuan": "相对完全预知下界差额（元）",
        "cost_gap_vs_perfect_pct": "相对完全预知下界差距（%）",
        "emergency_intervals": "紧急购电时段数",
        "emergency_days": "紧急购电天数",
        "supply_interruptions": "供电中断次数",
        "solve_seconds": "优化求解时间（秒）",
        "eligible": "参与可实施方法排名",
        "algorithm_note": "方法说明",
    }
    return frame[[column for column in columns if column in frame.columns]].rename(columns=columns)


def main() -> None:
    core.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    core.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    print("读取附件数据并建立严格的信息时间边界……", flush=True)
    data = core.load_inputs()
    forecasts = build_forecasts(data)

    forecast_results = [core.forecast_metrics(data, bundle) for bundle in forecasts.values()]
    forecast_rows: list[dict[str, Any]] = []
    for item in forecast_results:
        forecast_rows.append(
            {
                "预测编号": item["forecast_id"],
                "预测方法": item["forecast_name"],
                "负载MAE（kW）": item["load"]["mae_kw"],
                "负载RMSE（kW）": item["load"]["rmse_kw"],
                "光伏MAE（kW）": item["pv"]["mae_kw"],
                "光伏RMSE（kW）": item["pv"]["rmse_kw"],
                "净负荷MAE（kW）": item["net_load"]["mae_kw"],
                "净负荷RMSE（kW）": item["net_load"]["rmse_kw"],
                "净负荷偏差（kW）": item["net_load"]["bias_kw"],
                "训练耗时（秒）": item["fit_seconds"],
            }
        )
    pd.DataFrame(forecast_rows).to_csv(
        core.OUTPUT_DIR / "q2_forecast_comparison.csv", index=False, encoding="utf-8-sig"
    )
    (core.OUTPUT_DIR / "q2_forecast_metrics.json").write_text(
        json.dumps(core.json_ready(forecast_results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    create_forecast_figure(forecast_results, core.FIGURE_DIR / "04_causal_forecast_comparison.png")
    create_key_date_forecast_figure(data, forecasts, core.FIGURE_DIR / "06_key_date_forecasts.png")

    seasonal = forecasts["seasonal_7d"]
    ridge = forecasts["ridge_causal"]
    tree = forecasts["hist_gbdt_causal"]
    builders: list[tuple[str, str, bool, Callable[[], core.StrategyResult]]] = [
        ("seasonal_point_lp", "7日周期预测+日前LP", True, lambda: core.run_point_forecast_strategy(data, seasonal, method_id="seasonal_point_lp", name="7日周期预测+日前LP")),
        ("ridge_point_lp", "Ridge正则化时序预测+日前LP", True, lambda: core.run_point_forecast_strategy(data, ridge, method_id="ridge_point_lp", name="Ridge正则化时序预测+日前LP")),
        ("tree_point_lp", "梯度提升树预测+日前LP", True, lambda: core.run_point_forecast_strategy(data, tree, method_id="tree_point_lp", name="梯度提升树预测+日前LP")),
        ("direct_similar_saa", "历史相似日两阶段场景LP", True, lambda: core.run_scenario_strategy(data, method_id="direct_similar_saa", name="历史相似日两阶段场景LP", scenario_mode="direct_similar", scenario_count=6)),
        ("ridge_residual_saa", "Ridge残差场景两阶段LP", True, lambda: core.run_scenario_strategy(data, method_id="ridge_residual_saa", name="Ridge残差场景两阶段LP", scenario_mode="residual", forecast=ridge, scenario_count=6)),
        ("tree_residual_saa", "树模型残差场景两阶段LP", True, lambda: core.run_scenario_strategy(data, method_id="tree_residual_saa", name="树模型残差场景两阶段LP", scenario_mode="residual", forecast=tree, scenario_count=6)),
        ("tree_residual_cvar_010", "树模型残差场景LP+CVaR(0.10)", True, lambda: core.run_scenario_strategy(data, method_id="tree_residual_cvar_010", name="树模型残差场景LP+CVaR(0.10)", scenario_mode="residual", forecast=tree, scenario_count=6, risk_weight=0.10)),
        ("tree_residual_cvar_030", "树模型残差场景LP+CVaR(0.30)", True, lambda: core.run_scenario_strategy(data, method_id="tree_residual_cvar_030", name="树模型残差场景LP+CVaR(0.30)", scenario_mode="residual", forecast=tree, scenario_count=6, risk_weight=0.30)),
        ("perfect_foresight_lower_bound", "完全预知未来下界（不可实施）", False, lambda: core.run_perfect_foresight_lower_bound(data)),
    ]

    records: list[dict[str, Any]] = []
    for method_id, name, eligible, builder in builders:
        cached = load_cached_strategy(method_id)
        if cached is None:
            print(f"求解：{name}", flush=True)
            strategy = builder()
            metrics = save_method(data, strategy)
        else:
            print(f"读取缓存：{name}", flush=True)
            strategy = cached
            metrics = core.evaluate_strategy(data, strategy)
        metrics["eligible"] = eligible
        records.append(metrics)
        print(
            f"  完成：总费用={metrics['total_cost_yuan']:,.2f}元，"
            f"紧急电={metrics['emergency_energy_mwh']:.3f}MWh，中断={metrics['supply_interruptions']}",
            flush=True,
        )

    frame = pd.DataFrame(records)
    perfect_cost = float(frame.loc[frame["method_id"] == "perfect_foresight_lower_bound", "total_cost_yuan"].iloc[0])
    frame["cost_gap_vs_perfect_yuan"] = frame["total_cost_yuan"] - perfect_cost
    frame["cost_gap_vs_perfect_pct"] = frame["cost_gap_vs_perfect_yuan"] / perfect_cost * 100.0
    eligible_frame = frame.loc[frame["eligible"] & (frame["supply_interruptions"] == 0)].copy()
    if eligible_frame.empty:
        raise RuntimeError("没有满足零供电中断的可实施候选方法。")
    best = eligible_frame.sort_values(["total_cost_yuan", "solve_seconds"]).iloc[0]

    frame.to_csv(core.COMPARISON_DIR / "q2_method_comparison_internal.csv", index=False, encoding="utf-8-sig")
    chinese_comparison(frame).to_csv(core.OUTPUT_DIR / "q2_method_comparison.csv", index=False, encoding="utf-8-sig")
    create_method_figure(frame, core.FIGURE_DIR / "08_q2_method_comparison.png")
    create_risk_figure(frame, core.FIGURE_DIR / "09_q2_risk_tradeoff.png")
    best_payload = {
        "model_version": core.MODEL_VERSION,
        "selection_rule": "先要求供电中断次数为0，再在严格因果可实施方法中选择实际总购电费最低者",
        "method_id": str(best["method_id"]),
        "method_name": str(best["method_name"]),
        "total_cost_yuan": float(best["total_cost_yuan"]),
        "scheduled_cost_yuan": float(best["scheduled_cost_yuan"]),
        "emergency_cost_yuan": float(best["emergency_cost_yuan"]),
        "emergency_energy_mwh": float(best["emergency_energy_mwh"]),
        "perfect_foresight_lower_bound_yuan": perfect_cost,
        "gap_vs_perfect_yuan": float(best["cost_gap_vs_perfect_yuan"]),
        "gap_vs_perfect_pct": float(best["cost_gap_vs_perfect_pct"]),
    }
    (core.OUTPUT_DIR / "q2_best_method.json").write_text(
        json.dumps(core.json_ready(best_payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"最优可实施方法：{best_payload['method_name']}；总费用={best_payload['total_cost_yuan']:,.2f}元。",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"方法比较失败：{exc}", file=sys.stderr)
        raise
