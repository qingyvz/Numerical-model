#!/usr/bin/env python3
"""对第三问输出执行独立数值与工作簿结构检查。"""

from __future__ import annotations

import gzip
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from analyze_question3_data import load_inputs, parse_date, read_xlsx


ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = ROOT / "result" / "3" / "tmp"
WORKBOOK = ROOT / "result" / "3" / "result3.xlsx"
ETA_C = 0.90
ETA_D = 0.90
SOC_MIN = 1200.0
SOC_MAX = 10800.0
SOC_TARGET = 6000.0
MAX_INTERVAL_ENERGY = 5000.0 / 6.0
DT = 1.0 / 6.0
TOL = 2.0e-4


def main() -> None:
    data = load_inputs()
    with gzip.open(
        TMP_DIR / "question3_solution_arrays.json.gz", "rt", encoding="utf-8"
    ) as handle:
        solutions: dict[str, Any] = json.load(handle)
    summary = json.loads(
        (TMP_DIR / "question3_solution_summary.json").read_text(encoding="utf-8")
    )
    load = data["load"]
    pv = data["pv"]
    price = data["price"]

    minimum_balance_margin = math.inf
    minimum_soc = math.inf
    maximum_soc = -math.inf
    maximum_charge = 0.0
    maximum_discharge = 0.0
    simultaneous_intervals = 0
    maximum_cost_difference = 0.0
    total_emergency = 0.0
    total_adjusted = 0.0
    for day_text, day_result in solutions.items():
        day = datetime.strptime(day_text, "%Y-%m-%d").date()
        plan = day_result["combinations"]["00_only"]
        final = day_result["combinations"]["00_06_12_18"]
        q0 = plan["schedule"]["purchase_kwh"]
        q = final["schedule"]["purchase_kwh"]
        charge = final["schedule"]["charge_kwh"]
        discharge = final["schedule"]["discharge_kwh"]
        emergency = final["emergency_kwh"]
        soc = final["soc_kwh"]
        assert all(value >= -TOL for value in q0)
        assert all(value >= -TOL for value in q)
        assert all(-TOL <= value <= MAX_INTERVAL_ENERGY + TOL for value in charge)
        assert all(-TOL <= value <= MAX_INTERVAL_ENERGY + TOL for value in discharge)
        assert all(value >= -TOL for value in emergency)
        assert min(soc) >= SOC_MIN - TOL
        assert max(soc) <= SOC_MAX + TOL
        assert abs(soc[-1] - SOC_TARGET) <= TOL
        state = SOC_TARGET
        for index in range(144):
            state += ETA_C * charge[index] - discharge[index] / ETA_D
            assert abs(state - soc[index]) <= 5.0e-4
            actual_net = (load[day][index] - pv[day][index]) * DT
            margin = q[index] + discharge[index] + emergency[index] - charge[index] - actual_net
            minimum_balance_margin = min(minimum_balance_margin, margin)
            assert margin >= -TOL
            expected_emergency = max(
                0.0,
                actual_net + charge[index] - discharge[index] - q[index],
            )
            assert abs(expected_emergency - emergency[index]) <= TOL
            if charge[index] > TOL and discharge[index] > TOL:
                simultaneous_intervals += 1
        maximum_charge = max(maximum_charge, max(charge))
        maximum_discharge = max(maximum_discharge, max(discharge))
        minimum_soc = min(minimum_soc, min(soc))
        maximum_soc = max(maximum_soc, max(soc))
        total_emergency += sum(emergency)
        total_adjusted += sum(q)

        plan_cost = sum(price[index] * q0[index] for index in range(144))
        upward_cost = sum(
            1.5 * price[index] * max(q[index] - q0[index], 0.0)
            for index in range(144)
        )
        downward_refund = sum(
            0.5 * price[index] * max(q0[index] - q[index], 0.0)
            for index in range(144)
        )
        emergency_cost = sum(
            5.0 * price[index] * emergency[index] for index in range(144)
        )
        recomputed = plan_cost + upward_cost - downward_refund + emergency_cost
        recorded = final["costs"]["total_cost_yuan"]
        maximum_cost_difference = max(maximum_cost_difference, abs(recomputed - recorded))
        assert abs(recomputed - recorded) <= 1.0e-5

    workbook = read_xlsx(WORKBOOK)
    assert list(workbook) == ["计划购电量", "调整购电量", "充放电量", "紧急购电量"]
    assert len(workbook["计划购电量"]) == 335
    assert len(workbook["调整购电量"]) == 335
    assert len(workbook["充放电量"]) == 1 + 334 * 6
    assert workbook["计划购电量"][0][1:145] == [
        __import__("solve_question3").INTERVAL_LABELS[index] for index in range(144)
    ]
    first_day = date(2025, 2, 1)
    last_day = date(2025, 12, 31)
    assert parse_date(workbook["计划购电量"][1][0]) == first_day
    assert parse_date(workbook["计划购电量"][-1][0]) == last_day
    first_solution = solutions[first_day.isoformat()]
    for column in (1, 60, 73, 109, 144):
        expected = first_solution["combinations"]["00_only"]["schedule"][
            "purchase_kwh"
        ][column - 1]
        actual = float(workbook["计划购电量"][1][column])
        assert abs(expected - actual) <= 1.0e-7

    report = {
        "status": "PASS",
        "solution_days": len(solutions),
        "checked_intervals": len(solutions) * 144,
        "minimum_power_balance_margin_kwh": minimum_balance_margin,
        "minimum_soc_kwh": minimum_soc,
        "maximum_soc_kwh": maximum_soc,
        "maximum_charge_per_interval_kwh": maximum_charge,
        "maximum_discharge_per_interval_kwh": maximum_discharge,
        "simultaneous_charge_discharge_intervals": simultaneous_intervals,
        "maximum_daily_cost_recalculation_difference_yuan": maximum_cost_difference,
        "total_emergency_purchase_kwh": total_emergency,
        "total_adjusted_purchase_kwh": total_adjusted,
        "workbook_sheets": list(workbook),
        "workbook_rows": {name: len(rows) for name, rows in workbook.items()},
        "summary_days": summary["days"],
    }
    (TMP_DIR / "question3_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
