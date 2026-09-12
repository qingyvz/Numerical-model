"""C题问题四：附件4波动电价的可复核描述统计。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "program" / "_deps"))
sys.path.insert(0, str(ROOT / "program"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import solution2_core as core  # noqa: E402


OUTPUT = ROOT / "result" / "4" / "tmp"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = core.load_data()
    price = data.price_actual
    raw = price.reshape(-1)
    timestamps = pd.date_range("2025-01-01 00:10", periods=raw.size, freq="10min")
    min_index = int(np.argmin(raw))
    max_index = int(np.argmax(raw))
    daily_mean = price.mean(axis=1)
    daily_min = price.min(axis=1)
    daily_max = price.max(axis=1)
    daily = pd.DataFrame(
        {
            "日期": data.base.dates.strftime("%Y-%m-%d"),
            "日均价_元每kWh": daily_mean,
            "日最低价_元每kWh": daily_min,
            "日最高价_元每kWh": daily_max,
            "日峰谷差_元每kWh": daily_max - daily_min,
        }
    )
    core.dataframe_to_csv(daily, OUTPUT / "question4_daily_price_statistics.csv")
    month_index = data.base.dates.month
    monthly_rows = []
    for month in range(1, 13):
        selected = price[month_index == month]
        monthly_rows.append(
            {
                "月份": month,
                "观测点数": int(selected.size),
                "均价_元每kWh": float(selected.mean()),
                "标准差_元每kWh": float(selected.std()),
                "最低价_元每kWh": float(selected.min()),
                "最高价_元每kWh": float(selected.max()),
            }
        )
    core.dataframe_to_csv(pd.DataFrame(monthly_rows), OUTPUT / "question4_monthly_price_statistics.csv")

    fixed = np.tile(data.fixed_price, (len(data.base.dates), 1))
    difference = price - fixed
    metrics = {
        "时间范围": [str(timestamps[0]), str(timestamps[-1])],
        "观测点数": int(raw.size),
        "缺失或非有限值": int((~np.isfinite(raw)).sum()),
        "非正电价数": int((raw <= 0).sum()),
        "全年均价_元每kWh": float(raw.mean()),
        "中位数_元每kWh": float(np.median(raw)),
        "标准差_元每kWh": float(raw.std()),
        "P05_元每kWh": float(np.quantile(raw, 0.05)),
        "P95_元每kWh": float(np.quantile(raw, 0.95)),
        "全年最低价_元每kWh": float(raw[min_index]),
        "全年最低价时刻": str(timestamps[min_index]),
        "全年最高价_元每kWh": float(raw[max_index]),
        "全年最高价时刻": str(timestamps[max_index]),
        "日均价范围_元每kWh": [float(daily_mean.min()), float(daily_mean.max())],
        "平均日峰谷差_元每kWh": float(np.mean(daily_max - daily_min)),
        "一日滞后相关系数": float(np.corrcoef(price[1:].reshape(-1), price[:-1].reshape(-1))[0, 1]),
        "七日滞后相关系数": float(np.corrcoef(price[7:].reshape(-1), price[:-7].reshape(-1))[0, 1]),
        "附件1固定曲线MAE_元每kWh": float(np.mean(np.abs(difference))),
        "附件1固定曲线偏差_元每kWh": float(np.mean(fixed - price)),
    }
    core.write_json(OUTPUT / "question4_data_metrics.json", metrics)
    print(core.json_ready(metrics))


if __name__ == "__main__":
    main()
