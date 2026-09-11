"""载入方法比较选出的严格因果最优策略，并生成问题二最终数据。

运行：python -X utf8 program/2/solve_q2.py
"""

from __future__ import annotations

import json
import sys

import q2_causal_core as core


def main() -> None:
    best_path = core.OUTPUT_DIR / "q2_best_method.json"
    if not best_path.exists():
        print("尚无方法比较结果，先运行完整比较……", flush=True)
        from compare_q2_methods import main as compare_main

        compare_main()
    best = json.loads(best_path.read_text(encoding="utf-8"))
    if best.get("model_version") != core.MODEL_VERSION:
        print("方法比较缓存版本已过期，重新比较……", flush=True)
        from compare_q2_methods import main as compare_main

        compare_main()
        best = json.loads(best_path.read_text(encoding="utf-8"))

    method_id = str(best["method_id"])
    strategy_path = core.COMPARISON_DIR / f"{method_id}.npz"
    if not strategy_path.exists():
        raise FileNotFoundError(f"入选策略缓存不存在：{strategy_path}")
    data = core.load_inputs()
    strategy = core.load_strategy(strategy_path)
    metrics = core.write_final_outputs(data, strategy)
    print(
        f"最终策略：{strategy.name}\n"
        f"输出期总费用：{metrics['total_cost_yuan']:,.2f} 元\n"
        f"紧急购电：{metrics['emergency_energy_mwh']:.3f} MWh\n"
        f"供电中断：{metrics['supply_interruptions']} 次",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"求解失败：{exc}", file=sys.stderr)
        raise
