#!/usr/bin/env python3
"""为 C 题第三问绘制中文结果图（仅依赖 Python 标准库）。"""

from __future__ import annotations

import csv
import gzip
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TMP = ROOT / "result" / "3" / "tmp"


def svg_text(x: float, y: float, value: str, **attrs: object) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{item}"' for key, item in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{html.escape(value)}</text>'


def cost_saving_chart() -> None:
    with (TMP / "question3_policy_comparison.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    labels = ["仅 0:00", "加入 6:00", "再加入 12:00", "再加入 18:00"]
    total_costs = [float(row["total_cost_yuan"]) / 10000 for row in rows]
    savings = [(total_costs[0] - value) for value in total_costs]
    marginal = [0.0] + [total_costs[i - 1] - total_costs[i] for i in range(1, 4)]

    width, height = 960, 520
    left, right, top, bottom = 120, 45, 85, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum = 50.0
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#25334a}</style>',
        svg_text(width / 2, 38, "引入滚动预报后的累计购电费用节省", text_anchor="middle", font_size="24", font_weight="700"),
        svg_text(width / 2, 66, "比较基准：全天仅使用 0:00 预报（总费用 1498.94 万元）", text_anchor="middle", font_size="14", fill="#667085"),
    ]
    for tick in range(0, 51, 10):
        y = top + plot_height - tick / maximum * plot_height
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#d9e2f2" stroke-width="1"/>')
        pieces.append(svg_text(left - 12, y + 5, str(tick), text_anchor="end", font_size="13"))
    pieces.append(svg_text(27, top + plot_height / 2, "累计节省（万元）", text_anchor="middle", font_size="14", transform=f'rotate(-90 27 {top + plot_height / 2:.1f})'))
    slot = plot_width / len(rows)
    bar_width = 98
    colors = ["#b8c4d6", "#5b9bd5", "#70ad47", "#ed7d31"]
    for i, (label, saving, total, increment) in enumerate(zip(labels, savings, total_costs, marginal)):
        center = left + slot * (i + 0.5)
        bar_height = saving / maximum * plot_height
        y = top + plot_height - bar_height
        pieces.append(f'<rect x="{center-bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="5" fill="{colors[i]}"/>')
        pieces.append(svg_text(center, max(top + 17, y - 10), f"{saving:.2f}", text_anchor="middle", font_size="16", font_weight="700"))
        pieces.append(svg_text(center, top + plot_height + 28, label, text_anchor="middle", font_size="14", font_weight="600"))
        pieces.append(svg_text(center, top + plot_height + 50, f"总费用 {total:.2f} 万元", text_anchor="middle", font_size="12", fill="#667085"))
        if i:
            pieces.append(svg_text(center, top + plot_height + 70, f"本次边际节省 {increment:.2f} 万元", text_anchor="middle", font_size="12", fill="#667085"))
    pieces.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#25334a" stroke-width="1.5"/>')
    pieces.append("</svg>")
    (TMP / "question3_cost_saving.svg").write_text("\n".join(pieces), encoding="utf-8")


def purchase_curve_chart() -> None:
    with gzip.open(TMP / "question3_solution_arrays.json.gz", "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    days = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
    width, height = 1120, 760
    margin_x, top = 68, 72
    gap_x, gap_y = 45, 66
    panel_w = (width - 2 * margin_x - gap_x) / 2
    panel_h = (height - top - 70 - gap_y) / 2
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#25334a}</style>',
        svg_text(width / 2, 34, "指定日期的计划购电、最终调整购电与紧急购电曲线", text_anchor="middle", font_size="23", font_weight="700"),
        '<line x1="330" y1="57" x2="365" y2="57" stroke="#4472c4" stroke-width="3"/>',
        svg_text(372, 62, "计划购电", font_size="13"),
        '<line x1="475" y1="57" x2="510" y2="57" stroke="#ed7d31" stroke-width="3"/>',
        svg_text(517, 62, "最终调整购电", font_size="13"),
        '<line x1="650" y1="57" x2="685" y2="57" stroke="#c00000" stroke-width="2"/>',
        svg_text(692, 62, "紧急购电", font_size="13"),
    ]
    for panel_index, day in enumerate(days):
        row, column = divmod(panel_index, 2)
        x0 = margin_x + column * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        result = payload[day]["combinations"]
        planned = result["00_only"]["schedule"]["purchase_kwh"]
        adjusted = result["00_06_12_18"]["schedule"]["purchase_kwh"]
        emergency = result["00_06_12_18"]["emergency_kwh"]
        ymax = max(max(planned), max(adjusted), max(emergency), 1.0) * 1.10

        pieces.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" fill="#fbfcfe" stroke="#d9e2f2"/>')
        for tick in range(5):
            y = y0 + panel_h - tick / 4 * panel_h
            value = ymax * tick / 4
            pieces.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0+panel_w:.1f}" y2="{y:.1f}" stroke="#e9edf4"/>')
            pieces.append(svg_text(x0 - 8, y + 4, f"{value:.0f}", text_anchor="end", font_size="10"))
        for hour in (0, 6, 12, 18, 24):
            x = x0 + hour / 24 * panel_w
            pieces.append(svg_text(x, y0 + panel_h + 18, str(hour), text_anchor="middle", font_size="11"))
        pieces.append(svg_text(x0 + panel_w / 2, y0 - 10, day, text_anchor="middle", font_size="15", font_weight="700"))
        if column == 0:
            pieces.append(svg_text(x0 - 52, y0 + panel_h / 2, "电量/kWh", text_anchor="middle", font_size="12", transform=f'rotate(-90 {x0-52:.1f} {y0+panel_h/2:.1f})'))
        if row == 1:
            pieces.append(svg_text(x0 + panel_w / 2, y0 + panel_h + 38, "时刻/小时", text_anchor="middle", font_size="12"))

        def points(values: list[float]) -> str:
            return " ".join(
                f"{x0 + index / 143 * panel_w:.1f},{y0 + panel_h - value / ymax * panel_h:.1f}"
                for index, value in enumerate(values)
            )

        pieces.append(f'<polyline points="{points(planned)}" fill="none" stroke="#4472c4" stroke-width="2" stroke-linejoin="round"/>')
        pieces.append(f'<polyline points="{points(adjusted)}" fill="none" stroke="#ed7d31" stroke-width="2" stroke-linejoin="round"/>')
        pieces.append(f'<polyline points="{points(emergency)}" fill="none" stroke="#c00000" stroke-width="1.5" stroke-linejoin="round"/>')
    pieces.append("</svg>")
    (TMP / "question3_selected_purchase_curves.svg").write_text("\n".join(pieces), encoding="utf-8")


if __name__ == "__main__":
    cost_saving_chart()
    purchase_curve_chart()
