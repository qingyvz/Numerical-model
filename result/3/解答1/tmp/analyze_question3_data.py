#!/usr/bin/env python3
"""C 题第三问附件 1--3 的可复现数据分析（仅依赖 Python 标准库）。"""

from __future__ import annotations

import csv
import html
import json
import math
import posixpath
import re
import statistics
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
ATTACHMENT_DIR = ROOT / "CUMCM2026Problems" / "C题" / "附件"
OUTPUT_DIR = ROOT / "result" / "3" / "tmp"
EVAL_START = date(2025, 2, 1)
EVAL_END = date(2025, 12, 31)
ACTIVE_PV_THRESHOLD_KW = 10.0

NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


def _number_or_text(value: str | None) -> Any:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def read_xlsx(path: Path) -> dict[str, list[list[Any]]]:
    """Read values from a simple OOXML workbook without third-party packages."""
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall(f"{{{NS_MAIN}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t"))
                )

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationships = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root.findall(f"{{{NS_PKG_REL}}}Relationship")
        }

        output: dict[str, list[list[Any]]] = {}
        sheets = workbook_root.find(f"{{{NS_MAIN}}}sheets")
        if sheets is None:
            return output
        for sheet in sheets:
            name = sheet.attrib["name"]
            relationship_id = sheet.attrib[f"{{{NS_REL}}}id"]
            target = relationships[relationship_id].lstrip("/")
            if not target.startswith("xl/"):
                target = posixpath.normpath(posixpath.join("xl", target))
            sheet_root = ET.fromstring(archive.read(target))
            sheet_data = sheet_root.find(f"{{{NS_MAIN}}}sheetData")
            rows: list[list[Any]] = []
            if sheet_data is None:
                output[name] = rows
                continue
            for row_node in sheet_data.findall(f"{{{NS_MAIN}}}row"):
                cells: dict[int, Any] = {}
                max_index = -1
                for cell in row_node.findall(f"{{{NS_MAIN}}}c"):
                    index = _column_index(cell.attrib["r"])
                    max_index = max(max_index, index)
                    cell_type = cell.attrib.get("t")
                    if cell_type == "inlineStr":
                        inline = cell.find(f"{{{NS_MAIN}}}is")
                        value = "" if inline is None else "".join(
                            node.text or ""
                            for node in inline.iter(f"{{{NS_MAIN}}}t")
                        )
                    else:
                        value_node = cell.find(f"{{{NS_MAIN}}}v")
                        raw = None if value_node is None else value_node.text
                        if cell_type == "s" and raw is not None:
                            value = shared_strings[int(raw)]
                        elif cell_type == "b":
                            value = raw == "1"
                        else:
                            value = _number_or_text(raw)
                    cells[index] = value
                row = [None] * (max_index + 1)
                for index, value in cells.items():
                    row[index] = value
                rows.append(row)
            output[name] = rows
        return output


def parse_date(value: Any) -> date:
    if isinstance(value, (int, float)):
        return (datetime(1899, 12, 30) + timedelta(days=float(value))).date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date: {value!r}")


def parse_minutes(value: Any) -> int:
    if isinstance(value, (int, float)):
        minutes = int(round(float(value) * 24 * 60))
        return minutes
    text = str(value).strip()
    if text == "0:00+1":
        return 24 * 60
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match:
        raise ValueError(f"Unrecognized time: {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def pad(row: Sequence[Any], length: int) -> list[Any]:
    return list(row) + [None] * max(0, length - len(row))


def load_inputs() -> dict[str, Any]:
    book1 = read_xlsx(ATTACHMENT_DIR / "附件1.xlsx")
    rows1 = next(iter(book1.values()))
    price_end_minutes: list[int] = []
    price: list[float] = []
    exemplar_load: list[float] = []
    exemplar_pv: list[float] = []
    for raw_row in rows1[1:]:
        row = pad(raw_row, 4)
        price_end_minutes.append(parse_minutes(row[0]))
        price.append(float(row[1]))
        exemplar_load.append(float(row[2]))
        exemplar_pv.append(float(row[3]))

    book2 = read_xlsx(ATTACHMENT_DIR / "附件2.xlsx")
    matrices: dict[str, dict[date, list[float]]] = {}
    actual_end_minutes: list[int] | None = None
    for sheet_name, rows in book2.items():
        header = pad(rows[0], 145)
        end_minutes = [parse_minutes(value) for value in header[1:145]]
        if actual_end_minutes is None:
            actual_end_minutes = end_minutes
        elif actual_end_minutes != end_minutes:
            raise ValueError("Attachment 2 worksheets use different time grids")
        matrix: dict[date, list[float]] = {}
        for raw_row in rows[1:]:
            row = pad(raw_row, 145)
            day = parse_date(row[0])
            matrix[day] = [float(value) for value in row[1:145]]
        matrices[sheet_name] = matrix
    if actual_end_minutes is None:
        raise ValueError("Attachment 2 is empty")

    load_sheet = next(name for name in matrices if "负载" in name)
    pv_sheet = next(name for name in matrices if "光伏" in name)
    load = matrices[load_sheet]
    pv = matrices[pv_sheet]

    actual_pv_by_datetime: dict[datetime, float] = {}
    for day, values in pv.items():
        day_start = datetime.combine(day, datetime.min.time())
        for minute, value in zip(actual_end_minutes, values):
            actual_pv_by_datetime[day_start + timedelta(minutes=minute)] = value

    book3 = read_xlsx(ATTACHMENT_DIR / "附件3.xlsx")
    rows3 = next(iter(book3.values()))
    forecasts: dict[tuple[date, int], list[float]] = {}
    current_day: date | None = None
    for raw_row in rows3[1:]:
        row = pad(raw_row, 26)
        if row[0] not in (None, ""):
            current_day = parse_date(row[0])
        if current_day is None:
            raise ValueError("Attachment 3 starts with a blank date")
        issue_hour = parse_minutes(row[1]) // 60
        values = [float(value) for value in row[2:26]]
        forecasts[(current_day, issue_hour)] = values

    return {
        "price_end_minutes": price_end_minutes,
        "price": price,
        "exemplar_load": exemplar_load,
        "exemplar_pv": exemplar_pv,
        "actual_end_minutes": actual_end_minutes,
        "load": load,
        "pv": pv,
        "actual_pv_by_datetime": actual_pv_by_datetime,
        "forecasts": forecasts,
        "sheet_names": {
            "attachment_1": list(book1),
            "attachment_2": list(book2),
            "attachment_3": list(book3),
        },
    }


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    if x_ss == 0 or y_ss == 0:
        return math.nan
    return numerator / math.sqrt(x_ss * y_ss)


def describe(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "min": min(values),
        "p10": quantile(values, 0.10),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "max": max(values),
    }


def forecast_metrics(pairs: Sequence[tuple[float, float]]) -> dict[str, float | int]:
    if not pairs:
        return {
            "n": 0,
            "mae_kw": math.nan,
            "rmse_kw": math.nan,
            "bias_kw": math.nan,
            "wape_pct": math.nan,
            "p50_abs_error_kw": math.nan,
            "p90_abs_error_kw": math.nan,
            "p95_abs_error_kw": math.nan,
            "max_abs_error_kw": math.nan,
            "correlation": math.nan,
            "overforecast_rate_pct_nonzero": math.nan,
            "underforecast_rate_pct_nonzero": math.nan,
            "overforecast_sum_kw": 0.0,
            "underforecast_sum_kw": 0.0,
        }
    forecasts = [pair[0] for pair in pairs]
    actuals = [pair[1] for pair in pairs]
    errors = [forecast - actual for forecast, actual in pairs]
    absolute_errors = [abs(error) for error in errors]
    actual_sum = sum(abs(value) for value in actuals)
    nonzero_errors = [error for error in errors if abs(error) > 1e-12]
    return {
        "n": len(pairs),
        "mae_kw": mean(absolute_errors),
        "rmse_kw": math.sqrt(mean([error * error for error in errors])),
        "bias_kw": mean(errors),
        "wape_pct": 100 * sum(absolute_errors) / actual_sum if actual_sum else math.nan,
        "p50_abs_error_kw": quantile(absolute_errors, 0.50),
        "p90_abs_error_kw": quantile(absolute_errors, 0.90),
        "p95_abs_error_kw": quantile(absolute_errors, 0.95),
        "max_abs_error_kw": max(absolute_errors),
        "correlation": correlation(forecasts, actuals),
        "overforecast_rate_pct_nonzero": (
            100 * sum(error > 0 for error in nonzero_errors) / len(nonzero_errors)
            if nonzero_errors
            else math.nan
        ),
        "underforecast_rate_pct_nonzero": (
            100 * sum(error < 0 for error in nonzero_errors) / len(nonzero_errors)
            if nonzero_errors
            else math.nan
        ),
        "overforecast_sum_kw": sum(max(error, 0.0) for error in errors),
        "underforecast_sum_kw": sum(max(-error, 0.0) for error in errors),
    }


def active_pairs(pairs: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    return [
        pair
        for pair in pairs
        if max(abs(pair[0]), abs(pair[1])) >= ACTIVE_PV_THRESHOLD_KW
    ]


def endpoint_label(minute: int) -> str:
    if minute == 1440:
        return "24:00"
    return f"{minute // 60:02d}:{minute % 60:02d}"


def timestamp_label(day: date, minute: int) -> str:
    if minute == 1440:
        return f"{day.isoformat()} 24:00"
    return f"{day.isoformat()} {endpoint_label(minute)}"


def daily_energy_stats(
    day: date, load: dict[date, list[float]], pv: dict[date, list[float]]
) -> dict[str, float | str]:
    load_values = load[day]
    pv_values = pv[day]
    net = [l_value - p_value for l_value, p_value in zip(load_values, pv_values)]
    return {
        "date": day.isoformat(),
        "load_kwh": sum(load_values) / 6,
        "pv_kwh": sum(pv_values) / 6,
        "net_energy_kwh": sum(net) / 6,
        "gross_deficit_kwh": sum(max(value, 0.0) for value in net) / 6,
        "pv_surplus_kwh": sum(max(-value, 0.0) for value in net) / 6,
        "self_consumed_pv_kwh": sum(
            min(l_value, p_value) for l_value, p_value in zip(load_values, pv_values)
        )
        / 6,
        "peak_load_kw": max(load_values),
        "peak_pv_kw": max(pv_values),
        "peak_net_load_kw": max(net),
        "peak_reverse_flow_kw": max(max(-value, 0.0) for value in net),
    }


def choose_forecast(
    forecasts: dict[tuple[date, int], list[float]],
    day: date,
    target_hour: int,
    available_issues: Sequence[int],
) -> tuple[float, int]:
    valid_issues = [issue for issue in available_issues if issue < target_hour]
    if not valid_issues:
        raise ValueError(f"No forecast for {day} target hour {target_hour}")
    issue = max(valid_issues)
    horizon = target_hour - issue
    return forecasts[(day, issue)][horizon - 1], issue


def asymmetric_risk_proxy(
    records: Sequence[dict[str, Any]], hourly_prices: Sequence[float]
) -> float:
    """Mean daily diagnostic, not a settlement-cost calculation."""
    if not records:
        return math.nan
    days = {record["date"] for record in records}
    total = 0.0
    for record in records:
        error = record["forecast"] - record["actual"]
        price_value = hourly_prices[record["target_hour"] - 1]
        total += price_value * (5 * max(error, 0.0) + 0.5 * max(-error, 0.0))
    return total / len(days)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if math.isclose(low, high):
        return [low]
    span = high - low
    raw = span / count
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    first = math.floor(low / step) * step
    values = []
    value = first
    while value <= high + step * 0.5:
        if value >= low - step * 0.1:
            values.append(value)
        value += step
    return values


def line_chart_svg(
    path: Path,
    title: str,
    x_values: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
    y_floor_zero: bool = False,
) -> None:
    width, height = 960, 520
    left, right, top, bottom = 82, 28, 62, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_y = [value for _, values, _ in series for value in values if math.isfinite(value)]
    y_min = min(all_y)
    y_max = max(all_y)
    if y_floor_zero:
        y_min = min(0.0, y_min)
    padding = 0.05 * (y_max - y_min or 1)
    y_min -= padding
    y_max += padding
    x_min, x_max = min(x_values), max(x_values)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#222}.grid{stroke:#d9dee7;stroke-width:1}.axis{stroke:#46515f;stroke-width:1.3}.label{font-size:13px}.title{font-size:20px;font-weight:600}.legend{font-size:13px}</style>',
        f'<text class="title" x="{width/2}" y="32" text-anchor="middle">{html.escape(title)}</text>',
    ]
    for tick in _nice_ticks(y_min, y_max):
        y_pos = sy(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{width-right}" y2="{y_pos:.2f}"/>')
        parts.append(f'<text class="label" x="{left-10}" y="{y_pos+4:.2f}" text-anchor="end">{tick:.0f}</text>')
    for tick in range(math.ceil(x_min / 3) * 3, math.floor(x_max / 3) * 3 + 1, 3):
        x_pos = sx(tick)
        parts.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="label" x="{x_pos:.2f}" y="{height-bottom+23}" text-anchor="middle">{tick}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<text class="label" x="{left+plot_width/2}" y="{height-14}" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text class="label" transform="translate(19 {top+plot_height/2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left + 14
    for index, (name, values, color) in enumerate(series):
        points = " ".join(
            f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_values, values)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        lx = legend_x + index * 160
        parts.append(f'<line x1="{lx}" y1="{top-18}" x2="{lx+24}" y2="{top-18}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="legend" x="{lx+31}" y="{top-13}">{html.escape(name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def bar_chart_svg(
    path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
) -> None:
    width, height = 900, 500
    left, right, top, bottom = 86, 28, 64, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_max = max(values) * 1.18 if values else 1
    bar_space = plot_width / len(values)
    bar_width = bar_space * 0.58
    colors = ["#5975A4", "#5F9E6E", "#D8904A", "#B55D60"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#222}.grid{stroke:#d9dee7;stroke-width:1}.axis{stroke:#46515f;stroke-width:1.3}.label{font-size:13px}.title{font-size:20px;font-weight:600}.value{font-size:13px;font-weight:600}</style>',
        f'<text class="title" x="{width/2}" y="32" text-anchor="middle">{html.escape(title)}</text>',
    ]
    for tick in _nice_ticks(0, y_max):
        y_pos = top + (y_max - tick) / y_max * plot_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" x2="{width-right}" y2="{y_pos:.2f}"/>')
        parts.append(f'<text class="label" x="{left-10}" y="{y_pos+4:.2f}" text-anchor="end">{tick:.0f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>')
    parts.append(f'<text class="label" transform="translate(19 {top+plot_height/2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>')
    for index, (label, value) in enumerate(zip(labels, values)):
        x_pos = left + index * bar_space + (bar_space - bar_width) / 2
        bar_height = value / y_max * plot_height
        y_pos = height - bottom - bar_height
        color = colors[index % len(colors)]
        parts.append(f'<rect x="{x_pos:.2f}" y="{y_pos:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="3" fill="{color}"/>')
        parts.append(f'<text class="value" x="{x_pos+bar_width/2:.2f}" y="{y_pos-8:.2f}" text-anchor="middle">{value:.1f}</text>')
        parts.append(f'<text class="label" x="{x_pos+bar_width/2:.2f}" y="{height-bottom+25}" text-anchor="middle">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    price: list[float] = data["price"]
    price_end_minutes: list[int] = data["price_end_minutes"]
    actual_end_minutes: list[int] = data["actual_end_minutes"]
    load: dict[date, list[float]] = data["load"]
    pv: dict[date, list[float]] = data["pv"]
    forecasts: dict[tuple[date, int], list[float]] = data["forecasts"]
    actual_pv_by_datetime: dict[datetime, float] = data["actual_pv_by_datetime"]
    all_days = sorted(set(load) & set(pv))
    eval_days = [day for day in all_days if EVAL_START <= day <= EVAL_END]
    hourly_prices = [mean(price[index * 6 : (index + 1) * 6]) for index in range(24)]

    expected_minutes = list(range(10, 1441, 10))
    forecast_date_set = {key[0] for key in forecasts}
    missing_forecast_keys = [
        (day.isoformat(), issue)
        for day in all_days
        for issue in (0, 6, 12, 18)
        if (day, issue) not in forecasts
    ]
    attachment_integrity = {
        "attachment_1_rows": len(price),
        "attachment_1_time_grid_valid": price_end_minutes == expected_minutes,
        "attachment_2_days_load": len(load),
        "attachment_2_days_pv": len(pv),
        "attachment_2_date_min": min(all_days).isoformat(),
        "attachment_2_date_max": max(all_days).isoformat(),
        "attachment_2_time_points_per_day": len(actual_end_minutes),
        "attachment_2_time_grid_valid": actual_end_minutes == expected_minutes,
        "attachment_3_rows": len(forecasts),
        "attachment_3_dates": len(forecast_date_set),
        "attachment_3_date_min": min(forecast_date_set).isoformat(),
        "attachment_3_date_max": max(forecast_date_set).isoformat(),
        "attachment_3_missing_date_issue_pairs": missing_forecast_keys,
        "load_missing_cells": sum(
            not math.isfinite(value) for values in load.values() for value in values
        ),
        "pv_missing_cells": sum(
            not math.isfinite(value) for values in pv.values() for value in values
        ),
        "forecast_missing_cells": sum(
            not math.isfinite(value)
            for values in forecasts.values()
            for value in values
        ),
        "negative_load_cells": sum(value < 0 for values in load.values() for value in values),
        "negative_pv_cells": sum(value < 0 for values in pv.values() for value in values),
        "negative_forecast_cells": sum(
            value < 0 for values in forecasts.values() for value in values
        ),
        "evaluation_days": len(eval_days),
        "evaluation_start": EVAL_START.isoformat(),
        "evaluation_end": EVAL_END.isoformat(),
    }

    price_min = min(price)
    price_max = max(price)
    price_summary = {
        **describe(price),
        "std": statistics.pstdev(price),
        "coefficient_of_variation_pct": 100 * statistics.pstdev(price) / mean(price),
        "max_min_ratio": price_max / price_min,
        "min_times": [
            endpoint_label(minute)
            for minute, value in zip(price_end_minutes, price)
            if math.isclose(value, price_min)
        ],
        "max_times": [
            endpoint_label(minute)
            for minute, value in zip(price_end_minutes, price)
            if math.isclose(value, price_max)
        ],
        "hourly_mean_prices": hourly_prices,
        "six_hour_block_means": {
            f"{start:02d}:00-{start+6:02d}:00": mean(price[start * 6 : (start + 6) * 6])
            for start in (0, 6, 12, 18)
        },
    }

    daily_rows = [daily_energy_stats(day, load, pv) for day in all_days]
    eval_daily_rows = [row for row in daily_rows if EVAL_START.isoformat() <= row["date"] <= EVAL_END.isoformat()]
    energy_summary = {
        key: describe([float(row[key]) for row in eval_daily_rows])
        for key in (
            "load_kwh",
            "pv_kwh",
            "net_energy_kwh",
            "gross_deficit_kwh",
            "pv_surplus_kwh",
            "self_consumed_pv_kwh",
            "peak_load_kw",
            "peak_pv_kw",
            "peak_net_load_kw",
            "peak_reverse_flow_kw",
        )
    }
    total_load_kwh = sum(float(row["load_kwh"]) for row in eval_daily_rows)
    total_pv_kwh = sum(float(row["pv_kwh"]) for row in eval_daily_rows)
    total_surplus_kwh = sum(float(row["pv_surplus_kwh"]) for row in eval_daily_rows)
    total_self_kwh = sum(float(row["self_consumed_pv_kwh"]) for row in eval_daily_rows)
    aggregate_energy = {
        "total_load_kwh": total_load_kwh,
        "total_pv_kwh": total_pv_kwh,
        "pv_to_load_ratio_pct": 100 * total_pv_kwh / total_load_kwh,
        "total_pv_surplus_kwh": total_surplus_kwh,
        "surplus_share_of_pv_pct": 100 * total_surplus_kwh / total_pv_kwh,
        "self_consumed_pv_kwh": total_self_kwh,
        "self_consumption_share_of_pv_pct": 100 * total_self_kwh / total_pv_kwh,
        "days_with_pv_surplus": sum(float(row["pv_surplus_kwh"]) > 1e-9 for row in eval_daily_rows),
        "days_pv_surplus_above_9600_kwh": sum(
            float(row["pv_surplus_kwh"]) > 9600 for row in eval_daily_rows
        ),
    }
    daily_extremes: dict[str, dict[str, Any]] = {}
    for key in (
        "load_kwh",
        "pv_kwh",
        "net_energy_kwh",
        "gross_deficit_kwh",
        "pv_surplus_kwh",
    ):
        minimum_row = min(eval_daily_rows, key=lambda row: float(row[key]))
        maximum_row = max(eval_daily_rows, key=lambda row: float(row[key]))
        daily_extremes[key] = {
            "min": float(minimum_row[key]),
            "min_date": minimum_row["date"],
            "max": float(maximum_row[key]),
            "max_date": maximum_row["date"],
        }

    flattened_load = [value for day in eval_days for value in load[day]]
    flattened_pv = [value for day in eval_days for value in pv[day]]
    flattened_net = [
        l_value - p_value
        for day in eval_days
        for l_value, p_value in zip(load[day], pv[day])
    ]
    flattened_pairs = list(zip(flattened_load, flattened_pv))
    peak_load_value = max(flattened_load)
    peak_pv_value = max(flattened_pv)
    peak_net_value = max(flattened_net)
    peak_reverse_value = max(-value for value in flattened_net)

    def first_location(target: float, values_by_day: dict[date, list[float]], sign: float = 1.0) -> str:
        for day in eval_days:
            for minute, value in zip(actual_end_minutes, values_by_day[day]):
                if math.isclose(sign * value, target, rel_tol=1e-12, abs_tol=1e-9):
                    return timestamp_label(day, minute)
        return ""

    net_by_day = {
        day: [l_value - p_value for l_value, p_value in zip(load[day], pv[day])]
        for day in eval_days
    }
    extremes = {
        "peak_load_kw": peak_load_value,
        "peak_load_time": first_location(peak_load_value, load),
        "peak_pv_kw": peak_pv_value,
        "peak_pv_time": first_location(peak_pv_value, pv),
        "peak_net_load_kw": peak_net_value,
        "peak_net_load_time": first_location(peak_net_value, net_by_day),
        "peak_reverse_flow_kw": peak_reverse_value,
        "peak_reverse_flow_time": first_location(peak_reverse_value, net_by_day, sign=-1.0),
        "load_pv_point_correlation": correlation(
            [pair[0] for pair in flattened_pairs], [pair[1] for pair in flattened_pairs]
        ),
        "negative_net_load_slots": sum(value < 0 for value in flattened_net),
        "negative_net_load_slot_pct": 100 * sum(value < 0 for value in flattened_net) / len(flattened_net),
        "net_load_above_5000_kw_slots": sum(value > 5000 for value in flattened_net),
        "net_load_above_5000_kw_slot_pct": 100
        * sum(value > 5000 for value in flattened_net)
        / len(flattened_net),
        "reverse_flow_above_5000_kw_slots": sum(value < -5000 for value in flattened_net),
        "reverse_flow_above_5000_kw_slot_pct": 100
        * sum(value < -5000 for value in flattened_net)
        / len(flattened_net),
    }

    load_changes = []
    pv_changes = []
    for day in eval_days:
        load_changes.extend(abs(second - first) for first, second in zip(load[day], load[day][1:]))
        pv_changes.extend(abs(second - first) for first, second in zip(pv[day], pv[day][1:]))
    ramps = {
        "load_abs_10min_change_kw": describe(load_changes),
        "pv_abs_10min_change_kw": describe(pv_changes),
    }

    mean_load_profile = [mean([load[day][index] for day in eval_days]) for index in range(144)]
    mean_pv_profile = [mean([pv[day][index] for day in eval_days]) for index in range(144)]
    mean_net_profile = [l_value - p_value for l_value, p_value in zip(mean_load_profile, mean_pv_profile)]
    price_net_profile_correlation = correlation(price, mean_net_profile)

    def true_intervals(mask: Sequence[bool]) -> list[str]:
        intervals: list[str] = []
        start_index: int | None = None
        for index, value in enumerate(list(mask) + [False]):
            if value and start_index is None:
                start_index = index
            elif not value and start_index is not None:
                start_minute = (start_index + 1) * 10
                end_minute = index * 10
                intervals.append(
                    f"{endpoint_label(start_minute)}-{endpoint_label(end_minute)}"
                )
                start_index = None
        return intervals

    block_rows = []
    for start in (0, 6, 12, 18):
        indices = range(start * 6, (start + 6) * 6)
        block_rows.append(
            {
                "time_block": f"{start:02d}:00-{start+6:02d}:00",
                "mean_load_kw": mean([mean_load_profile[index] for index in indices]),
                "mean_pv_kw": mean([mean_pv_profile[index] for index in indices]),
                "mean_net_load_kw": mean([mean_net_profile[index] for index in indices]),
                "mean_price_yuan_per_kwh": mean([price[index] for index in indices]),
            }
        )

    monthly_rows = []
    for month in range(1, 13):
        month_rows = [row for row in daily_rows if int(str(row["date"])[5:7]) == month]
        monthly_rows.append(
            {
                "month": month,
                "days": len(month_rows),
                "mean_daily_load_kwh": mean([float(row["load_kwh"]) for row in month_rows]),
                "mean_daily_pv_kwh": mean([float(row["pv_kwh"]) for row in month_rows]),
                "pv_load_ratio_pct": 100
                * sum(float(row["pv_kwh"]) for row in month_rows)
                / sum(float(row["load_kwh"]) for row in month_rows),
                "mean_daily_gross_deficit_kwh": mean(
                    [float(row["gross_deficit_kwh"]) for row in month_rows]
                ),
                "mean_daily_pv_surplus_kwh": mean(
                    [float(row["pv_surplus_kwh"]) for row in month_rows]
                ),
                "days_with_pv_surplus": sum(
                    float(row["pv_surplus_kwh"]) > 1e-9 for row in month_rows
                ),
            }
        )

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_rows = []
    eval_daily_by_date = {parse_date(row["date"]): row for row in eval_daily_rows}
    for weekday_index, weekday_name in enumerate(weekday_names):
        rows = [
            row
            for day, row in eval_daily_by_date.items()
            if day.weekday() == weekday_index
        ]
        weekday_rows.append(
            {
                "weekday_index": weekday_index + 1,
                "weekday": weekday_name,
                "days": len(rows),
                "mean_daily_load_kwh": mean([float(row["load_kwh"]) for row in rows]),
                "mean_daily_pv_kwh": mean([float(row["pv_kwh"]) for row in rows]),
                "mean_daily_net_energy_kwh": mean(
                    [float(row["net_energy_kwh"]) for row in rows]
                ),
                "mean_daily_pv_surplus_kwh": mean(
                    [float(row["pv_surplus_kwh"]) for row in rows]
                ),
            }
        )
    weekday_rows_flat = [row for row in weekday_rows if int(row["weekday_index"]) <= 5]
    weekend_rows_flat = [row for row in weekday_rows if int(row["weekday_index"]) >= 6]
    weekday_weekend = {
        "weekday_mean_daily_load_kwh": sum(
            float(eval_daily_by_date[day]["load_kwh"])
            for day in eval_days
            if day.weekday() <= 4
        )
        / sum(day.weekday() <= 4 for day in eval_days),
        "weekend_mean_daily_load_kwh": sum(
            float(eval_daily_by_date[day]["load_kwh"])
            for day in eval_days
            if day.weekday() >= 5
        )
        / sum(day.weekday() >= 5 for day in eval_days),
        "weekday_mean_daily_pv_kwh": sum(
            float(eval_daily_by_date[day]["pv_kwh"])
            for day in eval_days
            if day.weekday() <= 4
        )
        / sum(day.weekday() <= 4 for day in eval_days),
        "weekend_mean_daily_pv_kwh": sum(
            float(eval_daily_by_date[day]["pv_kwh"])
            for day in eval_days
            if day.weekday() >= 5
        )
        / sum(day.weekday() >= 5 for day in eval_days),
    }
    weekday_weekend["weekend_load_discount_pct"] = 100 * (
        1
        - weekday_weekend["weekend_mean_daily_load_kwh"]
        / weekday_weekend["weekday_mean_daily_load_kwh"]
    )
    daily_load_series = [float(eval_daily_by_date[day]["load_kwh"]) for day in eval_days]
    daily_pv_series = [float(eval_daily_by_date[day]["pv_kwh"]) for day in eval_days]
    daily_net_series = [float(eval_daily_by_date[day]["net_energy_kwh"]) for day in eval_days]
    serial_dependence = {
        "load_daily_energy_lag1_correlation": correlation(
            daily_load_series[:-1], daily_load_series[1:]
        ),
        "load_daily_energy_lag7_correlation": correlation(
            daily_load_series[:-7], daily_load_series[7:]
        ),
        "pv_daily_energy_lag1_correlation": correlation(
            daily_pv_series[:-1], daily_pv_series[1:]
        ),
        "pv_daily_energy_lag7_correlation": correlation(
            daily_pv_series[:-7], daily_pv_series[7:]
        ),
        "net_daily_energy_lag1_correlation": correlation(
            daily_net_series[:-1], daily_net_series[1:]
        ),
        "net_daily_energy_lag7_correlation": correlation(
            daily_net_series[:-7], daily_net_series[7:]
        ),
    }

    policies: list[tuple[str, tuple[int, ...]]] = [
        ("00_only", (0,)),
        ("00_06", (0, 6)),
        ("00_06_12", (0, 6, 12)),
        ("00_06_12_18", (0, 6, 12, 18)),
    ]
    policy_records: dict[str, list[dict[str, Any]]] = {}
    policy_metrics_rows = []
    daily_policy_mae: dict[str, dict[date, float]] = {}
    for policy_name, issues in policies:
        records = []
        errors_by_day: dict[date, list[float]] = defaultdict(list)
        for day in eval_days:
            day_start = datetime.combine(day, datetime.min.time())
            for target_hour in range(1, 25):
                forecast, source_issue = choose_forecast(
                    forecasts, day, target_hour, issues
                )
                actual = actual_pv_by_datetime[day_start + timedelta(hours=target_hour)]
                records.append(
                    {
                        "date": day,
                        "target_hour": target_hour,
                        "forecast": forecast,
                        "actual": actual,
                        "source_issue": source_issue,
                    }
                )
                errors_by_day[day].append(abs(forecast - actual))
        policy_records[policy_name] = records
        pairs = [(record["forecast"], record["actual"]) for record in records]
        active = active_pairs(pairs)
        metrics_all = forecast_metrics(pairs)
        metrics_active = forecast_metrics(active)
        daily_mae = {day: mean(values) for day, values in errors_by_day.items()}
        daily_policy_mae[policy_name] = daily_mae
        baseline_daily = daily_policy_mae.get("00_only")
        policy_metrics_rows.append(
            {
                "policy": policy_name,
                "issues": ",".join(str(issue) for issue in issues),
                "n_all": metrics_all["n"],
                "mae_all_kw": metrics_all["mae_kw"],
                "rmse_all_kw": metrics_all["rmse_kw"],
                "bias_all_kw": metrics_all["bias_kw"],
                "wape_all_pct": metrics_all["wape_pct"],
                "n_active": metrics_active["n"],
                "mae_active_kw": metrics_active["mae_kw"],
                "rmse_active_kw": metrics_active["rmse_kw"],
                "bias_active_kw": metrics_active["bias_kw"],
                "wape_active_pct": metrics_active["wape_pct"],
                "p90_active_abs_error_kw": metrics_active["p90_abs_error_kw"],
                "correlation_active": metrics_active["correlation"],
                "asymmetric_risk_proxy_yuan_per_day": asymmetric_risk_proxy(
                    records, hourly_prices
                ),
                "days_lower_mae_than_00_only": (
                    sum(daily_mae[day] < baseline_daily[day] for day in eval_days)
                    if baseline_daily is not None
                    else 0
                ),
                "days_equal_mae_to_00_only": (
                    sum(math.isclose(daily_mae[day], baseline_daily[day]) for day in eval_days)
                    if baseline_daily is not None
                    else len(eval_days)
                ),
                "days_higher_mae_than_00_only": (
                    sum(daily_mae[day] > baseline_daily[day] for day in eval_days)
                    if baseline_daily is not None
                    else 0
                ),
            }
        )

    policy_daily_comparison_rows = []
    baseline_daily_mae = daily_policy_mae["00_only"]
    for policy_name, _ in policies[1:]:
        differences = [
            daily_policy_mae[policy_name][day] - baseline_daily_mae[day]
            for day in eval_days
        ]
        policy_daily_comparison_rows.append(
            {
                "policy": policy_name,
                "mean_daily_mae_change_kw": mean(differences),
                "median_daily_mae_change_kw": quantile(differences, 0.5),
                "p10_daily_mae_change_kw": quantile(differences, 0.1),
                "p90_daily_mae_change_kw": quantile(differences, 0.9),
                "days_lower": sum(value < -1e-12 for value in differences),
                "days_equal": sum(abs(value) <= 1e-12 for value in differences),
                "days_higher": sum(value > 1e-12 for value in differences),
                "higher_dates": ",".join(
                    day.isoformat()
                    for day, value in zip(eval_days, differences)
                    if value > 1e-12
                ),
            }
        )

    update_rows = []
    update_details: dict[str, Any] = {}
    for issue, previous_issue in ((6, 0), (12, 6), (18, 12)):
        records = []
        for day in eval_days:
            day_start = datetime.combine(day, datetime.min.time())
            for target_hour in range(issue + 1, 25):
                previous = forecasts[(day, previous_issue)][target_hour - previous_issue - 1]
                updated = forecasts[(day, issue)][target_hour - issue - 1]
                actual = actual_pv_by_datetime[day_start + timedelta(hours=target_hour)]
                records.append(
                    {
                        "date": day,
                        "target_hour": target_hour,
                        "previous": previous,
                        "updated": updated,
                        "actual": actual,
                    }
                )
        active_records = [
            record
            for record in records
            if max(abs(record["previous"]), abs(record["updated"]), abs(record["actual"]))
            >= ACTIVE_PV_THRESHOLD_KW
        ]
        previous_pairs = [(record["previous"], record["actual"]) for record in records]
        updated_pairs = [(record["updated"], record["actual"]) for record in records]
        previous_active_pairs = [
            (record["previous"], record["actual"]) for record in active_records
        ]
        updated_active_pairs = [
            (record["updated"], record["actual"]) for record in active_records
        ]
        previous_metrics = forecast_metrics(previous_pairs)
        updated_metrics = forecast_metrics(updated_pairs)
        previous_active_metrics = forecast_metrics(previous_active_pairs)
        updated_active_metrics = forecast_metrics(updated_active_pairs)
        improvements = [
            abs(record["previous"] - record["actual"])
            - abs(record["updated"] - record["actual"])
            for record in active_records
        ]
        improvement_mean = mean(improvements)
        improvement_se = (
            statistics.stdev(improvements) / math.sqrt(len(improvements))
            if len(improvements) > 1
            else math.nan
        )
        previous_daily: dict[date, list[float]] = defaultdict(list)
        updated_daily: dict[date, list[float]] = defaultdict(list)
        for record in records:
            previous_daily[record["date"]].append(abs(record["previous"] - record["actual"]))
            updated_daily[record["date"]].append(abs(record["updated"] - record["actual"]))
        previous_risk_records = [
            {
                "date": record["date"],
                "target_hour": record["target_hour"],
                "forecast": record["previous"],
                "actual": record["actual"],
            }
            for record in records
        ]
        updated_risk_records = [
            {
                "date": record["date"],
                "target_hour": record["target_hour"],
                "forecast": record["updated"],
                "actual": record["actual"],
            }
            for record in records
        ]
        row = {
            "issue_hour": issue,
            "previous_issue_hour": previous_issue,
            "target_hours": f"{issue+1}-24",
            "n_all": len(records),
            "n_active": len(active_records),
            "previous_mae_all_kw": previous_metrics["mae_kw"],
            "updated_mae_all_kw": updated_metrics["mae_kw"],
            "mae_change_all_pct": 100
            * (updated_metrics["mae_kw"] - previous_metrics["mae_kw"])
            / previous_metrics["mae_kw"],
            "previous_mae_active_kw": previous_active_metrics["mae_kw"],
            "updated_mae_active_kw": updated_active_metrics["mae_kw"],
            "mae_change_active_pct": 100
            * (updated_active_metrics["mae_kw"] - previous_active_metrics["mae_kw"])
            / previous_active_metrics["mae_kw"],
            "previous_rmse_active_kw": previous_active_metrics["rmse_kw"],
            "updated_rmse_active_kw": updated_active_metrics["rmse_kw"],
            "previous_bias_active_kw": previous_active_metrics["bias_kw"],
            "updated_bias_active_kw": updated_active_metrics["bias_kw"],
            "previous_wape_active_pct": previous_active_metrics["wape_pct"],
            "updated_wape_active_pct": updated_active_metrics["wape_pct"],
            "updated_closer_active_pct": (
                100 * sum(value > 1e-12 for value in improvements) / len(improvements)
                if improvements
                else math.nan
            ),
            "updated_equal_active_pct": (
                100 * sum(abs(value) <= 1e-12 for value in improvements) / len(improvements)
                if improvements
                else math.nan
            ),
            "mean_abs_error_reduction_active_kw": improvement_mean,
            "mean_abs_error_reduction_ci95_low_kw": improvement_mean - 1.96 * improvement_se,
            "mean_abs_error_reduction_ci95_high_kw": improvement_mean + 1.96 * improvement_se,
            "mean_abs_revision_active_kw": mean(
                [abs(record["updated"] - record["previous"]) for record in active_records]
            ),
            "days_updated_lower_mae": sum(
                mean(updated_daily[day]) < mean(previous_daily[day]) for day in eval_days
            ),
            "days_updated_equal_mae": sum(
                math.isclose(mean(updated_daily[day]), mean(previous_daily[day]))
                for day in eval_days
            ),
            "days_updated_higher_mae": sum(
                mean(updated_daily[day]) > mean(previous_daily[day]) for day in eval_days
            ),
            "max_actual_target_kw": max(record["actual"] for record in records),
            "max_previous_forecast_target_kw": max(
                record["previous"] for record in records
            ),
            "max_updated_forecast_target_kw": max(
                record["updated"] for record in records
            ),
            "nonzero_actual_target_count": sum(
                abs(record["actual"]) > 1e-12 for record in records
            ),
            "previous_asymmetric_risk_proxy_yuan_per_day": asymmetric_risk_proxy(
                previous_risk_records, hourly_prices
            ),
            "updated_asymmetric_risk_proxy_yuan_per_day": asymmetric_risk_proxy(
                updated_risk_records, hourly_prices
            ),
        }
        update_rows.append(row)
        update_details[str(issue)] = {
            "previous_all": previous_metrics,
            "updated_all": updated_metrics,
            "previous_active": previous_active_metrics,
            "updated_active": updated_active_metrics,
        }

    horizon_rows = []
    for horizon in range(1, 25):
        pairs = []
        issue_counts: dict[int, int] = defaultdict(int)
        for (issue_day, issue_hour), values in forecasts.items():
            if not (EVAL_START <= issue_day <= EVAL_END):
                continue
            target = datetime.combine(issue_day, datetime.min.time()) + timedelta(
                hours=issue_hour + horizon
            )
            if target not in actual_pv_by_datetime:
                continue
            pairs.append((values[horizon - 1], actual_pv_by_datetime[target]))
            issue_counts[issue_hour] += 1
        metrics_all = forecast_metrics(pairs)
        metrics_active = forecast_metrics(active_pairs(pairs))
        horizon_rows.append(
            {
                "horizon_hour": horizon,
                "n_all": metrics_all["n"],
                "mae_all_kw": metrics_all["mae_kw"],
                "rmse_all_kw": metrics_all["rmse_kw"],
                "bias_all_kw": metrics_all["bias_kw"],
                "wape_all_pct": metrics_all["wape_pct"],
                "n_active": metrics_active["n"],
                "mae_active_kw": metrics_active["mae_kw"],
                "rmse_active_kw": metrics_active["rmse_kw"],
                "bias_active_kw": metrics_active["bias_kw"],
                "wape_active_pct": metrics_active["wape_pct"],
                "p90_active_abs_error_kw": metrics_active["p90_abs_error_kw"],
                "correlation_active": metrics_active["correlation"],
            }
        )
    horizon_bin_rows = []
    for lower in (1, 7, 13, 19):
        upper = lower + 5
        pairs = []
        for horizon in range(lower, upper + 1):
            for (issue_day, issue_hour), values in forecasts.items():
                if not (EVAL_START <= issue_day <= EVAL_END):
                    continue
                target = datetime.combine(issue_day, datetime.min.time()) + timedelta(
                    hours=issue_hour + horizon
                )
                if target in actual_pv_by_datetime:
                    pairs.append((values[horizon - 1], actual_pv_by_datetime[target]))
        all_metrics = forecast_metrics(pairs)
        active_metrics = forecast_metrics(active_pairs(pairs))
        horizon_bin_rows.append(
            {
                "horizon_bin": f"{lower}-{upper}",
                "n_all": all_metrics["n"],
                "mae_all_kw": all_metrics["mae_kw"],
                "rmse_all_kw": all_metrics["rmse_kw"],
                "n_active": active_metrics["n"],
                "mae_active_kw": active_metrics["mae_kw"],
                "rmse_active_kw": active_metrics["rmse_kw"],
                "bias_active_kw": active_metrics["bias_kw"],
                "wape_active_pct": active_metrics["wape_pct"],
            }
        )

    baseline_endpoint_pairs = []
    baseline_hour_average_pairs = []
    for day in eval_days:
        day_start = datetime.combine(day, datetime.min.time())
        for target_hour in range(1, 25):
            forecast = forecasts[(day, 0)][target_hour - 1]
            endpoint_actual = actual_pv_by_datetime[day_start + timedelta(hours=target_hour)]
            hour_average_actual = mean(
                pv[day][(target_hour - 1) * 6 : target_hour * 6]
            )
            baseline_endpoint_pairs.append((forecast, endpoint_actual))
            baseline_hour_average_pairs.append((forecast, hour_average_actual))
    common_active_indices = [
        index
        for index, (endpoint_pair, average_pair) in enumerate(
            zip(baseline_endpoint_pairs, baseline_hour_average_pairs)
        )
        if max(
            abs(endpoint_pair[0]),
            abs(endpoint_pair[1]),
            abs(average_pair[1]),
        )
        >= ACTIVE_PV_THRESHOLD_KW
    ]
    alignment_check = {
        "endpoint_all": forecast_metrics(baseline_endpoint_pairs),
        "preceding_hour_average_all": forecast_metrics(baseline_hour_average_pairs),
        "endpoint_common_active": forecast_metrics(
            [baseline_endpoint_pairs[index] for index in common_active_indices]
        ),
        "preceding_hour_average_common_active": forecast_metrics(
            [baseline_hour_average_pairs[index] for index in common_active_indices]
        ),
    }

    monthly_forecast_rows = []
    baseline_by_key = {
        (record["date"], record["target_hour"]): record
        for record in policy_records["00_only"]
    }
    full_by_key = {
        (record["date"], record["target_hour"]): record
        for record in policy_records["00_06_12_18"]
    }
    for month in range(2, 13):
        keys = [key for key in baseline_by_key if key[0].month == month]
        baseline_pairs = [
            (baseline_by_key[key]["forecast"], baseline_by_key[key]["actual"])
            for key in keys
        ]
        full_pairs = [
            (full_by_key[key]["forecast"], full_by_key[key]["actual"]) for key in keys
        ]
        baseline_active = forecast_metrics(active_pairs(baseline_pairs))
        full_active = forecast_metrics(active_pairs(full_pairs))
        monthly_forecast_rows.append(
            {
                "month": month,
                "baseline_mae_active_kw": baseline_active["mae_kw"],
                "full_revision_mae_active_kw": full_active["mae_kw"],
                "mae_change_pct": 100
                * (full_active["mae_kw"] - baseline_active["mae_kw"])
                / baseline_active["mae_kw"],
                "baseline_rmse_active_kw": baseline_active["rmse_kw"],
                "full_revision_rmse_active_kw": full_active["rmse_kw"],
                "baseline_wape_active_pct": baseline_active["wape_pct"],
                "full_revision_wape_active_pct": full_active["wape_pct"],
            }
        )

    selected_days = [date(2025, 3, 20), date(2025, 6, 21), date(2025, 9, 23), date(2025, 12, 21)]
    selected_rows = []
    for day in selected_days:
        energy = daily_energy_stats(day, load, pv)
        baseline_records = [record for record in policy_records["00_only"] if record["date"] == day]
        full_records = [record for record in policy_records["00_06_12_18"] if record["date"] == day]
        baseline_metrics = forecast_metrics(
            [(record["forecast"], record["actual"]) for record in baseline_records]
        )
        full_metrics = forecast_metrics(
            [(record["forecast"], record["actual"]) for record in full_records]
        )
        common_active_hours = {
            record["target_hour"]
            for record in baseline_records
            if max(abs(record["forecast"]), abs(record["actual"]))
            >= ACTIVE_PV_THRESHOLD_KW
        } | {
            record["target_hour"]
            for record in full_records
            if max(abs(record["forecast"]), abs(record["actual"]))
            >= ACTIVE_PV_THRESHOLD_KW
        }
        baseline_active_metrics = forecast_metrics(
            [
                (record["forecast"], record["actual"])
                for record in baseline_records
                if record["target_hour"] in common_active_hours
            ]
        )
        full_active_metrics = forecast_metrics(
            [
                (record["forecast"], record["actual"])
                for record in full_records
                if record["target_hour"] in common_active_hours
            ]
        )
        selected_rows.append(
            {
                **energy,
                "baseline_mae_kw": baseline_metrics["mae_kw"],
                "full_revision_mae_kw": full_metrics["mae_kw"],
                "mae_change_pct": 100
                * (full_metrics["mae_kw"] - baseline_metrics["mae_kw"])
                / baseline_metrics["mae_kw"]
                if baseline_metrics["mae_kw"]
                else math.nan,
                "baseline_bias_kw": baseline_metrics["bias_kw"],
                "full_revision_bias_kw": full_metrics["bias_kw"],
                "active_point_count": len(common_active_hours),
                "baseline_active_mae_kw": baseline_active_metrics["mae_kw"],
                "full_revision_active_mae_kw": full_active_metrics["mae_kw"],
                "active_mae_change_pct": 100
                * (
                    full_active_metrics["mae_kw"]
                    - baseline_active_metrics["mae_kw"]
                )
                / baseline_active_metrics["mae_kw"]
                if baseline_active_metrics["mae_kw"]
                else math.nan,
            }
        )

    profile_rows = []
    for index, minute in enumerate(actual_end_minutes):
        profile_rows.append(
            {
                "end_time": endpoint_label(minute),
                "hour": minute / 60,
                "mean_load_kw": mean_load_profile[index],
                "mean_pv_kw": mean_pv_profile[index],
                "mean_net_load_kw": mean_net_profile[index],
                "price_yuan_per_kwh": price[index],
            }
        )

    lead_mae = [float(row["mae_active_kw"]) for row in horizon_rows]
    line_chart_svg(
        OUTPUT_DIR / "mean_load_pv_net_profile.svg",
        "2025-02-01—2025-12-31 平均日内功率曲线",
        [minute / 60 for minute in actual_end_minutes],
        [
            ("负荷", mean_load_profile, "#38598C"),
            ("光伏", mean_pv_profile, "#D69B2D"),
            ("净负荷", mean_net_profile, "#4F8F5B"),
        ],
        "时刻 / h",
        "功率 / kW",
        y_floor_zero=False,
    )
    line_chart_svg(
        OUTPUT_DIR / "forecast_mae_by_horizon.svg",
        "光伏预报有效点 MAE 随预报提前量的变化",
        list(range(1, 25)),
        [("有效点 MAE", lead_mae, "#B55D60")],
        "预报提前量 / h",
        "MAE / kW",
        y_floor_zero=True,
    )
    bar_chart_svg(
        OUTPUT_DIR / "forecast_policy_active_mae.svg",
        "不同预报更新组合的光伏有效点 MAE",
        ["仅 0:00", "+ 6:00", "+ 12:00", "+ 18:00"],
        [float(row["mae_active_kw"]) for row in policy_metrics_rows],
        "MAE / kW",
    )

    write_csv(OUTPUT_DIR / "daily_energy_summary.csv", daily_rows)
    write_csv(OUTPUT_DIR / "monthly_energy_summary.csv", monthly_rows)
    write_csv(OUTPUT_DIR / "weekday_energy_summary.csv", weekday_rows)
    write_csv(OUTPUT_DIR / "mean_10min_profile.csv", profile_rows)
    write_csv(OUTPUT_DIR / "six_hour_block_summary.csv", block_rows)
    write_csv(OUTPUT_DIR / "forecast_policy_metrics.csv", policy_metrics_rows)
    write_csv(
        OUTPUT_DIR / "forecast_policy_daily_comparison.csv",
        policy_daily_comparison_rows,
    )
    write_csv(OUTPUT_DIR / "forecast_update_metrics.csv", update_rows)
    write_csv(OUTPUT_DIR / "forecast_horizon_metrics.csv", horizon_rows)
    write_csv(OUTPUT_DIR / "forecast_horizon_bin_metrics.csv", horizon_bin_rows)
    write_csv(OUTPUT_DIR / "monthly_forecast_metrics.csv", monthly_forecast_rows)
    write_csv(OUTPUT_DIR / "selected_date_summary.csv", selected_rows)

    result = {
        "scope": {
            "root": str(ROOT),
            "evaluation_start": EVAL_START.isoformat(),
            "evaluation_end": EVAL_END.isoformat(),
            "active_pv_threshold_kw": ACTIVE_PV_THRESHOLD_KW,
        },
        "integrity": attachment_integrity,
        "price": price_summary,
        "aggregate_energy_eval_period": aggregate_energy,
        "daily_extremes_eval_period": daily_extremes,
        "daily_energy_distributions_eval_period": energy_summary,
        "power_extremes_eval_period": extremes,
        "ramps_eval_period": ramps,
        "profile": {
            "price_mean_net_profile_correlation": price_net_profile_correlation,
            "six_hour_blocks": block_rows,
            "mean_load_peak_time": endpoint_label(
                actual_end_minutes[mean_load_profile.index(max(mean_load_profile))]
            ),
            "mean_load_peak_kw": max(mean_load_profile),
            "mean_pv_peak_time": endpoint_label(
                actual_end_minutes[mean_pv_profile.index(max(mean_pv_profile))]
            ),
            "mean_pv_peak_kw": max(mean_pv_profile),
            "mean_net_load_peak_time": endpoint_label(
                actual_end_minutes[mean_net_profile.index(max(mean_net_profile))]
            ),
            "mean_net_load_peak_kw": max(mean_net_profile),
            "mean_net_load_min_time": endpoint_label(
                actual_end_minutes[mean_net_profile.index(min(mean_net_profile))]
            ),
            "mean_net_load_min_kw": min(mean_net_profile),
            "mean_net_load_negative_intervals": true_intervals(
                [value < 0 for value in mean_net_profile]
            ),
            "mean_pv_above_10kw_intervals": true_intervals(
                [value >= ACTIVE_PV_THRESHOLD_KW for value in mean_pv_profile]
            ),
        },
        "monthly_energy": monthly_rows,
        "weekday_energy": weekday_rows,
        "weekday_weekend": weekday_weekend,
        "daily_serial_dependence": serial_dependence,
        "forecast_alignment_check": alignment_check,
        "forecast_policies": policy_metrics_rows,
        "forecast_policy_daily_comparisons": policy_daily_comparison_rows,
        "forecast_updates": update_rows,
        "forecast_update_detail": update_details,
        "forecast_horizon_bins": horizon_bin_rows,
        "monthly_forecast": monthly_forecast_rows,
        "selected_dates": selected_rows,
    }
    (OUTPUT_DIR / "analysis_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
