# -*- coding: utf-8 -*-
"""
Phase II: Prior-day contextual levels and joins.

Computes prior-day reference levels (PDH, PDL, VAH, VAL, POC) and joins them
with Phase I outputs from mp2b_IBH_IBL.py for contextual analysis.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Dict, Iterable, List, Optional, Tuple

DATE_FORMAT = "%m/%d/%y %H:%M"

DEFAULTS = {
    "csv": "MNQ_1min_2023Jan_2026Jan.csv",
    "ib_metrics": "outputs/ib_metrics.csv",
    "output": "outputs/phase2_previous_day_levels.csv",
    "session_start": time(6, 30),
    "session_end": time(14, 0),
    "tick_size": 0.25,
    "value_area_pct": 0.7,
    "level_tolerance": 10.0,
}


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PriorDayLevels:
    session_date: date
    pdh: float
    pdl: float
    vah: float
    val: float
    poc: float
    session_volume: float


def parse_time(value: str) -> time:
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid time format '{value}'. Use HH:MM in 24h format."
        ) from exc


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date format '{value}'. Use YYYY-MM-DD."
        ) from exc


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return float(value)


def iter_bars(csv_path: str) -> Iterable[Bar]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_dt = row.get("DateTime") or row.get("\ufeffDateTime")
            if not raw_dt:
                continue
            timestamp = datetime.strptime(raw_dt.strip(), DATE_FORMAT)
            yield Bar(
                timestamp=timestamp,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume(from bar)", 0) or 0),
            )


def within_session(bar: Bar, session_start: time, session_end: time) -> bool:
    bar_time = bar.timestamp.time()
    return session_start <= bar_time <= session_end


def compute_volume_profile(bars: List[Bar], tick_size: float) -> Tuple[Dict[float, float], float]:
    profile: Dict[float, float] = {}
    total_volume = 0.0
    for bar in bars:
        typical_price = (bar.high + bar.low + bar.close) / 3
        price = round(typical_price / tick_size) * tick_size
        profile[price] = profile.get(price, 0.0) + bar.volume
        total_volume += bar.volume
    return profile, total_volume


def compute_value_area(
    profile: Dict[float, float], total_volume: float, value_area_pct: float
) -> Tuple[float, float, float]:
    if not profile or total_volume == 0:
        return 0.0, 0.0, 0.0

    prices = sorted(profile.keys())
    volumes = [profile[p] for p in prices]
    poc_index = max(range(len(prices)), key=lambda i: volumes[i])
    poc = prices[poc_index]

    cumulative = volumes[poc_index]
    upper_index = poc_index
    lower_index = poc_index

    while cumulative < total_volume * value_area_pct:
        next_upper = upper_index + 1 if upper_index + 1 < len(prices) else None
        next_lower = lower_index - 1 if lower_index - 1 >= 0 else None

        if next_upper is None and next_lower is None:
            break
        if next_upper is None:
            lower_index = next_lower
            cumulative += volumes[lower_index]
            continue
        if next_lower is None:
            upper_index = next_upper
            cumulative += volumes[upper_index]
            continue

        if volumes[next_upper] >= volumes[next_lower]:
            upper_index = next_upper
            cumulative += volumes[upper_index]
        else:
            lower_index = next_lower
            cumulative += volumes[lower_index]

    vah = prices[upper_index]
    val = prices[lower_index]
    return vah, val, poc


def compute_daily_levels(
    bars: Iterable[Bar],
    session_start: time,
    session_end: time,
    tick_size: float,
    value_area_pct: float,
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[PriorDayLevels]:
    grouped: Dict[date, List[Bar]] = {}
    for bar in bars:
        bar_date = bar.timestamp.date()
        if start_date and bar_date < start_date:
            continue
        if end_date and bar_date > end_date:
            break
        if not within_session(bar, session_start, session_end):
            continue
        grouped.setdefault(bar_date, []).append(bar)

    levels: List[PriorDayLevels] = []
    for session_date in sorted(grouped.keys()):
        day_bars = grouped[session_date]
        pdh = max(bar.high for bar in day_bars)
        pdl = min(bar.low for bar in day_bars)
        profile, total_volume = compute_volume_profile(day_bars, tick_size)
        vah, val, poc = compute_value_area(profile, total_volume, value_area_pct)
        levels.append(
            PriorDayLevels(
                session_date=session_date,
                pdh=pdh,
                pdl=pdl,
                vah=vah,
                val=val,
                poc=poc,
                session_volume=total_volume,
            )
        )
    return levels


def classify_relation(value: Optional[float], level: Optional[float], tol: float) -> Optional[str]:
    if value is None or level is None:
        return None
    if value > level + tol:
        return "above"
    if value < level - tol:
        return "below"
    return "within"


def touch_level(
    high: Optional[float], low: Optional[float], level: Optional[float], tol: float
) -> Optional[bool]:
    if high is None or low is None or level is None:
        return None
    return high >= level - tol and low <= level + tol


def breach_above(high: Optional[float], level: Optional[float], tol: float) -> Optional[bool]:
    if high is None or level is None:
        return None
    return high > level + tol


def breach_below(low: Optional[float], level: Optional[float], tol: float) -> Optional[bool]:
    if low is None or level is None:
        return None
    return low < level - tol


def nearest_level_to_open(
    opening_midpoint: Optional[float], prior_levels: Dict[str, Optional[float]]
) -> Tuple[Optional[str], Optional[float]]:
    if opening_midpoint is None:
        return None, None
    distances = {}
    for level_name, value in prior_levels.items():
        if value is None:
            continue
        distances[level_name] = abs(opening_midpoint - value)
    if not distances:
        return None, None
    nearest = min(distances, key=distances.get)
    return nearest, distances[nearest]


def load_ib_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def build_prior_level_map(levels: List[PriorDayLevels]) -> Dict[date, PriorDayLevels]:
    prior_map: Dict[date, PriorDayLevels] = {}
    levels_sorted = sorted(levels, key=lambda level: level.session_date)
    for idx, level in enumerate(levels_sorted):
        if idx == 0:
            continue
        prior_map[level.session_date] = levels_sorted[idx - 1]
    return prior_map


def add_interactions(
    ib_row: Dict[str, str],
    prior: Optional[PriorDayLevels],
    tolerance: float,
) -> Dict[str, object]:
    ib_high = parse_float(ib_row.get("ib_high"))
    ib_low = parse_float(ib_row.get("ib_low"))
    ib_range = parse_float(ib_row.get("ib_range"))
    rth_high = parse_float(ib_row.get("rth_high"))
    rth_low = parse_float(ib_row.get("rth_low"))
    rth_close = parse_float(ib_row.get("rth_close"))
    opening_high = parse_float(ib_row.get("opening_range_high"))
    opening_low = parse_float(ib_row.get("opening_range_low"))
    opening_close = parse_float(ib_row.get("opening_range_close"))

    opening_midpoint = None
    if opening_high is not None and opening_low is not None:
        opening_midpoint = (opening_high + opening_low) / 2

    prior_levels = {
        "pdh": prior.pdh if prior else None,
        "pdl": prior.pdl if prior else None,
        "vah": prior.vah if prior else None,
        "val": prior.val if prior else None,
        "poc": prior.poc if prior else None,
    }

    updates: Dict[str, object] = {
        "prev_pdh": prior_levels["pdh"],
        "prev_pdl": prior_levels["pdl"],
        "prev_vah": prior_levels["vah"],
        "prev_val": prior_levels["val"],
        "prev_poc": prior_levels["poc"],
        "prev_session_volume": prior.session_volume if prior else None,
        "opening_range_midpoint": opening_midpoint,
    }

    for level_name, level_value in prior_levels.items():
        updates[f"opening_touch_{level_name}"] = touch_level(
            opening_high, opening_low, level_value, tolerance
        )
        updates[f"rth_touch_{level_name}"] = touch_level(
            rth_high, rth_low, level_value, tolerance
        )
        updates[f"rth_breach_above_{level_name}"] = breach_above(
            rth_high, level_value, tolerance
        )
        updates[f"rth_breach_below_{level_name}"] = breach_below(
            rth_low, level_value, tolerance
        )
        updates[f"rth_close_vs_{level_name}"] = classify_relation(
            rth_close, level_value, tolerance
        )
        updates[f"opening_close_vs_{level_name}"] = classify_relation(
            opening_close, level_value, tolerance
        )

    updates["discovery_up"] = (
        rth_high is not None and ib_high is not None and rth_high > ib_high
    )
    updates["discovery_down"] = (
        rth_low is not None and ib_low is not None and rth_low < ib_low
    )
    discovery_up = updates["discovery_up"]
    discovery_down = updates["discovery_down"]
    if discovery_up is None or discovery_down is None:
        updates["balance_day"] = None
    else:
        updates["balance_day"] = not (discovery_up or discovery_down)

    extension_up = None
    if ib_high is not None and rth_high is not None:
        extension_up = max(0.0, rth_high - ib_high)
    extension_down = None
    if ib_low is not None and rth_low is not None:
        extension_down = max(0.0, ib_low - rth_low)

    if ib_range is None or ib_range == 0 or extension_up is None:
        updates["extension_up_normib"] = None
    else:
        updates["extension_up_normib"] = extension_up / ib_range
    if ib_range is None or ib_range == 0 or extension_down is None:
        updates["extension_down_normib"] = None
    else:
        updates["extension_down_normib"] = extension_down / ib_range

    nearest_name, nearest_distance = nearest_level_to_open(opening_midpoint, prior_levels)
    updates["nearest_prior_level_to_open"] = nearest_name
    updates["nearest_prior_level_to_open_distance"] = nearest_distance

    return updates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute prior-day levels and join with Phase I IB metrics."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULTS["csv"],
        help="Path to MNQ minute data CSV.",
    )
    parser.add_argument(
        "--ib-metrics",
        default=DEFAULTS["ib_metrics"],
        help="Path to Phase I IB metrics CSV (from mp2b_IBH_IBL.py).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULTS["output"],
        help="Output CSV path for joined Phase II data.",
    )
    parser.add_argument(
        "--session-start",
        type=parse_time,
        default=DEFAULTS["session_start"],
        help="Session start time for prior-day levels (HH:MM).",
    )
    parser.add_argument(
        "--session-end",
        type=parse_time,
        default=DEFAULTS["session_end"],
        help="Session end time for prior-day levels (HH:MM).",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=DEFAULTS["tick_size"],
        help="Tick size for volume profile binning.",
    )
    parser.add_argument(
        "--value-area-pct",
        type=float,
        default=DEFAULTS["value_area_pct"],
        help="Value area percent for VAH/VAL (e.g., 0.7 for 70%%).",
    )
    parser.add_argument(
        "--level-tolerance",
        type=float,
        default=DEFAULTS["level_tolerance"],
        help="Tolerance for level touch/confluence checks in points.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        help="Optional start date (YYYY-MM-DD) to filter sessions.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="Optional end date (YYYY-MM-DD) to filter sessions.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.session_end <= args.session_start:
        raise SystemExit("Session end time must be after session start time.")
    if args.tick_size <= 0:
        raise SystemExit("Tick size must be positive.")
    if not 0 < args.value_area_pct <= 1:
        raise SystemExit("Value area percent must be between 0 and 1.")

    levels = compute_daily_levels(
        iter_bars(args.csv),
        session_start=args.session_start,
        session_end=args.session_end,
        tick_size=args.tick_size,
        value_area_pct=args.value_area_pct,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    prior_map = build_prior_level_map(levels)

    ib_rows = load_ib_metrics(args.ib_metrics)
    if not ib_rows:
        raise SystemExit("IB metrics file is empty.")

    fieldnames = list(ib_rows[0].keys())
    extra_fields = [
        "prev_pdh",
        "prev_pdl",
        "prev_vah",
        "prev_val",
        "prev_poc",
        "prev_session_volume",
        "opening_range_midpoint",
        "opening_touch_pdh",
        "opening_touch_pdl",
        "opening_touch_vah",
        "opening_touch_val",
        "opening_touch_poc",
        "rth_touch_pdh",
        "rth_touch_pdl",
        "rth_touch_vah",
        "rth_touch_val",
        "rth_touch_poc",
        "rth_breach_above_pdh",
        "rth_breach_above_pdl",
        "rth_breach_above_vah",
        "rth_breach_above_val",
        "rth_breach_above_poc",
        "rth_breach_below_pdh",
        "rth_breach_below_pdl",
        "rth_breach_below_vah",
        "rth_breach_below_val",
        "rth_breach_below_poc",
        "rth_close_vs_pdh",
        "rth_close_vs_pdl",
        "rth_close_vs_vah",
        "rth_close_vs_val",
        "rth_close_vs_poc",
        "opening_close_vs_pdh",
        "opening_close_vs_pdl",
        "opening_close_vs_vah",
        "opening_close_vs_val",
        "opening_close_vs_poc",
        "discovery_up",
        "discovery_down",
        "balance_day",
        "extension_up_normib",
        "extension_down_normib",
        "nearest_prior_level_to_open",
        "nearest_prior_level_to_open_distance",
    ]
    fieldnames.extend([name for name in extra_fields if name not in fieldnames])

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in ib_rows:
            session_date_raw = row.get("session_date")
            if not session_date_raw:
                writer.writerow(row)
                continue
            session_date = parse_date(session_date_raw)
            prior = prior_map.get(session_date)
            updates = add_interactions(row, prior, args.level_tolerance)
            row.update(updates)
            writer.writerow(row)

    print(f"Saved Phase II levels to {args.output}")


if __name__ == "__main__":
    main()
