# -*- coding: utf-8 -*-
"""
MNQ Initial Balance analysis

Refactored to compute IBH/IBL and discovery metrics for a user-defined period.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Iterable, Iterator, List, Optional


DATE_FORMAT = "%m/%d/%y %H:%M"
# For quick VS Code runs, adjust the DEFAULTS below and run the script directly.
DEFAULTS = {
    "rth_start": time(6, 30),
    "rth_end": time(13, 0),
    "ib_start": time(6, 30),
    "ib_end": time(7, 30),
    "opening_window_minutes": 10,
    "start_date": date(2025, 1, 15),
    "end_date": date(2026, 1, 15),
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
class DayMetrics:
    session_date: date
    rth_start: str
    rth_end: str
    ib_high: float
    ib_low: float
    ib_range: float
    midpoint: float
    ib_volume: float
    total_volume: float
    relative_ib_volume: float
    rth_high: float
    rth_low: float
    rth_close: float
    rotation: bool
    failed_auction: str
    breakside: str
    breakside_rotation_up: Optional[float]
    breakside_rotation_down: Optional[float]
    breakside_retracement_points: Optional[float]
    breakside_retracement_normib: Optional[float]
    opening_window_minutes: int
    opening_range_high: Optional[float]
    opening_range_low: Optional[float]
    opening_range: Optional[float]
    opening_range_close: Optional[float]
    opening_direction: Optional[str]
    opening_type: Optional[str]
    opening_bar_open: Optional[float]
    opening_bar_close: Optional[float]
    opening_bar_open_close: Optional[float]
    opening_bar_volume: Optional[float]


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


def iter_bars(csv_path: str) -> Iterator[Bar]:
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


def compute_day_metrics(
    bars: List[Bar],
    rth_start: time,
    rth_end: time,
    ib_start: time,
    ib_end: time,
    opening_window_minutes: int,
) -> Optional[DayMetrics]:
    rth_bars = [
        bar for bar in bars if rth_start <= bar.timestamp.time() < rth_end
    ]
    if not rth_bars:
        return None

    ib_bars = [bar for bar in rth_bars if ib_start <= bar.timestamp.time() < ib_end]
    if not ib_bars:
        return None

    ib_high = max(bar.high for bar in ib_bars)
    ib_low = min(bar.low for bar in ib_bars)
    ib_range = ib_high - ib_low
    midpoint = (ib_high + ib_low) / 2

    ib_volume = sum(bar.volume for bar in ib_bars)
    total_volume = sum(bar.volume for bar in rth_bars)
    relative_ib_volume = ib_volume / total_volume if total_volume else 0.0

    rth_high = max(bar.high for bar in rth_bars)
    rth_low = min(bar.low for bar in rth_bars)
    rth_close = rth_bars[-1].close

    after_ib = [bar for bar in rth_bars if bar.timestamp.time() >= ib_end]
    touched_high = any(bar.high >= ib_high for bar in after_ib)
    touched_low = any(bar.low <= ib_low for bar in after_ib)
    rotation = touched_high and touched_low

    breakside = "none"
    for bar in after_ib:
        hit_high = bar.high >= ib_high
        hit_low = bar.low <= ib_low
        if hit_high and hit_low:
            breakside = "both"
            break
        if hit_high:
            breakside = "high"
            break
        if hit_low:
            breakside = "low"
            break

    failed_high = rth_high > ib_high and rth_close <= ib_high
    failed_low = rth_low < ib_low and rth_close >= ib_low
    if failed_high and failed_low:
        failed_auction = "failed_both"
    elif failed_high:
        failed_auction = "failed_high"
    elif failed_low:
        failed_auction = "failed_low"
    else:
        failed_auction = "none"

    breakside_rotation_up = None
    if failed_high:
        first_high_idx = next(
            (
                idx
                for idx, bar in enumerate(after_ib)
                if bar.high >= ib_high
            ),
            None,
        )
        if first_high_idx is not None:
            breakside_rotation_up = min(
                bar.low for bar in after_ib[first_high_idx:]
            )

    breakside_rotation_down = None
    if failed_low:
        first_low_idx = next(
            (
                idx
                for idx, bar in enumerate(after_ib)
                if bar.low <= ib_low
            ),
            None,
        )
        if first_low_idx is not None:
            breakside_rotation_down = max(
                bar.high for bar in after_ib[first_low_idx:]
            )

    breakside_retracement_points = None
    breakside_retracement_normib = None
    if breakside in {"high", "low"} and after_ib:
        first_break_idx = next(
            (
                idx
                for idx, bar in enumerate(after_ib)
                if (
                    breakside == "high"
                    and bar.high >= ib_high
                    or breakside == "low"
                    and bar.low <= ib_low
                )
            ),
            None,
        )
        if first_break_idx is not None:
            post_break = after_ib[first_break_idx:]
            if breakside == "high":
                lowest_low = min(bar.low for bar in post_break)
                breakside_retracement_points = ib_high - lowest_low
            else:
                highest_high = max(bar.high for bar in post_break)
                breakside_retracement_points = highest_high - ib_low
            if ib_range:
                breakside_retracement_normib = (
                    breakside_retracement_points / ib_range
                )

    opening_bar = next(
        (bar for bar in rth_bars if bar.timestamp.time() == rth_start),
        None,
    )
    opening_start_dt = datetime.combine(rth_bars[0].timestamp.date(), rth_start)
    opening_end_dt = opening_start_dt + timedelta(minutes=opening_window_minutes)
    opening_window_bars = [
        bar
        for bar in rth_bars
        if opening_start_dt <= bar.timestamp < opening_end_dt
    ]

    if opening_window_bars:
        opening_high = max(bar.high for bar in opening_window_bars)
        opening_low = min(bar.low for bar in opening_window_bars)
        opening_range = opening_high - opening_low
        opening_close = opening_window_bars[-1].close
        opening_move = (
            opening_close - opening_window_bars[0].open
            if opening_window_bars
            else 0.0
        )
        if opening_range:
            closing_location = (opening_close - opening_low) / opening_range
        else:
            closing_location = 0.5
        if opening_move > 0:
            opening_direction = "up"
        elif opening_move < 0:
            opening_direction = "down"
        else:
            opening_direction = "flat"
        drive_threshold = 0.6 * opening_range
        drive_close_high = closing_location >= 1.05
        drive_close_low = closing_location <= 0.05
        is_drive = abs(opening_move) >= drive_threshold and (drive_close_high or drive_close_low)
        opening_type = "drive" if is_drive else "auction"
    else:
        opening_high = None
        opening_low = None
        opening_range = None
        opening_close = None
        opening_direction = None
        opening_type = None

    opening_bar_open = opening_bar.open if opening_bar else None
    opening_bar_close = opening_bar.close if opening_bar else None
    opening_bar_open_close = (
        opening_bar_close - opening_bar_open
        if opening_bar_open is not None and opening_bar_close is not None
        else None
    )
    opening_bar_volume = opening_bar.volume if opening_bar else None

    return DayMetrics(
        session_date=rth_bars[0].timestamp.date(),
        rth_start=rth_start.isoformat(timespec="minutes"),
        rth_end=rth_end.isoformat(timespec="minutes"),
        ib_high=ib_high,
        ib_low=ib_low,
        ib_range=ib_range,
        midpoint=midpoint,
        ib_volume=ib_volume,
        total_volume=total_volume,
        relative_ib_volume=relative_ib_volume,
        rth_high=rth_high,
        rth_low=rth_low,
        rth_close=rth_close,
        rotation=rotation,
        failed_auction=failed_auction,
        breakside=breakside,
        breakside_rotation_up=breakside_rotation_up,
        breakside_rotation_down=breakside_rotation_down,
        breakside_retracement_points=breakside_retracement_points,
        breakside_retracement_normib=breakside_retracement_normib,
        opening_window_minutes=opening_window_minutes,
        opening_range_high=opening_high,
        opening_range_low=opening_low,
        opening_range=opening_range,
        opening_range_close=opening_close,
        opening_direction=opening_direction,
        opening_type=opening_type,
        opening_bar_open=opening_bar_open,
        opening_bar_close=opening_bar_close,
        opening_bar_open_close=opening_bar_open_close,
        opening_bar_volume=opening_bar_volume,
    )


def day_grouped_bars(
    bars: Iterable[Bar],
    start_date: Optional[date],
    end_date: Optional[date],
) -> Iterator[List[Bar]]:
    current_day: Optional[date] = None
    bucket: List[Bar] = []

    for bar in bars:
        bar_date = bar.timestamp.date()
        if start_date and bar_date < start_date:
            continue
        if end_date and bar_date > end_date:
            break

        if current_day is None:
            current_day = bar_date
        if bar_date != current_day:
            yield bucket
            bucket = []
            current_day = bar_date

        bucket.append(bar)

    if bucket:
        yield bucket


def write_metrics(metrics: List[DayMetrics], output_path: Optional[str]) -> None:
    if not metrics:
        print("No sessions matched the requested window.")
        return

    fieldnames = [field.name for field in DayMetrics.__dataclass_fields__.values()]

    if output_path:
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                writer.writerow(metric.__dict__)
        print(f"Saved analysis to {output_path}")
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for metric in metrics:
        writer.writerow(metric.__dict__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute MNQ initial balance metrics for a user-defined period."
    )
    parser.add_argument(
        "--csv",
        default="MNQ_1min_2023Jan_2026Jan.csv",
        help="Path to the MNQ minute data CSV.",
    )
    parser.add_argument(
        "--rth-start",
        type=parse_time,
        default=DEFAULTS["rth_start"],
        help="Regular Trading Hours start time (HH:MM, 24h).",
    )
    parser.add_argument(
        "--rth-end",
        type=parse_time,
        default=DEFAULTS["rth_end"],
        help="Regular Trading Hours end time (HH:MM, 24h).",
    )
    parser.add_argument(
        "--ib-start",
        type=parse_time,
        default=DEFAULTS["ib_start"],
        help="Initial balance start time (HH:MM, 24h).",
    )
    parser.add_argument(
        "--ib-end",
        type=parse_time,
        default=DEFAULTS["ib_end"],
        help="Initial balance end time (HH:MM, 24h).",
    )
    parser.add_argument(
        "--opening-window-minutes",
        type=int,
        default=DEFAULTS["opening_window_minutes"],
        help="Opening range window length in minutes from RTH start.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=DEFAULTS["start_date"],
        help="Optional start date (YYYY-MM-DD) to filter sessions.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=DEFAULTS["end_date"],
        help="Optional end date (YYYY-MM-DD) to filter sessions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/ib_metrics.csv",
        help="Optional output CSV path. If omitted, prints CSV to stdout.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.rth_end <= args.rth_start:
        raise SystemExit("RTH end time must be after RTH start time.")
    if args.ib_end <= args.ib_start:
        raise SystemExit("IB end time must be after IB start time.")
    if args.ib_start < args.rth_start or args.ib_end > args.rth_end:
        raise SystemExit("IB window must fall within the RTH window.")
    if args.opening_window_minutes <= 0:
        raise SystemExit("Opening window minutes must be a positive integer.")

    metrics: List[DayMetrics] = []
    for bars in day_grouped_bars(
        iter_bars(args.csv),
        start_date=args.start_date,
        end_date=args.end_date,
    ):
        day_metric = compute_day_metrics(
            bars,
            args.rth_start,
            args.rth_end,
            args.ib_start,
            args.ib_end,
            args.opening_window_minutes,
        )
        if day_metric:
            metrics.append(day_metric)

    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
