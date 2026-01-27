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
from datetime import datetime, date, time
from typing import Iterable, Iterator, List, Optional


DATE_FORMAT = "%m/%d/%y %H:%M"


@dataclass
class Bar:
    timestamp: datetime
    high: float
    low: float
    close: float
    volume: float


@dataclass
class DayMetrics:
    session_date: date
    ib_high: float
    ib_low: float
    ib_range: float
    midpoint: float
    balance_state: str
    ib_volume: float
    total_volume: float
    relative_ib_volume: float
    day_high: float
    day_low: float
    extension_up: float
    extension_down: float
    extension_1_5_up: float
    extension_2_up: float
    extension_1_5_down: float
    extension_2_down: float
    reached_1_5_up: bool
    reached_2_up: bool
    reached_1_5_down: bool
    reached_2_down: bool
    rotation: bool
    failed_auction: str


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
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume(from bar)", 0) or 0),
            )


def compute_day_metrics(
    bars: List[Bar],
    ib_start: time,
    ib_end: time,
) -> Optional[DayMetrics]:
    ib_bars = [bar for bar in bars if ib_start <= bar.timestamp.time() <= ib_end]
    if not ib_bars:
        return None

    ib_high = max(bar.high for bar in ib_bars)
    ib_low = min(bar.low for bar in ib_bars)
    ib_range = ib_high - ib_low
    midpoint = (ib_high + ib_low) / 2

    ib_volume = sum(bar.volume for bar in ib_bars)
    total_volume = sum(bar.volume for bar in bars)
    relative_ib_volume = ib_volume / total_volume if total_volume else 0.0

    day_high = max(bar.high for bar in bars)
    day_low = min(bar.low for bar in bars)
    day_close = bars[-1].close

    balance_state = "balance" if day_high <= ib_high and day_low >= ib_low else "discovery"

    extension_up = max(0.0, day_high - ib_high)
    extension_down = max(0.0, ib_low - day_low)

    extension_1_5_up = ib_high + 1.5 * ib_range
    extension_2_up = ib_high + 2.0 * ib_range
    extension_1_5_down = ib_low - 1.5 * ib_range
    extension_2_down = ib_low - 2.0 * ib_range

    reached_1_5_up = day_high >= extension_1_5_up
    reached_2_up = day_high >= extension_2_up
    reached_1_5_down = day_low <= extension_1_5_down
    reached_2_down = day_low <= extension_2_down

    after_ib = [bar for bar in bars if bar.timestamp.time() > ib_end]
    touched_high = any(bar.high >= ib_high for bar in after_ib)
    touched_low = any(bar.low <= ib_low for bar in after_ib)
    rotation = touched_high and touched_low

    failed_high = day_high > ib_high and day_close <= ib_high
    failed_low = day_low < ib_low and day_close >= ib_low
    if failed_high and failed_low:
        failed_auction = "failed_both"
    elif failed_high:
        failed_auction = "failed_high"
    elif failed_low:
        failed_auction = "failed_low"
    else:
        failed_auction = "none"

    return DayMetrics(
        session_date=bars[0].timestamp.date(),
        ib_high=ib_high,
        ib_low=ib_low,
        ib_range=ib_range,
        midpoint=midpoint,
        balance_state=balance_state,
        ib_volume=ib_volume,
        total_volume=total_volume,
        relative_ib_volume=relative_ib_volume,
        day_high=day_high,
        day_low=day_low,
        extension_up=extension_up,
        extension_down=extension_down,
        extension_1_5_up=extension_1_5_up,
        extension_2_up=extension_2_up,
        extension_1_5_down=extension_1_5_down,
        extension_2_down=extension_2_down,
        reached_1_5_up=reached_1_5_up,
        reached_2_up=reached_2_up,
        reached_1_5_down=reached_1_5_down,
        reached_2_down=reached_2_down,
        rotation=rotation,
        failed_auction=failed_auction,
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
        "--ib-start",
        type=parse_time,
        default=time(9, 30),
        help="Initial balance start time (HH:MM, 24h).",
    )
    parser.add_argument(
        "--ib-end",
        type=parse_time,
        default=time(10, 30),
        help="Initial balance end time (HH:MM, 24h).",
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
    parser.add_argument(
        "--output",
        help="Optional output CSV path. If omitted, prints CSV to stdout.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.ib_end <= args.ib_start:
        raise SystemExit("IB end time must be after IB start time.")

    metrics: List[DayMetrics] = []
    for bars in day_grouped_bars(
        iter_bars(args.csv),
        start_date=args.start_date,
        end_date=args.end_date,
    ):
        day_metric = compute_day_metrics(bars, args.ib_start, args.ib_end)
        if day_metric:
            metrics.append(day_metric)

    write_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
