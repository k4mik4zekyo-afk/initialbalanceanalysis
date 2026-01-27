#!/usr/bin/env python3
"""Initial Balance analytics for MNQ futures data.

Reads minute data from a CSV file and computes daily initial balance (IB)
metrics for a user-defined time window.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def parse_rows(path: Path, limit: Optional[int] = None) -> Iterator[Bar]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        count = 0
        for row in reader:
            if limit is not None and count >= limit:
                break
            timestamp = datetime.strptime(row["DateTime"], "%m/%d/%y %H:%M")
            yield Bar(
                timestamp=timestamp,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume(from bar)"]),
            )
            count += 1


def group_by_date(bars: Iterable[Bar]) -> dict[datetime.date, list[Bar]]:
    grouped: dict[datetime.date, list[Bar]] = {}
    for bar in bars:
        grouped.setdefault(bar.timestamp.date(), []).append(bar)
    return grouped


def compute_ib_metrics(
    bars: list[Bar],
    ib_start: time,
    ib_end: time,
    session_start: time,
    session_end: time,
    prior_ib_volume: Optional[float],
) -> Optional[dict[str, object]]:
    session_bars = [
        bar
        for bar in bars
        if session_start <= bar.timestamp.time() <= session_end
    ]
    if not session_bars:
        return None

    ib_bars = [
        bar
        for bar in session_bars
        if ib_start <= bar.timestamp.time() < ib_end
    ]
    if not ib_bars:
        return None

    ibh = max(bar.high for bar in ib_bars)
    ibl = min(bar.low for bar in ib_bars)
    ib_range = ibh - ibl
    ib_volume = sum(bar.volume for bar in ib_bars)
    ib_volume_rel = ib_volume / prior_ib_volume if prior_ib_volume else None
    midpoint = (ibh + ibl) / 2

    after_ib = [bar for bar in session_bars if bar.timestamp.time() >= ib_end]
    if after_ib:
        max_high = max(bar.high for bar in after_ib)
        min_low = min(bar.low for bar in after_ib)
    else:
        max_high = max(bar.high for bar in session_bars)
        min_low = min(bar.low for bar in session_bars)

    extension_above = max(0.0, max_high - ibh)
    extension_below = max(0.0, ibl - min_low)

    discovery_up = max_high > ibh
    discovery_down = min_low < ibl
    balance_vs_discovery = "discovery" if (discovery_up or discovery_down) else "balance"

    extension_1_5x_up = max_high >= ibh + 1.5 * ib_range
    extension_1_5x_down = min_low <= ibl - 1.5 * ib_range
    extension_2x_up = max_high >= ibh + 2 * ib_range
    extension_2x_down = min_low <= ibl - 2 * ib_range

    rotation_between = (
        any(bar.high >= ibh for bar in after_ib)
        and any(bar.low <= ibl for bar in after_ib)
    )

    failed_auction, break_side = detect_failed_auction(after_ib, ibh, ibl)

    session_close = session_bars[-1].close

    return {
        "date": session_bars[0].timestamp.date().isoformat(),
        "session_open": session_bars[0].open,
        "session_close": session_close,
        "session_high": max(bar.high for bar in session_bars),
        "session_low": min(bar.low for bar in session_bars),
        "balance_vs_discovery": balance_vs_discovery,
        "ibh": ibh,
        "ibl": ibl,
        "ib_mid": midpoint,
        "ib_range": ib_range,
        "ib_volume": ib_volume,
        "ib_volume_rel": ib_volume_rel,
        "discovery_up": discovery_up,
        "discovery_down": discovery_down,
        "max_extension_above": extension_above,
        "max_extension_below": extension_below,
        "extension_1_5x_up": extension_1_5x_up,
        "extension_1_5x_down": extension_1_5x_down,
        "extension_2x_up": extension_2x_up,
        "extension_2x_down": extension_2x_down,
        "rotation_between_ibh_ibl": rotation_between,
        "failed_auction": failed_auction,
        "failed_auction_side": break_side,
    }


def detect_failed_auction(after_ib: list[Bar], ibh: float, ibl: float) -> tuple[bool, Optional[str]]:
    if not after_ib:
        return False, None

    break_above_idx = next(
        (idx for idx, bar in enumerate(after_ib) if bar.high > ibh),
        None,
    )
    break_below_idx = next(
        (idx for idx, bar in enumerate(after_ib) if bar.low < ibl),
        None,
    )

    if break_above_idx is None and break_below_idx is None:
        return False, None

    if break_below_idx is None or (
        break_above_idx is not None and break_above_idx < break_below_idx
    ):
        side = "IBH"
        start_idx = break_above_idx
    else:
        side = "IBL"
        start_idx = break_below_idx

    reentered = any(
        ibl <= bar.close <= ibh for bar in after_ib[start_idx + 1 :]
    )
    return reentered, side


def analyze(
    path: Path,
    ib_start: time,
    ib_end: time,
    session_start: time,
    session_end: time,
    limit: Optional[int] = None,
) -> list[dict[str, object]]:
    bars = list(parse_rows(path, limit=limit))
    grouped = group_by_date(bars)
    results: list[dict[str, object]] = []
    prior_ib_volume: Optional[float] = None

    for session_date in sorted(grouped.keys()):
        metrics = compute_ib_metrics(
            grouped[session_date],
            ib_start=ib_start,
            ib_end=ib_end,
            session_start=session_start,
            session_end=session_end,
            prior_ib_volume=prior_ib_volume,
        )
        if metrics is None:
            continue
        results.append(metrics)
        prior_ib_volume = metrics["ib_volume"]

    return results


def write_csv(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initial Balance analytics for MNQ.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("MNQ_1min_2023Jan_2026Jan.csv"),
        help="CSV file containing MNQ minute data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("initial_balance_analysis.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--ib-start", type=parse_time, default=parse_time("09:30"))
    parser.add_argument("--ib-end", type=parse_time, default=parse_time("10:30"))
    parser.add_argument(
        "--session-start", type=parse_time, default=parse_time("09:30")
    )
    parser.add_argument(
        "--session-end", type=parse_time, default=parse_time("16:00")
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quicker testing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results = analyze(
        path=args.data,
        ib_start=args.ib_start,
        ib_end=args.ib_end,
        session_start=args.session_start,
        session_end=args.session_end,
        limit=args.limit,
    )

    write_csv(results, args.output)
    print(f"Wrote {len(results)} daily rows to {args.output}")


if __name__ == "__main__":
    main()
