#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script: prints IBH, IBL, RTH_high, RTH_low for the last RTH
session (or a specified date) along with the first and last 5 bars of the
IB window so the values can be manually cross-checked against a chart.
"""

from __future__ import annotations

import argparse
from datetime import date, time
from typing import List, Optional

from mp2b_IBH_IBL import (
    Bar,
    DayMetrics,
    compute_day_metrics,
    day_grouped_bars,
    iter_bars,
    parse_date,
    parse_time,
    DEFAULTS,
)


def find_session(
    csv_path: str,
    target_date: Optional[date],
    start_date: Optional[date],
    end_date: Optional[date],
    rth_start: time,
    rth_end: time,
    ib_start: time,
    ib_end: time,
    opening_window_minutes: int,
):
    """Return (DayMetrics, all_bars_for_day) for the target or last session."""
    last_metrics: Optional[DayMetrics] = None
    last_bars: List[Bar] = []

    for bars in day_grouped_bars(iter_bars(csv_path), start_date, end_date):
        bar_date = bars[0].timestamp.date()
        m = compute_day_metrics(
            bars, rth_start, rth_end, ib_start, ib_end, opening_window_minutes
        )
        if m is None:
            continue
        if target_date is not None:
            if bar_date == target_date:
                return m, bars
        else:
            last_metrics = m
            last_bars = bars

    if target_date is not None:
        return None, []
    return last_metrics, last_bars


def print_session_report(
    m: DayMetrics,
    bars: List[Bar],
    rth_start: time,
    rth_end: time,
    ib_start: time,
    ib_end: time,
):
    rth_bars = [
        b for b in bars if rth_start <= b.timestamp.time() < rth_end
    ]
    ib_bars = [
        b for b in bars if ib_start <= b.timestamp.time() < ib_end
    ]

    sep = "=" * 68
    print(sep)
    print(f"  Session Verification Report — {m.session_date}")
    print(sep)

    print(f"\n  IB window:  {ib_start.isoformat(timespec='minutes')} – "
          f"{ib_end.isoformat(timespec='minutes')} (exclusive)")
    print(f"  RTH window: {rth_start.isoformat(timespec='minutes')} – "
          f"{rth_end.isoformat(timespec='minutes')} (exclusive)")
    print(f"  IB bars:    {len(ib_bars)}")
    print(f"  RTH bars:   {len(rth_bars)}")

    print(f"\n  {'Metric':<12} {'Value':>12}")
    print(f"  {'-' * 26}")
    print(f"  {'IBH':<12} {m.ib_high:>12.2f}")
    print(f"  {'IBL':<12} {m.ib_low:>12.2f}")
    print(f"  {'IB Range':<12} {m.ib_range:>12.2f}")
    print(f"  {'RTH High':<12} {m.rth_high:>12.2f}")
    print(f"  {'RTH Low':<12} {m.rth_low:>12.2f}")
    print(f"  {'RTH Close':<12} {m.rth_close:>12.2f}")

    header = f"  {'Time':<14} {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10} {'Volume':>10}"
    row_sep = f"  {'-' * 66}"

    n = min(5, len(ib_bars))
    print(f"\n  First {n} IB bars:")
    print(header)
    print(row_sep)
    for b in ib_bars[:n]:
        t = b.timestamp.strftime("%m/%d/%y %H:%M")
        print(f"  {t:<14} {b.open:>10.2f} {b.high:>10.2f} {b.low:>10.2f} {b.close:>10.2f} {b.volume:>10.0f}")

    print(f"\n  Last {n} IB bars:")
    print(header)
    print(row_sep)
    for b in ib_bars[-n:]:
        t = b.timestamp.strftime("%m/%d/%y %H:%M")
        print(f"  {t:<14} {b.open:>10.2f} {b.high:>10.2f} {b.low:>10.2f} {b.close:>10.2f} {b.volume:>10.0f}")

    # Identify which bar produced IBH and IBL
    ibh_bar = max(ib_bars, key=lambda b: b.high)
    ibl_bar = min(ib_bars, key=lambda b: b.low)
    print(f"\n  IBH {m.ib_high:.2f} from bar at {ibh_bar.timestamp.strftime('%H:%M')}")
    print(f"  IBL {m.ib_low:.2f} from bar at {ibl_bar.timestamp.strftime('%H:%M')}")

    # Identify which bar produced RTH high and low
    rth_high_bar = max(rth_bars, key=lambda b: b.high)
    rth_low_bar = min(rth_bars, key=lambda b: b.low)
    print(f"  RTH High {m.rth_high:.2f} from bar at {rth_high_bar.timestamp.strftime('%H:%M')}")
    print(f"  RTH Low  {m.rth_low:.2f} from bar at {rth_low_bar.timestamp.strftime('%H:%M')}")

    print(f"\n{sep}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify IBH/IBL/RTH_high/RTH_low for a session."
    )
    parser.add_argument(
        "--csv",
        default="MNQ_1min_2023Jan_2026Jan.csv",
        help="Path to the MNQ minute data CSV.",
    )
    parser.add_argument(
        "--target-date",
        type=parse_date,
        default=None,
        help="Session date to verify (YYYY-MM-DD). Omit for last session.",
    )
    parser.add_argument(
        "--rth-start",
        type=parse_time,
        default=DEFAULTS["rth_start"],
    )
    parser.add_argument(
        "--rth-end",
        type=parse_time,
        default=DEFAULTS["rth_end"],
    )
    parser.add_argument(
        "--ib-start",
        type=parse_time,
        default=DEFAULTS["ib_start"],
    )
    parser.add_argument(
        "--ib-end",
        type=parse_time,
        default=DEFAULTS["ib_end"],
    )
    args = parser.parse_args()

    m, bars = find_session(
        csv_path=args.csv,
        target_date=args.target_date,
        start_date=None,
        end_date=None,
        rth_start=args.rth_start,
        rth_end=args.rth_end,
        ib_start=args.ib_start,
        ib_end=args.ib_end,
        opening_window_minutes=DEFAULTS["opening_window_minutes"],
    )

    if m is None:
        raise SystemExit(
            f"No session found"
            + (f" for {args.target_date}." if args.target_date else ".")
        )

    print_session_report(m, bars, args.rth_start, args.rth_end, args.ib_start, args.ib_end)


if __name__ == "__main__":
    main()
