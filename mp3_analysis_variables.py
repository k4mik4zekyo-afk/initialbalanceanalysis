# -*- coding: utf-8 -*-
"""
Decision-tree feature extraction from two consecutive sessions of 1-minute bars.

Reuses the IB and prior-day level algorithms from mp2b_IBH_IBL.py and
mp2a_previous_day_levels.py to compute a compact feature vector:

    [[relative_ib_volume, normalized_distance, opening_bar_open_close,
      opening_bar_volume, prev_session_volume]]

Intended for direct consumption by a sklearn-style decision tree via
``model.predict(features)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np


DATE_FORMAT = "%m/%d/%y %H:%M"

DEFAULTS = {
    "csv": "MNQ_1min_2023Jan_2026Jan.csv",
    "rth_start": time(6, 30),
    "rth_end": time(13, 0),
    "ib_start": time(6, 30),
    "ib_end": time(7, 30),
    "session_start": time(6, 30),
    "session_end": time(14, 0),
    "tick_size": 0.25,
    "value_area_pct": 0.7,
}

FEATURE_NAMES = [
    "relative_ib_volume",
    "normalized_distance",
    "opening_bar_open_close",
    "opening_bar_volume",
    "prev_session_volume",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# CSV parsing (identical to mp2b / mp2a)
# ---------------------------------------------------------------------------

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


def _detect_csv_format(header_line: str) -> str:
    """Return ``'tradingview'`` or ``'legacy'`` based on the CSV header."""
    lower = header_line.lower().lstrip("\ufeff").strip()
    if lower.startswith("time,"):
        return "tradingview"
    return "legacy"


def iter_bars(csv_path: str, tz_name: str = "America/Los_Angeles") -> Iterator[Bar]:
    with open(csv_path, newline="") as handle:
        header = handle.readline()
        handle.seek(0)
        fmt = _detect_csv_format(header)
        reader = csv.DictReader(handle)

        if fmt == "tradingview":
            target_tz = ZoneInfo(tz_name)
            for row in reader:
                raw_dt = row.get("time") or row.get("\ufefftime")
                if not raw_dt:
                    continue
                aware_dt = datetime.fromisoformat(raw_dt.strip())
                local_dt = aware_dt.astimezone(target_tz)
                naive_dt = local_dt.replace(tzinfo=None)
                yield Bar(
                    timestamp=naive_dt,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("Volume", 0) or 0),
                )
        else:
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


def collect_two_days(
    bars: Iterable[Bar],
    target_date: date,
) -> Tuple[List[Bar], List[Bar]]:
    """Return bars for the calendar day *before* target_date and target_date."""
    prev_date = target_date - timedelta(days=1)
    prev_bars: List[Bar] = []
    target_bars: List[Bar] = []

    for bar in bars:
        bar_date = bar.timestamp.date()
        if bar_date < prev_date:
            continue
        if bar_date > target_date:
            break
        if bar_date == prev_date:
            prev_bars.append(bar)
        elif bar_date == target_date:
            target_bars.append(bar)

    return prev_bars, target_bars


def collect_two_sessions(
    bars: Iterable[Bar],
    target_date: date,
) -> Tuple[List[Bar], List[Bar]]:
    """Return bars for the two most recent session days up to *target_date*.

    Unlike ``collect_two_days`` this does not assume the prior session is
    exactly one calendar day earlier — it simply takes the last two distinct
    dates with data on or before *target_date*.
    """
    by_date: Dict[date, List[Bar]] = {}
    for bar in bars:
        bar_date = bar.timestamp.date()
        if bar_date > target_date:
            break
        by_date.setdefault(bar_date, []).append(bar)

    dates = sorted(by_date.keys())
    if not dates:
        return [], []
    if len(dates) < 2:
        return [], by_date[dates[-1]]

    # Last two session dates up to and including target_date
    return by_date[dates[-2]], by_date[dates[-1]]


def collect_last_two_sessions(
    bars: Iterable[Bar],
    rth_start: time = DEFAULTS["rth_start"],
    rth_end: time = DEFAULTS["rth_end"],
) -> Tuple[List[Bar], List[Bar]]:
    """Return bars for the last two full session dates in the file.

    A date only counts as a valid session if it contains at least one bar
    within the RTH window, so partial/overnight-only days are skipped.
    """
    by_date: Dict[date, List[Bar]] = {}
    for bar in bars:
        by_date.setdefault(bar.timestamp.date(), []).append(bar)

    # Keep only dates that have RTH-eligible bars
    valid_dates = sorted(
        d for d, day_bars in by_date.items()
        if any(rth_start <= b.timestamp.time() <= rth_end for b in day_bars)
    )
    if not valid_dates:
        return [], []
    if len(valid_dates) < 2:
        return [], by_date[valid_dates[-1]]

    return by_date[valid_dates[-2]], by_date[valid_dates[-1]]


# ---------------------------------------------------------------------------
# Phase I helpers (from mp2b_IBH_IBL.py algorithm)
# ---------------------------------------------------------------------------

def compute_ib_features(
    bars: List[Bar],
    rth_start: time,
    rth_end: time,
    ib_start: time,
    ib_end: time,
) -> Optional[Dict[str, Optional[float]]]:
    """Compute the Phase I features needed for the feature vector."""
    rth_bars = [
        bar for bar in bars if rth_start <= bar.timestamp.time() <= rth_end
    ]
    if not rth_bars:
        return None

    ib_bars = [
        bar for bar in rth_bars if ib_start <= bar.timestamp.time() <= ib_end
    ]
    if not ib_bars:
        return None

    ib_high = max(bar.high for bar in ib_bars)
    ib_low = min(bar.low for bar in ib_bars)
    ib_range = ib_high - ib_low

    ib_volume = sum(bar.volume for bar in ib_bars)
    total_volume = sum(bar.volume for bar in rth_bars)
    relative_ib_volume = ib_volume / total_volume if total_volume else 0.0

    # Opening bar (first RTH minute)
    opening_bar = next(
        (bar for bar in rth_bars if bar.timestamp.time() == rth_start),
        None,
    )
    opening_bar_open_close = (
        opening_bar.close - opening_bar.open if opening_bar else None
    )
    opening_bar_volume = opening_bar.volume if opening_bar else None

    # Opening range midpoint (first 10 minutes, reuse mp2b logic)
    opening_start_dt = datetime.combine(rth_bars[0].timestamp.date(), rth_start)
    opening_end_dt = opening_start_dt + timedelta(minutes=10)
    opening_window_bars = [
        bar for bar in rth_bars
        if opening_start_dt <= bar.timestamp < opening_end_dt
    ]
    opening_midpoint = None
    if opening_window_bars:
        opening_high = max(bar.high for bar in opening_window_bars)
        opening_low = min(bar.low for bar in opening_window_bars)
        opening_midpoint = (opening_high + opening_low) / 2

    return {
        "relative_ib_volume": relative_ib_volume,
        "ib_range": ib_range,
        "opening_bar_open_close": opening_bar_open_close,
        "opening_bar_volume": opening_bar_volume,
        "opening_midpoint": opening_midpoint,
    }


# ---------------------------------------------------------------------------
# Phase II helpers (from mp2a_previous_day_levels.py algorithm)
# ---------------------------------------------------------------------------

def compute_volume_profile(
    bars: List[Bar], tick_size: float
) -> Tuple[Dict[float, float], float]:
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

    return prices[upper_index], prices[lower_index], poc


def compute_prior_session(
    bars: List[Bar],
    session_start: time,
    session_end: time,
    tick_size: float,
    value_area_pct: float,
) -> Optional[Dict[str, Optional[float]]]:
    """Compute prior-day levels and session volume from the previous session bars."""
    session_bars = [
        bar for bar in bars
        if session_start <= bar.timestamp.time() <= session_end
    ]
    if not session_bars:
        return None

    pdh = max(bar.high for bar in session_bars)
    pdl = min(bar.low for bar in session_bars)
    profile, total_volume = compute_volume_profile(session_bars, tick_size)
    vah, val, poc = compute_value_area(profile, total_volume, value_area_pct)

    return {
        "pdh": pdh,
        "pdl": pdl,
        "vah": vah,
        "val": val,
        "poc": poc,
        "session_volume": total_volume,
    }


def nearest_level_distance(
    opening_midpoint: Optional[float],
    prior_levels: Dict[str, Optional[float]],
) -> Optional[float]:
    """Absolute distance from the opening midpoint to the nearest prior level."""
    if opening_midpoint is None:
        return None
    distances = []
    for value in prior_levels.values():
        if value is not None:
            distances.append(abs(opening_midpoint - value))
    return min(distances) if distances else None


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def build_feature_vector(
    prev_bars: List[Bar],
    target_bars: List[Bar],
    rth_start: time,
    rth_end: time,
    ib_start: time,
    ib_end: time,
    session_start: time,
    session_end: time,
    tick_size: float,
    value_area_pct: float,
) -> Optional[List[Optional[float]]]:
    """Build ``[relative_ib_volume, normalized_distance, opening_bar_open_close,
    opening_bar_volume, prev_session_volume]`` from two sessions of bars."""

    ib_feats = compute_ib_features(
        target_bars, rth_start, rth_end, ib_start, ib_end
    )
    if ib_feats is None:
        return None

    prior = compute_prior_session(
        prev_bars, session_start, session_end, tick_size, value_area_pct
    )

    prev_session_volume = prior["session_volume"] if prior else None

    # Normalized distance: nearest-prior-level distance / ib_range
    normalized_distance = None
    if prior and ib_feats["opening_midpoint"] is not None:
        prior_levels = {
            k: prior[k] for k in ("pdh", "pdl", "vah", "val", "poc")
        }
        raw_distance = nearest_level_distance(
            ib_feats["opening_midpoint"], prior_levels
        )
        ib_range = ib_feats["ib_range"]
        if raw_distance is not None and ib_range and ib_range > 0:
            normalized_distance = raw_distance / ib_range

    return [
        ib_feats["relative_ib_volume"],
        normalized_distance,
        ib_feats["opening_bar_open_close"],
        ib_feats["opening_bar_volume"],
        prev_session_volume,
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract decision-tree features from two sessions of 1-min bars."
        ),
    )
    parser.add_argument(
        "--csv",
        default=DEFAULTS["csv"],
        help="Path to the MNQ minute data CSV.",
    )
    parser.add_argument(
        "--target-date",
        type=parse_date,
        default=None,
        help=(
            "Session date to compute features for (YYYY-MM-DD). "
            "Omit to auto-detect the last two sessions in the CSV."
        ),
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
        "--session-start",
        type=parse_time,
        default=DEFAULTS["session_start"],
        help="Session start for prior-day level computation (HH:MM).",
    )
    parser.add_argument(
        "--session-end",
        type=parse_time,
        default=DEFAULTS["session_end"],
        help="Session end for prior-day level computation (HH:MM).",
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
        help="Value area percent for VAH/VAL (e.g. 0.7 for 70%%).",
    )
    parser.add_argument(
        "--tz",
        default="America/Los_Angeles",
        help=(
            "IANA timezone for converting TradingView ISO timestamps "
            "(default: America/Los_Angeles).  Ignored for legacy CSVs."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file path. Writes JSON. Omit to print to stdout.",
    )
    parser.add_argument(
        "--model",
        default="decision_tree_model.pkl",
        help="Path to the trained decision tree model pickle file.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path. Writes full output to this file.",
    )
    return parser


def _output(text: str, log_handle=None) -> None:
    """Print to stdout and optionally write to a log file."""
    print(text)
    if log_handle is not None:
        log_handle.write(text + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.rth_end <= args.rth_start:
        raise SystemExit("RTH end time must be after RTH start time.")
    if args.ib_end <= args.ib_start:
        raise SystemExit("IB end time must be after IB start time.")
    if args.ib_start < args.rth_start or args.ib_end > args.rth_end:
        raise SystemExit("IB window must fall within the RTH window.")
    if args.session_end <= args.session_start:
        raise SystemExit("Session end time must be after session start time.")
    if args.tick_size <= 0:
        raise SystemExit("Tick size must be positive.")
    if not 0 < args.value_area_pct <= 1:
        raise SystemExit("Value area percent must be between 0 and 1.")

    # Open log file if requested
    log_handle = None
    if args.log_file:
        log_handle = open(args.log_file, "w")

    try:
        _output("=" * 60, log_handle)
        _output("  Decision Tree Feature Extraction & Prediction", log_handle)
        _output("=" * 60, log_handle)

        # --- Load model ---
        if not os.path.isfile(args.model):
            raise SystemExit(f"Model file not found: {args.model}")
        with open(args.model, "rb") as f:
            model = pickle.load(f)
        _output(f"\nModel loaded: {args.model}", log_handle)

        # --- Collect bars ---
        if args.target_date is not None:
            prev_bars, target_bars = collect_two_sessions(
                iter_bars(args.csv, tz_name=args.tz), args.target_date
            )
        else:
            prev_bars, target_bars = collect_last_two_sessions(
                iter_bars(args.csv, tz_name=args.tz), args.rth_start, args.rth_end
            )

        if not target_bars:
            raise SystemExit(
                "No bars found"
                + (f" for target date {args.target_date}." if args.target_date else
                   " in the CSV.")
            )
        if not prev_bars:
            print(
                "Warning: no prior session bars found; prev_session_volume and "
                "normalized_distance will be None.",
                file=sys.stderr,
            )

        target_session_date = target_bars[0].timestamp.date()
        _output(f"Target session date: {target_session_date}", log_handle)

        # --- Build features ---
        features = build_feature_vector(
            prev_bars,
            target_bars,
            rth_start=args.rth_start,
            rth_end=args.rth_end,
            ib_start=args.ib_start,
            ib_end=args.ib_end,
            session_start=args.session_start,
            session_end=args.session_end,
            tick_size=args.tick_size,
            value_area_pct=args.value_area_pct,
        )

        if features is None:
            raise SystemExit(
                f"Could not compute features for {args.target_date} "
                f"(missing RTH or IB bars)."
            )

        # --- Check for None inputs ---
        _output("\nFeatures:", log_handle)
        has_none = False
        for name, value in zip(FEATURE_NAMES, features):
            _output(f"  {name}: {value}", log_handle)
            if value is None:
                has_none = True

        if has_none:
            _output("\nNone as input", log_handle)
            _output("Cannot run prediction with missing feature values.", log_handle)
            if args.output:
                result = {
                    "feature_names": FEATURE_NAMES,
                    "features": [features],
                    "error": "None as input",
                }
                with open(args.output, "w") as handle:
                    handle.write(json.dumps(result, indent=2) + "\n")
            raise SystemExit(1)

        # --- Predict ---
        import pandas as pd

        feature_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        prediction = model.predict(feature_df)[0]
        probabilities = model.predict_proba(feature_df)[0]
        class_labels = model.classes_

        label = "Rotation" if prediction else "No Rotation"
        _output(f"\n{'=' * 60}", log_handle)
        _output(f"  Prediction: {label}", log_handle)
        _output(f"{'=' * 60}", log_handle)
        _output(f"\n  Probabilities:", log_handle)
        for cls, prob in zip(class_labels, probabilities):
            cls_label = "Rotation" if cls else "No Rotation"
            _output(f"    {cls_label}: {prob:.4f}", log_handle)

        # --- Build result ---
        result = {
            "feature_names": FEATURE_NAMES,
            "features": [features],
            "prediction": label,
            "prediction_raw": bool(prediction),
            "probabilities": {
                "Rotation": float(probabilities[list(class_labels).index(True)]),
                "No Rotation": float(probabilities[list(class_labels).index(False)]),
            },
        }

        payload = json.dumps(result, indent=2)
        _output(f"\nJSON output:\n{payload}", log_handle)

        if args.output:
            with open(args.output, "w") as handle:
                handle.write(payload + "\n")
            _output(f"\nSaved features + prediction to {args.output}", log_handle)

    finally:
        if log_handle is not None:
            log_handle.close()
            print(f"Log written to {args.log_file}")


if __name__ == "__main__":
    main()
