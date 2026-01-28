# MNQ Initial Balance Metrics Documentation

This document explains how each output column in `mp2b_IBH_IBL.py` is calculated.
Unless otherwise stated, calculations are performed **per session day** using only
**Regular Trading Hours (RTH)** bars between `rth_start` and `rth_end` (inclusive).

## Time Windows

- **RTH window**: `[rth_start, rth_end]`
- **IB window**: `[ib_start, ib_end]` (must fall within RTH)
- **Opening window**: `opening_window_minutes` starting at `rth_start`

## Output Fields and Calculations

### Session Identifiers

- **session_date**: `date` portion of the first RTH bar for the session.
- **rth_start / rth_end**: The configured RTH window, rendered as `HH:MM`.

### Initial Balance (IB) Metrics

Computed from bars where `ib_start <= bar.time <= ib_end` (within RTH).

- **ib_high**: Maximum `high` in the IB window.
- **ib_low**: Minimum `low` in the IB window.
- **ib_range**: `ib_high - ib_low`.
- **midpoint**: `(ib_high + ib_low) / 2`.
- **ib_volume**: Sum of `volume` over IB window bars.

### RTH Session Metrics

Computed from all RTH bars where `rth_start <= bar.time <= rth_end`.

- **total_volume**: Sum of `volume` over RTH bars.
- **relative_ib_volume**: `ib_volume / total_volume` (0 if total volume is 0).
- **rth_high**: Maximum `high` over RTH bars.
- **rth_low**: Minimum `low` over RTH bars.
- **rth_close**: `close` of the last RTH bar.

### Discovery Extensions

Using RTH high/low and IB range:

- **extension_up**: `max(0, rth_high - ib_high)`
- **extension_down**: `max(0, ib_low - rth_low)`
- **extension_1_5_up**: `ib_high + 1.5 * ib_range`
- **extension_2_up**: `ib_high + 2.0 * ib_range`
- **extension_1_5_down**: `ib_low - 1.5 * ib_range`
- **extension_2_down**: `ib_low - 2.0 * ib_range`
- **reached_1_5_up**: `rth_high >= extension_1_5_up`
- **reached_2_up**: `rth_high >= extension_2_up`
- **reached_1_5_down**: `rth_low <= extension_1_5_down`
- **reached_2_down**: `rth_low <= extension_2_down`

### Rotation

Rotation means price touched **both** IBH and IBL **after** the IB window ends.

- `after_ib` bars are those where `bar.time > ib_end` (still within RTH).
- **rotation**: `True` if any `after_ib` bar has `high >= ib_high` **and**
  any `after_ib` bar has `low <= ib_low`. Otherwise `False`.

### Failed Auction

Uses RTH extreme and close vs. IB levels:

- **failed_high**: `rth_high > ib_high` **and** `rth_close <= ib_high`
- **failed_low**: `rth_low < ib_low` **and** `rth_close >= ib_low`
- **failed_auction**:
  - `"failed_both"` if `failed_high` and `failed_low`
  - `"failed_high"` if only `failed_high`
  - `"failed_low"` if only `failed_low`
  - `"none"` otherwise

### Breakside (First IB Touch After IB Window)

Scan `after_ib` bars in time order and stop at the first touch:

- `"both"` if the first qualifying bar touches both `ib_high` **and** `ib_low`
- `"high"` if the first qualifying bar touches only `ib_high`
- `"low"` if the first qualifying bar touches only `ib_low`
- `"none"` if no `after_ib` bar touches either extreme

### Opening Range Metrics

Opening window bars are those where:
`rth_start <= bar.time < rth_start + opening_window_minutes`.

- **opening_window_minutes**: Configured window size.
- **opening_range_high**: Max `high` over opening window bars (or `None` if missing).
- **opening_range_low**: Min `low` over opening window bars (or `None` if missing).
- **opening_range**: `opening_range_high - opening_range_low` (or `None`).
- **opening_range_close**: `close` of the last opening-window bar (or `None`).
- **opening_direction**:
  - `"up"` if `opening_range_close > opening_window_first_open`
  - `"down"` if `opening_range_close < opening_window_first_open`
  - `"flat"` otherwise
- **opening_type**:
  - `"drive"` when:
    - `abs(opening_move) >= 0.6 * opening_range`, **and**
    - close location is `>= 1.05` or `<= 0.05` of the opening range
  - `"auction"` otherwise

Where:
- **opening_move**: `opening_range_close - opening_window_first_open`
- **close location**: `(opening_range_close - opening_range_low) / opening_range`

### Opening Bar Metrics (First RTH Minute)

These are derived from the bar whose timestamp equals `rth_start`:

- **opening_bar_open**: `open` of the first RTH bar (or `None` if missing).
- **opening_bar_close**: `close` of the first RTH bar (or `None` if missing).
- **opening_bar_open_close**: `opening_bar_close - opening_bar_open` (or `None`).
- **opening_bar_volume**: `volume` of the first RTH bar (or `None`).
