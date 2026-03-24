# New Features Added to Phase II Data

## Date: February 4, 2026

### 1. relative_ib2_volume
**Definition**: Current day's Initial Balance (IB) volume divided by the previous day's IB volume.

**Formula**: `relative_ib2_volume = current_day_ib_volume / previous_day_ib_volume`

**Purpose**: Measures relative activity level during the Initial Balance period compared to the previous session's IB period. This can indicate:
- Increased volatility/interest (ratio > 1.0)
- Decreased participation (ratio < 1.0)
- Potential changes in market regime

**Example Values**:
- 2025-01-21: ratio = 3.17 (very high IB activity vs previous day)
- 2025-01-20: ratio = 0.29 (very low IB activity vs previous day)

**Note**: First session in dataset will have empty/null value as there is no previous IB volume.

---

### 2. high_impact_during_rth
**Definition**: Boolean flag indicating whether a "High Impact" economic event occurred during Regular Trading Hours (RTH).

**RTH Hours**: 06:30 - 14:00 Pacific Time

**Data Source**: `Jan01_2025_December31_2025_events.csv`

**Criteria**: Event marked as "High Impact Expected" in the events CSV that occurs between RTH start and end times.

**Purpose**: Identifies trading sessions that may experience:
- Increased volatility
- Price discovery around economic releases
- Potential for larger directional moves
- Changed market dynamics compared to "quiet" sessions

**Example Events Flagged**:
- 2025-01-20: President Trump Speaks (09:00)
- 2025-01-21: JOLTS Job Openings, ISM Services PMI (07:00)
- 2025-01-23: Core Durable Goods Orders, Unemployment Claims (05:30)

**Implementation**: Events file is parsed to extract all high impact events, filter by RTH hours, and create a set of dates for fast lookup.

---

## Files Modified

### mp2a_previous_day_levels.py
- Added `load_high_impact_events()` function to parse events CSV
- Modified `add_interactions()` to calculate both new features
- Updated main loop to track `prev_ib_volume` across iterations
- Added `--events` argument (default: Jan01_2025_December31_2025_events.csv)
- Added both features to output CSV fieldnames

### Generated Output Files
All Phase II CSV files now include these two additional columns:
- `outputs/phase2_previous_day_levels.csv`
- `outputs/phase2_10min.csv`
- `outputs/phase2_15min.csv`

---

## Usage in Feature Engineering

Both features can now be used in machine learning models to predict rotation patterns:

```python
# Example usage in feature engineering
features = [
    'relative_ib_vol_pdv',        # IB volume / prev session volume (existing, corrected for no leakage)
    'relative_ib2_volume',        # IB volume / prev IB volume (NEW)
    'nearest_level_poc',
    'nearest_level_vah',
    'previous_day_confluences',
    'high_impact_during_rth'      # Boolean: high impact event in RTH (NEW)
]
```

**Hypothesis**: Sessions with high impact events may have different rotation behavior, and relative IB activity compared to previous day's IB may indicate changes in market participation that affect rotation probability.
