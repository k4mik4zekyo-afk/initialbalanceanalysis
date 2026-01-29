# Phase II Prior-Day Level Columns

This document explains the columns appended by `mp2a_previous_day_levels.py` when it
joins prior-day reference levels to the Phase I IB metrics output. All fields are
computed per session day and are derived strictly from *completed* prior sessions
to avoid forward-looking data leakage.【F:mp2a_previous_day_levels.py†L132-L219】

## Prior-Day Reference Levels (Inputs)

These levels are computed for each session day using bars within the configured
session window (defaults to 06:30–16:00). The script then shifts them so that each
current session receives *the previous session's* levels; the very first session
in the dataset has no prior-day levels and therefore receives `None`.【F:mp2a_previous_day_levels.py†L132-L219】

- **prev_pdh**: Previous day high, computed as the maximum `high` within the
  prior session window.【F:mp2a_previous_day_levels.py†L155-L174】
- **prev_pdl**: Previous day low, computed as the minimum `low` within the
  prior session window.【F:mp2a_previous_day_levels.py†L155-L174】
- **prev_vah**: Previous day value area high. The value area is derived from a
  volume profile built on *typical price* `((high + low + close) / 3)` rounded to
  the configured tick size. Starting from the POC, the profile expands up/down
  (choosing the higher-volume side next) until it reaches the configured
  `value_area_pct` of total session volume; the resulting upper price is `VAH`.【F:mp2a_previous_day_levels.py†L83-L131】
- **prev_val**: Previous day value area low, computed alongside `prev_vah` using
  the same expansion logic; the resulting lower price is `VAL`.【F:mp2a_previous_day_levels.py†L83-L131】
- **prev_poc**: Previous day point of control, defined as the price bin in the
  volume profile with the highest volume.【F:mp2a_previous_day_levels.py†L93-L131】
- **prev_session_volume**: Total volume over the prior session window, used to
  compute the volume profile and value area cutoffs.【F:mp2a_previous_day_levels.py†L83-L174】

## Opening Range Helper

- **opening_range_midpoint**: Midpoint of the current day opening range,
  `(opening_range_high + opening_range_low) / 2`. If either opening bound is
  missing, this value is `None`.【F:mp2a_previous_day_levels.py†L265-L297】

## Touch / Breach / Close-Relation Columns

All *touch*, *breach*, and *close relation* checks are computed using the
configured `level_tolerance` (default 0.25 points).【F:mp2a_previous_day_levels.py†L35-L41】【F:mp2a_previous_day_levels.py†L186-L236】

### Touch Columns

Definition: `True` when the relevant range overlaps the level within tolerance,
i.e., `high >= level - tol` **and** `low <= level + tol`.

- **opening_touch_{level}**: Touch using the opening range `high/low`
  (`opening_range_high`, `opening_range_low`).【F:mp2a_previous_day_levels.py†L265-L326】
- **rth_touch_{level}**: Touch using the RTH session range
  (`rth_high`, `rth_low`).【F:mp2a_previous_day_levels.py†L265-L326】

Valid `{level}` suffixes: `pdh`, `pdl`, `vah`, `val`, `poc`.

### Breach Columns

Definition:
- **breach_above**: `high > level + tol`
- **breach_below**: `low < level - tol`

Columns:
- **rth_breach_above_{level}**: Breach above using `rth_high`.【F:mp2a_previous_day_levels.py†L265-L333】
- **rth_breach_below_{level}**: Breach below using `rth_low`.【F:mp2a_previous_day_levels.py†L265-L333】

Valid `{level}` suffixes: `pdh`, `pdl`, `vah`, `val`, `poc`.

### Close-Relation Columns

Definition: Classification of the close relative to the level with tolerance.
- **above**: `close > level + tol`
- **below**: `close < level - tol`
- **within**: otherwise

Columns:
- **rth_close_vs_{level}**: Uses `rth_close`.【F:mp2a_previous_day_levels.py†L265-L333】
- **opening_close_vs_{level}**: Uses `opening_range_close`.【F:mp2a_previous_day_levels.py†L265-L333】

Valid `{level}` suffixes: `pdh`, `pdl`, `vah`, `val`, `poc`.

## Confluence Columns

Definition: `True` when the absolute distance between two levels is within the
configured tolerance: `abs(level_a - level_b) <= tol`.【F:mp2a_previous_day_levels.py†L201-L236】

- **confluence_{level}_ibh**: Prior-day `{level}` aligns with `ib_high`.
- **confluence_{level}_ibl**: Prior-day `{level}` aligns with `ib_low`.

Valid `{level}` suffixes: `pdh`, `pdl`, `vah`, `val`, `poc`.【F:mp2a_previous_day_levels.py†L265-L347】

Additional convenience aliases:
- **confluence_vah_ibh**: Same as `confluence_vah_ibh` (explicitly re-set).
- **confluence_val_ibl**: Same as `confluence_val_ibl` (explicitly re-set).
- **confluence_pdl_val**: Confluence between `prev_pdl` and `prev_val`.
- **confluence_pdh_vah**: Confluence between `prev_pdh` and `prev_vah`.【F:mp2a_previous_day_levels.py†L328-L353】

## Discovery / Balance Columns

- **discovery_up**: `True` when `rth_high > ib_high`; `False` otherwise. (If
  either input is missing, this is `None`).【F:mp2a_previous_day_levels.py†L355-L360】
- **discovery_down**: `True` when `rth_low < ib_low`; `False` otherwise. (If
  either input is missing, this is `None`).【F:mp2a_previous_day_levels.py†L359-L360】
- **balance_day**: `True` when *neither* discovery condition is met; `False` if
  any discovery occurs; `None` if discovery inputs are missing.【F:mp2a_previous_day_levels.py†L361-L366】

## Rotation Depth Columns

Rotation depth normalizes extension magnitude by the IB range:

- **rotation_depth_up**: `extension_up / ib_range` when both values exist and
  `ib_range != 0`; otherwise `None`.【F:mp2a_previous_day_levels.py†L367-L373】
- **rotation_depth_down**: `extension_down / ib_range` when both values exist and
  `ib_range != 0`; otherwise `None`.【F:mp2a_previous_day_levels.py†L373-L377】

## Nearest Prior Level to the Open

- **nearest_prior_level_to_open**: The name (`pdh`, `pdl`, `vah`, `val`, or `poc`)
  of the prior-day level closest to `opening_range_midpoint`.
- **nearest_prior_level_to_open_distance**: The absolute distance between
  `opening_range_midpoint` and that nearest prior level.

If the opening midpoint is missing, or if no prior levels exist, both fields are
`None`.【F:mp2a_previous_day_levels.py†L219-L248】【F:mp2a_previous_day_levels.py†L379-L382】
