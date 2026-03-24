# MNQ Initial Balance & Rotation Prediction

A data pipeline and machine learning system that analyzes **Micro Nasdaq-100 (MNQ) futures** market structure to predict whether a trading session will **rotate** (price touches both the Initial Balance High and Low) or **continue** (price trends in one direction).

## What is Initial Balance?

The **Initial Balance (IB)** is the price range established during the first hour of Regular Trading Hours (6:30–7:30am PT). Traders use it as a reference for gauging the day's likely behavior — rotation back through the range, or continuation/breakout beyond it.

## Pipeline Overview

```
Raw 1-min MNQ bars (2023–2026)
        │
        ▼
┌──────────────────────┐
│  Phase 1: mp2b       │  Compute IB high/low, range, rotation,
│  IBH_IBL.py          │  failed auctions, opening type
│                      │  → outputs/ib_metrics.csv
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Phase 2: mp2a       │  Add prior-day levels (PDH, PDL, VAH,
│  previous_day_       │  VAL, POC), level interactions, news
│  levels.py           │  events, relative volume
│                      │  → outputs/phase2_previous_day_levels.csv
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Phase 3: Notebook   │  Feature engineering, model training
│  feature_engineering  │  (RandomForest), cross-validation
│  _and_model_         │  → best_model.joblib
│  evaluation.ipynb    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Prediction Pipeline │  Load trained model, predict rotation
│  rotation_prediction │  probability for new sessions
│  _pipeline.ipynb     │
└──────────────────────┘
```

## Key Scripts

| File | Purpose |
|------|---------|
| `mp2b_IBH_IBL.py` | **Phase 1** — Computes initial balance metrics from raw 1-min bars. Identifies rotation, failed auctions, breakside, opening type. **Stable/locked.** |
| `mp2a_previous_day_levels.py` | **Phase 2** — Adds prior-day context: PDH, PDL, Value Area (VAH/VAL), POC, news events, relative volume. |
| `mp3_analysis_variables.py` | **Phase 3** — Feature extraction and ML training (script version). |
| `feature_engineering_and_model_evaluation.ipynb` | **Phase 3** — Interactive model evaluation, feature importance analysis, cross-validation. |
| `rotation_prediction_pipeline.ipynb` | Live/forward prediction using the trained model. |
| `verify_session.py` | Utility to spot-check computed metrics against chart data. |

## Features Engineered

The model uses features available **before or at the start** of the trading session to avoid data leakage:

| Feature | Description |
|---------|-------------|
| `relative_ib_vol_pdv` | IB volume / previous day's total RTH volume |
| `normalized_distance` | Distance to nearest prior-day level / previous day's range |
| `norm_distance_by_atr` | Normalized distance scaled by ATR |
| `nearest_prior_level_to_open` | Which level (PDH/PDL/VAH/VAL/POC) is closest to open — one-hot encoded |
| `norm_opening_volatility` | ATR(14) at 6:45am / 5-day average prior ATR(14) |
| `norm_opening_bar_volume` | First-minute volume / 10-day average |
| `news_event_during_rth` | Binary flag for high-impact USD economic events |
| `opening_bar_open_close` | First RTH minute price change |

## Model Results

### Best Configuration

**10-minute opening window + Random Forest (100 trees, max depth 5) + Normalized volume features**

- **Target:** Binary classification — rotation (1) vs. continuation (0)
- **Class distribution:** ~21% rotation, ~79% continuation (imbalanced)
- **Data:** 256 trading sessions from 2025

### Cross-Validation Performance (Stratified 5-Fold)

| Model | F1 | Notes |
|-------|-----|-------|
| **Random Forest (10-min)** | **0.451 ± 0.022** | Best overall |
| Decision Tree (10-min) | 0.420 ± 0.032 | |
| Decision Tree (15-min) | 0.415 ± 0.033 | |
| Random Forest (15-min) | 0.407 ± 0.026 | |

### Within-Year Stability

| Period | Sessions | Rotation % | F1 |
|--------|----------|------------|-----|
| 2023 | 248 | 22.6% | 0.358 ± 0.166 |
| 2024 | 251 | 19.9% | 0.482 ± 0.040 |
| 2025 | 256 | 20.7% | 0.451 ± 0.042 |

### Walk-Forward Validation (Train Year N → Test Year N+1)

| Split | F1 | Precision | Recall | Accuracy |
|-------|-----|-----------|--------|----------|
| 2023 → 2024 | 0.477 | 0.363 | 0.698 | 0.687 |
| 2024 → 2025 | 0.352 | 0.306 | 0.415 | 0.684 |
| 2023-24 → 2025 | 0.400 | 0.329 | 0.509 | 0.684 |

### Impact of Feature Normalization

| Metric | Before (raw volumes) | After (normalized) | Change |
|--------|---------------------|-------------------|--------|
| Best CV F1 | 0.444 ± 0.021 | **0.451 ± 0.022** | +1.6% |
| 2025 within-year F1 | 0.369 ± 0.106 | **0.451 ± 0.042** | **+22%** |
| Walk-forward 2024→2025 | 0.305 | **0.352** | **+15%** |

### Key Findings

1. **Normalizing volume features by rolling averages eliminated year-over-year drift**, boosting 2025 F1 by 22%.
2. **Walk-forward generalization improved** — training on 2024 and testing on 2025 went from F1=0.305 to 0.352.
3. **10-minute opening window outperforms 15-minute** with normalized features (F1 0.451 vs 0.407).
4. **2023 data is noisy** (F1 0.358 ± 0.166), likely due to different market dynamics or insufficient warm-up for rolling averages.
5. **Model selected 4 key features:** `relative_ib_vol_pdv`, `normalized_distance`, `norm_distance_by_atr`, `nearest_level_pdh`.

## Data

- **Input:** `MNQ_1min_2023Jan_2026Jan.csv` — 1-minute OHLCV bars (~62MB, not committed)
- **News events:** `Jan01_2025_December31_2025_events.csv` — Forex Factory economic calendar
- **Timezone:** Raw data in PST; code converts to `America/Los_Angeles` for DST handling
- **RTH window:** 6:30am–1:00pm PT

## Outputs

| File | Description |
|------|-------------|
| `outputs/ib_metrics.csv` | One row per session with IB metrics (~800 days) |
| `outputs/phase2_previous_day_levels.csv` | Full feature set with prior-day context |
| `best_model.joblib` | Trained RandomForest model |
| `misclassified_samples.csv` | Error analysis of misclassified sessions |

## Special Handling

- **Holiday adjustments:** A centralized dictionary skips holidays and contract rollover days in lookback calculations (volume averages, ATR). See `HOLIDAY_ADJUSTMENTS.md`.
- **Data leakage prevention:** All features use `.shift(1)` or prior-day data only. Volume normalization uses previous session, not current. ATR is computed at 6:45am (before IB window closes).

## Running the Pipeline

```bash
# Phase 1 — compute IB metrics
python mp2b_IBH_IBL.py

# Phase 2 — add prior-day levels and context
python mp2a_previous_day_levels.py

# Phase 3 — open notebook for feature engineering & model training
jupyter notebook feature_engineering_and_model_evaluation.ipynb

# Prediction — run predictions on new data
jupyter notebook rotation_prediction_pipeline.ipynb
```

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, joblib
- xgboost (optional)
- jupyter (for notebooks)

## Open Questions

1. Which prior-day level best predicts rotation depth?
2. Does acceptance beyond VAH/VAL increase discovery probability?
3. How stable are tendencies across different volatility regimes?
4. Walk-forward validation vs. current stratified k-fold for time-series data
