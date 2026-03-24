# Data Leakage Fix - Re-run Instructions

## Issue Identified
The `relative_ib_volume` feature had **data leakage**:
- **Old calculation (WRONG)**: `ib_volume / total_volume` (same-day data)
- **New calculation (CORRECT)**: `ib_volume / prev_session_volume` (prior-day data only)

## Changes Made

### 1. Code Updates
- **mp2a_previous_day_levels.py**: Now correctly calculates `relative_ib_volume` using `prev_session_volume`
- **mp2b_IBH_IBL.py**: Original calculation remains but is now overwritten by mp2a (deprecated)
- **mp3_analysis_variables.py**: Added warning comment about data leakage
- **METRICS.md**: Updated documentation
- **FEATURE_ENGINEERING_README.md**: Updated documentation

### 2. What This Means
The feature now correctly measures: **How aggressive is today's IB relative to yesterday's total activity?**

This is a legitimate predictive feature with no forward-looking bias.

## Re-run Instructions

### Step 1: Regenerate Phase 2 Data
Run the pipeline to regenerate all CSVs with the corrected feature:

```python
# In rotation_prediction_pipeline.ipynb or similar
# This will run mp2b → mp2a for each opening window
# The corrected relative_ib_volume will be in the output
```

Or run directly:
```bash
python mp2b_IBH_IBL.py --csv MNQ_1min_2023Jan_2026Jan.csv --start-date 2023-01-02 --end-date 2026-01-15 --opening-window-minutes 10 --output outputs/ib_metrics_10min.csv

python mp2a_previous_day_levels.py --csv MNQ_1min_2023Jan_2026Jan.csv --ib-metrics outputs/ib_metrics_10min.csv --output outputs/phase2_10min.csv
```

### Step 2: Re-run Feature Engineering
Open `feature_engineering_and_model_evaluation.ipynb` and re-run from the top to:
1. Load the corrected phase2 data
2. Perform feature engineering with corrected features
3. Run Sequential Feature Selection to find best features
4. Train and evaluate models
5. Save the new model as `best_model.joblib`

### Step 3: Verify Changes
Check that the new `relative_ib_volume` values make sense:
- Should be comparing IB volume to previous day's volume
- Values > 1 mean today's IB is more aggressive than all of yesterday
- Values < 1 mean today's IB is less aggressive

### Step 4: Update Live Prediction Pipeline
The `rotation_prediction_pipeline.ipynb` should load the new `best_model.joblib` automatically.

## Expected Impact
The corrected feature should:
- Potentially change which features are selected as "best"
- May improve or change model performance metrics
- Provides more legitimate predictive power
- Eliminates data leakage concerns

## Notes
- The first session in your data will have `relative_ib_volume = None` (no prior day)
- This is expected and will be handled by dropping NaN rows during modeling
- All existing CSV outputs should be regenerated to reflect the fix
