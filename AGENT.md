# Agent Instructions

## Project Structure

### Main File
- `mp2b_IBH_IBL.py` — **PRIMARY FILE TO MODIFY**
  - Main analysis script
  - All significant changes should be made here

### Data Files
- `MNQ_1min_2023Jan_2026Jan.csv` — **READ ONLY**
  - Minute-level data from Jan 2023 to Jan 2026
  - Do not modify this file

### Other Files
- All other `.py` files are dummies/templates  
- Ignore unless explicitly instructed otherwise

---

## Phase I — Intraday Structure & Opening Behavior (LOCKED)

### Phase I Objective

Establish reliable, reproducible intraday market structure metrics for MNQ during RTH, forming the foundation for higher-order contextual analysis.

---

## Refactoring Objective

Refactor the code to calculate Initial Balance metrics for a **user-defined period** (not hardcoded).

---

## Daily Calculations Required

For **Regular Trading Hours (RTH)** defined as **6:30 AM – 4:00 PM**, calculate and track the following on a per-day basis:

### 1. Balance vs Discovery
- Did price remain within Initial Balance?
- Did price enter Discovery (above IBH or below IBL)?

### 2. Initial Balance Metrics
- IBH (Initial Balance High)
- IBL (Initial Balance Low)
- IB Size (`IBH - IBL`)
- Midpoint (`(IBH + IBL) / 2`)
- Relative volume during Initial Balance period

### 3. Discovery Extensions (If Applicable)
- Extension beyond IB:
  - +1.5x IB
  - -1.5x IB
  - +2.0x IB
  - -2.0x IB

### 4. Market Structure
- Rotation: Did price rotate between IBH and IBL?
- Failed Auction: Did the auction fail during the session?

### 5. Opening Range Behavior
- Based on the 6:30 AM one-minute candle:
  - Open-to-close relationship
  - Volume
- Classify opening type:
  - Drive
  - Auction

---

## Output Requirements

- Output must be organized **by trading day**
- Each day must include all calculated metrics
- Output should be structured for downstream analysis:
  - DataFrame
  - Or clearly defined tabular format

---

## Testing Workflow

- After any code change:
  - Perform a smoke test
  - Capture all stdout/stderr output to `test-results.log`
- Upload logs as a GitHub Actions artifact:
  - `test-logs-${{ github.run_id }}.log`
- If tests fail:
  - Comment full log in the PR
  - Set PR status to **pending**

---

## Phase II — Contextual Levels & Tendency Mapping (ACTIVE)

### Phase II Rationale

Phase I established reliable intraday structure metrics (IBH/IBL, opening type, extensions) on a per-day basis.  
Phase II builds on this foundation by contextualizing intraday behavior against **prior-day reference levels** to identify repeatable market tendencies.  
The objective is to transition from descriptive structure metrics to **conditional behavior prediction**.

---

### Phase II Objectives

1. **Previous-Day Contextualization**
   - Extract prior-day reference levels:
     - Previous Day High (PDH)
     - Previous Day Low (PDL)
     - Value Area High (VAH)
     - Value Area Low (VAL)
     - Point of Control (POC)
   - Compare current-day interaction with prior-day levels.

2. **Level Significance Assessment**
   - Evaluate which prior-day levels most influence:
     - Opening behavior
     - Rotation depth
     - Discovery vs balance outcomes
   - Identify confluence between levels (e.g., VAH + IBH, PDL + VAL).

3. **Hypothesis Validation**
   - Use a **24-hour POC on the 1-minute timeframe** as a real-time contextual anchor.
   - Validate acceptance vs rejection hypotheses intraday.
   - Formalize hypothesis tests for historical replay.

4. **Tendency Map Construction**
   - Encode recurring behavior patterns such as:
     - Rotation → pullback → continuation
     - Early rejection → balance
     - Acceptance outside prior value → discovery
   - Express tendencies as conditional logic, not trade signals.

---

### Explicit Non-Objectives (Phase II)

- Live execution or order management
- Strategy optimization or parameter fitting
- Machine-learning models beyond simple decision trees
- Tick-level or order-book analysis

---

### Script Ownership & Refactoring Scope

#### Primary Files

- `mp2b_IBH_IBL.py`
  - Source of intraday structure metrics:
    - IBH / IBL
    - Opening type
    - Rotation and discovery labels

- `mp2a_previous_day_levels.py` — **FULL REFACTOR ALLOWED**
  - Build on outputs from `mp2b*`
  - Compute and attach prior-day contextual levels
  - Prepare structured outputs for level interaction analysis

---

### Phase II Script Registry

| Script | Status | Purpose | Depends On |
|------|--------|---------|------------|
| mp2b_IBH_IBL.py | stable | Intraday structure & opening metrics | raw bars |
| mp2a_previous_day_levels.py | active | Prior-day levels & contextual joins | mp2b |
| mp2c_level_interaction.py | planned | Touch / acceptance / rejection metrics | mp2a |
| mp2d_tendency_map.py | planned | Conditional behavior encoding | mp2c |

---

### Definitions & Constraints

#### Prior-Day Levels

- All prior-day levels must:
  - Be derived only from **completed sessions**
  - Be frozen before the next session begins
  - Explicitly specify session scope (RTH vs 24-hour)

#### Level Interaction Semantics

- **Touch**: Price reaches a level within tolerance
- **Breach**: Price trades beyond a level
- **Acceptance**: Sustained trade beyond a level (parameterized)
- **Rejection**: Failure to hold beyond a level

All definitions must be configurable and non-hardcoded.

---

### Design Constraints (Phase II)

- No forward-looking data leakage
- All comparisons must be reproducible bar-by-bar
- Explicit and consistent timezone handling
- Outputs must be join-friendly and structured
- Favor interpretability over complexity

---

### Key Phase II Decisions

- Rule-based logic precedes machine learning
- Prior-day levels provide contextual bias, not signals
- 24-hour POC is treated as a validation tool
- State machine deferred until tendencies stabilize

---

### Open Questions

- Which prior-day level best predicts rotation depth?
- Does acceptance beyond VAH/VAL increase discovery probability?
- Is POC influence symmetric across long and short sessions?
- How stable are tendencies across volatility regimes?

---

### Phase II Milestones

- **v0.1** — Prior-day levels computed and joined
- **v0.2** — Level interaction metrics validated
- **v0.3** — Tendency map encoded as conditional logic

---

### Phase II Litmus Test

If Phase II is complete, the system should answer:

> “Given today’s interaction with yesterday’s levels, what behavior is most likely next?”
