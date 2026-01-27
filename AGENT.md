# Agent Instructions

## Project Structure

### Main File
- `mp2b_IBH_IBL.py` - **PRIMARY FILE TO MODIFY**
  - This is the main analysis script
  - All significant changes should be made here

### Data Files
- `MNQ_1min_2023Jan_2026Jan.csv` - Input data file (READ ONLY)
  - Contains minute-level data from Jan 2023 to Jan 2026
  - Do not modify this file

### Other Files
- All other `.py` files are dummies/templates - ignore them unless specifically asked

---

## Refactoring Objective

Refactor the code to calculate IBH/IBL for a **user-defined period** (not hardcoded).

### Daily Calculations Required

For each trading day, calculate and track:

1. **Balance vs Discovery**
   - Did we stay within Initial Balance or enter Discovery?

2. **Initial Balance Metrics**
   - Size of Initial Balance (IBH - IBL)
   - Relative volume during Initial Balance period
   - IBH (Initial Balance High)
   - IBL (Initial Balance Low)
   - Midpoint ((IBH + IBL) / 2)

3. **Discovery Extensions** (if applicable)
   - How far did we extend beyond IB?
   - 1.5x extension (positive or negative)
   - 2x extension (positive or negative)

4. **Market Structure**
   - Rotation: Did price rotate between IBH and IBL?
   - Failed Auction: Did we have a failed auction for that day?

### Output Format
- Results should be organized by day
- Each day should have all metrics clearly labeled
- Consider a DataFrame or structured output for analysis
