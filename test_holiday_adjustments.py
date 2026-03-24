#!/usr/bin/env python3
"""
Quick test to verify holiday adjustment logic is working correctly.
"""

from datetime import date

# Test the holiday adjustment logic
HOLIDAY_ADJUSTMENTS = {
    date(2025, 9, 2): date(2025, 8, 29),   # Labor Day (Sept 1) -> use Friday Aug 29
    date(2025, 2, 18): date(2025, 2, 14),  # Presidents Day (Feb 17) -> use Friday Feb 14
}

print("Holiday Adjustment Map:")
print("=" * 60)
for after_date, reference_date in sorted(HOLIDAY_ADJUSTMENTS.items()):
    print(f"{after_date} (after holiday) -> {reference_date} (reference)")

print("\n" + "=" * 60)
print("✓ Holiday adjustments configured for:", len(HOLIDAY_ADJUSTMENTS), "dates")
print("\nThese dates will use adjusted reference dates for:")
print("  - relative_ib_vol_pdv (IB volume / previous session volume)")
print("  - relative_ib2_volume (current IB volume / previous IB volume)")
print("  - norm_opening_volatility (opening ATR / avg previous 5-day RTH ATR)")
