"""
roster_value_breakdown.py
Combines value_breakdown.py (batting) and pitching_value_breakdown.py
(pitching) into one unified report -- the whole roster's real value
sources in a single run, instead of two separate scripts.

Usage:
    python roster_value_breakdown.py
"""

from data_builder import build_all
from batting_value_breakdown import build_value_breakdown, print_breakdown
from pitching_value_breakdown import build_pitching_breakdown, print_breakdown as print_pitching_breakdown


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)

    bat_df = build_value_breakdown(data)
    if not bat_df.empty:
        print_breakdown(bat_df)
    else:
        print("No batting value data to show.")

    pit_df = build_pitching_breakdown(data)
    if not pit_df.empty:
        print_pitching_breakdown(pit_df)
    else:
        print("No pitching value data to show.")