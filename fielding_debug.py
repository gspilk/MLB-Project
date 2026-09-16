"""
fielding_debug.py
Checks the real, raw fielding table columns before building any
analysis on top of assumed column names -- learned from the RA9def
situation earlier tonight (guessed what a column meant, had to correct
it after seeing real data). Better to check first this time.

Usage:
    python fielding_debug.py
"""

from data_builder import build_all

data = build_all(2026)
fielding = data["seattle"].get("fielding")

if fielding is None or fielding.empty:
    print("No fielding table found at all.")
else:
    print("Real columns found:", fielding.columns.tolist())
    print(f"\n{len(fielding)} rows\n")
    print(fielding.head(10).to_string(index=False))