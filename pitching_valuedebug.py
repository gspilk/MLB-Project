"""
value_pitching_debug.py
Checks whether RA9def is actually mapped to the right real column --
the last run showed nearly every pitcher with an identical -0.02,
which is far too uniform to be real per-pitcher data. Prints the raw
column list and a few full rows so we can see what's actually there.

Usage:
    python value_pitching_debug.py
"""

from data_builder import build_all

data = build_all(2026)
val_pit = data["seattle"].get("value_pitching")

if val_pit is None or val_pit.empty:
    print("No value_pitching table found at all.")
else:
    print("Real columns found:", val_pit.columns.tolist())
    print(f"\n{len(val_pit)} rows\n")
    print(val_pit.head(10).to_string(index=False))