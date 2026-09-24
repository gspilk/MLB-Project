"""
data_structure_debug.py
The stats matcher got 0 of 278 matches, which is far too uniform to be
real individual name/data gaps -- almost certainly means the guessed
keys ("batting_leaders", "pitching_leaders") don't match the real
structure build_all() actually returns. Prints the real top-level keys
and a sample of whatever's really there, so the matcher can be fixed
against reality instead of another guess.

Usage:
    python data_structure_debug.py
"""

from data_builder import build_all

data = build_all(2026)

print("\nReal top-level keys in data:", list(data.keys()))

for key in data:
    val = data[key]
    if isinstance(val, dict):
        print(f"\n'{key}' is a dict with real keys: {list(val.keys())}")
        for subkey, subval in val.items():
            if hasattr(subval, "columns"):
                print(f"  '{subkey}': DataFrame, {len(subval)} rows, "
                      f"columns: {subval.columns.tolist()[:8]}...")
    elif hasattr(val, "columns"):
        print(f"\n'{key}': DataFrame, {len(val)} rows, "
              f"columns: {val.columns.tolist()[:8]}...")