"""
position_debug.py
The incumbent-mapping logic produced clearly wrong results (Cole Young
as "Catchers", Donovan as "First Basemen", Raleigh missing entirely) --
meaning the assumption that batting's real "Pos" column matches the
same format already confirmed for the fielding table was wrong. Prints
the real, raw Pos values for every Mariners batter to fix this
against reality.

Usage:
    python position_debug.py
"""

from data_builder import build_all

data = build_all(2026)
bat = data["seattle"]["batting"]

print("\nReal 'Pos' values in the batting table:\n")
for _, row in bat.iterrows():
    print(f"  {row.get('Name', '?'):<25} Pos={row.get('Pos')!r}  PA={row.get('PA')}")