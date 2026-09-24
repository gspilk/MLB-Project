"""
roster_debug.py

Quick check on the freshly-scraped 40-man roster cache -- prints every
player's real Name/OnActv/IL values so we can see exactly why Montes and
Rodden aren't getting marked active while Pereda and Arroyo correctly are,
now that the OnActv column itself is being preserved.

Run from the project folder:
    python roster_debug.py
"""

import pandas as pd

df = pd.read_parquet("data/cache/sea_roster_2026.parquet")

print(f"columns: {list(df.columns)}\n")
print(df[["Name", "OnActv", "IL"]].to_string())

print("\nOnActv dtype:", df["OnActv"].dtype)
print("OnActv raw unique values:", df["OnActv"].unique().tolist())

for name in ["Lazaro Montes", "Brock Rodden", "Jhonny Pereda", "Michael Arroyo"]:
    row = df[df["Name"].str.contains(name, case=False, na=False)]
    if row.empty:
        print(f"\n{name}: NOT FOUND in roster table")
    else:
        r = row.iloc[0]
        print(f"\n{name}: OnActv={r['OnActv']!r} (type {type(r['OnActv'])})  IL={r['IL']!r}")