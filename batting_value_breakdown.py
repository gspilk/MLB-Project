"""
value_breakdown.py
Breaks each batter's WAR into its real components, using bbref's own
Value Batting table -- not just the final WAR number player_grades.py
already uses, but WHERE that value actually comes from:

  Rbat   -- runs above average from hitting alone
  Rbaser -- runs above average from baserunning
  Rfield -- runs above average from fielding
  Rpos   -- positional adjustment (scarcity of the position played)

Two players can have identical WAR with very different real profiles --
one compiling value through defense/baserunning while scuffling at the
plate, another carrying the team offensively while being a real
liability in the field. Useful directly for trade-target evaluation:
a target's WAR alone doesn't say which of these he is.

Usage:
    python value_breakdown.py
"""

import pandas as pd
from data_builder import build_all


def build_value_breakdown(data: dict) -> pd.DataFrame:
    val_bat = data["seattle"].get("value_batting")
    if val_bat is None or val_bat.empty:
        print("  [warn] no value_batting table found")
        return pd.DataFrame()

    cols = ["Name", "Rbat", "Rbaser", "Rfield", "Rpos", "WAR"]
    real_cols = [c for c in cols if c in val_bat.columns]
    missing = set(cols) - set(real_cols)
    if missing:
        print(f"  [warn] missing expected columns: {missing} -- "
              f"check the real table structure with a debug print")

    df = val_bat[real_cols].copy()
    for c in real_cols:
        if c != "Name":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["WAR"]).sort_values("WAR", ascending=False)


def classify_profile(row: pd.Series) -> str:
    """
    Simple, real classification of WHERE a player's value is coming
    from, based on which component(s) dominate their real runs-above-
    average breakdown -- not a judgment on whether that's good or bad,
    just a description of the actual shape of their contribution.
    """
    rbat = row.get("Rbat", 0) or 0
    rbaser = row.get("Rbaser", 0) or 0
    rfield = row.get("Rfield", 0) or 0

    if rbat > 10 and rbat > (rfield + rbaser) * 1.5:
        return "Bat-driven"
    if rfield > 5 and rfield > rbat:
        return "Defense-driven"
    if rbat < -5 and rfield > 0:
        return "Glove carrying a weak bat"
    if rbat > 0 and rfield < -5:
        return "Bat covering for a weak glove"
    if abs(rbat) < 5 and abs(rfield) < 5:
        return "Balanced / unremarkable either way"
    return "Mixed"


def print_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("WAR COMPONENT BREAKDOWN -- where each player's value actually comes from")
    print("=" * 90)
    print(f"  {'Name':<24} {'Rbat':>6} {'Rbaser':>7} {'Rfield':>7} "
          f"{'Rpos':>6} {'WAR':>6}  Profile")
    print("  " + "-" * 86)
    for _, row in df.iterrows():
        profile = classify_profile(row)
        print(f"  {row['Name']:<24} {row.get('Rbat', 0):>6.1f} "
              f"{row.get('Rbaser', 0):>7.1f} {row.get('Rfield', 0):>7.1f} "
              f"{row.get('Rpos', 0):>6.1f} {row['WAR']:>6.1f}  {profile}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    df = build_value_breakdown(data)
    if not df.empty:
        print_breakdown(df)
    else:
        print("No data to show.")