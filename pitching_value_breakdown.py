"""
pitching_value_breakdown.py
Breaks down each pitcher's real value using bbref's Value Pitching
table -- specifically isolating how much of their runs-allowed is
attributable to the DEFENSE behind them (RA9def), not just their own
pitching. Different shape than the batting breakdown (no clean
Rbat/Rbaser/Rfield split for pitching) -- centered on RA9 and its
adjustments instead.

Real columns used:
  RA9     -- actual runs allowed per 9 innings
  RA9opp  -- adjusted for quality of opponents faced
  RA9def  -- adjustment for the defense playing behind them (the real
             signal for "is Seattle's known-bad defense dragging this
             guy's numbers down")
  RA9role -- adjustment for role (starter vs reliever have different
             real baselines)
  RA9avg  -- league-average RA9 for comparison
  gmLI    -- leverage index (how high-stakes their real appearances were)
  WAR     -- final value

Usage:
    python pitching_value_breakdown.py
"""

import pandas as pd
from data_builder import build_all


def build_pitching_breakdown(data: dict) -> pd.DataFrame:
    val_pit = data["seattle"].get("value_pitching")
    if val_pit is None or val_pit.empty:
        print("  [warn] no value_pitching table found")
        return pd.DataFrame()

    cols = ["Name", "RA9", "RA9opp", "RA9def", "RA9role", "RA9avg", "gmLI", "WAR"]
    real_cols = [c for c in cols if c in val_pit.columns]
    missing = set(cols) - set(real_cols)
    if missing:
        print(f"  [warn] missing expected columns: {missing} -- "
              f"check the real table structure with a debug print")

    df = val_pit[real_cols].copy()
    for c in real_cols:
        if c != "Name":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["WAR"]).sort_values("WAR", ascending=False)


def classify_defense_impact(row: pd.Series) -> str:
    """
    CORRECTED after checking real raw data: RA9def is a TEAM-WIDE
    defensive-efficiency adjustment, applied almost identically to
    every pitcher on the roster (they all pitched in front of
    essentially the same fielders all season) -- NOT a per-pitcher
    signal the way this was originally framed. A live check showed
    literally every Mariners pitcher at -0.02, confirming this is real,
    correctly-extracted data, just not something that differentiates
    individual pitchers the way the original design assumed.

    The column that DOES vary meaningfully per pitcher is RA9role
    (starters ~0.21, most relievers ~-0.30-ish -- a real role-based
    baseline difference), used here instead.
    """
    ra9role = row.get("RA9role")
    if ra9role is None or pd.isna(ra9role):
        return "No real role-adjustment data available"
    if ra9role >= 0.15:
        return "Starter-level role adjustment"
    if ra9role <= -0.15:
        return "Reliever-level role adjustment"
    return "Role adjustment near neutral"


def print_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("PITCHING VALUE BREAKDOWN -- real opponent/role-adjusted performance and leverage")
    print("=" * 100)

    # CORRECTED: RA9def turned out to be a real, near-constant TEAM-WIDE
    # value (confirmed via a live raw-column check -- every Mariners
    # pitcher showed -0.02), not a per-pitcher signal. Shown once here,
    # at the team level, instead of misleadingly repeated on every row
    # as if it varied person to person.
    if "RA9def" in df.columns and df["RA9def"].notna().any():
        team_ra9def = df["RA9def"].mode().iloc[0] if not df["RA9def"].mode().empty else df["RA9def"].mean()
        direction = "hurting" if team_ra9def < 0 else "helping" if team_ra9def > 0 else "neutral for"
        print(f"  Team-wide defensive context: {team_ra9def:+.2f} runs/9 -- "
              f"the defense has been mildly {direction} every pitcher's real "
              f"numbers roughly equally, since they share the same fielders")
        print()

    print(f"  {'Name':<24} {'RA9':>6} {'RA9opp':>7} {'RA9role':>8} "
          f"{'gmLI':>6} {'WAR':>6}  Role Adjustment")
    print("  " + "-" * 96)
    for _, row in df.iterrows():
        note = classify_defense_impact(row)
        ra9role = row.get("RA9role")
        gmli = row.get("gmLI")
        print(f"  {row['Name']:<24} {row.get('RA9', float('nan')):>6.2f} "
              f"{row.get('RA9opp', float('nan')):>7.2f} "
              f"{(ra9role if ra9role is not None else float('nan')):>8.2f} "
              f"{(gmli if gmli is not None else float('nan')):>6.2f} "
              f"{row['WAR']:>6.1f}  {note}")
    print("=" * 100)
    print("  RA9opp: opponent quality faced (higher = tougher opponents).")
    print("  RA9role: real baseline difference between starters and relievers,")
    print("  not a judgment of skill -- just accounts for the different real")
    print("  context each role pitches in.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    df = build_pitching_breakdown(data)
    if not df.empty:
        print_breakdown(df)
    else:
        print("No data to show.")