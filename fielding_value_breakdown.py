"""
fielding_breakdown.py
Per-player defensive value using bbref's real fielding table --
Rtot (Total Zone) and Rdrs (Defensive Runs Saved), two independent
real methodologies for measuring the same thing. Confirmed real
columns via fielding_debug.py -- MultiIndex headers like bbref's other
tables this project scrapes (('Total Zone', 'Rtot'), ('DRS', 'Rdrs')).

This is what connects the team-level "DefEff rank 23/30" (already
shown elsewhere in this project) down to WHICH specific players are
actually driving that, rather than the whole roster sharing blame
equally.

Usage:
    python fielding_breakdown.py
"""

import pandas as pd
from data_builder import build_all


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUG FIX: originally only handled a genuine pd.MultiIndex -- but the
    real columns arriving here turned out to already be FLAT STRINGS
    that literally contain tuple-repr text, e.g. the string
    "('Total Zone', 'Rtot')" rather than an actual (top, bottom) tuple.
    This almost certainly happened upstream in this project's own
    caching -- parquet doesn't support MultiIndex columns natively, so
    something in the existing pipeline likely already stringified them
    on save. A live run confirmed this exactly: every expected column
    came back "missing" because the real column names didn't match
    plain strings like "Rtot" at all.

    Handles BOTH a genuine MultiIndex (the original case this was
    written for) AND this already-stringified case, using
    ast.literal_eval to safely parse a string that looks like a real
    Python tuple literal back into one.
    """
    import ast

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = [str(bottom) for _, bottom in df.columns]
        df = df.copy()
        df.columns = new_cols
        return df

    # check if columns are stringified tuples instead
    sample = str(df.columns[0]) if len(df.columns) else ""
    if sample.startswith("(") and sample.endswith(")"):
        new_cols = []
        for col in df.columns:
            try:
                parsed = ast.literal_eval(str(col))
                new_cols.append(str(parsed[1]) if isinstance(parsed, tuple) else str(col))
            except (ValueError, SyntaxError):
                new_cols.append(str(col))
        df = df.copy()
        df.columns = new_cols
        return df

    return df


# BUG FIX: bbref's real fielding page includes non-player rows mixed
# in with the real roster -- a "Team Totals" summary row, plus separate
# team-wide shift/positioning category rows ("Infield Shifts", "Non-
# Shift Infield Positioning", "Outfield Positioning"). A live run
# showed "Team Totals" getting classified as if it were an individual
# player with a real defensive grade, which is meaningless for a
# summary row. Filtered out by name rather than guessed at some other
# way, since these are real, predictable bbref section labels.
NON_PLAYER_ROWS = {
    "Team Totals", "Infield Shifts", "Non-Shift Infield Positioning",
    "Outfield Positioning",
}


def build_fielding_breakdown(data: dict) -> pd.DataFrame:
    fielding = data["seattle"].get("fielding")
    if fielding is None or fielding.empty:
        print("  [warn] no fielding table found")
        return pd.DataFrame()

    df = _flatten_columns(fielding)
    cols = ["Player", "Pos", "G", "Inn", "Rtot", "Rtot/yr", "Rdrs", "Rdrs/yr"]
    real_cols = [c for c in cols if c in df.columns]
    missing = set(cols) - set(real_cols)
    if missing:
        print(f"  [warn] missing expected columns: {missing}")

    out = df[real_cols].copy()
    if "Player" in out.columns:
        before = len(out)
        out = out[~out["Player"].isin(NON_PLAYER_ROWS)]
        removed = before - len(out)
        if removed:
            print(f"  [info] filtered {removed} non-player summary/category "
                  f"row(s) from the real fielding table")

    for c in real_cols:
        if c not in ("Player", "Pos"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["Rtot", "Rdrs"], how="all").sort_values(
        "Rtot", ascending=True, na_position="last")


def classify_defense(row: pd.Series) -> str:
    """
    Cross-checks BOTH real methodologies (Total Zone and DRS) when both
    exist -- they're independently computed and can disagree, so
    agreement is a stronger signal than either alone.

    BUG FIX: Rtot (Total Zone) genuinely isn't computed for pitchers on
    bbref at all -- a real, known limitation of that specific
    methodology, not missing/bad data. The original code required BOTH
    metrics to be present, so every single pitcher showed "Incomplete
    defensive data" even when Rdrs (which IS available for pitchers)
    had a real, usable number. Falls back to Rdrs alone, clearly
    labeled as single-metric, rather than discarding real data pitchers
    actually have.
    """
    rtot, rdrs = row.get("Rtot"), row.get("Rdrs")

    if pd.isna(rtot) and not pd.isna(rdrs):
        if rdrs <= -5:
            return "Defensive minus (DRS only -- Rtot not computed for pitchers)"
        if rdrs >= 5:
            return "Defensive plus (DRS only -- Rtot not computed for pitchers)"
        return "Roughly average (DRS only -- Rtot not computed for pitchers)"

    if pd.isna(rtot) or pd.isna(rdrs):
        return "Incomplete defensive data"

    rtot_bad, rdrs_bad = rtot <= -5, rdrs <= -5
    rtot_good, rdrs_good = rtot >= 5, rdrs >= 5

    if rtot_bad and rdrs_bad:
        return "Real defensive minus (both metrics agree)"
    if rtot_good and rdrs_good:
        return "Real defensive plus (both metrics agree)"
    if (rtot_bad and rdrs_good) or (rtot_good and rdrs_bad):
        return "Metrics DISAGREE -- treat with caution"
    return "Roughly average defensively"


def print_breakdown(df: pd.DataFrame):
    print("\n" + "=" * 96)
    print("FIELDING BREAKDOWN -- real Total Zone and DRS defensive value, cross-checked")
    print("=" * 96)
    print(f"  {'Player':<22} {'Pos':<10} {'G':>5} {'Rtot':>6} "
          f"{'Rdrs':>6}  Real Read")
    print("  " + "-" * 92)
    for _, row in df.iterrows():
        note = classify_defense(row)
        print(f"  {row['Player']:<22} {str(row.get('Pos','')):<10} "
              f"{row.get('G', float('nan')):>5.0f} "
              f"{row.get('Rtot', float('nan')):>6.1f} "
              f"{row.get('Rdrs', float('nan')):>6.1f}  {note}")
    print("=" * 96)
    print("  Rtot (Total Zone) and Rdrs (DRS) are two independent real")
    print("  defensive metrics -- negative = runs cost by fielding,")
    print("  positive = runs saved. Shown together since they can disagree.")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    df = build_fielding_breakdown(data)
    if not df.empty:
        print_breakdown(df)
    else:
        print("No data to show.")