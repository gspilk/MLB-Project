"""
multi_year_comparison.py
Compares Mariners standings + team-level stats across multiple seasons
(default: 2022, 2025, 2026).

SCOPE -- deliberately team-level only, no individual player matching
---------------------------------------------------------------------
This does NOT try to match individual players across years (e.g. "how
did Julio do in 2022 vs 2026"). That's a much harder problem -- rosters
turn over, players change teams, get injured for a full season, retire,
etc. -- and would need the same kind of name-matching infrastructure
that caused real bugs even within a single season (see name_matching.py
and the roster-collision issues fixed there). This script sticks to
TEAM-LEVEL aggregates (record, run differential, team OPS, team ERA)
computed by summing/averaging each season's roster, which sidesteps
that problem entirely.

WHAT THIS PULLS
---------------
  Standings: record, win%, run differential, division rank
  Batting:   team totals -- HR, R, and team-wide average BA/OBP/SLG/OPS
  Pitching:  team totals -- team ERA (weighted by IP), total SO, WHIP

DATA NOTE
---------
Each season gets its own cache (mariners_stats.py and standings_scraper.py
both key their cache files by season), so pulling 2022/2025 data will
NOT touch or overwrite your existing 2026 cache. First run for a new
season will need real network access to bbref (not cached yet) -- expect
it to take longer than a normal main.py run.

Usage:
    python multi_year_comparison.py
    python multi_year_comparison.py --seasons 2022 2023 2025 2026
"""

import argparse
import pandas as pd

from standings_scraper import get_standings, get_team_row
from mariners_stats import get_seattle_stats


TEAM_NAME = "Seattle Mariners"


def _team_batting_summary(batting_df: pd.DataFrame) -> dict:
    if batting_df is None or batting_df.empty:
        return {}
    df = batting_df.copy()
    # weight rate stats by PA so part-time players don't skew the average
    pa_col = "PA" if "PA" in df.columns else None
    total_pa = df[pa_col].sum() if pa_col else None

    def _weighted_avg(col):
        if col not in df.columns or not pa_col or not total_pa:
            return None
        return round((df[col] * df[pa_col]).sum() / total_pa, 3)

    return {
        "team_HR":  int(df["HR"].sum()) if "HR" in df.columns else None,
        "team_R":   int(df["R"].sum()) if "R" in df.columns else None,
        "team_BA":  _weighted_avg("BA"),
        "team_OBP": _weighted_avg("OBP"),
        "team_SLG": _weighted_avg("SLG"),
        "team_OPS": _weighted_avg("OPS"),
    }


def _team_pitching_summary(pitching_df: pd.DataFrame) -> dict:
    if pitching_df is None or pitching_df.empty:
        return {}
    df = pitching_df.copy()
    ip_col = "IP" if "IP" in df.columns else None
    total_ip = df[ip_col].sum() if ip_col else None

    def _weighted_avg(col):
        if col not in df.columns or not ip_col or not total_ip:
            return None
        return round((df[col] * df[ip_col]).sum() / total_ip, 3)

    return {
        "team_ERA":  _weighted_avg("ERA"),
        "team_WHIP": _weighted_avg("WHIP"),
        "team_SO":   int(df["SO"].sum()) if "SO" in df.columns else None,
    }


def build_comparison(seasons: list) -> pd.DataFrame:
    rows = []
    for season in seasons:
        print(f"\n--- {season} ---")

        print(f"  fetching standings...")
        _, expanded = get_standings(season)
        sea_row = get_team_row(expanded, TEAM_NAME)

        row = {"season": season}
        if sea_row is not None:
            row["record"] = f"{sea_row.get('W','?')}-{sea_row.get('L','?')}"
            row["win_pct"] = sea_row.get("W-L%")
            row["run_diff"] = sea_row.get("Rdiff")
            row["pythag_WL"] = sea_row.get("pythWL")
        else:
            print(f"  [warn] Mariners not found in {season} standings")

        print(f"  fetching Mariners team stats...")
        stats = get_seattle_stats(season)
        row.update(_team_batting_summary(stats.get("batting")))
        row.update(_team_pitching_summary(stats.get("pitching")))

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-year Mariners comparison")
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2022, 2025, 2026],
                        help="Seasons to compare (default: 2022 2025 2026)")
    args = parser.parse_args()

    df = build_comparison(args.seasons)
    print("\n" + "=" * 70)
    print("MULTI-YEAR COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))

    out_path = "multi_year_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")