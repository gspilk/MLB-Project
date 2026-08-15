"""
player_year_comparison.py
Compares SPECIFIC named players' stats across seasons (default: 2022,
2025, 2026).

SCOPE -- deliberately targeted lookups, not bulk roster matching
------------------------------------------------------------------
This does NOT try to auto-match every player on every year's roster
against each other -- that's the harder, riskier problem (surname
collisions, players changing teams, retiring, etc. -- see
multi_year_comparison.py's docstring for why that was avoided there
too). Instead, YOU specify which players to look up by name, and this
does one targeted get_player() search per season per player -- the same
safe, tested lookup already used in deadline_trade_comparison.py.

Only works for players who were actually on the Mariners in the given
season -- a player on a different team that year won't be found (this
searches mariners_stats.py's Seattle-only tables, not league-wide
leaders).

Usage:
    python player_year_comparison.py --players "Julio Rodriguez" "Cal Raleigh"
    python player_year_comparison.py --players "Julio Rodriguez" --seasons 2022 2023 2024 2025 2026
"""

import argparse
import pandas as pd

from mariners_stats import get_seattle_stats
from name_matching import normalize


BATTING_COLS = ["PA", "BA", "OBP", "SLG", "OPS", "HR", "RBI", "SB"]
PITCHING_COLS = ["G", "GS", "IP", "ERA", "WHIP", "SO", "BB", "SV"]

# Core roster default -- players who've been on the Mariners across
# multiple of the default comparison years (2022/2025/2026), so a plain
# `python player_year_comparison.py` with no --players flag still gives
# a meaningful multi-player comparison instead of needing you to type
# names every time. Still fully overridable with --players.
CORE_ROSTER_DEFAULT = [
    "Julio Rodriguez", "Cal Raleigh", "J.P. Crawford", "Randy Arozarena",
    "Cole Young", "Josh Naylor", "Dominic Canzone",
    "Logan Gilbert", "Bryan Woo", "George Kirby",
    "Andres Munoz", "Eduard Bazardo",
]


def _get_player_accent_insensitive(df: pd.DataFrame, name: str) -> pd.Series:
    """
    Same job as mariners_stats.get_player(), but accent-insensitive --
    bbref stores names with accents (e.g. "Julio Rodríguez"), and a
    plain substring search for "Rodriguez" (no accent) won't match that.
    Reuses name_matching.normalize(), the same accent-stripping helper
    used throughout the rest of the project, instead of introducing a
    second, different way of handling this.
    """
    if df is None or df.empty or "Name" not in df.columns:
        return pd.Series()
    target = normalize(name)
    matches = df[df["Name"].apply(lambda n: target in normalize(n))]
    return matches.iloc[0] if not matches.empty else pd.Series()


def _lookup(df: pd.DataFrame, name: str, cols: list) -> dict:
    row = _get_player_accent_insensitive(df, name)
    if row.empty:
        return {"found": False}
    out = {"found": True}
    for c in cols:
        if c in row.index:
            out[c] = row[c]
    return out


def build_comparison(players: list, seasons: list) -> pd.DataFrame:
    rows = []
    for season in seasons:
        print(f"\n--- {season} ---")
        stats = get_seattle_stats(season)
        bat = stats.get("batting")
        pit = stats.get("pitching")

        for name in players:
            b = _lookup(bat, name, BATTING_COLS) if bat is not None else {"found": False}
            p = _lookup(pit, name, PITCHING_COLS) if pit is not None else {"found": False}

            if b.get("found"):
                row = {"season": season, "name": name, "role": "batter"}
                row.update({k: v for k, v in b.items() if k != "found"})
                rows.append(row)
                print(f"  {name} (batter): PA {b.get('PA','?')}  OPS {b.get('OPS','?')}")

            if p.get("found"):
                row = {"season": season, "name": name, "role": "pitcher"}
                row.update({k: v for k, v in p.items() if k != "found"})
                rows.append(row)
                print(f"  {name} (pitcher): IP {p.get('IP','?')}  ERA {p.get('ERA','?')}")

            if not b.get("found") and not p.get("found"):
                print(f"  {name}: not found on {season} Mariners roster")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Player-level multi-year comparison")
    parser.add_argument("--players", nargs="+", default=CORE_ROSTER_DEFAULT,
                        help='Player names, e.g. --players "Julio Rodriguez" "Cal Raleigh" '
                             '(default: core roster -- Julio, Raleigh, Crawford, Arozarena, '
                             'Young, Naylor, Canzone, Gilbert, Woo, Kirby, Munoz, Bazardo)')
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=[2022, 2025, 2026],
                        help="Seasons to compare (default: 2022 2025 2026)")
    args = parser.parse_args()

    df = build_comparison(args.players, args.seasons)
    print("\n" + "=" * 70)
    print("PLAYER-LEVEL MULTI-YEAR COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False) if not df.empty else "No matches found.")

    if not df.empty:
        out_path = "player_year_comparison.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")