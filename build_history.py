"""
Build league-wide player tables for past seasons and cache them, so they can be
loaded into MySQL (python load_mysql.py --players-only 2022 2023 2024 2025).

    python build_history.py                 # 2022-2025
    python build_history.py 2023 2024       # specific seasons
    python build_history.py --force 2025    # rebuild even if the file exists

Uses the same code as the main pipeline (mlb_api.build_player_tables): MLB Stats
API for the counting stats and ages, Baseball Savant for fielding runs. Each
season is cached, so a season is only downloaded once (the first one takes a few
minutes). Rows are one per player per team, like the 2026 files; players traded
mid-season have a row for each team (the combined "2TM" rows are dropped).
"""
import os
import sys

import mlb_api

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
DEFAULT_SEASONS = [2022, 2023, 2024, 2025]


def path(kind, season):
    return os.path.join(CACHE, f"{kind}_all_players_{season}.parquet")


def build_season(season, force=False, get_tables=None):
    get_tables = get_tables or mlb_api.get_player_tables
    bat_p, pit_p = path("batting", season), path("pitching", season)
    if not force and os.path.exists(bat_p) and os.path.exists(pit_p):
        print(f"  [skip] {season}: already built (use --force to rebuild)")
        return False
    print(f"  [build] {season} ... (first run takes a few minutes)")
    t = get_tables(season)
    os.makedirs(CACHE, exist_ok=True)
    for kind, key, cols in (("batting", "batting", mlb_api.BAT_COLS),
                            ("pitching", "pitching", mlb_api.PIT_COLS)):
        df = t[key]
        df = df[~df["_total"]] if "_total" in df.columns else df
        out = mlb_api.shape(df, cols)
        out["league"] = out["Lg"]
        out["stat_type"] = kind
        out.to_parquet(path(kind, season), index=False)
        print(f"  [ok]   {kind} {season}: {len(out):,} rows")
    return True


def main(argv):
    force = "--force" in argv
    seasons = [int(a) for a in argv if a.lstrip("-").isdigit()] or DEFAULT_SEASONS
    for s in seasons:
        try:
            build_season(s, force=force)
        except Exception as e:
            print(f"  [warn] {s} failed: {type(e).__name__}: {e}")
    print("\nNext:  python load_mysql.py --players-only " + " ".join(map(str, seasons)))


if __name__ == "__main__":
    main(sys.argv[1:])