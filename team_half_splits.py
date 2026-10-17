"""
team_half_splits.py  (SCRIPT_VERSION printed at startup -- if you're not
sure whether a fix landed, check that number against what I tell you)

1st-half vs. 2nd-half wOBA/xwOBA splits for the current Mariners roster.

Two sources, each used for the one thing it does reliably:

  1. MLB Stats API -- WHO is on the team. The full-season roster comes
     straight back with MLBAM player ids, so no separate id-translation
     step (the old source roster + Chadwick lookup) is needed.
  2. pybaseball's bulk statcast() pull from Baseball Savant -- the actual
     pitch-level stats. Two requests total (one per half), covering every
     pitch in every Mariners game.

The raw Statcast pull contains every OPPONENT player from those games too,
so the result is filtered down to the roster id set from step 1 rather
than trying to derive "was this the Mariners' player" from game context.

NOISE REDUCTION: pybaseball's tqdm progress bars and pandas FutureWarnings
are suppressed, and pybaseball's own on-disk cache is enabled, so a re-run
after a code change replays from cache instead of re-downloading.
"""

SCRIPT_VERSION = "2026-10-12.1"

import os
import time
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")           # kill pybaseball's per-day progress bars
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")
warnings.filterwarnings("ignore", message=".*nice request.*")  # pybaseball's cache-nag UserWarning

import pandas as pd

import mlb_api

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR   = os.path.join(DATA_DIR, "cache")
SUMMARY_CSV = os.path.join(DATA_DIR, "team_half_splits_summary.csv")
CACHE_TTL_HOURS = 24 * 4

TEAM_ABBR = "SEA"
TEAM_ID   = mlb_api.SEA_TEAM_ID
SEASON    = 2026

FIRST_HALF_START, FIRST_HALF_END   = f"{SEASON}-03-01", f"{SEASON}-07-13"
SECOND_HALF_START, SECOND_HALF_END = f"{SEASON}-07-15", f"{SEASON}-10-15"


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"team_half_{name}.parquet")


def _is_stale(path: str, ttl_hours: float = CACHE_TTL_HOURS) -> bool:
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > ttl_hours


def get_roster_mlbam(force_refresh: bool = False) -> dict:
    """
    Step 1: who's on the team -- everyone who appeared for Seattle this
    season (the "fullSeason" roster, so traded/optioned players still
    count), falling back to the 40-man. Returns {mlbam_id: display_name}.
    """
    for roster_type in ("fullSeason", "40Man"):
        try:
            roster = mlb_api.get_roster(TEAM_ID, SEASON, roster_type, force_refresh=force_refresh)
        except Exception as e:
            print(f"    [warn] {roster_type} roster unavailable: {e}")
            continue
        if not roster.empty:
            print(f"    [ok] {len(roster)} roster players ({roster_type})")
            return {int(r.player_id): r.Name for r in roster.itertuples() if pd.notna(r.player_id)}
    print("    [warn] no roster players found")
    return {}


def fetch_half_raw(label: str, start_dt: str, end_dt: str,
                   team: str = TEAM_ABBR, force_refresh: bool = False) -> pd.DataFrame:
    """Step 3: bulk pitch-level pull for one half. Still passes team= to
    pybaseball as a coarse pre-filter (cuts down what it has to download),
    but the REAL scoping happens afterward against the known roster IDs."""
    cache_path = _cache_path(label)
    if not force_refresh and not _is_stale(cache_path):
        print(f"  [cache] loading {label} raw statcast from disk")
        return pd.read_parquet(cache_path)

    from pybaseball import statcast
    print(f"  [fetch] statcast {label} ({start_dt} to {end_dt}, team={team}) ...")
    df = statcast(start_dt=start_dt, end_dt=end_dt, team=team, verbose=False)
    if df is None or df.empty:
        print(f"    [warn] no rows returned for {label}")
        return pd.DataFrame()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"    [ok] {len(df)} pitches")
    return df


def _pa_level(df: pd.DataFrame) -> pd.DataFrame:
    """PA-ending rows only, identified by woba_denom being populated -- NOT
    by `events` alone, which is also set for non-PA outcomes like stolen
    bases and caught stealing (a real bug from the previous version)."""
    if df.empty:
        return df
    df = df.copy()
    df["woba_value"] = pd.to_numeric(df.get("woba_value"), errors="coerce")
    df["woba_denom"] = pd.to_numeric(df.get("woba_denom"), errors="coerce")
    df["estimated_woba_using_speedangle"] = pd.to_numeric(
        df.get("estimated_woba_using_speedangle"), errors="coerce")
    pa = df[df["woba_denom"].notna()].copy()
    if pa.empty:
        return pa
    pa["woba_est"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"])
    return pa


def _aggregate_side(pa: pd.DataFrame, id_col: str, roster_ids: set) -> pd.DataFrame:
    """Groups PA rows by player id, filtered to the known roster set FIRST --
    this is the actual fix, replacing the previous version's attempt to
    derive team membership from game context."""
    if pa.empty:
        return pd.DataFrame(columns=[id_col, "PA", "wOBA", "xwOBA"])
    pa = pa[pa[id_col].isin(roster_ids)]
    if pa.empty:
        return pd.DataFrame(columns=[id_col, "PA", "wOBA", "xwOBA"])
    g = pa.groupby(id_col)
    out = pd.DataFrame({
        "PA":    g["woba_denom"].count(),
        "denom": g["woba_denom"].sum(min_count=1),
        "val":   g["woba_value"].sum(min_count=1),
        "est":   g["woba_est"].sum(min_count=1),
    }).reset_index()
    out["wOBA"]  = (out["val"] / out["denom"]).round(3)
    out["xwOBA"] = (out["est"] / out["denom"]).round(3)
    return out[[id_col, "PA", "wOBA", "xwOBA"]]


def build_half_split_table(first_raw: pd.DataFrame, second_raw: pd.DataFrame,
                           roster_names: dict) -> pd.DataFrame:
    roster_ids = set(roster_names.keys())
    first_pa  = _pa_level(first_raw)
    second_pa = _pa_level(second_raw)

    bat_1h = _aggregate_side(first_pa, "batter", roster_ids).rename(
        columns={"batter": "id", "PA": "1H_PA", "wOBA": "1H_wOBA", "xwOBA": "1H_xwOBA"})
    bat_2h = _aggregate_side(second_pa, "batter", roster_ids).rename(
        columns={"batter": "id", "PA": "2H_PA", "wOBA": "2H_wOBA", "xwOBA": "2H_xwOBA"})
    pit_1h = _aggregate_side(first_pa, "pitcher", roster_ids).rename(
        columns={"pitcher": "id", "PA": "1H_PA", "wOBA": "1H_wOBA", "xwOBA": "1H_xwOBA"})
    pit_2h = _aggregate_side(second_pa, "pitcher", roster_ids).rename(
        columns={"pitcher": "id", "PA": "2H_PA", "wOBA": "2H_wOBA", "xwOBA": "2H_xwOBA"})

    bat = pd.merge(bat_1h, bat_2h, on="id", how="outer"); bat["Type"] = "batter"
    pit = pd.merge(pit_1h, pit_2h, on="id", how="outer"); pit["Type"] = "pitcher"
    combined = pd.concat([bat, pit], ignore_index=True)
    if combined.empty:
        return combined

    combined["Name"] = combined["id"].map(roster_names)
    combined["wOBA_Change"] = (combined["2H_wOBA"] - combined["1H_wOBA"]).round(3)

    def trend(row):
        if pd.isna(row["1H_wOBA"]) and pd.notna(row["2H_wOBA"]):
            return "2ND HALF ONLY"
        if pd.notna(row["1H_wOBA"]) and pd.isna(row["2H_wOBA"]):
            return "1ST HALF ONLY"
        if pd.isna(row["wOBA_Change"]):
            return ""
        change = row["wOBA_Change"]
        if row["Type"] == "pitcher":
            return "IMPROVED" if change < -0.015 else ("DECLINED" if change > 0.015 else "STEADY")
        return "IMPROVED" if change > 0.015 else ("DECLINED" if change < -0.015 else "STEADY")

    combined["Trend"] = combined.apply(trend, axis=1)
    cols = ["Name", "Type", "1H_PA", "1H_wOBA", "1H_xwOBA",
            "2H_PA", "2H_wOBA", "2H_xwOBA", "wOBA_Change", "Trend"]
    combined = combined[cols]
    combined["_abs"] = combined["wOBA_Change"].abs()
    combined = combined.sort_values("_abs", ascending=False, na_position="last")
    return combined.drop(columns="_abs").reset_index(drop=True)


def get_team_half_splits(force_refresh: bool = False) -> pd.DataFrame:
    try:
        import pybaseball
        pybaseball.cache.enable()  # pybaseball's OWN cache -- separate from ours,
                                   # means even after a code change, re-running
                                   # doesn't re-download from Savant
    except Exception:
        pass

    roster_names = get_roster_mlbam(force_refresh=force_refresh)
    if not roster_names:
        print("  [warn] no roster -- aborting")
        return pd.DataFrame()

    first  = fetch_half_raw("1st_half", FIRST_HALF_START, FIRST_HALF_END, force_refresh=force_refresh)
    second = fetch_half_raw("2nd_half", SECOND_HALF_START, SECOND_HALF_END, force_refresh=force_refresh)
    return build_half_split_table(first, second, roster_names)


if __name__ == "__main__":
    print(f"team_half_splits.py version {SCRIPT_VERSION}")
    summary = get_team_half_splits()
    if summary.empty:
        print("\nNo half-split data retrieved -- see warnings above.")
    else:
        print(f"\n── Team half-split summary ({len(summary)} roster players, "
              f"biggest wOBA swings first) ──")
        print(summary.to_string(index=False))
        os.makedirs(DATA_DIR, exist_ok=True)
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"\n[saved] {SUMMARY_CSV}  ({len(summary)} players -- open directly in Excel)")