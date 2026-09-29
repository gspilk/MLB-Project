"""
team_half_splits.py  (SCRIPT_VERSION printed at startup -- if you're not
sure whether a fix landed, check that number against what I tell you)

1st-half vs. 2nd-half wOBA/xwOBA splits for the current Mariners roster.
REPLACES both player_half_splits.py (bbref, one request per player -- kept
hitting persistent rate-limit blocks on ~14 players every run) and the first
version of statcast_half_splits.py (bulk pybaseball pull, but scoped to
"team" using derived home/away/inning logic -- came back with 537 players
across the whole league instead of ~53 Mariners, because pybaseball's
`team=` filter selects GAMES that team played in, not that team's players).

NEW STRATEGY: use each data source for only the one thing it's actually
proven reliable at, instead of asking either one to do the whole job:

  1. Baseball-Reference's roster page -- WHO is on the team. This has
     worked perfectly on every single run so far (53/53 real bbref player
     IDs, every time) because it's ONE request, not one per player.
  2. pybaseball's Chadwick register (playerid_reverse_lookup) -- bridges
     bbref IDs to MLBAM IDs. One bulk lookup, not per-player.
  3. pybaseball's bulk statcast() pull -- the actual stats. Two requests
     total (one per half), covering every pitch in every Mariners game.

Step 3's raw pull still contains every OPPONENT player from those games
too (confirmed in two real runs) -- the fix is to filter the final result
down to the roster ID set from step 1/2, not to try to derive "was this the
Mariners' player" from game context (home/away/inning), which is exactly
the fragile approach that just failed twice in a row.

NOISE REDUCTION: pybaseball prints a wall of tqdm progress bars (one per
daily sub-query) and pandas FutureWarnings to the terminal, which made the
last two runs' output nearly impossible to read past. Both are suppressed
here. pybaseball's own on-disk cache is also enabled, so even a fresh run
of this script after a code change doesn't re-download from Savant --
it replays from pybaseball's cache almost instantly.

NEEDS A REAL TEST RUN, same caveat as always: the roster-scrape and
Chadwick-lookup pieces are proven from prior real runs; the ID-filtering
logic is tested here against synthetic data shaped like the real schema,
but hasn't been run against an actual live pull yet.
"""

SCRIPT_VERSION = "2026-09-28.1"

import os
import re
import time
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")           # kill pybaseball's per-day progress bars
warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")
warnings.filterwarnings("ignore", message=".*nice request.*")  # pybaseball's cache-nag UserWarning

import pandas as pd
import requests

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR   = os.path.join(DATA_DIR, "cache")
SUMMARY_CSV = os.path.join(DATA_DIR, "team_half_splits_summary.csv")
CACHE_TTL_HOURS = 24 * 4

TEAM_ABBR = "SEA"
SEASON    = 2026
ROSTER_URL = f"https://www.baseball-reference.com/teams/{TEAM_ABBR}/{SEASON}-roster.shtml"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
PLAYER_LINK_RE = re.compile(r"/players/[a-z]/([a-z0-9]+)\.shtml")

FIRST_HALF_START, FIRST_HALF_END   = f"{SEASON}-03-01", f"{SEASON}-07-13"
SECOND_HALF_START, SECOND_HALF_END = f"{SEASON}-07-15", f"{SEASON}-10-15"


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"team_half_{name}.parquet")


def _is_stale(path: str, ttl_hours: float = CACHE_TTL_HOURS) -> bool:
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > ttl_hours


def get_roster_bbref_ids(force_refresh: bool = False) -> list:
    """
    Step 1: who's on the team. This is the ONE part of the whole saga that
    has worked identically and correctly on every real run so far.
    """
    cache_path = _cache_path("roster_ids")
    if not force_refresh and not _is_stale(cache_path):
        print("  [cache] loading roster bbref ids from disk")
        return pd.read_parquet(cache_path)["bbref_id"].tolist()

    print(f"  [fetch] roster page ({TEAM_ABBR} {SEASON}) ...")
    resp = requests.get(ROSTER_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text.replace("<!--", "").replace("-->", "")

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="appearances") or soup.find("table", id="roster")
    candidates = (table.find_all("a", href=PLAYER_LINK_RE) if table is not None
                  else soup.find_all("a", href=PLAYER_LINK_RE))

    ids = []
    seen = set()
    for a in candidates:
        m = PLAYER_LINK_RE.search(a.get("href", ""))
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            ids.append(m.group(1))

    if not ids:
        print("    [warn] no player ids found on roster page")
        return []

    pd.DataFrame({"bbref_id": ids}).to_parquet(cache_path, index=False)
    print(f"    [ok] {len(ids)} roster players")
    return ids


def bbref_ids_to_mlbam(bbref_ids: list) -> dict:
    """
    Step 2: bridge bbref IDs -> MLBAM IDs via pybaseball's Chadwick register
    lookup. One bulk call, not per-player. Returns {mlbam_id: display_name}.
    """
    from pybaseball import playerid_reverse_lookup
    lookup = playerid_reverse_lookup(bbref_ids, key_type="bbref")
    if lookup.empty:
        return {}
    out = {}
    for _, row in lookup.iterrows():
        mlbam = row.get("key_mlbam")
        if pd.notna(mlbam):
            out[int(mlbam)] = f"{str(row['name_first']).title()} {str(row['name_last']).title()}"
    return out


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

    bbref_ids = get_roster_bbref_ids(force_refresh=force_refresh)
    if not bbref_ids:
        print("  [warn] no roster -- aborting")
        return pd.DataFrame()

    roster_names = bbref_ids_to_mlbam(bbref_ids)
    print(f"  [ok] resolved {len(roster_names)}/{len(bbref_ids)} roster players to MLBAM ids")

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