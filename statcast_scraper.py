"""
statcast_scraper.py
Two sources of Statcast data:

  1. League-wide leaders (get_statcast) -- still a manual CSV export from
     Baseball Savant's leaderboard, used only for league baselines and
     MLB-wide rankings. Save CSVs as:
       data/statcast_batters_2026.csv
       data/statcast_pitchers_2026.csv

  2. Mariners-direct (get_team_statcast) -- LIVE, no manual download.
     BUG FIX 2026-09-27: for months, EVERY xwOBA-based grade/note in this
     project (Miller "elite underneath", every luck callout, every
     "ERA misleading" note) was silently computed off a one-time manual
     CSV export from June 5 that nobody thought to refresh, because
     get_statcast() only ever reads a static local file -- `--refresh`
     never touched it. Root cause wasn't a code bug, it was a design gap:
     there was no live-fetch path for Statcast at all.
     get_team_statcast() closes that gap by hitting Baseball Savant's own
     leaderboard endpoint filtered to one team (?team=136&csv=true), which
     returns real, current, Mariners-only rows directly -- no manual
     download, and no need for the roster-matching layer that's caused
     most of this project's other real bugs (Brash/Gilbert/Davila WAR
     collision, Montes/Rodden marker mismatch, etc.), since the data is
     already scoped to one team server-side.
"""

import os
import io
import time
import pandas as pd
import requests

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
BATTER_FILE  = os.path.join(DATA_DIR, "statcast_batters_2026.csv")
PITCHER_FILE = os.path.join(DATA_DIR, "statcast_pitchers_2026.csv")

SEA_TEAM_ID  = 136
CACHE_DIR    = os.path.join(DATA_DIR, "cache")
CACHE_TTL_HOURS = 6   # matches the "seattle stats" TTL -- updates daily-ish

STATCAST_LEADERBOARD_URL = "https://baseballsavant.mlb.com/leaderboard/expected_statistics"

# Column names Savant's leaderboard actually returns when filtered by team
# (confirmed 2026-09-27 against a real ?team=136&min=1&csv=true response --
# this is a DIFFERENT schema than the plain league-wide export below, e.g.
# "est_woba" here vs. "xwoba" there, and this one includes era/xera directly).
TEAM_RENAME = {
    "last_name, first_name": "Name",
    "pa":                    "PA",
    "bip":                   "BIP",
    "ba":                    "BA",
    "est_ba":                "xBA",
    "slg":                   "SLG",
    "est_slg":               "xSLG",
    "woba":                  "wOBA",
    "est_woba":              "xwOBA",
    "era":                   "ERA",
    "xera":                  "xERA",
}


def _team_cache_path(player_type: str, team: int, year: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"statcast_team{team}_{player_type}_{year}.parquet")


def _is_stale(path: str, ttl_hours: float = CACHE_TTL_HOURS) -> bool:
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > ttl_hours


def _fetch_team_leaderboard(player_type: str, team: int = SEA_TEAM_ID,
                            year: int = 2026, timeout: int = 20) -> pd.DataFrame:
    """
    Live-fetches Baseball Savant's expected-statistics leaderboard filtered
    to one team. player_type: "batter" or "pitcher".

    min=1 is required -- BUG FIX 2026-09-27: without an explicit min,
    Savant's leaderboard defaults to a "qualified" PA/BF threshold that
    silently drops short-relief arms and September call-ups. A real test
    against this exact URL with no min returned only 10 of Seattle's ~26
    pitchers. min=1 is the most permissive value Savant accepts and is
    the only way to get the FULL roster, not just the qualified players.
    """
    params = {
        "type": player_type,
        "year": year,
        "team": team,
        "min":  1,
        "csv":  "true",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0.0.0 Safari/537.36"}
    resp = requests.get(STATCAST_LEADERBOARD_URL, params=params,
                        headers=headers, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(columns=TEAM_RENAME)

    if player_type == "pitcher":
        df = df.rename(columns={"wOBA": "wOBA_against", "xwOBA": "xwOBA_against"})

    skip = {"Name", "year", "player_id"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def get_team_statcast(team: int = SEA_TEAM_ID, year: int = 2026,
                      force_refresh: bool = False):
    """
    Returns (batters_df, pitchers_df) for one team, live from Baseball
    Savant, cached to parquet for CACHE_TTL_HOURS so a normal run doesn't
    hit the network every time. Falls back to a stale cache (with a
    warning) if the live fetch fails, rather than crashing the whole
    pipeline over a network hiccup -- same philosophy as the roster-keys
    fallback in roster.py.
    """
    results = {}
    for player_type in ("batter", "pitcher"):
        cache_path = _team_cache_path(player_type, team, year)
        label = "sea_batters" if player_type == "batter" else "sea_pitchers"

        if not force_refresh and not _is_stale(cache_path):
            print(f"  [cache] loading team statcast {player_type}s from disk")
            results[label] = pd.read_parquet(cache_path)
            continue

        try:
            print(f"  [fetch] team statcast {player_type}s (team={team}) ...")
            df = _fetch_team_leaderboard(player_type, team, year)
            df.to_parquet(cache_path, index=False)
            print(f"    [ok]  {len(df)} {player_type}s")
            results[label] = df
        except Exception as e:
            print(f"    [warn] live fetch failed ({e})")
            if os.path.exists(cache_path):
                print(f"    [warn] using stale cache instead")
                results[label] = pd.read_parquet(cache_path)
            else:
                print(f"    [warn] no cache available -- returning empty")
                results[label] = pd.DataFrame()

    return results["sea_batters"], results["sea_pitchers"]

# exact CSV column names from savant
RENAME = {
    "last_name, first_name": "Name",
    "pa":                    "PA",
    "k_percent":             "K%",
    "bb_percent":            "BB%",
    "woba":                  "wOBA",
    "xwoba":                 "xwOBA",
    "sweet_spot_percent":    "Sweet%",
    "barrel_batted_rate":    "Barrel%",
    "hard_hit_percent":      "HardHit%",
    "avg_best_speed":        "EV50",
    "avg_hyper_speed":       "AdjEV",
    "whiff_percent":         "Whiff%",
    "swing_percent":         "Swing%",
}

# FALLBACK ONLY -- these are used solely if the live 40-man roster scrape
# (mariners_stats.py -> data["seattle"]["roster"]) is unavailable, e.g. the
# very first run before any cache exists, or bbref is unreachable. Normal
# runs use the live roster passed in via `roster_last_names` below instead,
# so this list is NOT kept up to date with trades/call-ups on purpose --
# don't rely on it being current. See roster.py.
_FALLBACK_MARINERS_BATTERS = [
    "Rodríguez, Julio", "Arozarena, Randy", "Raley, Luke",
    "Young, Cole", "Crawford, J.P.", "Raleigh, Cal",
    "Naylor, Josh", "Canzone, Dominic", "Donovan, Brendan",
    "Emerson, Colt", "Garver, Mitch", "Pereda, Jhonny",
    "Refsnyder, Rob", "Rivas, Leo", "Robles, Víctor",
    "Wisdom, Patrick", "Joe, Connor", "Bliss, Ryan",
]

_FALLBACK_MARINERS_PITCHERS = [
    "Kirby, George", "Woo, Bryan", "Hancock, Emerson",
    "Gilbert, Logan", "Castillo, Luis", "Miller, Bryce",
    "Muñoz, Andrés", "Brash, Matt", "Ferrer, José A.",
    "Bazardo, Eduard", "Criswell, Cooper", "Speier, Gabe",
    "Hoppe, Alex", "Legumina, Casey", "Davila, Nick",
    "Wilcox, Cole",
]


def _load(filepath: str, is_pitcher: bool = False) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"  [warn] not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df = df.rename(columns=RENAME)

    # for pitchers rename wOBA/xwOBA to _against
    if is_pitcher:
        df = df.rename(columns={
            "wOBA":  "wOBA_against",
            "xwOBA": "xwOBA_against",
        })

    # numeric conversion
    skip = {"Name", "year", "player_id"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def get_statcast(batter_file=BATTER_FILE,
                 pitcher_file=PITCHER_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[load] statcast batters ...")
    bat = _load(batter_file, is_pitcher=False)
    if not bat.empty:
        print(f"  [ok]  {len(bat)} batters, {len(bat.columns)} cols")

    print("[load] statcast pitchers ...")
    pit = _load(pitcher_file, is_pitcher=True)
    if not pit.empty:
        print(f"  [ok]  {len(pit)} pitchers, {len(pit.columns)} cols")

    return bat, pit


def _filter_by_roster(df, roster_keys, fallback_names):
    """
    Filters a Statcast dataframe (Name = "Last, First") down to players on
    the current roster. Prefers the live `roster_keys` set (precise
    'lastname_firstinitial' keys from roster.get_roster_keys -- NOT
    last-name-only, since the roster has real last-name collisions with
    other teams, e.g. more than one "Wilson" in MLB); falls back to the
    bundled hardcoded list -- with a warning -- if the live roster wasn't
    available.
    """
    from name_matching import key_from_last_first

    if roster_keys:
        sea = df[df["Name"].apply(key_from_last_first).isin(roster_keys)]
        return sea.reset_index(drop=True)

    print("  [warn] no live roster available -- falling back to bundled "
          "Mariners name list, which may be stale (see statcast_scraper.py)")
    sea = df[df["Name"].isin(fallback_names)]
    if sea.empty:
        last_names = [n.split(",")[0] for n in fallback_names]
        sea = df[df["Name"].str.split(",").str[0].isin(last_names)]
    return sea.reset_index(drop=True)


def get_mariners_batters(df, roster_keys=None):
    if df.empty or "Name" not in df.columns:
        return pd.DataFrame()
    return _filter_by_roster(df, roster_keys, _FALLBACK_MARINERS_BATTERS)


def get_mariners_pitchers(df, roster_keys=None):
    if df.empty or "Name" not in df.columns:
        return pd.DataFrame()
    return _filter_by_roster(df, roster_keys, _FALLBACK_MARINERS_PITCHERS)


def get_luck_analysis(df, is_pitcher=False):
    woba = "wOBA_against" if is_pitcher else "wOBA"
    xwoba = "xwOBA_against" if is_pitcher else "xwOBA"
    if df.empty or woba not in df.columns or xwoba not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["luck_gap"] = (d[woba] - d[xwoba]).round(3)
    d["verdict"]  = d["luck_gap"].apply(
        lambda x: "LUCKY" if x > 0.020
        else ("UNLUCKY" if x < -0.020 else "NEUTRAL")
    )
    if is_pitcher:
        d["verdict"] = d["luck_gap"].apply(
            lambda x: "UNLUCKY (better than ERA shows)" if x > 0.020
            else ("LUCKY (worse than ERA shows)" if x < -0.020
                  else "NEUTRAL")
        )
    cols = [c for c in ["Name","PA","BF",woba,xwoba,"luck_gap",
                         "verdict","Barrel%","HardHit%","EV50",
                         "K%","Whiff%"]
            if c in d.columns]
    return d[cols].sort_values("luck_gap", ascending=False).reset_index(drop=True)


def get_rankings(df, stat, min_pa=50, top=20, ascending=False):
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    pa_col = "BF" if "BF" in d.columns else "PA"
    if pa_col in d.columns:
        d = d[pd.to_numeric(d[pa_col], errors="coerce") >= min_pa]
    want = ["Name", pa_col, stat, "K%", "BB%", "Whiff%",
            "Barrel%", "HardHit%", "EV50"]
    cols = [c for c in want if c in d.columns]
    return (d[cols].dropna(subset=[stat])
            .sort_values(stat, ascending=ascending)
            .head(top)
            .reset_index(drop=True))


def rank_player(df, name, stat, ascending=False):
    if df.empty or stat not in df.columns:
        return {}
    col     = pd.to_numeric(df[stat], errors="coerce")
    matches = df[df["Name"].str.contains(name, case=False, na=False)]
    if matches.empty:
        return {"error": f"{name} not found"}
    val    = col[matches.index[0]]
    ranked = col.rank(ascending=ascending, method="min")
    total  = int(col.notna().sum())
    rank   = int(ranked[matches.index[0]])
    return {
        "name":   matches.iloc[0]["Name"],
        "value":  round(val, 3),
        "rank":   rank,
        "total":  total,
        "pct":    round(rank / total * 100, 1),
    }


if __name__ == "__main__":
    batters, pitchers = get_statcast()

    if batters.empty and pitchers.empty:
        print("\nSave CSVs to:")
        print(f"  {BATTER_FILE}")
        print(f"  {PITCHER_FILE}")
    else:
        print("\n── Mariners batting (sorted by xwOBA) ──")
        sea_bat = get_mariners_batters(batters)
        if not sea_bat.empty:
            cols = [c for c in ["Name","PA","wOBA","xwOBA",
                                 "Barrel%","HardHit%","EV50","K%","BB%"]
                    if c in sea_bat.columns]
            print(sea_bat[cols].sort_values("xwOBA", ascending=False)
                  .to_string(index=False))

        print("\n── Mariners batting luck ──")
        luck = get_luck_analysis(sea_bat, is_pitcher=False)
        print(luck.to_string(index=False) if not luck.empty else "  no data")

        print("\n── Mariners pitching (sorted by xwOBA) ──")
        sea_pit = get_mariners_pitchers(pitchers)
        if not sea_pit.empty:
            cols = [c for c in ["Name","BF","xwOBA_against","wOBA_against",
                                 "K%","BB%","Whiff%",
                                 "HardHit%_against","Barrel%_against"]
                    if c in sea_pit.columns]
            print(sea_pit[cols].sort_values("xwOBA_against")
                  .to_string(index=False))

        print("\n── Mariners pitching luck ──")
        pluck = get_luck_analysis(sea_pit, is_pitcher=True)
        print(pluck.to_string(index=False) if not pluck.empty else "  no data")

        print("\n── MLB xwOBA leaders batters (min 50 PA) ──")
        print(get_rankings(batters, "xwOBA", min_pa=50, top=15,
                           ascending=False).to_string(index=False))

        print("\n── MLB xwOBA leaders pitchers (min 50 BF) ──")
        print(get_rankings(pitchers, "xwOBA_against", min_pa=50, top=15,
                           ascending=True).to_string(index=False))