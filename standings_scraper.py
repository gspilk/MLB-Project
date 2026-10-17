"""
standings_scraper.py
Builds MLB standings from the official MLB Stats API (statsapi.mlb.com)
instead of scraping the old source.

Every regular-season game result is pulled from MLB's schedule endpoint,
and both tables are calculated from those results:
  1. Division standings  (AL/NL x East/Central/West)
  2. Expanded standings  (all 30 teams: run diff, Pythagorean record,
     luck, SOS, SRS, home/road, 1-run, extra innings, recent form)

Same functions, column names, and return values as the old
the old source version, so nothing else in the project has to change.

vRHP / vLHP (record vs right/left-handed starters) are not in game results,
so they come from the Stats API's official standings split records
(type "right" / "left"). If that call fails they stay None.

Usage:
    from standings_scraper import get_standings
    division, expanded = get_standings(2026)

Install:
    pip install MLB-StatsAPI

Data (c) MLB Advanced Media, L.P. Personal, non-commercial use:
http://gdx.mlb.com/components/copyright.txt
"""

import os
import time
from collections import defaultdict

import pandas as pd

import mlb_api

# ── cache config ──────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > CACHE_TTL_HOURS


# ── team / division lookup ────────────────────────────────────────────────────
DIVISION_ORDER = [
    "AL_East", "AL_Central", "AL_West",
    "NL_East", "NL_Central", "NL_West",
]

# column names the old expanded table used for "record vs each division"
REGION_COL = {"East": "vEast", "Central": "vCent", "West": "vWest"}

def _fetch_teams(season: int, force_refresh: bool = False) -> dict:
    """team id -> {name, division, league}"""
    teams = mlb_api.get_teams(season, force_refresh=force_refresh)
    out = {int(r.team_id): {"name": r.name, "division": r.division, "league": r.league}
           for r in teams.itertuples()}
    print(f"  [ok]   {len(out)} teams")
    return out


def _fetch_hand_splits(season: int, force_refresh: bool = False) -> dict:
    """
    team id -> {"left": "W-L", "right": "W-L"} (record vs LHP / RHP starters)
    from the Stats API standings endpoint. Returns {} if anything goes wrong,
    so the rest of the standings still build.
    """
    try:
        data = mlb_api.api_get(
            "standings",
            {"leagueId": "103,104", "season": season,
             "standingsTypes": "regularSeason", "hydrate": "team"},
            force_refresh=force_refresh)
        out = {}
        for rec in data.get("records", []):
            for tr in rec.get("teamRecords", []):
                tid = tr.get("team", {}).get("id")
                splits = (tr.get("records") or {}).get("splitRecords") or []
                by_type = {s.get("type"): f"{int(s['wins'])}-{int(s['losses'])}"
                           for s in splits if "wins" in s and "losses" in s}
                if tid is not None and ("left" in by_type or "right" in by_type):
                    out[int(tid)] = {"left": by_type.get("left"),
                                     "right": by_type.get("right")}
        if not out:
            print("  [warn] no vs-LHP/RHP split records in the standings feed -- leaving them blank")
        return out
    except Exception as e:
        print(f"  [warn] vs-LHP/RHP split records unavailable ({e}) -- leaving them blank")
        return {}


def _fetch_games(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """One row per completed regular-season game."""
    print(f"  [GET] MLB Stats API schedule {season}")
    games = mlb_api.get_season_games(season, force_refresh=force_refresh)
    games = games[games["played"]].copy() if not games.empty else games
    print(f"  [ok]   {len(games)} completed games")
    return games


# ── calculations ──────────────────────────────────────────────────────────────
def _wl(w: int, l: int) -> str:
    return f"{w}-{l}"


def _team_logs(games: pd.DataFrame) -> dict:
    """team id -> list of that team's games, in date order, from its own view."""
    logs = defaultdict(list)
    for g in games.itertuples():
        for tid, opp, rs, ra, home in (
            (g.home_id, g.away_id, g.home_runs, g.away_runs, True),
            (g.away_id, g.home_id, g.away_runs, g.home_runs, False),
        ):
            logs[tid].append({"opp": opp, "rs": rs, "ra": ra, "home": home,
                              "won": rs > ra, "innings": g.innings})
    return logs


def _srs(logs: dict, iterations: int = 100):
    """
    Simple Rating System, same idea other sites use:
    rating = average run margin + average opponent rating.
    Returns (srs, sos) dicts in runs per game.
    """
    mov = {t: sum(g["rs"] - g["ra"] for g in gl) / len(gl) for t, gl in logs.items()}
    srs = dict(mov)
    for _ in range(iterations):
        sos = {t: sum(srs[g["opp"]] for g in gl) / len(gl) for t, gl in logs.items()}
        srs = {t: mov[t] + sos[t] for t in logs}
        mean = sum(srs.values()) / len(srs)
        srs = {t: v - mean for t, v in srs.items()}
    sos = {t: sum(srs[g["opp"]] for g in gl) / len(gl) for t, gl in logs.items()}
    return srs, sos


def _build_expanded(teams: dict, games: pd.DataFrame, hand_splits: dict = None) -> pd.DataFrame:
    hand_splits = hand_splits or {}
    logs = _team_logs(games)
    srs, sos = _srs(logs)

    season_pct = {t: sum(g["won"] for g in gl) / len(gl) for t, gl in logs.items()}

    rows = []
    for tid, info in teams.items():
        gl = logs.get(tid, [])
        if not gl:
            continue
        w = sum(g["won"] for g in gl)
        l = len(gl) - w
        rs = sum(g["rs"] for g in gl)
        ra = sum(g["ra"] for g in gl)

        def rec(filt):
            sub = [g for g in gl if filt(g)]
            won = sum(g["won"] for g in sub)
            return _wl(won, len(sub) - won)

        def last(n):
            sub = gl[-n:]
            won = sum(g["won"] for g in sub)
            return _wl(won, len(sub) - won)

        # Pythagorean record (exponent 1.83, a standard choice)
        pyth_pct = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83) if rs + ra else 0.5
        pyth_w = round(pyth_pct * len(gl))

        # current streak, e.g. "W 3" / "L 2"
        streak = 0
        for g in reversed(gl):
            if g["won"] != gl[-1]["won"]:
                break
            streak += 1

        row = {
            "Tm": info["name"], "Lg": info["league"],
            "G": len(gl), "W": w, "L": l,
            "W-L%": round(w / len(gl), 3),
            "Strk": f"{'W' if gl[-1]['won'] else 'L'} {streak}",
            "R": round(rs / len(gl), 2), "RA": round(ra / len(gl), 2),
            "Rdiff": round((rs - ra) / len(gl), 2),
            "SOS": round(sos[tid], 2), "SRS": round(srs[tid], 2),
            "pythWL": _wl(pyth_w, len(gl) - pyth_w),
            "Luck": w - pyth_w,
        }
        for region, col in REGION_COL.items():
            row[col] = rec(lambda g, r=region: teams.get(g["opp"], {}).get("division", "")
                           == f"{info['league']}_{r}")
        row["Inter"] = rec(lambda g: teams.get(g["opp"], {}).get("league") != info["league"])
        row["Home"] = rec(lambda g: g["home"])
        row["Road"] = rec(lambda g: not g["home"])
        row["ExInn"] = rec(lambda g: g["innings"] > 9)
        row["1Run"] = rec(lambda g: abs(g["rs"] - g["ra"]) == 1)
        row["vRHP"] = hand_splits.get(tid, {}).get("right")
        row["vLHP"] = hand_splits.get(tid, {}).get("left")
        row["≥.500"] = rec(lambda g: season_pct.get(g["opp"], 0) >= 0.5)
        row["<.500"] = rec(lambda g: season_pct.get(g["opp"], 0) < 0.5)
        row["last10"] = last(10)
        row["last20"] = last(20)
        row["last30"] = last(30)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("W-L%", ascending=False).reset_index(drop=True)
    df.insert(0, "Rk", range(1, len(df) + 1))
    print(f"  [ok]   expanded standings: {len(df)} teams, {len(df.columns)} cols")
    return df


def _games_back(df: pd.DataFrame) -> pd.Series:
    lead = df.iloc[0]
    gb = ((lead["W"] - df["W"]) + (df["L"] - lead["L"])) / 2
    return gb.map(lambda x: "--" if x == 0 else f"{x:.1f}")


def _build_division_tables(teams: dict, expanded: pd.DataFrame) -> dict:
    name_to_div = {i["name"]: i["division"] for i in teams.values()}
    df = expanded.copy()
    df["division"] = df["Tm"].map(name_to_div)
    df["league"] = df["division"].str.split("_").str[0]

    tables = {}
    for div in DIVISION_ORDER:
        d = df[df["division"] == div].sort_values("W-L%", ascending=False)
        if d.empty:
            print(f"  [warn] missing: {div}")
            continue
        d = d[["Tm", "W", "L", "W-L%", "division", "league"]].reset_index(drop=True)
        d.insert(4, "GB", _games_back(d))
        tables[div] = d
        print(f"  [ok]   {div}: {len(d)} teams")
    return tables


# ── public helpers (unchanged) ────────────────────────────────────────────────
def get_al_west(division_tables: dict) -> pd.DataFrame:
    return division_tables.get("AL_West", pd.DataFrame())


def get_nl_west(division_tables: dict) -> pd.DataFrame:
    return division_tables.get("NL_West", pd.DataFrame())


def get_team_row(expanded_df, team: str = "Seattle Mariners"):
    if expanded_df is None or expanded_df.empty:
        return None
    matches = expanded_df[expanded_df["Tm"] == team]
    if matches.empty:
        print(f"  [warn] '{team}' not found in expanded standings")
        return None
    return matches.iloc[0]


def get_all_teams(division_tables: dict) -> pd.DataFrame:
    """All 30 teams across all 6 divisions, sorted by W-L%."""
    frames = [division_tables[d] for d in DIVISION_ORDER if d in division_tables]
    if not frames:
        return pd.DataFrame()
    all_teams = pd.concat(frames, ignore_index=True)
    for col in ["W-L%", "W", "L", "GB"]:
        all_teams[col] = pd.to_numeric(all_teams[col], errors="coerce")
    return all_teams.sort_values("W-L%", ascending=False).reset_index(drop=True)


def get_playoff_picture(division_tables: dict) -> pd.DataFrame:
    """All AL teams sorted by W-L%, with GB measured from the 3rd wild card."""
    al_divs = ["AL_East", "AL_Central", "AL_West"]
    frames = [division_tables[d] for d in al_divs if d in division_tables]
    if not frames:
        return pd.DataFrame()
    al = pd.concat(frames, ignore_index=True)
    al["W-L%"] = pd.to_numeric(al["W-L%"], errors="coerce")
    al = al.sort_values("W-L%", ascending=False).reset_index(drop=True)

    # Wild card games back: division winners don't count toward the 3 WC spots
    leaders = {f[["Tm", "W-L%"]].iloc[0]["Tm"] for f in frames}
    wc_pool = al[~al["Tm"].isin(leaders)].reset_index(drop=True)
    if len(wc_pool) >= 3:
        cut = wc_pool.iloc[2]
        gb = ((cut["W"] - al["W"]) + (al["L"] - cut["L"])) / 2
        al["GB"] = gb.map(lambda x: "--" if x <= 0 else f"{x:.1f}")
        al.loc[al["Tm"].isin(leaders), "GB"] = "--"
    return al


def get_mariners_context(division_tables: dict, expanded_df) -> dict:
    """
    Returns a dict with:
      mlb_rank      — Mariners rank among all 30 teams
      div_rank      — rank in AL West
      wc_gap        — W-L% gap vs the 6th-best AL team (negative = behind)
      wc_games_back — real games back of the 3rd wild card
      wc_in_reach   — True if within ~6 games
      expanded_row  — full row from expanded standings
    """
    all_teams = get_all_teams(division_tables)
    mlb_rank = all_teams[all_teams["Tm"] == "Seattle Mariners"].index
    mlb_rank = int(mlb_rank[0]) + 1 if len(mlb_rank) else None

    al_west = get_al_west(division_tables).reset_index(drop=True)
    div_rank = al_west[al_west["Tm"] == "Seattle Mariners"].index
    div_rank = int(div_rank[0]) + 1 if len(div_rank) else None

    al_pic = get_playoff_picture(division_tables)
    wc_cutoff = al_pic.iloc[5]["W-L%"] if len(al_pic) >= 6 else None
    sea_row = al_pic[al_pic["Tm"] == "Seattle Mariners"]
    sea_pct = float(sea_row["W-L%"].values[0]) if not sea_row.empty else None
    wc_gap = round(sea_pct - wc_cutoff, 3) if sea_pct and wc_cutoff else None

    wc_games_back = None
    if not sea_row.empty and "GB" in sea_row.columns:
        gb_val = sea_row["GB"].values[0]
        wc_games_back = 0.0 if str(gb_val).strip() == "--" else \
                        pd.to_numeric(gb_val, errors="coerce")

    return {
        "mlb_rank":      mlb_rank,
        "div_rank":      div_rank,
        "wc_gap":        wc_gap,
        "wc_games_back": wc_games_back,
        "wc_in_reach":   wc_gap is not None and wc_gap > -0.060,
        "expanded_row":  get_team_row(expanded_df, "Seattle Mariners"),
    }


def get_category_leaders(expanded_df, team: str = "Seattle Mariners") -> pd.DataFrame:
    """
    For every numeric column in the expanded standings: MLB leader, Mariners
    value, Mariners rank out of 30, and gap. Sorted worst rank first.
    """
    if expanded_df is None or expanded_df.empty:
        print("  [warn] no expanded data for category leaders")
        return pd.DataFrame()

    lower_is_better = {"RA", "L"}
    skip = {"Rk", "Tm", "Lg", "G", "Strk", "pythWL", "vEast", "vCent", "vWest", "Inter",
            "Home", "Road", "ExInn", "1Run", "vRHP", "vLHP", "≥.500", "<.500",
            "last10", "last20", "last30"}
    num_cols = [c for c in expanded_df.columns if c not in skip
                and pd.to_numeric(expanded_df[c], errors="coerce").notna().any()]

    sea = expanded_df[expanded_df["Tm"] == team]
    if sea.empty:
        print(f"  [warn] '{team}' not found")
        return pd.DataFrame()

    rows = []
    for col in num_cols:
        series = pd.to_numeric(expanded_df[col], errors="coerce")
        sea_val = pd.to_numeric(sea[col].values[0], errors="coerce")
        if pd.isna(sea_val):
            continue
        ascending = col in lower_is_better
        ranked = series.rank(ascending=ascending, method="min")
        leader_idx = series.idxmin() if ascending else series.idxmax()
        rows.append({
            "category":    col,
            "leader_team": expanded_df.loc[leader_idx, "Tm"],
            "leader_val":  series[leader_idx],
            "sea_val":     sea_val,
            "sea_rank":    int(ranked[sea.index[0]]),
            "gap":         round(sea_val - series[leader_idx], 3),
        })

    return (pd.DataFrame(rows).sort_values("sea_rank", ascending=False)
            .reset_index(drop=True))


def print_mariners_vs_leaders(expanded_df, team: str = "Seattle Mariners"):
    df = get_category_leaders(expanded_df, team)
    if df.empty:
        return
    print(f"\n{'category':<10} {'leader':<22} {'leader val':>10} "
          f"{'SEA val':>9} {'SEA rank':>9} {'gap':>8}")
    print("─" * 75)
    for _, row in df.iterrows():
        print(f"{row['category']:<10} {row['leader_team']:<22} "
              f"{row['leader_val']:>10.3f} {row['sea_val']:>9.3f} "
              f"{str(int(row['sea_rank'])) + '/30':>9} {row['gap']:>8.3f}")


# ── main public function ──────────────────────────────────────────────────────
def get_standings(season: int = 2026, force_refresh: bool = False):
    """
    Returns:
        division_tables  dict of 6 DataFrames keyed by division name
        expanded_df      single DataFrame, all 30 teams + advanced splits
    """
    expanded_cache = _cache_path(f"standings_expanded_{season}")
    division_cache = _cache_path(f"standings_division_{season}")

    if (not force_refresh
            and not _is_stale(expanded_cache)
            and not _is_stale(division_cache)):
        print("[cache] loading standings from disk")
        expanded_df = pd.read_parquet(expanded_cache)
        combined = pd.read_parquet(division_cache)
        division_tables = {d: grp.reset_index(drop=True)
                           for d, grp in combined.groupby("division")}
        return division_tables, expanded_df

    print(f"[fetch] standings {season} (MLB Stats API) ...")
    teams = _fetch_teams(season, force_refresh)
    games = _fetch_games(season, force_refresh)
    hand_splits = _fetch_hand_splits(season, force_refresh)
    expanded_df = _build_expanded(teams, games, hand_splits)
    division_tables = _build_division_tables(teams, expanded_df)

    pd.concat(division_tables.values(), ignore_index=True).to_parquet(division_cache, index=False)
    expanded_df.to_parquet(expanded_cache, index=False)
    return division_tables, expanded_df


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    division, expanded = get_standings(2026, force_refresh=True)

    print("\n── all 30 teams (sorted by W-L%) ──")
    all30 = get_all_teams(division)
    print(all30[["Tm", "W", "L", "W-L%", "division", "league"]].to_string(index=False))

    print("\n── AL West ──")
    print(get_al_west(division)[["Tm", "W", "L", "W-L%", "GB"]].to_string(index=False))

    print("\n── Mariners context ──")
    ctx = get_mariners_context(division, expanded)
    print(f"  MLB rank:    {ctx['mlb_rank']} of 30")
    print(f"  Division:    {ctx['div_rank']} of 5 in AL West")
    print(f"  WC gap:      {ctx['wc_gap']} (negative = behind wild card)")
    print(f"  WC games back: {ctx['wc_games_back']}")
    print(f"  WC in reach: {ctx['wc_in_reach']}")
    if ctx["expanded_row"] is not None:
        print("\n── Mariners expanded row ──")
        print(ctx["expanded_row"].to_string())

    print("\n── Mariners vs MLB leaders (worst categories first) ──")
    print_mariners_vs_leaders(expanded)