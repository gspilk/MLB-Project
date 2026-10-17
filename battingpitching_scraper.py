"""
batting_pitching_scraper.py
League-wide MLB batting, pitching AND fielding leaders for both AL and NL,
built from the official MLB Stats API (statsapi.mlb.com) instead of
the old source's league pages.

Same return shape and column names as the old source version,
so nothing downstream changes:
  al_teams / nl_teams       team totals ("Name"/"Tm" = full team name)
  al_players / nl_players   individual stats ("Name", "Team"/"Tm" = 3-letter code)
  all_players / all_teams   AL + NL combined

Traded players get a combined row ("2TM", "3TM") plus one row per team,
like the old source showed them.

Columns:
  batting   Name Age Team Tm Lg WAR G PA AB R H 2B 3B HR RBI SB CS BB SO
            BA OBP SLG OPS OPS+ TB GIDP HBP SH SF IBB Pos
  pitching  Name Age Team Tm Lg WAR W L W-L% ERA G GS GF CG SHO SV HLD IP
            H R ER HR BB IBB SO HBP BK WP BF ERA+ FIP WHIP H9 HR9 BB9 SO9 SO/W
  fielding  Name Age Team Tm Lg Pos G GS Inn Ch PO A E DP Fld% Rtot Rtot/yr
            Rdrs Rdrs/yr  (Rtot = Rdrs = Statcast fielding runs, see mlb_api.py)

WAR is this project's own calculation (mlb_api.py), not another site's.

Usage:
    from batting_pitching_scraper import get_batting, get_pitching, get_fielding
    bat = get_batting(2026)
    pit = get_pitching(2026)
    fld = get_fielding(2026)
"""

import os
import time
import pandas as pd

import mlb_api

# ── config ────────────────────────────────────────────────────────────────────
LEAGUES         = ["AL", "NL"]
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > CACHE_TTL_HOURS


# ── build tables ──────────────────────────────────────────────────────────────
TEAM_BAT_COLS = ["Name", "Tm", "Lg", "G", "R/G", "PA", "AB", "R", "H", "2B", "3B", "HR",
                 "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+", "TB",
                 "GIDP", "HBP", "SH", "SF", "IBB", "WAR"]
TEAM_PIT_COLS = ["Name", "Tm", "Lg", "RA/G", "W", "L", "W-L%", "ERA", "G", "GS", "GF", "CG",
                 "SHO", "SV", "IP", "H", "R", "ER", "HR", "BB", "IBB", "SO", "HBP", "BK", "WP",
                 "BF", "ERA+", "FIP", "WHIP", "H9", "HR9", "BB9", "SO9", "SO/W", "WAR"]
TEAM_FLD_COLS = ["Name", "Tm", "Lg", "G", "Inn", "Ch", "PO", "A", "E", "DP", "Fld%", "Rtot", "Rdrs"]


def _team_table(players: pd.DataFrame, stat_type: str, season: int, lc: dict,
                force_refresh: bool = False) -> pd.DataFrame:
    group = {"batting": "hitting", "pitching": "pitching", "fielding": "fielding"}[stat_type]
    games = mlb_api.games_played_by_team(season, force_refresh=force_refresh)
    teams = mlb_api.team_totals(players, group, season)
    if teams.empty:
        return teams
    g_games = teams["team_id"].map(games)
    if stat_type == "batting":
        teams = mlb_api.add_batting_rates(teams, lc)
        teams["G"] = g_games
        teams["R/G"] = (teams["R"] / teams["G"]).round(2)
        cols = TEAM_BAT_COLS
    elif stat_type == "pitching":
        teams = mlb_api.add_pitching_rates(teams, lc)
        teams["G"] = g_games
        teams["RA/G"] = (teams["R"] / teams["G"]).round(2)
        cols = TEAM_PIT_COLS
    else:
        teams = mlb_api.add_fielding_rates(teams)
        teams["G"] = g_games
        cols = TEAM_FLD_COLS
    teams["Name"] = teams["Tm"]
    out = mlb_api.shape(teams, cols)
    out["league"] = teams["league"].values
    out["stat_type"] = stat_type
    return out


def _player_table(players: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    cols = {"batting": mlb_api.BAT_COLS, "pitching": mlb_api.PIT_COLS,
            "fielding": mlb_api.FLD_COLS}[stat_type]
    out = mlb_api.shape(players, cols)
    out["league"] = players["league"].values
    out["stat_type"] = stat_type
    out["_total"] = players["_total"].values
    return out


def _parse_all(stat_type: str, season: int, force_refresh: bool = False) -> dict:
    """
    Returns the same six tables the old per-league page parsing produced:
    al_teams, al_players, nl_teams, nl_players, all_players, all_teams.
    A traded player's combined row goes in a league table only when all
    his teams were in that league; his per-team rows always do.
    """
    tables = mlb_api.get_player_tables(season, force_refresh=force_refresh)
    key = {"batting": "batting", "pitching": "pitching", "fielding": "fielding"}[stat_type]
    players = _player_table(tables[key], stat_type)
    teams = _team_table(tables[key], stat_type, season, tables["context"], force_refresh)

    result = {}
    for lg in LEAGUES:
        lg_players = players[players["league"] == lg].drop(columns="_total")
        lg_teams = teams[teams["league"] == lg] if not teams.empty else teams
        result[f"{lg.lower()}_players"] = lg_players.reset_index(drop=True)
        result[f"{lg.lower()}_teams"] = lg_teams.reset_index(drop=True)
        print(f"  [ok]   {lg} {stat_type} teams:   {len(lg_teams)} teams")
        print(f"  [ok]   {lg} {stat_type} players: {len(lg_players)} players")
    return result


# ── analysis helpers ──────────────────────────────────────────────────────────
def get_leaders(df: pd.DataFrame, stat: str,
                top: int = 10, ascending: bool = False,
                league: str = None) -> pd.DataFrame:
    """Top N players by any stat, optionally filtered by league."""
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    if league:
        df = df[df["league"] == league]
    cols = [c for c in ["Name", "Tm", "league", stat]
            if c in df.columns]
    return (df[cols]
            .dropna(subset=[stat])
            .sort_values(stat, ascending=ascending)
            .head(top)
            .reset_index(drop=True))


def get_mariners_rank(df: pd.DataFrame, stat: str,
                      ascending: bool = False) -> dict:
    """
    Where do the Mariners rank among all players in a given stat.
    Returns rank, value, and league average.
    """
    if df.empty or stat not in df.columns:
        return {}

    col = pd.to_numeric(df[stat], errors="coerce")
    lg_avg = col.mean()

    sea_mask = df["Tm"].str.contains("SEA", na=False)
    if not sea_mask.any():
        return {}

    sea_vals = col[sea_mask]
    ranked   = col.rank(ascending=ascending, method="min")

    results = []
    for idx in sea_vals.index:
        results.append({
            "Name":    df.loc[idx, "Name"],
            "value":   sea_vals[idx],
            "rank":    int(ranked[idx]),
            "total":   len(col.dropna()),
            "lg_avg":  round(lg_avg, 3),
            "gap":     round(sea_vals[idx] - lg_avg, 3),
        })
    return results


def get_team_rankings(teams_df: pd.DataFrame,
                      team: str = "Seattle Mariners") -> pd.DataFrame:
    """
    Rank all teams by every numeric stat.
    Shows where SEA stands vs all AL/NL teams.
    """
    if teams_df.empty:
        return pd.DataFrame()

    # find the team name column
    name_col = next((c for c in ["Name", "Tm", "Team", "Franchise"]
                      if c in teams_df.columns), None)
    if not name_col:
        print(f"  [warn] no name column in teams. columns: {list(teams_df.columns)}")
        return pd.DataFrame()

    numeric_cols = [c for c in teams_df.columns
                    if c not in {"Name", "Tm", "Lg", "league", "stat_type"}
                    and pd.to_numeric(teams_df[c], errors="coerce").notna().any()]

    sea = teams_df[teams_df[name_col].str.contains("SEA|Seattle", na=False)]
    if sea.empty:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        series  = pd.to_numeric(teams_df[col], errors="coerce")
        sea_val = pd.to_numeric(sea[col].values[0], errors="coerce")
        if pd.isna(sea_val):
            continue
        lg_avg  = series.mean()
        ranked  = series.rank(ascending=False, method="min")
        sea_rank = int(ranked[sea.index[0]])
        rows.append({
            "stat":     col,
            "sea_val":  sea_val,
            "lg_avg":   round(lg_avg, 3),
            "rank":     sea_rank,
            "total":    len(series.dropna()),
            "gap":      round(sea_val - lg_avg, 3),
        })

    return (pd.DataFrame(rows)
            .sort_values("rank", ascending=False)
            .reset_index(drop=True))


def compare_mariners_to_league(players_df: pd.DataFrame,
                               teams_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each SEA player, show how they rank vs all players.
    Auto-detects the team column name.
    """
    if players_df.empty:
        return pd.DataFrame()

    # find team column — Tm, Team, or similar
    tm_col = next((c for c in players_df.columns
                   if c.lower() in ("tm", "team", "franchise")), None)
    if tm_col is None:
        print(f"  [warn] no team column found. columns: {list(players_df.columns)}")
        return pd.DataFrame()

    sea_players = players_df[
        players_df[tm_col].astype(str).str.contains("SEA", na=False)
    ].copy()

    print(f"  [ok]   found {len(sea_players)} SEA players")
    return sea_players.reset_index(drop=True)


# ── main public functions ─────────────────────────────────────────────────────
def _get(stat_type: str, season: int, force_refresh: bool, sort_players, sort_teams) -> dict:
    cache_keys  = ["al_teams", "al_players", "nl_teams",
                   "nl_players", "all_players", "all_teams"]
    cache_paths = {k: _cache_path(f"{stat_type}_{k}_{season}") for k in cache_keys}

    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    if not force_refresh and all_cached and not _is_stale(cache_paths["al_players"]):
        print(f"[cache] loading {stat_type} from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] {stat_type} leaders {season} (MLB Stats API) ...")
    result = _parse_all(stat_type, season, force_refresh)

    all_players = pd.concat([result["al_players"], result["nl_players"]], ignore_index=True)
    all_teams   = pd.concat([result["al_teams"], result["nl_teams"]], ignore_index=True)
    col, asc = sort_players
    if col in all_players.columns:
        all_players = all_players.sort_values(col, ascending=asc)
    col, asc = sort_teams
    if col and col in all_teams.columns:
        all_teams = all_teams.sort_values(col, ascending=asc)
    result["all_players"] = all_players.reset_index(drop=True)
    result["all_teams"] = all_teams.reset_index(drop=True)

    for k, df in result.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)
    return result


def get_batting(season: int = 2026,
                force_refresh: bool = False) -> dict:
    """
    Returns dict:
        al_teams      AL team batting totals
        al_players    AL individual batting stats
        nl_teams      NL team batting totals
        nl_players    NL individual batting stats
        all_players   AL + NL combined, sorted by OPS
        all_teams     AL + NL combined, sorted by R
    """
    return _get("batting", season, force_refresh, ("OPS", False), ("R", False))


def get_pitching(season: int = 2026,
                 force_refresh: bool = False) -> dict:
    """
    Returns dict:
        al_teams      AL team pitching totals
        al_players    AL individual pitching stats
        nl_teams      NL team pitching totals
        nl_players    NL individual pitching stats
        all_players   AL + NL combined, sorted by ERA
        all_teams     AL + NL combined, sorted by ERA
    """
    return _get("pitching", season, force_refresh, ("ERA", True), ("ERA", True))


def get_fielding(season: int = 2026,
                 force_refresh: bool = False) -> dict:
    """
    Returns dict:
        al_teams      AL team fielding totals
        al_players    AL individual fielding stats (one row per position played)
        nl_teams      NL team fielding totals
        nl_players    NL individual fielding stats
        all_players   AL + NL combined, sorted by Rtot descending (best first)
        all_teams     AL + NL combined

    Rtot and Rdrs both hold Statcast fielding runs (Outs Above Average based;
    catchers: framing runs) -- see mlb_api.py.
    """
    return _get("fielding", season, force_refresh, ("Rtot", False), (None, False))


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── batting ──────────────────────────────────────────────────────────────
    bat = get_batting(2026, force_refresh=True)  # always fresh

    print("\n── AL team batting (sorted by R) ──")
    al_t = bat["al_teams"]
    if not al_t.empty:
        cols = [c for c in ["Name","G","R","H","HR","RBI","BA","OBP","SLG","OPS"]
                if c in al_t.columns]
        print(al_t[cols].to_string(index=False))

    print("\n── AL batting leaders OPS (top 15) ──")
    print(get_leaders(bat["al_players"], "OPS", top=15, league="AL")
          .to_string(index=False))

    print("\n── AL batting leaders HR (top 15) ──")
    print(get_leaders(bat["al_players"], "HR", top=15, league="AL")
          .to_string(index=False))

    print("\n── SEA players vs AL ──")
    sea_bat = compare_mariners_to_league(bat["al_players"], al_t)
    if not sea_bat.empty:
        cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI"]
                if c in sea_bat.columns]
        print(sea_bat[cols].to_string(index=False))

    # ── pitching ─────────────────────────────────────────────────────────────
    pit = get_pitching(2026, force_refresh=True)  # always fresh

    print("\n── AL team pitching (sorted by ERA) ──")
    al_pt = pit["al_teams"]
    if not al_pt.empty:
        cols = [c for c in ["Name","ERA","WHIP","SO","BB","HR","FIP"]
                if c in al_pt.columns]
        print(al_pt[cols].to_string(index=False))

    print("\n── AL ERA leaders starters (top 15) ──")
    starters = pit["al_players"]
    if not starters.empty:
        gs_col = "GS" if "GS" in starters.columns else None
        if gs_col:
            starters = starters.copy()
            starters[gs_col] = pd.to_numeric(starters[gs_col],
                                             errors="coerce").fillna(0)
            sp = starters[starters[gs_col] >= 5]
        else:
            sp = starters
        cols = [c for c in ["Name","Tm","GS","IP","ERA","WHIP","SO","BB","FIP"]
                if c in sp.columns]
        print(sp[cols].sort_values("ERA").head(15).to_string(index=False))

    print("\n── AL saves leaders (top 10) ──")
    print(get_leaders(pit["al_players"], "SV", top=10, league="AL")
          .to_string(index=False))

    # ── fielding ─────────────────────────────────────────────────────────────
    fld = get_fielding(2026, force_refresh=True)  # always fresh

    print("\n── AL fielding leaders by Rtot (top 15) ──")
    sort_col = "Rtot" if "Rtot" in fld["al_players"].columns else "Rdrs"
    print(get_leaders(fld["al_players"], sort_col, top=15, league="AL", ascending=False)
          .to_string(index=False))