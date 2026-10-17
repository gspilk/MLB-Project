"""
seattle_scraper.py
Seattle Mariners player stats from the official MLB Stats API
(statsapi.mlb.com), replacing the old source team page.

Tables (same names and columns as the old source version):
    batting          standard batting per player
    pitching         standard pitching per player
    fielding         fielding per player per position (Rtot/Rdrs = Statcast
                     fielding runs; "Player" duplicates "Name" for scripts
                     that used the old source's fielding header)
    value_batting    WAR + components (Rbat, Rbaser, Rfield, Rpos, Rrep ...)
    value_pitching   WAR + RA9 components (RA9, RA9def, RA9role, RA9avg, gmLI)
    roster           40-man roster with OnActv ("*" = active) and IL columns

Players currently on the injured list get the same "(10-day IL)" style note
on their Name in batting/pitching that the old source used, which
recommender.py relies on.

WAR is this project's own calculation (see mlb_api.py).

Usage:
    from seattle_scraper import get_seattle_stats
    stats = get_seattle_stats(2026)
"""

import os
import time
import pandas as pd

import mlb_api

# ── config ────────────────────────────────────────────────────────────────────
TEAM            = "SEA"
TEAM_ID         = mlb_api.SEA_TEAM_ID
SEASON          = 2026
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
def _team_rows(df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    return df[(df["team_id"] == team_id) & ~df["_total"]].copy()


def _with_il_note(df: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if roster.empty or df.empty:
        return df
    il = roster.dropna(subset=["IL"]).set_index("player_id")["IL"]
    df = df.copy()
    note = df["player_id"].map(il)
    df["Name"] = [f"{n} ({x})" if isinstance(x, str) else n for n, x in zip(df["Name"], note)]
    return df


def build_team_stats(team_id: int, season: int, force_refresh: bool = False) -> dict:
    """All six tables for any team (used for Seattle here, and by
    trade_cost_package.py for players' new teams)."""
    tables = mlb_api.get_player_tables(season, force_refresh=force_refresh)
    try:
        roster = mlb_api.get_roster(team_id, season, "40Man", force_refresh=force_refresh)
    except Exception as e:
        print(f"  [warn] 40-man roster unavailable: {e}")
        roster = pd.DataFrame(columns=["player_id", "Name", "Pos", "OnActv", "IL", "B", "T",
                                       "birth_date", "status"])

    bat = _with_il_note(_team_rows(tables["batting"], team_id), roster)
    pit = _with_il_note(_team_rows(tables["pitching"], team_id), roster)
    fld = _team_rows(tables["fielding"], team_id)

    stats = {
        "batting":        mlb_api.shape(bat.sort_values("PA", ascending=False), mlb_api.BAT_COLS),
        "pitching":       mlb_api.shape(pit.sort_values("_outs", ascending=False), mlb_api.PIT_COLS),
        "fielding":       mlb_api.shape(fld.sort_values("_inn_outs", ascending=False), mlb_api.FLD_COLS),
        "value_batting":  mlb_api.shape(bat.sort_values("WAR", ascending=False), mlb_api.VAL_BAT_COLS),
        "value_pitching": mlb_api.shape(pit.sort_values("WAR", ascending=False), mlb_api.VAL_PIT_COLS),
    }
    stats["fielding"].insert(1, "Player", stats["fielding"]["Name"])

    if not roster.empty:
        r = roster.copy()
        r["Age"] = r["birth_date"].map(lambda d: mlb_api.season_age(d, season))
        stats["roster"] = r[["Name", "OnActv", "IL", "Pos", "B", "T", "Age", "status",
                             "player_id"]].reset_index(drop=True)
    else:
        stats["roster"] = pd.DataFrame()

    for label, df in stats.items():
        print(f"  [ok]   {label}: {len(df)} players, {len(df.columns)} cols")
    return stats


# ── analysis helpers ──────────────────────────────────────────────────────────
def get_rotation(pitching_df: pd.DataFrame) -> pd.DataFrame:
    """Starting pitchers — GS >= 1, sorted by GS."""
    if pitching_df.empty or "GS" not in pitching_df.columns:
        return pd.DataFrame()
    df = pitching_df.copy()
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
    return (df[df["GS"] >= 1]
            .sort_values("GS", ascending=False)
            .reset_index(drop=True))


def get_bullpen(pitching_df: pd.DataFrame) -> pd.DataFrame:
    """Relief pitchers — GS = 0."""
    if pitching_df.empty or "GS" not in pitching_df.columns:
        return pd.DataFrame()
    df = pitching_df.copy()
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
    df["G"]  = pd.to_numeric(df["G"],  errors="coerce").fillna(0)
    return (df[df["GS"] == 0]
            .sort_values("G", ascending=False)
            .reset_index(drop=True))


def get_struggling_batters(batting_df: pd.DataFrame,
                           avg_threshold: float = 0.200,
                           min_pa: int = 50) -> pd.DataFrame:
    """Batters hitting below threshold with min PA."""
    if batting_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI"]
            if c in batting_df.columns]
    df   = batting_df[cols].dropna(subset=["BA","PA"])
    df   = df[df["PA"] >= min_pa]
    return (df[df["BA"] <= avg_threshold]
            .sort_values("BA")
            .reset_index(drop=True))


def get_struggling_pitchers(pitching_df: pd.DataFrame,
                            era_threshold: float = 5.00,
                            min_ip: float = 5.0) -> pd.DataFrame:
    """Pitchers with ERA above threshold and min IP."""
    if pitching_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Name","G","GS","IP","ERA","WHIP","SO","BB"]
            if c in pitching_df.columns]
    df   = pitching_df[cols].dropna(subset=["ERA","IP"])
    df   = df[df["IP"] >= min_ip]
    return (df[df["ERA"] >= era_threshold]
            .sort_values("ERA", ascending=False)
            .reset_index(drop=True))


def get_war_leaders(value_df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Top players by WAR."""
    if value_df.empty:
        return pd.DataFrame()
    war_col = next((c for c in value_df.columns
                    if "WAR" in str(c) and "162" not in str(c)), None)
    if not war_col:
        return pd.DataFrame()
    return (value_df[["Name", war_col]]
            .dropna()
            .sort_values(war_col, ascending=False)
            .head(top)
            .reset_index(drop=True))


def get_leaders(df: pd.DataFrame, stat: str,
                top: int = 5, ascending: bool = False) -> pd.DataFrame:
    """Top N players by any stat."""
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    return (df[["Name", stat]]
            .dropna()
            .sort_values(stat, ascending=ascending)
            .head(top)
            .reset_index(drop=True))


def get_player(df: pd.DataFrame, name: str) -> pd.Series:
    """Pull single player by partial name match."""
    if df.empty or "Name" not in df.columns:
        return pd.Series()
    matches = df[df["Name"].str.contains(name, case=False, na=False)]
    return matches.iloc[0] if not matches.empty else pd.Series()


# ── main public function ──────────────────────────────────────────────────────
def get_seattle_stats(season: int = 2026,
                      force_refresh: bool = False) -> dict:
    """
    Returns dict:
        batting          standard batting per player
        pitching         standard pitching per player
        fielding         fielding stats per player
        value_batting    WAR + value per batter
        value_pitching   WAR + value per pitcher
        roster           40-man roster
    """
    keys        = ["batting","pitching","fielding",
                   "value_batting","value_pitching","roster"]
    cache_paths = {k: _cache_path(f"sea_{k}_{season}") for k in keys}

    all_cached = all(os.path.exists(p) for p in cache_paths.values())
    if not force_refresh and all_cached and not _is_stale(cache_paths["batting"]):
        print("[cache] loading seattle stats from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] seattle stats {season} (MLB Stats API) ...")
    stats = build_team_stats(TEAM_ID, season, force_refresh)

    for k, df in stats.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return stats


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stats = get_seattle_stats(2026, force_refresh=True)

    # ── batting ──
    bat = stats["batting"]
    print("\n── all batters (sorted by OPS) ──")
    if not bat.empty:
        cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI","SB"]
                if c in bat.columns]
        df   = bat[cols].copy()
        for c in ["OPS","PA","BA","OBP","SLG","HR","RBI","SB"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["OPS"]) if "OPS" in df.columns else df
        print(df.sort_values("OPS", ascending=False).to_string(index=False))

    # ── rotation ──
    pit = stats["pitching"]
    print("\n── rotation (GS >= 1) ──")
    rot = get_rotation(pit)
    if not rot.empty:
        cols = [c for c in ["Name","GS","IP","ERA","WHIP","SO","BB","W","L"]
                if c in rot.columns]
        print(rot[cols].to_string(index=False))

    # ── bullpen ──
    print("\n── bullpen ──")
    bp = get_bullpen(pit)
    if not bp.empty:
        cols = [c for c in ["Name","G","IP","ERA","WHIP","SO","BB","SV"]
                if c in bp.columns]
        print(bp[cols].to_string(index=False))

    # ── WAR leaders ──
    print("\n── WAR leaders (batting) ──")
    war = get_war_leaders(stats["value_batting"])
    print(war.to_string(index=False) if not war.empty else "  no data")

    print("\n── WAR leaders (pitching) ──")
    war = get_war_leaders(stats["value_pitching"])
    print(war.to_string(index=False) if not war.empty else "  no data")

    # ── struggling ──
    print("\n── struggling batters (BA <= .200, min 50 PA) ──")
    s = get_struggling_batters(bat)
    print(s.to_string(index=False) if not s.empty else "  none")

    print("\n── struggling pitchers (ERA >= 5.00, min 5 IP) ──")
    sp = get_struggling_pitchers(pit)
    print(sp.to_string(index=False) if not sp.empty else "  none")