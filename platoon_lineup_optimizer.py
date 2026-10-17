"""
platoon_lineup_optimizer.py
Real vs-LHP/vs-RHP platoon splits for the Mariners' own current
roster -- a genuinely bounded, small set of real players (~25-40),
unlike the 278-player free agent list where bulk splits were
confirmed impractical. Given today's opposing starter's real
handedness, recommends the actual better lineup using real splits
data instead of season-total stats, which is all the rest of this
project currently sees.

Data: the official MLB Stats API (statsapi.mlb.com) "statSplits" with
situation codes vl (vs left-handed pitching) and vr (vs right-handed
pitching), for players and for the team as a whole. Player ids are
MLBAM ids straight from the roster, so no separate id lookup is needed.

Usage:
    from platoon_lineup_optimizer import get_roster_ids, get_platoon_splits, recommend_lineup
"""

import pandas as pd

import mlb_api

SEASON = 2026
TEAM_ID = mlb_api.SEA_TEAM_ID
CACHE_TTL_HOURS = 24 * 7  # splits change slowly during a season


def get_roster_ids(season: int = SEASON) -> dict:
    """{player_name: mlbam_id} for the current Mariners 40-man roster."""
    roster = mlb_api.get_roster(TEAM_ID, season, "40Man")
    ids = {r.Name: int(r.player_id) for r in roster.itertuples() if pd.notna(r.player_id)}
    print(f"  [ok] {len(ids)} player IDs found on the roster")
    return ids


# old name kept so existing calls keep working
get_roster_player_ids = get_roster_ids


def _stat(split: dict, key: str):
    return pd.to_numeric((split or {}).get(key), errors="coerce")


def get_platoon_splits(player_name: str, player_id: int, player_type: str = "batter",
                       season: int = SEASON, force_refresh: bool = False) -> dict:
    """
    vs-LHP / vs-RHP for ONE player: OPS for batters, ERA for pitchers.
    Returns {"name", "vs_LHP", "vs_RHP"} (None where there's no data).
    """
    group = "hitting" if player_type == "batter" else "pitching"
    stat_key = "ops" if player_type == "batter" else "era"
    try:
        splits = mlb_api.get_player_platoon(int(player_id), season, group, force_refresh)
    except Exception as e:
        print(f"  [warn] no platoon splits for {player_name}: {e}")
        splits = {}
    vs_l, vs_r = _stat(splits.get("vl"), stat_key), _stat(splits.get("vr"), stat_key)
    return {
        "name": player_name,
        "vs_LHP": None if pd.isna(vs_l) else float(vs_l),
        "vs_RHP": None if pd.isna(vs_r) else float(vs_r),
    }


def get_team_platoon_splits(team: str = "SEA", season: int = SEASON) -> dict:
    """Team-wide OPS vs LHP and vs RHP ("is the TEAM better vs one hand")."""
    team_id = mlb_api.team_id_for(team, season)
    try:
        splits = mlb_api.get_team_platoon(team_id, season, "hitting")
    except Exception as e:
        print(f"  [warn] no team platoon splits: {e}")
        splits = {}
    vs_l, vs_r = _stat(splits.get("vl"), "ops"), _stat(splits.get("vr"), "ops")
    return {
        "team": team,
        "vs_LHP_OPS": None if pd.isna(vs_l) else float(vs_l),
        "vs_RHP_OPS": None if pd.isna(vs_r) else float(vs_r),
    }


def recommend_lineup(all_splits: list, opposing_starter_throws: str) -> pd.DataFrame:
    """
    Given real splits for a group of players and today's real opposing
    starter's handedness ('L' or 'R'), ranks them by their real
    performance in that specific matchup -- the actual point of
    pulling platoon splits in the first place.
    """
    stat_col = "vs_LHP" if opposing_starter_throws == "L" else "vs_RHP"
    df = pd.DataFrame(all_splits)
    df = df[df[stat_col].notna()].sort_values(stat_col, ascending=False)
    return df[["name", stat_col]]


if __name__ == "__main__":
    print("Getting player IDs from the current Seattle roster...")
    ids = get_roster_ids()
    for name, pid in list(ids.items())[:10]:
        print(f"  {name}: {pid}")

    print("\nGetting real team-wide platoon splits...")
    team_splits = get_team_platoon_splits()
    print(team_splits)
