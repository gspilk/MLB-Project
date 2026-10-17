"""
minor_league_scraper.py

Seattle Mariners minor-league affiliate + organization stats from the
official MLB Stats API (statsapi.mlb.com), replacing the old source's
affiliate page.

Tables returned by get_minor_league_data() (same keys as before):
    affiliate_batting    one row per affiliate team: Tm, Lev, team totals
    affiliate_pitching   one row per affiliate team
    affiliate_fielding   one row per player per position per affiliate
    org_batting          one row per player across the whole system:
                         Name, Age, Tm ("Tacoma,Arkansas"), Lev ("AA,AAA"),
                         G PA AB R H 2B 3B HR RBI SB CS BB SO BA OBP SLG OPS
    org_pitching         Name, Age, Tm, Lev, W L ERA G GS SV IP H R ER HR BB
                         SO WHIP H9 HR9 BB9 SO9 SO/W

Levels: AAA, AA, A+, A, Rk -- the same tokens minor_league_shortlist.py
already filters on.

Top prospect rankings (Baseball America / MLB Pipeline / BP lists) aren't in
the MLB Stats API, so get_top_prospects() reads a small CSV you keep:
    data/inputs/top_prospects.csv   columns: source, rank, name, position
"""

import os
import time

import numpy as np
import pandas as pd

import mlb_api

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
TTL_HOURS = 24

TABLE_KEYS = ["affiliate_batting", "affiliate_pitching", "affiliate_fielding",
              "org_batting", "org_pitching"]
LEVEL_ORDER = {"Rk": 0, "A": 1, "A+": 2, "AA": 3, "AAA": 4}


def _cache_path(team: str, season: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"minor_league_{team}_{season}.parquet")


def _is_stale(path: str, ttl_hours: int = TTL_HOURS) -> bool:
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > ttl_hours


def get_affiliates(team_id: int, season: int, force_refresh: bool = False) -> pd.DataFrame:
    """team_id, name, sport_id, Lev for every minor-league affiliate."""
    resp = mlb_api.api_get("teams_affiliates", {"teamIds": team_id, "season": season},
                           ttl_hours=24 * 7, force_refresh=force_refresh)
    rows = []
    for t in resp.get("teams", []):
        sport = (t.get("sport") or {}).get("id")
        if sport in mlb_api.LEVELS:
            rows.append({"team_id": t["id"], "name": t.get("name"), "sport_id": sport,
                         "Lev": mlb_api.LEVELS[sport]})
    return pd.DataFrame(rows, columns=["team_id", "name", "sport_id", "Lev"])


def _fetch_level(affiliates: pd.DataFrame, group: str, season: int,
                 force_refresh: bool) -> pd.DataFrame:
    frames = []
    for a in affiliates.itertuples():
        try:
            df = mlb_api.fetch_player_stats(season, group, team_id=a.team_id,
                                            sport_id=a.sport_id, force_refresh=force_refresh)
        except Exception as e:
            print(f"  [warn] {a.name} {group}: {e}")
            continue
        df["team_id"] = a.team_id
        df["Tm"] = a.name
        df["Lev"] = a.Lev
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _join_levels(values) -> str:
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return ",".join(sorted(seen, key=lambda x: LEVEL_ORDER.get(x, 9)))


def _org_rollup(df: pd.DataFrame, group: str, ages: dict) -> pd.DataFrame:
    """One row per player across every affiliate he played for."""
    if df.empty:
        return df
    counting = [c for c in mlb_api.GROUP_MAP[group].values() if c in df.columns]
    agg = df.groupby("player_id")[counting].sum(min_count=1)
    meta = df.groupby("player_id").agg(Name=("Name", "first"),
                                       Tm=("Tm", lambda s: ",".join(dict.fromkeys(s))),
                                       Lev=("Lev", _join_levels))
    out = meta.join(agg).reset_index()
    out["Age"] = out["player_id"].map(ages)
    if group == "hitting":
        out = mlb_api.add_batting_rates(out)
        cols = ["Name", "Age", "Tm", "Lev", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI",
                "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "TB", "GIDP", "HBP", "SH",
                "SF", "IBB", "player_id"]
    else:
        out = mlb_api.add_pitching_rates(out)
        cols = ["Name", "Age", "Tm", "Lev", "W", "L", "W-L%", "ERA", "G", "GS", "GF", "CG",
                "SHO", "SV", "IP", "H", "R", "ER", "HR", "BB", "IBB", "SO", "HBP", "BK", "WP",
                "BF", "WHIP", "H9", "HR9", "BB9", "SO9", "SO/W", "player_id"]
    return mlb_api.shape(out, cols)


def _affiliate_rollup(df: pd.DataFrame, group: str, affiliates: pd.DataFrame) -> pd.DataFrame:
    """One row per affiliate team (team totals)."""
    if df.empty:
        return df
    counting = [c for c in mlb_api.GROUP_MAP[group].values() if c in df.columns]
    agg = df.groupby("team_id")[counting].sum(min_count=1).reset_index()
    agg = agg.merge(affiliates.rename(columns={"name": "Tm"})[["team_id", "Tm", "Lev"]], on="team_id")
    agg["Aff"] = "SEA"
    if group == "hitting":
        agg = mlb_api.add_batting_rates(agg)
        cols = ["Tm", "Lev", "Aff", "G", "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB",
                "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS"]
    else:
        agg = mlb_api.add_pitching_rates(agg)
        cols = ["Tm", "Lev", "Aff", "W", "L", "ERA", "G", "GS", "SV", "IP", "H", "R", "ER",
                "HR", "BB", "SO", "WHIP", "SO9", "BB9"]
    agg["_lev_order"] = agg["Lev"].map(LEVEL_ORDER)
    agg = agg.sort_values("_lev_order", ascending=False)
    return mlb_api.shape(agg, cols)


def get_minor_league_data(team: str = "SEA", season: int = 2026, force_refresh: bool = False) -> dict:
    """
    Affiliate + organizational batting/pitching/fielding tables for one
    team-season, cached to disk (parquet), same TTL convention as the rest
    of this project. Any table with no data comes back as an empty
    DataFrame rather than raising.
    """
    cache_path = _cache_path(team, season)
    if not force_refresh and not _is_stale(cache_path):
        print("[cache] loading minor league data from disk")
        cached = pd.read_parquet(cache_path)
        return {
            name: cached[cached["_table"] == name].drop(columns=["_table"])
                  .dropna(axis=1, how="all").reset_index(drop=True)
            for name in TABLE_KEYS
        }

    team_id = mlb_api.team_id_for(team, season)
    print(f"[fetch] {team} minor league system {season} (MLB Stats API) ...")
    affiliates = get_affiliates(team_id, season, force_refresh)
    print(f"  [ok] {len(affiliates)} affiliates: "
          + ", ".join(f"{a.name} ({a.Lev})" for a in affiliates.itertuples()))

    hit = _fetch_level(affiliates, "hitting", season, force_refresh)
    pit = _fetch_level(affiliates, "pitching", season, force_refresh)
    fld = _fetch_level(affiliates, "fielding", season, force_refresh)

    ids = set(hit.get("player_id", [])) | set(pit.get("player_id", []))
    people = mlb_api.get_people(ids, force_refresh=force_refresh)
    ages = {r.player_id: mlb_api.season_age(r.birth_date, season) for r in people.itertuples()}

    if not fld.empty:
        fld = mlb_api.add_fielding_rates(fld)
        fld["Age"] = fld["player_id"].map(ages)
        fld = mlb_api.shape(fld, ["Name", "Age", "Tm", "Lev", "Pos", "G", "GS", "Inn", "Ch",
                                  "PO", "A", "E", "DP", "Fld%", "player_id"])

    results = {
        "affiliate_batting": _affiliate_rollup(hit, "hitting", affiliates),
        "affiliate_pitching": _affiliate_rollup(pit, "pitching", affiliates),
        "affiliate_fielding": fld,
        "org_batting": _org_rollup(hit, "hitting", ages),
        "org_pitching": _org_rollup(pit, "pitching", ages),
    }
    for key, frame in results.items():
        print(f"  [ok] {key}: {len(frame)} rows")

    tagged = []
    for name, frame in results.items():
        if frame.empty:
            continue
        f = frame.copy()
        f["_table"] = name
        tagged.append(f)
    if tagged:
        combined = pd.concat(tagged, ignore_index=True)
        for c in combined.columns:
            if combined[c].dtype == object:
                combined[c] = combined[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        combined.to_parquet(cache_path)

    return results


def get_top_prospects(team: str = "SEA", season: int = 2026, force_refresh: bool = False) -> pd.DataFrame:
    """
    Top prospect rankings for the system, from a CSV you maintain by hand
    (rankings aren't part of the MLB Stats API):

        data/inputs/top_prospects.csv
        source,rank,name,position
        MLB Pipeline,1,Colt Emerson,SS

    Returns columns: source, rank, name, position. Empty if the file hasn't
    been filled in yet (a blank template is created the first time).
    """
    df = mlb_api.read_input_csv("top_prospects.csv", ["source", "rank", "name", "position"],
                                "top prospect rankings")
    if not df.empty:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"]).sort_values(["source", "rank"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = get_minor_league_data()
    for name, frame in data.items():
        print(f"\n=== {name} ({len(frame)} rows) ===")
        if not frame.empty:
            print(frame.head(10).to_string(index=False))

    prospects = get_top_prospects()
    print(f"\n=== Top Prospects ({len(prospects)} rows) ===")
    if not prospects.empty:
        print(prospects.to_string(index=False))
