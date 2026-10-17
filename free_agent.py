"""
free_agent.py
This winter's free agent class -- replaces the old MLB Trade Rumors scrape.

Two sources, in order:

  1. data/inputs/free_agents.csv -- a list you keep by hand. Columns:
         name, age_2027, position_group, option_detail
     position_group uses the same groups the rest of the project expects:
         Catchers, First Basemen, Second Basemen, Shortstops, Third Basemen,
         Left Fielders, Center Fielders, Right Fielders, Designated Hitters,
         Starting Pitchers, Right-Handed Relievers, Left-Handed Relievers
     option_detail is free text (e.g. "$18.7M player option") or blank.

  2. If that file is empty or missing, MLB's own free-agent list from the
     MLB Stats API (people/freeAgents). Position groups are filled in from
     each player's position, bats/throws, and this season's starts. This
     list only fills up once the offseason starts, and its exact contents
     depend on what MLB publishes -- check it against the real class.

Output columns (unchanged): name, age_2027, position_group, option_detail.

Usage:
    from free_agent import get_free_agents
    fa = get_free_agents()
"""

import os
import time

import pandas as pd

import mlb_api

CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24
SEASON          = 2026
COLUMNS         = ["name", "age_2027", "position_group", "option_detail"]

POSITION_HEADERS = [
    "Catchers", "First Basemen", "Second Basemen", "Shortstops",
    "Third Basemen", "Left Fielders", "Center Fielders", "Right Fielders",
    "Designated Hitters", "Starting Pitchers", "Right-Handed Relievers",
    "Left-Handed Relievers",
]
POS_TO_GROUP = {
    "C": "Catchers", "1B": "First Basemen", "2B": "Second Basemen",
    "3B": "Third Basemen", "SS": "Shortstops", "LF": "Left Fielders",
    "CF": "Center Fielders", "RF": "Right Fielders", "DH": "Designated Hitters",
}


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > CACHE_TTL_HOURS


def _from_csv() -> pd.DataFrame:
    df = mlb_api.read_input_csv("free_agents.csv", COLUMNS, "free agent list")
    if df.empty:
        return df
    bad = sorted(set(df["position_group"].dropna()) - set(POSITION_HEADERS))
    if bad:
        print(f"  [warn] unknown position_group values in free_agents.csv: {bad}")
    df["age_2027"] = pd.to_numeric(df["age_2027"], errors="coerce")
    df["option_detail"] = df["option_detail"].where(df["option_detail"].notna(), None)
    return df[COLUMNS].reset_index(drop=True)


def _position_group(pid, pos, throws, tables) -> str:
    if pos in ("P", "TWP"):
        pit = tables["pitching"]
        rows = pit[(pit["player_id"] == pid) & ~pit["_total"]]
        starter = not rows.empty and rows["GS"].sum() >= 0.5 * max(rows["G"].sum(), 1)
        if starter:
            return "Starting Pitchers"
        return "Left-Handed Relievers" if throws == "L" else "Right-Handed Relievers"
    if pos == "OF" or pos not in POS_TO_GROUP:
        fld = tables["fielding"]
        rows = fld[(fld["player_id"] == pid) & ~fld["_total"] & (fld["Pos"] != "P")]
        if not rows.empty:
            pos = rows.groupby("Pos")["_inn_outs"].sum().idxmax()
    return POS_TO_GROUP.get(pos)


def _from_api(force_refresh: bool = False) -> pd.DataFrame:
    try:
        resp = mlb_api.api_get("people_freeAgents", {"season": SEASON},
                               ttl_hours=CACHE_TTL_HOURS, force_refresh=force_refresh, force=True)
    except Exception as e:
        print(f"  [warn] MLB free-agent list unavailable: {e}")
        return pd.DataFrame(columns=COLUMNS)
    items = resp.get("freeAgents") or resp.get("people") or []
    if not items:
        print("  [info] MLB's free-agent list is empty right now (it fills in after the season)")
        return pd.DataFrame(columns=COLUMNS)

    ids = [((i.get("player") or i).get("id")) for i in items]
    people = mlb_api.get_people([x for x in ids if x], force_refresh=force_refresh).set_index("player_id")
    tables = mlb_api.get_player_tables(SEASON)
    rows = []
    for item in items:
        player = item.get("player") or item
        pid = player.get("id")
        if pid not in people.index:
            continue
        p = people.loc[pid]
        pos = (item.get("position") or {}).get("abbreviation") or p["primary_pos"]
        group = _position_group(pid, pos, p["throws"], tables)
        if group is None:
            continue
        rows.append({"name": p["full_name"] or player.get("fullName"),
                     "age_2027": mlb_api.season_age(p["birth_date"], SEASON + 1),
                     "position_group": group,
                     "option_detail": item.get("notes") or None})
    print(f"  [ok] {len(rows)} free agents from the MLB Stats API")
    return pd.DataFrame(rows, columns=COLUMNS)


def get_free_agents(force_refresh: bool = False) -> pd.DataFrame:
    """Free agent class: your CSV if it has rows, else MLB's own list."""
    df = _from_csv()
    if not df.empty:
        print(f"  [ok] {len(df)} free agents from data/inputs/free_agents.csv")
        return df

    cache_path = _cache_path(f"free_agents_api_{SEASON}")
    if not force_refresh and not _is_stale(cache_path):
        print("[cache] loading free agents from disk")
        return pd.read_parquet(cache_path)

    print(f"[fetch] {SEASON}-{str(SEASON + 1)[2:]} free agent class (MLB Stats API) ...")
    df = _from_api(force_refresh)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


# old name kept so existing imports keep working
get_mlbtr_free_agents = get_free_agents


def find_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Direct check -- e.g. find_player(fa, 'Crawford')."""
    if df.empty:
        return df
    return df[df["name"].str.contains(name, case=False, na=False)]


if __name__ == "__main__":
    fa = get_free_agents()

    if not fa.empty:
        print("\n-- FREE AGENT CLASS --")
        print(fa.to_string(index=False))

        print("\n-- CHECKING FOR CURRENT/RECENT MARINERS --")
        for name in ["Arozarena", "Crawford", "Robles", "Garver", "Refsnyder", "Ward"]:
            match = find_player(fa, name)
            if not match.empty:
                print(f"  {name}: FOUND -- {match.iloc[0]['position_group']}, "
                      f"age {match.iloc[0]['age_2027']} in 2027"
                      + (f" ({match.iloc[0]['option_detail']})"
                         if match.iloc[0]['option_detail'] else ""))
            else:
                print(f"  {name}: not found on the free agent list")
    else:
        print("No free agents yet -- fill in data/inputs/free_agents.csv")
