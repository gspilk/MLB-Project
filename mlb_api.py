"""
mlb_api.py
One shared place for every MLB Stats API call in this project, plus the
project's own simplified WAR.

Every other data module (standings, schedule, Mariners stats, league
leaders, team overview, minor leagues, platoon splits, trades) builds its
tables from the helpers here, so caching, error handling and team/league
lookups only live in one place.

DATA SOURCES
  * MLB Stats API (statsapi.mlb.com) through the MLB-StatsAPI package --
    standings, schedules, rosters, player and team stats, minor leagues,
    platoon splits.
  * Baseball Savant (baseballsavant.mlb.com) -- Outs Above Average
    fielding runs, used for the fielding part of WAR (Rtot/Rdrs columns).

  Data (c) MLB Advanced Media, L.P. Personal, non-commercial use only:
  http://gdx.mlb.com/components/copyright.txt

PROJECT WAR (not copied from any other site)
  Batters:  WAR = (Rbat + Rbaser + Rfield + Rpos + Rrep) / runs-per-win
    Rbat    wOBA above league average, in runs (standard linear weights)
    Rbaser  stolen-base runs above league rate (wSB)
    Rfield  Statcast fielding runs prevented (Outs Above Average);
            catchers get framing runs when Savant provides them
    Rpos    standard positional adjustment, prorated by innings/games
    Rrep    replacement level, 20 runs per 600 PA
  Pitchers: RA9-based, a standard method
    RA9avg  league RA9 adjusted for role (starter/reliever, measured from
            this season's data) and the team's defense behind him
    RAR     runs saved vs. a replacement pitcher (.380 starter / .470
            reliever win%), relievers scaled by an estimated leverage
  Numbers will be close to, not identical to, other sites' WAR.

Usage:
    from mlb_api import get_player_tables, get_teams
    tables = get_player_tables(2026)       # league-wide batting/pitching/fielding + WAR
"""

import hashlib
import io
import json
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
import statsapi

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache", "api")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")

SEA_TEAM_ID = 136
DEFAULT_TTL_HOURS = 24

AL_ID, NL_ID = 103, 104
DIVISION_IDS = {
    201: "AL_East", 202: "AL_Central", 200: "AL_West",
    204: "NL_East", 205: "NL_Central", 203: "NL_West",
}

# Same 3-letter codes the rest of the project already uses (recommender.py's
# TEAM_ABBR), so schedule "Opp" values keep matching simulator/montecarlo.
TEAM_ABBR_BY_NAME = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
}

# Minor-league levels by MLB Stats API sport id
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "Rk"}

# Positional adjustments, runs per 162 games (standard values)
POS_ADJ = {"C": 12.5, "SS": 7.5, "2B": 2.5, "3B": 2.5, "CF": 2.5,
           "LF": -7.5, "RF": -7.5, "1B": -12.5, "DH": -17.5}
POS_NUM = {"P": 1, "C": 2, "1B": 3, "2B": 4, "3B": 5, "SS": 6, "LF": 7, "CF": 8, "RF": 9}

# wOBA weights (recent-season standard values) and scale
WOBA_W = {"uBB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.25, "3B": 1.59, "HR": 2.03}
WOBA_SCALE = 1.20
RUNS_SB, RUNS_CS = 0.2, -0.42
REPL_RUNS_PER_600 = 20.0
INNINGS_PER_SEASON = 1458.0          # 162 games x 9 innings
SP_REPL_FACTOR = np.sqrt(1 / 0.380 - 1)   # replacement starter wins .380
RP_REPL_FACTOR = np.sqrt(1 / 0.470 - 1)   # replacement reliever wins .470


# ── cached API access ─────────────────────────────────────────────────────────
def _cache_file(endpoint: str, params: dict) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{endpoint}_{key}.json")


def api_get(endpoint: str, params: dict, ttl_hours: float = DEFAULT_TTL_HOURS,
            force_refresh: bool = False, force: bool = False) -> dict:
    """
    statsapi.get() with an on-disk JSON cache. If the live call fails and an
    older cached copy exists, that copy is used (with a warning) instead of
    crashing the whole pipeline over a network hiccup.
    """
    path = _cache_file(endpoint, params)
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 3600 < ttl_hours
    if fresh and not force_refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    try:
        data = statsapi.get(endpoint, params, force=force)
    except Exception as e:
        if os.path.exists(path):
            print(f"  [warn] MLB Stats API call failed ({endpoint}: {e}) -- using older cached copy")
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        raise
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def frame_cache(name: str, ttl_hours: float, force_refresh: bool, build):
    """Parquet cache for a single built DataFrame."""
    os.makedirs(os.path.join(DATA_DIR, "cache"), exist_ok=True)
    path = os.path.join(DATA_DIR, "cache", f"{name}.parquet")
    if (not force_refresh and os.path.exists(path)
            and (time.time() - os.path.getmtime(path)) / 3600 < ttl_hours):
        return pd.read_parquet(path)
    df = build()
    if df is not None and not df.empty:
        safe = df.copy()
        for c in safe.columns:
            if safe[c].dtype == object:
                safe[c] = safe[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        safe.to_parquet(path, index=False)
    return df


# ── small helpers ─────────────────────────────────────────────────────────────
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def ip_to_outs(ip) -> float:
    """'123.1' (baseball notation) -> 370 outs."""
    if ip is None or (isinstance(ip, float) and np.isnan(ip)):
        return np.nan
    whole, _, frac = str(ip).partition(".")
    try:
        return int(whole or 0) * 3 + int(frac[:1] or 0)
    except ValueError:
        return np.nan


def outs_to_ip(outs) -> float:
    """370 outs -> 123.1 (standard baseball notation)."""
    if outs is None or np.isnan(outs):
        return np.nan
    outs = int(round(outs))
    return outs // 3 + (outs % 3) / 10


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.asarray(b, dtype=float) > 0, np.asarray(a, dtype=float) / np.asarray(b, dtype=float), np.nan)
    return out


def season_age(birth_date: str, season: int):
    """Age on June 30 of the season (the standard convention)."""
    try:
        y, m, d = map(int, str(birth_date)[:10].split("-"))
    except ValueError:
        return np.nan
    return season - y - (1 if (m, d) > (6, 30) else 0)


# ── teams ─────────────────────────────────────────────────────────────────────
def get_teams(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """id, name, abbr, league (AL/NL), division for the 30 MLB teams."""
    resp = api_get("teams", {"sportId": 1, "season": season}, ttl_hours=24 * 7,
                   force_refresh=force_refresh)
    rows = []
    for t in resp.get("teams", []):
        div = DIVISION_IDS.get(t.get("division", {}).get("id"))
        if not div:
            continue
        rows.append({"team_id": t["id"], "name": t["name"],
                     "abbr": TEAM_ABBR_BY_NAME.get(t["name"], t.get("abbreviation", "")),
                     "league": div.split("_")[0], "division": div})
    return pd.DataFrame(rows)


def team_id_for(code_or_name: str, season: int) -> int:
    """'CHW', 'Chicago White Sox' or 'white sox' -> 145."""
    teams = get_teams(season)
    key = str(code_or_name).strip().lower()
    for _, t in teams.iterrows():
        if key in (t["abbr"].lower(), t["name"].lower()) or key in t["name"].lower():
            return int(t["team_id"])
    raise ValueError(f"unknown team: {code_or_name}")


# ── schedule ──────────────────────────────────────────────────────────────────
NOT_PLAYED = {"Postponed", "Cancelled", "Suspended"}


def get_season_games(season: int, force_refresh: bool = False,
                     ttl_hours: float = 6) -> pd.DataFrame:
    """
    Every regular-season game (played and upcoming), one row per game:
    game_pk, date, game_time, home_id, away_id, home_runs, away_runs,
    innings, day_night, status, played.
    """
    resp = api_get("schedule", {"sportId": 1, "season": season, "gameTypes": "R",
                                "hydrate": "linescore"},
                   ttl_hours=ttl_hours, force_refresh=force_refresh)
    rows = {}
    for day in resp.get("dates", []):
        for g in day.get("games", []):
            state = g.get("status", {}).get("detailedState", "")
            if state in NOT_PLAYED:
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            hs, as_ = home.get("score"), away.get("score")
            final = g.get("status", {}).get("abstractGameState") == "Final"
            played = final and hs is not None and as_ is not None and hs != as_
            rows[g["gamePk"]] = {
                "game_pk": g["gamePk"],
                "date": g.get("officialDate", day.get("date")),
                "game_time": g.get("gameDate", ""),
                "home_id": home["team"]["id"], "away_id": away["team"]["id"],
                "home_runs": hs if played else None, "away_runs": as_ if played else None,
                "innings": (g.get("linescore", {}) or {}).get("currentInning", 9) or 9,
                "day_night": g.get("dayNight", ""),
                "status": state, "played": played,
            }
    games = pd.DataFrame(rows.values())
    if not games.empty:
        games = games.sort_values(["date", "game_time"]).reset_index(drop=True)
    return games


# ── people & rosters ──────────────────────────────────────────────────────────
def get_people(ids, force_refresh: bool = False) -> pd.DataFrame:
    """Birth date, bats/throws and primary position for a list of MLBAM ids."""
    ids = sorted({int(i) for i in ids if pd.notna(i)})
    rows = []
    for start in range(0, len(ids), 150):
        chunk = ids[start:start + 150]
        resp = api_get("people", {"personIds": ",".join(map(str, chunk))},
                       ttl_hours=24 * 7, force_refresh=force_refresh)
        for p in resp.get("people", []):
            rows.append({
                "player_id": p["id"], "full_name": p.get("fullName"),
                "birth_date": p.get("birthDate"),
                "bats": p.get("batSide", {}).get("code"),
                "throws": p.get("pitchHand", {}).get("code"),
                "primary_pos": p.get("primaryPosition", {}).get("abbreviation"),
            })
    return pd.DataFrame(rows, columns=["player_id", "full_name", "birth_date",
                                       "bats", "throws", "primary_pos"])


IL_CODES = {"D7": "7-day IL", "D10": "10-day IL", "D15": "15-day IL", "D60": "60-day IL"}


def get_roster(team_id: int, season: int, roster_type: str = "40Man",
               force_refresh: bool = False) -> pd.DataFrame:
    """
    A team's roster: player_id, Name, Pos, status_code, status, OnActv ("*"
    when on the active roster -- same marker the old source's 40-man
    table used), IL ("60-day IL" etc. or None), B, T, birth_date.
    """
    resp = api_get("team_roster", {"teamId": team_id, "rosterType": roster_type,
                                   "season": season, "hydrate": "person"},
                   ttl_hours=6, force_refresh=force_refresh)
    rows = []
    for r in resp.get("roster", []):
        person = r.get("person", {})
        code = r.get("status", {}).get("code", "")
        rows.append({
            "player_id": person.get("id"), "Name": person.get("fullName"),
            "Pos": r.get("position", {}).get("abbreviation"),
            "status_code": code, "status": r.get("status", {}).get("description"),
            "OnActv": "*" if code == "A" else None,
            "IL": IL_CODES.get(code),
            "B": person.get("batSide", {}).get("code"),
            "T": person.get("pitchHand", {}).get("code"),
            "birth_date": person.get("birthDate"),
        })
    return pd.DataFrame(rows, columns=["player_id", "Name", "Pos", "status_code", "status",
                                       "OnActv", "IL", "B", "T", "birth_date"])


# ── raw player stats ──────────────────────────────────────────────────────────
HIT_MAP = {"gamesPlayed": "G", "plateAppearances": "PA", "atBats": "AB", "runs": "R",
           "hits": "H", "doubles": "2B", "triples": "3B", "homeRuns": "HR", "rbi": "RBI",
           "stolenBases": "SB", "caughtStealing": "CS", "baseOnBalls": "BB",
           "intentionalWalks": "IBB", "strikeOuts": "SO", "hitByPitch": "HBP",
           "sacBunts": "SH", "sacFlies": "SF", "groundIntoDoublePlay": "GIDP",
           "totalBases": "TB"}
PIT_MAP = {"wins": "W", "losses": "L", "gamesPlayed": "G", "gamesStarted": "GS",
           "gamesFinished": "GF", "completeGames": "CG", "shutouts": "SHO", "saves": "SV",
           "holds": "HLD", "outs": "_outs", "hits": "H", "runs": "R", "earnedRuns": "ER",
           "homeRuns": "HR", "baseOnBalls": "BB", "intentionalWalks": "IBB",
           "strikeOuts": "SO", "hitBatsmen": "HBP", "balks": "BK", "wildPitches": "WP",
           "battersFaced": "BF"}
FLD_MAP = {"gamesPlayed": "G", "gamesStarted": "GS", "chances": "Ch", "putOuts": "PO",
           "assists": "A", "errors": "E", "doublePlays": "DP"}
GROUP_MAP = {"hitting": HIT_MAP, "pitching": PIT_MAP, "fielding": FLD_MAP}


def fetch_player_stats(season: int, group: str, team_id: int = None, sport_id: int = 1,
                       force_refresh: bool = False, ttl_hours: float = DEFAULT_TTL_HOURS) -> pd.DataFrame:
    """
    Raw season counting stats, one row per player per team (fielding: per
    player per team per position). A row with team_id NaN is a multi-team
    season total, when the API provides one.
    """
    params = {"stats": "season", "group": group, "playerPool": "ALL", "season": season,
              "sportIds": sport_id, "gameType": "R", "limit": 10000}
    if team_id is not None:
        params["teamId"] = team_id
    resp = api_get("stats", params, ttl_hours=ttl_hours, force_refresh=force_refresh)
    stat_map = GROUP_MAP[group]
    rows = []
    for block in resp.get("stats", []):
        for s in block.get("splits", []):
            st = s.get("stat", {})
            team = s.get("team") or {}
            row = {"player_id": s.get("player", {}).get("id"),
                   "Name": s.get("player", {}).get("fullName"),
                   "team_id": team.get("id"), "team_name": team.get("name"),
                   "league_id": (s.get("league") or {}).get("id")}
            for api_key, col in stat_map.items():
                row[col] = _num(st.get(api_key))
            if group == "pitching":
                if np.isnan(row["_outs"]):
                    row["_outs"] = ip_to_outs(st.get("inningsPitched"))
                if np.isnan(row["HBP"]):
                    row["HBP"] = _num(st.get("hitByPitch"))
            if group == "fielding":
                row["Pos"] = (s.get("position") or st.get("position") or {}).get("abbreviation")
                row["_inn_outs"] = ip_to_outs(st.get("innings"))
            rows.append(row)
    cols = ["player_id", "Name", "team_id", "team_name", "league_id"] + list(stat_map.values())
    if group == "fielding":
        cols += ["Pos", "_inn_outs"]
    df = pd.DataFrame(rows, columns=cols)
    return df


def _label_teams(df: pd.DataFrame, season: int, key_cols=("player_id",)) -> pd.DataFrame:
    """
    Adds Team/Tm (3-letter code), league (AL/NL) and _total (True for a
    player's combined multi-team row). Builds the combined row from team rows
    when the API didn't send one, labeled like the old source did ("2TM").
    """
    if df.empty:
        return df.assign(Team=None, Tm=None, league=None, _total=False)
    teams = get_teams(season).set_index("team_id")
    df = df.copy()
    df["Team"] = df["team_id"].map(teams["abbr"])
    df["league"] = df["team_id"].map(teams["league"])
    df.loc[df["team_id"].isna(), "league"] = None
    df["_total"] = df["team_id"].isna()

    counting = [c for c in df.columns if c not in
                ("player_id", "Name", "team_id", "team_name", "league_id", "Pos",
                 "Team", "league", "_total")]
    out = []
    for _, grp in df.groupby(list(key_cols), sort=False):
        per_team = grp[~grp["_total"]]
        totals = grp[grp["_total"]]
        if len(per_team) > 1:
            n = per_team["team_id"].nunique()
            if totals.empty:
                tot = per_team.iloc[[0]].copy()
                tot[counting] = per_team[counting].sum(min_count=1).values
                tot["team_id"] = np.nan
            else:
                tot = totals.iloc[[0]].copy()
            leagues = per_team["league"].dropna().unique()
            tot["Team"] = f"{n}TM"
            tot["league"] = leagues[0] if len(leagues) == 1 else None
            tot["_total"] = True
            out.append(pd.concat([tot, per_team]))
        else:
            out.append(per_team if not per_team.empty else totals)
    df = pd.concat(out, ignore_index=True)
    df["Tm"] = df["Team"]
    return df


# ── league context ────────────────────────────────────────────────────────────
def _woba_parts(df):
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    ubb = df["BB"] - df["IBB"].fillna(0)
    num = (WOBA_W["uBB"] * ubb + WOBA_W["HBP"] * df["HBP"] + WOBA_W["1B"] * singles
           + WOBA_W["2B"] * df["2B"] + WOBA_W["3B"] * df["3B"] + WOBA_W["HR"] * df["HR"])
    den = df["AB"] + ubb + df["SF"] + df["HBP"]
    return num, den, singles, ubb


def league_context(bat: pd.DataFrame, pit: pd.DataFrame) -> dict:
    """League totals from per-team rows (never the combined rows)."""
    b = bat[~bat["_total"]].fillna(0)
    p = pit[~pit["_total"]].fillna(0)
    num, den, singles, ubb = _woba_parts(b)
    ab, h, bb, hbp, sf = (b[c].sum() for c in ("AB", "H", "BB", "HBP", "SF"))
    ip = p["_outs"].sum() / 3
    runs = p["R"].sum()
    lg_ra9 = 9 * runs / ip if ip else 4.4
    starters = p[p["GS"] >= 0.5 * p["G"].clip(lower=1)]
    relievers = p[p["GS"] < 0.5 * p["G"].clip(lower=1)]
    sp_ip, rp_ip = starters["_outs"].sum() / 3, relievers["_outs"].sum() / 3
    raw_fip = ((13 * p["HR"].sum() + 3 * (p["BB"].sum() + p["HBP"].sum())
                - 2 * p["SO"].sum()) / ip) if ip else 0
    lg_era = 9 * p["ER"].sum() / ip if ip else 4.2
    sb_opps = (singles + ubb + b["HBP"]).sum()
    return {
        "woba": num.sum() / den.sum() if den.sum() else 0.315,
        "obp": (h + bb + hbp) / (ab + bb + hbp + sf) if ab else 0.315,
        "slg": b["TB"].sum() / ab if ab else 0.400,
        "era": lg_era,
        "ra9": lg_ra9,
        "ra9_sp": 9 * starters["R"].sum() / sp_ip if sp_ip else lg_ra9,
        "ra9_rp": 9 * relievers["R"].sum() / rp_ip if rp_ip else lg_ra9,
        "fip_const": lg_era - raw_fip,
        "wsb": (RUNS_SB * b["SB"].sum() + RUNS_CS * b["CS"].sum()) / sb_opps if sb_opps else 0,
        # Tango's runs-per-win approximation from league run environment
        "rpw": 9 * (runs / ip if ip else 0.49) * 1.5 + 3,
        "pa": b["PA"].sum(),
    }


# ── rate stats ────────────────────────────────────────────────────────────────
def add_batting_rates(df: pd.DataFrame, lc: dict = None) -> pd.DataFrame:
    df = df.copy()
    df["BA"] = np.round(_div(df["H"], df["AB"]), 3)
    df["OBP"] = np.round(_div(df["H"] + df["BB"] + df["HBP"].fillna(0),
                              df["AB"] + df["BB"] + df["HBP"].fillna(0) + df["SF"].fillna(0)), 3)
    df["SLG"] = np.round(_div(df["TB"], df["AB"]), 3)
    df["OPS"] = np.round(df["OBP"] + df["SLG"], 3)
    if lc:
        df["OPS+"] = np.round(100 * (df["OBP"] / lc["obp"] + df["SLG"] / lc["slg"] - 1))
    return df


def add_pitching_rates(df: pd.DataFrame, lc: dict = None) -> pd.DataFrame:
    df = df.copy()
    ip = df["_outs"] / 3
    df["IP"] = df["_outs"].map(outs_to_ip)
    df["W-L%"] = np.round(_div(df["W"], df["W"] + df["L"]), 3)
    df["ERA"] = np.round(_div(9 * df["ER"], ip), 2)
    df["WHIP"] = np.round(_div(df["H"] + df["BB"], ip), 3)
    df["H9"] = np.round(_div(9 * df["H"], ip), 1)
    df["HR9"] = np.round(_div(9 * df["HR"], ip), 1)
    df["BB9"] = np.round(_div(9 * df["BB"], ip), 1)
    df["SO9"] = np.round(_div(9 * df["SO"], ip), 1)
    df["SO/W"] = np.round(_div(df["SO"], df["BB"]), 2)
    if lc:
        df["FIP"] = np.round(_div(13 * df["HR"] + 3 * (df["BB"] + df["HBP"].fillna(0)) - 2 * df["SO"], ip)
                             + lc["fip_const"], 2)
        df["ERA+"] = np.round(_div(100 * lc["era"], df["ERA"]))
    return df


def add_fielding_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Inn"] = df["_inn_outs"].map(outs_to_ip)
    df["Fld%"] = np.round(_div(df["PO"] + df["A"], df["Ch"]), 3)
    return df


SAVANT_HEADERS = {"User-Agent": "mlb-project (personal portfolio project)"}


def _savant_csv(url: str, params: dict) -> pd.DataFrame:
    resp = requests.get(url, params=params, headers=SAVANT_HEADERS, timeout=30)
    resp.raise_for_status()
    text = resp.content.decode("utf-8-sig")
    return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()


def _pick(df: pd.DataFrame, *fragments):
    for frag in fragments:
        for c in df.columns:
            if frag in str(c).lower():
                return c
    return None


def get_fielding_runs(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """
    Statcast fielding runs per player per position: player_id, Pos, frv.
    Infield/outfield from Outs Above Average (fielding runs prevented),
    catchers from framing runs. Empty frame (with a warning) if Savant
    can't be reached -- WAR then treats fielding as average (0 runs).
    """
    def build():
        frames = []
        for pos, num in POS_NUM.items():
            if pos in ("P", "C"):
                continue
            try:
                raw = _savant_csv("https://baseballsavant.mlb.com/leaderboard/outs_above_average",
                                  {"type": "Fielder", "startYear": season, "endYear": season,
                                   "split": "no", "team": "", "range": "year", "min": 0,
                                   "pos": num, "roles": "", "viz": "hide", "csv": "true"})
            except Exception as e:
                print(f"  [warn] Savant OAA ({pos}) unavailable: {e}")
                continue
            if raw.empty:
                continue
            id_col = _pick(raw, "player_id")
            runs_col = _pick(raw, "fielding_runs", "runs_prevented")
            oaa_col = _pick(raw, "outs_above_average")
            if id_col is None or (runs_col is None and oaa_col is None):
                print(f"  [warn] unexpected Savant OAA columns for {pos}: {list(raw.columns)[:8]}")
                continue
            frv = raw[runs_col] if runs_col else raw[oaa_col] * 0.9
            frames.append(pd.DataFrame({"player_id": pd.to_numeric(raw[id_col], errors="coerce"),
                                        "Pos": pos, "frv": pd.to_numeric(frv, errors="coerce")}))
        try:
            raw = _savant_csv("https://baseballsavant.mlb.com/catcher_framing",
                              {"year": season, "team": "", "min": 0, "type": "catcher", "csv": "true"})
            id_col = _pick(raw, "player_id", "id")
            runs_col = _pick(raw, "runs_extra_strikes", "rv_tot", "framing_runs")
            if id_col and runs_col:
                frames.append(pd.DataFrame({"player_id": pd.to_numeric(raw[id_col], errors="coerce"),
                                            "Pos": "C", "frv": pd.to_numeric(raw[runs_col], errors="coerce")}))
        except Exception as e:
            print(f"  [warn] Savant catcher framing unavailable: {e}")
        if not frames:
            return pd.DataFrame(columns=["player_id", "Pos", "frv"])
        out = pd.concat(frames, ignore_index=True).dropna(subset=["player_id"])
        out["player_id"] = out["player_id"].astype(int)
        return out.groupby(["player_id", "Pos"], as_index=False)["frv"].sum()

    df = frame_cache(f"savant_fielding_runs_{season}", 24, force_refresh, build)
    if df is None or df.empty:
        print("  [warn] no Statcast fielding runs -- fielding counted as average (0) in WAR")
        return pd.DataFrame(columns=["player_id", "Pos", "frv"])
    return df


# ── WAR ───────────────────────────────────────────────────────────────────────
def _allocate_fielding(fld: pd.DataFrame, frv: pd.DataFrame) -> pd.DataFrame:
    """
    Gives every fielding row (player x team x position) its share of the
    player's season Statcast runs at that position, split by innings when a
    player played the position for more than one team.
    """
    fld = fld.copy()
    if frv.empty:
        fld["Rfield"] = 0.0
        return fld
    rows = fld[~fld["_total"]]
    inn_total = rows.groupby(["player_id", "Pos"])["_inn_outs"].transform("sum")
    share = (rows["_inn_outs"] / inn_total.replace(0, np.nan)).fillna(1.0)
    frv_map = frv.set_index(["player_id", "Pos"])["frv"]
    keys = list(zip(rows["player_id"], rows["Pos"]))
    fld.loc[rows.index, "Rfield"] = [frv_map.get(k, 0.0) for k in keys] * share.values
    if fld["_total"].any():
        sums = fld[~fld["_total"]].groupby(["player_id", "Pos"])["Rfield"].sum()
        tot = fld["_total"]
        fld.loc[tot, "Rfield"] = [sums.get(k, 0.0) for k in zip(fld.loc[tot, "player_id"], fld.loc[tot, "Pos"])]
    fld["Rfield"] = fld["Rfield"].fillna(0.0)
    return fld


def compute_batting_war(bat: pd.DataFrame, fld: pd.DataFrame, lc: dict) -> pd.DataFrame:
    """Per batting row (player x team, plus combined rows): Rbat ... WAR."""
    b = bat.copy()
    f = fld.copy()
    num, den, singles, ubb = _woba_parts(b.fillna({"IBB": 0, "HBP": 0, "SF": 0}))
    b["wOBA"] = np.round(_div(num, den), 3)
    b["Rbat"] = (b["wOBA"].fillna(lc["woba"]) - lc["woba"]) / WOBA_SCALE * b["PA"].fillna(0)
    sb_opps = singles + ubb + b["HBP"].fillna(0)
    b["Rbaser"] = (RUNS_SB * b["SB"].fillna(0) + RUNS_CS * b["CS"].fillna(0)) - lc["wsb"] * sb_opps.fillna(0)
    b["Rdp"] = 0.0

    # fielding + positional runs: a player x team batting row gets that
    # team's fielding rows; a combined "2TM" row gets all of the player's
    f = f[~f["_total"]].copy()
    f["_pos_runs"] = np.array([POS_ADJ.get(p, 0.0) for p in f["Pos"]]) * (
        f["_inn_outs"].fillna(0) / 3 / INNINGS_PER_SEASON)
    f_non_p = f[f["Pos"] != "P"]
    by_team = f_non_p.groupby(["player_id", "team_id"])
    by_player = f_non_p.groupby("player_id")
    rfield, rpos, fgames = {}, {}, {}
    for (pid, tid), g in by_team:
        rfield[(pid, tid, False)] = g["Rfield"].sum()
        rpos[(pid, tid, False)] = g["_pos_runs"].sum()
        fgames[(pid, tid, False)] = g["G"].sum()
    for pid, g in by_player:
        rfield[(pid, -1, True)] = g["Rfield"].sum()
        rpos[(pid, -1, True)] = g["_pos_runs"].sum()
        fgames[(pid, -1, True)] = g["G"].sum()
    keys = [(pid, -1, True) if tot else (pid, tid, False)
            for pid, tid, tot in zip(b["player_id"], b["team_id"], b["_total"])]
    b["Rfield"] = [rfield.get(k, 0.0) for k in keys]
    dh_games = (b["G"].fillna(0) - np.array([fgames.get(k, 0.0) for k in keys])).clip(lower=0)
    pitcher_only = np.array([k not in fgames for k in keys]) & (b["PA"].fillna(0) < 20)
    b["Rpos"] = np.array([rpos.get(k, 0.0) for k in keys]) + np.where(
        pitcher_only, 0.0, POS_ADJ["DH"] * dh_games / 162)
    b["Rrep"] = REPL_RUNS_PER_600 * b["PA"].fillna(0) / 600
    b["RAA"] = b["Rbat"] + b["Rbaser"] + b["Rdp"] + b["Rfield"] + b["Rpos"]
    b["RAR"] = b["RAA"] + b["Rrep"]
    rpw = lc["rpw"]
    b["WAA"] = b["RAA"] / rpw
    b["WAR"] = b["RAR"] / rpw
    b["oWAR"] = (b["Rbat"] + b["Rbaser"] + b["Rdp"] + b["Rrep"]) / rpw
    b["dWAR"] = (b["Rfield"] + b["Rpos"]) / rpw
    for c in ("Rbat", "Rbaser", "Rdp", "Rfield", "Rpos", "Rrep", "RAA", "RAR"):
        b[c] = b[c].round(1)
    for c in ("WAA", "WAR", "oWAR", "dWAR"):
        b[c] = b[c].round(1)
    return b


def compute_pitching_war(pit: pd.DataFrame, team_def_runs: dict, lc: dict) -> pd.DataFrame:
    """Per pitching row: RA9, RA9opp/def/role/avg, gmLI, RAA, WAA, RAR, WAR."""
    p = pit.copy()
    ip = p["_outs"].fillna(0) / 3
    p["RA9"] = np.round(_div(9 * p["R"], ip), 2)
    starter = p["GS"].fillna(0) >= 0.5 * p["G"].fillna(0).clip(lower=1)
    p["RA9role"] = np.where(starter, lc["ra9_sp"] - lc["ra9"], lc["ra9_rp"] - lc["ra9"]).round(2)
    p["RA9opp"] = 0.0      # opponent-quality adjustment not modeled
    # team defense: good fielders behind him lower what an average pitcher allows
    team_ip = p[~p["_total"]].groupby("team_id")["_outs"].sum() / 3
    def_per9 = {t: -9 * runs / team_ip[t] for t, runs in team_def_runs.items()
                if t in team_ip and team_ip[t] > 0}
    p["RA9def"] = p["team_id"].map(def_per9).fillna(0.0).round(2)
    p["RA9avg"] = (lc["ra9"] + p["RA9opp"] + p["RA9def"] + p["RA9role"]).round(2)
    relief_lev = 1.0 + 0.9 * _div(p["SV"].fillna(0) + p["HLD"].fillna(0), p["G"].fillna(0))
    p["gmLI"] = np.where(starter, 1.0, np.clip(np.nan_to_num(relief_lev, nan=1.0), 0.7, 2.0)).round(2)
    ra9 = p["RA9"].fillna(p["RA9avg"])
    p["RAA"] = ((p["RA9avg"] - ra9) * ip / 9).round(1)
    repl = p["RA9avg"] * np.where(starter, SP_REPL_FACTOR, RP_REPL_FACTOR)
    p["Rrep"] = ((repl - p["RA9avg"]) * ip / 9).round(1)
    p["RAR"] = (p["RAA"] + p["Rrep"]).round(1)
    lev = np.where(starter, 1.0, (1 + p["gmLI"]) / 2)
    p["WAA"] = (p["RAA"] * lev / lc["rpw"]).round(1)
    p["WAR"] = (p["RAR"] * lev / lc["rpw"]).round(1)
    return p


# ── one call for every league-wide player table ───────────────────────────────
_TABLES_MEMO = {}   # season -> (built_at, tables): reused within one run


def get_player_tables(season: int, force_refresh: bool = False) -> dict:
    """
    Memoized wrapper around build_player_tables(): several modules need the
    same league tables in one run, so they're built once per process (a
    --refresh run refetches once, not once per module).
    """
    hit = _TABLES_MEMO.get(season)
    if hit and (not force_refresh or time.time() - hit[0] < 600):
        return hit[1]
    tables = build_player_tables(season, force_refresh=force_refresh)
    _TABLES_MEMO[season] = (time.time(), tables)
    return tables


def build_player_tables(season: int, force_refresh: bool = False) -> dict:
    """
    League-wide, every MLB player this season, WAR included:
      batting, pitching, fielding  -- per player x team rows plus combined
                                       "2TM" rows for traded players
      context                      -- league averages used for OPS+/ERA+/WAR
    Each table has player_id, Name, Age, Team/Tm (3-letter code), league.
    """
    bat = _label_teams(fetch_player_stats(season, "hitting", force_refresh=force_refresh), season)
    pit = _label_teams(fetch_player_stats(season, "pitching", force_refresh=force_refresh), season)
    fld = _label_teams(fetch_player_stats(season, "fielding", force_refresh=force_refresh), season,
                       key_cols=("player_id", "Pos"))

    ids = set(bat["player_id"]) | set(pit["player_id"]) | set(fld["player_id"])
    people = get_people(ids, force_refresh=force_refresh).set_index("player_id")
    for df in (bat, pit, fld):
        if not df.empty:
            df["Age"] = df["player_id"].map(people["birth_date"]).map(lambda d: season_age(d, season))

    lc = league_context(bat, pit)
    frv = get_fielding_runs(season, force_refresh=force_refresh)
    fld = _allocate_fielding(fld, frv)
    fld = add_fielding_rates(fld)
    fld["Rtot"] = fld["Rfield"].round(1)
    fld["Rdrs"] = fld["Rtot"]
    inn = fld["_inn_outs"] / 3
    fld["Rtot/yr"] = np.round(_div(fld["Rtot"] * 1200, inn))
    fld["Rdrs/yr"] = fld["Rtot/yr"]

    bat = add_batting_rates(bat, lc)
    bat = compute_batting_war(bat, fld, lc)
    # primary position per row: most innings in the field, else DH
    f_rows = fld[(fld["Pos"] != "P") & ~fld["_total"]]
    by_team_pos = f_rows.groupby(["player_id", "team_id", "Pos"])["_inn_outs"].sum()
    by_player_pos = f_rows.groupby(["player_id", "Pos"])["_inn_outs"].sum()
    prim = {}
    for (pid, tid, pos), outs in by_team_pos.sort_values(ascending=False).items():
        prim.setdefault((pid, tid), pos)
    for (pid, pos), outs in by_player_pos.sort_values(ascending=False).items():
        prim.setdefault((pid, -1), pos)
    bat["Pos"] = [prim.get((pid, -1) if tot else (pid, tid)) or
                  ("P" if people["primary_pos"].get(pid) == "P" else "DH")
                  for pid, tid, tot in zip(bat["player_id"], bat["team_id"], bat["_total"])]

    team_def = fld[~fld["_total"]].groupby("team_id")["Rfield"].sum().to_dict()
    pit = add_pitching_rates(pit, lc)
    pit = compute_pitching_war(pit, team_def, lc)

    return {"batting": bat, "pitching": pit, "fielding": fld, "context": lc,
            "people": people.reset_index()}


# ── output tables in the old source column layout ────────────────
BAT_COLS = ["Name", "Age", "Team", "Tm", "Lg", "WAR", "G", "PA", "AB", "R", "H", "2B", "3B",
            "HR", "RBI", "SB", "CS", "BB", "SO", "BA", "OBP", "SLG", "OPS", "OPS+", "TB",
            "GIDP", "HBP", "SH", "SF", "IBB", "Pos", "player_id"]
PIT_COLS = ["Name", "Age", "Team", "Tm", "Lg", "WAR", "W", "L", "W-L%", "ERA", "G", "GS",
            "GF", "CG", "SHO", "SV", "HLD", "IP", "H", "R", "ER", "HR", "BB", "IBB", "SO",
            "HBP", "BK", "WP", "BF", "ERA+", "FIP", "WHIP", "H9", "HR9", "BB9", "SO9", "SO/W",
            "player_id"]
FLD_COLS = ["Name", "Age", "Team", "Tm", "Lg", "Pos", "G", "GS", "Inn", "Ch", "PO", "A",
            "E", "DP", "Fld%", "Rtot", "Rtot/yr", "Rdrs", "Rdrs/yr", "player_id"]
VAL_BAT_COLS = ["Name", "Age", "Tm", "PA", "Rbat", "Rbaser", "Rdp", "Rfield", "Rpos",
                "RAA", "WAA", "Rrep", "RAR", "WAR", "oWAR", "dWAR", "player_id"]
VAL_PIT_COLS = ["Name", "Age", "Tm", "IP", "G", "GS", "R", "RA9", "RA9opp", "RA9def",
                "RA9role", "RA9avg", "RAA", "WAA", "gmLI", "Rrep", "RAR", "WAR", "player_id"]


def shape(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Selects/orders the given columns (adding any missing ones as empty)."""
    out = df.copy()
    if "Lg" in cols and "Lg" not in out.columns and "league" in out.columns:
        out["Lg"] = out["league"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].reset_index(drop=True)


def team_totals(players: pd.DataFrame, group: str, season: int,
                games_by_team: dict = None) -> pd.DataFrame:
    """Team totals summed from per-team player rows; Tm = full team name."""
    if players.empty:
        return pd.DataFrame()
    rows = players[~players["_total"]]
    counting = [c for c in GROUP_MAP[group].values()] + (["_inn_outs"] if group == "fielding" else [])
    counting = [c for c in counting if c in rows.columns]
    agg = rows.groupby("team_id")[counting].sum(min_count=1)
    if "WAR" in rows.columns:
        agg["WAR"] = rows.groupby("team_id")["WAR"].sum().round(1)
    if "Rfield" in rows.columns:
        agg["Rtot"] = rows.groupby("team_id")["Rfield"].sum().round(0)
        agg["Rdrs"] = agg["Rtot"]
    teams = get_teams(season).set_index("team_id")
    agg = agg.reset_index()
    agg["Tm"] = agg["team_id"].map(teams["name"])
    agg["abbr"] = agg["team_id"].map(teams["abbr"])
    agg["league"] = agg["team_id"].map(teams["league"])
    agg["_total"] = False
    if games_by_team is not None:
        agg["G"] = agg["team_id"].map(games_by_team)
    return agg


def games_played_by_team(season: int, force_refresh: bool = False) -> dict:
    games = get_season_games(season, force_refresh=force_refresh)
    played = games[games["played"]] if not games.empty else games
    counts = {}
    for col in ("home_id", "away_id"):
        for t, n in played[col].value_counts().items():
            counts[t] = counts.get(t, 0) + int(n)
    return counts


# ── splits (platoon) ──────────────────────────────────────────────────────────
def _parse_splits(stat_blocks) -> dict:
    out = {}
    for block in stat_blocks or []:
        for s in block.get("splits", []):
            code = (s.get("split") or {}).get("code")
            if code in ("vl", "vr"):
                out[code] = s.get("stat", {})
    return out


def get_player_platoon(player_id: int, season: int, group: str = "hitting",
                       force_refresh: bool = False) -> dict:
    """{'vl': stat dict, 'vr': stat dict} for one player."""
    resp = api_get("person", {"personId": player_id,
                              "hydrate": f"stats(group=[{group}],type=[statSplits],"
                                         f"sitCodes=[vl,vr],season={season})"},
                   ttl_hours=24 * 7, force_refresh=force_refresh)
    people = resp.get("people", [])
    return _parse_splits(people[0].get("stats") if people else [])


def get_team_platoon(team_id: int, season: int, group: str = "hitting",
                     force_refresh: bool = False) -> dict:
    resp = api_get("team_stats", {"teamId": team_id, "season": season, "group": group,
                                  "stats": "statSplits", "sitCodes": "vl,vr"},
                   ttl_hours=24, force_refresh=force_refresh)
    return _parse_splits(resp.get("stats"))


# ── CSV inputs you maintain by hand ───────────────────────────────────────────
def read_input_csv(filename: str, columns: list, what: str) -> pd.DataFrame:
    """
    Reads data/inputs/<filename>. If it doesn't exist yet, writes an empty
    template with the right headers and returns an empty frame, so the
    feature that needs it skips gracefully instead of crashing.
    """
    os.makedirs(INPUT_DIR, exist_ok=True)
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        print(f"  [info] {what}: no data yet -- fill in {path} (template created)")
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    if df.empty:
        print(f"  [info] {what}: {path} is empty -- fill it in to enable this feature")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"  [warn] {path} is missing columns {missing}")
    return df


if __name__ == "__main__":
    t = get_player_tables(2026)
    print("league context:", {k: round(v, 3) for k, v in t["context"].items()})
    top = t["batting"][t["batting"]["PA"] >= 300].sort_values("WAR", ascending=False)
    print(top[["Name", "Tm", "PA", "OPS", "Rbat", "Rfield", "Rpos", "WAR"]].head(15).to_string(index=False))
    top = t["pitching"][t["pitching"]["_outs"] >= 300].sort_values("WAR", ascending=False)
    print(top[["Name", "Tm", "IP", "ERA", "RA9", "RA9avg", "WAR"]].head(15).to_string(index=False))
