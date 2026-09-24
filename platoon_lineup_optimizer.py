"""
platoon_lineup_optimizer.py
Real vs-LHP/vs-RHP platoon splits for the Mariners' own current
roster -- a genuinely bounded, small set of real players (~25-40),
unlike the 278-player free agent list where bulk splits were
confirmed impractical. Given today's opposing starter's real
handedness, recommends the actual better lineup using real splits
data instead of season-total stats, which is all the rest of this
project currently sees.

HONEST STATUS: the real bbref splits page URL pattern used here
(baseball-reference.com/players/split.fcgi?id={id}&year={year}&t=b)
is a well-established, real bbref convention, but could NOT be
directly verified tonight -- a live fetch attempt hit bot detection
(same as a couple of other sites tonight), and this project's own
network doesn't reach baseball-reference.com either. Treat this as a
first real attempt, same as every other scraper's first pass tonight
-- run platoon_debug.py first to confirm the real page structure
before trusting this for real lineup decisions.

Getting a player's real bbref ID (e.g. "raleica01") -- a DIFFERENT ID
system than the MLBAM ids already used elsewhere in this project (the
free agent matcher's statsapi.mlb.com lookup) -- comes from the actual
href links already embedded in the roster pages this project already
successfully scrapes every night, not a new lookup service.

Usage:
    from platoon_lineup_optimizer import get_roster_bbref_ids, get_platoon_splits, recommend_lineup
"""

import os
import io
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24 * 7  # splits change slowly during a season -- a
                         # week-long cache is reasonable, unlike the
                         # 24-hour TTL used for daily-moving stats
                         # elsewhere in this project
SEASON = 2026


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.baseball-reference.com/",
    })
    return session


def get_roster_bbref_ids(season: int = SEASON) -> dict:
    """
    Real bbref player IDs for the current Mariners roster, pulled
    directly from the actual player-name links on Seattle's real team
    page -- the same page seattle_scraper.py already fetches for
    batting/pitching stats, just reading the href attributes this time
    instead of the table text. Returns {player_name: bbref_id}.
    """
    url = f"https://www.baseball-reference.com/teams/SEA/{season}.shtml"
    print(f"  [GET] {url}")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)

    soup = BeautifulSoup(resp.text, "html.parser")
    ids = {}
    for link in soup.find_all("a", href=re.compile(r"/players/\w/\w+\d+\.shtml")):
        match = re.search(r"/players/\w/(\w+\d+)\.shtml", link["href"])
        if match and link.text.strip():
            ids[link.text.strip()] = match.group(1)

    print(f"  [ok] {len(ids)} real bbref player IDs found on the roster page")
    return ids


def get_platoon_splits(player_name: str, bbref_id: str, player_type: str = "batter",
                       season: int = SEASON, force_refresh: bool = False) -> dict:
    """
    Real vs-LHP/vs-RHP splits for ONE specific player, using their
    real bbref ID. NOT independently verified against a live fetch --
    see module docstring. Run platoon_debug.py against a real player
    first to confirm this URL and table structure before trusting the
    numbers this returns.
    """
    cache_path = _cache_path(f"platoon_{bbref_id}_{season}")
    if not force_refresh and os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < CACHE_TTL_HOURS:
            print(f"  [cache] loading {player_name}'s splits from disk")
            return pd.read_parquet(cache_path).to_dict("records")[0]

    t = "b" if player_type == "batter" else "p"
    url = f"https://www.baseball-reference.com/players/split.fcgi?id={bbref_id}&year={season}&t={t}"
    print(f"  [GET] {url}")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)

    # BUG FIX: a real debug run found 25 tables on this page, ALL
    # sharing the exact same generic columns (Split, G, GS, PA, AB,
    # R...) -- the original "any column contains split" check matched
    # literally every one of them and just grabbed the first, which
    # wasn't the real platoon table at all. The debug run also
    # confirmed the real, specific, reliable id for the one that
    # actually matters: id="plato" -- targeting it directly, the same
    # established _find_table pattern (live HTML first, then HTML
    # comments) already used successfully throughout this whole
    # project for bbref's other tables, rather than guessing by
    # column content a second time.
    from bs4 import Comment
    soup_check = BeautifulSoup(resp.text, "html.parser")
    platoon_tag = soup_check.find("table", {"id": "plato"})
    if platoon_tag is None:
        for comment in soup_check.find_all(string=lambda t: isinstance(t, Comment)):
            if "plato" not in comment:
                continue
            c_soup = BeautifulSoup(comment, "html.parser")
            platoon_tag = c_soup.find("table", {"id": "plato"})
            if platoon_tag:
                break

    if platoon_tag is None:
        print(f"  [warn] no real table with id='plato' found for "
              f"{player_name} -- the real page structure may have "
              f"changed since this was confirmed, check with "
              f"platoon_table_debug.py")
        return {"name": player_name, "vs_LHP": None, "vs_RHP": None}

    platoon_table = pd.read_html(io.StringIO(str(platoon_tag)))[0]
    split_col = next((c for c in platoon_table.columns if "split" in str(c).lower()), None)
    stat_col = "OPS" if player_type == "batter" else "ERA"

    vs_lhp_row = platoon_table[platoon_table[split_col].astype(str).str.contains("LHP", na=False)]
    vs_rhp_row = platoon_table[platoon_table[split_col].astype(str).str.contains("RHP", na=False)]

    result = {
        "name": player_name,
        "vs_LHP": pd.to_numeric(vs_lhp_row[stat_col].values[0], errors="coerce") if not vs_lhp_row.empty and stat_col in vs_lhp_row.columns else None,
        "vs_RHP": pd.to_numeric(vs_rhp_row[stat_col].values[0], errors="coerce") if not vs_rhp_row.empty and stat_col in vs_rhp_row.columns else None,
    }
    pd.DataFrame([result]).to_parquet(cache_path, index=False)
    return result


def get_team_platoon_splits(team: str = "SEA", season: int = SEASON) -> dict:
    """
    Real, confirmed team-wide platoon splits -- a genuinely different,
    simpler question than the per-player lineup optimizer above ("is
    the TEAM as a whole better vs one handedness", not "which specific
    player should start today"). Confirmed real via a live page paste
    tonight: real URL, real "Platoon Splits" section with real vs RHP/
    vs LHP rows. Same id="plato" table id as the per-player pages,
    same real fix already applies here.
    """
    url = f"https://www.baseball-reference.com/teams/split.cgi?t=b&team={team}&year={season}"
    print(f"  [GET] {url}")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)

    from bs4 import Comment
    soup = BeautifulSoup(resp.text, "html.parser")
    tag = soup.find("table", {"id": "plato"})
    if tag is None:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if "plato" not in comment:
                continue
            c_soup = BeautifulSoup(comment, "html.parser")
            tag = c_soup.find("table", {"id": "plato"})
            if tag:
                break

    if tag is None:
        print(f"  [warn] no real team platoon table found -- check the "
              f"real page structure manually")
        return {"team": team, "vs_LHP_OPS": None, "vs_RHP_OPS": None}

    df = pd.read_html(io.StringIO(str(tag)))[0]
    split_col = next((c for c in df.columns if "split" in str(c).lower()), None)
    vs_lhp = df[df[split_col].astype(str) == "vs LHP"]
    vs_rhp = df[df[split_col].astype(str) == "vs RHP"]

    return {
        "team": team,
        "vs_LHP_OPS": pd.to_numeric(vs_lhp["OPS"].values[0], errors="coerce") if not vs_lhp.empty else None,
        "vs_RHP_OPS": pd.to_numeric(vs_rhp["OPS"].values[0], errors="coerce") if not vs_rhp.empty else None,
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
    print("Getting real bbref player IDs from the current Seattle roster...")
    ids = get_roster_bbref_ids()
    for name, pid in list(ids.items())[:10]:
        print(f"  {name}: {pid}")

    print("\nGetting real team-wide platoon splits...")
    team_splits = get_team_platoon_splits()
    print(team_splits)