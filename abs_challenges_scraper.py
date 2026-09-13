"""
abs_challenge_scraper.py
Scrapes real 2026 MLB ABS (Automated Ball-Strike) challenge system data
from baseball-reference.com's dedicated ABS analysis page:
https://www.baseball-reference.com/friv/abs-challenges.shtml

MLB introduced the ABS challenge system in 2026 -- batters, pitchers,
and catchers can challenge a ball/strike call, re-judged automatically.
This scrapes the real "Challenges by Team" and "Challenges by Player"
tables to answer things like "how often do Mariners hitters/pitchers
actually win their challenges."

Both target tables are hidden inside HTML comments -- confirmed via a
live fetch (see abs_debug.py) -- same pattern already handled by
mariners_stats.py's _find_table() for fielding/40-man data, reused
directly here rather than reimplemented.

Real, confirmed table ids (2026-09-13):
  by_team        -- every team's own challenge stats
  player_stats   -- every player's own challenge stats

Usage:
    from abs_challenge_scraper import get_abs_challenges
    abs_data = get_abs_challenges()
    mariners_row = abs_data["by_team"][abs_data["by_team"]["Name"] == "Seattle Mariners"]
"""

import os
import io
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ── config ────────────────────────────────────────────────────────────────────
URL             = "https://www.baseball-reference.com/friv/abs-challenges.shtml"
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24
TEAM_NAME       = "Seattle Mariners"


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _is_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours > CACHE_TTL_HOURS


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def _fetch_page() -> BeautifulSoup:
    print(f"  [GET] {URL}")
    session = _make_session()
    resp = session.get(URL, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)
    return BeautifulSoup(resp.text, "html.parser")


def _find_table(soup: BeautifulSoup, table_id: str):
    """Same pattern as mariners_stats.py -- checks live HTML first,
    then searches inside HTML comments (bbref hides many tables this
    way, confirmed for both target tables here via abs_debug.py)."""
    tag = soup.find("table", {"id": table_id})
    if tag:
        return tag
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id not in comment:
            continue
        c_soup = BeautifulSoup(comment, "html.parser")
        tag = c_soup.find("table", {"id": table_id})
        if tag:
            return tag
    return None


def _read_html(tag) -> pd.DataFrame:
    return pd.read_html(io.StringIO(str(tag)))[0]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUG FIX: both real ABS tables have genuine multi-level headers
    (e.g. "All Challenges" spanning "Chal"/"OT"/"OT%" sub-columns) --
    pandas.read_html() parses this into a MultiIndex, which shows up
    as tuples like ('All Challenges', 'Chal') rather than a plain
    string column name. The rest of this file's string-based matching
    (looking for "Name" or "Tm") can't match a tuple at all, so it was
    silently failing and leaving the raw, unflattened columns in place.

    Flattens to plain strings. Structural identifier columns (Tm, G,
    Player, Name) are ALWAYS kept standalone regardless of their group
    label -- e.g. bbref's real header nests the team-name column under
    a generic "Team" group ('Team', 'Tm'), which isn't the same string
    as "Tm" itself, so a naive top==bottom check missed it and produced
    an awkward "Team_Tm" column that nothing downstream was looking
    for. Only genuinely repeated stat columns (Chal, OT%, etc., which
    really do need their group prefix to stay distinguishable across
    "As Batter" vs "As Catcher" vs "As Pitcher") get the group-prefixed
    treatment.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    STANDALONE = {"tm", "g", "player", "name"}
    new_cols = []
    for top, bottom in df.columns:
        top_str, bottom_str = str(top), str(bottom)
        if (top_str.startswith("Unnamed") or top_str == bottom_str
                or bottom_str.lower() in STANDALONE):
            new_cols.append(bottom_str)
        else:
            new_cols.append(f"{top_str}_{bottom_str}")
    df = df.copy()
    df.columns = new_cols
    return df


def _clean(df: pd.DataFrame, name_col: str = None) -> pd.DataFrame:
    df = _flatten_columns(df)
    if name_col is None:
        for candidate in ["Player", "Name", "Tm"]:
            if candidate in df.columns:
                name_col = candidate
                break
    if name_col is None or name_col not in df.columns:
        return df
    df = df[df[name_col].notna()].copy()
    df = df[~df[name_col].isin([name_col, "Name", "Player", "Tm", "Totals",
                                 "Team Totals", "League Totals"])]
    df[name_col] = df[name_col].astype(str).str.replace(r"[*#]", "", regex=True).str.strip()
    if name_col != "Name":
        df = df.rename(columns={name_col: "Name"})
    skip = {"Name", "Pos", "Tm", "Lg"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def get_abs_challenges(force_refresh: bool = False) -> dict:
    """
    Returns:
        {
            "by_team":   DataFrame -- every team's real challenge stats
            "by_player": DataFrame -- every player's real challenge stats
        }
    """
    cache_paths = {"by_team": _cache_path("abs_by_team"),
                  "by_player": _cache_path("abs_by_player")}
    all_cached = all(os.path.exists(p) for p in cache_paths.values())

    if not force_refresh and all_cached and not any(_is_stale(p) for p in cache_paths.values()):
        print("[cache] loading ABS challenge data from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print("[fetch] ABS challenge data from bbref ...")
    soup = _fetch_page()

    by_team_tag = _find_table(soup, "by_team")
    by_team = _clean(_read_html(by_team_tag)) if by_team_tag is not None else pd.DataFrame()
    print(f"  [ok]   by_team: {len(by_team)} rows, {len(by_team.columns)} cols"
          if not by_team.empty else "  [warn] by_team not found")

    player_tag = _find_table(soup, "player_stats")
    by_player = _clean(_read_html(player_tag)) if player_tag is not None else pd.DataFrame()
    print(f"  [ok]   by_player: {len(by_player)} rows, {len(by_player.columns)} cols"
          if not by_player.empty else "  [warn] player_stats not found")

    result = {"by_team": by_team, "by_player": by_player}
    for k, df in result.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return result


def get_mariners_abs_summary(abs_data: dict = None) -> dict:
    """
    Convenience helper: pulls just the Mariners' own row from the team
    table, and every Mariners player's row from the player table.
    """
    if abs_data is None:
        abs_data = get_abs_challenges()

    by_team = abs_data.get("by_team", pd.DataFrame())
    team_row = pd.DataFrame()
    if not by_team.empty and "Name" in by_team.columns:
        team_row = by_team[by_team["Name"] == TEAM_NAME]

    by_player = abs_data.get("by_player", pd.DataFrame())
    mariners_players = pd.DataFrame()
    if not by_player.empty and "Tm" in by_player.columns:
        mariners_players = by_player[by_player["Tm"] == "SEA"]
    elif not by_player.empty and "Name" in by_player.columns:
        # fallback if there's no team column on this table -- would
        # need a roster cross-reference instead, flagged rather than
        # silently returning nothing
        print("  [warn] no 'Tm' column on player_stats table -- can't "
              "filter to Mariners players directly, check the real "
              "columns with abs_debug.py")

    return {"team": team_row, "players": mariners_players}


if __name__ == "__main__":
    abs_data = get_abs_challenges()
    summary = get_mariners_abs_summary(abs_data)

    print("\n-- SEATTLE MARINERS TEAM ABS STATS --")
    if not summary["team"].empty:
        print(summary["team"].to_string(index=False))
    else:
        print("  (not found -- check real column/team-name format with abs_debug.py)")

    print("\n-- SEATTLE MARINERS PLAYER ABS STATS --")
    if not summary["players"].empty:
        print(summary["players"].to_string(index=False))
    else:
        print("  (not found -- check real column/team-name format with abs_debug.py)")