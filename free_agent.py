"""
mlbtr_free_agents_scraper.py
Scrapes MLB Trade Rumors' real, comprehensive 2026-27 free agent class
list -- structured completely differently than bbref's tables (plain
text grouped under position headers like "Catchers", "First Basemen",
etc.), so this uses real regex text-parsing rather than pandas table
reading.

Confirmed real page structure (via a live fetch tonight):
  https://www.mlbtraderumors.com/2025/08/2026-27-mlb-free-agents.html
  "Updated 4-9-26" marker, then position-header sections, each
  followed by "Name (Age)" or "Name (Age) – option/contract detail"
  lines, ending after "Left-Handed Relievers" -- everything after that
  is reader comments, not part of the real list, and must be excluded.

Usage:
    from mlbtr_free_agents_scraper import get_mlbtr_free_agents
    fa = get_mlbtr_free_agents()
"""

import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24
URL             = "https://www.mlbtraderumors.com/2025/08/2026-27-mlb-free-agents.html"

# Real, confirmed position headers from the actual page tonight -- used
# both to detect section boundaries and to tag each parsed player with
# their real position group. "Left-Handed Relievers" is the real LAST
# section on the page; everything after it is reader comments.
POSITION_HEADERS = [
    "Catchers", "First Basemen", "Second Basemen", "Shortstops",
    "Third Basemen", "Left Fielders", "Center Fielders", "Right Fielders",
    "Designated Hitters", "Starting Pitchers", "Right-Handed Relievers",
    "Left-Handed Relievers",
]

# Matches "Name (Age)" or "Name (Age) – trailing option/contract detail"
PLAYER_LINE_RE = re.compile(r"^([A-Za-zÀ-ÿ.'\-\s]+?)\s*\((\d+)\)(?:\s*[–\-]\s*(.+))?$")


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
    })
    return session


def _fetch_lines() -> list:
    print(f"  [GET] {URL}")
    session = _make_session()
    resp = session.get(URL, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(2)

    # BUG FIX: originally tried to guess the real article container via
    # a loose class-name regex ("entry|post|content"), which a live run
    # showed matching the WRONG element entirely -- a small comment
    # widget (5 lines: a username, "1 year ago", a single comment,
    # "Reply"), not the real article body at all. Rather than guess
    # again at a more specific selector without being able to verify it
    # live, this just grabs the ENTIRE page's text -- parse_free_agents()
    # already has real, robust boundary detection built in (skips
    # everything before the first known position header, stops after
    # the last one), so it correctly ignores all the surrounding
    # navigation/sidebar/ad noise on its own without needing a precise
    # container guess at all.
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n")
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse_free_agents(lines: list) -> pd.DataFrame:
    """
    Real parsing logic: walks the page text line by line, tracking the
    current position header, extracting real "Name (Age)" matches, and
    stopping entirely once the last known real section (Left-Handed
    Relievers) ends -- everything after that is reader comments, not
    real list data, and would otherwise get incorrectly parsed as
    additional "players" (comment text can coincidentally look close
    to the pattern in places).
    """
    rows = []
    current_pos = None
    in_real_list = False
    reached_last_section = False

    for line in lines:
        if line in POSITION_HEADERS:
            current_pos = line
            in_real_list = True
            if line == POSITION_HEADERS[-1]:
                reached_last_section = True
            continue

        if not in_real_list:
            continue

        match = PLAYER_LINE_RE.match(line)
        if match:
            name, age, detail = match.groups()
            rows.append({
                "name": name.strip(),
                "age_2027": int(age),
                "position_group": current_pos,
                "option_detail": detail.strip() if detail else None,
            })
        elif reached_last_section:
            # first non-matching line after the real last section
            # started -- this is where reader comments begin, stop here
            break

    return pd.DataFrame(rows)


def get_mlbtr_free_agents(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = _cache_path("mlbtr_free_agents_2026_27")

    if not force_refresh and not _is_stale(cache_path):
        print("[cache] loading MLBTR free agents from disk")
        return pd.read_parquet(cache_path)

    print("[fetch] 2026-27 free agent class from MLB Trade Rumors ...")
    lines = _fetch_lines()
    df = parse_free_agents(lines)

    print(f"  [ok] {len(df)} real free agents parsed across "
          f"{df['position_group'].nunique() if not df.empty else 0} position groups")
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def find_player(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Real, direct check -- e.g. find_player(fa, 'Crawford') or
    find_player(fa, 'Arozarena') to confirm a specific player's real
    free agent status without scanning the whole list by eye."""
    if df.empty:
        return df
    return df[df["name"].str.contains(name, case=False, na=False)]


if __name__ == "__main__":
    fa = get_mlbtr_free_agents()

    if not fa.empty:
        print("\n-- FULL 2026-27 FREE AGENT CLASS --")
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
                print(f"  {name}: not found on the real free agent list")
    else:
        print("No data parsed -- check the real page structure, it may have changed")