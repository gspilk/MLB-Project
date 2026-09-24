"""
minor_league_scraper.py

Real Seattle Mariners minor-league affiliate + top-prospects scraper, built
from bbref's free affiliate page:

    https://www.baseball-reference.com/register/affiliate.cgi?id=SEA&year=2026

Confirmed (via a live fetch of the real page) real section anchors:
    all_top_prospects   -> three separate ranking tables (BA / MLB.com / BP)
    all_aff_batting     -> real table id: aff_batting
    all_aff_pitching    -> real table id: aff_pitching
    all_league_fielding -> real table id: league_fielding   (NOT aff_fielding -- confirmed real id)
    all_team_batting    -> real table id: team_batting   (organizational batting)
    all_team_pitching   -> real table id: team_pitching  (organizational pitching)

Like every other bbref table in this project, the aff_*/team_*/league_fielding
tables are expected to be hidden inside HTML comments (bbref's standard
"login wall" trick for tables outside the main box score pages), so this
file uses the same live-HTML-first / comment-search-second pattern as
seattle_scraper.py, standings_scraper.py, and schedule_scraper.py.

If any real id below turns out to still be wrong once you actually run this
against your own cached/live fetch, run minor_league_debug.py first --
it prints every live table id, every commented-out table id, and every
h2/h3 heading actually on the page, so you can fix the id here instead of
guessing again.
"""

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# Real, confirmed table ids on the affiliate page (see module docstring)
REAL_TABLE_IDS = {
    "affiliate_batting": "aff_batting",
    "affiliate_pitching": "aff_pitching",
    "affiliate_fielding": "league_fielding",
    "org_batting": "team_batting",
    "org_pitching": "team_pitching",
}

TTL_HOURS = 24


def _affiliate_url(team: str, season: int) -> str:
    return f"https://www.baseball-reference.com/register/affiliate.cgi?id={team}&year={season}"


def _is_stale(path: Path, ttl_hours: int = TTL_HOURS) -> bool:
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours > ttl_hours


def _fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _find_table(soup: BeautifulSoup, table_id: str):
    """
    Real, established pattern for this project: bbref hides most secondary
    tables inside HTML comments. Check live HTML first, then search every
    comment block for the same id.
    """
    table = soup.find("table", id=table_id)
    if table is not None:
        return table

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if table_id not in c:
            continue
        inner = BeautifulSoup(c, "html.parser")
        table = inner.find("table", id=table_id)
        if table is not None:
            return table
    return None


def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in ("Name", "Tm", "Team", "Lg", "Level", "Pos", "Affiliate"):
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # real non-numeric column (e.g. "Managers", combined Tm strings like "ARK,TAC") -- leave as-is
    return df


def _table_to_df(table) -> pd.DataFrame:
    df = pd.read_html(io.StringIO(str(table)))[0]

    # Flatten a MultiIndex header if bbref grouped this one (fielding tables
    # in this project consistently do; batting/pitching usually don't, but
    # this keeps the function safe either way).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            c[-1] if not str(c[-1]).startswith("Unnamed") else c[0]
            for c in df.columns
        ]

    # Drop bbref's repeated mid-table header rows (e.g. "Rk" == "Rk")
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"]

    df = df.reset_index(drop=True)
    return _clean_numeric(df)


def get_minor_league_data(team: str = "SEA", season: int = 2026, force_refresh: bool = False) -> dict:
    """
    Real affiliate + organizational batting/pitching/fielding tables for one
    team-season, e.g. Cache -> disk (parquet), same TTL convention as the
    rest of this project's scrapers.

    Returns a dict with keys: affiliate_batting, affiliate_pitching,
    affiliate_fielding, org_batting, org_pitching -- any table bbref doesn't
    have for this team/season comes back as an empty DataFrame rather than
    raising, since not every affiliate reports every stat category.
    """
    cache_path = CACHE_DIR / f"minor_league_{team}_{season}.parquet"
    meta_ok = not force_refresh and not _is_stale(cache_path)

    if meta_ok:
        print("[cache] loading minor league data from disk")
        cached = pd.read_parquet(cache_path)
        return {
            name: cached[cached["_table"] == name].drop(columns=["_table"]).reset_index(drop=True)
            for name in REAL_TABLE_IDS
        }

    url = _affiliate_url(team, season)
    print(f"[fetch] {url}")
    soup = _fetch_soup(url)

    results = {}
    for key, table_id in REAL_TABLE_IDS.items():
        table = _find_table(soup, table_id)
        if table is None:
            print(f"  [warn] table id {table_id!r} not found for {team} {season}")
            results[key] = pd.DataFrame()
            continue
        results[key] = _table_to_df(table)
        print(f"  [ok] {key} ({table_id}): {len(results[key])} rows")

    # cache combined (tag each frame with which table it came from so we can
    # split it back apart on load, same trick used elsewhere in this project
    # for multi-frame caches)
    tagged = []
    for name, frame in results.items():
        if frame.empty:
            continue
        f = frame.copy()
        f["_table"] = name
        tagged.append(f)
    if tagged:
        pd.concat(tagged, ignore_index=True).to_parquet(cache_path)

    return results


def get_top_prospects(team: str = "SEA", season: int = 2026, force_refresh: bool = False) -> pd.DataFrame:
    """
    Real Top Prospects rankings from the same affiliate page -- three
    separate ranking lists (Baseball America / MLB.com / BaseballProspectus)
    live inside the all_top_prospects section, each a small rank/name/
    position table rather than the aff_*/team_* stat tables above.

    Returns one combined DataFrame with columns: source, rank, name, position.
    """
    cache_path = CACHE_DIR / f"top_prospects_{team}_{season}.parquet"
    if not force_refresh and not _is_stale(cache_path):
        print("[cache] loading top prospects from disk")
        return pd.read_parquet(cache_path)

    url = _affiliate_url(team, season)
    print(f"[fetch] {url}")
    soup = _fetch_soup(url)

    wrapper = soup.find(id="all_top_prospects")
    if wrapper is None:
        # fall back to comment search for the whole wrapper div, same as
        # the individual-table case above
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if "all_top_prospects" not in c:
                continue
            inner = BeautifulSoup(c, "html.parser")
            wrapper = inner.find(id="all_top_prospects")
            if wrapper is not None:
                break

    if wrapper is None:
        print("  [warn] top prospects section not found")
        return pd.DataFrame(columns=["source", "rank", "name", "position"])

    rows = []
    # Real fix: bbref's own <caption> tag lives INSIDE each prospect table
    # (confirmed by minor_league_debug.py: the three live tables print with
    # caption='Baseball America' / 'MLB.com' / 'BaseballProspectus.com' and
    # no separate id). Reading the caption off each table directly -- rather
    # than tracking headings in document-traversal order -- avoids the
    # off-by-one bug where table 1's real data got labeled with table 2's
    # source name, table 2's with table 3's name, etc.
    for table_el in wrapper.find_all("table"):
        caption_el = table_el.find("caption")
        source = caption_el.get_text(strip=True) if caption_el else "Unknown"

        try:
            df = pd.read_html(io.StringIO(str(table_el)))[0]
        except ValueError:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]

        # normalize expected columns: something rank-like, a name, a position
        cols_lower = {c: str(c).strip().lower() for c in df.columns}
        rank_col = next((c for c, lc in cols_lower.items() if "rk" in lc or "rank" in lc), df.columns[0])
        name_col = next((c for c, lc in cols_lower.items() if "name" in lc or "player" in lc), df.columns[1] if len(df.columns) > 1 else df.columns[0])
        pos_col = next((c for c, lc in cols_lower.items() if lc in ("pos", "position")), None)
        if pos_col is None and len(df.columns) >= 3:
            # real prospect tables are just rank/name/position -- if the
            # header text didn't say "Pos" outright, the 3rd column is it
            leftover = [c for c in df.columns if c not in (rank_col, name_col)]
            pos_col = leftover[0] if leftover else None

        for _, r in df.iterrows():
            name = str(r[name_col]).strip()
            if not name or name.lower() == "name":
                continue
            # real rank is always numeric (7, 25, 58...); bbref also embeds
            # a non-data separator row ("Before 2026 Season") in this table
            # that isn't a real prospect -- drop anything whose rank isn't
            # actually a number
            try:
                rank_val = int(str(r[rank_col]).strip())
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "source": source,
                    "rank": rank_val,
                    "name": name,
                    "position": str(r[pos_col]).strip() if pos_col is not None else None,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_parquet(cache_path)
    return result


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