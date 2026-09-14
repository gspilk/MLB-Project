"""
spotrac_scraper.py
Scrapes Seattle Mariners real salary/payroll data from spotrac.com
https://www.spotrac.com/mlb/seattle-mariners/payroll/

UNLIKE bbref, Spotrac's page has no clean table `id` attributes to
target directly -- it's several distinct tables stacked on one page
(Active Payroll, Injured List, two Retained Salary sections, a Payroll
Summary, and Minor League/40-man depth), identified only by position
on the page and a preceding text header, not a machine-readable id.
Confirmed 2026-09-11 against a live fetch (via spotrac_debug.py) --
this scraper uses that real, verified table order, not a guess.

CONFIRMED WORKING (2026-09-11): the plain, year-less URL
(.../payroll/, no year in the path) reliably returns the CURRENT
season's data -- verified directly against a live fetch. Do NOT use
the year-in-path variant (.../payroll/2026/) -- testing showed it
actually serves a DIFFERENT (older) year's data despite the year in
the URL, a real, confirmed labeling inconsistency on Spotrac's own
site, not a guess.

This is a genuinely different, more fragile scrape than bbref's pages
-- column names, section order, or table count could all change
without warning, since there's no stable id to anchor to. Treat
failures here as "check the real page structure again," not
necessarily a bug in this file.

Usage:
    from spotrac_scraper import get_mariners_payroll
    payroll = get_mariners_payroll()
"""

import os
import io
import time
import pandas as pd
import requests

# ── config ────────────────────────────────────────────────────────────────────
TEAM_SLUG       = "seattle-mariners"  # kept as the default for backward
                                      # compatibility with get_mariners_payroll()
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24
PAYROLL_URL     = f"https://www.spotrac.com/mlb/{TEAM_SLUG}/payroll/"

# Real team slugs for Spotrac's standard hyphenated-lowercase-name URL
# pattern (confirmed for seattle-mariners via a live fetch tonight).
# NOT independently verified for the other 29 -- if a team's real page
# doesn't match, it likely just needs the slug corrected here, same as
# how the Mariners page needed one fix (the year-in-path quirk) before
# this worked cleanly.
TEAM_SLUGS = {
    "Arizona Diamondbacks":  "arizona-diamondbacks",
    "Atlanta Braves":        "atlanta-braves",
    "Baltimore Orioles":     "baltimore-orioles",
    "Boston Red Sox":        "boston-red-sox",
    "Chicago Cubs":          "chicago-cubs",
    "Chicago White Sox":     "chicago-white-sox",
    "Cincinnati Reds":       "cincinnati-reds",
    "Cleveland Guardians":   "cleveland-guardians",
    "Colorado Rockies":      "colorado-rockies",
    "Detroit Tigers":        "detroit-tigers",
    "Houston Astros":        "houston-astros",
    "Kansas City Royals":    "kansas-city-royals",
    "Los Angeles Angels":    "los-angeles-angels",
    "Los Angeles Dodgers":   "los-angeles-dodgers",
    "Miami Marlins":         "miami-marlins",
    "Milwaukee Brewers":     "milwaukee-brewers",
    "Minnesota Twins":       "minnesota-twins",
    "New York Mets":         "new-york-mets",
    "New York Yankees":      "new-york-yankees",
    "Athletics":             "athletics",
    "Philadelphia Phillies": "philadelphia-phillies",
    "Pittsburgh Pirates":    "pittsburgh-pirates",
    "San Diego Padres":      "san-diego-padres",
    "San Francisco Giants":  "san-francisco-giants",
    "Seattle Mariners":      "seattle-mariners",
    "St. Louis Cardinals":   "st-louis-cardinals",
    "Tampa Bay Rays":        "tampa-bay-rays",
    "Texas Rangers":         "texas-rangers",
    "Toronto Blue Jays":     "toronto-blue-jays",
    "Washington Nationals":  "washington-nationals",
}


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
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return session


def _fetch_page(url: str) -> str:
    print(f"  [GET] {url}")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)
    return resp.text


def _find_table_by_columns(tables: list, required_cols: list) -> pd.DataFrame:
    """
    Given the list of DataFrames pandas.read_html() finds on the page,
    returns the first one whose columns contain ALL of required_cols
    (case-insensitive substring match, since Spotrac's headers can carry
    odd whitespace/casing). Returns an empty DataFrame if none match.
    Used only as a sanity check now -- see get_mariners_payroll()'s
    real positional mapping below for why.
    """
    for df in tables:
        cols_lower = [str(c).lower() for c in df.columns]
        if all(any(req.lower() in c for c in cols_lower) for req in required_cols):
            return df
    return pd.DataFrame()


def _parse_totals_table(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Table 4 (Payroll Summary) comes out of pandas.read_html() with a
    broken header -- Spotrac's real HTML uses a merged/spanning header
    cell here that read_html can't split properly, so every column
    shows up as "Unnamed: N" and the real header text ("Payroll
    Summary") ends up duplicated into the first data row instead.
    Rebuilds a clean 2-column (label, value) table from what's usable.
    """
    if raw.empty or len(raw.columns) < 2:
        return pd.DataFrame()
    clean = raw.iloc[1:, :2].copy()  # skip the broken header-as-data row
    clean.columns = ["category", "value"]
    return clean.reset_index(drop=True)


def get_team_payroll(team_name: str = "Seattle Mariners", force_refresh: bool = False) -> dict:
    """
    Returns:
        {
            "active_payroll":  DataFrame -- active roster salaries
            "injured_list":    DataFrame -- IL players, salary still counts
            "retained":        DataFrame -- salary owed to traded/released players
            "totals":          DataFrame -- payroll summary (category, value)
            "minors":          DataFrame -- 40-man/minor-league salaries
        }
    Each DataFrame is real, live-scraped (or cached) data -- no
    hardcoded salary figures anywhere in this file.

    `team_name` must be a real key in TEAM_SLUGS (e.g. "Houston Astros").
    Generalized 2026-09-13 from a Mariners-only version so the same real
    scraping logic (see REAL TABLE ORDER note below) works for any of
    the 30 real team slugs -- needed to compare Seattle's real spending
    against actual rivals, not just look at it in isolation.

    REAL TABLE ORDER (confirmed 2026-09-11 against a live Mariners
    fetch, since Spotrac's page has no id attributes to target directly
    like bbref): 0: active payroll (26-man)   1: injured list   2 & 3:
    retained (traded/released, in two separate small tables -- combined
    here)   4: payroll totals (broken/merged header, needs special
    parsing -- see _parse_totals_table())   5: minor league / 40-man
    depth. NOT independently re-verified for the other 29 teams --
    if another team's page has a different number of sections (e.g. no
    retained-salary tables at all for a team with no recent trades),
    this positional mapping could be wrong for that specific team. The
    sanity check below only catches table 0 drift, not a shifted count
    elsewhere -- treat other teams' results as first-pass, not as
    thoroughly proven as the Mariners case.
    """
    if team_name not in TEAM_SLUGS:
        print(f"  [warn] '{team_name}' not in TEAM_SLUGS -- check spelling "
              f"against the real dict keys")
        return {}
    slug = TEAM_SLUGS[team_name]
    url = f"https://www.spotrac.com/mlb/{slug}/payroll/"
    safe_name = slug.replace("-", "_")

    cache_paths = {k: _cache_path(f"spotrac_{safe_name}_{k}") for k in
                   ("active_payroll", "injured_list", "retained", "totals", "minors")}
    all_cached = all(os.path.exists(p) for p in cache_paths.values())

    if not force_refresh and all_cached and not any(_is_stale(p) for p in cache_paths.values()):
        print(f"[cache] loading {team_name} payroll from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print(f"[fetch] {team_name} payroll from Spotrac ...")
    html = _fetch_page(url)
    tables = pd.read_html(io.StringIO(html))
    print(f"  [ok] {len(tables)} tables found on page")

    if len(tables) < 6:
        print(f"  [warn] expected 6 tables, found {len(tables)} for {team_name} -- "
              f"this team's page structure may differ from the Mariners "
              f"case this scraper was built against, check manually")
        return {k: pd.DataFrame() for k in cache_paths}

    active_payroll = tables[0]
    injured_list   = tables[1]
    # BUG FIX: tables 2 and 3 use DIFFERENT column names for the same
    # logical "player name" field (table 2: "Player (10)", table 3:
    # "Player (2)"), and table 3 has one extra column ("VIA") table 2
    # doesn't. A plain pd.concat() correctly-but-unhelpfully keeps both
    # as separate columns instead of merging them, leaving table 3's
    # rows (e.g. Knizner, Suarez) with a blank name in the "wrong"
    # column. Rename both to a shared "player" column before combining.
    t2 = tables[2].rename(columns={tables[2].columns[0]: "player"})
    t3 = tables[3].rename(columns={tables[3].columns[0]: "player"})
    retained = pd.concat([t2, t3], ignore_index=True)
    totals         = _parse_totals_table(tables[4])
    minors         = tables[5]

    # sanity check: table 0 should actually look like a payroll table --
    # catches silent drift if Spotrac reorders sections
    sanity = _find_table_by_columns([active_payroll], ["Payroll Salary"])
    if sanity.empty:
        print(f"  [warn] table 0 doesn't look like active payroll data for "
              f"{team_name} (no 'Payroll Salary' column found) -- the "
              f"positional mapping above may not hold for this team")

    for label, df in [("active_payroll", active_payroll), ("injured_list", injured_list),
                      ("retained", retained), ("totals", totals), ("minors", minors)]:
        print(f"  [ok]   {label}: {len(df)} rows, {len(df.columns)} cols")

    result = {
        "active_payroll": active_payroll, "injured_list": injured_list,
        "retained": retained, "totals": totals, "minors": minors,
    }
    for k, df in result.items():
        if not df.empty:
            df.to_parquet(cache_paths[k], index=False)

    return result


def get_league_cash_totals(force_refresh: bool = False) -> pd.DataFrame:
    """
    Real, much simpler alternative to get_all_teams_payroll(): Spotrac
    has a single page (spotrac.com/mlb/cash) listing every team's real
    cash payroll totals (Active, Disabled List, Retained, Buried,
    Suspended, Total) in ONE flat table -- no MultiIndex headers, no
    30 separate fetches needed.

    CAVEAT, found 2026-09-13: fetching this URL has shown genuinely
    inconsistent year data between a search-index snippet ("2026") and
    a direct fetch ("2024") -- same kind of staleness/caching quirk hit
    earlier with the team-specific payroll page, which turned out to be
    fetch-side, not a real site inconsistency. VERIFY what year you
    actually get back before trusting this -- see the year-detection
    check below, which prints a clear warning if it doesn't look like
    the current season.
    """
    cache_path = _cache_path("spotrac_league_cash")

    if not force_refresh and not _is_stale(cache_path):
        print("[cache] loading league cash totals from disk")
        return pd.read_parquet(cache_path)

    url = "https://www.spotrac.com/mlb/cash"
    print(f"[fetch] league-wide cash totals from Spotrac ...")
    html = _fetch_page(url)

    tables = pd.read_html(io.StringIO(html))
    if not tables:
        print("  [warn] no tables found on the page")
        return pd.DataFrame()

    df = tables[0]
    # BUG FIX: originally filtered out a row literally containing
    # "League Average" text, based on an earlier fetch that had a
    # divider row with that exact label. The real, current page
    # instead has TWO trailing summary rows labeled "Totals" and
    # "Averages" -- different text, so the old filter missed both,
    # counting 32 "teams" instead of the real 30. Both trailing rows
    # (and any future differently-worded ones) share one reliable
    # trait: no real Rank number, since they're not actually ranked
    # teams. Filtering on that directly is more robust than matching
    # specific label text that can change.
    rank_col = df.columns[0]
    df = df[pd.to_numeric(df[rank_col], errors="coerce").notna()]

    print(f"  [ok] {len(df)} teams found")

    # Real sanity check, replacing an earlier fragile regex-based year
    # search (looked for exact text that turned out not to match this
    # page's real structure, producing a false alarm despite the data
    # actually being correct). Uses the real "Record" column instead --
    # a mostly-empty or all-zero record column would be the real,
    # checkable sign of stale/wrong data, not a guessed text pattern.
    record_col = next((c for c in df.columns if "record" in str(c).lower()), None)
    if record_col:
        sample = df[record_col].dropna().head(3).tolist()
        print(f"  [info] sample records found: {sample} -- cross-check "
              f"one of these against a team's real current record to "
              f"confirm this fetch got current data, not stale")
    else:
        print(f"  [warn] no 'Record' column found -- can't sanity-check "
              f"against a real, known win-loss total, verify manually")

    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def get_mariners_payroll(force_refresh: bool = False) -> dict:
    """Backward-compatible wrapper -- same as get_team_payroll('Seattle Mariners')."""
    return get_team_payroll("Seattle Mariners", force_refresh)


def get_all_teams_payroll(force_refresh: bool = False) -> dict:
    """
    Returns {team_name: payroll_dict} for all 30 real MLB teams.

    SLOW BY DESIGN: each team is a separate real fetch with the same
    polite 3-second delay used throughout this project (see
    _fetch_page()) -- 30 teams means this takes several minutes
    minimum, not something to run casually or often. Uses each team's
    own cache independently, so a second run within 24 hours mostly
    just loads from disk and is fast.
    """
    results = {}
    for i, team_name in enumerate(TEAM_SLUGS, 1):
        print(f"\n[{i}/30] {team_name}")
        results[team_name] = get_team_payroll(team_name, force_refresh)
    return results


if __name__ == "__main__":
    payroll = get_mariners_payroll()
    for label, df in payroll.items():
        print(f"\n-- {label.upper()} --")
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  (empty)")