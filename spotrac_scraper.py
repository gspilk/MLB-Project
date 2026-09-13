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
TEAM_SLUG       = "seattle-mariners"
CACHE_DIR       = os.path.join(os.path.dirname(__file__), "data", "cache")
CACHE_TTL_HOURS = 24
PAYROLL_URL     = f"https://www.spotrac.com/mlb/{TEAM_SLUG}/payroll/"


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


def get_mariners_payroll(force_refresh: bool = False) -> dict:
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

    REAL TABLE ORDER (confirmed 2026-09-11 against a live fetch, since
    Spotrac's page has no id attributes to target directly like bbref):
      0: active payroll (26-man)   1: injured list   2 & 3: retained
      (traded/released, in two separate small tables -- combined here)
      4: payroll totals (broken/merged header, needs special parsing --
      see _parse_totals_table())   5: minor league / 40-man depth

    This positional mapping is inherently more fragile than bbref's
    id-based lookups -- if Spotrac adds/removes/reorders a section,
    this breaks silently rather than erroring. The sanity check below
    (verifying table 0 actually looks like a payroll table) exists to
    catch that kind of drift early rather than returning wrong data
    quietly.
    """
    cache_paths = {k: _cache_path(f"spotrac_{k}") for k in
                   ("active_payroll", "injured_list", "retained", "totals", "minors")}
    all_cached = all(os.path.exists(p) for p in cache_paths.values())

    if not force_refresh and all_cached and not any(_is_stale(p) for p in cache_paths.values()):
        print("[cache] loading Spotrac payroll from disk")
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    print("[fetch] Seattle Mariners payroll from Spotrac ...")
    html = _fetch_page(PAYROLL_URL)
    tables = pd.read_html(io.StringIO(html))
    print(f"  [ok] {len(tables)} tables found on page")

    if len(tables) < 6:
        print(f"  [warn] expected 6 tables, found {len(tables)} -- "
              f"Spotrac's page structure has likely changed, check "
              f"the real page manually (run spotrac_debug.py)")
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
        print(f"  [warn] table 0 doesn't look like active payroll data "
              f"(no 'Payroll Salary' column found) -- the positional "
              f"mapping above may be stale, re-run spotrac_debug.py "
              f"to check the real current structure")

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


if __name__ == "__main__":
    payroll = get_mariners_payroll()
    for label, df in payroll.items():
        print(f"\n-- {label.upper()} --")
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  (empty)")