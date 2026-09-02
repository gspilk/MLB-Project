"""
trade_cost_package.py
Two things:
  1. Team-wide Mariners performance before vs. after the Aug 3 deadline
     (same method as team_before_after_deadline.py -- real, complete
     game log, no gaps).
  2. Real, LIVE tracking of the two MLB-level players Seattle traded
     away -- Luis Castillo (now White Sox) and Alex Hoppe (now
     Orioles) -- pulled directly from THEIR new teams' bbref pages,
     not a frozen snapshot.

Brock Moore and Harrison Kreiling (the two pitching prospects also sent
to Baltimore in the Ward trade) are minor leaguers -- same situation as
Jones/Smith in trade_return_package.py, no MiLB scraper exists in this
project, so they're a real, dated, sourced snapshot rather than live
data.

Usage:
    python trade_cost_package.py
"""

import time
import requests
import pandas as pd
from datetime import date
from bs4 import BeautifulSoup, Comment
from data_builder import build_all

DEADLINE_DATE = date(2026, 8, 3)

# Real, sourced minor-league snapshot as of the trade (early Aug 2026).
# Not live data -- see trade_return_package.py's docstring for why.
MOORE_SNAPSHOT = {
    "name": "Brock Moore", "as_of": "2026-08-04 (trade date)",
    "level": "Double-A Arkansas (Mariners org pre-trade)",
    "note": "2024 7th-round pick. Recently promoted to Double-A after a "
           "1.13 ERA, 4 saves stint with the AquaSox. Ranked No. 24 in "
           "Seattle's system -- the headline prospect in the return. "
           "Sits upper-90s from a funky, 6'6\" delivery; still working "
           "on command (~16% walk rate at Double-A).",
}
KREILING_SNAPSHOT = {
    "name": "Harrison Kreiling", "as_of": "2026-08-04 (trade date)",
    "level": "Recently debuted (Mariners org pre-trade)",
    "note": "2024 17th-round pick out of Nebraska-Omaha. Missed all of "
           "2025 recovering from injury; made his pro debut in 2026, "
           "debuting mid-July. Not yet a ranked Top 30 prospect -- the "
           "least established of the four pieces sent out.",
}


def _period_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df["W/L"].astype(str).str.startswith("W").sum()
    losses = len(df) - wins
    r = pd.to_numeric(df["R"], errors="coerce")
    ra = pd.to_numeric(df["RA"], errors="coerce")
    games = len(df)
    return {
        "games": games, "record": f"{wins}-{losses}",
        "win_pct": round(wins / games, 3) if games else None,
        "rs_g": round(r.mean(), 2) if not r.empty else None,
        "ra_g": round(ra.mean(), 2) if not ra.empty else None,
        "run_diff": round((r.sum() - ra.sum()) / games, 2) if games else None,
    }


def team_before_after(data: dict) -> dict:
    completed = data["schedule"]["completed"].copy()
    completed["date"] = pd.to_datetime(completed["date"]).dt.date
    before = completed[completed["date"] < DEADLINE_DATE]
    after = completed[completed["date"] >= DEADLINE_DATE]
    return {"before": _period_stats(before), "after": _period_stats(after)}


def _fetch_team_page(team_code: str, season: int = 2026):
    """Reuses the same fetch/parse pattern as mariners_stats.py, just
    parameterized to any team instead of hardcoded to SEA."""
    url = f"https://www.baseball-reference.com/teams/{team_code}/{season}.shtml"
    print(f"  [GET] {url}")
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9",
    })
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)
    return BeautifulSoup(resp.text, "html.parser")


def _find_table(soup, table_id: str):
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


def get_player_on_new_team(team_code: str, table_id: str, name_fragment: str) -> dict:
    """Live-scrapes a player's current-season stats from their NEW
    team's bbref page. Used for Castillo (White Sox) and Hoppe
    (Orioles) -- both still active MLB players, just not Mariners
    anymore, so their real current numbers exist on a different page
    than the one this project normally scrapes.

    BUG FIX: an earlier version of this function parsed the raw table
    with plain pd.read_html() and assumed a "Name" column existed
    exactly as-is -- that's wrong for bbref's actual table structure
    (confirmed by a real run: KeyError: 'Name'). mariners_stats.py's
    _clean() function already handles this correctly (auto-detects the
    name column, strips bbref's asterisks/hashes, drops repeated header
    rows) -- it's proven correct every single main.py run against this
    exact same table structure for Seattle, just applied here to a
    different team's page instead of reimplementing it badly."""
    from mariners_stats import _clean, _read_html

    try:
        soup = _fetch_team_page(team_code)
        tag = _find_table(soup, table_id)
        if tag is None:
            return {}
        raw_df = _read_html(tag)
        df = _clean(raw_df)
        if "Name" not in df.columns:
            print(f"  [warn] no Name column after cleaning {team_code} table "
                  f"(columns: {list(df.columns)[:5]}...)")
            return {}
        row = df[df["Name"].astype(str).str.contains(name_fragment, na=False)]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()
    except Exception as e:
        print(f"  [warn] could not fetch {team_code} page: {e}")
        return {}


def print_report(before_after: dict, castillo: dict, hoppe: dict):
    b, a = before_after["before"], before_after["after"]
    print("\n" + "=" * 70)
    print(f"MARINERS TEAM -- BEFORE vs. AFTER THE {DEADLINE_DATE} DEADLINE")
    print("=" * 70)
    print(f"  {'':<14} {'BEFORE':>12} {'AFTER':>12}")
    print(f"  {'Games':<14} {b.get('games',''):>12} {a.get('games',''):>12}")
    print(f"  {'Record':<14} {b.get('record',''):>12} {a.get('record',''):>12}")
    print(f"  {'Win %':<14} {b.get('win_pct',''):>12} {a.get('win_pct',''):>12}")
    print(f"  {'RS/G':<14} {b.get('rs_g',''):>12} {a.get('rs_g',''):>12}")
    print(f"  {'RA/G':<14} {b.get('ra_g',''):>12} {a.get('ra_g',''):>12}")
    print(f"  {'Run diff/G':<14} {b.get('run_diff',''):>12} {a.get('run_diff',''):>12}")

    print("\n" + "=" * 70)
    print("TRADE COST PACKAGE -- everyone the Mariners gave up")
    print("=" * 70)
    print("\n-- MLB-level, live on their new team --")
    if castillo:
        print(f"  Luis Castillo (now CHW): {castillo.get('IP','?')} IP, "
              f"{castillo.get('ERA','?')} ERA (full-season CHW line)")
    else:
        print("  Luis Castillo (now CHW): could not fetch live data")
    if hoppe:
        print(f"  Alex Hoppe (now BAL):    {hoppe.get('IP','?')} IP, "
              f"{hoppe.get('ERA','?')} ERA (full-season BAL line)")
    else:
        print("  Alex Hoppe (now BAL):    could not fetch live data")

    print("\n-- Minor league (real, dated snapshot -- not live) --")
    for p in (MOORE_SNAPSHOT, KREILING_SNAPSHOT):
        print(f"  {p['name']} ({p['level']})  [as of {p['as_of']}]")
        print(f"    -> {p['note']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    before_after = team_before_after(data)

    print("\nFetching Castillo's current White Sox stats...")
    castillo = get_player_on_new_team("CHW", "players_standard_pitching", "Castillo")

    print("Fetching Hoppe's current Orioles stats...")
    hoppe = get_player_on_new_team("BAL", "players_standard_pitching", "Hoppe")

    print_report(before_after, castillo, hoppe)