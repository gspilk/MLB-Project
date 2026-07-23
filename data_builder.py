"""
data_builder.py
Assembles all scrapers into clean datasets.
Everything else imports from here — never import scrapers directly.
"""

import os
import time
import pandas as pd

from standings_scraper         import get_standings, get_mariners_context, get_all_teams
from mariners_schedule_scraper import get_schedule
from mariners_stats            import get_seattle_stats
from battingpitching_scraper  import get_batting, get_pitching
from mlb_team_stats            import get_mlb_overview
from statcast_scraper          import (get_statcast, get_mariners_batters,
                                       get_mariners_pitchers, get_luck_analysis)
from roster                    import get_roster_keys

SEASON = 2026

# ── cache TTL per data type ───────────────────────────────────────────────────
CACHE_TTL = {
    "standings": 6,    # updates after every game
    "schedule":  6,    # same
    "seattle":   6,    # player lines update daily
    "batting":   24,   # leaders stable
    "pitching":  24,   # leaders stable
    "overview":  24,   # team totals stable
    "statcast":  72,   # manual CSV download
}


def build_standings(season=SEASON, force_refresh=False) -> dict:
    print("[build] standings ...")
    division, expanded = get_standings(season, force_refresh=force_refresh)
    return {
        "division":  division,
        "expanded":  expanded,
        "mariners":  get_mariners_context(division, expanded),
        "all_teams": get_all_teams(division),
    }


def build_schedule(season=SEASON, force_refresh=False) -> dict:
    print("[build] schedule ...")
    completed, next7, remaining = get_schedule(season, force_refresh=force_refresh)
    return {
        "completed": completed,
        "next7":     next7,
        "remaining": remaining,
    }


def build_seattle(season=SEASON, force_refresh=False) -> dict:
    print("[build] seattle stats ...")
    return get_seattle_stats(season, force_refresh=force_refresh)


def build_batting(season=SEASON, force_refresh=False) -> dict:
    print("[build] batting leaders ...")
    return get_batting(season, force_refresh=force_refresh)


def build_pitching(season=SEASON, force_refresh=False) -> dict:
    print("[build] pitching leaders ...")
    return get_pitching(season, force_refresh=force_refresh)


def build_overview(season=SEASON, force_refresh=False) -> dict:
    print("[build] MLB overview ...")
    return get_mlb_overview(season, force_refresh=force_refresh)


def build_statcast(roster_keys=None) -> dict:
    print("[build] statcast ...")
    batters, pitchers = get_statcast()
    sea_bat = get_mariners_batters(batters, roster_keys)
    sea_pit = get_mariners_pitchers(pitchers, roster_keys)
    return {
        "batters":     batters,
        "pitchers":    pitchers,
        "sea_batters": sea_bat,
        "sea_pitchers":sea_pit,
        "bat_luck":    get_luck_analysis(sea_bat, is_pitcher=False),
        "pit_luck":    get_luck_analysis(sea_pit, is_pitcher=True),
    }


def build_all(season=SEASON, force_refresh=False) -> dict:
    start = time.time()
    print(f"\n{'='*50}")
    print(f"Building MLB data for {season} season...")
    print(f"{'='*50}\n")

    data = {
        "standings": build_standings(season, force_refresh),
        "schedule":  build_schedule(season,  force_refresh),
        "seattle":   build_seattle(season,   force_refresh),
        "batting":   build_batting(season,   force_refresh),
        "pitching":  build_pitching(season,  force_refresh),
        "overview":  build_overview(season,  force_refresh),
    }

    # roster comes from the live 40-man scrape above -- used to identify
    # which Statcast rows belong to the Mariners without a hardcoded name
    # list. Uses precise last+first-initial keys (not last-name-only) since
    # the roster has real last-name collisions with other teams' players
    # (e.g. more than one "Wilson" in MLB). If the roster table failed to
    # scrape, this comes back empty and get_mariners_batters/pitchers fall
    # back to their bundled list with a warning (see statcast_scraper.py).
    roster_keys = get_roster_keys(data)
    print(f"[build] live roster: {len(roster_keys)} players")

    data["statcast"] = build_statcast(roster_keys)

    elapsed = round(time.time() - start, 1)
    print(f"\n{'='*50}")
    print(f"All data built in {elapsed}s")
    print(f"{'='*50}\n")
    return data


# ── convenience accessors ─────────────────────────────────────────────────────
def get_mariners_batting(data):
    return data.get("seattle", {}).get("batting", pd.DataFrame())

def get_mariners_pitching(data):
    return data.get("seattle", {}).get("pitching", pd.DataFrame())

def get_mariners_statcast_bat(data):
    return data.get("statcast", {}).get("sea_batters", pd.DataFrame())

def get_mariners_statcast_pit(data):
    return data.get("statcast", {}).get("sea_pitchers", pd.DataFrame())

def get_schedule_next7(data):
    return data.get("schedule", {}).get("next7", pd.DataFrame())

def get_al_west(data):
    return data.get("standings", {}).get("division", {}).get("AL_West", pd.DataFrame())

def get_mariners_record(data):
    return data.get("standings", {}).get("mariners", {})


if __name__ == "__main__":
    data = build_all(2026)

    print("\n── Mariners record ──")
    ctx = get_mariners_record(data)
    for k in ["mlb_rank","div_rank","wc_gap","record"]:
        print(f"  {k}: {ctx.get(k)}")

    print("\n── AL West ──")
    al_west = get_al_west(data)
    if not al_west.empty:
        cols = [c for c in ["Tm","W","L","W-L%","GB"] if c in al_west.columns]
        print(al_west[cols].to_string(index=False))

    # debug batting all_players columns
    bat = data.get("batting",{}).get("all_players")
    if bat is not None:
        print(f"\n── batting all_players cols: {list(bat.columns[:10])}")
        print(f"   Tm sample: {bat['Tm'].value_counts().head(5).to_dict() if 'Tm' in bat.columns else 'NO TM'}")
        print(f"   rows: {len(bat)}")