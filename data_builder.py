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
from battingpitching_scraper  import get_batting, get_pitching, get_fielding
from mlb_team_stats            import get_mlb_overview
from statcast_scraper          import (get_statcast, get_mariners_batters,
                                       get_mariners_pitchers, get_luck_analysis,
                                       get_team_statcast, SEA_TEAM_ID)
from roster                    import get_roster_keys

SEASON = 2026

# ── cache TTL per data type ───────────────────────────────────────────────────
CACHE_TTL = {
    "standings": 6,    # updates after every game
    "schedule":  6,    # same
    "seattle":   6,    # player lines update daily
    "batting":   24,   # leaders stable
    "pitching":  24,   # leaders stable
    "fielding":  24,   # same -- season-long defensive totals don't move fast
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


def build_fielding(season=SEASON, force_refresh=False) -> dict:
    """
    NEW: league-wide fielding leaders, same shape as build_batting()/
    build_pitching() (al_players/nl_players/all_players/...). Added to
    close a real gap in free_agent_recommendations.py -- external free
    agents had no defensive data anywhere in this pipeline, only
    internal Mariners players did (via build_seattle()'s team-level
    fielding table). See battingpitching_scraper.get_fielding()'s
    docstring for the "needs a real test run" caveat on this specific
    page's structure.
    """
    print("[build] fielding leaders ...")
    return get_fielding(season, force_refresh=force_refresh)


def build_overview(season=SEASON, force_refresh=False) -> dict:
    print("[build] MLB overview ...")
    return get_mlb_overview(season, force_refresh=force_refresh)


def build_statcast(roster_keys=None, force_refresh=False) -> dict:
    print("[build] statcast ...")
    # league-wide (still a manual CSV export) -- used only for league
    # baselines and MLB-wide rankings, never for grading Mariners players.
    batters, pitchers = get_statcast()

    # BUG FIX 2026-09-27: Mariners-specific stats used to come from
    # roster-filtering the league-wide file above (get_mariners_batters/
    # get_mariners_pitchers), which (a) depended on the league-wide CSV
    # being current -- it silently wasn't, for months -- and (b) went
    # through the roster-matching layer that's caused most of this
    # project's other real bugs. get_team_statcast() fetches Seattle's
    # rows LIVE and directly from Savant's own team filter, sidestepping
    # both problems. Falls back to the old roster-filter path only if the
    # live fetch comes back completely empty (e.g. this environment can't
    # reach Savant at all).
    sea_bat, sea_pit = get_team_statcast(SEA_TEAM_ID, force_refresh=force_refresh)

    if sea_bat.empty:
        print("  [warn] live Mariners batter statcast empty -- "
              "falling back to roster-filtered league file")
        sea_bat = get_mariners_batters(batters, roster_keys)
    if sea_pit.empty:
        print("  [warn] live Mariners pitcher statcast empty -- "
              "falling back to roster-filtered league file")
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
        "fielding":  build_fielding(season,  force_refresh),
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
    data["roster_keys"] = roster_keys          # <-- add this line

    data["statcast"] = build_statcast(roster_keys, force_refresh=force_refresh)

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

    # debug fielding all_players columns -- new, unverified against a real
    # page yet (see battingpitching_scraper.get_fielding()'s docstring)
    fld = data.get("fielding", {}).get("all_players")
    if fld is not None:
        print(f"\n── fielding all_players cols: {list(fld.columns)}")
        print(f"   rows: {len(fld)}")