"""
data_builder.py
Assembles all scrapers into clean datasets.
Everything else imports from here — never import scrapers directly.

Usage:
    from data_builder import build_all
    data = build_all(2026)
    print(data["seattle"]["batting"])
    print(data["standings"]["division"])
"""

import os
import time
import pandas as pd

# ── imports — match YOUR actual filenames ─────────────────────────────────────
from standings_scraper          import get_standings, get_mariners_context, get_all_teams
from mariners_schedule_scraper  import get_schedule
from mariners_stats             import get_seattle_stats
from battingpitching_scraper    import get_batting, get_pitching
from mlb_team_stats             import get_mlb_overview
from statcast_scraper           import get_statcast, get_mariners_batters, get_mariners_pitchers, get_luck_analysis

SEASON = 2026


# ── individual build functions ────────────────────────────────────────────────

def build_standings(season=SEASON, force_refresh=False) -> dict:
    """
    Returns:
        division        dict of 6 division DataFrames
        expanded        all 30 teams expanded standings
        mariners        Mariners context dict (rank, WC gap, etc)
        all_teams       all 30 teams sorted by W-L%
    """
    print("[build] standings ...")
    division, expanded = get_standings(season, force_refresh=force_refresh)
    return {
        "division":   division,
        "expanded":   expanded,
        "mariners":   get_mariners_context(division, expanded),
        "all_teams":  get_all_teams(division),
    }


def build_schedule(season=SEASON, force_refresh=False) -> dict:
    """
    Returns:
        completed       games played so far
        next7           next 7 games checklist
        remaining       remaining 104 games
    """
    print("[build] schedule ...")
    completed, next7, remaining = get_schedule(season, force_refresh=force_refresh)
    return {
        "completed":  completed,
        "next7":      next7,
        "remaining":  remaining,
    }


def build_seattle(season=SEASON, force_refresh=False) -> dict:
    """
    Returns:
        batting         standard batting per player
        pitching        standard pitching per player
        fielding        fielding stats per player
        value_batting   WAR + value per batter
        value_pitching  WAR + value per pitcher
        roster          40-man roster
    """
    print("[build] seattle stats ...")
    return get_seattle_stats(season, force_refresh=force_refresh)


def build_batting(season=SEASON, force_refresh=False) -> dict:
    """
    Returns:
        al_teams        AL team batting totals
        al_players      AL individual batting
        nl_teams        NL team batting totals
        nl_players      NL individual batting
        all_players     combined sorted by OPS
        all_teams       combined sorted by R
    """
    print("[build] batting leaders ...")
    return get_batting(season, force_refresh=force_refresh)


def build_pitching(season=SEASON, force_refresh=False) -> dict:
    """
    Returns:
        al_teams        AL team pitching totals
        al_players      AL individual pitching
        nl_teams        NL team pitching totals
        nl_players      NL individual pitching
        all_players     combined sorted by ERA
        all_teams       combined sorted by ERA
    """
    print("[build] pitching leaders ...")
    return get_pitching(season, force_refresh=force_refresh)


def build_overview(season=SEASON, force_refresh=False) -> dict:
    """
    Returns:
        batting         all 30 teams batting
        pitching        all 30 teams pitching
        fielding        all 30 teams fielding
        war_positions   WAR by position all 30 teams
    """
    print("[build] MLB overview ...")
    return get_mlb_overview(season, force_refresh=force_refresh)


def build_statcast() -> dict:
    """
    Returns:
        batters         full MLB batter Statcast
        pitchers        full MLB pitcher Statcast
        sea_batters     Mariners batters only
        sea_pitchers    Mariners pitchers only
        bat_luck        batter luck analysis
        pit_luck        pitcher luck analysis
    """
    print("[build] statcast ...")
    batters, pitchers = get_statcast()
    sea_bat  = get_mariners_batters(batters)
    sea_pit  = get_mariners_pitchers(pitchers)
    return {
        "batters":      batters,
        "pitchers":     pitchers,
        "sea_batters":  sea_bat,
        "sea_pitchers": sea_pit,
        "bat_luck":     get_luck_analysis(sea_bat,  is_pitcher=False),
        "pit_luck":     get_luck_analysis(sea_pit,  is_pitcher=True),
    }


# ── main build_all function ───────────────────────────────────────────────────

def build_all(season=SEASON, force_refresh=False) -> dict:
    """
    Runs all scrapers and returns complete dataset.
    Pass force_refresh=True to bypass all caches.

    Returns dict with keys:
        standings
        schedule
        seattle
        batting
        pitching
        overview
        statcast
    """
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
        "statcast":  build_statcast(),
    }

    elapsed = round(time.time() - start, 1)
    print(f"\n{'='*50}")
    print(f"All data built in {elapsed}s")
    print(f"{'='*50}\n")

    return data


# ── convenience accessors ─────────────────────────────────────────────────────

def get_mariners_batting(data: dict) -> pd.DataFrame:
    """Quick access to Mariners batting from full dataset."""
    return data.get("seattle", {}).get("batting", pd.DataFrame())


def get_mariners_pitching(data: dict) -> pd.DataFrame:
    """Quick access to Mariners pitching from full dataset."""
    return data.get("seattle", {}).get("pitching", pd.DataFrame())


def get_mariners_statcast_bat(data: dict) -> pd.DataFrame:
    """Quick access to Mariners Statcast batting."""
    return data.get("statcast", {}).get("sea_batters", pd.DataFrame())


def get_mariners_statcast_pit(data: dict) -> pd.DataFrame:
    """Quick access to Mariners Statcast pitching."""
    return data.get("statcast", {}).get("sea_pitchers", pd.DataFrame())


def get_schedule_next7(data: dict) -> pd.DataFrame:
    """Quick access to next 7 games."""
    return data.get("schedule", {}).get("next7", pd.DataFrame())


def get_al_west(data: dict) -> pd.DataFrame:
    """Quick access to AL West standings."""
    division = data.get("standings", {}).get("division", {})
    return division.get("AL_West", pd.DataFrame())


def get_mariners_record(data: dict) -> dict:
    """Quick access to Mariners record and context."""
    return data.get("standings", {}).get("mariners", {})


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = build_all(2026)

    print("\n── Mariners record ──")
    ctx = get_mariners_record(data)
    print(f"  MLB rank:    {ctx.get('mlb_rank')} of 30")
    print(f"  Division:    {ctx.get('div_rank')} of 5 AL West")
    print(f"  WC gap:      {ctx.get('wc_gap')}")
    print(f"  WC in reach: {ctx.get('wc_in_reach')}")

    print("\n── AL West ──")
    al_west = get_al_west(data)
    if not al_west.empty:
        cols = [c for c in ["Tm","W","L","W-L%","GB"] if c in al_west.columns]
        print(al_west[cols].to_string(index=False))

    print("\n── next 7 games ──")
    next7 = get_schedule_next7(data)
    if not next7.empty:
        print(next7[["Gm#","date","home_away","Opp"]].to_string(index=False))

    print("\n── Mariners batting (top 5 OPS) ──")
    bat = get_mariners_batting(data)
    if not bat.empty and "OPS" in bat.columns:
        cols = [c for c in ["Name","PA","BA","OBP","SLG","OPS","HR","RBI"]
                if c in bat.columns]
        print(bat[cols].dropna(subset=["OPS"])
              .sort_values("OPS", ascending=False)
              .head(5).to_string(index=False))

    print("\n── Mariners Statcast batting ──")
    sc_bat = get_mariners_statcast_bat(data)
    if not sc_bat.empty:
        cols = [c for c in ["Name","xwOBA","Barrel%","HardHit%","EV50"]
                if c in sc_bat.columns]
        print(sc_bat[cols].sort_values("xwOBA", ascending=False)
              .to_string(index=False))

    print("\n── Mariners pitching luck ──")
    pit_luck = data.get("statcast", {}).get("pit_luck", pd.DataFrame())
    if not pit_luck.empty:
        print(pit_luck[["Name","verdict","luck_gap"]].to_string(index=False))