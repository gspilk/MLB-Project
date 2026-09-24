"""
free_agent_stats_matcher.py
Takes the REAL, full 2026-27 free agent list (all position groups,
batters and pitchers alike) and cross-references every name against
the real, current-season league-wide leader tables this project
already scrapes -- the same tables recommender.py's own trade-target
search already uses. Turns a list of "who's available" into "who's
available AND how good they actually are right now."

Real logic: position_group tells us whether to look a name up in the
batting leaders or pitching leaders table (Catchers/First Basemen/.../
Designated Hitters = batter; Starting Pitchers/Right-Handed Relievers/
Left-Handed Relievers = pitcher).

Usage:
    python free_agent_stats_matcher.py
"""

import pandas as pd
from data_builder import build_all
from free_agent import get_mlbtr_free_agents

PITCHER_GROUPS = {"Starting Pitchers", "Right-Handed Relievers", "Left-Handed Relievers"}


def _match_name(name: str, leaders: pd.DataFrame) -> pd.Series:
    """Real, direct name match against a leaders table -- exact match
    first (most reliable), falls back to a contains-match for name
    format differences (e.g. suffix handling) between the two real,
    independently-scraped sources."""
    if leaders.empty or "Name" not in leaders.columns:
        return pd.Series()
    exact = leaders[leaders["Name"].astype(str).str.lower() == name.lower()]
    if not exact.empty:
        return exact.iloc[0]
    contains = leaders[leaders["Name"].astype(str).str.contains(
        name.split()[-1], case=False, na=False, regex=False)]
    return contains.iloc[0] if not contains.empty else pd.Series()


def enrich_free_agents(fa_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    # BUG FIX: real, confirmed structure (via a live debug run) is
    # data["batting"]["all_players"] and data["pitching"]["all_players"]
    # -- the guessed "batting_leaders"/"pitching_leaders" keys don't
    # exist at all, which is why every single one of 278 free agents
    # came back unmatched (a silently-empty DataFrame from .get()'s
    # default, not 278 individual real data gaps). all_players is the
    # already-combined AL+NL table (861/1056 real rows), the cleanest
    # single source to match against rather than checking al_players/
    # nl_players separately.
    bat_leaders = data.get("batting", {}).get("all_players", pd.DataFrame())
    pit_leaders = data.get("pitching", {}).get("all_players", pd.DataFrame())

    rows = []
    for _, agent in fa_df.iterrows():
        is_pitcher = agent["position_group"] in PITCHER_GROUPS
        leaders = pit_leaders if is_pitcher else bat_leaders
        match = _match_name(agent["name"], leaders)

        row = {
            "name": agent["name"], "age_2027": agent["age_2027"],
            "position_group": agent["position_group"],
            "option_detail": agent["option_detail"],
            "matched": not match.empty,
            "source": "Free Agent",
        }
        if not match.empty:
            # BUG FIX: a live run showed every single matched row with
            # a blank Team column, even though the real stats (OPS,
            # ERA, etc.) all matched correctly -- meaning name-matching
            # and most columns work fine, but something about "Team"
            # specifically isn't resolving as expected on the real,
            # combined all_players table (possibly a duplicate-column
            # or concat quirk between the AL/NL source tables that
            # wasn't visible in the truncated debug column list). Tries
            # a few real, plausible column name variants rather than
            # assuming one exact name a second time.
            team_val = None
            for candidate in ["Team", "Tm", "Team "]:
                if candidate in match.index and pd.notna(match.get(candidate)):
                    team_val = match.get(candidate)
                    break
            row["Team"] = team_val

            if is_pitcher:
                # WAR confirmed present in the real all_players table
                # (via a live debug run earlier tonight) -- expanded
                # beyond just ERA for a real, fuller depth-chart-style
                # comparison rather than one single number per player
                for stat in ["IP", "ERA", "WHIP", "SO", "BB", "WAR"]:
                    row[stat] = match.get(stat)
            else:
                for stat in ["PA", "OPS", "OBP", "SLG", "HR", "RBI", "BA", "WAR"]:
                    row[stat] = match.get(stat)
        rows.append(row)

    return pd.DataFrame(rows)


def print_matched(df: pd.DataFrame, min_pa_or_ip: float = None):
    matched = df[df["matched"]].copy()
    unmatched_count = len(df) - len(matched)

    print("\n" + "=" * 100)
    print(f"FREE AGENTS WITH REAL, CURRENT-SEASON STATS MATCHED "
          f"({len(matched)} of {len(df)} total)")
    print("=" * 100)

    bat = matched[~matched["position_group"].isin(PITCHER_GROUPS)]
    pit = matched[matched["position_group"].isin(PITCHER_GROUPS)]

    if not bat.empty:
        print("\n-- BATTERS (sorted by OPS) --")
        bat_sorted = bat.copy()
        bat_sorted["OPS"] = pd.to_numeric(bat_sorted["OPS"], errors="coerce")
        cols = ["name", "age_2027", "position_group", "Team", "PA", "OPS", "HR", "RBI", "BA"]
        cols = [c for c in cols if c in bat_sorted.columns]
        print(bat_sorted.sort_values("OPS", ascending=False)[cols].to_string(index=False))

    if not pit.empty:
        print("\n-- PITCHERS (sorted by ERA) --")
        pit_sorted = pit.copy()
        pit_sorted["ERA"] = pd.to_numeric(pit_sorted["ERA"], errors="coerce")
        cols = ["name", "age_2027", "position_group", "Team", "IP", "ERA", "WHIP", "SO"]
        cols = [c for c in cols if c in pit_sorted.columns]
        print(pit_sorted.sort_values("ERA")[cols].to_string(index=False))

    print(f"\n  {unmatched_count} free agents had no real current-season "
          f"match found (likely injured all year, minor-league only, "
          f"or a name-format mismatch between sources -- not "
          f"necessarily unavailable, just unverified here)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("Building league data (uses cache if fresh)...")
    data = build_all(2026)

    print("Getting the real free agent list...")
    fa = get_mlbtr_free_agents()

    if fa.empty:
        print("No free agent data available.")
    else:
        enriched = enrich_free_agents(fa, data)
        print_matched(enriched)