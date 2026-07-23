"""
roster.py
Single source of truth for "who is currently on the Mariners roster?"

Previously this was answered by three separately hardcoded name sets
(recommender.py's MARINERS_ROSTER, statcast_scraper.py's MARINERS_BATTERS /
MARINERS_PITCHERS, and debug_tables.py's MARINERS) that had to be updated
by hand after every trade, call-up, or DFA -- and could silently drift out
of sync with each other. This module instead reads the 40-man roster table
that mariners_stats.py already scrapes from bbref
(data["seattle"]["roster"]), so roster membership updates automatically
every time build_all() refreshes.

Usage:
    from roster import get_roster_last_names
    roster = get_roster_last_names(data)   # set of normalized last names
    "munoz" in roster                       # True
"""

import pandas as pd

from name_matching import last_name_only, key_from_first_last


def _roster_names(data: dict):
    roster_df = data.get("seattle", {}).get("roster")
    if roster_df is None or not isinstance(roster_df, pd.DataFrame) or roster_df.empty:
        return []
    if "Name" not in roster_df.columns:
        return []
    return [n for n in roster_df["Name"].dropna().tolist() if str(n).strip()]


def get_roster_last_names(data: dict) -> set:
    """
    Returns a set of normalized LAST NAMES ONLY for everyone on the current
    40-man roster, pulled from data["seattle"]["roster"].

    Use this only where over-matching is the SAFE failure mode -- e.g.
    excluding current Mariners from a league-wide trade-target search,
    where accidentally excluding a same-surnamed non-Mariner just means
    one fewer candidate shown, not corrupted data. For anything that
    identifies "is this row actually a Mariner" (building the team's own
    stat lines), use get_roster_keys() instead -- the roster has real
    last-name collisions with other teams (e.g. multiple "Wilson"s and
    "Ortiz"/"Vargas" surnames both on and off the roster), so last-name-only
    matching WILL pull in wrong players there.

    Returns an empty set (not an exception) if the roster table is
    missing or failed to scrape -- callers should treat an empty set as
    "roster unknown" and fall back gracefully rather than assume no one
    is on the roster.
    """
    return {last_name_only(n) for n in _roster_names(data)}


def get_roster_keys(data: dict) -> set:
    """
    Returns a set of precise 'lastname_firstinitial' keys for everyone on
    the current 40-man roster (e.g. {'rodriguez_j', 'munoz_a', ...}).

    Use this to identify which rows in another data source (e.g. Statcast)
    actually belong to the Mariners -- it's precise enough to avoid
    cross-team last-name collisions, which get_roster_last_names() is not.
    """
    return {key_from_first_last(n) for n in _roster_names(data)}


def is_on_roster(name_or_key: str, roster_last_names: set) -> bool:
    """
    Convenience check. Accepts either a raw name (any format) or an
    already-built 'lastname_firstinitial' match key.
    """
    if not roster_last_names:
        return False
    if "_" in name_or_key and " " not in name_or_key and "," not in name_or_key:
        # looks like an already-built match key, e.g. "munoz_a"
        last = name_or_key.split("_")[0]
    else:
        last = last_name_only(name_or_key)
    return last in roster_last_names