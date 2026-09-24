"""
minor_league_shortlist.py

Filters the real org_batting/org_pitching tables (133 batters, 147 pitchers
across the whole system) down to the players actually worth tracking as
near-term help: real playing time at AA or AAA -- the two levels close
enough to the majors to matter for a "who's on the way" read.

Rookie ball / A-ball / A+ names are dropped here on purpose, not because
they're bad prospects, but because they're multiple years out and mostly
just add noise to a "what could help the 2026/2027 Mariners" question.

Thresholds (adjust freely -- these are meant as a reasonable floor, not a
scouting rule):
    batters:  played some AA or AAA time this season AND 200+ PA
    pitchers: pitched some AA or AAA time this season AND 40+ IP
"""

from minor_league_stats import get_minor_league_data

MIN_PA = 200
MIN_IP = 40.0
NEAR_MLB_LEVELS = {"AA", "AAA"}


def get_active_mariners_keys(data: dict) -> set:
    """
    Real fix, round 3: get_roster_keys() (round 2's fix) reads the real
    40-man roster table, but 40-man membership and ACTIVE roster status
    are different things -- being optioned to AAA or placed on the IL
    doesn't remove someone from the 40-man. Confirmed directly against the
    real live page: Wisdom/Bliss/Brennen Davis(60-day IL) are all real
    40-man members with no active-roster mark, so get_roster_keys() was
    still wrongly excluding them even with the right table.

    get_active_roster_keys() (added to roster.py) filters that same real
    table down to rows actually marked active (the page's own "*" in an
    "OnActv" column) -- but only works once seattle_scraper.py's _clean()
    stops numeric-coercing that column into NaN for every row (add
    "OnActv"/"IL" to its skip set). This is the real, correct source.
    """
    from roster import get_active_roster_keys
    return get_active_roster_keys(data)


def exclude_active_mariners(df, active_keys: set):
    """
    active_keys: a set of "lastname_firstinitial" keys, as returned by
    roster.get_roster_keys() -- built with the same key_from_first_last()
    this project already uses elsewhere, so a minor-league name with
    bbref's usual */# markers (Kade Anderson*, J.T. Arruda#) keys the
    same way a plain roster name does without any extra cleanup here.
    """
    if not active_keys or "Name" not in df.columns:
        return df
    from name_matching import key_from_first_last
    mask = ~df["Name"].apply(lambda n: key_from_first_last(n) in active_keys)
    return df[mask].reset_index(drop=True)


def _played_at_near_mlb_level(lev_value) -> bool:
    """
    Real Lev values can be a single level ("AA") or a comma-joined list for
    a player who got promoted mid-season ("AA,AAA"). True if AA or AAA
    appears anywhere in that list -- an exact-token check, not a substring
    check, since "AAA" and "A" would otherwise both match a naive "in" test.
    """
    if lev_value is None or (isinstance(lev_value, float) and lev_value != lev_value):
        return False
    tokens = [t.strip() for t in str(lev_value).split(",")]
    return any(t in NEAR_MLB_LEVELS for t in tokens)


def batting_shortlist(org_batting, active_keys: set = None):
    df = org_batting.copy()
    df = df[df["Lev"].apply(_played_at_near_mlb_level)]
    df = df[df["PA"].fillna(0) >= MIN_PA]
    if active_keys:
        df = exclude_active_mariners(df, active_keys)
    df = df.sort_values("OPS", ascending=False).reset_index(drop=True)
    return df


def pitching_shortlist(org_pitching, active_keys: set = None):
    df = org_pitching.copy()
    df = df[df["Lev"].apply(_played_at_near_mlb_level)]
    df = df[df["IP"].fillna(0) >= MIN_IP]
    if active_keys:
        df = exclude_active_mariners(df, active_keys)
    df = df.sort_values("ERA", ascending=True).reset_index(drop=True)
    return df


def print_shortlist(team: str = "SEA", season: int = 2026, pipeline_data: dict = None):
    """
    pipeline_data: pass this project's real `data` dict (from main.py /
    data_builder.py) to automatically drop anyone already on the current
    40-man roster (Arroyo, Anderson, Montes, etc.) from the shortlist,
    using the real roster.py roster table rather than cumulative stats.
    Running this file standalone without wiring that in still works, it
    just can't tell who's already been called up.
    """
    data = get_minor_league_data(team=team, season=season)
    active_keys = get_active_mariners_keys(pipeline_data) if pipeline_data else set()

    if pipeline_data and not active_keys:
        print("[warn] pipeline_data was passed but roster.get_roster_keys(data) came back "
              "empty -- data['seattle']['roster'] may have failed to scrape; "
              "falling back to no exclusions")

    bat_short = batting_shortlist(data["org_batting"], active_keys)
    pit_short = pitching_shortlist(data["org_pitching"], active_keys)

    print(f"\n=== Batters with real AA/AAA time, {MIN_PA}+ PA ({len(bat_short)}) ===")
    cols = [c for c in ["Name", "Age", "Tm", "Lev", "PA", "OPS", "HR", "SB", "Pos Summary"] if c in bat_short.columns]
    if not bat_short.empty:
        print(bat_short[cols].to_string(index=False))
    else:
        print("  (none met the threshold)")

    print(f"\n=== Pitchers with real AA/AAA time, {MIN_IP}+ IP ({len(pit_short)}) ===")
    cols = [c for c in ["Name", "Age", "Tm", "Lev", "IP", "ERA", "WHIP", "SO9"] if c in pit_short.columns]
    if not pit_short.empty:
        print(pit_short[cols].to_string(index=False))
    else:
        print("  (none met the threshold)")

    return bat_short, pit_short


if __name__ == "__main__":
    # Same real build call free_agent_recs.py uses (data_builder.build_all),
    # so this file also knows who's already on the current MLB roster and
    # can exclude them -- without this, Arroyo/Anderson/Montes show up as
    # if they're still down on the farm.
    from data_builder import build_all

    print("Building league + Mariners data (uses cache if fresh)...")
    data = build_all()

    print_shortlist(pipeline_data=data)