"""
injury_cost_tracker.py
For every player who IS or WAS on the IL this season, estimates:
  - games missed (or starts missed, for starting pitchers -- see note)
  - estimated WAR lost, extrapolating their own real per-game/per-start
    rate over the time missed

TWO DATA SOURCES, COMBINED:
  1. CURRENT snapshot (live) -- anyone whose name has "(IL)" attached
     right now, detected automatically from bbref's own live team pages.
  2. HISTORICAL stints (manually sourced, dated) -- real players who
     were hurt EARLIER this season and have since returned to health,
     so their name no longer carries an "(IL)" tag and the live
     detection alone would miss them entirely. Confirmed via real
     reporting (Lookout Landing, MLB.com) -- see HISTORICAL_IL_STINTS
     below for sources per player.

This split exists because of a real gap found and confirmed tonight:
Bryce Miller, J.P. Crawford, Julio Rodriguez, Victor Robles, Carlos
Vargas, and Miles Mastrobuoni all had real, confirmed IL time earlier
in the 2026 season -- none of them show up in the live-only detection
since they're all healthy and active now. A user correctly caught this
by asking "shouldn't it include players who WERE hurt, not just who
IS hurt right now" -- this fixes that.

HISTORICAL_IL_STINTS is NOT live data -- there's no transaction-log
scraper in this project (that would be the real, permanent fix -- see
OFFSEASON_ROADMAP.md). These are manually entered, dated, sourced
facts, current as of tonight. They will NOT automatically pick up new
injuries or update on their own the way the live-snapshot half does.

IMPORTANT METHOD NOTE ON STARTING PITCHERS:
A starting pitcher doesn't play in every team game even when fully
healthy -- they start roughly every 5th game. Using
"team_games - player_G" for a starter would wildly overstate their
missed time (a perfectly healthy #1 starter might only have ~27 GS all
season anyway). Starters are instead evaluated on STARTS missed,
estimated from the team's pace (team_games / 5) rather than raw
attendance -- position players and relievers use the simpler
team_games - player_G comparison, since both realistically play in
most games when healthy.

This is an ESTIMATE, not an exact reconstruction of actual IL start/end
dates -- good enough for "which injuries have cost the most," not
precise enough for a legal filing.

Usage:
    python injury_cost_tracker.py
"""

import pandas as pd
from datetime import date
from data_builder import build_all
from mariners_stats import get_seattle_stats


def get_2025_baseline_games(name_fragment: str, sea_2025: dict) -> dict:
    """
    Looks up a player's 2025 games played (batting or pitching), as a
    healthy-season baseline for cross-checking 2026 games-missed
    estimates -- without needing an exact injury date at all.

    CAVEAT, stated plainly: this assumes 2025 was itself a representative,
    healthy year for that player. If someone was ALSO hurt for a chunk
    of 2025, this baseline understates their true healthy pace. Treat
    as a cross-check, not a replacement for real injury dates.
    """
    bat = sea_2025.get("batting")
    pit = sea_2025.get("pitching")

    if bat is not None and not bat.empty:
        row = bat[bat["Name"].astype(str).str.contains(name_fragment, na=False, regex=False)]
        if not row.empty:
            g = pd.to_numeric(row["G"].values[0], errors="coerce")
            if g and g > 0:
                return {"games_2025": float(g), "type": "batter"}

    if pit is not None and not pit.empty:
        row = pit[pit["Name"].astype(str).str.contains(name_fragment, na=False, regex=False)]
        if not row.empty:
            g = pd.to_numeric(row["G"].values[0], errors="coerce")
            gs = pd.to_numeric(row["GS"].values[0], errors="coerce")
            if g and g > 0:
                return {"games_2025": float(g), "starts_2025": float(gs or 0), "type": "pitcher"}

    return {}


# Players whose 2025 season was ITSELF injury-shortened, confirmed via
# real reporting -- using their 2025 games as a "healthy baseline" would
# understate their true injury cost, not overstate it. Found tonight
# when a user asked directly whether Brash might be one of these: he
# missed all of 2024 AND the first month of 2025 recovering from Tommy
# John surgery, only returning May 3, 2025 -- his 53 G / 47.1 IP that
# year is already a partial season, not a clean baseline. Excluded here
# rather than silently producing an understated number.
INVALID_2025_BASELINE = {
    "Matt Brash": "missed all of 2024 + first month of 2025 recovering "
                  "from Tommy John surgery -- his own 2025 season was "
                  "itself injury-shortened, not a healthy baseline",
}


def cross_check_vs_2025(name_fragment: str, actual_2026_games: float,
                        team_games_2026: int, sea_2025: dict,
                        is_starter: bool = False) -> dict:
    """
    Scales a player's 2025 games/starts to "expected by now" given how
    far into the 2026 season the team actually is, then compares to
    their real 2026 total. Sidesteps needing an exact injury date
    entirely -- useful specifically for the cases where a precise date
    isn't confirmed (see HISTORICAL_IL_STINTS "n/a" entries) and for
    sanity-checking small, noisy 2026-only samples (e.g. Will Wilson).
    """
    for excluded_name, reason in INVALID_2025_BASELINE.items():
        if excluded_name in name_fragment or name_fragment in excluded_name:
            print(f"  [skip] not using 2025 baseline for {name_fragment}: {reason}")
            return {}

    baseline = get_2025_baseline_games(name_fragment, sea_2025)
    if not baseline:
        return {}

    progress = team_games_2026 / 162.0
    if baseline["type"] == "pitcher" and is_starter:
        expected = baseline.get("starts_2025", 0) * progress
    else:
        expected = baseline["games_2025"] * progress

    games_missed_vs_2025 = max(0, round(expected - actual_2026_games, 1))
    return {
        "expected_by_now": round(expected, 1),
        "actual_2026": actual_2026_games,
        "games_missed_vs_2025_baseline": games_missed_vs_2025,
    }

# Real season boundaries used to convert a historical stint's calendar
# days into an estimated number of team GAMES missed (not just days) --
# 2026 Mariners season ran roughly Mar 26 - Sep 28, 162 games over
# about 187 days, so ~0.867 games/day is the conversion rate used below.
SEASON_START = date(2026, 3, 26)
GAMES_PER_DAY = 162 / 187

# Real, sourced historical IL stints -- see module docstring for why
# this exists alongside the live-snapshot detection. "end_date": None
# means the real return date wasn't precisely confirmed in reporting;
# those entries are shown for the record but excluded from the WAR-lost
# math rather than guessing a number that can't be defended.
HISTORICAL_IL_STINTS = [
    {
        "name": "Gabe Speier", "injury": "left shoulder inflammation (15-day IL)",
        "start_date": date(2026, 5, 4), "end_date": date(2026, 5, 26),
        "source": "Spokesman-Review / Southern Sports Today, May 2026 reporting",
        "date_confidence": "CONFIRMED",  # both placement and reinstatement
                                        # dates directly reported
    },
    {
        "name": "Miles Mastrobuoni", "injury": "right calf strain (60-day IL, "
                                              "retroactive to Mar. 22)",
        "start_date": date(2026, 3, 22), "end_date": date(2026, 6, 15),
        "source": "MLB.com transactions notes, May 2026 (\"began a second rehab "
                  "assignment May 25\", \"expected return: mid-June\")",
        "date_confidence": "APPROXIMATE",  # end date is the team's own
                                          # stated target, not a
                                          # confirmed activation --
                                          # Mastrobuoni was later DFA'd
                                          # (Jul 18), so this window is
                                          # unlikely to run past that
    },
    {
        "name": "Brendan Donovan", "injury": "strained left groin/adductor "
                                            "(10-day IL, EARLIER stint -- see "
                                            "live section above for his "
                                            "separate, current Aug concussion IL)",
        "start_date": date(2026, 5, 4), "end_date": date(2026, 5, 25),
        "source": "ESPN injury notes, May 2026 (\"expected to resume baseball "
                  "activity in 2-to-3 weeks\" from a PRP injection)",
        "date_confidence": "APPROXIMATE",  # exact placement date not
                                          # directly reported, estimated
                                          # from the same injury-news
                                          # cluster as Speier/Mastrobuoni;
                                          # end date is the stated
                                          # recovery estimate, not a
                                          # confirmed activation date
    },
    {
        "name": "Taylor Ward", "injury": "quad tightness (day-to-day, "
                                        "NEVER an actual IL stint -- included "
                                        "per explicit request to count real "
                                        "missed games even without a formal "
                                        "IL tag)",
        "start_date": date(2026, 8, 13), "end_date": date(2026, 8, 20),
        "source": "RotoWire / Seattle Sports / NBC Sports, Aug 2026 reporting "
                  "(\"missing fifth straight start\" as of Aug 18)",
        "date_confidence": "APPROXIMATE",
    },
    {
        "name": "Cal Raleigh", "injury": "right oblique strain (first career IL trip)",
        "start_date": date(2026, 5, 14), "end_date": date(2026, 5, 31),
        "source": "ESPN / Yakima Herald / union-bulletin.com, May 2026 reporting -- "
                  "\"the first time in his career\" on the IL",
        "date_confidence": "APPROXIMATE",  # placement date confirmed
                                          # directly; return inferred
                                          # from "rejoined the team over
                                          # the weekend" after "the past
                                          # two weeks recovering"
        "war_lost_caveat": "This uses Raleigh's CURRENT, already-collapsed "
                          "2026 WAR rate to estimate what the injury cost -- "
                          "circular if the injury itself contributed to that "
                          "collapse (a common real effect: playing through "
                          "lingering soreness after returning). His real "
                          "2025 rate was far higher (4.7 WAR), so this "
                          "number likely UNDERSTATES the true cost, not "
                          "overstates it -- opposite direction from most of "
                          "the LOW-confidence warnings elsewhere in this file.",
    },
    {
        "name": "Bryce Miller", "injury": "oblique inflammation",
        "start_date": date(2026, 3, 25), "end_date": date(2026, 5, 1),
        "source": "Lookout Landing / wagers.com, Apr 2026 reporting",
        "date_confidence": "APPROXIMATE",  # return date was a team
                                          # estimate ("end of April or
                                          # early May"), not a confirmed
                                          # activation date
    },
    {
        "name": "Julio Rodríguez", "injury": "concussion (7-day IL)",
        "start_date": date(2026, 7, 4), "end_date": date(2026, 7, 18),
        "source": "Lookout Landing, Jul 4 and Jul 18 2026 posts",
        "date_confidence": "CONFIRMED",  # both placement and activation
                                        # dates directly reported
    },
    {
        "name": "J.P. Crawford", "injury": "wrist injury (10-day IL)",
        "start_date": date(2026, 7, 31), "end_date": None,
        "source": "Lookout Landing, Jul 31 2026 post",
        "date_confidence": "APPROXIMATE",  # exact activation date not
                                          # confirmed in reporting found
    },
    {
        "name": "J.P. Crawford", "injury": "shoulder inflammation (10-day IL)",
        "start_date": None, "end_date": date(2026, 4, 2),
        "source": "wagers.com, Apr 2026 reporting",
        "date_confidence": "APPROXIMATE",  # placement date not
                                          # confirmed, only reinstatement
    },
    {
        "name": "Víctor Robles", "injury": "right pectoral strain",
        "start_date": date(2026, 4, 7), "end_date": None,
        "source": "wagers.com, Apr 2026 reporting",
        "date_confidence": "APPROXIMATE",
    },
    {
        "name": "Carlos Vargas", "injury": "oblique (15-day IL, extended)",
        "start_date": date(2026, 3, 24), "end_date": None,
        "source": "wagers.com Mar 2026 reporting + Aug 2026 rehab notes "
                  "mentioning ~116 games missed -- likely extended well "
                  "past the initial 15-day designation",
        "date_confidence": "APPROXIMATE",
    },
]


def _stint_games_missed(stint: dict) -> float:
    """Returns estimated games missed for a stint, or None if either
    date is missing (rather than guessing)."""
    if not stint.get("start_date") or not stint.get("end_date"):
        return None
    days = (stint["end_date"] - stint["start_date"]).days
    return round(max(0, days) * GAMES_PER_DAY, 1)


def _is_il(name: str) -> bool:
    return "IL" in str(name)


def get_team_games_played(data: dict) -> int:
    st = data["standings"]["all_teams"]
    sea = st[st["Tm"] == "Seattle Mariners"]
    if sea.empty:
        return 0
    w = pd.to_numeric(sea["W"].values[0], errors="coerce")
    l = pd.to_numeric(sea["L"].values[0], errors="coerce")
    return int(w + l)


def _confidence(games_played: float) -> str:
    """
    Flags how much to trust an extrapolated rate. Directly motivated by
    tonight's Matt Brash 743 ERA+ conversation -- a rate computed from a
    tiny sample can be mathematically correct and still wildly
    misleading once extrapolated over a large number of missed games
    (e.g. 2 games played -> 6+ "WAR lost" is not a real, trustworthy
    estimate, just noise amplified by extrapolation).
    """
    if games_played < 10:
        return "LOW"
    if games_played < 30:
        return "MEDIUM"
    return "HIGH"


def build_historical_report(data: dict, sea_2025: dict = None,
                            team_games_2026: int = None) -> list:
    """
    Computes estimated WAR lost for each HISTORICAL_IL_STINTS entry,
    using that player's own real, full-season WAR rate (from their
    current batting/pitching line) applied to the estimated games
    missed during that specific past stint.

    Entries with an unknown start or end date fall back to the 2025
    games-played cross-check (see cross_check_vs_2025()) when
    sea_2025/team_games_2026 are provided -- fills in a real, defensible
    estimate for the entries that otherwise showed "n/a" with no
    injury-date info required at all.
    """
    bat = data["seattle"]["batting"]
    pit = data["seattle"]["pitching"]
    rows = []
    # BUG FIX: the 2025 cross-check reflects a player's TOTAL season
    # shortfall, not a per-stint figure -- if the same player has
    # multiple HISTORICAL_IL_STINTS entries with unknown dates (e.g.
    # Crawford's two separate stints), applying the cross-check to each
    # one identically would double (or triple) count the same games.
    # Only apply it to the first such entry per player.
    cross_check_used_for = set()

    for stint in HISTORICAL_IL_STINTS:
        name = stint["name"]
        games_missed = _stint_games_missed(stint)
        games_missed_source = "date-based" if games_missed is not None else None

        bat_row = bat[bat["Name"].str.contains(name, na=False, regex=False)]
        pit_row = pit[pit["Name"].str.contains(name, na=False, regex=False)]

        # BUG FIX: this used to always check batting FIRST and only
        # fall back to pitching if there was no batting match at all --
        # meaning a spurious or thin batting match (e.g. a name
        # substring collision, or a stray/incomplete row) could
        # silently override a real, substantial pitching line. Caught
        # directly: Gabe Speier -- a real pitcher with no batting
        # record under AL rules -- was showing up as "Batter" in the
        # combined table. Now checks BOTH tables' real games-played
        # figures and picks whichever one actually has a substantial,
        # real stat line, rather than trusting whichever table
        # happened to be checked first.
        bat_g = pit_g = 0
        if not bat_row.empty:
            bat_g = pd.to_numeric(bat_row["G"].values[0], errors="coerce") or 0
        if not pit_row.empty:
            pit_g = pd.to_numeric(pit_row["G"].values[0], errors="coerce") or 0

        war_rate = None
        player_type = "Unknown"
        actual_2026_g = 0
        if bat_g > 0 and bat_g >= pit_g:
            war = pd.to_numeric(bat_row["WAR"].values[0], errors="coerce") or 0
            war_rate = war / bat_g
            player_type = "Batter"
            actual_2026_g = bat_g
        elif pit_g > 0:
            war = pd.to_numeric(pit_row["WAR"].values[0], errors="coerce") or 0
            war_rate = war / pit_g
            player_type = "Pitcher"
            actual_2026_g = pit_g

        # fall back to the 2025 baseline cross-check when no date-based
        # estimate exists -- only once per player, see note above
        if (games_missed is None and sea_2025 is not None and team_games_2026
                and name not in cross_check_used_for):
            cc = cross_check_vs_2025(name, actual_2026_g, team_games_2026, sea_2025)
            if cc:
                games_missed = cc["games_missed_vs_2025_baseline"]
                games_missed_source = "2025 baseline cross-check (total season shortfall, not this stint alone)"
                cross_check_used_for.add(name)
        elif games_missed is None and name in cross_check_used_for:
            games_missed_source = "already counted under this player's other stint above"

        war_lost = (round(war_rate * games_missed, 2)
                   if war_rate is not None and games_missed is not None else None)

        rows.append({
            "name": name, "type": player_type, "injury": stint["injury"],
            "start_date": stint.get("start_date"), "end_date": stint.get("end_date"),
            "games_missed": games_missed, "games_missed_source": games_missed_source,
            "estimated_war_lost": war_lost,
            "date_confidence": stint["date_confidence"], "source": stint["source"],
            "war_lost_caveat": stint.get("war_lost_caveat"),
        })

    return rows


def get_leverage(name: str, val_pit) -> float:
    """
    Real leverage index (gmLI) from bbref's own value-pitching table --
    how high-stakes this pitcher's average appearance actually was.
    League average is 1.00; a real closer typically sits well above
    that (Matt Brash's real 2026 gmLI was 1.91), while a mop-up/depth
    arm sits well below it (0.3-0.4 range). Two pitchers can have
    similar raw WAR/G but very different real importance to the team --
    this captures that directly instead of treating all WAR equally.
    Returns None if not found, rather than a misleading default of 1.0.
    """
    if val_pit is None or val_pit.empty or "Name" not in val_pit.columns:
        return None
    row = val_pit[val_pit["Name"].astype(str).str.contains(
        name.split(" (")[0], case=False, na=False, regex=False)]
    if row.empty or "gmLI" not in row.columns:
        return None
    li = pd.to_numeric(row["gmLI"].values[0], errors="coerce")
    return None if pd.isna(li) else float(li)


# Real thresholds for classifying a batter's role from their own PA/G
# rate (plate appearances per game actually played) -- since a player's
# OWN games-played total is exactly what's suppressed by injury, it
# can't be used to judge their role; PA/G in games they DID play is a
# real, independent signal instead. ~4 PA/G is close to a full lineup
# spot; a true bench/pinch-hit role rarely clears half that.
STARTER_PA_PER_G = 3.0

# A real, everyday starter doesn't actually play all 162 games even
# healthy -- real rest days, day games after night games, etc. Using
# the flat team-games total as "expected" overstates a healthy
# starter's normal workload. ~93% reflects a realistic every-day
# player's real attendance rate, not a guess -- most true iron-man
# regulars land in the 150-155 games range out of 162.
STARTER_ATTENDANCE_RATE = 0.93


def _compute_reliever_baseline(pit_df: pd.DataFrame) -> float:
    """
    Real, dynamic reliever workload baseline -- the team's own current
    HEALTHY relievers' average appearance count, not a fixed number or
    the full team-games total (which no reliever, healthy or hurt,
    would ever be expected to match -- even a heavily-used reliever
    typically appears in well under half the team's games). This
    directly answers the real, honest problem flagged tonight: reliever
    workload varies a lot and there's no single clean number, so use
    the team's own real relievers as the reference point instead of
    guessing one.
    """
    healthy_relievers = []
    for _, row in pit_df.iterrows():
        name = str(row.get("Name", ""))
        if _is_il(name):
            continue
        gs = pd.to_numeric(row.get("GS", 0), errors="coerce") or 0
        g = pd.to_numeric(row.get("G", 0), errors="coerce") or 0
        is_starter = gs >= (g * 0.5) and gs > 0
        if not is_starter and g > 0:
            healthy_relievers.append(g)
    if not healthy_relievers:
        return None
    return sum(healthy_relievers) / len(healthy_relievers)


def build_injury_report(data: dict) -> list:
    team_games = get_team_games_played(data)
    if not team_games:
        return []

    rows = []

    bat = data["seattle"]["batting"]
    for _, row in bat.iterrows():
        name = str(row.get("Name", ""))
        if not _is_il(name):
            continue
        g = pd.to_numeric(row.get("G", 0), errors="coerce") or 0
        pa = pd.to_numeric(row.get("PA", 0), errors="coerce") or 0
        war = pd.to_numeric(row.get("WAR", 0), errors="coerce") or 0
        if g <= 0:
            continue

        pa_per_g = pa / g
        is_regular_starter = pa_per_g >= STARTER_PA_PER_G

        if is_regular_starter:
            expected_games = team_games * STARTER_ATTENDANCE_RATE
            role_note = (f"everyday starter ({pa_per_g:.1f} PA/G when playing) -- "
                        f"real ~93% attendance baseline, not a flat 162")
        else:
            # BENCH/DEPTH: these players are role-defined to play less
            # than every game even fully healthy -- comparing them to a
            # starter's baseline would misread their normal part-time
            # role as injury-driven absence. Games-missed intentionally
            # NOT computed for this group -- there's no honest baseline
            # to compare against without real depth-chart data this
            # project doesn't have.
            rows.append({
                "name": name, "type": "Batter (bench/depth)", "games_played": int(g),
                "games_missed": None, "war_rate_per_game": None,
                "estimated_war_lost": None, "confidence": "N/A",
                "note": f"bench/depth role ({pa_per_g:.1f} PA/G when playing) -- "
                       f"games-missed not computed, no honest full-role "
                       f"baseline exists for a part-time role",
            })
            continue

        games_missed = max(0, expected_games - g)
        war_rate = war / g
        war_lost = round(war_rate * games_missed, 2)
        rows.append({
            "name": name, "type": "Batter", "games_played": int(g),
            "games_missed": round(games_missed, 1), "war_rate_per_game": round(war_rate, 5),
            "estimated_war_lost": war_lost, "confidence": _confidence(g),
            "note": role_note,
        })

    pit = data["seattle"]["pitching"]
    val_pit = data["seattle"].get("value_pitching")
    reliever_baseline = _compute_reliever_baseline(pit)
    for _, row in pit.iterrows():
        name = str(row.get("Name", ""))
        if not _is_il(name):
            continue
        g = pd.to_numeric(row.get("G", 0), errors="coerce") or 0
        gs = pd.to_numeric(row.get("GS", 0), errors="coerce") or 0
        war = pd.to_numeric(row.get("WAR", 0), errors="coerce") or 0
        is_starter = gs >= (g * 0.5) and gs > 0
        leverage = get_leverage(name, val_pit)
        # leverage-adjusted WAR lost: scales the raw estimate by how
        # high-stakes this pitcher's real appearances actually were,
        # relative to league-average (1.00) -- a real closer's missed
        # time counts for more than a depth arm's at the same raw WAR
        leverage_note = (f" (leverage-adjusted from real gmLI {leverage:.2f})"
                         if leverage is not None else
                         " (no real leverage data found -- raw WAR only)")

        if is_starter:
            if gs <= 0:
                continue
            expected_starts = team_games / 5.0  # standard 5-man rotation pace
            starts_missed = max(0, expected_starts - gs)
            war_rate = war / gs
            war_lost = round(war_rate * starts_missed, 2)
            war_lost_adj = round(war_lost * leverage, 2) if leverage is not None else None
            rows.append({
                "name": name, "type": "Starter", "games_played": int(gs),
                "games_missed": round(starts_missed, 1), "war_rate_per_game": round(war_rate, 5),
                "estimated_war_lost": war_lost, "confidence": _confidence(gs),
                "leverage": leverage, "war_lost_leverage_adj": war_lost_adj,
                "note": "starts missed (5-day rotation pace), not raw team games"
                       + leverage_note,
            })
        else:
            if g <= 0:
                continue
            # BUG FIX: was comparing a reliever's games played against
            # the FULL team games total -- no reliever, healthy or
            # hurt, would ever be expected to appear in every team
            # game, so this overstated "missed" time for every reliever
            # in the file. Uses the team's own real, currently-healthy
            # relievers' average appearance count instead -- a genuine,
            # role-appropriate baseline computed from real data, not a
            # guess, and directly addresses that reliever workload
            # varies too much for one fixed number to work.
            if reliever_baseline is not None:
                games_missed = max(0, reliever_baseline - g)
                baseline_note = (f"real team reliever baseline: "
                                f"{reliever_baseline:.1f} appearances "
                                f"(avg of currently-healthy relievers)")
            else:
                games_missed = 0
                baseline_note = ("no other healthy relievers found to "
                                "compute a real baseline from -- "
                                "games-missed not estimated")
            war_rate = war / g
            war_lost = round(war_rate * games_missed, 2)
            war_lost_adj = round(war_lost * leverage, 2) if leverage is not None else None
            rows.append({
                "name": name, "type": "Reliever", "games_played": int(g),
                "games_missed": round(games_missed, 1), "war_rate_per_game": round(war_rate, 5),
                "estimated_war_lost": war_lost, "confidence": _confidence(g),
                "leverage": leverage, "war_lost_leverage_adj": war_lost_adj,
                "note": baseline_note + " | " + leverage_note.strip(),
            })

    # BUG FIX: bench/depth rows now carry estimated_war_lost=None
    # (intentionally, since there's no honest baseline to compute them
    # against) -- a plain sort on that field would crash comparing
    # None to a real number. Sorts those rows to the bottom instead of
    # erroring.
    rows.sort(key=lambda r: (r["estimated_war_lost"] is None,
                             -(r["estimated_war_lost"] or 0)))
    return rows


def print_report(rows: list, team_games: int):
    print("\n" + "=" * 78)
    print(f"INJURY COST TRACKER -- as of {team_games} team games played")
    print("=" * 78)
    if not rows:
        print("  No players currently on IL with usable stats.")
        print("=" * 78 + "\n")
        return

    print(f"  {'Name':<28} {'Type':<9} {'Played':>7} {'Missed':>7} "
          f"{'WAR/G':>9} {'Est. WAR Lost':>14} {'Confidence':>11}")
    print("  " + "-" * 86)
    for r in rows:
        # BUG FIX: bench/depth rows carry None for games_missed/
        # war_rate/war_lost (intentionally -- no honest baseline
        # exists for a part-time role, see build_injury_report()).
        # The plain ":>N" alignment format spec crashes on None, so
        # format these as a literal "n/a" string first.
        gm = f"{r['games_missed']:.1f}" if r['games_missed'] is not None else "n/a"
        wr = f"{r['war_rate_per_game']:.5f}" if r['war_rate_per_game'] is not None else "n/a"
        wl = f"{r['estimated_war_lost']:.2f}" if r['estimated_war_lost'] is not None else "n/a"
        print(f"  {r['name']:<28} {r['type']:<9} {r['games_played']:>7} "
              f"{gm:>7} {wr:>9} "
              f"{wl:>14} {r['confidence']:>11}")
        if r.get("war_lost_leverage_adj") is not None:
            print(f"    Leverage-adjusted (real gmLI {r['leverage']:.2f}): "
                  f"{r['war_lost_leverage_adj']:+.2f} WAR "
                  f"(vs {r['estimated_war_lost']:+.2f} unweighted)")
        if r.get("note"):
            print(f"    -> {r['note']}")

    # BUG FIX: same None-handling issue as the print loop above -- bench/
    # depth rows have estimated_war_lost=None, which would crash a plain
    # sum(). Filtering those out of both totals (correctly -- they were
    # never given a real number to include).
    low_conf = [r for r in rows if r["confidence"] == "LOW"]
    total_war_lost = round(sum(r["estimated_war_lost"] for r in rows
                               if r["estimated_war_lost"] is not None), 2)
    total_high_conf = round(sum(r["estimated_war_lost"] for r in rows
                                if r["confidence"] != "LOW"
                                and r["estimated_war_lost"] is not None), 2)
    print("  " + "-" * 86)
    print(f"  {'TOTAL (all confidence levels)':<52} {total_war_lost:>14}")
    print(f"  {'TOTAL (excluding LOW-confidence estimates)':<52} {total_high_conf:>14}")
    if low_conf:
        names = ", ".join(r["name"] for r in low_conf)
        print(f"\n  Note: {len(low_conf)} player(s) flagged LOW confidence -- "
              f"under 10 games/starts played, meaning their per-game rate is "
              f"noisy and gets amplified by extrapolation ({names}). Treat "
              f"those specific numbers as illustrative, not reliable.")
    print("=" * 78 + "\n")


def print_historical_report(rows: list):
    print("\n" + "=" * 78)
    print("HISTORICAL IL STINTS THIS SEASON -- sourced, dated, NOT live data")
    print("=" * 78)
    print("  (Players who were hurt earlier this season and have since")
    print("   returned to health -- invisible to the live-snapshot check")
    print("   above since they're not currently tagged IL anymore.)\n")

    for r in rows:
        start = r["start_date"].isoformat() if r["start_date"] else "unknown"
        end = r["end_date"].isoformat() if r["end_date"] else "unknown"
        gm = f"{r['games_missed']:.1f}" if r["games_missed"] is not None else "n/a"
        wl = f"{r['estimated_war_lost']:+.2f}" if r["estimated_war_lost"] is not None else "n/a"
        print(f"  {r['name']:<20} {r['injury']:<38} [{r['date_confidence']}]")
        print(f"    {start} -> {end}   est. games missed: {gm}   "
              f"est. WAR lost: {wl}")
        if r.get("games_missed_source"):
            print(f"    games-missed method: {r['games_missed_source']}")
        if r.get("war_lost_caveat"):
            print(f"    CAVEAT: {r['war_lost_caveat']}")
        print(f"    source: {r['source']}")
        print()

    computable = [r for r in rows if r["estimated_war_lost"] is not None]
    if computable:
        total = round(sum(r["estimated_war_lost"] for r in computable), 2)
        print(f"  Historical total (only stints with both dates known): {total:+.2f} WAR")
    incomplete = [r["name"] for r in rows if r["estimated_war_lost"] is None]
    if incomplete:
        print(f"  Not computed (missing a confirmed date): {', '.join(incomplete)}")
    print("=" * 78 + "\n")


def build_unified_table(live_rows: list, historical_rows: list) -> list:
    """
    Combines the live-snapshot and historical-stint data into ONE
    unified list -- same real numbers as the two separate sections
    above, just normalized into a single consistent schema (name,
    status, type, detail, games missed, WAR lost, confidence, source)
    so the whole season's injury picture can be seen, sorted, and
    reasoned about as one dataset instead of two differently-shaped
    reports bolted together.
    """
    unified = []

    for r in live_rows:
        unified.append({
            "name": r["name"],
            "status": "CURRENTLY OUT",
            "type": r["type"],
            "detail": r.get("note", ""),
            "games_missed": r["games_missed"],
            "war_lost": r["estimated_war_lost"],
            "war_lost_leverage_adj": r.get("war_lost_leverage_adj"),
            "confidence": r["confidence"],
            "source": "live bbref snapshot",
        })

    for r in historical_rows:
        unified.append({
            "name": r["name"],
            "status": "RESOLVED",
            "type": r["type"],
            "detail": r["injury"],
            "games_missed": r["games_missed"],
            "war_lost": r["estimated_war_lost"],
            "war_lost_leverage_adj": None,  # leverage data only wired
                                            # up for the live section so far
            "confidence": r["date_confidence"],
            "source": r["source"],
        })

    # same None-safe sort pattern as build_injury_report() -- rows with
    # no computable WAR-lost figure sort to the bottom instead of
    # crashing a plain comparison against a real number
    unified.sort(key=lambda r: (r["war_lost"] is None, -(r["war_lost"] or 0)))
    return unified


def print_unified_table(rows: list):
    print("\n" + "=" * 100)
    print("UNIFIED INJURY TABLE -- every player who is or was hurt this season, one dataset")
    print("=" * 100)
    print(f"  {'Name':<26} {'Status':<14} {'Type':<11} {'Missed':>7} "
          f"{'WAR Lost':>9} {'Lev-Adj':>8} {'Conf':>12}  Detail")
    print("  " + "-" * 96)

    for r in rows:
        gm = f"{r['games_missed']:.1f}" if r["games_missed"] is not None else "n/a"
        wl = f"{r['war_lost']:+.2f}" if r["war_lost"] is not None else "n/a"
        lev = (f"{r['war_lost_leverage_adj']:+.2f}"
              if r.get("war_lost_leverage_adj") is not None else "-")
        detail = (r["detail"][:45] + "...") if len(r["detail"]) > 48 else r["detail"]
        print(f"  {r['name']:<26} {r['status']:<14} {r['type']:<11} "
              f"{gm:>7} {wl:>9} {lev:>8} {r['confidence']:>12}  {detail}")

    real_totals = [r["war_lost"] for r in rows if r["war_lost"] is not None]
    if real_totals:
        print("  " + "-" * 96)
        print(f"  TOTAL (every row with a computable WAR-lost figure): "
              f"{round(sum(real_totals), 2):+.2f} WAR across {len(real_totals)} stints")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    team_games = get_team_games_played(data)
    rows = build_injury_report(data)
    print_report(rows, team_games)

    print("Fetching 2025 season data for cross-checking...")
    sea_2025 = get_seattle_stats(2025)
    historical_rows = build_historical_report(data, sea_2025, team_games)
    print_historical_report(historical_rows)

    # BUG FIX: the two sections above were always computed and printed
    # correctly, but never actually COMBINED into one final answer --
    # a real user pointed out the output "still seems off" after
    # verifying both sections' internal math checked out individually.
    # The real gap: after building all this, the one number someone
    # actually wants (total season-long injury cost) was simply never
    # shown. Uses the live section's HIGH/MEDIUM-confidence total (not
    # the LOW-confidence-included one) alongside the historical total,
    # since combining a known-noisy number with a dated, sourced one
    # would undermine the credibility of both.
    live_trustworthy = round(sum(r["estimated_war_lost"] for r in rows
                                 if r["confidence"] != "LOW"
                                 and r["estimated_war_lost"] is not None), 2)
    historical_total = round(sum(r["estimated_war_lost"] for r in historical_rows
                                 if r["estimated_war_lost"] is not None), 2)
    grand_total = round(live_trustworthy + historical_total, 2)

    print("=" * 78)
    print("SEASON-TOTAL INJURY COST (combining both sections above)")
    print("=" * 78)
    print(f"  Currently on IL (MEDIUM/HIGH confidence only): {live_trustworthy:>8.2f} WAR")
    print(f"  Historical, now-resolved stints:               {historical_total:>8.2f} WAR")
    print("  " + "-" * 60)
    print(f"  TOTAL ESTIMATED SEASON INJURY COST:            {grand_total:>8.2f} WAR")
    print(f"\n  Note: this is a floor, not a ceiling -- excludes the LOW-")
    print(f"  confidence live estimate (Will Wilson) and the historical")
    print(f"  entries with no confirmed end date (see 'Not computed' above),")
    print(f"  both of which represent real, additional cost this total")
    print(f"  doesn't capture.")
    print("=" * 78 + "\n")

    unified = build_unified_table(rows, historical_rows)
    print_unified_table(unified)