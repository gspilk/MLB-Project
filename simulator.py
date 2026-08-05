"""
simulator.py
Simulates the Mariners rest-of-season record under different scenarios.

Models:
  - IL returns (known dates, hand-maintained in IL_RETURNS)
  - Confirmed deadline acquisitions (Ward, Dominguez -- real, not
    hypothetical; the pre-deadline "what if we trade for X" scenario
    system was retired 2026-08-03 once the deadline passed)
  - Schedule difficulty
  - Luck correction
  - Pythagorean projection

Usage:
    from simulator import run_simulation, print_simulation
    results = run_simulation()
    print_simulation(results)
"""

import pandas as pd
from datetime import date, datetime

# -- current team state --------------------------------------------------------
CURRENT_DATE      = date.today()
SEASON_GAMES      = 162
CURRENT_W         = 47
CURRENT_L         = 47
GAMES_REMAINING   = 68   # 162 - 94 completed
CURRENT_RS_G      = 3.82   # runs scored per game
CURRENT_RA_G      = 3.57   # runs allowed per game
CURRENT_LUCK      = -2.0   # luck stat from standings
CURRENT_ERA_RANK  = 4      # ERA rank out of 30

# -- IL return schedule --------------------------------------------------------
# NOTE: all return dates are ESTIMATES from team - actual returns may vary
# early_return = optimistic scenario
# est_return   = team estimate (baseline)
# late_return  = if setback occurs

IL_RETURNS = [
    # Julio Rodriguez removed 2026-07-23: he's back and his real production
    # is already baked into CURRENT_RS_G (pulled live from the season-to-date
    # batting overview). Leaving his entry in would double-count him -- once
    # in the real team stats, once again as a projected future boost. This
    # is a general gotcha with IL_RETURNS: _calc_il_impact() only checks the
    # return date against season-end, not against today, so an entry with a
    # past return_date keeps contributing at close to full weight instead of
    # zeroing out. Remove (don't just leave) an entry as soon as a player is
    # confirmed back, rather than trusting the date math to handle it.
    #
    # Brendan Donovan removed 2026-08-03: MLB.com confirms he's been cleared
    # and activated (same reasoning as Julio above -- don't leave a "removed"
    # player's entry in place with a stale future date).
    #
    # Rob Refsnyder entry removed too: rs_impact/ra_impact were already 0
    # ("DFA on return - no impact"), so it was contributing nothing -- dead
    # weight, not worth keeping around.
    #
    # J.P. Crawford ADDED 2026-08-03: placed on 10-day IL July 19 with wrist
    # inflammation (Fox 13 Seattle). Hollander said "couple weeks" -- treated
    # as MEDIUM confidence given that's a specific-enough estimate.
    #
    # Vargas/Brash/Criswell dates updated 2026-08-03 from GM Justin
    # Hollander's own public timeline updates (union-bulletin.com,
    # sports.mynorthwest.com, MLB.com, Yahoo Sports -- all within the last
    # week) -- the old dates here (Jul 20 for Brash, Aug 3 for Vargas, Aug 15
    # for Criswell) were significantly out of date; all three slipped later
    # into August than originally hoped.
    {
        "name":         "Will Wilson",
        "pos":          "3B",
        "return_date":  date(2026, 7, 20),   # UNVERIFIED as of 2026-08-03 --
        "early_return": date(2026, 7, 17),   # this date is carried over from
        "late_return":  date(2026, 7, 27),   # an earlier update and was NOT
        "rs_impact":    0.04,                # rechecked in this pass. Confirm
        "ra_impact":    0.00,                # his actual current status before
        "confidence":   "LOW",               # trusting this entry -- lowered
        "note":         "Bench depth -- status not reconfirmed 2026-08-03, "
                        "treat this date as stale until verified",
    },
    {
        "name":         "J.P. Crawford",
        "pos":          "SS",
        "return_date":  date(2026, 8, 2),    # "couple weeks" from Jul 19
        "early_return": date(2026, 7, 28),
        "late_return":  date(2026, 8, 9),
        "rs_impact":    0.10,
        "ra_impact":    0.00,
        "confidence":   "MEDIUM",
        "note":         "Elite OBP/walk rate -- 10-day IL, wrist inflammation, "
                        "placed Jul 19 (Rivas recalled in corresponding move)",
    },
    {
        "name":         "Matt Brash",
        "pos":          "RP",
        "return_date":  date(2026, 8, 25),   # "Aug 22-28 range" per Hollander
        "early_return": date(2026, 8, 20),
        "late_return":  date(2026, 9, 1),
        "rs_impact":    0.00,
        "ra_impact":    0.10,
        "confidence":   "MEDIUM",
        "note":         "0.54 ERA closer -- right lat inflammation, "
                        "targeting Aug 22-28 return",
    },
    {
        "name":         "Carlos Vargas",
        "pos":          "RP",
        "return_date":  date(2026, 8, 15),   # Hollander's stated target
        "early_return": date(2026, 8, 12),
        "late_return":  date(2026, 8, 22),
        "rs_impact":    0.00,
        "ra_impact":    0.05,
        "confidence":   "MEDIUM",            # was LOW; now a specific,
                                              # recently-reaffirmed date
        "note":         "Bullpen depth -- right lat strain, tentatively "
                        "scheduled to return Aug 15",
    },
    {
        "name":         "Cooper Criswell",
        "pos":          "RP",
        "return_date":  date(2026, 8, 30),   # "Aug 28-Sept 1" per Hollander
        "early_return": date(2026, 8, 28),
        "late_return":  date(2026, 9, 5),
        "rs_impact":    0.00,
        "ra_impact":    0.04,
        "confidence":   "LOW",               # furthest out, most uncertain
        "note":         "Right shoulder/pec strain -- targeting Aug 28-Sept 1 return",
    },
]

# -- confirmed deadline acquisitions ---------------------------------------
# RETIRED 2026-08-03: this used to hold ~10 hypothetical trade targets
# (Dominic Smith, JJ Bleday, Carlos Cortes, Dylan Lee, Raisel Iglesias,
# etc.) for pre-deadline "what if we get X" scenario modeling. The
# deadline has now passed -- none of those players were actually
# acquired, and no more trades are coming this season. Modeling win
# projections around them any further would be pure fiction.
#
# What replaces it: the two players ACTUALLY acquired (Ward, Dominguez),
# treated the same way IL_RETURNS treats a returning player -- a
# temporary modeled boost that should be REMOVED (not just left with a
# stale flag) once their production is naturally absorbed into
# CURRENT_RS_G/CURRENT_RA_G. Those two are pulled live from the team-wide
# batting/pitching overview in main.py, so once Ward and Dominguez have
# played enough games as Mariners for the team-wide averages to reflect
# them, this entry becomes redundant/double-counting -- same failure mode
# documented for Julio and Donovan in IL_RETURNS above. Check periodically
# and remove once their stats are clearly baked into the live team
# numbers rather than trusting a fixed date.
ACTUAL_ACQUISITIONS = {
    "Taylor Ward": {
        "pos":       "OF",
        "xwoba":     0.346,
        "rs_impact": 0.10,   # .729 OPS, 98th-percentile walk rate --
                              # clear offensive upgrade at a real need
        "ra_impact": 0.00,
        "cost":      "Alex Hoppe + 2 low-level pitching prospects",
        "note":      "Acquired from BAL 2026-08-03. .383 OBP -- highest "
                     "of any regular in the lineup. Pushes Robles to bench.",
        "acquired_date": date(2026, 8, 3),
    },
    "Seranthony Dominguez": {
        "pos":       "RP",
        "xwoba_against": 0.327,
        "rs_impact": 0.00,
        "ra_impact": 0.02,   # modest -- his .327 xwOBA against is worse
                              # than every current core bullpen arm; this
                              # is bullpen depth, not a clear upgrade (see
                              # deadline_trade_comparison.py for the full
                              # stat-by-stat breakdown backing this call)
        "cost":      "Luis Castillo",
        "note":      "Acquired from CHW 2026-08-03 for Castillo. Lost "
                     "closer job in late June; projects as setup/middle "
                     "relief behind Munoz, not closer competition.",
        "acquired_date": date(2026, 8, 3),
    },
}

# -- schedule difficulty -------------------------------------------------------
SCHEDULE = {
    "easy_games":    27,   # vs sub-.500 teams (24 + ~3 from next 7)
    "hard_games":    16,   # vs TB/NYY/LAD etc (15 + TBR series)
    "neutral_games": 25,   # everything else
    "easy_winpct":   0.600,
    "hard_winpct":   0.400,
    "neutral_winpct":0.515,
}
# easy + hard + neutral = 68 total remaining

# -- division context ----------------------------------------------------------
DIVISION = {
    "SEA": {"w": 47, "l": 47, "name": "Seattle Mariners"},
    "TEX": {"w": 47, "l": 46, "name": "Texas Rangers"},
    "HOU": {"w": 46, "l": 49, "name": "Houston Astros"},
    "ATH": {"w": 41, "l": 52, "name": "Athletics"},
    "LAA": {"w": 24, "l": 72, "name": "Los Angeles Angels"},
}


# -- core simulation functions -------------------------------------------------

def _pythagorean_winpct(rs_g: float, ra_g: float,
                        exp: float = 1.83) -> float:
    """Pythagorean win% formula."""
    if ra_g == 0:
        return 1.0
    return (rs_g ** exp) / (rs_g ** exp + ra_g ** exp)


def _project_games(win_pct: float,
                   schedule: dict = SCHEDULE) -> int:
    """Project wins over remaining schedule."""
    easy    = schedule["easy_games"]    * schedule["easy_winpct"]
    hard    = schedule["hard_games"]    * schedule["hard_winpct"]
    neutral = schedule["neutral_games"] * schedule["neutral_winpct"]
    # blend schedule difficulty with team win%
    base_wins  = easy + hard + neutral
    base_wpct  = base_wins / GAMES_REMAINING
    # weight team quality 60%, schedule 40%
    blended    = (win_pct * 0.60) + (base_wpct * 0.40)
    return round(GAMES_REMAINING * blended)


def _luck_correction(luck: float, games_remaining: int) -> float:
    """
    Convert luck stat to additional run differential.
    Luck = games over/under pythagorean expectation.
    Negative luck = team winning fewer than expected = correction coming.
    """
    # spread luck correction over remaining games
    correction_per_game = -luck / games_remaining
    return correction_per_game


def _calc_il_impact(return_date: date,
                    current_date: date = CURRENT_DATE,
                    games_remaining: int = GAMES_REMAINING) -> float:
    """
    Calculate fraction of remaining games a returning player impacts.
    Uses estimated return date - actual may vary.
    """
    season_end      = date(2026, 9, 28)
    days_remaining  = (season_end - current_date).days
    days_after_return = max(0, (season_end - return_date).days)
    if days_remaining <= 0:
        return 0.0
    return min(1.0, days_after_return / days_remaining)


def _calc_il_impact_optimistic(player: dict) -> float:
    """Use early_return date."""
    return _calc_il_impact(player.get("early_return", player["return_date"]))


def _calc_il_impact_pessimistic(player: dict) -> float:
    """Use late_return date."""
    return _calc_il_impact(player.get("late_return", player["return_date"]))


def _build_scenario(name: str,
                    acquisitions: list,
                    il_returns: list = None,
                    include_luck: bool = True,
                    use_early_returns: bool = False,
                    use_late_returns: bool = False) -> dict:
    """
    Build a single scenario.

    Parameters:
        name:              scenario name
        acquisitions:      list of player names from ACTUAL_ACQUISITIONS
        il_returns:        list of IL player names (None = all)
        include_luck:      apply luck correction
        use_early_returns: use optimistic early return dates
        use_late_returns:  use pessimistic late return dates
    """
    rs_g = CURRENT_RS_G
    ra_g = CURRENT_RA_G

    impacts = []

    # -- IL returns --
    returns_to_use = IL_RETURNS if il_returns is None else [
        r for r in IL_RETURNS if r["name"] in il_returns
    ]
    for player in returns_to_use:
        if use_early_returns:
            frac = _calc_il_impact_optimistic(player)
        elif use_late_returns:
            frac = _calc_il_impact_pessimistic(player)
        else:
            frac = _calc_il_impact(player["return_date"])
        if frac <= 0:
            continue
        rs_add = player["rs_impact"] * frac
        ra_sub = player["ra_impact"] * frac
        rs_g  += rs_add
        ra_g  -= ra_sub
        if rs_add > 0 or ra_sub > 0:
            impacts.append({
                "source":  f"IL return: {player['name']}",
                "rs":      round(rs_add, 3),
                "ra":      round(-ra_sub, 3),
                "games":   round(frac * GAMES_REMAINING),
                "note":    player["note"],
            })

    # -- confirmed acquisitions --
    # Full weight (not a fractional "days since deadline" estimate like the
    # old hardcoded July 31 reference used) -- Ward and Dominguez are
    # CONFIRMED active on the roster right now, not a hypothetical future
    # pickup, so there's no ambiguity about "when did they join" to model
    # around. All GAMES_REMAINING games get their impact applied.
    for acq_name in acquisitions:
        if acq_name not in ACTUAL_ACQUISITIONS:
            continue
        acq    = ACTUAL_ACQUISITIONS[acq_name]
        rs_add = acq["rs_impact"]
        ra_sub = acq["ra_impact"]
        rs_g  += rs_add
        ra_g  -= ra_sub
        impacts.append({
            "source": f"Acquired: {acq_name}",
            "rs":     round(rs_add, 3),
            "ra":     round(-ra_sub, 3),
            "games":  GAMES_REMAINING,
            "note":   acq["note"],
            "cost":   acq.get("cost",""),
        })

    # -- luck correction --
    luck_wins = 0
    if include_luck and CURRENT_LUCK < 0:
        luck_correction = -CURRENT_LUCK * 0.50  # half corrects
        luck_wins       = round(luck_correction)
        impacts.append({
            "source": "Luck correction",
            "rs":     0,
            "ra":     0,
            "games":  0,
            "note":   f"Luck {CURRENT_LUCK} - ~{luck_wins} free wins",
        })

    # -- project --
    win_pct    = _pythagorean_winpct(rs_g, ra_g)
    proj_wins  = _project_games(win_pct) + luck_wins
    proj_losses = GAMES_REMAINING - (proj_wins - luck_wins) + 0
    proj_losses = GAMES_REMAINING - proj_wins

    final_w    = CURRENT_W + proj_wins
    final_l    = CURRENT_L + proj_losses
    final_wpct = round(final_w / (final_w + final_l), 3)

    return {
        "name":       name,
        "rs_g":       round(rs_g, 2),
        "ra_g":       round(ra_g, 2),
        "win_pct":    round(win_pct, 3),
        "proj_wins":  proj_wins,
        "final_w":    final_w,
        "final_l":    final_l,
        "final_wpct": final_wpct,
        "luck_wins":  luck_wins,
        "impacts":    impacts,
    }


# -- preset scenarios ----------------------------------------------------------
def run_simulation(custom_acquisitions: list = None) -> dict:
    """
    Runs all preset scenarios + optional custom scenario.

    RETIRED 2026-08-03: this used to include 5 scenarios built around
    hypothetical pre-deadline targets (yes_targets, yes_plus_pitching,
    maybe_targets, best_case, and part of il_pessimistic). The deadline
    has passed and none of those players were acquired -- keeping them
    would mean reporting win projections based on trades that never
    happened. Replaced with scenarios reflecting what's actually true
    now: IL-return timing (the real remaining uncertainty this season)
    and the confirmed real acquisitions (Ward, Dominguez).

    Returns dict of scenario results.
    """
    print("\n[simulate] Running post-deadline scenarios...")

    scenarios = {}

    # scenario 0: baseline (no IL returns, no acquisition impact modeled)
    scenarios["baseline"] = _build_scenario(
        "Baseline - no IL returns",
        acquisitions=[],
        il_returns=[],
        include_luck=False,
    )

    # scenario 1: IL returns only, acquisitions not modeled separately
    # (their impact is already flowing into CURRENT_RS_G/RA_G naturally
    # once they've played enough games -- see ACTUAL_ACQUISITIONS note)
    scenarios["il_only"] = _build_scenario(
        "IL Returns Only",
        acquisitions=[],
        il_returns=None,
        include_luck=True,
    )

    # scenario 2: IL returns + confirmed deadline acquisitions
    scenarios["with_acquisitions"] = _build_scenario(
        "IL Returns + Ward + Dominguez",
        acquisitions=["Taylor Ward", "Seranthony Dominguez"],
        il_returns=None,
        include_luck=True,
    )

    # scenario 3: worst case - no IL returns land on schedule
    scenarios["worst_case"] = _build_scenario(
        "Worst Case - IL returns slip/setback",
        acquisitions=[],
        il_returns=["Matt Brash", "Carlos Vargas", "Cooper Criswell",
                     "J.P. Crawford", "Will Wilson"],
        include_luck=False,
        use_late_returns=True,
    )

    # scenario 4: optimistic IL - everyone returns early
    scenarios["il_optimistic"] = _build_scenario(
        "Optimistic IL - Everyone Returns Early",
        acquisitions=["Taylor Ward", "Seranthony Dominguez"],
        il_returns=None,
        include_luck=True,
        use_early_returns=True,
    )

    # scenario 5: pessimistic IL - setbacks happen
    scenarios["il_pessimistic"] = _build_scenario(
        "Pessimistic IL - Setbacks, Late Returns",
        acquisitions=["Taylor Ward", "Seranthony Dominguez"],
        il_returns=None,
        include_luck=False,
        use_late_returns=True,
    )

    # custom scenario
    if custom_acquisitions:
        scenarios["custom"] = _build_scenario(
            f"Custom - {', '.join(custom_acquisitions)}",
            acquisitions=custom_acquisitions,
            il_returns=None,
            include_luck=True,
        )

    print(f"[simulate] {len(scenarios)} scenarios complete.")
    return scenarios


# -- division projection -------------------------------------------------------
def project_division(sea_final_w: int) -> dict:
    """Project division/playoff outcome based on final wins."""
    # rough projections for other teams
    tex_proj = 83   # TEX at 47-46, similar trajectory
    hou_proj = 79   # HOU fading, 46-49
    ath_proj = 72   # ATH young team, 41-52

    division_winner = sea_final_w > tex_proj
    wc_position     = None

    if division_winner:
        wc_position = "Division winner"
    elif sea_final_w >= 87:
        wc_position = "Wild Card 1"
    elif sea_final_w >= 84:
        wc_position = "Wild Card 2"
    elif sea_final_w >= 82:
        wc_position = "Wild Card 3 (bubble)"
    elif sea_final_w >= 79:
        wc_position = "Wild Card bubble - risky"
    else:
        wc_position = "Likely miss playoffs"

    return {
        "sea":              sea_final_w,
        "tex_proj":         tex_proj,
        "hou_proj":         hou_proj,
        "division_winner":  division_winner,
        "playoff_position": wc_position,
        "gap_to_tex":       sea_final_w - tex_proj,
        "games_ahead_hou":  sea_final_w - hou_proj,
    }


# -- pretty print --------------------------------------------------------------
def print_simulation(scenarios: dict):
    print(f"\n{'='*70}")
    print(f"SEATTLE MARINERS - STRETCH RUN SIMULATOR")
    print(f"As of: {CURRENT_DATE}  |  Record: {CURRENT_W}-{CURRENT_L}")
    print(f"Games remaining: {GAMES_REMAINING}  |  Luck: {CURRENT_LUCK}")
    print(f"{'='*70}")

    # summary table
    print(f"\n-- SCENARIO SUMMARY --")
    print(f"  {'Scenario':<45} {'RS/G':>5} {'RA/G':>5} "
          f"{'Win%':>5} {'Proj W':>6} {'Final':>8} {'Playoff'}")
    print("  " + "-"*95)

    for key, s in scenarios.items():
        div = project_division(s["final_w"])
        print(f"  {s['name'][:44]:<45} "
              f"{s['rs_g']:>5.2f} {s['ra_g']:>5.2f} "
              f"{s['win_pct']:>5.3f} {s['proj_wins']:>6} "
              f"{s['final_w']}-{s['final_l']:>2}  "
              f"{div['playoff_position']}")

    # detail each scenario
    for key, s in scenarios.items():
        print(f"\n-- {s['name'].upper()} --")
        print(f"  RS/G: {s['rs_g']}  RA/G: {s['ra_g']}  "
              f"Win%: {s['win_pct']}  Luck wins: {s['luck_wins']}")
        print(f"  Projected: {s['final_w']}-{s['final_l']}  "
              f"({s['final_wpct']} win%)")

        div = project_division(s["final_w"])
        print(f"  Playoff: {div['playoff_position']}")
        print(f"  vs TEX:  {div['gap_to_tex']:+d} games")
        print(f"  vs HOU:  {div['games_ahead_hou']:+d} games")

        if s["impacts"]:
            print(f"  Factors:")
            for imp in s["impacts"]:
                rs_str = f"+{imp['rs']:.3f} RS/G" if imp["rs"] > 0 else ""
                ra_str = f"{imp['ra']:.3f} RA/G" if imp["ra"] < 0 else ""
                stat_str = "  ".join(filter(None, [rs_str, ra_str]))
                conf = f"[{imp.get('confidence','')}]" if imp.get('confidence') else ""
                print(f"    * {imp['source']:<35} "
                      f"{stat_str:<18} {conf} {imp['note']}")

    print(f"\n{'='*70}")
    print(f"NOTES:")
    print(f"  * IL return dates are TEAM ESTIMATES - actual returns may vary")
    print(f"  * HIGH confidence = 7-day IL, short stint")
    print(f"  * MEDIUM confidence = 10-15 day IL")
    print(f"  * LOW confidence = 60-day IL, longer recovery")
    print(f"  * RS/G and RA/G impacts estimated from xwOBA differentials")
    print(f"  * TEX projected ~83W, HOU ~79W based on current trajectory")
    print(f"{'='*70}\n")


# -- test ----------------------------------------------------------------------
if __name__ == "__main__":
    scenarios = run_simulation()
    print_simulation(scenarios)

    # example custom scenario -- IL returns without Dominguez's modest
    # impact, isolating just Ward's contribution
    print("\n-- CUSTOM SCENARIO: Just Ward, no Dominguez --")
    custom = run_simulation(["Taylor Ward"])
    print_simulation({"custom": custom["custom"]})