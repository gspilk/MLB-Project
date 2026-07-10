"""
simulator.py
Simulates the Mariners rest-of-season record under different scenarios.

Models:
  - IL returns (known dates)
  - Deadline acquisitions (hypothetical)
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
    {
        "name":         "Julio Rodriguez",
        "pos":          "CF",
        "return_date":  date(2026, 7, 10),   # team estimate
        "early_return": date(2026, 7, 10),   # today
        "late_return":  date(2026, 7, 17),   # if setback
        "rs_impact":    0.18,
        "ra_impact":    0.00,
        "confidence":   "HIGH",              # 7-day IL, short
        "note":         "Franchise CF, .337 xwOBA, 14 HR - est Jul 10",
    },
    {
        "name":         "Rob Refsnyder",
        "pos":          "DH",
        "return_date":  date(2026, 7, 10),
        "early_return": date(2026, 7, 10),
        "late_return":  date(2026, 7, 10),
        "rs_impact":    0.00,
        "ra_impact":    0.00,
        "confidence":   "N/A",
        "note":         "DFA on return - no impact",
    },
    {
        "name":         "Brendan Donovan",
        "pos":          "3B",
        "return_date":  date(2026, 7, 17),   # team estimate
        "early_return": date(2026, 7, 14),
        "late_return":  date(2026, 7, 24),
        "rs_impact":    0.14,
        "ra_impact":    0.00,
        "confidence":   "MEDIUM",
        "note":         ".839 OPS when healthy - est Jul 17",
    },
    {
        "name":         "Will Wilson",
        "pos":          "3B",
        "return_date":  date(2026, 7, 20),   # team estimate
        "early_return": date(2026, 7, 17),
        "late_return":  date(2026, 7, 27),
        "rs_impact":    0.04,
        "ra_impact":    0.00,
        "confidence":   "MEDIUM",
        "note":         "Bench depth - est Jul 20",
    },
    {
        "name":         "Matt Brash",
        "pos":          "RP",
        "return_date":  date(2026, 7, 20),   # estimated from 15-day IL
        "early_return": date(2026, 7, 17),
        "late_return":  date(2026, 7, 31),
        "rs_impact":    0.00,
        "ra_impact":    0.10,
        "confidence":   "MEDIUM",
        "note":         "0.54 ERA closer - est Jul 20 from 15-day IL",
    },
    {
        "name":         "Carlos Vargas",
        "pos":          "RP",
        "return_date":  date(2026, 8, 3),    # team estimate
        "early_return": date(2026, 7, 28),
        "late_return":  date(2026, 8, 17),
        "rs_impact":    0.00,
        "ra_impact":    0.05,
        "confidence":   "LOW",               # 60-day IL, uncertain
        "note":         "Bullpen depth - est Aug 3 from 60-day IL",
    },
    {
        "name":         "Cooper Criswell",
        "pos":          "RP",
        "return_date":  date(2026, 8, 15),   # estimated
        "early_return": date(2026, 8, 1),
        "late_return":  date(2026, 9, 1),
        "rs_impact":    0.00,
        "ra_impact":    0.04,
        "confidence":   "LOW",
        "note":         "60-day IL - return date uncertain",
    },
]

# -- deadline acquisition profiles --------------------------------------------
ACQUISITION_PROFILES = {
    "Bryce Eldridge": {
        "pos":       "1B/DH",
        "xwoba":     0.405,
        "rs_impact": 0.12,   # replaces below average 1B production
        "ra_impact": 0.00,
        "cost":      "Low prospect",
        "note":      "NL power bat, .405 xwOBA, upside",
        "available": True,
    },
    "JJ Bleday": {
        "pos":       "OF/DH",
        "xwoba":     0.399,
        "rs_impact": 0.10,
        "ra_impact": 0.00,
        "cost":      "Low prospect",
        "note":      "OF depth, .399 xwOBA, 14 HR",
        "available": True,
    },
    "Ryan Jeffers": {
        "pos":       "C/DH",
        "xwoba":     0.389,
        "rs_impact": 0.08,
        "ra_impact": 0.00,
        "cost":      "Low-mid prospect",
        "note":      ".949 OPS, insurance/DH flexibility",
        "available": True,
    },
    "Dillon Dingler": {
        "pos":       "C/DH",
        "xwoba":     0.409,
        "rs_impact": 0.11,
        "ra_impact": 0.00,
        "cost":      "Mid prospect",
        "note":      "Tigers seller, .409 xwOBA, 19 HR",
        "available": True,
    },
    "Miguel Vargas": {
        "pos":       "3B/DH",
        "xwoba":     0.397,
        "rs_impact": 0.09,
        "ra_impact": 0.00,
        "cost":      "Low prospect",
        "note":      "Versatile, .397 xwOBA, 20 HR",
        "available": True,
    },
    "Willson Contreras": {
        "pos":       "C/DH",
        "xwoba":     0.395,
        "rs_impact": 0.07,
        "ra_impact": 0.00,
        "cost":      "Low prospect",
        "note":      "Veteran, playoff experience",
        "available": True,
    },
    "Dylan Lee": {
        "pos":       "RP",
        "xwoba_against": 0.209,
        "rs_impact": 0.00,
        "ra_impact": 0.09,
        "cost":      "Waiver/Low prospect",
        "note":      "Elite .209 xwOBA, 1.52 ERA",
        "available": True,
    },
    "Raisel Iglesias": {
        "pos":       "RP",
        "xwoba_against": 0.227,
        "rs_impact": 0.00,
        "ra_impact": 0.07,
        "cost":      "Low-mid prospect",
        "note":      "Veteran closer/setup, 2.30 ERA",
        "available": True,
    },
    "Louis Varland": {
        "pos":       "RP",
        "xwoba_against": 0.216,
        "rs_impact": 0.00,
        "ra_impact": 0.08,
        "cost":      "Low prospect",
        "note":      "Elite .216 xwOBA, 0.94 ERA",
        "available": True,
    },
    "Dylan Dodd": {
        "pos":       "RP",
        "xwoba_against": 0.229,
        "rs_impact": 0.00,
        "ra_impact": 0.06,
        "cost":      "Waiver",
        "note":      "Solid depth, 2.08 ERA",
        "available": True,
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
        acquisitions:      list of player names from ACQUISITION_PROFILES
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

    # -- acquisitions --
    deadline_date   = date(2026, 7, 31)
    deadline_frac   = _calc_il_impact(deadline_date)
    for acq_name in acquisitions:
        if acq_name not in ACQUISITION_PROFILES:
            continue
        acq    = ACQUISITION_PROFILES[acq_name]
        rs_add = acq["rs_impact"] * deadline_frac
        ra_sub = acq["ra_impact"] * deadline_frac
        rs_g  += rs_add
        ra_g  -= ra_sub
        impacts.append({
            "source": f"Acquired: {acq_name}",
            "rs":     round(rs_add, 3),
            "ra":     round(-ra_sub, 3),
            "games":  round(deadline_frac * GAMES_REMAINING),
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

    Returns dict of scenario results.
    """
    print("\n[simulate] Running deadline scenarios...")

    scenarios = {}

    # scenario 0: baseline (no moves, no IL returns)
    scenarios["baseline"] = _build_scenario(
        "Baseline - no moves, no IL returns",
        acquisitions=[],
        il_returns=[],
        include_luck=False,
    )

    # scenario 1: IL returns only (no deadline moves)
    scenarios["il_only"] = _build_scenario(
        "IL Returns Only - no deadline moves",
        acquisitions=[],
        il_returns=None,
        include_luck=True,
    )

    # scenario 2: IL returns + your YES acquisitions
    scenarios["yes_targets"] = _build_scenario(
        "IL Returns + YES Targets (Eldridge + Bleday + Jeffers)",
        acquisitions=["Bryce Eldridge", "JJ Bleday", "Ryan Jeffers"],
        il_returns=None,
        include_luck=True,
    )

    # scenario 3: IL returns + YES + pitching
    scenarios["yes_plus_pitching"] = _build_scenario(
        "IL Returns + YES + Pitching (+ Lee + Varland)",
        acquisitions=["Bryce Eldridge", "JJ Bleday", "Ryan Jeffers",
                      "Dylan Lee", "Louis Varland"],
        il_returns=None,
        include_luck=True,
    )

    # scenario 4: IL returns + MAYBE targets
    scenarios["maybe_targets"] = _build_scenario(
        "IL Returns + MAYBE (Dingler + Vargas + Contreras)",
        acquisitions=["Dillon Dingler", "Miguel Vargas", "Willson Contreras"],
        il_returns=None,
        include_luck=True,
    )

    # scenario 5: best case - all IL + all YES + pitching
    scenarios["best_case"] = _build_scenario(
        "Best Case - All IL + YES + Pitching",
        acquisitions=["Bryce Eldridge", "JJ Bleday", "Ryan Jeffers",
                      "Dylan Lee", "Louis Varland", "Raisel Iglesias"],
        il_returns=None,
        include_luck=True,
    )

    # scenario 6: worst case - no moves, injuries linger
    scenarios["worst_case"] = _build_scenario(
        "Worst Case - No moves, Julio slow return",
        acquisitions=[],
        il_returns=["Matt Brash", "Carlos Vargas", "Cooper Criswell"],
        include_luck=False,
    )

    # scenario 7: optimistic IL - everyone returns early
    scenarios["il_optimistic"] = _build_scenario(
        "Optimistic IL - Everyone Returns Early",
        acquisitions=[],
        il_returns=None,
        include_luck=True,
        use_early_returns=True,
    )

    # scenario 8: pessimistic IL - setbacks happen
    scenarios["il_pessimistic"] = _build_scenario(
        "Pessimistic IL - Setbacks, Late Returns",
        acquisitions=["Bryce Eldridge", "Dylan Lee"],
        il_returns=None,
        include_luck=False,
        use_early_returns=False,
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
    print(f"SEATTLE MARINERS - DEADLINE ACQUISITION SIMULATOR")
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

    # example custom scenario
    print("\n-- CUSTOM SCENARIO: Just Eldridge + Lee --")
    custom = run_simulation(["Bryce Eldridge", "Dylan Lee"])
    print_simulation({"custom": custom["custom"]})
