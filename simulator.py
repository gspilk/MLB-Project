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
# PLACEHOLDER DEFAULTS ONLY. Every one of these except CURRENT_DATE and
# SEASON_GAMES gets overwritten by main.py before any real simulation runs
# (search main.py for "sim_mod.CURRENT_W =", etc. -- it pulls your real,
# live record/RS_G/RA_G/luck/ERA rank and sets these module attributes
# directly). The numbers sitting here (47-47, RS_G 3.82...) are just
# whatever was true whenever this file was last hand-edited -- they will
# look stale almost immediately and that's expected, not a bug, AS LONG AS
# you're running this through `python main.py`. Running simulator.py
# directly skips that override step entirely and uses these placeholders
# for real (see the loud warning in the __main__ block below) -- don't
# trust standalone output, and don't "fix" these values by hand thinking
# it'll help; main.py's live override makes the specific numbers here
# irrelevant for any real run.
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
        "return_date":  date(2026, 9, 15),   # confirmed still on 60-day IL
        "early_return": date(2026, 9, 1),    # (thumb) as of Aug 6-9 per
        "late_return":  date(2026, 9, 28),   # KJR radio report -- no specific
        "rs_impact":    0.02,                # target date given anywhere,
        "ra_impact":    0.00,                # so this is a wide, genuinely
        "confidence":   "LOW",               # low-confidence guess, not a
        "note":         "Bench depth -- confirmed still on 60-day IL "
                        "(thumb) as of Aug 6-9, no target date reported",
    },
    {
        "name":         "J.P. Crawford",
        "pos":          "SS",
        "return_date":  date(2026, 9, 5),    # the old Aug 2 estimate was
        "early_return": date(2026, 8, 25),   # WRONG -- confirmed via a live
        "late_return":  date(2026, 9, 15),   # bbref fetch that he's STILL on
        "rs_impact":    0.10,                # the IL as of Aug 14. A separate
        "ra_impact":    0.00,                # report said "down 3-5 days
        "confidence":   "LOW",               # after an injection, then ramp
        "note":         "Elite OBP/walk rate -- still on IL as of Aug 14 "
                        "(confirmed via live bbref fetch), well past the "
                        "earlier Aug 2 estimate. Re-verify before trusting "
                        "this date either.",
    },
    {
        "name":         "Cole Wilcox",
        "pos":          "RP",
        "return_date":  date(2026, 8, 28),   # oblique strains typically
        "early_return": date(2026, 8, 21),   # 3-4 weeks; official word is
        "late_return":  date(2026, 9, 10),   # just "TBD" as of Aug 7, so
        "rs_impact":    0.00,                # this is an estimate, not a
        "ra_impact":    0.03,                # reported target
        "confidence":   "LOW",
        "note":         "Left oblique strain, placed Aug 7. Was a "
                        "dependable reliever pre-injury (4.00 ERA, 27 IP) "
                        "-- no official target date reported yet",
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
        "return_date":  date(2026, 8, 22),   # pushed back from Aug 15 --
        "early_return": date(2026, 8, 18),   # confirmed "about to begin a
        "late_return":  date(2026, 8, 28),   # rehab assignment" as of
        "rs_impact":    0.00,                # ~Aug 14, and rehab stints
        "ra_impact":    0.05,                # themselves typically run
        "confidence":   "MEDIUM",            # 1-2 weeks before MLB
                                              # activation
        "note":         "Bullpen depth -- right lat strain, beginning "
                        "rehab assignment as of mid-Aug after missing "
                        "116 straight games",
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
    # UPDATED 2026-08-14 -- both entries below originally carried PRE-TRADE
    # stats (Ward's Angels/Orioles numbers, Dominguez's White Sox numbers),
    # which was wrong the moment they actually started playing for Seattle.
    # Their real Mariners debuts, confirmed via a live bbref fetch:
    #   Ward:      8 G, 32 PA,  .156 BA / .375 OPS
    #   Dominguez: 5 G, 3.1 IP, 10.80 ERA, 3.000 WHIP
    # Both samples are tiny and noisy -- not something to build confident
    # going-forward expectations on. More importantly: since CURRENT_RS_G/
    # CURRENT_RA_G are pulled live from TEAM-WIDE stats in main.py, these
    # few games are already baked into that team average. Keeping a large
    # separate rs_impact/ra_impact boost here would double-count them, the
    # same failure mode documented for Julio/Donovan in IL_RETURNS above --
    # just smaller in scale (a few games out of ~120 played, not a whole
    # player's absence). Impacts brought down toward near-zero for that
    # reason, not because they've been bad -- there just isn't a clean way
    # to model "expected future value" on top of team-wide stats that
    # already include them.
    "Taylor Ward": {
        "pos":       "OF",
        "xwoba":     0.346,   # xwOBA still their pre-trade rate; early
                              # Mariners OPS (.375) is real but too small
                              # a sample to trust over that established rate
        "rs_impact": 0.02,
        "ra_impact": 0.00,
        "cost":      "Alex Hoppe + 2 low-level pitching prospects",
        "note":      "Acquired from BAL 2026-08-03. Rough Mariners debut "
                     "so far (8 G, .375 OPS) -- small sample, already "
                     "reflected in team-wide stats, watch for regression "
                     "toward his established .729 OPS rate either way.",
        "acquired_date": date(2026, 8, 3),
    },
    "Seranthony Dominguez": {
        "pos":       "RP",
        "xwoba_against": 0.327,
        "rs_impact": 0.00,
        "ra_impact": 0.00,
        "cost":      "Luis Castillo",
        "note":      "Acquired from CHW 2026-08-03 for Castillo. Rough "
                     "Mariners debut (5 G, 3.1 IP, 10.80 ERA) -- tiny "
                     "sample, already reflected in team-wide stats. Was "
                     "already a modest-at-best add per deadline_trade_"
                     "comparison.py's stat-by-stat breakdown even before this.",
        "acquired_date": date(2026, 8, 3),
    },
}

# -- schedule difficulty -------------------------------------------------------
SCHEDULE = {
    "easy_games":    27,   # DEPRECATED FALLBACK ONLY -- see
    "hard_games":    16,   # compute_schedule_difficulty() below. This
    "neutral_games": 25,   # fixed dict summed to 68 games, which was only
    "easy_winpct":   0.600,  # correct back when GAMES_REMAINING was 68 (near
    "hard_winpct":   0.400,  # the trade deadline). As the season correctly
    "neutral_winpct":0.515,  # ticks GAMES_REMAINING down, this fixed count
}                            # implies MORE remaining games than actually
# exist, which silently produces a mathematically impossible >100% blended
# win rate in _project_games() the closer the season gets to ending -- a
# real bug that got worse every single day post-deadline (verified: at
# GAMES_REMAINING=68 this gives a sane 52.2% base rate; at 25 it gives an
# impossible 141.9%). Only used now as an emergency fallback if live
# schedule/standings data isn't available for some reason.


def apply_live_state(data: dict, analysis: dict) -> None:
    """
    Pulls real, current team state (record, RS/G, RA/G, luck, ERA rank)
    from already-built data/analysis and updates this module's globals.

    MOVED HERE 2026-08-26 from main.py, where this exact logic used to
    live as an external "reach into simulator.py and overwrite its
    module attributes" block. That pattern was fragile in practice --
    it's exactly what caused several real sync bugs this session, where
    simulator.py got updated but main.py's override block didn't (or
    vice versa), silently producing wrong results. Moving it here makes
    simulator.py responsible for its own live state instead of
    depending on an external caller to patch it correctly every time.

    Call this before run_simulation() if you want live data applied --
    or just pass `data`/`analysis` directly to run_simulation(), which
    calls this automatically. If this is never called, the placeholder
    constants above remain in effect (with the loud __main__ warning).
    """
    global CURRENT_W, CURRENT_L, GAMES_REMAINING, CURRENT_LUCK
    global CURRENT_ERA_RANK, CURRENT_RS_G, CURRENT_RA_G

    standings = analysis.get("standings", {})
    record = standings.get("record")
    if not record:
        print("  [warn] no live record available in apply_live_state() -- "
              "leaving current module state as-is (may be placeholders)")
        return

    try:
        w, l = map(int, record.split("-"))
    except (ValueError, AttributeError):
        print(f"  [warn] could not parse record '{record}' in apply_live_state()")
        return

    CURRENT_W       = w
    CURRENT_L       = l
    GAMES_REMAINING = 162 - w - l
    CURRENT_LUCK    = float(standings.get("luck", -2.0) or -2.0)

    era_rank = analysis.get("pitching", {}).get("team_era_rank")
    if era_rank:
        CURRENT_ERA_RANK = era_rank

    ov_bat = data.get("overview", {}).get("batting")
    if ov_bat is not None and not ov_bat.empty:
        tm = next((c for c in ["Tm", "Team"] if c in ov_bat.columns), None)
        if tm:
            sea = ov_bat[ov_bat[tm].str.contains("Seattle", na=False)]
            if not sea.empty:
                r_val = pd.to_numeric(sea["R"].values[0], errors="coerce")
                g_val = pd.to_numeric(sea["G"].values[0], errors="coerce")
                if r_val and g_val and g_val > 0:
                    CURRENT_RS_G = round(r_val / g_val, 2)

    ov_pit = data.get("overview", {}).get("pitching")
    if ov_pit is not None and not ov_pit.empty:
        tm = next((c for c in ["Tm", "Team"] if c in ov_pit.columns), None)
        if tm:
            sea = ov_pit[ov_pit[tm].str.contains("Seattle", na=False)]
            if not sea.empty:
                for col in ["RA", "R", "RA9"]:
                    if col in sea.columns:
                        ra_val = pd.to_numeric(sea[col].values[0], errors="coerce")
                        g_val = pd.to_numeric(sea["G"].values[0], errors="coerce")
                        if ra_val and g_val and g_val > 0:
                            CURRENT_RA_G = round(ra_val / g_val, 2)
                            break

    print(f"  [sim] Live state applied: {CURRENT_W}-{CURRENT_L}  "
          f"RS/G {CURRENT_RS_G}  RA/G {CURRENT_RA_G}  Luck {CURRENT_LUCK}")


def compute_schedule_difficulty(data: dict) -> dict:
    """
    Builds a SCHEDULE-shaped dict from the ACTUAL remaining schedule and
    ACTUAL current opponent records, instead of a fixed guess made once
    near the trade deadline. Classifies each remaining game as easy/hard/
    neutral based on the opponent's real current win%, so this scales
    correctly with GAMES_REMAINING no matter how far into the season you
    are -- it can never produce more games than actually remain.

    Falls back to the stale SCHEDULE dict above (with a warning) only if
    live schedule/standings data isn't available.
    """
    # BUG FIX 2026-08-26: mariners_schedule_scraper.py deliberately splits
    # upcoming games into next7 (the next 7 games, for a separate
    # checklist display) and remaining (everything AFTER those 7) -- so
    # data["schedule"]["remaining"] alone is NOT the full remaining
    # schedule, it's missing exactly the next 7 games. This silently
    # undercounted the real remaining schedule by 7 games (about a
    # quarter of what's actually left this late in the season) in every
    # calculation that used this function. Combine both pieces to get
    # the true full remaining schedule.
    remaining_only = data.get("schedule", {}).get("remaining")
    next7 = data.get("schedule", {}).get("next7")
    if remaining_only is not None and next7 is not None and not next7.empty:
        remaining = pd.concat([next7, remaining_only], ignore_index=True)
    else:
        remaining = remaining_only

    all_teams = data.get("standings", {}).get("all_teams")

    if remaining is None or remaining.empty or all_teams is None or all_teams.empty:
        print("  [warn] no live schedule/standings -- falling back to "
              "stale hardcoded SCHEDULE, results may not be trustworthy")
        return dict(SCHEDULE)

    # map full team name -> win% for quick lookup; "Opp" column uses
    # 3-letter abbreviations, so build an abbreviation map too. Lowercase
    # both sides consistently -- TEAM_ABBR (reused from recommender.py)
    # stores lowercase full names, but standings data has proper-case
    # names ("Chicago Cubs"), so this would silently never match without
    # normalizing both to the same case.
    name_to_pct = dict(zip(
        all_teams["Tm"].str.lower(),
        pd.to_numeric(all_teams["W-L%"], errors="coerce")
    ))

    try:
        from recommender import TEAM_ABBR
        abbr_lookup = {v.upper(): k for k, v in TEAM_ABBR.items()}
    except ImportError:
        abbr_lookup = {}

    easy = hard = neutral = 0
    for opp in remaining["Opp"].dropna():
        opp_key = str(opp).strip().upper()
        full_name = abbr_lookup.get(opp_key)  # lowercase full name, or None
        pct = name_to_pct.get(full_name) if full_name else None
        if pct is None:
            neutral += 1  # unknown opponent -- treat as neutral, safest default
        elif pct < 0.450:
            easy += 1
        elif pct > 0.550:
            hard += 1
        else:
            neutral += 1

    total = easy + hard + neutral
    if total == 0:
        print("  [warn] could not classify any remaining games -- falling "
              "back to stale hardcoded SCHEDULE")
        return dict(SCHEDULE)

    print(f"  [schedule] {total} remaining games classified: "
          f"{easy} easy, {hard} hard, {neutral} neutral (live opponent records)")

    return {
        "easy_games": easy, "hard_games": hard, "neutral_games": neutral,
        "easy_winpct": 0.600, "hard_winpct": 0.400, "neutral_winpct": 0.515,
    }

# -- division context ----------------------------------------------------------
# REMOVED 2026-08-26: this DIVISION dict (hardcoded records from way back --
# "SEA 47-47", "TEX 47-46") was never actually referenced anywhere else in
# this file. Fully dead code, fully superseded by compute_rival_projections()
# above, which does the same job with live data. Left this note instead of
# just silently deleting it in case anyone goes looking for it later.


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
                    use_late_returns: bool = False,
                    schedule: dict = None) -> dict:
    """
    Build a single scenario.

    Parameters:
        name:              scenario name
        acquisitions:      list of player names from ACTUAL_ACQUISITIONS
        il_returns:        list of IL player names (None = all)
        include_luck:      apply luck correction
        use_early_returns: use optimistic early return dates
        use_late_returns:  use pessimistic late return dates
        schedule:          real schedule-difficulty dict from
                           compute_schedule_difficulty(data), or None to
                           fall back to the stale hardcoded SCHEDULE
                           (see that dict's docstring for why the
                           fallback shouldn't be trusted this late in
                           the season)
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
    proj_wins  = _project_games(win_pct, schedule=schedule or SCHEDULE) + luck_wins
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
def run_simulation(custom_acquisitions: list = None, schedule: dict = None,
                   data: dict = None, analysis: dict = None) -> dict:
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

    `data`/`analysis`: pass the already-built data_builder/team_analyzer
    output and this function applies live state (apply_live_state()) and
    computes the real schedule (compute_schedule_difficulty()) itself --
    no more need for the caller to manually patch module globals from
    outside, which is what main.py used to do and what caused real sync
    bugs this session. If you already have a `schedule` dict computed,
    pass it directly and it takes priority over auto-computing one.

    `schedule` should be the output of compute_schedule_difficulty(data)
    -- pass it in (or pass `data` and let this compute it) so every
    scenario uses the REAL remaining schedule instead of the stale
    hardcoded fallback (see SCHEDULE's docstring for why that fallback
    becomes actively wrong, not just imprecise, as the season goes on).

    Returns dict of scenario results.
    """
    if data is not None and analysis is not None:
        apply_live_state(data, analysis)
    if schedule is None and data is not None:
        schedule = compute_schedule_difficulty(data)

    print("\n[simulate] Running post-deadline scenarios...")

    scenarios = {}

    # scenario 0: baseline (no IL returns, no acquisition impact modeled)
    scenarios["baseline"] = _build_scenario(
        "Baseline - no IL returns",
        acquisitions=[],
        il_returns=[],
        include_luck=False,
        schedule=schedule,
    )

    # scenario 1: IL returns only, acquisitions not modeled separately
    # (their impact is already flowing into CURRENT_RS_G/RA_G naturally
    # once they've played enough games -- see ACTUAL_ACQUISITIONS note)
    scenarios["il_only"] = _build_scenario(
        "IL Returns Only",
        acquisitions=[],
        il_returns=None,
        include_luck=True,
        schedule=schedule,
    )

    # scenario 2: IL returns + confirmed deadline acquisitions
    scenarios["with_acquisitions"] = _build_scenario(
        "IL Returns + Ward + Dominguez",
        acquisitions=["Taylor Ward", "Seranthony Dominguez"],
        il_returns=None,
        include_luck=True,
        schedule=schedule,
    )

    # scenario 3: worst case - no IL returns land on schedule
    scenarios["worst_case"] = _build_scenario(
        "Worst Case - IL returns slip/setback",
        acquisitions=[],
        il_returns=["Matt Brash", "Carlos Vargas", "Cooper Criswell",
                     "J.P. Crawford", "Will Wilson", "Cole Wilcox"],
        include_luck=False,
        use_late_returns=True,
        schedule=schedule,
    )

    # scenario 4: optimistic IL - everyone returns early
    scenarios["il_optimistic"] = _build_scenario(
        "Optimistic IL - Everyone Returns Early",
        acquisitions=["Taylor Ward", "Seranthony Dominguez"],
        il_returns=None,
        include_luck=True,
        use_early_returns=True,
        schedule=schedule,
    )

    # scenario 5: pessimistic IL - setbacks happen
    scenarios["il_pessimistic"] = _build_scenario(
        "Pessimistic IL - Setbacks, Late Returns",
        acquisitions=["Taylor Ward", "Seranthony Dominguez"],
        il_returns=None,
        include_luck=False,
        use_late_returns=True,
        schedule=schedule,
    )

    # custom scenario
    if custom_acquisitions:
        scenarios["custom"] = _build_scenario(
            f"Custom - {', '.join(custom_acquisitions)}",
            acquisitions=custom_acquisitions,
            il_returns=None,
            include_luck=True,
            schedule=schedule,
        )

    print(f"[simulate] {len(scenarios)} scenarios complete.")
    return scenarios


def compute_rival_projections(data: dict) -> dict:
    """
    Projects the other AL West teams' final win totals from their REAL,
    current pace -- replaces the hardcoded tex_proj/hou_proj/ath_proj
    constants that used to live here (set once, weeks ago, referencing a
    Texas record from back near the trade deadline -- "TEX at 47-46").
    Same staleness problem as the old fixed SCHEDULE dict had: those
    numbers don't update as the season progresses, so comparisons built
    on them (e.g. "vs HOU: -1 games") can look far closer than reality
    once rivals have kept winning while the hardcoded number stood still.

    Uses the same simple pace-extension method as monte_carlo.py's
    estimate_playoff_cutoff()/estimate_division_leader() -- extends each
    team's current win% over their own remaining games. Not a full
    simulation of their remaining schedule, just a live, current
    estimate instead of a frozen one.
    """
    all_teams = data.get("standings", {}).get("all_teams")
    if all_teams is None or all_teams.empty:
        print("  [warn] no live standings for rival projections -- "
              "falling back to stale hardcoded estimates")
        return {"tex_proj": 83, "hou_proj": 79, "ath_proj": 72}

    def _project(team_name, fallback):
        row = all_teams[all_teams["Tm"] == team_name]
        if row.empty:
            return fallback
        w = pd.to_numeric(row["W"].values[0], errors="coerce")
        l = pd.to_numeric(row["L"].values[0], errors="coerce")
        pct = pd.to_numeric(row["W-L%"].values[0], errors="coerce")
        if pd.isna(w) or pd.isna(l) or pd.isna(pct):
            return fallback
        games_remaining = 162 - w - l
        return round(w + games_remaining * pct)

    return {
        "tex_proj": _project("Texas Rangers", 83),
        "hou_proj": _project("Houston Astros", 79),
        "ath_proj": _project("Athletics", 72),
    }


# -- division projection -------------------------------------------------------
def project_division(sea_final_w: int, rival_projections: dict = None) -> dict:
    """Project division/playoff outcome based on final wins.

    `rival_projections` should come from compute_rival_projections(data)
    -- pass it through from main.py so this uses live current pace
    instead of the stale fallback constants below."""
    if rival_projections is None:
        print("  [warn] project_division() called without live rival "
              "projections -- using stale hardcoded fallback values")
        rival_projections = {"tex_proj": 83, "hou_proj": 79, "ath_proj": 72}

    tex_proj = rival_projections.get("tex_proj", 83)
    hou_proj = rival_projections.get("hou_proj", 79)
    ath_proj = rival_projections.get("ath_proj", 72)

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
def print_simulation(scenarios: dict, rival_projections: dict = None):
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
        div = project_division(s["final_w"], rival_projections)
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

        div = project_division(s["final_w"], rival_projections)
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
    if rival_projections:
        print(f"  * TEX projected ~{rival_projections.get('tex_proj','?')}W, "
              f"HOU ~{rival_projections.get('hou_proj','?')}W "
              f"(live pace-based estimate, see compute_rival_projections())")
    else:
        print(f"  * TEX/HOU rival projections unavailable -- rival_projections "
              f"was not passed to print_simulation()")
    print(f"{'='*70}\n")


# -- test ----------------------------------------------------------------------
if __name__ == "__main__":
    # UPDATED 2026-08-26: running this file directly used to silently use
    # stale placeholder constants (with a loud warning telling you to run
    # main.py instead). Now it builds live data itself via data_builder/
    # team_analyzer and applies it the same way main.py does -- standalone
    # runs are correct by default, no separate "real" entry point needed.
    print("[standalone] Building live data for a real standalone run...")
    try:
        from data_builder import build_all
        from team_analyzer import analyze_team

        _data = build_all(2026)
        _analysis = analyze_team(_data)

        scenarios = run_simulation(data=_data, analysis=_analysis)
        rival_projections = compute_rival_projections(_data)
        print_simulation(scenarios, rival_projections)

        # example custom scenario -- IL returns without Dominguez's modest
        # impact, isolating just Ward's contribution
        print("\n-- CUSTOM SCENARIO: Just Ward, no Dominguez --")
        custom = run_simulation(["Taylor Ward"], data=_data, analysis=_analysis)
        print_simulation({"custom": custom["custom"]}, rival_projections)

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"! Could not build live data ({e}) -- falling back to")
        print(f"! placeholder record {CURRENT_W}-{CURRENT_L}. Results below")
        print("! are NOT based on real data.")
        print("!" * 70)
        scenarios = run_simulation()
        print_simulation(scenarios)
