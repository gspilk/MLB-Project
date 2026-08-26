"""
monte_carlo.py
Full league-wide Monte Carlo simulator: simulates every MLB team's
remaining games (not just Seattle's), determines the real playoff field
per simulated season, then runs the actual postseason bracket through
every round.

WHY THE PREVIOUS VERSION WAS TOO SIMPLE
-----------------------------------------
An earlier version of this script only simulated Seattle's games and
assumed every OTHER team (the Wild Card cutoff team, the division
leader) would hold their exact current pace with zero variance. That
badly understated Seattle's real odds -- in a mediocre division where
the "leader" is sitting at .500, that leader has plenty of room to
slump too, which opens up real paths for Seattle that a fixed-target
model completely misses. This version fixes that by simulating
EVERYONE's remaining games, so the competition is exactly as uncertain
as Seattle's own record is.

METHOD
------
- Seattle: opponent-aware, using the real remaining schedule + log5 +
  home/away (same as before -- this is the one team we have full
  schedule detail for).
- Every other team: their remaining games are simulated as independent
  Bernoulli trials at THEIR OWN current win%. This is a real
  simplification (no opponent-specific matchups for teams other than
  Seattle, since we don't have every team's schedule scraped), but it's
  a standard, reasonable approximation used by simplified playoff-odds
  calculators, and a big improvement over treating other teams' paces
  as fixed and certain.
- Playoff field per simulated season: division winner in each of the 6
  divisions (best simulated record), plus the next-best 3 teams per
  league as Wild Cards.
- Postseason bracket (fixed seeding, not full re-seeding -- see note in
  simulate_postseason()): Wild Card round (best-of-3) -> Division round
  (best-of-5) -> Championship round (best-of-7) -> World Series
  (best-of-7). Each series game uses log5 between the two teams' final
  simulated win% for that season.

CAVEATS WORTH KNOWING
-----------------------
- No opponent-specific schedule for teams other than Seattle (see
  above).
- Fixed bracket seeding rather than MLB's actual re-seeding-after-each-
  round format -- a real simplification for a personal project, not
  tournament-official.
- Doesn't model day-to-day factors: injuries, hot/cold streaks,
  specific starting pitchers, etc. -- same limitation as any log5-based
  simulation.
- Given all of the above, treat the postseason-round percentages as
  directionally useful, not precise probabilities.

Usage:
    python monte_carlo.py
    python monte_carlo.py --sims 5000
"""

import argparse
import random
import pandas as pd

from data_builder import build_all
from team_analyzer import analyze_team


HOME_FIELD_BUMP_FALLBACK = 0.04  # only used if live data is unavailable


def compute_home_field_bump(data: dict) -> float:
    """
    Computes the real, current league-wide home-field advantage from
    actual home/road win-loss splits (data["standings"]["expanded"]'s
    Home/Road columns), instead of a flat guessed constant.

    Previously HOME_FIELD_BUMP was hardcoded to 0.04 -- a reasonable-
    sounding round number, but not derived from anything. Real 2026
    data gives 0.026, meaningfully smaller (about 35% less) than the
    guess. Same pattern as every other hardcoded-constant bug fixed
    tonight: prefer computing from live data over a plausible-looking
    fixed number, since the fixed number silently drifts from reality
    and there's no way to tell just by looking at it.
    """
    expanded = data.get("standings", {}).get("expanded")
    if expanded is None or expanded.empty or "Home" not in expanded.columns:
        print("  [warn] no live home/road split data -- using fallback "
              f"home field bump of {HOME_FIELD_BUMP_FALLBACK}")
        return HOME_FIELD_BUMP_FALLBACK

    def _parse(s):
        try:
            w, l = str(s).split("-")
            return int(w), int(l)
        except (ValueError, AttributeError):
            return None, None

    total_hw = total_hl = total_rw = total_rl = 0
    for _, row in expanded.iterrows():
        hw, hl = _parse(row.get("Home"))
        rw, rl = _parse(row.get("Road"))
        if hw is not None:
            total_hw += hw
            total_hl += hl
        if rw is not None:
            total_rw += rw
            total_rl += rl

    if (total_hw + total_hl) == 0:
        return HOME_FIELD_BUMP_FALLBACK

    home_pct = total_hw / (total_hw + total_hl)
    bump = home_pct - 0.5
    print(f"  [home field] live league-wide home win%: {home_pct:.4f} "
          f"(bump: {bump:+.4f}, vs old hardcoded {HOME_FIELD_BUMP_FALLBACK:+.4f})")
    return bump
MY_TEAM = "Seattle Mariners"


def log5(team_pct: float, opp_pct: float) -> float:
    num = team_pct - (team_pct * opp_pct)
    den = team_pct + opp_pct - (2 * team_pct * opp_pct)
    if den <= 0:
        return team_pct
    return num / den


def build_game_probabilities(data: dict, sea_win_pct: float,
                             home_field_bump: float = None) -> list:
    """Seattle's per-game win probabilities using the real remaining
    schedule (log5 + home/away vs each actual opponent).

    Returns a list of (probability, opponent_full_name) tuples instead
    of bare probabilities -- the opponent name is needed downstream to
    correlate head-to-head results against division rivals (see
    simulate_regular_season()'s handling of games vs Texas/Houston).

    `home_field_bump` should come from compute_home_field_bump(data) --
    falls back to HOME_FIELD_BUMP_FALLBACK if not provided."""
    if home_field_bump is None:
        home_field_bump = HOME_FIELD_BUMP_FALLBACK

    # BUG FIX 2026-08-26: same issue as simulator.py's
    # compute_schedule_difficulty() -- data["schedule"]["remaining"] alone
    # excludes the next 7 games (tracked separately as "next7" for a
    # checklist display), silently undercounting the real remaining
    # schedule by 7 games. Combine both to get the true full schedule.
    remaining_only = data["schedule"]["remaining"]
    next7 = data["schedule"].get("next7")
    if next7 is not None and not next7.empty:
        remaining = pd.concat([next7, remaining_only], ignore_index=True)
    else:
        remaining = remaining_only

    all_teams = data["standings"]["all_teams"]

    try:
        from recommender import TEAM_ABBR
        abbr_lookup = {v.upper(): k for k, v in TEAM_ABBR.items()}
    except ImportError:
        abbr_lookup = {}

    name_to_pct = dict(zip(
        all_teams["Tm"].str.lower(),
        pd.to_numeric(all_teams["W-L%"], errors="coerce")
    ))

    probs = []
    for _, row in remaining.iterrows():
        opp_key = str(row.get("Opp", "")).strip().upper()
        home_away = str(row.get("home_away", "")).strip().lower()
        full_name = abbr_lookup.get(opp_key)
        opp_pct = name_to_pct.get(full_name) if full_name else None
        if opp_pct is None or pd.isna(opp_pct):
            opp_pct = 0.500

        p = log5(sea_win_pct, opp_pct)
        if home_away == "home":
            p += home_field_bump
        elif home_away == "away":
            p -= home_field_bump
        probs.append((max(0.05, min(0.95, p)), full_name))

    return probs


def get_sea_true_talent_pct(data: dict) -> float:
    """
    Seattle's forward-looking win% -- NOT the flat season-to-date record.

    Uses simulator.py's own IL-return-adjusted RS/G and RA/G (its
    "IL Returns Only" scenario) run through the Pythagorean formula.
    This matters a lot: Seattle's season-to-date win% includes months of
    games played WITHOUT Crawford/Donovan/Brash/Criswell/Vargas -- their
    actual current roster health, once those players are back, projects
    to meaningfully better run-scoring/prevention rates than the full-
    season average reflects (simulator.py's own scenario output shows
    this: RS/G 3.95->4.03, RA/G 4.34->4.13 once IL returns are applied).
    Using the flat season record as "team strength" for every remaining
    game -- as an earlier version of this script did -- ignores that
    entirely and was likely the single biggest reason this model's
    playoff odds ran far below other public projections (e.g. ESPN).
    """
    import simulator as sim_mod

    standings = data.get("standings", {})
    sea_row = data["standings"]["all_teams"]
    sea_row = sea_row[sea_row["Tm"] == MY_TEAM].iloc[0]
    w, l = int(sea_row["W"]), int(sea_row["L"])

    sim_mod.CURRENT_W = w
    sim_mod.CURRENT_L = l
    sim_mod.GAMES_REMAINING = 162 - w - l
    ov_bat = data.get("overview", {}).get("batting")
    ov_pit = data.get("overview", {}).get("pitching")
    # reuse the same RS_G/RA_G live-update logic main.py already does
    try:
        if ov_bat is not None and not ov_bat.empty:
            tm = next((c for c in ["Tm", "Team"] if c in ov_bat.columns), None)
            if tm:
                sea = ov_bat[ov_bat[tm].str.contains("Seattle", na=False)]
                if not sea.empty:
                    r = pd.to_numeric(sea["R"].values[0], errors="coerce")
                    g = pd.to_numeric(sea["G"].values[0], errors="coerce")
                    if r and g:
                        sim_mod.CURRENT_RS_G = round(r / g, 2)
        if ov_pit is not None and not ov_pit.empty:
            tm = next((c for c in ["Tm", "Team"] if c in ov_pit.columns), None)
            if tm:
                sea = ov_pit[ov_pit[tm].str.contains("Seattle", na=False)]
                if not sea.empty:
                    for col in ["RA", "R", "RA9"]:
                        if col in sea.columns:
                            ra = pd.to_numeric(sea[col].values[0], errors="coerce")
                            g = pd.to_numeric(sea["G"].values[0], errors="coerce")
                            if ra and g:
                                sim_mod.CURRENT_RA_G = round(ra / g, 2)
                                break
    except Exception as e:
        print(f"  [warn] could not refresh RS_G/RA_G for true-talent calc: {e}")

    real_schedule = compute_schedule_difficulty_for_simulator(data)
    il_scenario = sim_mod._build_scenario(
        "IL Returns Only (true talent estimate)",
        acquisitions=[], il_returns=None, include_luck=False,
        schedule=real_schedule,
    )
    il_adjusted_pct = il_scenario["win_pct"]

    # CONSISTENCY FIX (added 2026-08-26): every rival team's true-talent
    # estimate now blends in their real last-30-games record (see
    # _blended_true_talent_pct()), but Seattle's own number here was
    # still purely the IL-adjusted Pythagorean estimate with no recent-
    # form component at all -- meaning a genuine hot or cold Mariners
    # stretch wasn't affecting Seattle's own strength the way it now
    # affects every other team's. Blending it in here too, same 60/40
    # weighting, closes that asymmetry.
    expanded = data["standings"].get("expanded")
    last30_str = None
    if expanded is not None and not expanded.empty and "last30" in expanded.columns:
        row = expanded[expanded["Tm"] == MY_TEAM]
        if not row.empty:
            last30_str = row["last30"].values[0]

    if last30_str:
        last30_pct = _pyth_pct(last30_str, il_adjusted_pct)
        blended_pct = (last30_pct * 0.60) + (il_adjusted_pct * 0.40)
        print(f"  [true talent] flat season win%: {w/(w+l):.3f}  "
              f"IL-adjusted win%: {il_adjusted_pct:.3f}  "
              f"last30: {last30_str} ({last30_pct:.3f})  "
              f"blended: {blended_pct:.3f}")
        return blended_pct

    print(f"  [true talent] flat season win%: {w/(w+l):.3f}  "
          f"IL-adjusted win%: {il_adjusted_pct:.3f}  "
          f"(no last30 data available -- using IL-adjusted only)")
    return il_adjusted_pct


def compute_schedule_difficulty_for_simulator(data: dict) -> dict:
    """Bridges to simulator.py's own schedule-difficulty function so the
    true-talent calc above uses the same real, current schedule this
    script already computes for Seattle's own game-by-game simulation."""
    import simulator as sim_mod
    return sim_mod.compute_schedule_difficulty(data)


def _pyth_pct(pyth_wl_str: str, fallback: float) -> float:
    """Parses a 'W-L' Pythagorean record string (e.g. '60-72') into a
    win%. Falls back to the given raw win% if parsing fails."""
    try:
        w_str, l_str = str(pyth_wl_str).split("-")
        w, l = int(w_str), int(l_str)
        return w / (w + l) if (w + l) > 0 else fallback
    except (ValueError, AttributeError):
        return fallback


def _blended_true_talent_pct(pyth_wl_str: str, last30_str: str, raw_pct: float) -> float:
    """
    Blends last-30-games win% (recent form) with Pythagorean win%
    (season-long, luck-corrected), 60/40 -- the SAME weighting
    team_analyzer.py's own projected_wins formula already uses for
    Seattle specifically. Applying it to every team, not just Seattle,
    is what's new here: a team on a hot or cold recent streak (like
    Houston's actual recent surge) wasn't being captured for anyone but
    the Mariners before. Falls back gracefully through pyth -> raw if
    last30 can't be parsed, and to raw win% if neither can be parsed.
    """
    pyth_pct = _pyth_pct(pyth_wl_str, raw_pct)
    last30_pct = _pyth_pct(last30_str, pyth_pct)  # same "W-L" string parser works here
    return (last30_pct * 0.60) + (pyth_pct * 0.40)


def simulate_regular_season(data: dict, sea_true_talent_pct: float = None) -> pd.DataFrame:
    """
    One simulated regular season for all 30 teams. Returns a DataFrame
    with columns: Tm, league, division, final_wins, final_pct.

    HEAD-TO-HEAD CORRELATION (added 2026-08-26): Seattle's remaining
    games against Texas and Houston were previously simulated
    completely independently of Texas's/Houston's own simulated
    seasons -- meaning a trial could have Seattle "beat" Houston while
    Houston's own independently-drawn win total didn't reflect that
    loss at all. Those are the same games; a Seattle win in a head-to-
    head game IS a Houston loss, not two unrelated coin flips. This
    matters because it's a real, sizeable chunk of the schedule right
    now (5 of Seattle's 22 remaining games are directly vs TEX/HOU) --
    real reporting on 2026's remaining schedule strength confirms the
    three AL West contenders are bunched together in no small part
    because they keep playing each other down the stretch, which this
    correlation now properly reflects instead of double-counting.

    PYTHAGOREAN TRUE TALENT (added 2026-08-26): every team besides
    Seattle previously projected forward using their raw season win%,
    which bakes in real, sometimes sizeable in-season luck. A live
    check against Baseball-Reference's own official playoff-odds model
    found Seattle rated far higher there (22.0%) than this script was
    producing (~3-4%) even after the fixes above -- and one real,
    concrete reason: BOTH Seattle (60-72 Pythag vs 63-69 actual) AND
    Houston (61-70 Pythag vs 65-66 actual) have been running notably
    lucky this year. Using raw record for Houston while giving Seattle
    a more sophisticated true-talent adjustment was an apples-to-
    oranges comparison that likely overstated Houston's real forward-
    looking strength relative to Seattle's. This doesn't fully close
    the gap to Baseball-Reference's own model (which very likely uses
    real player-level roster projections we don't have access to), but
    it's a legitimate, standard sabermetric correction worth making
    consistently across all 30 teams rather than just one.

    RECENT-FORM BLEND (added 2026-08-26, same day): pure season-long
    Pythagorean win% still misses a team that's genuinely playing better
    or worse RIGHT NOW than its full-season average suggests (a hot
    streak, a key player back from injury, etc.) -- exactly the kind of
    thing team_analyzer.py already accounts for when projecting Seattle
    specifically (60% weight on last-30-games record, 40% on season
    pace). That same blending is now applied to every team in this
    league-wide simulation, not just Seattle, using each team's real
    last-30-games record from the standings data. Still not equivalent
    to real player-level projections, but a closer, more current
    estimate of where a team actually stands than a single-season
    average.
    """
    all_teams = data["standings"]["all_teams"].copy()
    all_teams["W"] = pd.to_numeric(all_teams["W"], errors="coerce")
    all_teams["L"] = pd.to_numeric(all_teams["L"], errors="coerce")
    all_teams["W-L%"] = pd.to_numeric(all_teams["W-L%"], errors="coerce")

    # pull pythWL and last30 from the expanded standings table (has both
    # for all 30 teams; all_teams does not) and merge in by team name
    expanded = data["standings"].get("expanded")
    pyth_lookup = {}
    last30_lookup = {}
    if expanded is not None and not expanded.empty:
        if "pythWL" in expanded.columns:
            pyth_lookup = dict(zip(expanded["Tm"], expanded["pythWL"]))
        if "last30" in expanded.columns:
            last30_lookup = dict(zip(expanded["Tm"], expanded["last30"]))

    sea_row = all_teams[all_teams["Tm"] == MY_TEAM].iloc[0]
    sea_pct = sea_true_talent_pct if sea_true_talent_pct is not None else \
              sea_row["W"] / (sea_row["W"] + sea_row["L"])
    sea_probs = build_game_probabilities(data, sea_pct)  # list of (prob, opponent_name)

    # simulate Seattle's own games first, tracking head-to-head results
    # against division rivals separately so they can be correlated below
    sea_wins_added = 0
    h2h_results = {}  # opponent full name -> (seattle_wins, seattle_losses) in this trial
    for p, opp_name in sea_probs:
        sea_won = random.random() < p
        if sea_won:
            sea_wins_added += 1
        if opp_name in ("Texas Rangers", "Houston Astros"):
            w_l = h2h_results.get(opp_name, [0, 0])
            w_l[0 if sea_won else 1] += 1
            h2h_results[opp_name] = w_l

    results = []
    for _, row in all_teams.iterrows():
        w, l = row["W"], row["L"]
        team_name = row["Tm"]

        if team_name == MY_TEAM:
            wins_added = sea_wins_added
        else:
            games_remaining = int(162 - w - l)
            raw_pct = row["W-L%"]
            team_pct = _blended_true_talent_pct(
                pyth_lookup.get(team_name), last30_lookup.get(team_name), raw_pct
            )

            # correlate head-to-head games vs Seattle instead of
            # simulating them independently -- see docstring above
            h2h = h2h_results.get(team_name)
            if h2h:
                sea_wins_vs_them, sea_losses_vs_them = h2h
                h2h_games = sea_wins_vs_them + sea_losses_vs_them
                # this team's wins in those specific games = Seattle's losses in them
                correlated_wins = sea_losses_vs_them
                games_remaining -= h2h_games
                wins_added = correlated_wins + sum(
                    1 for _ in range(max(0, games_remaining))
                    if random.random() < team_pct
                )
            else:
                wins_added = sum(1 for _ in range(games_remaining)
                                 if random.random() < team_pct)

        final_w = w + wins_added
        final_l = 162 - final_w
        results.append({
            "Tm": team_name, "league": row["league"], "division": row["division"],
            "final_wins": final_w, "final_pct": final_w / 162,
        })

    return pd.DataFrame(results)


def determine_playoff_field(season: pd.DataFrame, league: str) -> dict:
    """
    Given one simulated season's final records, determines the playoff
    field for one league: 3 division winners (seeded 1-3 by record,
    #1 gets the bye) and 3 Wild Card teams (seeded 4-6).
    """
    lg = season[season["league"] == league].copy()

    div_winners = (lg.sort_values("final_wins", ascending=False)
                     .groupby("division").head(1))
    div_winners = div_winners.sort_values("final_wins", ascending=False)

    non_winners = lg[~lg["Tm"].isin(div_winners["Tm"])]
    wild_cards = non_winners.sort_values("final_wins", ascending=False).head(3)

    seeds = list(div_winners["Tm"]) + list(wild_cards["Tm"])
    seed_wins = dict(zip(season["Tm"], season["final_pct"]))
    return {"seeds": seeds, "seed_pct": seed_wins}


def simulate_series(team_a: str, team_b: str, pct_lookup: dict, best_of: int) -> str:
    """Simulates a short series game-by-game using log5. Returns the winner."""
    wins_needed = best_of // 2 + 1
    wins_a = wins_b = 0
    pa, pb = pct_lookup[team_a], pct_lookup[team_b]
    p_a_wins_game = log5(pa, pb)
    while wins_a < wins_needed and wins_b < wins_needed:
        if random.random() < p_a_wins_game:
            wins_a += 1
        else:
            wins_b += 1
    return team_a if wins_a == wins_needed else team_b


def simulate_postseason(field: dict) -> dict:
    """
    Runs one league's bracket: WC round (best-of-3) -> Division round
    (best-of-5) -> Championship round (best-of-7). Fixed seeding, not
    full re-seeding (see module docstring). Returns the pennant winner
    plus which round each seed was eliminated in (used for aggregating
    "made playoffs" / "won WC round" / etc. percentages).
    """
    seeds = field["seeds"]  # [1,2,3,4,5,6] by list position
    pct = field["seed_pct"]
    if len(seeds) < 6:
        return {"pennant_winner": None, "results": {}}

    s1, s2, s3, s4, s5, s6 = seeds
    results = {s: "made_playoffs" for s in seeds}

    # Wild Card round: (3 vs 6), (4 vs 5); #1 seed has a bye
    wc_winner_36 = simulate_series(s3, s6, pct, best_of=3)
    wc_winner_45 = simulate_series(s4, s5, pct, best_of=3)
    for s in (wc_winner_36, wc_winner_45):
        results[s] = "won_wc_round"

    # Division round: #1 (bye) vs winner(4v5); #2 vs winner(3v6)
    div_winner_1 = simulate_series(s1, wc_winner_45, pct, best_of=5)
    div_winner_2 = simulate_series(s2, wc_winner_36, pct, best_of=5)
    for s in (div_winner_1, div_winner_2):
        results[s] = "won_division_round"

    # Championship round
    pennant_winner = simulate_series(div_winner_1, div_winner_2, pct, best_of=7)
    results[pennant_winner] = "won_pennant"

    return {"pennant_winner": pennant_winner, "results": results}


def run_monte_carlo(n_sims: int = 5000):
    print("Building data (uses cache if fresh)...")
    data = build_all(2026)
    analysis = analyze_team(data)
    standings = analysis.get("standings", {})
    record = standings.get("record")
    print(f"\nCurrent Mariners record: {record}")

    sea_true_talent_pct = get_sea_true_talent_pct(data)

    print(f"Simulating {n_sims:,} full 30-team seasons + postseason brackets...")
    print("(this is a much bigger computation than before -- may take a bit longer)")

    counters = {
        "made_playoffs": 0, "won_division": 0, "wild_card": 0,
        "won_wc_round": 0, "won_division_round": 0,
        "won_pennant": 0, "won_world_series": 0,
    }

    for i in range(n_sims):
        season = simulate_regular_season(data, sea_true_talent_pct)

        al_field = determine_playoff_field(season, "AL")
        nl_field = determine_playoff_field(season, "NL")

        sea_al_seed = al_field["seeds"].index(MY_TEAM) if MY_TEAM in al_field["seeds"] else None

        if sea_al_seed is not None:
            counters["made_playoffs"] += 1
            if sea_al_seed < 3:
                counters["won_division"] += 1
            else:
                counters["wild_card"] += 1

            al_result = simulate_postseason(al_field)
            nl_result = simulate_postseason(nl_field)
            sea_stage = al_result["results"].get(MY_TEAM, "made_playoffs")

            if sea_stage in ("won_wc_round", "won_division_round", "won_pennant"):
                counters["won_wc_round"] += 1
            if sea_stage in ("won_division_round", "won_pennant"):
                counters["won_division_round"] += 1
            if sea_stage == "won_pennant":
                counters["won_pennant"] += 1
                ws_winner = simulate_series(
                    al_result["pennant_winner"], nl_result["pennant_winner"],
                    {**al_field["seed_pct"], **nl_field["seed_pct"]}, best_of=7
                )
                if ws_winner == MY_TEAM:
                    counters["won_world_series"] += 1

        if (i + 1) % max(1, n_sims // 10) == 0:
            print(f"  ...{i+1:,} / {n_sims:,} simulations complete")

    print("\n" + "=" * 60)
    print(f"MONTE CARLO RESULTS -- {MY_TEAM}")
    print("=" * 60)
    print(f"  Make playoffs (any path):  {counters['made_playoffs']/n_sims*100:5.1f}%")
    print(f"  Win division:              {counters['won_division']/n_sims*100:5.1f}%")
    print(f"  Wild Card berth:           {counters['wild_card']/n_sims*100:5.1f}%")
    print(f"  Win Wild Card round:       {counters['won_wc_round']/n_sims*100:5.1f}%")
    print(f"  Win Division round:        {counters['won_division_round']/n_sims*100:5.1f}%")
    print(f"  Win pennant (Champ round): {counters['won_pennant']/n_sims*100:5.1f}%")
    print(f"  Win World Series:          {counters['won_world_series']/n_sims*100:5.1f}%")
    print("=" * 60)
    return counters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full league-wide Monte Carlo playoff simulator")
    parser.add_argument("--sims", type=int, default=5000,
                        help="Number of simulated seasons (default: 5000 -- this "
                             "version is much heavier per-sim than before)")
    args = parser.parse_args()
    run_monte_carlo(args.sims)
