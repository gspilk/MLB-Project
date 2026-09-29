# Seattle Mariners Stretch-Run Analyzer

A Python pipeline that scrapes live 2026 MLB data — batting, pitching, standings, Statcast, payroll, and minor-league affiliate stats — and turns it into player grades, roster decisions, rest-of-season projections, and free-agent/prospect recommendations, all validated against real professional benchmarks.

## Project description

**Data**: 2026 season — batting, pitching, standings, and payroll, scraped live from Baseball Reference, Baseball Savant (Statcast), MLB Trade Rumors, and Spotrac.

**Core pipeline** (`python main.py`):
- Gets live standings across all 30 MLB teams
- Gets the Mariners' full season schedule (completed games + upcoming)
- Gets batting stats and advanced Statcast metrics, league-wide
- Gets pitching stats and advanced Statcast metrics, league-wide
- Gets the Mariners' own batting, pitching, fielding, and Statcast data specifically
- Grades every Mariners player and drives roster decisions — keep, DFA, option to AAA, monitor — based on real performance, not gut feel
- Grades batters on OPS, xwOBA, and luck (xwOBA vs. actual production gap); grades pitchers on ERA, WHIP, K/9, xwOBA against, and luck
- Runs a rest-of-season simulation (live record, run differential, luck correction, real schedule difficulty, IL-return timelines) to project the final record and playoff odds
- Runs a standalone Monte Carlo simulation (thousands of simulated seasons + postseason brackets) for a higher-resolution playoff-odds estimate

**Additional tools** (run separately from the core pipeline):
- **Free-agent recommendations** (`free_agent.py`, `free_agent_stats_matcher.py`, `free_agent_recommendations.py`) — scrapes the real MLB Trade Rumors free-agent class, matches players to league-wide stats, and scores every free agent *and* internal roster candidate against the Mariners' actual incumbent at each position, weighted by need and reliability (sample-size shrinkage)
- **Minor-league prospect tracking** (`minor_league_stats.py`, `minor_league_shortlist.py`) — pulls the full organizational batting/pitching stats and outside top-prospect rankings, filtered down to real, near-MLB-ready talent (AA/AAA, meaningful sample size, excluding anyone already on the active roster)
- **Injury cost tracking** (`injury_cost_track.py`) — role-aware estimate of real wins lost to injury, weighted by leverage for relievers and games-missed for everyday players
- **Positional WAR review** (`position_summary_review.py`) and **roster decision review** — rolls up combined WAR by position and flags real underperforming roster spots
- **WAR component breakdowns** (`batting_value_breakdown.py`, `pitching_value_breakdown.py`, `fielding_value_breakdown.py`, `combined_valuebreakdown.py`) — splits each player's value into batting/baserunning/fielding/positional components
- **Payroll tracking** (`spotrac_scraper.py`) — league-wide payroll and cash-commitment comparisons
- **Platoon lineup optimizer** (`platoon_lineup_optimizer.py`) — vs-LHP/vs-RHP lineup construction from real split data
- **Trade tracking** (`trade_return_package.py`, `trade_cost_package.py`) — what the team gave up vs. got back on specific trades, including live stats for players now on other teams
- **Historical validation** (`fifty_four_percent_test.py`, `multi_year_comparison.py`) — tested the front office's public "54% win rate" playoff-odds claim against real multi-year MLB data to find the actual empirical win threshold, and tracks player/team trends year over year

## What makes it different

- **Pulls from multiple independent sources, not one.** Baseball Reference for traditional stats and standings, Baseball Savant for Statcast (xwOBA, Barrel%, Whiff%), MLB Trade Rumors for free agency, Spotrac for payroll — cross-referenced and reconciled by player identity, since sources format names differently (handled centrally in `name_matching.py`).
- **Recommendations come from real statistical need, not a fixed wishlist.** Both the free-agent tool and the (now-dated, see note below) in-season trade-target logic identify the team's actual weak spots by position and stat, league-wide, rather than starting from a hand-picked list.
- **Validated against real, independent models, not just self-consistent.** The playoff-odds simulator was checked against both ESPN's and Baseball-Reference's own published odds. That process surfaced and fixed a real bug that had been silently excluding a chunk of the remaining schedule from every projection — the tool catching its own mistakes rather than just producing a number and trusting it. The same discipline (build a small debug script to confirm real page structure before trusting a scraper) has been used to catch and fix real bugs in almost every scraper in this project.

> **Note on trade targets:** `recommender.py`'s seller-based trade-target logic and the simulator's "deadline simulation" step were built for the in-season trade deadline. That's now passed for 2026 — the code still runs, but the free-agent/internal-candidate pipeline is the more current answer to "how does this team improve." Worth deciding whether to keep, retire, or repurpose the deadline-era code for next year's stretch run.

## Steps (core pipeline)

```
STEP 1/6 -- Building data...
STEP 2/6 -- Analyzing team...
STEP 3/6 -- Grading players...
STEP 4/6 -- Generating recommendations...

[recommend] Generating recommendations...
  [targets] filtering batters to positions the team actually needs: [...]
  [targets] N teams flagged as likely sellers based on live standings
  [targets] N batter targets found
  [targets] N pitcher targets found
[recommend] Done.

STEP 5/6 -- Running rest-of-season simulation...
STEP 6/6 -- Generating Excel report...
```

Run it with:
```
python main.py --refresh
```

Output is a formatted Excel report (Team Summary, Player Grades, Statcast, Stats to Improve, Trade Targets, Schedule, Simulation) plus an appended row in `data/history.parquet` for tracking trends over time in Power BI.

For a standalone, higher-resolution playoff-odds estimate:
```
python montecarlo.py --sims 5000
```

For tracking a specific trade's principals over time:
```
python trade_return_package.py
python trade_cost_package.py
```

For free-agent and internal roster recommendations:
```
python free_agent_recommendations.py
```

For real, near-MLB-ready minor-league talent (excludes anyone already on the active roster):
```
python minor_league_shortlist.py
```

## Future opportunities

- Fold minor-league shortlist candidates into the free-agent recommendation scoring, so "sign a free agent," "promote a prospect," and "play the bench guy we already have" are one ranked list instead of three separate tools
- Compare against previous seasons (multi-year player and team trend analysis — partially built, see `multi_year_comparison.py` and `player_year_comparison.py`)
- Backtest the rest-of-season projection system against the actual final outcome once the season ends
- Extend Spotrac payroll data into genuine value-per-dollar analysis (real cost data now exists; the value-per-dollar comparison itself doesn't yet)
- Decide the fate of the trade-deadline-specific code (see note above) before next season's stretch run
