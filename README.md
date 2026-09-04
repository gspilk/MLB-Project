# Seattle Mariners Stretch-Run Analyzer

A Python pipeline that scrapes live 2026 MLB data — batting, pitching, and standings — and turns it into player grades, roster recommendations, and a Monte Carlo playoff-odds simulator, all validated against real professional benchmarks.

## Project description

**Data**: 2026 season — batting, pitching, and standings, scraped live from Baseball Reference and Baseball Savant (Statcast).

**What it does**:
- Gets live standings across all 30 MLB teams
- Gets the Mariners' full season schedule (completed games + upcoming)
- Gets batting stats and advanced Statcast metrics, league-wide
- Gets pitching stats and advanced Statcast metrics, league-wide
- Gets the Mariners' own batting, pitching, and Statcast data specifically
- Uses that data to grade every Mariners player and drive roster decisions — keep, DFA, option to AAA, monitor — based on real performance, not gut feel
- Grades batters on OPS, xwOBA, and luck (xwOBA vs. actual production gap); grades pitchers on ERA, WHIP, K/9, xwOBA against, and luck
- Runs a Monte Carlo simulation (thousands of simulated seasons + postseason brackets) to estimate real playoff odds
- Tracks specific trades — what the team gave up vs. what it got back, including live stats for players now on other teams

## What makes it different

- **Pulls from multiple independent sources, not one.** Baseball Reference for traditional stats and standings, Baseball Savant for Statcast (xwOBA, Barrel%, Whiff%) — cross-referenced and reconciled by player identity, since the two sources format names differently.
- **Generates trade targets from real statistical needs, not a fixed wishlist.** It identifies the team's actual weak spots (by position, by stat, ranked league-wide), filters for players meeting a real performance bar on teams likely to sell, and surfaces a ranked list — updated automatically as the season and standings change.
- **Validated against real, independent models, not just self-consistent.** The playoff-odds simulator was checked against both ESPN's and Baseball-Reference's own official published odds. That process surfaced and fixed a real bug that had been silently excluding a chunk of the remaining schedule from every projection — a good example of the tool catching its own mistakes rather than just producing a number and trusting it.

## Steps (pipeline)

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

STEP 5/6 -- Running deadline simulation...
STEP 6/6 -- Generating Excel report...
```

Run it with:
```
python main.py --refresh
```

Output is a formatted Excel report (Team Summary, Player Grades, Statcast, Stats to Improve, Trade Targets, Schedule, Deadline Simulation) plus an appended row in `data/history.parquet` for tracking trends over time in Power BI.

For a standalone, higher-resolution playoff-odds estimate:
```
python montecarlo.py --sims 5000
```

For tracking a specific trade's principals over time:
```
python trade_return_package.py
python trade_cost_package.py
```

## Future opportunities

- Compare against previous seasons (multi-year player and team trend analysis — partially built, see `multi_year_comparison.py` and `player_year_comparison.py`)
- Offseason mode: free-agent decision support (who to keep, re-sign, or let walk) once real contract/free-agent data becomes available at season's end
- Backtesting the projection system against actual final outcomes once the season ends
- Real cost/salary data, to move from "who performs well" toward genuine value-per-dollar analysis