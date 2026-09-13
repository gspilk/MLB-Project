"""
fifty_four_percent_test.py
Real, historical test of the "Dipoto 54%" question: of every MLB team
across 2019-2025 that finished with a win% close to .540 (Dipoto's
stated target), how many actually made the playoffs?

Uses this project's existing get_standings() infrastructure to pull
REAL win-loss for all 30 teams across each season -- no invented
numbers. Playoff participants per year are a real, sourced, hand-
maintained list (see PLAYOFF_TEAMS below) since that's genuine
historical fact, not something derivable from a single standings
scrape.

YEARS INCLUDED: 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025.
  - 2020 excluded deliberately: the COVID-shortened 60-game season had
    an anomalous 16-team playoff format (more than half the league
    qualified), which would badly distort this specific question.
  - 2026 excluded deliberately: the season isn't over yet as of this
    tool being built (mid-September, games still being played) -- you
    can't test "did they make the playoffs" on a field that hasn't
    been decided.
  - 2016-2019/2021 used the OLDER 10-team format; 2022-2025 used the
    CURRENT 12-team format. Both are kept in the same table (with the
    format visible) and analyzed SEPARATELY in the banded breakdown,
    since the bar for "making it" genuinely differs between the two
    eras -- fewer playoff spots under the old format means a given
    win% was less safe than the same win% is today.

Usage:
    python fifty_four_percent_test.py
"""

import pandas as pd
from standings_scraper import get_standings

YEARS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
FORMAT_BY_YEAR = {2016: "10-team", 2017: "10-team", 2018: "10-team",
                  2019: "10-team", 2021: "10-team",
                  2022: "12-team", 2023: "12-team",
                  2024: "12-team", 2025: "12-team"}

# Real, sourced playoff participants per year (full team names, matching
# bbref's own naming). NOT derived from a live scrape -- this is genuine
# historical fact, hand-verified against real reporting for each year.
PLAYOFF_TEAMS = {
    2016: {  # 10-team format
        "Cleveland Indians", "Texas Rangers", "Boston Red Sox",        # AL div
        "Toronto Blue Jays", "Baltimore Orioles",                      # AL WC
        "Chicago Cubs", "Los Angeles Dodgers", "Washington Nationals", # NL div
        "New York Mets", "San Francisco Giants",                       # NL WC
    },
    2017: {  # 10-team format
        "Cleveland Indians", "Houston Astros", "Boston Red Sox",       # AL div
        "New York Yankees", "Minnesota Twins",                        # AL WC
        "Los Angeles Dodgers", "Chicago Cubs", "Washington Nationals", # NL div
        "Arizona Diamondbacks", "Colorado Rockies",                    # NL WC
    },
    2018: {  # 10-team format
        "Boston Red Sox", "Houston Astros", "Cleveland Indians",       # AL div
        "New York Yankees", "Oakland Athletics",                       # AL WC
        "Atlanta Braves", "Milwaukee Brewers", "Los Angeles Dodgers",  # NL div
        "Chicago Cubs", "Colorado Rockies",                            # NL WC
    },
    2019: {  # 10-team format
        "New York Yankees", "Minnesota Twins", "Houston Astros",       # AL div
        "Tampa Bay Rays", "Oakland Athletics",                        # AL WC
        "Atlanta Braves", "St. Louis Cardinals", "Los Angeles Dodgers",  # NL div
        "Washington Nationals", "Milwaukee Brewers",                   # NL WC
    },
    2021: {  # 10-team format
        "Tampa Bay Rays", "Chicago White Sox", "Houston Astros",       # AL div
        "Boston Red Sox", "New York Yankees",                         # AL WC
        "San Francisco Giants", "Milwaukee Brewers", "Los Angeles Dodgers",  # NL div
        "Atlanta Braves", "St. Louis Cardinals",                       # NL WC
    },
    2022: {  # first 12-team format year
        "New York Yankees", "Cleveland Guardians", "Houston Astros",   # AL div
        "Toronto Blue Jays", "Seattle Mariners", "Tampa Bay Rays",     # AL WC
        "Atlanta Braves", "St. Louis Cardinals", "Los Angeles Dodgers", # NL div
        "New York Mets", "San Diego Padres", "Philadelphia Phillies",  # NL WC
    },
    2023: {
        "Baltimore Orioles", "Minnesota Twins", "Houston Astros",      # AL div
        "Tampa Bay Rays", "Texas Rangers", "Toronto Blue Jays",        # AL WC
        "Atlanta Braves", "Milwaukee Brewers", "Los Angeles Dodgers",  # NL div
        "Philadelphia Phillies", "Miami Marlins", "Arizona Diamondbacks",  # NL WC
    },
    2024: {
        "Baltimore Orioles", "Cleveland Guardians", "Houston Astros",  # AL div
        "New York Yankees", "Kansas City Royals", "Detroit Tigers",    # AL WC
        "Philadelphia Phillies", "Milwaukee Brewers", "Los Angeles Dodgers",  # NL div
        "San Diego Padres", "New York Mets", "Atlanta Braves",         # NL WC
    },
    2025: {
        "Toronto Blue Jays", "Seattle Mariners", "Cleveland Guardians",  # AL div
        "New York Yankees", "Boston Red Sox", "Detroit Tigers",        # AL WC
        "Milwaukee Brewers", "Philadelphia Phillies", "Los Angeles Dodgers",  # NL div
        "Chicago Cubs", "San Diego Padres", "Cincinnati Reds",         # NL WC
    },
}


# Real, sourced World Series champion per year -- for testing the
# related question "of teams that WON it all, what was their real
# regular-season win total." The postseason itself is a short, high-
# variance tournament, so this can (and does) include some real
# surprises below the "safe to make it" threshold found above.
WORLD_SERIES_CHAMPION = {
    2016: "Chicago Cubs",
    2017: "Houston Astros",
    2018: "Boston Red Sox",
    2019: "Washington Nationals",
    2021: "Atlanta Braves",
    2022: "Houston Astros",
    2023: "Texas Rangers",
    2024: "Los Angeles Dodgers",
    2025: "Los Angeles Dodgers",
}


def build_dataset() -> list:
    rows = []
    for year in YEARS:
        print(f"Fetching {year} standings...")
        try:
            _, expanded = get_standings(year)
        except Exception as e:
            print(f"  [warn] could not fetch {year}: {e}")
            continue

        for _, row in expanded.iterrows():
            name = row.get("Tm")
            w = pd.to_numeric(row.get("W"), errors="coerce")
            l = pd.to_numeric(row.get("L"), errors="coerce")
            if pd.isna(w) or pd.isna(l) or (w + l) == 0:
                continue
            win_pct = w / (w + l)
            made_playoffs = name in PLAYOFF_TEAMS.get(year, set())
            rows.append({
                "year": year, "team": name, "W": int(w), "L": int(l),
                "win_pct": round(win_pct, 3), "format": FORMAT_BY_YEAR[year],
                "made_playoffs": made_playoffs,
            })
    return rows


def run_test(rows: list, band: tuple = (0.525, 0.555)) -> None:
    """
    band: the win% window counted as "near 54%". Default (.525, .555)
    is roughly 85-90 wins over 162 games -- a defensible +/-1.5%
    window around Dipoto's stated .540 target, wide enough to get a
    real sample size without being so wide it stops meaning anything.
    """
    low, high = band
    near_target = [r for r in rows if low <= r["win_pct"] <= high]

    print("\n" + "=" * 70)
    print(f"TEAMS THAT FINISHED NEAR A 54% WIN RATE ({low:.3f}-{high:.3f}), "
          f"2019-2025 (excl. 2020, 2026)")
    print("=" * 70)
    for r in sorted(near_target, key=lambda x: (x["year"], -x["win_pct"])):
        made = "MADE PLAYOFFS" if r["made_playoffs"] else "missed"
        print(f"  {r['year']} {r['team']:<24} {r['W']}-{r['L']} "
              f"({r['win_pct']:.3f})  [{r['format']}]  -> {made}")

    total = len(near_target)
    made = sum(1 for r in near_target if r["made_playoffs"])
    print("\n" + "-" * 70)
    if total:
        pct = round(made / total * 100, 1)
        print(f"  Of {total} team-seasons that finished near 54%: "
              f"{made} made the playoffs ({pct}%), {total - made} missed.")
    else:
        print("  No team-seasons found in this window.")
    print("=" * 70 + "\n")


def run_banded_analysis(rows: list, fmt: str) -> None:
    """
    Breaks every team-season into win% bands and shows the real,
    empirical playoff hit rate for each -- to find where the curve
    actually goes from "good bet" to "essentially guaranteed," rather
    than just checking one target number. Run once per format, since
    the old 10-team era had fewer playoff spots than today's 12-team
    era, and blending them would hide that real difference.
    """
    bands = [
        (0.475, 0.500), (0.500, 0.525), (0.525, 0.550), (0.550, 0.575),
        (0.575, 0.600), (0.600, 0.625), (0.625, 1.000),
    ]

    print("\n" + "=" * 70)
    print(f"REAL HIT RATE BY WIN% BAND -- {fmt.upper()} ERA")
    print("=" * 70)
    print(f"  {'Win% band':<16} {'Approx wins':<14} {'Made':>5} {'Total':>6} {'Hit rate':>9}")
    print("  " + "-" * 60)

    for low, high in bands:
        band_rows = [r for r in rows if low <= r["win_pct"] < high
                    and r["format"] == fmt]
        total = len(band_rows)
        made = sum(1 for r in band_rows if r["made_playoffs"])
        approx_wins = f"{round(low*162)}-{round(high*162)}"
        if total:
            rate = round(made / total * 100, 1)
            print(f"  {low:.3f}-{high:.3f}   {approx_wins:<14} {made:>5} "
                  f"{total:>6} {rate:>8.1f}%")
        else:
            print(f"  {low:.3f}-{high:.3f}   {approx_wins:<14} {'--':>5} "
                  f"{'0':>6} {'n/a':>9}")

    print("=" * 70 + "\n")


def run_champion_analysis(rows: list) -> None:
    """
    Of the teams that actually WON the World Series each year, what was
    their real regular-season win total? Different question than "safe
    to make the playoffs" -- once you're in, the postseason is a short,
    high-variance tournament, so this can include real surprises.
    """
    champ_rows = []
    for year, champ_name in WORLD_SERIES_CHAMPION.items():
        match = [r for r in rows if r["year"] == year and r["team"] == champ_name]
        if match:
            champ_rows.append(match[0])
        else:
            print(f"  [warn] no data found for {year} champion ({champ_name})")

    print("\n" + "=" * 70)
    print("WORLD SERIES CHAMPIONS -- REAL REGULAR-SEASON WIN TOTALS")
    print("=" * 70)
    for r in sorted(champ_rows, key=lambda x: x["W"]):
        print(f"  {r['year']} {r['team']:<24} {r['W']}-{r['L']} "
              f"({r['win_pct']:.3f})  [{r['format']}]")

    if champ_rows:
        wins = [r["W"] for r in champ_rows]
        avg = round(sum(wins) / len(wins), 1)
        print("\n" + "-" * 70)
        print(f"  Range: {min(wins)}-{max(wins)} wins")
        print(f"  Average: {avg} wins across {len(champ_rows)} champions")
        below_safe = [r for r in champ_rows if r["W"] < 90]
        if below_safe:
            names = ", ".join(f"{r['year']} {r['team']} ({r['W']}-{r['L']})"
                              for r in below_safe)
            print(f"  Won it all with FEWER than the 90-win 'safe' "
                  f"threshold found above: {names}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    rows = build_dataset()
    run_test(rows)
    run_banded_analysis(rows, "10-team")
    run_banded_analysis(rows, "12-team")
    run_champion_analysis(rows)