"""
offseason_report.py
The single combined offseason report -- runs every already-built,
already-verified analysis in this project in one pass and prints them
in a logical order, instead of running four separate scripts and
mentally stitching the results together yourself.

WHY THIS EXISTS: by this point in the project there are four real,
independently-tested tools that each answer a different piece of "what
should the Mariners do this offseason":
  1. team_diagnosis.py       -- WHAT's wrong, ranked by severity
  2. year_over_year_decomposition.py -- WHY it changed from last season,
     broken down per player
  3. free_agent_recommendations.py -- WHO fixes it, ranked by fit AND
     by value-for-cost, with real Spotrac pricing
  4. fielding_targets.py     -- a fielding-FIRST cut of the same market,
     since fielding is the single biggest diagnosed weakness and the
     recommendation engine only ever treats defense as a minor bonus,
     not the primary question
This script does not reimplement any of that logic -- it imports the
real, tested functions from each module and calls them in sequence,
building the shared expensive inputs (league/Mariners data, the free
agent list, the enriched/matched pool, real market data) ONCE and
passing them to every section, instead of each module rebuilding them
independently (which is what running the four scripts separately does).

Usage:
    python offseason_report.py
    python offseason_report.py --from-season 2025 --to-season 2026
"""
import argparse
import os
import pandas as pd

from data_builder import build_all
from free_agent import get_mlbtr_free_agents
from free_agent_stats_matcher import enrich_free_agents
from free_agent_market import (
    load_market_data, attach_market_data, print_position_conflicts,
)
from team_diagnosis import diagnose_team, get_war_distribution, _team_fielding
from mariners_stats import get_seattle_stats
from year_over_year_decomposition import compute_all_decompositions, _print_block
from free_agent_recs import (
    load_diagnosed_needs, build_mariners_incumbents, build_internal_candidates,
    add_upgrade_deltas, compute_recommendation_score, print_recommendations,
    save_top_upgrade_chart, print_depth_chart, find_bench_candidates,
    print_bench_candidates, generate_comparison_html, _get_current_roster_names,
)
from fielding_targets import (
    get_mariners_fielding_weak_spots, get_full_team_fielding_ledger,
    print_full_team_fielding_ledger, get_fielding_targets, print_fielding_report,
)

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "data", "exports")


def _section(title: str):
    print("\n\n" + "#" * 100)
    print(f"# {title}")
    print("#" * 100)


def export_report_csvs(needs: pd.DataFrame, scored: pd.DataFrame,
                       full_ledger: pd.DataFrame, weak_spots: pd.DataFrame,
                       targets: dict, decompositions: dict) -> list:
    """
    Writes every table this report produces to flat CSVs under
    data/exports/, so a tool downstream of this project (Power BI, Excel,
    anything that isn't this terminal) has something real to point at --
    no chart library commitment implied or required to get this far.
    Returns the list of paths actually written (a table with nothing in
    it is skipped rather than writing an empty, confusing file).
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    written = []

    def _write(df: pd.DataFrame, filename: str):
        if df is None or df.empty:
            return
        path = os.path.join(EXPORT_DIR, filename)
        df.to_csv(path, index=False)
        written.append(path)

    _write(needs, "diagnosed_needs.csv")
    _write(scored, "free_agent_scored.csv")
    _write(full_ledger, "fielding_ledger.csv")
    _write(weak_spots, "fielding_weak_spots.csv")

    # fielding_targets.py's targets dict is {position_group: DataFrame} --
    # each options frame already carries its own "position_group" column
    # (inherited from the enriched free agent pool), so concatenating is
    # safe and gives one real, Power-BI-friendly table instead of N tiny
    # per-position files.
    if targets:
        _write(pd.concat(targets.values(), ignore_index=True), "fielding_targets.csv")

    # year-over-year decompositions: each entry's "players" table gets
    # its own file, named after the stat it decomposes (e.g.
    # "year_over_year_Batting_BA.csv") -- real per-player rows, not the
    # team-level summary numbers (those are just a handful of scalars,
    # already visible in the terminal output and not worth a CSV).
    for title, (res, _is_rate) in decompositions.items():
        safe_name = title.replace(" ", "_").replace("(", "").replace(")", "")
        _write(res.get("players"), f"year_over_year_{safe_name}.csv")

    return written


def run_report(from_season: int = 2025, to_season: int = 2026):
    # ── shared, expensive inputs -- built ONCE, reused by every section ──
    print("Building league + Mariners data (uses cache if fresh)...")
    data = build_all(to_season)

    print("Getting the real free agent list...")
    fa = get_mlbtr_free_agents()
    enriched_all = enrich_free_agents(fa, data)
    current_roster = _get_current_roster_names(data)

    print("Loading real Spotrac free agent market data (cost/contract info)...")
    market = load_market_data()

    # ────────────────────────────────────────────────────────────────────
    _section(f"1. ROSTER DIAGNOSIS ({to_season}) -- WHAT's wrong, ranked by severity")
    # ────────────────────────────────────────────────────────────────────
    needs = diagnose_team()
    if needs.empty:
        print("No diagnosis produced -- check that batting/pitching data loaded.")
    else:
        print(needs.to_string(index=False))

    war = get_war_distribution()
    if war:
        print("\n-- WAR distribution (context, not a weakness/strength verdict) --")
        for k, v in war.items():
            print(f"  {k}: {v}")

    # ────────────────────────────────────────────────────────────────────
    _section(f"2. YEAR-OVER-YEAR DECOMPOSITION: {from_season} -> {to_season} "
              f"-- WHY it changed, per player")
    # ────────────────────────────────────────────────────────────────────
    stats_a = get_seattle_stats(from_season)
    stats_b = get_seattle_stats(to_season)
    decompositions = compute_all_decompositions(stats_a, stats_b, str(from_season), str(to_season))
    for title, (res, is_rate) in decompositions.items():
        _print_block(title, res, str(from_season), str(to_season), is_rate=is_rate)

    # ────────────────────────────────────────────────────────────────────
    _section("3. FREE AGENT RECOMMENDATIONS -- WHO fixes it (fit + real cost/value)")
    # ────────────────────────────────────────────────────────────────────
    incumbents = build_mariners_incumbents(data)
    for group, info in incumbents.items():
        stat = info.get("OPS", info.get("ERA"))
        print(f"  {group}: {info['name']} ({stat})")

    enriched = enriched_all[~enriched_all["name"].isin(current_roster)].copy()
    excluded = len(enriched_all) - len(enriched)
    if excluded:
        print(f"  [info] excluded {excluded} free agent(s) who are also current "
              f"Mariners from the target pool")

    internal = build_internal_candidates(data, incumbents)
    print(f"\nFound {len(internal)} real internal roster candidates")

    combined = pd.concat([enriched, internal], ignore_index=True)
    upgraded = add_upgrade_deltas(combined, incumbents)
    scored = compute_recommendation_score(upgraded, incumbents, needs)

    if not market.empty:
        print(f"  [ok] {len(market)} real free agents loaded from Spotrac")
        scored = attach_market_data(scored, market)
        matched_cost = scored[scored["Market_Matched"] == True]
        print(f"  {len(matched_cost)} of {len(scored[scored['recommendation_score'].notna()])} "
              f"scored recommendations have a real cost match")

    print_recommendations(scored)
    if "Est_Cost_M" in scored.columns:
        print_recommendations(scored, by_value=True)
        print_position_conflicts(scored)

    html_path = generate_comparison_html(scored, incumbents)
    print(f"\nReal comparison page saved: {html_path}")

    batter_incumbents = {g: info for g, info in incumbents.items() if "OPS" in info}
    weakest = sorted(batter_incumbents.items(), key=lambda kv: kv[1]["OPS"])[:3]
    print("\nReal weakest positions by incumbent performance:")
    for group, info in weakest:
        print(f"  {group}: {info['name']} ({info['OPS']:.3f} OPS)")

    print("\nSaving real chart files for the actual weakest positions...")
    for group, _ in weakest:
        path = save_top_upgrade_chart(upgraded, group)
        if path:
            print(f"  saved: {path}")

    for group, _ in weakest:
        print_depth_chart(scored, incumbents, group)

    bench = find_bench_candidates(enriched)
    print_bench_candidates(bench)

    # ────────────────────────────────────────────────────────────────────
    _section("4. FIELDING-FIRST TARGETS -- the biggest diagnosed weakness, "
              "asked directly")
    # ────────────────────────────────────────────────────────────────────
    real_team_total = None
    fielding_df = data["seattle"].get("fielding", pd.DataFrame())
    if not fielding_df.empty:
        real_team_total = _team_fielding(fielding_df).get("Rtot")

    print_full_team_fielding_ledger(data, real_team_total=real_team_total)

    weak_spots = get_mariners_fielding_weak_spots(data)
    full_ledger = get_full_team_fielding_ledger(data)
    targets = get_fielding_targets(enriched_all, weak_spots, market, current_roster)
    print_fielding_report(weak_spots, targets, needs)

    # ────────────────────────────────────────────────────────────────────
    _section("5. CSV EXPORTS -- for Power BI, Excel, or anything else downstream")
    # ────────────────────────────────────────────────────────────────────
    written = export_report_csvs(needs, scored, full_ledger, weak_spots, targets, decompositions)
    if written:
        print(f"  Wrote {len(written)} CSV file(s) to {EXPORT_DIR}:")
        for path in written:
            print(f"    {os.path.basename(path)}")
    else:
        print("  No non-empty tables to export.")

    _section("END OF REPORT")
    print(f"  Diagnosis saved to data/diagnosed_needs.csv")
    print(f"  Free agent comparison page saved to {html_path}")
    print(f"  CSV exports saved to {EXPORT_DIR}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Mariners offseason report -- diagnosis, "
                     "year-over-year, free agent fit/cost, and fielding targets.")
    parser.add_argument("--from-season", type=int, default=2025)
    parser.add_argument("--to-season", type=int, default=2026)
    args = parser.parse_args()

    run_report(from_season=args.from_season, to_season=args.to_season)