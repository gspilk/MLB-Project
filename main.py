"""
main.py
Runs the complete Seattle Mariners Midseason Analyzer.

Usage:
    python main.py                    # full run with cache
    python main.py --refresh          # force refresh all data
    python main.py --no-report        # skip Excel report
    python main.py --print-only       # print to console only
"""

import sys
import time
import argparse
from datetime import date


def parse_args():
    p = argparse.ArgumentParser(description="Seattle Mariners Midseason Analyzer")
    p.add_argument("--refresh",    action="store_true", help="Force refresh all data")
    p.add_argument("--no-report",  action="store_true", help="Skip Excel report")
    p.add_argument("--print-only", action="store_true", help="Console only, no Excel")
    return p.parse_args()


def main():
    args  = parse_args()
    start = time.time()

    print(f"\n{'='*65}")
    print(f"  SEATTLE MARINERS MIDSEASON ANALYZER")
    print(f"  {date.today()}")
    print(f"  Data source: Baseball Reference (bbref.com)")
    print(f"  Note: bbref updates 6-24 hours after games")
    print(f"{'='*65}\n")

    # step 1 -- build data
    print("STEP 1/6 -- Building data...")
    from data_builder import build_all
    data = build_all(2026, force_refresh=args.refresh)

    # step 2 -- analyze team
    print("\nSTEP 2/6 -- Analyzing team...")
    from team_analyzer import analyze_team, print_analysis
    analysis = analyze_team(data)

    # step 3 -- grade players
    print("\nSTEP 3/6 -- Grading players...")
    from player_grades import grade_players, print_grades
    grades = grade_players(data)

    # step 4 -- recommendations
    print("\nSTEP 4/6 -- Generating recommendations...")
    from recommender import generate_recommendations, print_recommendations
    recs = generate_recommendations(data, analysis, grades)

    # step 5 -- simulation
    print("\nSTEP 5/6 -- Running deadline simulation...")
    from simulator import run_simulation, print_simulation
    import simulator as sim_mod

    # update simulator with live data
    try:
        standings = analysis.get("standings", {})
        record    = standings.get("record", "47-47") or "47-47"
        w, l      = map(int, record.split("-"))
        sim_mod.CURRENT_W       = w
        sim_mod.CURRENT_L       = l
        sim_mod.GAMES_REMAINING = 162 - w - l
        sim_mod.CURRENT_LUCK    = float(standings.get("luck", -2.0) or -2.0)
        era_rank = analysis.get("pitching", {}).get("team_era_rank")
        if era_rank:
            sim_mod.CURRENT_ERA_RANK = era_rank

        # update RS/G from batting overview
        import pandas as pd
        ov_bat = data.get("overview", {}).get("batting")
        if ov_bat is not None and not ov_bat.empty:
            tm = next((c for c in ["Tm","Team"] if c in ov_bat.columns), None)
            if tm:
                sea = ov_bat[ov_bat[tm].str.contains("Seattle", na=False)]
                if not sea.empty:
                    r_val = pd.to_numeric(sea["R"].values[0],  errors="coerce")
                    g_val = pd.to_numeric(sea["G"].values[0],  errors="coerce")
                    if r_val and g_val and g_val > 0:
                        sim_mod.CURRENT_RS_G = round(r_val / g_val, 2)

        # update RA/G from pitching overview
        ov_pit = data.get("overview", {}).get("pitching")
        if ov_pit is not None and not ov_pit.empty:
            tm = next((c for c in ["Tm","Team"] if c in ov_pit.columns), None)
            if tm:
                sea = ov_pit[ov_pit[tm].str.contains("Seattle", na=False)]
                if not sea.empty:
                    for col in ["RA", "R", "RA9"]:
                        if col in sea.columns:
                            ra_val = pd.to_numeric(sea[col].values[0], errors="coerce")
                            g_val  = pd.to_numeric(sea["G"].values[0],  errors="coerce")
                            if ra_val and g_val and g_val > 0:
                                sim_mod.CURRENT_RA_G = round(ra_val / g_val, 2)
                                break

        print(f"  [sim] Record {w}-{l}  RS/G {sim_mod.CURRENT_RS_G}"
              f"  RA/G {sim_mod.CURRENT_RA_G}  Luck {sim_mod.CURRENT_LUCK}")
    except Exception as e:
        print(f"  [warn] could not update simulator: {e}")

    scenarios = run_simulation()

    # step 6 -- generate report
    if not args.no_report and not args.print_only:
        print("\nSTEP 6/6 -- Generating Excel report...")
        from output_report import generate_report
        path = generate_report(data, analysis, grades, recs, scenarios)
    else:
        print("\nSTEP 6/6 -- Skipping Excel report")
        path = None

    # print to console
    print_analysis(analysis)
    print_grades(grades)
    print_recommendations(recs)
    print_simulation(scenarios)

    # summary
    elapsed = round(time.time() - start, 1)
    o = analysis.get("overall", {})
    s = analysis.get("standings", {})

    print(f"\n{'='*65}")
    print(f"  COMPLETE -- {elapsed}s")
    print(f"{'='*65}")
    print(f"  Record:         {s.get('record')}")
    print(f"  Grade:          {o.get('grade')} -- {o.get('verdict')}")
    print(f"  Projected wins: {o.get('projected_wins')}")
    print(f"  Range:          {o.get('floor')}--{o.get('ceiling')} wins")
    print(f"  Data as of:     {date.today()} (refresh with --refresh)")
    if path:
        print(f"  Report:         {path}")
    print(f"{'='*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())