"""
main.py
Runs the complete Seattle Mariners Midseason Analyzer.
One command to build all data, analyze, grade, recommend, and report.

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
    p = argparse.ArgumentParser(
        description="Seattle Mariners Midseason Analyzer"
    )
    p.add_argument("--refresh",    action="store_true",
                   help="Force refresh all cached data")
    p.add_argument("--no-report",  action="store_true",
                   help="Skip Excel report generation")
    p.add_argument("--print-only", action="store_true",
                   help="Print to console only, no Excel")
    return p.parse_args()


def main():
    args  = parse_args()
    start = time.time()

    print(f"\n{'='*65}")
    print(f"  SEATTLE MARINERS MIDSEASON ANALYZER")
    print(f"  {date.today()}")
    print(f"{'='*65}\n")

    # ── step 1: build data ──
    print("STEP 1/5 — Building data...")
    from data_builder import build_all
    data = build_all(2026, force_refresh=args.refresh)

    # ── step 2: analyze team ──
    print("\nSTEP 2/5 — Analyzing team...")
    from team_analyzer import analyze_team, print_analysis
    analysis = analyze_team(data)

    # ── step 3: grade players ──
    print("\nSTEP 3/5 — Grading players...")
    from player_grades import grade_players, print_grades
    grades = grade_players(data)

    # ── step 4: generate recommendations ──
    print("\nSTEP 4/5 — Generating recommendations...")
    from recommender import generate_recommendations, print_recommendations
    recs = generate_recommendations(data, analysis, grades)

    # ── step 5: generate report ──
    if not args.no_report and not args.print_only:
        print("\nSTEP 5/5 — Generating Excel report...")
        from output_report import generate_report
        path = generate_report(data, analysis, grades, recs)
    else:
        print("\nSTEP 5/5 — Skipping Excel report")
        path = None

    # ── print to console ──
    print_analysis(analysis)
    print_grades(grades)
    print_recommendations(recs)

    # ── summary ──
    elapsed = round(time.time() - start, 1)
    o = analysis.get("overall", {})
    s = analysis.get("standings", {})

    print(f"\n{'='*65}")
    print(f"  COMPLETE — {elapsed}s")
    print(f"{'='*65}")
    print(f"  Record:         {s.get('record')}")
    print(f"  Grade:          {o.get('grade')} — {o.get('verdict')}")
    print(f"  Projected wins: {o.get('projected_wins')}")
    print(f"  Range:          {o.get('floor')}—{o.get('ceiling')} wins")
    if path:
        print(f"  Report:         {path}")
    print(f"{'='*65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())