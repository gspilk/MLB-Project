"""
abs_challenge_summary.py
Real, formatted report on the Mariners' ABS challenge performance:
  - team-level overturn rate vs. real league average, by role
  - a "worst individual decision-makers" leaderboard -- who keeps
    burning challenges they had no business making

Uses abs_challenge_scraper.py's real, live-scraped data. No hardcoded
numbers -- league averages are computed live from all 30 teams' real
data each run.

Usage:
    python abs_challenge_summary.py
"""

import pandas as pd
from abs_challenges_scraper import get_abs_challenges, TEAM_NAME

MIN_CHALLENGES = 5   # minimum challenges (as batter) to appear on the
                     # "worst decision-makers" leaderboard -- filters
                     # out 1-2 challenge samples that could show 0% or
                     # 100% purely from noise, same reasoning as the
                     # confidence-tagging built elsewhere in this
                     # project tonight (don't trust a tiny sample at
                     # face value)


def compute_league_averages(by_team: pd.DataFrame) -> dict:
    """Real, live-computed league averages (challenge-weighted, not a
    simple average-of-percentages) across all 30 teams for the current
    season -- not a hardcoded reference number."""
    def weighted_pct(chal_col, ot_col):
        chal = pd.to_numeric(by_team[chal_col], errors="coerce")
        ot_pct = pd.to_numeric(by_team[ot_col], errors="coerce")
        valid = chal.notna() & ot_pct.notna() & (chal > 0)
        if not valid.any():
            return None
        ot_count = chal[valid] * (ot_pct[valid] / 100)
        return round(ot_count.sum() / chal[valid].sum() * 100, 1)

    return {
        "overall": weighted_pct("All Challenges_Chal", "All Challenges_OT%"),
        "batter":  weighted_pct("As Batter_Chal", "As Batter_OT%"),
        "catcher": weighted_pct("As Catcher_Chal", "As Catcher_OT%"),
    }


def print_team_comparison(by_team: pd.DataFrame):
    league_avg = compute_league_averages(by_team)
    team_row = by_team[by_team["Name"] == TEAM_NAME]

    print("\n" + "=" * 70)
    print(f"{TEAM_NAME.upper()} vs. REAL LEAGUE AVERAGE -- ABS CHALLENGES")
    print("=" * 70)

    if team_row.empty:
        print(f"  {TEAM_NAME} not found in team data.")
        return

    row = team_row.iloc[0]
    comparisons = [
        ("Overall", "All Challenges_OT%", league_avg["overall"]),
        ("As Batter", "As Batter_OT%", league_avg["batter"]),
        ("As Catcher", "As Catcher_OT%", league_avg["catcher"]),
    ]
    print(f"  {'Role':<12} {'Mariners':>10} {'League Avg':>12} {'Gap':>8}")
    print("  " + "-" * 44)
    for label, col, avg in comparisons:
        team_val = row.get(col)
        if pd.isna(team_val) or avg is None:
            continue
        gap = round(team_val - avg, 1)
        sign = "+" if gap >= 0 else ""
        note = " (better)" if gap > 0 else " (worse)" if gap < 0 else ""
        print(f"  {label:<12} {team_val:>9.1f}% {avg:>11.1f}% "
              f"{sign}{gap:>6.1f}{note}")
    print("=" * 70)


def print_worst_decision_makers(by_player: pd.DataFrame):
    sea = by_player[by_player["Tm"] == "SEA"].copy()
    sea["chal"] = pd.to_numeric(sea["As Batter_Chal"], errors="coerce")
    sea["ot_pct"] = pd.to_numeric(sea["As Batter_OT%"], errors="coerce")
    qualified = sea[sea["chal"] >= MIN_CHALLENGES].sort_values("ot_pct")

    print("\n" + "=" * 70)
    print(f"WORST DECISION-MAKERS -- AS BATTER (min {MIN_CHALLENGES} challenges)")
    print("=" * 70)
    if qualified.empty:
        print(f"  No Mariners batters with {MIN_CHALLENGES}+ challenges found.")
    else:
        print(f"  {'Player':<20} {'Challenges':>11} {'Won':>6} {'Win %':>8}")
        print("  " + "-" * 48)
        for _, r in qualified.iterrows():
            won = round(r["chal"] * r["ot_pct"] / 100)
            print(f"  {r['Name']:<20} {int(r['chal']):>11} {int(won):>6} "
                  f"{r['ot_pct']:>7.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    data = get_abs_challenges()
    print_team_comparison(data["by_team"])
    print_worst_decision_makers(data["by_player"])