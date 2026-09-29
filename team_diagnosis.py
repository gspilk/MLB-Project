"""
team_diagnosis.py
Diagnoses roster weaknesses from seattle_scraper.py's get_seattle_stats()
output, instead of relying on half-split data (baseball-reference is
currently hard-blocking our IP even on a single request -- see
team_half_splits.py's 429 -- so this deliberately reuses data you already
have cached from main.py --refresh, no new bbref hits required).

Approach:
  1. Pull batting/pitching/fielding/value_* from get_seattle_stats()
     (cache-first, so this doesn't need to hit bbref again if you've
     refreshed recently).
  2. Roll player rows up into team-level aggregates, weighted properly
     (PA-weighted for batting rates, IP-weighted for pitching rates,
     summed counting-stat ratios for %-based stats) -- NOT a naive
     column .mean(), which would let a 3-PA call-up's .500 BA count the
     same as your everyday starter's real sample.
  3. Split pitching into rotation vs. bullpen (reuses get_rotation/
     get_bullpen from seattle_scraper.py) and diagnose each separately,
     since a great rotation can mask a terrible bullpen in a blended ERA.
  4. Compare every metric to a league-average benchmark and rank the
     gaps -- biggest weakness first. That ranked table is what
     free_agent_recs.py should consume next: instead of "find good
     players," it becomes "find players who fix THIS specific gap."

STATS COVERED (v2 -- expanded per request to dig past just the basic
rate stats):
  Batting:  BA, OBP, SLG, OPS, ISO (power vs. contact), BB%, K% (plate
            discipline -- explains WHY OBP/BA move the way they do)
  Rotation: ERA, WHIP, K9, BB9, HR9 (getting hit hard vs. giving up
            the long ball specifically)
  Bullpen:  same five as rotation
  Fielding: team total Rtot/DRS (defense can explain part of an ERA/WHIP
            gap that looks like a pitching problem but isn't)
  WAR distribution: reported separately as CONTEXT, not a weakness/
            strength verdict -- it doesn't mean "good" or "bad," it
            means "fragile" or "deep," which matters for free-agent
            planning but shouldn't corrupt the severity ranking the
            other metrics use.

LEAGUE_AVG below are approximate modern-era MLB averages, not scraped --
flagged clearly so you know they're a placeholder. If/when bbref access
comes back (or you want to wire in the MLB Stats API, which is a
different host and untouched by the current block), swap
get_league_averages() for a real pull and nothing else here needs to
change -- the diagnosis logic just consumes whatever dict it returns.

Usage:
    from team_diagnosis import diagnose_team, get_war_distribution
    needs = diagnose_team()          # ranked weakness/strength table, saves CSV
    war   = get_war_distribution()   # separate context dict, not scored
"""

import os
import pandas as pd

from mariners_stats import get_seattle_stats, get_rotation, get_bullpen

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NEEDS_CSV = os.path.join(DATA_DIR, "diagnosed_needs.csv")

# Approximate modern-era MLB league-average benchmarks. PLACEHOLDER --
# swap get_league_averages() for a real source when you have one; every
# other function here just consumes this dict, so nothing else changes.
LEAGUE_AVG = {
    "BA":    0.245,
    "OBP":   0.315,
    "SLG":   0.400,
    "OPS":   0.715,
    "ISO":   0.155,
    "BB_pct": 8.5,
    "K_pct":  22.5,

    "rotation_ERA":  4.20,
    "rotation_WHIP": 1.25,
    "rotation_K9":   8.7,
    "rotation_BB9":  2.9,
    "rotation_HR9":  1.20,

    "bullpen_ERA":   4.00,
    "bullpen_WHIP":  1.28,
    "bullpen_K9":    9.2,
    "bullpen_BB9":   3.3,
    "bullpen_HR9":   1.10,

    # team defensive runs saved is defined relative to league average, so
    # the league "average" for it is 0 by construction (30 teams' Rtot
    # nets out around zero across the league)
    "fielding_Rtot": 0.0,
}

# How big a gap has to be (in the stat's own units) before we call it a
# real strength/weakness instead of noise. Rate stats only.
THRESHOLDS = {
    "BA": 0.008, "OBP": 0.008, "SLG": 0.015, "OPS": 0.020,
    "ISO": 0.010, "BB_pct": 1.0, "K_pct": 1.5,

    "rotation_ERA": 0.25, "rotation_WHIP": 0.05, "rotation_K9": 0.5,
    "rotation_BB9": 0.3, "rotation_HR9": 0.15,

    "bullpen_ERA": 0.30, "bullpen_WHIP": 0.06, "bullpen_K9": 0.5,
    "bullpen_BB9": 0.3, "bullpen_HR9": 0.15,

    "fielding_Rtot": 5.0,
}

# Stats where LOWER is better need the gap sign flipped before ranking
# "weakness" -- a positive gap there is bad, not good.
LOWER_IS_BETTER = {
    "rotation_ERA", "rotation_WHIP", "rotation_BB9", "rotation_HR9",
    "bullpen_ERA", "bullpen_WHIP", "bullpen_BB9", "bullpen_HR9",
    "K_pct",   # strikeouts are bad for a HITTER
}


def get_league_averages() -> dict:
    """Placeholder -- see module docstring. Swap this out for a real pull
    (MLB Stats API is a different host than bbref, unaffected by the
    current block) and diagnose_team() doesn't need to change at all."""
    return LEAGUE_AVG


def _weighted_batting(batting: pd.DataFrame) -> dict:
    """PA-weighted team batting line -- NOT a plain column average, so a
    September call-up's tiny sample doesn't count the same as a
    150-game starter's. Rate stats (BA/OBP/SLG) are PA-weighted; ISO is
    derived from the SAME weighted BA/SLG (not weighted separately) so
    it stays internally consistent; BB%/K% are true aggregate rates
    (sum(BB)/sum(PA)), which is the statistically correct way to combine
    a %-of-PA stat across players, not an average of each player's own
    percentage."""
    if batting.empty or "PA" not in batting.columns:
        return {}
    df = batting.dropna(subset=["PA"]).copy()
    df = df[df["PA"] > 0]
    total_pa = df["PA"].sum()
    if total_pa == 0:
        return {}
    out = {"PA": total_pa}
    for stat in ["BA", "OBP", "SLG", "OPS"]:
        if stat in df.columns:
            valid = df.dropna(subset=[stat])
            if valid["PA"].sum() > 0:
                out[stat] = (valid[stat] * valid["PA"]).sum() / valid["PA"].sum()

    if "BA" in out and "SLG" in out:
        out["ISO"] = out["SLG"] - out["BA"]

    if "BB" in df.columns:
        out["BB_pct"] = (df["BB"].sum() / total_pa) * 100
    if "SO" in df.columns:
        out["K_pct"] = (df["SO"].sum() / total_pa) * 100
    return out


def _weighted_pitching(pitching: pd.DataFrame) -> dict:
    """IP-weighted line for a pitching group (rotation or bullpen)."""
    if pitching.empty or "IP" not in pitching.columns:
        return {}
    df = pitching.dropna(subset=["IP"]).copy()
    df = df[df["IP"] > 0]
    total_ip = df["IP"].sum()
    if total_ip == 0:
        return {}
    out = {"IP": total_ip}
    if "ERA" in df.columns:
        valid = df.dropna(subset=["ERA"])
        if valid["IP"].sum() > 0:
            out["ERA"] = (valid["ERA"] * valid["IP"]).sum() / valid["IP"].sum()
    if "WHIP" in df.columns:
        valid = df.dropna(subset=["WHIP"])
        if valid["IP"].sum() > 0:
            out["WHIP"] = (valid["WHIP"] * valid["IP"]).sum() / valid["IP"].sum()
    if "SO" in df.columns:
        out["K9"] = (df["SO"].sum() / total_ip) * 9
    if "BB" in df.columns:
        out["BB9"] = (df["BB"].sum() / total_ip) * 9
    if "HR" in df.columns:
        out["HR9"] = (df["HR"].sum() / total_ip) * 9
    return out


def _team_fielding(fielding: pd.DataFrame) -> dict:
    """Team defensive runs saved -- an ADDITIVE total across players, not
    a weighted average (Rtot/Rdrs are already 'runs above/below average'
    per player, so summing them gives the team's total defensive value).
    Tries Rtot first (Total Zone, the older/more universal bbref column),
    falls back to Rdrs (Defensive Runs Saved, newer bbref tables) since
    which one bbref serves has varied across seasons."""
    if fielding.empty:
        return {}
    col = "Rtot" if "Rtot" in fielding.columns else (
          "Rdrs" if "Rdrs" in fielding.columns else None)
    if col is None:
        return {}
    total = pd.to_numeric(fielding[col], errors="coerce").sum()
    return {"Rtot": total}


def get_war_distribution(force_refresh: bool = False, stats: dict = None) -> dict:
    """CONTEXT only -- not a weakness/strength verdict. Tells you whether
    production is concentrated in a few players (fragile -- one injury
    and it's gone) or spread out (deep). Deliberately kept separate from
    diagnose_team()'s ranked table so it can't skew the severity ranking
    that free_agent_recs.py is meant to consume."""
    stats = stats if stats is not None else get_seattle_stats(force_refresh=force_refresh)
    out = {}
    for label, key in [("batting", "value_batting"), ("pitching", "value_pitching")]:
        df = stats.get(key, pd.DataFrame())
        if df.empty:
            continue
        war_col = next((c for c in df.columns if "WAR" in str(c) and "162" not in str(c)), None)
        if not war_col:
            continue
        war = pd.to_numeric(df[war_col], errors="coerce").dropna()
        war = war[war > 0]  # negative-WAR players don't count as "production" to concentrate
        total = war.sum()
        if total <= 0:
            continue
        top3_share = war.sort_values(ascending=False).head(3).sum() / total * 100
        out[f"{label}_total_WAR"] = round(total, 1)
        out[f"{label}_top3_share_pct"] = round(top3_share, 1)
    return out


def diagnose_team(force_refresh: bool = False) -> pd.DataFrame:
    stats = get_seattle_stats(force_refresh=force_refresh)
    batting  = stats.get("batting", pd.DataFrame())
    pitching = stats.get("pitching", pd.DataFrame())
    fielding = stats.get("fielding", pd.DataFrame())
    league = get_league_averages()

    rows = []

    team_bat = _weighted_batting(batting)
    for stat in ["BA", "OBP", "SLG", "OPS", "ISO", "BB_pct", "K_pct"]:
        if stat not in team_bat or stat not in league:
            continue
        team_val, lg_val = team_bat[stat], league[stat]
        gap = round(team_val - lg_val, 3)
        rows.append(_build_row("Batting", stat, team_val, lg_val, gap, stat in LOWER_IS_BETTER))

    rot = get_rotation(pitching)
    team_rot = _weighted_pitching(rot)
    for stat, key in [("ERA", "rotation_ERA"), ("WHIP", "rotation_WHIP"),
                       ("K9", "rotation_K9"), ("BB9", "rotation_BB9"),
                       ("HR9", "rotation_HR9")]:
        if stat not in team_rot or key not in league:
            continue
        team_val, lg_val = team_rot[stat], league[key]
        gap = round(team_val - lg_val, 3)
        rows.append(_build_row("Rotation", key, team_val, lg_val, gap, key in LOWER_IS_BETTER))

    bp = get_bullpen(pitching)
    team_bp = _weighted_pitching(bp)
    for stat, key in [("ERA", "bullpen_ERA"), ("WHIP", "bullpen_WHIP"),
                       ("K9", "bullpen_K9"), ("BB9", "bullpen_BB9"),
                       ("HR9", "bullpen_HR9")]:
        if stat not in team_bp or key not in league:
            continue
        team_val, lg_val = team_bp[stat], league[key]
        gap = round(team_val - lg_val, 3)
        rows.append(_build_row("Bullpen", key, team_val, lg_val, gap, key in LOWER_IS_BETTER))

    team_field = _team_fielding(fielding)
    if "Rtot" in team_field and "fielding_Rtot" in league:
        team_val, lg_val = team_field["Rtot"], league["fielding_Rtot"]
        gap = round(team_val - lg_val, 3)
        rows.append(_build_row("Fielding", "fielding_Rtot", team_val, lg_val, gap, False))

    if not rows:
        return pd.DataFrame()

    needs = pd.DataFrame(rows)
    # rank by how bad the weakness is (severity_score already flips sign
    # for lower-is-better stats so "bigger = worse" is consistent)
    needs = needs.sort_values("Severity_Score", ascending=False).reset_index(drop=True)

    # Severity_Score used to get dropped here before saving/returning --
    # fine when only a human read the printed table, but
    # free_agent_recommendations.py needs the actual number (not just
    # ranking order) to turn "this category is a weakness" into "by how
    # much," so it stays in both the saved CSV and the returned frame now.
    os.makedirs(DATA_DIR, exist_ok=True)
    needs.to_csv(NEEDS_CSV, index=False)
    return needs


def _build_row(category: str, metric: str, team_val: float, lg_val: float,
               gap: float, lower_is_better: bool = False) -> dict:
    threshold = THRESHOLDS.get(metric, 0.02)
    # severity_score: positive = weakness, negative = strength, scaled so
    # "how many thresholds below/above average" is comparable across
    # stats with very different units (OPS vs. ERA vs. K/9)
    signed_gap = -gap if lower_is_better else gap
    severity_score = -signed_gap / threshold if threshold else 0

    if abs(gap) < threshold:
        verdict = "AVERAGE"
    elif signed_gap > 0:
        verdict = "STRENGTH"
    else:
        verdict = "WEAKNESS"

    return {
        "Category": category,
        "Metric": metric,
        "Team_Value": round(team_val, 3),
        "League_Avg": round(lg_val, 3),
        "Gap": gap,
        "Verdict": verdict,
        "Severity_Score": round(severity_score, 2),
    }


if __name__ == "__main__":
    needs = diagnose_team()
    if needs.empty:
        print("No diagnosis produced -- check that batting/pitching data loaded.")
    else:
        print("── Roster diagnosis (biggest weaknesses first) ──")
        print(needs.to_string(index=False))
        print(f"\n[saved] {NEEDS_CSV}")

    war = get_war_distribution()
    if war:
        print("\n── WAR distribution (context, not a weakness/strength verdict) ──")
        for k, v in war.items():
            print(f"  {k}: {v}")