"""
stat_breakdown.py
Breaks a team-level rate stat (e.g. team BA = .232, bullpen ERA = 4.41,
team fielding Rtot = -30) down into WHO is responsible for it and BY HOW
MUCH, instead of stopping at the single team number the way
team_diagnosis.py does. Answers three questions team_diagnosis.py
deliberately doesn't:

  1. WHO -- which specific players are dragging the team number down (or
     propping it up), ranked by real weighted CONTRIBUTION, not raw
     rate. A .180 hitter in 20 PA barely moves the team number; a .240
     hitter in 550 PA moves it a lot, even though .240 "looks" less
     alarming than .180 on its own. Same idea for pitching (IP-weighted)
     and fielding (already additive, no weighting needed -- see below).

  2. WHY -- adds supporting context alongside each player's raw rate so
     a few different real explanations that all look identical as a
     single number can be told apart:
       Batting (BA): BB%/K%/ISO --
         - low BA + normal K%/BB%/ISO   -> plausible bad luck/BABIP
         - low BA + high K%             -> real contact problem
         - low BA + low ISO too         -> broad decline, not BA-specific
       Pitching (ERA/WHIP): K9/BB9/HR9 --
         - bad ERA + high HR9           -> getting hit for the long ball
         - bad ERA + high BB9           -> control/free-baserunner problem
         - bad ERA + low K9, normal else -> contact-management/defense-
                                            dependent, not a clean miss-bats issue
     (this project has no batted-ball/exit-velo data per player, so this
     is a real, honest triage -- pointing at WHICH follow-up question to
     ask -- not a diagnosis on its own)

  3. HOW MUCH BETTER -- simulate_bring_to_average() shows exactly how
     much the team number would move if the N biggest drags were
     brought up to league average, so "how can this get better" has a
     real, computed answer instead of a guess. simulate_fielding_fix()
     does the same for team defense (a straight sum, not a weighted
     average -- see diagnose_fielding()'s docstring for why).

Usage:
    from stat_breakdown import diagnose_batting_stat, diagnose_pitching_stat, \\
        diagnose_fielding, simulate_bring_to_average, simulate_fielding_fix
    print_all_breakdowns()   # batting BA, rotation+bullpen ERA, fielding -- one run
"""

import pandas as pd

from mariners_stats import get_seattle_stats, get_rotation, get_bullpen
from team_diagnosis import LEAGUE_AVG, LOWER_IS_BETTER

MIN_PA_FOR_CONTEXT = 20   # below this, BB%/K%/ISO are too noisy to read
MIN_IP_FOR_CONTEXT = 10   # same idea for pitchers
MIN_INN_FOR_FIELDING = 100  # same idea for fielding innings played


# ── batting ─────────────────────────────────────────────────────────────────
def diagnose_batting_stat(stat: str = "BA", force_refresh: bool = False,
                          stats: dict = None) -> pd.DataFrame:
    """
    Per-player breakdown of one team batting rate stat. Every player's
    CONTRIBUTION is in the SAME units as the stat itself and sums
    exactly to (team weighted average - league average) -- so if the
    team is 13 points of BA below league average, the Contribution
    column's total tells you exactly how those 13 points are split
    across the roster, not just who "looks" bad in isolation.
    """
    if stat not in LEAGUE_AVG:
        raise ValueError(f"No league average configured for '{stat}' -- "
                         f"add it to team_diagnosis.LEAGUE_AVG first.")
    league_avg = LEAGUE_AVG[stat]

    stats = stats if stats is not None else get_seattle_stats(force_refresh=force_refresh)
    batting = stats.get("batting", pd.DataFrame())
    if batting.empty or stat not in batting.columns or "PA" not in batting.columns:
        return pd.DataFrame()

    df = batting.dropna(subset=[stat, "PA"]).copy()
    df = df[df["PA"] > 0]
    if df.empty:
        return pd.DataFrame()

    total_pa = df["PA"].sum()
    df["Gap"] = (df[stat] - league_avg).round(3)
    df["Contribution_Pts"] = (df["PA"] * df["Gap"] / total_pa).round(4)

    team_weighted = (df[stat] * df["PA"]).sum() / total_pa
    team_gap = team_weighted - league_avg
    df["Pct_of_Team_Gap"] = (df["Contribution_Pts"] / team_gap * 100).round(1) if abs(team_gap) > 1e-9 else 0.0

    if "BB" in df.columns:
        df["BB_pct"] = (df["BB"] / df["PA"] * 100).round(1)
    if "SO" in df.columns:
        df["K_pct"] = (df["SO"] / df["PA"] * 100).round(1)
    if "SLG" in df.columns and "BA" in df.columns:
        df["ISO"] = (df["SLG"] - df["BA"]).round(3)

    df["Low_Sample"] = df["PA"] < MIN_PA_FOR_CONTEXT
    df["Weight"] = df["PA"]

    cols = ["Name", "PA", stat, "Gap", "Contribution_Pts", "Pct_of_Team_Gap",
            "BB_pct", "K_pct", "ISO", "Low_Sample", "Weight"]
    real_cols = [c for c in cols if c in df.columns]
    return df[real_cols].sort_values("Contribution_Pts").reset_index(drop=True)


def explain_batting_row(row: pd.Series, stat: str) -> str:
    """Honest, pattern-matched read on WHY a batter's gap looks the way
    it does -- not a real diagnosis (no batted-ball data per player)."""
    if stat != "BA" or pd.isna(row.get("Gap")):
        return ""
    if row.get("Low_Sample"):
        return "small sample -- don't read much into this yet"
    gap = row["Gap"]
    k_pct, iso = row.get("K_pct"), row.get("ISO")
    if gap >= 0:
        return "at or above league average -- not part of the problem"
    if pd.notna(k_pct) and k_pct >= LEAGUE_AVG.get("K_pct", 22.5) + 3:
        return "real contact problem -- strikeout rate is well above average"
    if pd.notna(iso) and iso < LEAGUE_AVG.get("ISO", 0.155) - 0.02:
        return "broad decline, not BA-specific -- power is down too"
    return "BA down but K%/power look normal -- plausible bad luck/BABIP, not a clear skill red flag"


# ── pitching ────────────────────────────────────────────────────────────────
def diagnose_pitching_stat(stat: str = "ERA", role: str = "bullpen",
                           force_refresh: bool = False, stats: dict = None) -> pd.DataFrame:
    """
    Per-player breakdown of one rotation-or-bullpen pitching rate stat,
    IP-weighted the same way diagnose_batting_stat() is PA-weighted.
    'role' must be "rotation" or "bullpen" -- these get different league
    averages in team_diagnosis.LEAGUE_AVG (rotation_ERA vs. bullpen_ERA)
    since a bullpen ERA of 4.00 and a rotation ERA of 4.00 don't mean the
    same real thing.

    Sign convention matches diagnose_batting_stat(): negative
    Contribution_Pts always means "this player is making the team number
    WORSE," regardless of whether the underlying stat is one where
    higher is better (K9) or lower is better (ERA/WHIP/BB9/HR9) -- so
    sorting ascending always puts the real problem at the top for every
    stat, not just BA.
    """
    key = f"{role}_{stat}"
    if key not in LEAGUE_AVG:
        raise ValueError(f"No league average configured for '{key}' -- "
                         f"check team_diagnosis.LEAGUE_AVG.")
    league_avg = LEAGUE_AVG[key]
    lower_is_better = key in LOWER_IS_BETTER

    stats = stats if stats is not None else get_seattle_stats(force_refresh=force_refresh)
    pitching = stats.get("pitching", pd.DataFrame())
    group = get_rotation(pitching) if role == "rotation" else get_bullpen(pitching)
    if group.empty or stat not in group.columns or "IP" not in group.columns:
        return pd.DataFrame()

    df = group.dropna(subset=[stat, "IP"]).copy()
    df = df[df["IP"] > 0]
    if df.empty:
        return pd.DataFrame()

    total_ip = df["IP"].sum()
    raw_gap = df[stat] - league_avg
    df["Gap"] = raw_gap.round(3)
    # flip sign for lower-is-better stats so negative always = "worse for the team"
    signed_gap = -raw_gap if lower_is_better else raw_gap
    df["Contribution_Pts"] = (df["IP"] * signed_gap / total_ip).round(4)

    team_weighted = (df[stat] * df["IP"]).sum() / total_ip
    team_signed_gap = (league_avg - team_weighted) if lower_is_better else (team_weighted - league_avg)
    df["Pct_of_Team_Gap"] = (df["Contribution_Pts"] / team_signed_gap * 100).round(1) if abs(team_signed_gap) > 1e-9 else 0.0

    if "SO" in df.columns:
        df["K9"] = (df["SO"] / df["IP"] * 9).round(2)
    if "BB" in df.columns:
        df["BB9"] = (df["BB"] / df["IP"] * 9).round(2)
    if "HR" in df.columns:
        df["HR9"] = (df["HR"] / df["IP"] * 9).round(2)

    df["Low_Sample"] = df["IP"] < MIN_IP_FOR_CONTEXT
    df["Weight"] = df["IP"]

    cols = ["Name", "IP", stat, "Gap", "Contribution_Pts", "Pct_of_Team_Gap",
            "K9", "BB9", "HR9", "Low_Sample", "Weight"]
    real_cols = [c for c in cols if c in df.columns]
    return df[real_cols].sort_values("Contribution_Pts").reset_index(drop=True)


def explain_pitching_row(row: pd.Series, stat: str) -> str:
    """Honest, pattern-matched read on WHY a pitcher's gap looks the way
    it does, using K9/BB9/HR9 -- same triage spirit as explain_batting_row,
    not a real diagnosis (no batted-ball data per pitcher)."""
    if stat not in ("ERA", "WHIP") or pd.isna(row.get("Contribution_Pts")):
        return ""
    if row.get("Low_Sample"):
        return "small sample -- don't read much into this yet"
    if row["Contribution_Pts"] >= 0:
        return "at or better than league average -- not part of the problem"
    hr9, bb9, k9 = row.get("HR9"), row.get("BB9"), row.get("K9")
    role_hr9_avg = LEAGUE_AVG.get("bullpen_HR9", 1.10)
    role_bb9_avg = LEAGUE_AVG.get("bullpen_BB9", 3.30)
    role_k9_avg  = LEAGUE_AVG.get("bullpen_K9", 9.20)
    if pd.notna(hr9) and hr9 >= role_hr9_avg + 0.3:
        return "getting hit for the long ball -- HR rate is well above average"
    if pd.notna(bb9) and bb9 >= role_bb9_avg + 0.5:
        return "control problem -- walk rate is well above average, too many free baserunners"
    if pd.notna(k9) and k9 <= role_k9_avg - 1.0:
        return "not missing bats -- leaning heavily on the defense behind him"
    return "rate down but K9/BB9/HR9 all look roughly normal -- plausible bad luck/sequencing, not a clear red flag"


# ── fielding ────────────────────────────────────────────────────────────────
def diagnose_fielding(force_refresh: bool = False, stats: dict = None) -> pd.DataFrame:
    """
    Per-player breakdown of team fielding. Different math from batting/
    pitching on purpose: Rtot/Rdrs is ALREADY a runs-above-or-below-
    average number per player (bbref computes it that way), so there's
    no separate "league average" to subtract or PA/IP to weight by --
    each player's own Rtot IS their real contribution, and the team
    total is just the sum. Sorting ascending still puts the real
    problem (most negative Rtot) at the top, same convention as the
    other two functions.
    """
    stats = stats if stats is not None else get_seattle_stats(force_refresh=force_refresh)
    fielding = stats.get("fielding", pd.DataFrame())
    if fielding.empty:
        return pd.DataFrame()

    col = "Rtot" if "Rtot" in fielding.columns else ("Rdrs" if "Rdrs" in fielding.columns else None)
    if col is None:
        return pd.DataFrame()

    df = fielding.dropna(subset=[col]).copy()
    df["Contribution_Pts"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Contribution_Pts"])
    if df.empty:
        return pd.DataFrame()

    team_total = df["Contribution_Pts"].sum()
    df["Pct_of_Team_Total"] = (df["Contribution_Pts"] / team_total * 100).round(1) if abs(team_total) > 1e-9 else 0.0

    inn_col = "Inn" if "Inn" in df.columns else None
    if inn_col:
        df["Low_Sample"] = pd.to_numeric(df[inn_col], errors="coerce") < MIN_INN_FOR_FIELDING
    else:
        df["Low_Sample"] = False

    cols = ["Name", "Pos", inn_col, col, "Contribution_Pts", "Pct_of_Team_Total", "Low_Sample"]
    real_cols = [c for c in cols if c and c in df.columns]
    return df[real_cols].sort_values("Contribution_Pts").reset_index(drop=True)


# ── "how much better" simulations ───────────────────────────────────────────
def simulate_bring_to_average(breakdown: pd.DataFrame, value_col: str, weight_col: str,
                              league_avg: float, top_n: int = 3) -> dict:
    """
    Shared by batting and pitching: takes the top_n biggest real drags
    (by Contribution_Pts, excluding tiny samples) and computes what the
    team number would be if EACH simply hit league average, holding
    everyone else's real production constant. A real, computed answer,
    not a guess -- and honestly just a "what if exactly average" scenario,
    not a prediction they will be.
    """
    candidates = breakdown[(breakdown["Contribution_Pts"] < 0) & (~breakdown["Low_Sample"])]
    candidates = candidates.sort_values("Contribution_Pts").head(top_n)
    if candidates.empty:
        return {"team_current": None, "team_if_fixed": None, "delta": None, "players": []}

    total_weight = breakdown[weight_col].sum()
    current_team = (breakdown[value_col] * breakdown[weight_col]).sum() / total_weight

    fixed = breakdown.copy()
    fixed.loc[candidates.index, value_col] = league_avg
    fixed_team = (fixed[value_col] * fixed[weight_col]).sum() / total_weight

    return {
        "team_current": round(current_team, 4),
        "team_if_fixed": round(fixed_team, 4),
        "delta": round(fixed_team - current_team, 4),
        "players": candidates["Name"].tolist(),
    }


def simulate_fielding_fix(breakdown: pd.DataFrame, top_n: int = 3) -> dict:
    """Fielding version of the same idea -- a straight sum, not a
    weighted average, since Rtot is already additive (see
    diagnose_fielding()'s docstring)."""
    candidates = breakdown[(breakdown["Contribution_Pts"] < 0) & (~breakdown["Low_Sample"])]
    candidates = candidates.sort_values("Contribution_Pts").head(top_n)
    if candidates.empty:
        return {"team_current": None, "team_if_fixed": None, "delta": None, "players": []}

    current_total = breakdown["Contribution_Pts"].sum()
    fixed_total = current_total - candidates["Contribution_Pts"].sum()  # bring each to 0 (average)
    return {
        "team_current": round(current_total, 1),
        "team_if_fixed": round(fixed_total, 1),
        "delta": round(fixed_total - current_total, 1),
        "players": candidates["Name"].tolist(),
    }


# ── printing ─────────────────────────────────────────────────────────────────
def print_batting_breakdown(stat: str = "BA", force_refresh: bool = False, top_n: int = 15):
    breakdown = diagnose_batting_stat(stat, force_refresh=force_refresh)
    if breakdown.empty:
        print(f"No data available to break down batting '{stat}'.")
        return
    league_avg = LEAGUE_AVG[stat]
    total_pa = breakdown["PA"].sum()
    team_weighted = (breakdown[stat] * breakdown["PA"]).sum() / total_pa

    print(f"\n── Team {stat}: {team_weighted:.3f} (league avg {league_avg:.3f}, "
          f"gap {team_weighted - league_avg:+.3f}) -- who's responsible ──")
    display = breakdown.copy()
    display["Why"] = display.apply(lambda r: explain_batting_row(r, stat), axis=1)
    cols = [c for c in ["Name", "PA", stat, "Gap", "Contribution_Pts",
                        "Pct_of_Team_Gap", "BB_pct", "K_pct", "ISO", "Why"] if c in display.columns]
    print(display[cols].head(top_n).to_string(index=False))

    sim = simulate_bring_to_average(breakdown, stat, "PA", league_avg, top_n=3)
    if sim["players"]:
        print(f"\n── What if the {len(sim['players'])} biggest drags "
              f"({', '.join(sim['players'])}) simply hit league average? ──")
        print(f"  Team {stat}: {sim['team_current']:.3f} -> {sim['team_if_fixed']:.3f} ({sim['delta']:+.3f})")


def print_pitching_breakdown(stat: str = "ERA", role: str = "bullpen",
                             force_refresh: bool = False, top_n: int = 10):
    breakdown = diagnose_pitching_stat(stat, role, force_refresh=force_refresh)
    if breakdown.empty:
        print(f"No data available to break down {role} '{stat}'.")
        return
    league_avg = LEAGUE_AVG[f"{role}_{stat}"]
    total_ip = breakdown["IP"].sum()
    team_weighted = (breakdown[stat] * breakdown["IP"]).sum() / total_ip

    print(f"\n── {role.title()} {stat}: {team_weighted:.2f} (league avg {league_avg:.2f}, "
          f"gap {team_weighted - league_avg:+.2f}) -- who's responsible ──")
    display = breakdown.copy()
    display["Why"] = display.apply(lambda r: explain_pitching_row(r, stat), axis=1)
    cols = [c for c in ["Name", "IP", stat, "Gap", "Contribution_Pts",
                        "Pct_of_Team_Gap", "K9", "BB9", "HR9", "Why"] if c in display.columns]
    print(display[cols].head(top_n).to_string(index=False))

    sim = simulate_bring_to_average(breakdown, stat, "IP", league_avg, top_n=2)
    if sim["players"]:
        print(f"\n── What if the {len(sim['players'])} biggest drags "
              f"({', '.join(sim['players'])}) simply hit league average? ──")
        print(f"  {role.title()} {stat}: {sim['team_current']:.2f} -> {sim['team_if_fixed']:.2f} ({sim['delta']:+.2f})")


def print_fielding_breakdown(force_refresh: bool = False, top_n: int = 15):
    breakdown = diagnose_fielding(force_refresh=force_refresh)
    if breakdown.empty:
        print("No fielding data available to break down.")
        return
    team_total = breakdown["Contribution_Pts"].sum()
    print(f"\n── Team fielding (Rtot): {team_total:+.1f} runs -- who's responsible ──")
    cols = [c for c in ["Name", "Pos", "Inn", "Rtot", "Rdrs", "Contribution_Pts",
                        "Pct_of_Team_Total", "Low_Sample"] if c in breakdown.columns]
    print(breakdown[cols].head(top_n).to_string(index=False))

    sim = simulate_fielding_fix(breakdown, top_n=3)
    if sim["players"]:
        print(f"\n── What if the {len(sim['players'])} worst gloves "
              f"({', '.join(sim['players'])}) simply played average defense? ──")
        print(f"  Team Rtot: {sim['team_current']:+.1f} -> {sim['team_if_fixed']:+.1f} ({sim['delta']:+.1f})")


def print_all_breakdowns(force_refresh: bool = False):
    print_batting_breakdown("BA", force_refresh=force_refresh)
    print_pitching_breakdown("ERA", "bullpen", force_refresh=force_refresh)
    print_pitching_breakdown("ERA", "rotation", force_refresh=force_refresh)
    print_fielding_breakdown(force_refresh=force_refresh)


if __name__ == "__main__":
    print_all_breakdowns()