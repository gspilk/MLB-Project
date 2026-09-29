"""
free_agent_recommendations.py
Builds on free_agent_stats_matcher.py: pulls the Mariners' own current
roster stats, maps positions between bbref's codes and MLBTR's
position-group labels, computes a real upgrade delta for every matched
free agent against whoever's actually playing that spot for Seattle
right now, and saves real chart files (not just printed tables) for
the biggest real gaps.

DIAGNOSIS-WEIGHTED SCORING: now also loads data/diagnosed_needs.csv
(built by team_diagnosis.py) and uses it to weight recommendation scores
on top of the existing per-position incumbent comparison. The two
signals answer different questions and are meant to combine, not
replace each other:
  - the existing incumbent-based need_weight (compute_recommendation_score,
    unchanged) answers "which SPECIFIC position is weak" (e.g. Second
    Basemen has a low OPS incumbent)
  - team_diagnosis.py's category severity answers "which WHOLE SIDE OF
    THE BALL is weak" (e.g. the offense overall has a real hit-tool
    problem, the bullpen overall is below average) -- something no
    single position's incumbent stat can see on its own
See get_team_need_multipliers()'s docstring for the defense-adjusted
bullpen discount, and get_batting_need_profile()/contact_power_fit_bonus()
for the hit-tool-vs-power distinction.

Usage:
    python free_agent_recommendations.py
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just saving files
import matplotlib.pyplot as plt

from data_builder import build_all
from free_agent import get_mlbtr_free_agents
from free_agent_stats_matcher import enrich_free_agents
from team_diagnosis import LEAGUE_AVG

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "free_agent_charts")
NEEDS_CSV  = os.path.join(os.path.dirname(__file__), "data", "diagnosed_needs.csv")

# BUG FIX: this originally assumed the batting table's real "Pos"
# column used the same multi-character scorecard-digit format already
# confirmed for the FIELDING table ("*4/6H", parsed by taking the first
# character against a digit->position map). A live debug run showed
# the real batting table actually uses plain, clean abbreviations
# instead ("1B", "2B", "3B", "C", "SS", etc.) -- a completely different,
# much simpler format. The old digit-based parsing was silently taking
# the first CHARACTER of these plain strings and matching it against
# the wrong lookup table: "2B" starts with "2", which the old digit map
# pointed at Catchers; "3B" starts with "3", pointed at First Basemen --
# exactly the two real, confirmed wrong results a live run produced
# (Cole Young, an actual 2B, showing as the Catchers incumbent; Brendan
# Donovan, an actual 3B, showing as the First Basemen incumbent).
# Direct string matching on the real, confirmed values instead --
# no character-parsing needed at all now that the real format is known.
BBREF_TO_MLBTR_POS = {
    "C": "Catchers", "1B": "First Basemen", "2B": "Second Basemen",
    "3B": "Third Basemen", "SS": "Shortstops", "LF": "Left Fielders",
    "CF": "Center Fielders", "RF": "Right Fielders", "DH": "Designated Hitters",
    # OF/IF/CI/UT/P are real, but genuinely ambiguous (a generic
    # outfield/infield/utility label, not one specific MLBTR position
    # group) -- intentionally left unmapped rather than guessing which
    # single position group to force a multi-position/utility player
    # into.
}

# Which diagnosis category (team_diagnosis.py's Category column) each
# MLBTR position group rolls up into, for the team-level severity
# multiplier. Every batting group maps to "Batting" -- the diagnosis
# doesn't (yet) have a per-position breakdown, only offense-as-a-whole,
# so every batter candidate gets the same offense-wide multiplier
# regardless of which specific position they'd play.
POSITION_GROUP_TO_DIAGNOSIS_CATEGORY = {
    "Catchers": "Batting", "First Basemen": "Batting", "Second Basemen": "Batting",
    "Third Basemen": "Batting", "Shortstops": "Batting", "Left Fielders": "Batting",
    "Center Fielders": "Batting", "Right Fielders": "Batting", "Designated Hitters": "Batting",
    "Starting Pitchers": "Rotation",
    "Right-Handed Relievers": "Bullpen", "Left-Handed Relievers": "Bullpen",
}


def _get_current_roster_names(data: dict) -> set:
    """
    BUG FIX: a real run showed J.P. Crawford recommended as the "best
    target" to replace himself at Shortstop -- he's genuinely both a
    current Mariner AND a real free agent this winter (confirmed
    earlier tonight), so he legitimately sits in both the current
    roster data and the free agent pool. With no other shortstop
    option scoring higher than his own real number, the tool ended up
    "recommending" re-signing the exact same guy as if it were an
    external upgrade. Real, current roster names (stripped of any
    "(60-day IL)"-style suffixes) get excluded from the target pool
    entirely -- a true free-agent signing means bringing in someone
    who isn't already on the team.
    """
    names = set()
    for table_key in ("batting", "pitching"):
        table = data["seattle"].get(table_key)
        if table is None or table.empty:
            continue
        for raw_name in table["Name"].dropna():
            clean = str(raw_name).split(" (")[0].strip()
            names.add(clean)
    return names


def _parse_primary_position(pos_str: str) -> str:
    """Real, confirmed format: batting table's Pos is already a
    plain, clean abbreviation ("1B", "C", "SS", etc.) -- no parsing
    needed, just normalize and look it up directly."""
    if not pos_str or pd.isna(pos_str):
        return None
    return str(pos_str).strip()


def build_internal_candidates(data: dict, incumbents: dict) -> pd.DataFrame:
    """
    Pipeline 3 -- the same real comparison already built for free
    agents, now pointed at the team's OWN bench/40-man/depth players.
    A real internal option (a bench bat quietly outperforming the
    incumbent, a prospect ready for more time) should show up on the
    exact same ranked list as an external free agent target, not live
    in a separate, disconnected tool -- "should we sign someone or
    just play the guy we already have" is one real question, not two.

    Excludes whoever is already the real incumbent at each position
    (comparing a player to themselves is meaningless) and anyone with
    zero real PA/IP this season (nothing to compare yet).

    Also carries real fielding (Rtot/Rdrs) through for batters when
    available in data["seattle"]["fielding"] -- Mariners players get
    theirs from the TEAM-level fielding table here; external free
    agents now get theirs from free_agent_stats_matcher.py's match
    against bbref's LEAGUE-WIDE fielding leaderboard instead (added
    2026-09-28, see get_fielding_bonus()'s docstring), so both sources
    end up on the same "Rtot" column by the time compute_recommendation_
    score() reads it.
    """
    rows = []
    bat = data["seattle"]["batting"]
    fielding = data["seattle"].get("fielding", pd.DataFrame())
    fielding_col = None
    if not fielding.empty:
        fielding_col = "Rtot" if "Rtot" in fielding.columns else (
                       "Rdrs" if "Rdrs" in fielding.columns else None)

    for _, row in bat.iterrows():
        name = row.get("Name")
        primary = _parse_primary_position(row.get("Pos"))
        group = BBREF_TO_MLBTR_POS.get(primary)
        if not group:
            continue
        incumbent = incumbents.get(group)
        if incumbent is None or name == incumbent.get("name"):
            continue
        pa = pd.to_numeric(row.get("PA"), errors="coerce") or 0
        ops = pd.to_numeric(row.get("OPS"), errors="coerce")
        if pa <= 0 or pd.isna(ops):
            continue
        entry = {
            "name": name, "position_group": group, "source": "Internal (roster)",
            "OPS": ops, "PA": pa, "matched": True,
            "BA": pd.to_numeric(row.get("BA"), errors="coerce"),
            "SLG": pd.to_numeric(row.get("SLG"), errors="coerce"),
        }
        if fielding_col is not None:
            f_row = fielding[fielding["Name"] == name]
            if not f_row.empty:
                entry["Rtot"] = pd.to_numeric(f_row.iloc[0][fielding_col], errors="coerce")
        rows.append(entry)

    pit = data["seattle"]["pitching"]
    for _, row in pit.iterrows():
        name = row.get("Name")
        gs = pd.to_numeric(row.get("GS"), errors="coerce") or 0
        g = pd.to_numeric(row.get("G"), errors="coerce") or 0
        is_starter = gs >= (g * 0.5) and gs > 0
        group = "Starting Pitchers" if is_starter else "Right-Handed Relievers"
        incumbent = incumbents.get(group)
        if incumbent is None or name == incumbent.get("name"):
            continue
        ip = pd.to_numeric(row.get("IP"), errors="coerce") or 0
        era = pd.to_numeric(row.get("ERA"), errors="coerce")
        if ip <= 0 or pd.isna(era):
            continue
        rows.append({
            "name": name, "position_group": group, "source": "Internal (roster)",
            "ERA": era, "IP": ip, "matched": True,
        })

    return pd.DataFrame(rows)


def build_mariners_incumbents(data: dict) -> dict:
    """
    Real, current Mariners production by position group -- for each
    real MLBTR position label, finds whichever Seattle player has the
    most playing time at that primary position (the real incumbent),
    using the same real Seattle batting/pitching tables already
    scraped elsewhere in this project.

    Returns {position_group: {"name":..., "OPS":..., "PA":...}} for
    batters, plus a single "Starting Pitchers"/"Right-Handed Relievers"
    rotation/bullpen-average entry for pitchers (individual matchups
    make less sense there than a real staff-wide baseline).
    """
    incumbents = {}
    bat = data["seattle"]["batting"]

    for _, row in bat.iterrows():
        primary = _parse_primary_position(row.get("Pos"))
        group = BBREF_TO_MLBTR_POS.get(primary)
        if not group:
            continue
        pa = pd.to_numeric(row.get("PA"), errors="coerce") or 0
        existing = incumbents.get(group)
        if existing is None or pa > existing.get("PA", 0):
            incumbents[group] = {
                "name": row.get("Name"), "OPS": pd.to_numeric(row.get("OPS"), errors="coerce"),
                "PA": pa,
            }

    # real, current team-wide rotation/bullpen baselines (an individual
    # pitcher-vs-pitcher matchup is less meaningful than "how does this
    # free agent compare to what the real staff has been doing")
    pit = data["seattle"]["pitching"]
    if not pit.empty:
        pit_num = pit.copy()
        pit_num["IP"] = pd.to_numeric(pit_num["IP"], errors="coerce")
        pit_num["ERA"] = pd.to_numeric(pit_num["ERA"], errors="coerce")
        starters = pit_num[pd.to_numeric(pit_num["GS"], errors="coerce") >= 5]
        relievers = pit_num[pd.to_numeric(pit_num["GS"], errors="coerce") < 5]
        if not starters.empty:
            rot_era = (starters["ERA"] * starters["IP"]).sum() / starters["IP"].sum()
            incumbents["Starting Pitchers"] = {"name": "Real rotation average", "ERA": round(rot_era, 2)}
        if not relievers.empty:
            bp_era = (relievers["ERA"] * relievers["IP"]).sum() / relievers["IP"].sum()
            for group in ["Right-Handed Relievers", "Left-Handed Relievers"]:
                incumbents[group] = {"name": "Real bullpen average", "ERA": round(bp_era, 2)}

    return incumbents


def add_upgrade_deltas(enriched: pd.DataFrame, incumbents: dict) -> pd.DataFrame:
    df = enriched[enriched["matched"]].copy()
    df["incumbent_name"] = None
    df["incumbent_stat"] = None
    df["upgrade"] = None

    for idx, row in df.iterrows():
        inc = incumbents.get(row["position_group"])
        if inc is None:
            continue
        df.at[idx, "incumbent_name"] = inc["name"]
        if "OPS" in inc and pd.notna(row.get("OPS")):
            df.at[idx, "incumbent_stat"] = inc["OPS"]
            df.at[idx, "upgrade"] = round(row["OPS"] - inc["OPS"], 3) if pd.notna(inc["OPS"]) else None
        elif "ERA" in inc and pd.notna(row.get("ERA")):
            df.at[idx, "incumbent_stat"] = inc["ERA"]
            # lower ERA is better, so upgrade = incumbent ERA - FA ERA
            df.at[idx, "upgrade"] = round(inc["ERA"] - row["ERA"], 2)

    return df


def save_top_upgrade_chart(df: pd.DataFrame, position_group: str, top_n: int = 5):
    """Saves a real .png bar chart comparing the current Mariners
    incumbent against the top real upgrade options at one position."""
    subset = df[(df["position_group"] == position_group) & df["upgrade"].notna()]
    subset = subset.sort_values("upgrade", ascending=False).head(top_n)
    if subset.empty:
        return None

    is_pitcher = "ERA" in subset.columns and subset["ERA"].notna().any()
    stat_col = "ERA" if is_pitcher else "OPS"
    incumbent_name = subset.iloc[0]["incumbent_name"]
    incumbent_val = subset.iloc[0]["incumbent_stat"]

    labels = [f"{incumbent_name}\n(current)"] + subset["name"].tolist()
    values = [incumbent_val] + subset[stat_col].tolist()
    colors = ["#888888"] + ["#2ca02c"] * len(subset)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color=colors)
    ax.set_title(f"{position_group}: Current Mariners vs. Top Free Agent Options ({stat_col})")
    ax.set_ylabel(stat_col)
    if is_pitcher:
        ax.invert_yaxis()  # lower ERA = better, so flip for visual clarity
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    safe_name = re.sub(r"[^\w]+", "_", position_group.lower())
    path = os.path.join(OUTPUT_DIR, f"{safe_name}_upgrade.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ── diagnosis-weighted scoring ────────────────────────────────────────────────
def load_diagnosed_needs() -> pd.DataFrame:
    """Loads the ranked weakness/strength table from team_diagnosis.py.
    Falls back to an empty DataFrame (with a warning) if diagnosis
    hasn't been run yet -- everything below is written to degrade
    gracefully to the old incumbent-only behavior (multiplier 1.0
    everywhere, no contact/power fit bonus) when this is empty, so this
    file still runs even on a machine where team_diagnosis.py hasn't
    been run yet."""
    if not os.path.exists(NEEDS_CSV):
        print("  [warn] data/diagnosed_needs.csv not found -- run "
              "team_diagnosis.py first for need-weighted scoring. "
              "Falling back to incumbent-only need weighting for now.")
        return pd.DataFrame()
    return pd.read_csv(NEEDS_CSV)


def _category_severity(needs: pd.DataFrame, category: str, metrics: list = None) -> float:
    """Average Severity_Score for a diagnosis category (optionally
    restricted to specific metric rows). Positive = net weakness,
    negative = net strength, 0 = no signal (empty/missing)."""
    if needs.empty or "Severity_Score" not in needs.columns:
        return 0.0
    rows = needs[needs["Category"] == category]
    if metrics:
        rows = rows[rows["Metric"].isin(metrics)]
    return rows["Severity_Score"].mean() if not rows.empty else 0.0


def get_team_need_multipliers(needs: pd.DataFrame) -> dict:
    """
    Turns team_diagnosis.py's ranked severity table into a per-category
    multiplier layered ON TOP of the existing position-level need_weight
    in compute_recommendation_score (which already correctly identifies
    WHICH specific position is weak, from real incumbent OPS/ERA). This
    adds a signal per-position comparison can't see on its own: team
    _diagnosis.py can tell you the whole PITCHING staff's bullpen is
    homer-prone, or the whole OFFENSE has a real hit-tool problem, in a
    way no single position's incumbent stat captures.

    DEFENSE-ADJUSTED BULLPEN DISCOUNT: a real run of team_diagnosis.py
    flagged team fielding (Rtot) as a severe weakness (-30 runs) at the
    same time the bullpen's ERA/WHIP were flagged weak. A meaningful
    chunk of "the bullpen looks bad" is downstream of the defense
    playing behind it, not the relievers themselves -- so when BOTH are
    flagged, this discounts the bullpen multiplier proportional to how
    much of the combined signal is coming from the glove work, capped
    at a 50% discount so a real bullpen need never gets fully explained
    away by defense alone (there's no way to know the exact split
    without pitch-by-pitch batted-ball data this project doesn't have,
    so this is a deliberately conservative, bounded adjustment, not a
    precise attribution).
    """
    if needs.empty:
        return {"Batting": 1.0, "Rotation": 1.0, "Bullpen": 1.0}

    batting_sev  = _category_severity(needs, "Batting")
    rotation_sev = _category_severity(needs, "Rotation")
    bullpen_sev  = _category_severity(needs, "Bullpen")
    fielding_sev = _category_severity(needs, "Fielding")

    def scale(sev):
        # +1 severity point (roughly "one threshold past average") nudges
        # the multiplier by 8%, bounded to [0.75, 1.6] -- meant to nudge
        # the existing position-level ranking, not override it outright
        return max(0.75, min(1.6, 1.0 + sev * 0.08))

    bullpen_mult = scale(bullpen_sev)
    if bullpen_sev > 0 and fielding_sev > 0:
        discount = min(0.5, fielding_sev / (bullpen_sev + fielding_sev))
        bullpen_mult = 1.0 + (bullpen_mult - 1.0) * (1 - discount)

    return {
        "Batting": scale(batting_sev),
        "Rotation": scale(rotation_sev),
        "Bullpen": bullpen_mult,
    }


def get_batting_need_profile(needs: pd.DataFrame) -> dict:
    """
    Reads whether the team's real batting weakness is specifically a
    hit-tool problem (BA/K% down) vs. a power problem (ISO itself
    flagged) vs. a plate-discipline problem (BB% flagged) -- confirmed
    from a real run where BA/SLG/OPS were weaknesses but ISO/K%/BB% all
    came back AVERAGE, meaning the real gap is contact/hit-tool, not
    power or patience. A free agent who fixes the SPECIFIC diagnosed gap
    should outrank one with a similar OPS built the "wrong" way (e.g. a
    three-true-outcomes power bat doesn't fix a contact problem even at
    a similar OPS to a real average-hitting alternative).

    Returns continuous weights (not a single label) so
    contact_power_fit_bonus() can blend them rather than hard-branching.
    """
    profile = {"contact_weight": 0.0, "power_weight": 0.0}
    if needs.empty:
        return profile
    lookup = {row["Metric"]: row for _, row in needs[needs["Category"] == "Batting"].iterrows()}

    if "BA" in lookup and lookup["BA"]["Verdict"] == "WEAKNESS":
        profile["contact_weight"] += lookup["BA"]["Severity_Score"]
    if "K_pct" in lookup and lookup["K_pct"]["Verdict"] == "WEAKNESS":
        profile["contact_weight"] += lookup["K_pct"]["Severity_Score"] * 0.5
    if "ISO" in lookup and lookup["ISO"]["Verdict"] == "WEAKNESS":
        profile["power_weight"] += lookup["ISO"]["Severity_Score"]
    return profile


def contact_power_fit_bonus(row: pd.Series, profile: dict) -> float:
    """Small, bounded bonus/penalty (+/-20%, before clamping) rewarding a
    free agent whose own real BA/power split matches whichever specific
    offensive gap the team actually has -- computed from BA/SLG already
    present on every matched batter row (ISO = SLG - BA), no new
    scraping needed. Returns 1.0 (no effect) when there's no diagnosed
    profile yet or the row is missing BA/SLG."""
    ba, slg = row.get("BA"), row.get("SLG")
    if pd.isna(ba) or pd.isna(slg) or (profile["contact_weight"] <= 0 and profile["power_weight"] <= 0):
        return 1.0
    iso = slg - ba
    bonus = 1.0
    if profile["contact_weight"] > 0:
        bonus += max(-0.2, min(0.2, (ba - LEAGUE_AVG["BA"]) / 0.05 * 0.2))
    if profile["power_weight"] > 0:
        bonus += max(-0.2, min(0.2, (iso - LEAGUE_AVG["ISO"]) / 0.05 * 0.2))
    return max(0.7, min(1.3, bonus))


def get_fielding_bonus(rtot, needs: pd.DataFrame) -> float:
    """
    Small defense bonus/penalty from a real per-player Rtot/Rdrs value.

    GAP CLOSED 2026-09-28: this used to be internal-candidates-only,
    since Rtot only existed for Mariners players (bbref's team-level
    fielding table). free_agent_stats_matcher.py now also matches every
    free agent against bbref's LEAGUE-WIDE fielding leaderboard
    (battingpitching_scraper.get_fielding()), so an external free
    agent's row can carry a real "Rtot" too -- this function doesn't
    care where the value came from, it just uses it when present.

    Not every free agent will have a match (part-timers, DH-only
    players with minimal defensive innings, or a name-format mismatch
    between the two independently-scraped sources) -- those come
    through as NaN and this is correctly a no-op (1.0) for them, same
    honest "matched" pattern used elsewhere in this pipeline rather than
    silently assuming average defense for someone never actually
    checked.

    Only applied when team_diagnosis.py has actually flagged team
    fielding as a real weakness; otherwise a no-op (1.0) regardless of
    whether a real Rtot value exists.
    """
    if needs.empty or pd.isna(rtot):
        return 1.0
    if _category_severity(needs, "Fielding") <= 0:
        return 1.0
    return max(0.85, min(1.3, 1.0 + rtot / 50))


def compute_recommendation_score(df: pd.DataFrame, incumbents: dict,
                                 needs: pd.DataFrame = None) -> pd.DataFrame:
    """
    A genuine, combined recommendation score -- not just a raw stat
    delta sort. Real adjustments on top of the upgrade delta:

    1. SCALE NORMALIZATION: OPS deltas and ERA deltas aren't
       comparable numbers (a 4.48 ERA delta and a .213 OPS delta don't
       mean the same real thing) -- normalizes each onto a comparable
       "how many real, meaningful chunks of improvement is this" scale
       (.100 OPS = 1.0 point; 1.00 ERA = 1.0 point, both genuinely
       substantial real differences in baseball terms).

    2. POSITION-LEVEL NEED WEIGHTING: an upgrade at a real, confirmed
       weak position should outrank a bigger raw upgrade at a position
       that's already fine. Positions below the team's own average
       incumbent OPS/ERA get a 1.5x multiplier; everyone else gets 1.0x.

    3. TEAM-LEVEL DIAGNOSIS MULTIPLIER (new): layered on top of #2 using
       team_diagnosis.py's category severity -- see
       get_team_need_multipliers()'s docstring for the defense-adjusted
       bullpen discount specifically.

    4. CONTACT/POWER FIT BONUS (new, batters only): rewards a free agent
       whose own BA/ISO profile matches the SPECIFIC diagnosed offensive
       gap (hit-tool vs. power vs. patience) -- see
       get_batting_need_profile()/contact_power_fit_bonus().

    5. FIELDING BONUS: rewards a real plus glove (Rtot) when team
       defense is diagnosed as weak -- now works for BOTH internal
       candidates (always had real Rtot, from bbref's team-level
       fielding table) AND external free agents (now matched against
       bbref's league-wide fielding leaderboard too -- see
       get_fielding_bonus()'s docstring for that gap being closed, and
       free_agent_stats_matcher.py's "field_matched" flag for which
       free agents actually got a real match vs. came back unmatched).

    6. RELIABILITY DISCOUNT: same shrinkage philosophy already used in
       player_grades.py all night -- a great rate over a tiny sample
       (15 PA, 5 IP) is noisy and shouldn't be trusted at face value.
       Shrinks the upgrade toward zero for small samples using the same
       k-based approach (k=100 PA for batters, k=30 IP for pitchers,
       roughly matching established sabermetric convention for how much
       regression a given sample size needs).
    """
    df = df.copy()
    df["scaled_upgrade"] = None
    df["reliability"] = None
    df["need_weight"] = None
    df["diagnosis_multiplier"] = None
    df["fit_bonus"] = None
    df["recommendation_score"] = None

    needs = needs if needs is not None else pd.DataFrame()
    team_multipliers = get_team_need_multipliers(needs)
    batting_profile = get_batting_need_profile(needs)

    # BUG FIX: this used to pick a FIXED COUNT as "weak" (bottom 3 of
    # 9 batting groups, bottom 2 of only 3 pitching groups) -- a real
    # run showed this treats 33% of batting spots as "weak" but 67% of
    # pitching spots as "weak", a completely different, inconsistent
    # bar for the two sides. With only 3 real pitching groups (SP/RHP/
    # LHP) and the bullpen's real ERA already worse than the rotation's,
    # the bullpen was ALMOST STRUCTURALLY GUARANTEED to count as "weak"
    # regardless of how it actually compared to real batting needs --
    # not a fair, principled standard, just an artifact of pitching
    # having fewer buckets to begin with. Uses a consistent real
    # threshold instead: below the TEAM'S OWN average performance
    # level, the same standard applied identically to both sides
    # regardless of how many groups either category happens to have.
    batter_incumbents = {g: info for g, info in incumbents.items() if "OPS" in info}
    pitcher_incumbents = {g: info for g, info in incumbents.items() if "ERA" in info}

    avg_ops = (sum(info["OPS"] for info in batter_incumbents.values())
              / len(batter_incumbents)) if batter_incumbents else None
    avg_era = (sum(info["ERA"] for info in pitcher_incumbents.values())
              / len(pitcher_incumbents)) if pitcher_incumbents else None

    weak_batter_positions = {g for g, info in batter_incumbents.items()
                             if avg_ops is not None and info["OPS"] < avg_ops}
    weak_pitcher_positions = {g for g, info in pitcher_incumbents.items()
                              if avg_era is not None and info["ERA"] > avg_era}
    weak_positions = weak_batter_positions | weak_pitcher_positions

    for idx, row in df.iterrows():
        if pd.isna(row.get("upgrade")):
            continue

        is_pitcher = "ERA" in row and pd.notna(row.get("ERA"))
        if is_pitcher:
            scaled = row["upgrade"] / 1.00
            sample = row.get("IP", 0) or 0
            k = 30  # real innings-based shrinkage constant
        else:
            scaled = row["upgrade"] / 0.100
            sample = row.get("PA", 0) or 0
            k = 100  # real PA-based shrinkage constant, matching
                     # player_grades.py's own established convention

        reliability = sample / (sample + k) if sample else 0
        position_need_weight = 1.5 if row["position_group"] in weak_positions else 1.0

        diagnosis_category = POSITION_GROUP_TO_DIAGNOSIS_CATEGORY.get(row["position_group"])
        diagnosis_mult = team_multipliers.get(diagnosis_category, 1.0)

        if is_pitcher:
            fit_bonus = 1.0
        else:
            fit_bonus = contact_power_fit_bonus(row, batting_profile)
            # fielding bonus now applies regardless of source -- internal
            # candidates always had a real Rtot (team-level fielding
            # table); external free agents can have one too now that
            # free_agent_stats_matcher.py matches against bbref's
            # league-wide fielding leaderboard. get_fielding_bonus()
            # itself is a no-op (1.0) when Rtot is NaN, so an unmatched
            # free agent's score is unaffected either way.
            fit_bonus *= get_fielding_bonus(row.get("Rtot"), needs)

        combined_need_weight = position_need_weight * diagnosis_mult

        df.at[idx, "scaled_upgrade"] = round(scaled, 2)
        df.at[idx, "reliability"] = round(reliability, 2)
        df.at[idx, "need_weight"] = round(combined_need_weight, 2)
        df.at[idx, "diagnosis_multiplier"] = round(diagnosis_mult, 2)
        df.at[idx, "fit_bonus"] = round(fit_bonus, 2)
        df.at[idx, "recommendation_score"] = round(
            scaled * reliability * combined_need_weight * fit_bonus, 2)

    return df


def print_recommendations(df: pd.DataFrame, top_n: int = 15):
    scored = df[df["recommendation_score"].notna()].copy()
    scored = scored.sort_values("recommendation_score", ascending=False)

    print("\n" + "=" * 110)
    print("GENUINE RECOMMENDATIONS -- scale-normalized, need-weighted (position + team diagnosis), "
          "fit-adjusted, reliability-discounted")
    print("=" * 110)
    print(f"  {'Name':<20} {'Source':<18} {'Position':<20} {'Raw upg':>8} "
          f"{'Reliab':>7} {'NeedW':>6} {'Fit':>5} {'SCORE':>7}")
    print("  " + "-" * 100)
    for _, row in scored.head(top_n).iterrows():
        print(f"  {row['name']:<20} {row.get('source','?'):<18} "
              f"{row['position_group']:<20} "
              f"{row['upgrade']:>8.2f} "
              f"{row['reliability']:>7.2f} "
              f"{row['need_weight']:>6.2f} "
              f"{row['fit_bonus']:>5.2f} "
              f"{row['recommendation_score']:>7.2f}")
    print("=" * 110)
    print("  Score = (upgrade / real scale) * (sample / (sample + k)) * need_weight * fit_bonus")
    print("  need_weight = position-level need (1.0/1.5) * team diagnosis category multiplier")
    print("    (see team_diagnosis.py / data/diagnosed_needs.csv -- run that first for this to be live)")
    print("  fit_bonus = contact/power profile match (batters) * defense bonus (when a real Rtot match exists)")
    print("  Reliability approaches 1.0 with a real, substantial sample;")
    print("  stays low for tiny samples regardless of how good the raw rate looks.")
    print("=" * 110 + "\n")


def print_depth_chart(scored: pd.DataFrame, incumbents: dict, position_group: str):
    """
    Real depth-chart-style view for ONE position -- every real,
    matched free agent option at that spot, side by side on the full
    real stat line (not collapsed to a single number), ranked by
    recommendation_score, with the current Mariners incumbent shown
    for direct reference at the top.
    """
    incumbent = incumbents.get(position_group)
    options = scored[(scored["position_group"] == position_group)
                     & scored["matched"]].copy()
    is_pitcher = "ERA" in options.columns and options["ERA"].notna().any()

    print("\n" + "=" * 100)
    print(f"DEPTH CHART -- {position_group}")
    print("=" * 100)

    if incumbent:
        stat = incumbent.get("ERA", incumbent.get("OPS"))
        print(f"  CURRENT: {incumbent['name']} ({stat})")
        print("  " + "-" * 96)

    if options.empty:
        print("  No real matched free agent options at this position.")
        print("=" * 100 + "\n")
        return

    if is_pitcher:
        cols = ["name", "Team", "IP", "ERA", "WHIP", "SO", "BB", "WAR", "recommendation_score"]
        sort_col = "recommendation_score"
    else:
        cols = ["name", "Team", "PA", "OPS", "OBP", "SLG", "HR", "RBI", "BA", "WAR", "recommendation_score"]
        sort_col = "recommendation_score"

    real_cols = [c for c in cols if c in options.columns]
    options_sorted = options.sort_values(sort_col, ascending=False, na_position="last")
    print(options_sorted[real_cols].to_string(index=False))
    print("=" * 100 + "\n")


BENCH_PA_MIN  = 150   # real enough sample to trust, not just a cup of coffee
BENCH_PA_MAX  = 450   # above this they're basically a full-time player, not a bench piece
BENCH_OPS_MIN = 0.680  # replacement-level-or-better, NOT a starter bar


def find_bench_candidates(enriched: pd.DataFrame,
                          position_groups=("Left Fielders", "Center Fielders",
                                           "Right Fielders", "Designated Hitters")
                          ) -> pd.DataFrame:
    """
    A genuinely different question than compute_recommendation_score()'s
    "does this beat our starter's OPS" -- a 4th outfielder / bench bat
    isn't trying to out-produce Arozarena or Julio, they're trying to be
    a competent complementary option in limited playing time. Filtering
    by "beats the incumbent" (the existing recommendation logic) has no
    lane for that question at all, which is why a real user correctly
    pointed out Robles-caliber bench needs never showed up in the
    original recommendation output -- it was never being asked.

    Filters real matched free agents down to a genuine part-time PA
    range (BENCH_PA_MIN-BENCH_PA_MAX) with at least a replacement-level
    OPS (BENCH_OPS_MIN), independent of whatever the starter at that
    spot is doing.

    REAL GAP THIS DOESN'T CLOSE: there's no bats-handed data anywhere in
    this pipeline -- bbref's league batting leaderboard (what
    free_agent_stats_matcher.py scrapes) doesn't carry a Bats column,
    and per get_platoon_splits()'s own docstring, bulk per-player
    split/bio scraping for the full ~278-player free agent pool was
    already tried and ruled genuinely impractical (rate-limiting/
    blocking). This returns real, qualified bench-caliber names;
    confirming which ones actually bat right-handed (or how they
    actually hit lefties/righties) still needs a manual, one-player-
    at-a-time check -- get_platoon_splits(), or a targeted web search --
    on whichever names from this list look like real fits, the same
    honest, scoped approach already used elsewhere in this file rather
    than a new scraper likely to get blocked.
    """
    if enriched.empty:
        return pd.DataFrame()

    d = enriched[enriched["matched"] & enriched["position_group"].isin(position_groups)].copy()
    d["PA"]  = pd.to_numeric(d.get("PA"), errors="coerce")
    d["OPS"] = pd.to_numeric(d.get("OPS"), errors="coerce")
    d = d[(d["PA"] >= BENCH_PA_MIN) & (d["PA"] <= BENCH_PA_MAX)
         & (d["OPS"] >= BENCH_OPS_MIN)]

    cols = [c for c in ["name", "Team", "position_group", "PA", "OPS",
                        "OBP", "SLG", "HR", "RBI", "BA", "WAR"]
           if c in d.columns]
    return d[cols].sort_values("OPS", ascending=False).reset_index(drop=True)


def print_bench_candidates(df: pd.DataFrame, top_n: int = 10):
    print("\n" + "=" * 100)
    print("BENCH / DEPTH CANDIDATES -- corner OF/DH, part-time PA range, replacement-level+")
    print("=" * 100)
    if df.empty:
        print("  No real matched free agents in the bench PA/OPS range.")
    else:
        print(df.head(top_n).to_string(index=False))
    print("=" * 100)
    print(f"  Filter: {BENCH_PA_MIN}-{BENCH_PA_MAX} PA, OPS >= {BENCH_OPS_MIN} "
          f"(NOT compared against the starter -- a different question)")
    print("  NOTE: no bats-handed data in this pipeline. To check a specific")
    print("  name's real platoon fit, use get_platoon_splits(name) or a")
    print("  manual web search -- bulk splits for the full FA pool were")
    print("  already confirmed impractical (see get_platoon_splits() docstring).")
    print("=" * 100 + "\n")


def get_platoon_splits(player_name: str) -> dict:
    """
    Real, ON-DEMAND single-player platoon split lookup -- vs LHP / vs
    RHP splits live on a separate bbref page PER PLAYER, not in the
    league-wide leaderboard tables this project already scrapes. Doing
    this in bulk for all 278 free agents would mean 278 separate page
    fetches, which is genuinely impractical and likely to get rate-
    limited or blocked. This is the honest, realistic scope instead --
    a targeted lookup for one specific player at a time, the same real
    approach used manually for Jake Bauers earlier tonight.

    NOT yet wired to a live fetch -- bbref's real per-player splits
    page structure would need to be confirmed the same way every other
    scraper in this project was (a debug pass against the real page)
    before this can pull real data automatically. Returns a clear
    placeholder for now rather than fabricating numbers.
    """
    print(f"  [info] real, automated platoon-split fetching for "
          f"'{player_name}' isn't wired up yet -- would need a real "
          f"bbref splits-page scraper (one player at a time), built "
          f"and debug-verified the same way every other scraper in "
          f"this project was. For now, check a specific player's real "
          f"splits manually (web search + bbref's own splits page) "
          f"the same way this got answered for Jake Bauers tonight.")
    return {"name": player_name, "vs_LHP_OPS": None, "vs_RHP_OPS": None,
            "note": "not yet implemented -- see function docstring"}


def generate_comparison_html(scored: pd.DataFrame, incumbents: dict) -> str:
    """
    Real, reusable version of the side-by-side matchup-style comparison
    -- automatically built for EVERY position with a real upgrade,
    using whichever free agent actually scored highest at that
    position (recommendation_score), not a hand-picked example. Saves
    a real, standalone .html file that opens in any browser.
    """
    best_per_position = (scored[scored["recommendation_score"].notna()]
                         .sort_values("recommendation_score", ascending=False)
                         .groupby("position_group").first())

    rows_html = ""
    for group, target in best_per_position.iterrows():
        incumbent = incumbents.get(group)
        if incumbent is None:
            continue
        is_pitcher = "ERA" in incumbent
        stat_label = "ERA" if is_pitcher else "OPS"
        incumbent_stat = incumbent.get("ERA" if is_pitcher else "OPS")
        target_stat = target.get("ERA" if is_pitcher else "OPS")
        if pd.isna(incumbent_stat) or pd.isna(target_stat):
            continue

        # for ERA, lower is better; for OPS, higher is better
        target_wins = (target_stat < incumbent_stat) if is_pitcher else (target_stat > incumbent_stat)
        check_left = "" if target_wins else "&#10003;"
        check_right = "&#10003;" if target_wins else ""

        # BUG FIX: an f-string format spec can't contain a
        # conditional expression ({x:.3f if cond else .2f} is invalid
        # and raises ValueError at runtime, not caught by a plain
        # compile check) -- format each stat as a plain string first.
        incumbent_stat_str = f"{incumbent_stat:.2f}" if is_pitcher else f"{incumbent_stat:.3f}"
        target_stat_str = f"{target_stat:.2f}" if is_pitcher else f"{target_stat:.3f}"

        rows_html += f"""
        <tr>
          <td class="name-cell">
            <div class="name">{incumbent['name']}</div>
            <div class="stat">{incumbent_stat_str} {stat_label}</div>
          </td>
          <td class="check-cell">{check_left}</td>
          <td class="pos-cell">{group.replace(' ', '<br>')}</td>
          <td class="check-cell">{check_right}</td>
          <td class="name-cell name-cell-right">
            <div class="name">{target['name']}</div>
            <div class="stat">{target_stat_str} {stat_label}</div>
          </td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Mariners vs. Free Agent Targets</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 700px; margin: 2rem auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 10px 12px; color: #666; font-size: 12px; font-weight: 400; }}
  th.right {{ text-align: right; }}
  th.center {{ text-align: center; }}
  td {{ padding: 12px; border-top: 1px solid #e0e0e0; vertical-align: middle; }}
  .name-cell-right {{ text-align: right; }}
  .name {{ font-weight: 500; }}
  .stat {{ color: #666; font-size: 13px; }}
  .check-cell {{ text-align: center; color: #1a9e5c; font-weight: bold; width: 30px; }}
  .pos-cell {{ text-align: center; font-weight: 500; width: 70px; }}
  h1 {{ font-size: 18px; font-weight: 500; }}
</style></head>
<body>
<h1>Seattle Mariners — Current vs. Best Real Free Agent Targets, by Position</h1>
<table>
  <tr><th>Mariners (current)</th><th class="center"></th><th class="center">Pos</th><th class="center"></th><th class="right">Best real target</th></tr>
  {rows_html}
</table>
</body></html>"""

    path = os.path.join(OUTPUT_DIR, "mariners_vs_targets.html")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


if __name__ == "__main__":
    print("Building league + Mariners data (uses cache if fresh)...")
    data = build_all(2026)

    print("Loading roster diagnosis (team_diagnosis.py output)...")
    needs = load_diagnosed_needs()
    if not needs.empty:
        print("  [ok] diagnosed_needs.csv loaded -- recommendation scoring "
              "will use team-level severity + contact/power fit weighting")

    print("Getting the real free agent list...")
    fa = get_mlbtr_free_agents()

    print("Building real Mariners incumbent comparisons...")
    incumbents = build_mariners_incumbents(data)
    for group, info in incumbents.items():
        stat = info.get("OPS", info.get("ERA"))
        print(f"  {group}: {info['name']} ({stat})")

    enriched = enrich_free_agents(fa, data)
    current_roster = _get_current_roster_names(data)
    before_count = len(enriched)
    enriched = enriched[~enriched["name"].isin(current_roster)]
    excluded = before_count - len(enriched)
    if excluded:
        print(f"  [info] excluded {excluded} free agent(s) who are also "
              f"current Mariners (e.g. Crawford, Arozarena) from the "
              f"target pool -- re-signing your own guy isn't a real "
              f"external upgrade to compare against")
    # Pipeline 3: the team's own internal bench/40-man options,
    # scored on the exact same real basis as external free agents --
    # "should we sign someone or just play the guy we already have"
    # is one real question with one real answer, not two disconnected
    # tools that never talk to each other.
    internal = build_internal_candidates(data, incumbents)
    print(f"\nFound {len(internal)} real internal roster candidates "
          f"(non-incumbent Mariners with real 2026 playing time)")

    combined = pd.concat([enriched, internal], ignore_index=True)
    upgraded = add_upgrade_deltas(combined, incumbents)

    # replaces the old raw upgrade-delta sort, which let pitching
    # deltas dominate purely by numeric scale and had no concept of
    # which positions actually need help -- now also folds in
    # team_diagnosis.py's whole-side-of-the-ball severity and the
    # contact/power fit bonus (see compute_recommendation_score's
    # docstring)
    scored = compute_recommendation_score(upgraded, incumbents, needs)
    print_recommendations(scored)

    html_path = generate_comparison_html(scored, incumbents)
    print(f"\nReal comparison page saved: {html_path}")

    # BUG FIX: this used to hardcode ["First Basemen", "Designated
    # Hitters", "Right Fielders"] as "the top needs" -- a real user
    # correctly called this out, since it was never actually derived
    # from the data. A live run showed Designated Hitters (Canzone,
    # .814 OPS) is genuinely one of the STRONGEST positions on the
    # team, not a need at all -- including it in "top needs" was
    # simply wrong. Now picks positions by the real, lowest incumbent
    # OPS/highest incumbent ERA instead -- an actual, principled
    # ranking rather than a guess.
    print("\nReal weakest positions by incumbent performance:")
    batter_incumbents = {g: info for g, info in incumbents.items() if "OPS" in info}
    weakest = sorted(batter_incumbents.items(), key=lambda kv: kv[1]["OPS"])[:3]
    for group, info in weakest:
        print(f"  {group}: {info['name']} ({info['OPS']:.3f} OPS)")

    print("\nSaving real chart files for the actual weakest positions...")
    for group, _ in weakest:
        path = save_top_upgrade_chart(upgraded, group)
        if path:
            print(f"  saved: {path}")

    for group, _ in weakest:
        print_depth_chart(scored, incumbents, group)

    # BUG FIX: the recommendation logic above only ever asks "does this
    # free agent beat our starter" -- a real user correctly pointed out
    # that a bench/depth need (e.g. a complementary corner-OF bat behind
    # a below-average part-time player like Robles) never showed up
    # anywhere, because that's a different question the tool never asked.
    bench = find_bench_candidates(enriched)
    print_bench_candidates(bench)