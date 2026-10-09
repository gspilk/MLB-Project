"""
year_over_year_decomposition.py
Explains HOW a Mariners team stat changed from one season to the next --
not just "team BA went .244 -> .232" but "here's exactly which players
account for that."

Builds directly on two already-proven pieces of this project:
  - mariners_stats.get_seattle_stats(season) -- same per-player batting/
    pitching/fielding tables multi_year_comparison.py already uses (real
    columns confirmed against that file: Name/PA/BA/OBP/SLG/OPS/HR for
    batting, Name/G/GS/IP/ERA/WHIP/SO/BB/W/L for pitching, Name/.../Rtot
    for fielding -- same shape team_diagnosis.py and stat_breakdown.py
    already trust).
  - The PA/IP-weighted "contribution" math from team_diagnosis.py /
    stat_breakdown.py -- same idea, just pointed at two seasons of the
    SAME team instead of one season vs. the league average.

THE MATH
--------
For a rate stat (BA, OBP, SLG, OPS, ERA, WHIP), each player's weighted
contribution to the team rate in a season is:

    contribution_i = (weight_i / total_weight) * rate_i

  (weight = PA for batting, IP for pitching -- same convention as
  team_diagnosis.py). These contributions sum EXACTLY to the team's real
  weighted-average rate for that season (same identity multi_year_
  comparison.py's team_BA/team_ERA already rely on).

A first version of this compared each player's raw contribution to
ZERO for players missing from one season (departed/new). That's
provably wrong and was caught in testing: since a rate's weighted
shares always sum to 1.0 within a season, a departing player's term
disappearing is mechanically "positive" for a lower-is-better stat
NO MATTER how good he was -- it showed a 2.80-ERA reliever's departure
as helping team ERA. The fix: every player's term is measured as
weight * (rate - {season_A}_team_average) -- a GAP, exactly like
team_diagnosis.py's LEAGUE_AVG gaps, just referenced against this
team's own prior season instead of the league. The year-over-year
CHANGE in that gap-weighted term:

    residual_i = weight_i,B * (rate_i,B - ref) - weight_i,A * (rate_i,A - ref)

(where ref = season A's real team average, weight_i = 0 for a season a
player didn't play in) sums EXACTLY to the real team-level change for
ANY fixed reference value -- verified algebraically and in
test_year_over_year.py -- while now giving the correct sign for every
category: a departed player who was BETTER than that season's team
average always shows as a real loss, a new player better than it always
shows as a real gain, for every stat direction.

For a counting stat (HR, fielding Rtot), there's no weighting at all --
delta_i = value_i(season_B) - value_i(season_A), and these already sum
exactly to the team-level change (same additive convention
team_diagnosis.py's fielding decomposition already uses, since Rtot is
already a runs-above/below-average number, not a rate).

Contribution_Pts is sign-flipped for "lower is better" stats (ERA, WHIP)
so that POSITIVE Contribution_Pts ALWAYS means "this player made the
team stat move in the good direction year-over-year" and NEGATIVE always
means "made it worse" -- same convention team_diagnosis.py's
Severity_Score already uses for LOWER_IS_BETTER stats, so reading this
output doesn't require remembering which raw direction is good per stat.

NAME MATCHING ACROSS SEASONS
-----------------------------
Both seasons come from the same source (bbref) so names are spelled
consistently -- the only real mismatch risk is accented characters
(e.g. "Julio Rodríguez") rendering identically both years, which they do
on bbref. _normalize_name() strips accents/case/punctuation before
matching so this isn't an issue in practice; no fuzzy/partial matching
is attempted, since two full MLB rosters from the same source is a much
safer matching problem than the free-agent-name-matching one this
project already solved carefully elsewhere -- an exact normalized match
is enough here, and if it ever isn't, a player would show up as a false
"departed" + false "new" pair, which is very easy to spot in output.

Every reported table includes a printed reconciliation check
(sum of deltas vs. the real team-level change) so a silent math bug
can't hide -- same "prove it sums exactly" convention as
team_diagnosis.py and stat_breakdown.py.

Usage:
    python year_over_year_decomposition.py --from-season 2025 --to-season 2026
"""

import argparse
import unicodedata

import pandas as pd

from mariners_stats import get_seattle_stats

LOWER_IS_BETTER = {"ERA", "WHIP", "BB9"}


# ── name normalization ───────────────────────────────────────────────────────
def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.strip().lower()


# ── weighted contribution table for one season ───────────────────────────────
def _weighted_table(df: pd.DataFrame, weight_col: str, stat_col: str) -> dict:
    """
    {normalized_name: {"name": real name, "weight": raw weight,
                        "share": weight / total_weight, "rate": stat value}}

    BUG FIX (found via a real run, not obvious from synthetic data alone):
    a player who appears on the team page under the same name in MORE
    THAN ONE ROW for a single season (optioned/recalled multiple times,
    DFA'd and reclaimed, a mid-season role change bbref splits into two
    lines, etc. -- confirmed as the real cause of a real run showing
    "reconciles exactly: False" for pitching ERA/WHIP specifically,
    where multi-stint relievers are common, while batting/fielding
    reconciled fine) used to silently OVERWRITE the earlier row in this
    dict, since it's keyed by normalized name -- that row's weight stays
    counted in `total` (the denominator) but vanishes from the
    numerator, so the remaining shares no longer sum to 1.0 and the
    whole "deltas sum exactly to the team-level change" identity this
    module depends on breaks. Fixed by AGGREGATING same-name rows within
    a season: their weights sum, and their rate becomes the
    weight-blended rate across the stints -- exactly how bbref's own
    season-total rows are computed, so this matches the real, intended
    per-player season line rather than an arbitrary single row.
    """
    if df is None or df.empty or weight_col not in df.columns or stat_col not in df.columns:
        return {}
    d = df.dropna(subset=[weight_col, stat_col]).copy()
    d = d[d[weight_col] > 0]
    total = d[weight_col].sum()
    if total == 0:
        return {}

    agg = {}
    for _, row in d.iterrows():
        key = _normalize_name(row["Name"])
        if not key:
            continue
        w = float(row[weight_col])
        if key not in agg:
            agg[key] = {"name": row["Name"], "weight_sum": 0.0, "weighted_rate_sum": 0.0}
        agg[key]["weight_sum"] += w
        agg[key]["weighted_rate_sum"] += w * float(row[stat_col])

    out = {}
    for key, v in agg.items():
        out[key] = {
            "name":   v["name"],
            "weight": v["weight_sum"],
            "share":  v["weight_sum"] / total,
            "rate":   v["weighted_rate_sum"] / v["weight_sum"],
        }
    return out


def decompose_rate_stat(df_a: pd.DataFrame, df_b: pd.DataFrame,
                        weight_col: str, stat_col: str,
                        label_a: str, label_b: str) -> dict:
    """
    Year-over-year decomposition of a PA/IP-weighted rate stat.
    Returns {"team_a", "team_b", "team_delta", "players": DataFrame,
             "reconciles": bool}.

    IMPORTANT METHOD NOTE (found via testing, not obvious up front):
    a player's contribution is computed from their weighted value ABOVE
    or BELOW the {label_a} team average (a "gap", exactly like
    team_diagnosis.py's LEAGUE_AVG gaps) -- NOT from their raw weighted
    rate compared to zero. This matters specifically for departed/new
    players. A naive "compare to zero" version was tried first and is
    provably wrong: since a rate stat's weighted shares always sum to
    1.0 within a season, a departing player's own weighted term
    disappearing is mechanically "positive" for a lower-is-better stat
    REGARDLESS of whether he was a good or bad pitcher (any positive
    ERA vanishing looks like an improvement in isolation) -- it showed
    a 2.80-ERA reliever's departure as "helping" team ERA and a new
    3.20-ERA arm's arrival as "hurting" it, both backwards. Comparing
    to the {label_a} team average instead fixes this: a departed player
    who was BETTER than that average always shows as a real loss
    (negative Contribution_Pts) and a new player BETTER than that
    average always shows as a real gain (positive), for every stat
    direction, while the exact reconciliation identity below still
    holds for any fixed reference value -- verified in
    test_year_over_year.py.
    """
    table_a = _weighted_table(df_a, weight_col, stat_col)
    table_b = _weighted_table(df_b, weight_col, stat_col)

    team_a = sum(v["share"] * v["rate"] for v in table_a.values())
    team_b = sum(v["share"] * v["rate"] for v in table_b.values())
    team_delta = team_b - team_a
    reference = team_a  # fixed baseline: bridging FROM season A's own team average

    lower_is_better = stat_col in LOWER_IS_BETTER
    rows = []
    delta_sum = 0.0
    for key in set(table_a) | set(table_b):
        a = table_a.get(key)
        b = table_b.get(key)
        term_a = a["share"] * (a["rate"] - reference) if a else 0.0
        term_b = b["share"] * (b["rate"] - reference) if b else 0.0
        residual = term_b - term_a
        delta_sum += residual

        if a and b:
            status = "returning"
        elif a and not b:
            status = "departed"
        else:
            status = "new"

        rows.append({
            "Name":   (b or a)["name"],
            "status": status,
            f"{label_a}_{stat_col}": round(a["rate"], 3) if a else None,
            f"{label_b}_{stat_col}": round(b["rate"], 3) if b else None,
            f"{label_a}_share_pct": round(a["share"] * 100, 1) if a else 0.0,
            f"{label_b}_share_pct": round(b["share"] * 100, 1) if b else 0.0,
            "Contribution_Pts": round(residual * (-1 if lower_is_better else 1), 5),
        })

    players = (pd.DataFrame(rows)
               .sort_values("Contribution_Pts")
               .reset_index(drop=True))

    return {
        "team_a": round(team_a, 4),
        "team_b": round(team_b, 4),
        "team_delta": round(team_delta, 4),
        "players": players,
        "reconciles": abs(delta_sum - team_delta) < 1e-6,
    }


def decompose_sum_stat(df_a: pd.DataFrame, df_b: pd.DataFrame,
                       stat_col: str, label_a: str, label_b: str) -> dict:
    """
    Year-over-year decomposition of a counting/additive stat (HR, fielding
    Rtot) -- no weighting, deltas sum exactly to the team-level change by
    construction.
    """
    def _table(df):
        if df is None or df.empty or stat_col not in df.columns or "Name" not in df.columns:
            return {}
        d = df.dropna(subset=[stat_col])
        out = {}
        for _, row in d.iterrows():
            key = _normalize_name(row["Name"])
            if not key:
                continue
            out[key] = {"name": row["Name"], "value": float(row[stat_col])}
        return out

    table_a = _table(df_a)
    table_b = _table(df_b)

    team_a = sum(v["value"] for v in table_a.values())
    team_b = sum(v["value"] for v in table_b.values())
    team_delta = team_b - team_a

    rows = []
    delta_sum = 0.0
    for key in set(table_a) | set(table_b):
        a = table_a.get(key)
        b = table_b.get(key)
        val_a = a["value"] if a else 0.0
        val_b = b["value"] if b else 0.0
        delta = val_b - val_a
        delta_sum += delta

        if a and b:
            status = "returning"
        elif a and not b:
            status = "departed"
        else:
            status = "new"

        rows.append({
            "Name":   (b or a)["name"],
            "status": status,
            f"{label_a}_{stat_col}": val_a if a else None,
            f"{label_b}_{stat_col}": val_b if b else None,
            "Contribution": round(delta, 3),
        })

    # BUG FIX: if stat_col genuinely isn't present in either season's
    # table (e.g. an HR-less batting export), _table() returns {} for
    # both, so `rows` stays empty and pd.DataFrame(rows) has NO columns
    # at all -- sort_values("Contribution") then raised a real KeyError
    # instead of just reporting "no data" like every other empty-input
    # path in this project already does.
    if not rows:
        players = pd.DataFrame(columns=["Name", "status", f"{label_a}_{stat_col}",
                                        f"{label_b}_{stat_col}", "Contribution"])
    else:
        players = (pd.DataFrame(rows)
                   .sort_values("Contribution")
                   .reset_index(drop=True))

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_delta": round(team_delta, 3),
        "players": players,
        "reconciles": abs(delta_sum - team_delta) < 1e-6,
    }


# ── printing ──────────────────────────────────────────────────────────────────
def _print_block(title: str, result: dict, label_a: str, label_b: str,
                 top_n: int = 6, is_rate: bool = True):
    print(f"\n── {title}: {label_a} {result['team_a']} -> {label_b} {result['team_b']} "
          f"(delta {result['team_delta']:+}) ──")
    print(f"   reconciles exactly: {result['reconciles']}")

    df = result["players"]
    contrib_col = "Contribution_Pts" if is_rate else "Contribution"

    hurt = df[df[contrib_col] < 0].sort_values(contrib_col).head(top_n)
    helped = df[df[contrib_col] > 0].sort_values(contrib_col, ascending=False).head(top_n)

    print(f"\n   Biggest drags ({label_a} -> {label_b}):")
    if hurt.empty:
        print("     none")
    else:
        print(hurt.to_string(index=False))

    print(f"\n   Biggest helps ({label_a} -> {label_b}):")
    if helped.empty:
        print("     none")
    else:
        print(helped.to_string(index=False))

    departed = df[df["status"] == "departed"]
    new = df[df["status"] == "new"]
    print(f"\n   Roster turnover: {len(departed)} departed, {len(new)} new "
          f"(of {len(df)} total names touching this stat)")


def compute_all_decompositions(stats_a: dict, stats_b: dict,
                               label_a: str, label_b: str) -> dict:
    """
    Every decomposition this module knows how to run, computed ONCE and
    returned as {label: (result_dict, is_rate)} -- the real shared step
    between print_all_decompositions() (which just formats these for the
    terminal) and anything that wants the raw per-player "players"
    DataFrames instead (e.g. offseason_report.py's CSV export, so a
    Power BI/Excel pass downstream doesn't need to re-derive this math
    itself from scratch).
    """
    results = {}

    for stat in ["BA", "OBP", "SLG", "OPS"]:
        res = decompose_rate_stat(stats_a.get("batting"), stats_b.get("batting"),
                                  "PA", stat, label_a, label_b)
        results[f"Batting {stat}"] = (res, True)

    res = decompose_sum_stat(stats_a.get("batting"), stats_b.get("batting"),
                             "HR", label_a, label_b)
    results["Batting HR (team total)"] = (res, False)

    for stat in ["ERA", "WHIP"]:
        res = decompose_rate_stat(stats_a.get("pitching"), stats_b.get("pitching"),
                                  "IP", stat, label_a, label_b)
        results[f"Pitching {stat}"] = (res, True)

    # fielding (additive, not weighted -- same convention as team_diagnosis.py)
    fld_a, fld_b = stats_a.get("fielding"), stats_b.get("fielding")
    stat_col = "Rtot" if (fld_b is not None and "Rtot" in getattr(fld_b, "columns", [])) else "Rdrs"
    res = decompose_sum_stat(fld_a, fld_b, stat_col, label_a, label_b)
    results[f"Fielding {stat_col} (team total)"] = (res, False)

    return results


def print_all_decompositions(stats_a: dict, stats_b: dict,
                             label_a: str, label_b: str):
    results = compute_all_decompositions(stats_a, stats_b, label_a, label_b)
    for title, (res, is_rate) in results.items():
        _print_block(title, res, label_a, label_b, is_rate=is_rate)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompose a Mariners team stat's year-over-year change into per-player deltas")
    parser.add_argument("--from-season", type=int, default=2025)
    parser.add_argument("--to-season", type=int, default=2026)
    args = parser.parse_args()

    label_a, label_b = str(args.from_season), str(args.to_season)

    print(f"Loading {label_a} and {label_b} Mariners stats (uses cache if fresh)...")
    stats_a = get_seattle_stats(args.from_season)
    stats_b = get_seattle_stats(args.to_season)

    print("\n" + "=" * 70)
    print(f"YEAR-OVER-YEAR DECOMPOSITION: {label_a} -> {label_b}")
    print("=" * 70)
    print("Contribution_Pts/Contribution is sign-flipped for ERA/WHIP so that")
    print("POSITIVE always means 'helped the team improve', NEGATIVE always")
    print("means 'made it worse' -- same convention as team_diagnosis.py.")

    print_all_decompositions(stats_a, stats_b, label_a, label_b)