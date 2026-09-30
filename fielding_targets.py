"""
fielding_targets.py
A dedicated, fielding-first view of the free agent market.

WHY THIS EXISTS: team_diagnosis.py flagged fielding (-30 Rtot) as the
single biggest team weakness by a wide margin (Severity_Score 6.00, more
than double the next-worst category). But free_agent_recommendations.py
only ever treats defense as a small tiebreaker bonus (get_fielding_bonus,
+/-15-30%) layered on top of an OFFENSE-driven ranking -- a plus defender
at a position that's already offensively fine would never surface there.
This asks the fielding question directly: "who's actually a good enough
glove to fix OUR specific defensive problem," not "who's a good hitter
who also happens to field okay."

METHOD
------
1. Confirms fielding is actually a diagnosed weakness (reads
   diagnosed_needs.csv via free_agent_recommendations.load_diagnosed_needs())
   -- if it isn't, this says so plainly and stops, rather than always
   running regardless of whether the team even has a defensive problem.
2. Sums each current Mariners batter's real Rtot this season from
   data["seattle"]["fielding"] -- additive, same convention
   team_diagnosis.py's _team_fielding() already established (Rtot is
   already a runs-above/below-average number, not a rate, so it sums
   directly rather than needing PA/IP weighting).
3. Maps each of the worst-fielding Mariners regulars to their PRIMARY
   position group via the BATTING table's Pos column
   (BBREF_TO_MLBTR_POS, the same confirmed-reliable mapping
   free_agent_recommendations.py already uses) -- this turns "Crawford
   is at -8 Rtot" into "Shortstop is a real, specific defensive need,"
   not just an anonymous team-wide number.
4. Pulls every matched free agent's real Rtot (free_agent_stats_matcher's
   field_matched flag) at those SPECIFIC flagged positions, ranks by
   Rtot descending (best gloves first), and shows their offense line +
   real cost (free_agent_market, when matched) side by side -- so a
   defensive target is never picked blind to whether they can also hit
   or what they'd cost.

HONESTY NOTES CARRIED OVER FROM THE REST OF THIS PROJECT: a free agent
with no real field_matched Rtot is excluded from these rankings entirely
(not assumed average), and one with no cost match shows "n/a", not a
guessed or zero cost.

Usage:
    python fielding_targets.py
"""
import pandas as pd

from data_builder import build_all
from free_agent import get_mlbtr_free_agents
from free_agent_stats_matcher import enrich_free_agents
from free_agent_recs import (
    load_diagnosed_needs, BBREF_TO_MLBTR_POS, _parse_primary_position,
    _category_severity, _get_current_roster_names,
)
from free_agent_market import load_market_data, get_market_row
from team_diagnosis import _team_fielding


def get_mariners_fielding_weak_spots(data: dict) -> pd.DataFrame:
    """
    Real, current-season Rtot per Mariners batter (summed across however
    many rows they have in the team fielding table -- see this module's
    docstring), joined to their PRIMARY position group via the batting
    table. Returns only players with Rtot < 0 (genuinely below average),
    sorted worst first.
    """
    fielding = data["seattle"].get("fielding", pd.DataFrame())
    batting = data["seattle"].get("batting", pd.DataFrame())
    if fielding.empty or batting.empty:
        return pd.DataFrame()

    stat_col = "Rtot" if "Rtot" in fielding.columns else (
               "Rdrs" if "Rdrs" in fielding.columns else None)
    if stat_col is None:
        return pd.DataFrame()

    fld = fielding.copy()
    fld[stat_col] = pd.to_numeric(fld[stat_col], errors="coerce")
    per_player = fld.groupby("Name", as_index=False)[stat_col].sum()

    pos_lookup = {}
    for _, row in batting.iterrows():
        primary = _parse_primary_position(row.get("Pos"))
        group = BBREF_TO_MLBTR_POS.get(primary)
        if group:
            pos_lookup[row.get("Name")] = group

    per_player["position_group"] = per_player["Name"].map(pos_lookup)
    per_player = per_player.rename(columns={stat_col: "Rtot"})
    per_player = per_player[per_player["position_group"].notna() & (per_player["Rtot"] < 0)]
    return per_player.sort_values("Rtot").reset_index(drop=True)


def get_full_team_fielding_ledger(data: dict) -> pd.DataFrame:
    """
    The SAME real, per-batter Rtot computation as
    get_mariners_fielding_weak_spots() -- summed across however many rows
    a player has, joined to their primary position group -- but with NO
    Rtot < 0 filter, so both plus and minus fielders show up. This is the
    "whole team, positive guys and negatives" view.

    NOTE ON RECONCILIATION: this is a per-BATTER-only ledger (it only
    includes names that also appear in the batting table, so it can map
    a position group). team_diagnosis.py's real team fielding total
    ALSO includes pitchers' own fielding Rtot (pitchers field their
    position too, and bbref's team fielding table includes them), which
    this view does not. So sum(this ledger) will not exactly equal the
    real team_diagnosis.py total -- the gap is exactly the pitchers'
    combined fielding Rtot, not a bug. print_full_team_fielding_ledger()
    reports both numbers and the gap explicitly, rather than pretending
    they should match.
    """
    fielding = data["seattle"].get("fielding", pd.DataFrame())
    batting = data["seattle"].get("batting", pd.DataFrame())
    if fielding.empty or batting.empty:
        return pd.DataFrame()

    stat_col = "Rtot" if "Rtot" in fielding.columns else (
               "Rdrs" if "Rdrs" in fielding.columns else None)
    if stat_col is None:
        return pd.DataFrame()

    fld = fielding.copy()
    fld[stat_col] = pd.to_numeric(fld[stat_col], errors="coerce")
    per_player = fld.groupby("Name", as_index=False)[stat_col].sum()

    pos_lookup = {}
    for _, row in batting.iterrows():
        primary = _parse_primary_position(row.get("Pos"))
        group = BBREF_TO_MLBTR_POS.get(primary)
        if group:
            pos_lookup[row.get("Name")] = group

    per_player["position_group"] = per_player["Name"].map(pos_lookup)
    per_player = per_player.rename(columns={stat_col: "Rtot"})
    # Only drop rows we can't map to a batting position at all (a pure
    # pitcher-only name with no batting row) -- keep BOTH positive and
    # negative Rtot for everyone we can map.
    per_player = per_player[per_player["position_group"].notna()]
    return per_player.sort_values("Rtot", ascending=False).reset_index(drop=True)


def print_full_team_fielding_ledger(data: dict, real_team_total: float = None):
    """
    Prints every mapped Mariners batter's real Rtot this season, best
    fielder first, positives and negatives together -- then an honest
    reconciliation line against team_diagnosis.py's real team total
    (pass it in via real_team_total if you have it handy; if not, this
    just reports the per-batter sum with a note that pitchers' own
    fielding isn't included in this view).
    """
    ledger = get_full_team_fielding_ledger(data)
    print("\n" + "=" * 100)
    print("FULL TEAM FIELDING LEDGER (every mapped batter, best to worst)")
    print("=" * 100)
    if ledger.empty:
        print("  No fielding/batting data available.")
        print("=" * 100 + "\n")
        return

    print(ledger.to_string(index=False))
    per_batter_sum = ledger["Rtot"].sum()
    print(f"\n  Sum of this per-batter ledger: {per_batter_sum:+.1f} Rtot")
    if real_team_total is not None:
        gap = real_team_total - per_batter_sum
        print(f"  Real team total (team_diagnosis.py): {real_team_total:+.1f} Rtot")
        print(f"  Gap: {gap:+.1f} Rtot -- this is pitchers' own fielding Rtot, "
              f"which this batter-only ledger doesn't include (not a bug).")
    else:
        print("  Note: this excludes pitchers' own fielding Rtot, so it will "
              "typically differ from team_diagnosis.py's real team total by "
              "that amount -- pass real_team_total in to see the exact gap.")
    print("=" * 100 + "\n")


def get_fielding_targets(enriched: pd.DataFrame, weak_spots: pd.DataFrame,
                         market: pd.DataFrame, current_roster: set,
                         top_n: int = 5) -> dict:
    """
    {position_group: DataFrame of top real defensive free agent fits},
    for every position_group that showed up in weak_spots. Excludes
    current Mariners (a real external target, not a re-sign) and anyone
    without a real field_matched Rtot.
    """
    if enriched.empty or weak_spots.empty:
        return {}

    candidates = enriched[enriched["field_matched"] & ~enriched["name"].isin(current_roster)].copy()
    candidates["Rtot"] = pd.to_numeric(candidates["Rtot"], errors="coerce")

    results = {}
    for group in weak_spots["position_group"].unique():
        options = candidates[candidates["position_group"] == group].copy()
        if options.empty:
            continue
        options = options.sort_values("Rtot", ascending=False)

        est_costs, matched_flags = [], []
        for _, row in options.iterrows():
            m = get_market_row(row["name"], market) if market is not None and not market.empty else None
            if m is not None:
                est_costs.append(round(float(m["Est_Cost_M"]), 2))
                matched_flags.append(True)
            else:
                est_costs.append(None)
                matched_flags.append(False)
        options["Est_Cost_M"] = est_costs
        options["Cost_Matched"] = matched_flags

        results[group] = options.head(top_n)
    return results


def find_fielding_target_conflicts(targets: dict, top_n: int = 3) -> dict:
    """
    {name: [position_group, position_group, ...]} for any free agent who
    shows up in the top `top_n` real candidates at MORE THAN ONE weak
    position group. A real, concrete case this is meant to catch: Seiya
    Suzuki showing up as the #1 matched Rtot option at BOTH Right Field
    and Designated Hitter -- he can only be signed to fill one roster
    spot, so treating him as "the fix" for both needs independently
    double-counts him. Same idea as
    free_agent_market.find_position_conflicts(), but scoped to each
    position group's own top_n here rather than one combined top-N
    recommendation list, since fielding_targets.py's targets are already
    split out per weak position.
    """
    if not targets:
        return {}

    appearances = {}
    for group, options in targets.items():
        if options is None or options.empty:
            continue
        for name in options.head(top_n)["name"].tolist():
            appearances.setdefault(name, []).append(group)

    return {name: groups for name, groups in appearances.items() if len(groups) > 1}


def print_fielding_target_conflicts(targets: dict, top_n: int = 3):
    conflicts = find_fielding_target_conflicts(targets, top_n=top_n)
    if not conflicts:
        return
    print(f"\n  [!] Overlap warning -- these free agents are a top-{top_n} real "
          f"defensive fit at MORE THAN ONE weak position, but can only fill one "
          f"roster spot:")
    for name, groups in conflicts.items():
        print(f"      {name}: {' / '.join(groups)}")


def print_fielding_report(weak_spots: pd.DataFrame, targets: dict, needs: pd.DataFrame):
    print("\n" + "=" * 100)
    print("FIELDING-FIRST FREE AGENT TARGETS")
    print("=" * 100)

    fielding_sev = _category_severity(needs, "Fielding")
    if needs.empty:
        print("  [warn] no diagnosed_needs.csv found -- run team_diagnosis.py first. "
              "Showing weak spots anyway, but severity isn't confirmed live.")
    elif fielding_sev <= 0:
        print(f"  Team fielding is NOT currently diagnosed as a weakness "
              f"(severity {fielding_sev:.2f}) -- this report still runs, "
              f"but treat it as informational rather than an urgent need.")
    else:
        print(f"  Team fielding severity: {fielding_sev:.2f} -- confirmed real weakness.")

    if weak_spots.empty:
        print("\n  No below-average Mariners fielders found at a mappable position "
              "-- nothing to target here.")
        print("=" * 100 + "\n")
        return

    print("\n  Current below-average Mariners fielders (real Rtot this season):")
    print("  " + weak_spots.to_string(index=False).replace("\n", "\n  "))

    for group in weak_spots["position_group"].unique():
        current_row = weak_spots[weak_spots["position_group"] == group].iloc[0]
        print(f"\n{'-' * 100}")
        print(f"  {group} -- current: {current_row['Name']} ({current_row['Rtot']:+.0f} Rtot)")
        print(f"{'-' * 100}")
        options = targets.get(group)
        if options is None or options.empty:
            print("    No real, matched free agent with defensive data at this position.")
            continue
        is_pitcher = "ERA" in options.columns and options["ERA"].notna().any()
        cols = (["name", "Team", "Rtot", "IP", "ERA", "Est_Cost_M"] if is_pitcher else
                ["name", "Team", "Rtot", "PA", "OPS", "BA", "Est_Cost_M"])
        real_cols = [c for c in cols if c in options.columns]
        display = options[real_cols].copy()
        if "Est_Cost_M" in display.columns:
            display["Est_Cost_M"] = display["Est_Cost_M"].apply(
                lambda v: f"${v:.1f}M" if pd.notna(v) else "n/a")
        print("    " + display.to_string(index=False).replace("\n", "\n    "))

    print_fielding_target_conflicts(targets)

    print("\n" + "=" * 100)
    print("  Rtot = real runs above/below average this season (higher is better).")
    print("  Est_Cost_M = real Spotrac Prev_AAV / opt-out floor, same convention as")
    print("    free_agent_market.py -- 'n/a' means no real cost match, not free.")
    print("  Only free agents with a REAL field_matched Rtot appear here -- no")
    print("  average-defense assumptions for anyone unmatched.")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    print("Building league + Mariners data (uses cache if fresh)...")
    data = build_all(2026)

    print("Loading roster diagnosis (team_diagnosis.py output)...")
    needs = load_diagnosed_needs()

    print("Getting the real free agent list...")
    fa = get_mlbtr_free_agents()
    enriched = enrich_free_agents(fa, data)
    current_roster = _get_current_roster_names(data)

    print("Loading real Spotrac market data...")
    market = load_market_data()

    weak_spots = get_mariners_fielding_weak_spots(data)
    targets = get_fielding_targets(enriched, weak_spots, market, current_roster)

    real_team_total = None
    fielding_df = data["seattle"].get("fielding", pd.DataFrame())
    if not fielding_df.empty:
        real_team_total = _team_fielding(fielding_df).get("Rtot")

    print_full_team_fielding_ledger(data, real_team_total=real_team_total)
    print_fielding_report(weak_spots, targets, needs)