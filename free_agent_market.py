"""
free_agent_market.py
Real Spotrac market data (Prev_AAV / contract mechanism) for this
winter's free agent class -- pasted 2026-09-29 from Spotrac's 1B/2B/3B/SS
(combined infield filter), OF, and RP pages, parsed and deduplicated by
build_market_csv.py into free_agent_market.csv.

WHY Prev_AAV, NOT A "MARKET VALUE" PROJECTION
----------------------------------------------
Spotrac's page has a "Market Value AAV" column for a NEXT-contract
projection, but it came back blank for every row in what was pasted (no
market-value projections had been populated yet for this class at scrape
time). Prev_AAV -- what each player actually made THIS PAST season -- is
the real number available, and is used here as an honest, clearly-labeled
proxy/anchor for asking price rather than inventing a projection. It's
not a prediction of their next deal (which could go up on a strong year,
or down for an aging/declining player) -- it's the best real number this
project actually has, and is labeled Est_Cost_M (not "Market_Value") for
exactly that reason.

ONE REAL REFINEMENT: OPT-OUT players (Bo Bichette, Yuki Matsui) walked
away from a KNOWN guaranteed option value to test the market -- that
forgone option is real evidence of a self-imposed price floor (nobody
opts out to sign for less), so Est_Cost_M uses max(Prev_AAV, opt-out
value) for those specific players. Every other contract mechanism
(UFA/MUTUAL/CLUB/CONDITIONAL-CLUB/PLAYER option) just uses Prev_AAV
directly -- those values describe what would have happened if a team/
player HAD exercised the option, not a signal about the open-market price
a new suitor should expect to pay.

NAME MATCHING
-------------
Same approach as year_over_year_decomposition.py's _normalize_name --
strips accents/case before an exact match. Spotrac and MLBTR (this
project's free agent source, via free_agent.py) both use standard
ASCII-safe player names in practice, so no fuzzy/partial matching is
attempted here -- an unmatched name shows up honestly as "no market data"
rather than a silent guess.

Usage:
    from free_agent_market import load_market_data, attach_market_data
    market = load_market_data()
    scored = attach_market_data(scored, market)   # adds Est_Cost_M, Contract_Type, Value_Score
"""
import os
import re
import unicodedata

import pandas as pd

MARKET_CSV = os.path.join(os.path.dirname(__file__), "free_agent_market.csv")


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return stripped.strip().lower()


def _opt_out_value(contract_type: str):
    """Parses 'OPT-OUT / $18.7M' -> 18.7. Returns None for any other type."""
    if not isinstance(contract_type, str) or "OPT-OUT" not in contract_type:
        return None
    m = re.search(r"\$([\d.]+)M", contract_type)
    return float(m.group(1)) if m else None


def load_market_data(path: str = MARKET_CSV) -> pd.DataFrame:
    """
    Returns a DataFrame keyed for lookup, with:
      Name, Pos, Age, Prev_Team, Prev_AAV, Contract_Type, Arm,
      Est_Cost_M (millions -- see module docstring for the opt-out floor)
    Empty DataFrame (with a warning) if the file hasn't been built yet.
    """
    if not os.path.exists(path):
        print(f"  [warn] {path} not found -- run build_market_csv.py first, "
              f"or paste fresh Spotrac data. Falling back to no cost data "
              f"(recommendations will show 'no market data' for everyone).")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["Prev_AAV_M"] = df["Prev_AAV"] / 1_000_000
    df["_opt_out_floor_M"] = df["Contract_Type"].apply(_opt_out_value)
    df["Est_Cost_M"] = df[["Prev_AAV_M", "_opt_out_floor_M"]].max(axis=1, skipna=True)
    df["_key"] = df["Name"].apply(_normalize_name)
    return df


def get_market_row(name: str, market: pd.DataFrame):
    """Single-player lookup. Returns a pd.Series row or None."""
    if market is None or market.empty:
        return None
    key = _normalize_name(name)
    match = market[market["_key"] == key]
    return match.iloc[0] if not match.empty else None


def attach_market_data(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Est_Cost_M, Contract_Type, Market_Matched, and Value_Score
    (recommendation_score / Est_Cost_M) to a scored recommendations
    DataFrame (the output of compute_recommendation_score()).

    Internal roster candidates (source == "Internal (roster)") get
    Est_Cost_M = 0.0 and Market_Matched = "internal" -- promoting a
    player already on the 40-man has no NEW signing cost, so they
    shouldn't be penalized by (or confused with) "no market data found"
    the way a genuinely-unmatched free agent is.

    A free agent with no real Spotrac match gets Est_Cost_M = NaN and
    Value_Score = NaN -- explicit, honest missing data, not a silent
    assumption of average or zero cost.
    """
    out = df.copy()
    out["Est_Cost_M"] = None
    out["Contract_Type"] = None
    # object dtype (not bool) since this holds True/False/"internal"
    out["Market_Matched"] = pd.Series([False] * len(out), dtype=object, index=out.index)

    for idx, row in out.iterrows():
        if row.get("source") == "Internal (roster)":
            out.at[idx, "Est_Cost_M"] = 0.0
            out.at[idx, "Contract_Type"] = "already on roster"
            out.at[idx, "Market_Matched"] = "internal"
            continue
        m = get_market_row(row.get("name"), market)
        if m is not None:
            out.at[idx, "Est_Cost_M"] = round(float(m["Est_Cost_M"]), 2)
            out.at[idx, "Contract_Type"] = m["Contract_Type"]
            out.at[idx, "Market_Matched"] = True

    out["Est_Cost_M"] = pd.to_numeric(out["Est_Cost_M"], errors="coerce")

    def _value_score(row):
        score = row.get("recommendation_score")
        cost = row.get("Est_Cost_M")
        if pd.isna(score):
            return None
        if row.get("Market_Matched") == "internal":
            return None  # no meaningful "value per dollar" for a free roster move
        if pd.isna(cost) or cost <= 0:
            return None
        return round(score / cost, 3)

    out["Value_Score"] = out.apply(_value_score, axis=1)
    return out


# ── position-conflict flagging ──────────────────────────────────────────────
# Roster spots you can only fill ONCE -- unlike bullpen/rotation slots,
# which realistically take several new arms in one offseason, a batting
# lineup spot is a single seat. If two+ recommendations land at the same
# one of these, that's not "sign them both", it's "pick one" -- worth
# flagging explicitly rather than letting the ranked list imply otherwise.
SINGLE_SLOT_POSITIONS = {
    "Catchers", "First Basemen", "Second Basemen", "Third Basemen",
    "Shortstops", "Left Fielders", "Center Fielders", "Right Fielders",
    "Designated Hitters",
}


def find_position_conflicts(scored: pd.DataFrame, top_n: int = 15) -> dict:
    """
    Looks at the top N ranked recommendations and flags any single-slot
    position group appearing more than once -- e.g. Ty France AND Jake
    Bauers both showing up as First Basemen recommendations. Returns
    {position_group: [row, row, ...]} for every conflicted group.
    """
    top = (scored[scored["recommendation_score"].notna()]
           .sort_values("recommendation_score", ascending=False)
           .head(top_n))
    conflicts = {}
    for group in SINGLE_SLOT_POSITIONS:
        rows = top[top["position_group"] == group]
        if len(rows) > 1:
            conflicts[group] = rows.to_dict("records")
    return conflicts


def print_position_conflicts(scored: pd.DataFrame, top_n: int = 15):
    conflicts = find_position_conflicts(scored, top_n)
    if not conflicts:
        return
    print("\n" + "=" * 90)
    print(f"POSITION CONFLICTS -- more than one top-{top_n} recommendation for a single roster spot")
    print("=" * 90)
    for group, rows in conflicts.items():
        names = ", ".join(f"{r['name']} (score {r['recommendation_score']:.2f}, "
                          f"${r.get('Est_Cost_M') or 0:.1f}M)" for r in rows)
        print(f"  {group}: {names}")
    print("  You can only roster one of these per spot -- treat the rest as depth/")
    print("  alternate-scenario options, not simultaneous real targets.")
    print("=" * 90 + "\n")