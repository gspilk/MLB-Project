"""
age_risk.py
Adds an "age risk" to each free agent from the aging curves built in R
(r/aging_curves.R -> data/exports/aging_by_age_band.csv).

For a hitter, find the age band he is in during the 2027 season and look up how
much hitters like him (same group: 1B/3B/SS vs everyone else) typically change
the next season, in WAR per 600 plate appearances. More negative = more decline
expected.

It is an average, not a forecast for one player. Bands with fewer than MIN_PAIRS
player-season pairs are reported as "Too few to say" (never a guessed number).
Pitchers are not covered (the aging table is hitters only) and stay blank.

    from age_risk import add_age_risk
    df = add_age_risk(df, "data/exports/aging_by_age_band.csv")
"""
import os

import numpy as np
import pandas as pd

INFIELD_GROUPS = {"First Basemen", "Third Basemen", "Shortstops"}
BAND_CUTS = [-np.inf, 26, 29, 32, 35, np.inf]              # same cuts as aging_curves.R
BAND_NAMES = ["26 and under", "27-29", "30-32", "33-35", "36+"]
IF_LABEL, OTHER_LABEL = "1B / 3B / SS", "All other hitters"
MIN_PAIRS = 20                                              # same bar as aging_curves.R
LOW_ABOVE, HIGH_BELOW = -0.25, -0.60                        # WAR/600 per season


def _is_hitter_group(group) -> bool:
    g = str(group)
    return bool(g) and g != "nan" and "Pitcher" not in g and "Reliever" not in g


def _risk_label(change: float) -> str:
    if change > LOW_ABOVE:
        return "Low"
    if change < HIGH_BELOW:
        return "High"
    return "Moderate"


def add_age_risk(df: pd.DataFrame, csv_path: str, age_col: str = "age_2027",
                 group_col: str = "position_group") -> pd.DataFrame:
    """Returns a copy of df with age_band, age_war600_per_yr, age_risk columns.
    If the CSV or the age column is missing, df comes back unchanged."""
    if age_col not in df.columns or not os.path.exists(csv_path):
        return df
    bands = pd.read_csv(csv_path)
    lookup = {(r["group"], r["band"]): (r["d_war"], r["n"]) for _, r in bands.iterrows()}

    out = df.copy()
    age = pd.to_numeric(out[age_col], errors="coerce")
    band = pd.cut(age, bins=BAND_CUTS, labels=BAND_NAMES).astype(object)
    out["age_band"] = band
    change, risk = [], []
    for g, b in zip(out[group_col] if group_col in out.columns else [None] * len(out), band):
        if not _is_hitter_group(g) or pd.isna(b):
            change.append(np.nan); risk.append("")
            continue
        grp = IF_LABEL if g in INFIELD_GROUPS else OTHER_LABEL
        d_war, n = lookup.get((grp, b), (np.nan, 0))
        if pd.isna(d_war) or n < MIN_PAIRS:
            change.append(np.nan); risk.append("Too few to say")
        else:
            change.append(round(float(d_war), 2)); risk.append(_risk_label(d_war))
    out["age_war600_per_yr"] = change
    out["age_risk"] = risk
    return out