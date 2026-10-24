"""
story_app.py
An interactive "here's the problem -> here's what to do" walkthrough of the
Mariners offseason analysis. It runs NO analysis itself: it only reads the CSVs
that offseason_report.py already wrote to data/exports/, so after you rerun the
report, refresh the page and the story updates.

Run (from the MLB PROJECT folder):
    pip install streamlit
    streamlit run story_app.py

Screens (sidebar):
  1. The problem      -- team grade and weaknesses ranked by severity
  2. Why it happened  -- per-player year-over-year changes
  3. What to do       -- ranked free agents with filters + cost vs score
  4. The plan         -- pick a signing slate, see total cost and conflicts
  5. Who fixes it     -- for each weak stat, the players strong at it (from
                         weakness_targets.py; any MLB player, not just free agents)

Optional extras, used automatically when the files exist:
  free_agent_handedness.csv      (from mlb_statsapi.py)  -> bats filter
  free_agent_projection_2027.csv (from projection_2027.py) -> projected OPS
"""
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

from age_risk import add_age_risk

DEFAULT_DIR = os.environ.get(
    "MLB_EXPORT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "exports"))

st.set_page_config(page_title="Mariners Offseason: Problem to Plan", layout="wide")


# ───────────────────────── data loading ─────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(path: str, mtime: float) -> pd.DataFrame:
    """mtime is part of the cache key so a re-run of the report is picked up."""
    return pd.read_csv(path)


def read_export(folder: str, filename: str):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return None
    return load_csv(path, os.path.getmtime(path))


def load_scored(folder: str):
    """free_agent_scored.csv, plus handedness / projection columns when present."""
    df = read_export(folder, "free_agent_scored.csv")
    if df is None:
        return None
    df = df.copy()
    if "recommendation_score" in df.columns:
        df = df[df["recommendation_score"].notna()]
    hand = read_export(folder, "free_agent_handedness.csv")
    if hand is not None and "bats" in hand.columns:
        df = df.merge(hand[["name", "bats", "throws"]].drop_duplicates("name"), on="name", how="left")
    proj = read_export(folder, "free_agent_projection_2027.csv")
    if proj is not None:
        pcols = [c for c in proj.columns if c.startswith("proj_OPS_")]
        if pcols:
            df = df.merge(proj[["name"] + pcols].drop_duplicates("name"), on="name", how="left")
    # age risk, from the R aging curves (data/exports/aging_by_age_band.csv); a no-op if that file is missing
    df = add_age_risk(df, os.path.join(folder, "aging_by_age_band.csv"))
    return df


def external_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drops the Mariners' own roster players (source 'Internal (roster)'), leaving real outside targets."""
    if "source" not in df.columns:
        return df
    return df[~df["source"].astype(str).str.contains("Internal", na=False)]


def priced(df: pd.DataFrame) -> pd.Series:
    """True only for rows with a REAL cost match -- unmatched is 'unknown', never free."""
    if "Est_Cost_M" not in df.columns:
        return pd.Series(False, index=df.index)
    ok = df["Est_Cost_M"].notna()
    if "Market_Matched" in df.columns:
        ok &= df["Market_Matched"].astype(str).str.lower().eq("true")
    return ok


STANDARD_GROUPS = [
    "Catchers", "First Basemen", "Second Basemen", "Shortstops", "Third Basemen", "Left Fielders",
    "Center Fielders", "Right Fielders", "Designated Hitters", "Starting Pitchers",
    "Right-Handed Relievers", "Left-Handed Relievers"]


def is_pitch_group(group: str) -> bool:
    return "Pitcher" in group or "Reliever" in group


def build_depth(all_rows: pd.DataFrame, added: pd.DataFrame, starters: pd.DataFrame = None) -> dict:
    """{position_group: (is_pitching, DataFrame[Player, Stat, Role])}.
    Roles: the current starter (or staff average for pitching groups), other
    Mariners at that position ('Roster depth'), and the players added in the
    what-if ('Added'). Stat is 2026 OPS for hitters and ERA for pitchers; a
    missing stat stays blank, never a guess.
    The Mariners side comes from mariners_starters.csv when it exists (built from
    the Mariners' own stats, so every position is covered); otherwise it falls
    back to the incumbent/Internal rows inside free_agent_scored.csv."""
    out = {}
    has_st = starters is not None and not starters.empty and "position_group" in starters.columns
    groups = set(STANDARD_GROUPS)
    if "position_group" in all_rows.columns:
        groups |= set(all_rows["position_group"].dropna())
    if has_st:
        groups |= set(starters["position_group"].dropna())
    for g in [x for x in STANDARD_GROUPS if x in groups] + sorted(groups - set(STANDARD_GROUPS)):
        sub = all_rows[all_rows["position_group"] == g] if "position_group" in all_rows.columns else all_rows.iloc[0:0]
        st = starters[starters["position_group"] == g] if has_st else None
        pitch = is_pitch_group(g)
        col = "ERA" if pitch else "OPS"
        entries, seen = [], set()
        base = st[st["role"].isin(["Current starter", "Staff average"])] if st is not None else None
        if base is not None and not base.empty:
            r = base.iloc[0]
            entries.append({"Player": r["player"], "Stat": pd.to_numeric(r.get("stat"), errors="coerce"),
                            "Role": r["role"]})
            seen.add(r["player"])
        elif "incumbent_name" in sub.columns:
            inc = sub.dropna(subset=["incumbent_name"])
            if not inc.empty:
                r = inc.iloc[0]
                entries.append({"Player": r["incumbent_name"],
                                "Stat": pd.to_numeric(r.get("incumbent_stat"), errors="coerce"),
                                "Role": "Staff average" if pitch else "Current starter"})
                seen.add(r["incumbent_name"])
        if st is not None and (st["role"] == "Roster depth").any():
            for _, r in st[st["role"] == "Roster depth"].iterrows():
                if r["player"] not in seen:
                    entries.append({"Player": r["player"], "Stat": pd.to_numeric(r.get("stat"), errors="coerce"),
                                    "Role": "Roster depth"})
                    seen.add(r["player"])
        elif "source" in sub.columns and col in sub.columns:
            for _, r in sub[sub["source"].astype(str).str.contains("Internal", na=False)].iterrows():
                if r["name"] not in seen:
                    entries.append({"Player": r["name"], "Stat": pd.to_numeric(r.get(col), errors="coerce"),
                                    "Role": "Roster depth"})
                    seen.add(r["name"])
        for _, r in added[added["position_group"] == g].iterrows():
            entries.append({"Player": r["name"], "Stat": pd.to_numeric(r.get(col), errors="coerce"),
                            "Role": "Added"})
        if entries:
            out[g] = (pitch, pd.DataFrame(entries))
    return out


def merge_position_starters(starters, pn):
    """Replaces the hitters' 'Current starter' rows with the Mariners-only starters from
    position_needs.csv (most plate appearances at the position while a Mariner). Pitching rows and
    anything missing stay as they were."""
    if pn is None or pn.empty or not {"position_group", "starter", "starter_ops_with_sea"} <= set(pn.columns):
        return starters
    new = pn[pn["starter_ops_with_sea"].notna() & (pn["starter"] != "no clear starter")]
    new = pd.DataFrame({"position_group": new["position_group"], "player": new["starter"],
                        "stat": new["starter_ops_with_sea"], "role": "Current starter"})
    if starters is None or starters.empty:
        return new
    keep = starters[~((starters["role"] == "Current starter") & starters["position_group"].isin(new["position_group"]))]
    return pd.concat([keep, new], ignore_index=True)


def starter_now_and_after(pitch: bool, depth: pd.DataFrame):
    """Starter now = the incumbent row. With signings = the best stat among the
    incumbent and the added players (roster depth is shown but not promoted)."""
    base = depth[depth["Role"].isin(["Current starter", "Staff average"])]
    now = base.iloc[0] if not base.empty else None
    pool = pd.concat([base, depth[depth["Role"] == "Added"]])
    pool = pool.dropna(subset=["Stat"])
    after = None
    if not pool.empty:
        after = pool.loc[pool["Stat"].idxmin() if pitch else pool["Stat"].idxmax()]
    return now, after


# ───────────────────────── sidebar ─────────────────────────
st.sidebar.title("Mariners offseason")
folder = DEFAULT_DIR
screen = st.sidebar.radio(
    "Story",
    ["1. The problem", "2. Why it happened", "3. What to do", "4. The plan", "5. Who fixes it"])
st.sidebar.caption("Reads the CSVs from offseason_report.py. Rerun the report, then refresh this page.")


def _pool_status(folder_):
    """(outside free agents, last-updated text) from free_agent_scored.csv, or (None, None)."""
    path = os.path.join(folder_, "free_agent_scored.csv")
    if not os.path.exists(path):
        return None, None
    d = load_csv(path, os.path.getmtime(path))
    d = d[d["recommendation_score"].notna()] if "recommendation_score" in d.columns else d
    n = len(external_only(d))
    stamp = pd.Timestamp(os.path.getmtime(path), unit="s").strftime("%b %d, %Y")
    return n, stamp

if not os.path.isdir(folder):
    st.error(f"Folder not found: {folder}. Run offseason_report.py first.")
    st.stop()

_n_pool, _stamp = _pool_status(folder)
if _n_pool is not None:
    st.sidebar.metric("Outside free agents scored", _n_pool)
    st.sidebar.caption(f"Report last run {_stamp}. The pool is small until free agency opens after the World Series.")


# ───────────────────────── 1. the problem ─────────────────────────
if screen == "1. The problem":
    st.title("1. The problem")
    needs = read_export(folder, "diagnosed_needs.csv")
    if needs is None:
        st.warning("diagnosed_needs.csv not found. Run offseason_report.py first.")
        st.stop()

    needs = needs.sort_values("Severity_Score", ascending=False)
    weak = needs[needs["Verdict"] == "WEAKNESS"]
    st.write(f"The diagnosis flags **{len(weak)}** weaknesses out of {len(needs)} measured areas. "
             "Severity combines how far the team is from league average with how much that area matters.")

    top = weak.head(3)
    cols = st.columns(max(len(top), 1))
    for col, (_, r) in zip(cols, top.iterrows()):
        col.metric(f"{r['Category']}: {r['Metric']}", f"{r['Team_Value']:g}",
                   f"{r['Gap']:+g} vs league {r['League_Avg']:g}", delta_color="off")
        col.caption(f"Severity {r['Severity_Score']:.2f}")

    st.caption("Severity = the gap to league average divided by a typical team-to-team spread for that stat. "
               "Around 1 means one typical spread from average; 2 or more is a clear outlier.")

    st.subheader("Severity by area")
    cats = sorted(needs["Category"].unique())
    pick = st.multiselect("Categories", cats, default=cats)
    view = needs[needs["Category"].isin(pick)]
    st.bar_chart(view.sort_values("Severity_Score", ascending=False).set_index("Metric")["Severity_Score"])
    with st.expander("Full table"):
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download this table (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name="mariners_diagnosed_needs.csv", mime="text/csv")
    pn = read_export(folder, "position_needs.csv")
    if pn is not None and not pn.empty:
        st.subheader("Where the wins are lost, by position")
        st.caption("Mariners WAR at each position compared with the league average at that position, using only "
                   "what players did while on the Mariners. Positive = wins below average. Run "
                   "position_needs.py (or offseason_report.py) to refresh.")
        import altair as alt
        chart_df = pn[["position", "wins_below_avg"]].copy()
        chart_df["side"] = np.where(chart_df["wins_below_avg"] > 0, "Below league average", "At or above average")
        chart = (alt.Chart(chart_df).mark_bar()
                 .encode(y=alt.Y("position:N", sort=alt.EncodingSortField("wins_below_avg", order="descending"), title=None),
                         x=alt.X("wins_below_avg:Q", title="Wins below league average at the position (negative = better than average)"),
                         color=alt.Color("side:N", legend=None,
                                         scale=alt.Scale(domain=["Below league average", "At or above average"],
                                                         range=["#c0392b", "#2e7d9a"])),
                         tooltip=["position", "wins_below_avg"])
                 .properties(height=300))
        st.altair_chart(chart, width="stretch")
        pcols = [c for c in ["position", "need", "sea_war", "league_avg_war", "sea_rank", "wins_below_avg",
                             "starter", "starter_pa_with_sea", "starter_ops_with_sea", "depth"] if c in pn.columns]
        st.dataframe(pn[pcols], width="stretch", hide_index=True,
                     column_config={"sea_war": st.column_config.NumberColumn("Mariners WAR", format="%.1f"),
                                    "league_avg_war": st.column_config.NumberColumn("League avg WAR", format="%.1f"),
                                    "sea_rank": st.column_config.NumberColumn("Rank of 30", format="%d"),
                                    "wins_below_avg": st.column_config.NumberColumn("Wins below avg", format="%.1f"),
                                    "starter_pa_with_sea": st.column_config.NumberColumn("Starter PA", format="%d"),
                                    "starter_ops_with_sea": st.column_config.NumberColumn("Starter OPS", format="%.3f")})
    st.info("Next: screen 2 shows who drove the change, screen 3 shows who could fix it.")


# ───────────────────────── 2. why it happened ─────────────────────────
elif screen == "2. Why it happened":
    st.title("2. Why it happened")
    files = sorted(glob.glob(os.path.join(folder, "year_over_year_*.csv")))
    if not files:
        st.warning("No year_over_year_*.csv files found. Run offseason_report.py first.")
        st.stop()
    labels = {os.path.basename(f)[len("year_over_year_"):-4].replace("_", " "): f for f in files}
    choice = st.selectbox("Stat", list(labels))
    df = load_csv(labels[choice], os.path.getmtime(labels[choice]))
    st.caption("How each player's change moved the team number from last season to this one.")

    name_col = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), None)
    if "Contribution" in df.columns and name_col:
        d = df.dropna(subset=["Contribution"]).sort_values("Contribution")
        n = st.slider("Players to show on each side", 3, 15, 8)
        extremes = pd.concat([d.head(n), d.tail(n)]).drop_duplicates()
        st.bar_chart(extremes.set_index(name_col)["Contribution"])
        st.write("Left of zero hurt the team number; right of zero helped (for rate stats like ERA, "
                 "check the direction in the table: lower can be better).")
    with st.expander("Full table", expanded="Contribution" not in df.columns):
        st.dataframe(df, width="stretch", hide_index=True)


# ───────────────────────── 3. what to do ─────────────────────────
elif screen == "3. What to do":
    st.title("3. What to do")
    fa = load_scored(folder)
    if fa is None:
        st.warning("free_agent_scored.csv not found. Run offseason_report.py first.")
        st.stop()

    st.write("Free agents ranked by how well they fit the diagnosed needs. Cost comes from the prices in free_agent_market.csv, which you enter yourself.")

    c1, c2, c3, c4 = st.columns(4)
    groups = sorted(fa["position_group"].dropna().unique()) if "position_group" in fa.columns else []
    pos = c1.multiselect("Position", groups, default=groups)
    only_priced = c2.toggle("Priced only", value=False)
    max_cost = None
    if "Est_Cost_M" in fa.columns and fa["Est_Cost_M"].notna().any():
        hi = float(np.ceil(fa["Est_Cost_M"].max()))
        if hi > 0:
            max_cost = c3.slider("Max cost ($M)", 0.0, hi, hi)
    bats = None
    if "bats" in fa.columns:
        bats = c4.multiselect("Bats", ["L", "R", "S"], default=["L", "R", "S"])

    show_internal = st.toggle("Also show current Mariners (internal options)", value=False,
                              help="Bench and depth players already on the roster, scored the same way, "
                                   "for the 'sign someone or play the guy we have' question.")
    v = fa.copy() if show_internal else external_only(fa)
    if not show_internal and len(v) < len(fa):
        st.caption(f"Hiding {len(fa) - len(v)} current Mariners. Showing outside free agents only.")
    if v.empty:
        st.info("No outside free agents in the scored file yet. Free agency opens after the World Series: "
                "then run python mlb_free_agents.py and rerun the report.")
    elif not show_internal and len(v) < 25:
        st.info(f"Only {len(v)} outside free agents are listed so far, so this is a short list, not the market. "
                "Screen 5 ranks any MLB player on the Mariners' weak stats and works now.")
    if pos:
        v = v[v["position_group"].isin(pos)]
    p = priced(v)
    if only_priced:
        v = v[p]
    if max_cost is not None:
        v = v[(v["Est_Cost_M"].isna()) | (v["Est_Cost_M"] <= max_cost)] if not only_priced else v[v["Est_Cost_M"] <= max_cost]
    if bats is not None and len(bats) < 3:
        v = v[v["bats"].isin(bats)]

    v = v.sort_values("recommendation_score", ascending=False)
    st.caption(f"{len(v)} players match.")

    show = [c for c in ["name", "position_group", "age_2027", "age_risk", "age_war600_per_yr", "Team", "bats",
                        "recommendation_score",
                        "Est_Cost_M", "OPS", "proj_OPS_2027", "Rtot", "option_detail"] if c in v.columns]
    st.dataframe(v[show].head(50), width="stretch", hide_index=True,
                 column_config={"recommendation_score": st.column_config.NumberColumn("Fit score", format="%.2f"),
                                "Est_Cost_M": st.column_config.NumberColumn("Est. cost ($M)", format="%.1f"),
                                "age_war600_per_yr": st.column_config.NumberColumn("WAR/600 per yr", format="%.2f")})
    if not v.empty:
        st.download_button("Download this list (CSV)", v[show].to_csv(index=False).encode("utf-8"),
                           file_name="free_agent_shortlist.csv", mime="text/csv")
    if "age_risk" in v.columns:
        st.caption("Age risk = how much hitters of that age and position group usually decline in a season "
                   "(WAR per 600 PA, from the R aging curves). It is an average, not a forecast for one player; "
                   "'Too few to say' means not enough players at that age. Hitters only.")

    pv = v[priced(v)]
    if not pv.empty:
        st.subheader("Cost vs. fit (priced players)")
        st.caption("Upper left = high fit, low cost.")
        st.scatter_chart(pv, x="Est_Cost_M", y="recommendation_score", color="position_group"
                         if "position_group" in pv.columns else None)

    ft = read_export(folder, "fielding_targets.csv")
    if ft is not None and not ft.empty:
        st.subheader("Fielding-first targets")
        st.caption("Matched fielding runs (Rtot) for the positions where the team is weakest.")
        fcols = [c for c in ["position_group", "name", "Team", "Rtot", "Est_Cost_M"] if c in ft.columns]
        st.dataframe(ft[fcols], width="stretch", hide_index=True)


# ───────────────────────── 5. who fixes it ─────────────────────────
elif screen == "5. Who fixes it":
    st.title("5. Who fixes it")
    tg = read_export(folder, "weakness_targets.csv")
    if tg is None:
        st.warning("weakness_targets.csv not found. Run  python weakness_targets.py  first.")
        st.stop()
    st.write("The players who are clearly better than the Mariners at the stats the team is weakest in. "
             "These are **candidates, not free agents**: any non-Mariners player appears, whatever his contract. "
             "Once free agency opens, screens 3 and 4 narrow this to players who can actually be signed.")
    needs = read_export(folder, "diagnosed_needs.csv")
    summ = read_export(folder, "weakness_targets_summary.csv")

    metrics = list(dict.fromkeys(tg["metric"]))
    if not metrics:
        st.info("No players beat the Mariners on the weak stats with enough playing time.")
        st.stop()
    sev = {}
    if needs is not None:
        sev = needs.set_index("Metric")["Severity_Score"].to_dict()
        metrics.sort(key=lambda m: -sev.get(m, 0))

    # the weak stats, worst first, so the filters below make sense
    cols_m = st.columns(len(metrics))
    for c, mm in zip(cols_m, metrics):
        tv = float(tg.loc[tg["metric"] == mm, "mariners"].iloc[0])
        c.metric(mm, f"{tv:g}")
        if mm in sev:
            c.caption(f"Severity {sev[mm]:.2f}")

    # filters shared by both views
    OF_LABEL = "Outfield (LF/CF/RF)"
    pos_all = (sorted({str(p) for p in tg["position"].dropna() if str(p)}) if "position" in tg.columns else [])
    f1, f2, f3 = st.columns(3)
    pos_choice = f1.multiselect("Position", ([OF_LABEL] if any(p in ("LF", "CF", "RF") for p in pos_all) else []) + pos_all,
                                help="Applies to hitters and fielders. Pitching stats have no position.") if pos_all else []
    has_bats = "bats" in tg.columns and tg["bats"].fillna("").astype(str).str.len().gt(0).any()
    bats_choice = f2.multiselect("Bats", ["L", "R", "S"],
                                 help="From player_handedness.csv (run mlb_stats_api.py to build it).") if has_bats else []
    search = f3.text_input("Find a player", placeholder="part of a name")

    def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
        if pos_choice and "position" in d.columns and d["position"].fillna("").astype(str).str.len().gt(0).any():
            allowed = {p for p in pos_choice if p != OF_LABEL} | ({"LF", "CF", "RF"} if OF_LABEL in pos_choice else set())
            d = d[d["position"].fillna("").astype(str).isin(allowed)]
        if bats_choice and "bats" in d.columns and d["bats"].fillna("").astype(str).str.len().gt(0).any():
            d = d[d["bats"].fillna("").astype(str).isin(bats_choice)]
        if search.strip() and "name" in d.columns:
            d = d[d["name"].astype(str).str.contains(search.strip(), case=False, na=False, regex=False)]
        return d

    tab_fit, tab_stat = st.tabs(["Best overall fit", "By weak stat"])

    with tab_fit:
        if summ is None or summ.empty:
            st.info("weakness_targets_summary.csv is missing or empty. Run  python weakness_targets.py.")
        else:
            st.caption("Fit score = how far the player beats the Mariners on each weak stat (in league standard "
                       "deviations), weighted by how severe that weakness is. Higher is better. "
                       "The best candidates beat the Mariners on several weak stats at once.")
            pool_names = {"hit": "Hitters", "rel": "Relievers", "sp": "Starters"}
            pools = [p for p in ["hit", "rel", "sp"] if p in set(summ["pool"])]
            kind = st.radio("Show", [pool_names[p] for p in pools], horizontal=True)
            inv = {v: k for k, v in pool_names.items()}
            n_show = st.slider("How many to show", 5, 50, 15)
            s2 = apply_filters(summ[summ["pool"] == inv[kind]])
            if s2.empty:
                st.info("No players match these filters.")
            else:
                s2 = s2.head(n_show).rename(columns={
                    "weak_metrics_beaten": "Weak stats beaten", "severity_covered": "Severity covered",
                    "fit_score": "Fit score"})
                keep = [c for c in ["name", "Team", "position", "bats", "Fit score", "Weak stats beaten",
                                    "Severity covered", "metrics"] if c in s2.columns
                        and (c not in ("position", "bats") or s2[c].fillna("").astype(str).str.len().gt(0).any())]
                st.dataframe(s2[keep], width="stretch", hide_index=True,
                             column_config={"Fit score": st.column_config.NumberColumn(format="%.2f"),
                                            "Severity covered": st.column_config.NumberColumn(format="%.2f")})
                st.download_button("Download this list (CSV)", s2[keep].to_csv(index=False).encode("utf-8"),
                                   file_name="best_fit_candidates.csv", mime="text/csv")

    with tab_stat:
        m = st.selectbox("Weak stat", metrics)
        row_all = tg[tg["metric"] == m]
        team_val = float(row_all["mariners"].iloc[0])
        row = apply_filters(row_all)
        is_fld = m == "fielding_Rtot"
        show_pos = "position" in row.columns and row["position"].fillna("").astype(str).str.len().gt(0).any()
        show_bats = "bats" in row.columns and row["bats"].fillna("").astype(str).str.len().gt(0).any()
        samp_label = "Innings" if is_fld else "PA or IP"
        if "sample" in row.columns and row["sample"].notna().any():
            hi = int(np.ceil(row["sample"].max()))
            min_s = st.slider(f"Minimum {samp_label}", 0, max(hi, 1), 0,
                              help="Raise this to drop small-sample players whose number may not hold up.") if hi > 0 else 0
            row = row[row["sample"] >= min_s]
        st.metric(f"Mariners {m}", f"{team_val:g}")
        # for fielding, value and better_by are the same number (runs above average), so show it once
        cols = (["rank", "name", "Team"] + (["position"] if show_pos else []) + (["bats"] if show_bats else [])
                + ["sample", "value"] + ([] if is_fld else ["better_by"]))
        show = row[cols].head(25).rename(columns={"sample": samp_label,
                                                  "value": "Runs above average" if is_fld else m,
                                                  "better_by": "Better than Mariners by"})
        if is_fld:
            st.caption("Fielding runs are already measured against an average fielder (0), so this is the best gloves "
                       "in the league, not a comparison with the Mariners' team total.")
        if len(row) < len(row_all):
            st.caption(f"Filters on: {len(row)} of {len(row_all)} players match (players whose position or bat side "
                       "is unknown are hidden when those filters are used).")
        if row.empty:
            st.info("No players match these filters for this stat.")
        else:
            st.dataframe(show, width="stretch", hide_index=True)
            st.download_button("Download this list (CSV)", show.to_csv(index=False).encode("utf-8"),
                               file_name=f"targets_{m}.csv", mime="text/csv")

    if "fielding_Rtot" not in metrics:
        st.info("No fielding list here: weakness_targets.py could not load player fielding data. "
                "The fielding targets on screen 3 still work.")


# ───────────────────────── 4. the plan ─────────────────────────
elif screen == "4. The plan":
    st.title("4. The plan")
    fa = load_scored(folder)
    if fa is None:
        st.warning("free_agent_scored.csv not found. Run offseason_report.py first.")
        st.stop()

    st.write("Build a signing slate. The totals use last year's salary as a rough stand-in; "
             "players with no price match are counted separately, not as $0.")
    fa_all = fa                      # keeps the Mariners' own rows for the depth chart
    fa = external_only(fa).sort_values("recommendation_score", ascending=False).reset_index(drop=True)
    if fa.empty:
        st.info("No outside free agents to build a slate from yet. Screen 5 works now; this screen fills in "
                "after free agency opens.")
        st.stop()
    fa["label"] = fa["name"] + " (" + fa.get("position_group", pd.Series("", index=fa.index)).fillna("").astype(str) + ")"
    fa = fa.drop_duplicates("label")   # the same player listed twice at one position counts once

    # Pick a position, then choose players at that position. The slate is kept in
    # session state, so picks made at one position stay when you switch to another.
    ALL = "All positions (top 60)"
    groups_avail = sorted(fa["position_group"].dropna().unique()) if "position_group" in fa.columns else []
    view = st.selectbox("Pick a position", [ALL] + groups_avail)
    options = (fa.head(60) if view == ALL else fa[fa["position_group"] == view])["label"].tolist()
    slate = st.session_state.setdefault("slate", [])
    picked = st.multiselect(
        "Players to sign" if view == ALL else f"Players to sign at {view}", options,
        default=[l for l in slate if l in options], key=f"pick_{view}")
    st.session_state["slate"] = [l for l in slate if l not in options] + picked
    chosen = st.session_state["slate"]
    if chosen and st.button("Clear my slate"):
        st.session_state["slate"] = []
        for k in [k for k in st.session_state if k.startswith("pick_")]:
            del st.session_state[k]
        st.rerun()
    sel = fa[fa["label"].isin(chosen)]

    # A player listed at two positions is still one person with one salary.
    one = sel.drop_duplicates("name")
    ok = priced(one)
    total = float(one.loc[ok, "Est_Cost_M"].sum()) if "Est_Cost_M" in one.columns else 0.0
    a, b, c = st.columns(3)
    a.metric("Players", len(one))
    b.metric("Priced total ($M/yr)", f"{total:.1f}")
    c.metric("Unpriced players", int((~ok).sum()))
    multi = sel["name"].value_counts()
    for nm in multi[multi > 1].index:
        st.caption(f"{nm} is on your slate at {int(multi[nm])} positions -- his salary is counted once.")

    if not sel.empty:
        if "position_group" in sel.columns:
            counts = sel["position_group"].value_counts()
            for g, n in counts[counts > 1].items():
                st.warning(f"{n} players chosen at {g} -- they compete for the same job.")
        show = [c for c in ["name", "position_group", "recommendation_score", "Est_Cost_M", "bats",
                            "proj_OPS_2027", "age_risk"] if c in sel.columns]
        st.dataframe(sel[show], width="stretch", hide_index=True)
        st.download_button("Download my slate (CSV)", sel[show].to_csv(index=False).encode("utf-8"),
                           file_name="signing_slate.csv", mime="text/csv")

        needs = read_export(folder, "diagnosed_needs.csv")
        if needs is not None:
            worst = needs[needs["Verdict"] == "WEAKNESS"].sort_values("Severity_Score", ascending=False).head(3)
            st.subheader("Biggest weaknesses to keep in mind")
            for _, r in worst.iterrows():
                st.write(f"- **{r['Category']} / {r['Metric']}**: team {r['Team_Value']:g} vs league {r['League_Avg']:g}")
    else:
        st.info("Pick players above to see the total cost and any position overlaps.")

    # ---- depth chart: current vs. with the players added above ----
    st.header("Depth chart")
    st.write("How each position looks now and with the players you added. The starter changes only "
             "when an added player beats the current starter on 2026 stats (OPS for hitters, ERA for "
             "pitchers; pitching groups are compared with the staff average).")
    starters_df = merge_position_starters(read_export(folder, "mariners_starters.csv"),
                                          read_export(folder, "position_needs.csv"))
    depth = build_depth(fa_all, sel, starters_df)
    rows, now_ops, after_ops = [], [], []
    for g, (pitch, d) in depth.items():
        now, after = starter_now_and_after(pitch, d)
        fmt = (lambda x: f"{x:.2f}") if pitch else (lambda x: f"{x:.3f}")
        if now is None:
            continue
        row = {"Position": g,
               "Now": f"{now['Player']} ({fmt(now['Stat'])})" if pd.notna(now["Stat"]) else f"{now['Player']} (n/a)",
               "With signings": "n/a", "Change": ""}
        if after is not None:
            row["With signings"] = f"{after['Player']} ({fmt(after['Stat'])})"
            if pd.notna(now["Stat"]):
                diff = after["Stat"] - now["Stat"]
                row["Change"] = f"{diff:+.2f}" if pitch else f"{diff:+.3f}"
                if not pitch:
                    now_ops.append(now["Stat"]); after_ops.append(after["Stat"])
        rows.append(row)
    # Positions with no rows in the scored file can't be shown; list them instead of dropping them.
    have = {r["Position"] for r in rows}
    missing = [g for g in STANDARD_GROUPS if g not in have]
    for g in missing:
        rows.append({"Position": g, "Now": "no data yet", "With signings": "", "Change": ""})
    order = {g: i for i, g in enumerate(STANDARD_GROUPS)}
    rows.sort(key=lambda r: order.get(r["Position"], 99))
    if missing:
        st.caption("No current starter is available at " + ", ".join(missing) + ". Run  python mariners_starters.py  "
                   "to build the Mariners' starters from their own stats.")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if now_ops:
        n_now, n_after = float(np.mean(now_ops)), float(np.mean(after_ops))
        st.metric(f"Average starting OPS ({len(now_ops)} hitter spots)", f"{n_after:.3f}",
                  f"{n_after - n_now:+.3f} vs now ({n_now:.3f})")
        st.caption("A simple average of the starting spots, not a team projection.")
    if depth:
        names = list(depth)
        g = st.selectbox("See the full depth at a position", names,
                         index=names.index(view) if view in names else 0)
        pitch, d = depth[g]
        d = d.sort_values("Stat", ascending=pitch, na_position="last").reset_index(drop=True)
        d.index = d.index + 1
        st.dataframe(d, width="stretch")