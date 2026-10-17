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
    return df


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

if not os.path.isdir(folder):
    st.error(f"Folder not found: {folder}. Run offseason_report.py first.")
    st.stop()


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

    st.subheader("Severity by area")
    cats = sorted(needs["Category"].unique())
    pick = st.multiselect("Categories", cats, default=cats)
    view = needs[needs["Category"].isin(pick)]
    st.bar_chart(view.set_index("Metric")["Severity_Score"])
    with st.expander("Full table"):
        st.dataframe(view, width="stretch", hide_index=True)
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
        max_cost = c3.slider("Max cost ($M)", 0.0, hi, hi)
    bats = None
    if "bats" in fa.columns:
        bats = c4.multiselect("Bats", ["L", "R", "S"], default=["L", "R", "S"])

    v = fa.copy()
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

    show = [c for c in ["name", "position_group", "age_2027", "Team", "bats", "recommendation_score",
                        "Est_Cost_M", "OPS", "proj_OPS_2027", "Rtot", "option_detail"] if c in v.columns]
    st.dataframe(v[show].head(50), width="stretch", hide_index=True)

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
    st.write("For each weak stat, the players who are clearly better at it than the Mariners are. "
             "These are **candidates, not free agents**: any non-Mariners player appears, whatever his contract.")
    needs = read_export(folder, "diagnosed_needs.csv")
    summ = read_export(folder, "weakness_targets_summary.csv")

    metrics = list(dict.fromkeys(tg["metric"]))
    if not metrics:
        st.info("No players beat the Mariners on the weak stats with enough playing time.")
        st.stop()
    if needs is not None:
        sev = needs.set_index("Metric")["Severity_Score"].to_dict()
        metrics.sort(key=lambda m: -sev.get(m, 0))
    m = st.selectbox("Weak stat", metrics)

    # optional filters, so you can say what you actually need (e.g. right-handed outfielders)
    OF_LABEL = "Outfield (LF/CF/RF)"
    pos_all = (sorted({str(p) for p in tg["position"].dropna() if str(p)}) if "position" in tg.columns else [])
    f1, f2 = st.columns(2)
    pos_choice = f1.multiselect("Position", ([OF_LABEL] if any(p in ("LF", "CF", "RF") for p in pos_all) else []) + pos_all,
                                help="Applies to hitters and fielders. Pitching stats have no position.") if pos_all else []
    has_bats = "bats" in tg.columns and tg["bats"].fillna("").astype(str).str.len().gt(0).any()
    bats_choice = f2.multiselect("Bats", ["L", "R", "S"],
                                 help="From player_handedness.csv (run mlb_stats_api.py to build it).") if has_bats else []

    def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
        if pos_choice and "position" in d.columns and d["position"].fillna("").astype(str).str.len().gt(0).any():
            allowed = {p for p in pos_choice if p != OF_LABEL} | ({"LF", "CF", "RF"} if OF_LABEL in pos_choice else set())
            d = d[d["position"].fillna("").astype(str).isin(allowed)]
        if bats_choice and "bats" in d.columns and d["bats"].fillna("").astype(str).str.len().gt(0).any():
            d = d[d["bats"].fillna("").astype(str).isin(bats_choice)]
        return d

    row_all = tg[tg["metric"] == m]
    team_val = float(row_all["mariners"].iloc[0])
    row = apply_filters(row_all)
    st.metric(f"Mariners {m}", f"{team_val:g}")
    is_fld = m == "fielding_Rtot"
    show_pos = "position" in row.columns and row["position"].fillna("").astype(str).str.len().gt(0).any()
    show_bats = "bats" in row.columns and row["bats"].fillna("").astype(str).str.len().gt(0).any()
    # for fielding, value and better_by are the same number (runs above average), so show it once
    cols = (["rank", "name", "Team"] + (["position"] if show_pos else []) + (["bats"] if show_bats else [])
            + ["sample", "value"] + ([] if is_fld else ["better_by"]))
    show = row[cols].head(25).rename(columns={"sample": "Innings" if is_fld else "PA or IP",
                                              "value": "Runs above average" if is_fld else m,
                                              "better_by": "Better than Mariners by"})
    if is_fld:
        st.caption("Fielding runs are already measured against an average fielder (0), so this is the best gloves "
                   "in the league, not a comparison with the Mariners' team total.")
    if (pos_choice or bats_choice) and len(row) < len(row_all):
        st.caption(f"Filters on: {len(row)} of {len(row_all)} players match (players whose position or bat side "
                   "is unknown are hidden).")
    if row.empty:
        st.info("No players match these filters for this stat.")
    else:
        st.dataframe(show, width="stretch", hide_index=True)

    if summ is not None and not summ.empty:
        st.subheader("Fix several problems at once")
        st.caption("Fit score = how far the player beats the Mariners on each weak stat (in league standard deviations), weighted by how severe that weakness is. Higher is better.")
        pool_names = {"hit": "Hitters", "rel": "Relievers", "sp": "Starters"}
        pools = [p for p in ["hit", "rel", "sp"] if p in set(summ["pool"])]
        kind = st.radio("Show", [pool_names[p] for p in pools], horizontal=True)
        inv = {v: k for k, v in pool_names.items()}
        s2 = apply_filters(summ[summ["pool"] == inv[kind]]).head(15).rename(columns={
            "weak_metrics_beaten": "Weak stats beaten", "severity_covered": "Severity covered", "fit_score": "Fit score"})
        st.dataframe(s2[[c for c in ["name", "Team", "position", "bats", "Fit score", "Weak stats beaten",
                                     "Severity covered", "metrics"] if c in s2.columns
                         and (c not in ("position", "bats") or s2[c].fillna("").astype(str).str.len().gt(0).any())]],
                     width="stretch", hide_index=True)
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
    fa_all = fa
    fa = fa.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
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
                            "proj_OPS_2027"] if c in sel.columns]
        st.dataframe(sel[show], width="stretch", hide_index=True)

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
    depth = build_depth(fa_all, sel, read_export(folder, "mariners_starters.csv"))
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