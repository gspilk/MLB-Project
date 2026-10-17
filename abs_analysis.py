"""
abs_analysis.py -- 2026 ABS (automated ball-strike) challenge analysis.

Reads the CSVs you download by hand from Baseball Savant's ABS challenge
leaderboard and saved in data/inputs/. File names are matched loosely:
  starts with "abs", contains batting/batter, catching/catcher, pitching/pitcher
  or fielding/fielder,
  and contains "team" for the team view (otherwise it is treated as the
  player view).  e.g.  abs-challenges-2026-batting-team.csv
If a file was downloaded twice, the newest one is used.

Writes data/exports/abs_summary.csv and prints how the Mariners rank
(runs above average, overturn rate vs. the rate expected for the same pitches)
and, for player files, which Mariners (and which league players) gain or lose
the most runs with their challenges.

Usage:   python abs_analysis.py            (from the project folder)
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(ROOT, "data", "inputs")
EXPORTS = os.path.join(ROOT, "data", "exports")
# words that can appear in a Savant file name -> the role they mean
ROLE_WORDS = {"batting": "batting", "batter": "batting", "catching": "catching",
              "catcher": "catching", "pitching": "pitching", "pitcher": "pitching",
              "fielding": "fielding", "fielder": "fielding"}
NEEDED = ["entity_name", "team_abbr", "n_challenges", "n_overturns", "rate_overturns",
          "exp_rate_overturns", "runs_above_average", "runs_gained", "runs_cost"]
OUT_COLS = ["role", "level", "name", "team", "challenges", "overturns", "overturn_pct",
            "expected_pct", "gap_pct", "runs_above_average", "runs_gained", "runs_cost",
            "rank_runs"]
MIN_PLAYER_CHALLENGES = 1


def find_files(folder=INPUTS):
    """{(role, level): path} -- newest file wins."""
    found = {}
    if not os.path.isdir(folder):
        return found
    for fn in os.listdir(folder):
        low = fn.lower()
        if not (low.startswith("abs") and low.endswith(".csv")):
            continue
        role = next((r for w, r in ROLE_WORDS.items() if w in low), None)
        if role is None:
            continue
        level = "team" if "team" in low else "player"
        path = os.path.join(folder, fn)
        key = (role, level)
        if key not in found or os.path.getmtime(path) > os.path.getmtime(found[key]):
            found[key] = path
    return found


def load(path, role, level):
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns {missing}")
    out = pd.DataFrame({
        "role": role, "level": level,
        "name": df["entity_name"], "team": df["team_abbr"],
        "challenges": pd.to_numeric(df["n_challenges"], errors="coerce"),
        "overturns": pd.to_numeric(df["n_overturns"], errors="coerce"),
        "overturn_pct": pd.to_numeric(df["rate_overturns"], errors="coerce") * 100,
        "expected_pct": pd.to_numeric(df["exp_rate_overturns"], errors="coerce") * 100,
        "runs_above_average": pd.to_numeric(df["runs_above_average"], errors="coerce"),
        "runs_gained": pd.to_numeric(df["runs_gained"], errors="coerce"),
        "runs_cost": pd.to_numeric(df["runs_cost"], errors="coerce"),
    })
    out["gap_pct"] = out["overturn_pct"] - out["expected_pct"]
    out["rank_runs"] = out["runs_above_average"].rank(ascending=False, method="min")
    for c in ["overturn_pct", "expected_pct", "gap_pct", "runs_above_average",
              "runs_gained", "runs_cost"]:
        out[c] = out[c].round(2)
    return out[OUT_COLS]


def run(inputs=INPUTS, exports=EXPORTS, team="SEA", verbose=True):
    files = find_files(inputs)
    if not files:
        raise RuntimeError(
            f"No ABS files found in {inputs}. Download them from Baseball Savant's "
            "ABS challenge leaderboard (Download CSV) and save them there.")
    frames = [load(p, role, level) for (role, level), p in sorted(files.items())]
    allrows = pd.concat(frames, ignore_index=True)
    os.makedirs(exports, exist_ok=True)
    allrows.to_csv(os.path.join(exports, "abs_summary.csv"), index=False)
    if verbose:
        _print(allrows, team)
        print(f"\nWrote {len(allrows)} rows -> {os.path.join(exports, 'abs_summary.csv')}")
    return allrows


def _print(allrows, team):
    for (role, level), g in allrows.groupby(["role", "level"], sort=False):
        tot_ch, tot_ov = g["challenges"].sum(), g["overturns"].sum()
        print(f"\n== {role} / {level} ==  league overturn rate "
              f"{100 * tot_ov / tot_ch:.1f}% on {int(tot_ch)} challenges")
        if level == "team":
            row = g[g["team"] == team]
            if not row.empty:
                r = row.iloc[0]
                print(f"  {team}: {int(r.challenges)} challenges, {r.overturn_pct:.1f}% overturned "
                      f"(expected {r.expected_pct:.1f}%, gap {r.gap_pct:+.1f}), "
                      f"{r.runs_above_average:+.2f} runs above average, "
                      f"rank {int(r.rank_runs)}/{len(g)}")
            print("  best :", ", ".join(f"{x.team} {x.runs_above_average:+.1f}"
                                         for x in g.nlargest(3, "runs_above_average").itertuples()))
            print("  worst:", ", ".join(f"{x.team} {x.runs_above_average:+.1f}"
                                         for x in g.nsmallest(3, "runs_above_average").itertuples()))
        else:
            q = g[g["challenges"] >= MIN_PLAYER_CHALLENGES]
            mine = q[q["team"] == team].sort_values("runs_above_average")
            if not mine.empty:
                print(f"  {team} players ({MIN_PLAYER_CHALLENGES}+ challenges), worst first:")
                for x in mine.itertuples():
                    print(f"    {x.name:<24} {int(x.challenges):>3} chal  {x.overturn_pct:>5.1f}% "
                          f"(exp {x.expected_pct:.1f}%)  {x.runs_above_average:+.2f} runs")
            print("  league best :", ", ".join(f"{x.name} {x.runs_above_average:+.1f}"
                                                for x in q.nlargest(5, "runs_above_average").itertuples()))
            print("  league worst:", ", ".join(f"{x.name} {x.runs_above_average:+.1f}"
                                                for x in q.nsmallest(5, "runs_above_average").itertuples()))


if __name__ == "__main__":
    try:
        run()
    except (RuntimeError, ValueError) as e:
        print(f"\n{e}")
        sys.exit(1)