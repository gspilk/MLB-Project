"""
statcast_scraper.py
Reads Statcast CSV exports from Baseball Savant leaderboards.

Save CSVs as:
  data/statcast_batters_2026.csv
  data/statcast_pitchers_2026.csv
"""

import os
import pandas as pd

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
BATTER_FILE  = os.path.join(DATA_DIR, "statcast_batters_2026.csv")
PITCHER_FILE = os.path.join(DATA_DIR, "statcast_pitchers_2026.csv")

# exact CSV column names from savant
RENAME = {
    "last_name, first_name": "Name",
    "pa":                    "PA",
    "k_percent":             "K%",
    "bb_percent":            "BB%",
    "woba":                  "wOBA",
    "xwoba":                 "xwOBA",
    "sweet_spot_percent":    "Sweet%",
    "barrel_batted_rate":    "Barrel%",
    "hard_hit_percent":      "HardHit%",
    "avg_best_speed":        "EV50",
    "avg_hyper_speed":       "AdjEV",
    "whiff_percent":         "Whiff%",
    "swing_percent":         "Swing%",
}

MARINERS_BATTERS = [
    "Rodríguez, Julio", "Arozarena, Randy", "Raley, Luke",
    "Young, Cole", "Crawford, J.P.", "Raleigh, Cal",
    "Naylor, Josh", "Canzone, Dominic", "Donovan, Brendan",
    "Emerson, Colt", "Garver, Mitch", "Pereda, Jhonny",
    "Refsnyder, Rob", "Rivas, Leo", "Robles, Víctor",
    "Wisdom, Patrick", "Joe, Connor", "Bliss, Ryan",
]

MARINERS_PITCHERS = [
    "Kirby, George", "Woo, Bryan", "Hancock, Emerson",
    "Gilbert, Logan", "Castillo, Luis", "Miller, Bryce",
    "Muñoz, Andrés", "Brash, Matt", "Ferrer, José A.",
    "Bazardo, Eduard", "Criswell, Cooper", "Speier, Gabe",
    "Hoppe, Alex", "Legumina, Casey", "Davila, Nick",
    "Wilcox, Cole",
]


def _load(filepath: str, is_pitcher: bool = False) -> pd.DataFrame:
    if not os.path.exists(filepath):
        print(f"  [warn] not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df = df.rename(columns=RENAME)

    # for pitchers rename wOBA/xwOBA to _against
    if is_pitcher:
        df = df.rename(columns={
            "wOBA":  "wOBA_against",
            "xwOBA": "xwOBA_against",
        })

    # numeric conversion
    skip = {"Name", "year", "player_id"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def get_statcast(batter_file=BATTER_FILE,
                 pitcher_file=PITCHER_FILE):
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[load] statcast batters ...")
    bat = _load(batter_file, is_pitcher=False)
    if not bat.empty:
        print(f"  [ok]  {len(bat)} batters, {len(bat.columns)} cols")

    print("[load] statcast pitchers ...")
    pit = _load(pitcher_file, is_pitcher=True)
    if not pit.empty:
        print(f"  [ok]  {len(pit)} pitchers, {len(pit.columns)} cols")

    return bat, pit


def get_mariners_batters(df):
    if df.empty or "Name" not in df.columns:
        return pd.DataFrame()
    sea = df[df["Name"].isin(MARINERS_BATTERS)]
    if sea.empty:
        last_names = [n.split(",")[0] for n in MARINERS_BATTERS]
        sea = df[df["Name"].str.split(",").str[0].isin(last_names)]
    return sea.reset_index(drop=True)


def get_mariners_pitchers(df):
    if df.empty or "Name" not in df.columns:
        return pd.DataFrame()
    sea = df[df["Name"].isin(MARINERS_PITCHERS)]
    if sea.empty:
        last_names = [n.split(",")[0] for n in MARINERS_PITCHERS]
        sea = df[df["Name"].str.split(",").str[0].isin(last_names)]
    return sea.reset_index(drop=True)


def get_luck_analysis(df, is_pitcher=False):
    woba = "wOBA_against" if is_pitcher else "wOBA"
    xwoba = "xwOBA_against" if is_pitcher else "xwOBA"
    if df.empty or woba not in df.columns or xwoba not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["luck_gap"] = (d[woba] - d[xwoba]).round(3)
    d["verdict"]  = d["luck_gap"].apply(
        lambda x: "LUCKY" if x > 0.020
        else ("UNLUCKY" if x < -0.020 else "NEUTRAL")
    )
    if is_pitcher:
        d["verdict"] = d["luck_gap"].apply(
            lambda x: "UNLUCKY (better than ERA shows)" if x > 0.020
            else ("LUCKY (worse than ERA shows)" if x < -0.020
                  else "NEUTRAL")
        )
    cols = [c for c in ["Name","PA","BF",woba,xwoba,"luck_gap",
                         "verdict","Barrel%","HardHit%","EV50",
                         "K%","Whiff%"]
            if c in d.columns]
    return d[cols].sort_values("luck_gap", ascending=False).reset_index(drop=True)


def get_rankings(df, stat, min_pa=50, top=20, ascending=False):
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    pa_col = "BF" if "BF" in d.columns else "PA"
    if pa_col in d.columns:
        d = d[pd.to_numeric(d[pa_col], errors="coerce") >= min_pa]
    want = ["Name", pa_col, stat, "K%", "BB%", "Whiff%",
            "Barrel%", "HardHit%", "EV50"]
    cols = [c for c in want if c in d.columns]
    return (d[cols].dropna(subset=[stat])
            .sort_values(stat, ascending=ascending)
            .head(top)
            .reset_index(drop=True))


def rank_player(df, name, stat, ascending=False):
    if df.empty or stat not in df.columns:
        return {}
    col     = pd.to_numeric(df[stat], errors="coerce")
    matches = df[df["Name"].str.contains(name, case=False, na=False)]
    if matches.empty:
        return {"error": f"{name} not found"}
    val    = col[matches.index[0]]
    ranked = col.rank(ascending=ascending, method="min")
    total  = int(col.notna().sum())
    rank   = int(ranked[matches.index[0]])
    return {
        "name":   matches.iloc[0]["Name"],
        "value":  round(val, 3),
        "rank":   rank,
        "total":  total,
        "pct":    round(rank / total * 100, 1),
    }


if __name__ == "__main__":
    batters, pitchers = get_statcast()

    if batters.empty and pitchers.empty:
        print("\nSave CSVs to:")
        print(f"  {BATTER_FILE}")
        print(f"  {PITCHER_FILE}")
    else:
        print("\n── Mariners batting (sorted by xwOBA) ──")
        sea_bat = get_mariners_batters(batters)
        if not sea_bat.empty:
            cols = [c for c in ["Name","PA","wOBA","xwOBA",
                                 "Barrel%","HardHit%","EV50","K%","BB%"]
                    if c in sea_bat.columns]
            print(sea_bat[cols].sort_values("xwOBA", ascending=False)
                  .to_string(index=False))

        print("\n── Mariners batting luck ──")
        luck = get_luck_analysis(sea_bat, is_pitcher=False)
        print(luck.to_string(index=False) if not luck.empty else "  no data")

        print("\n── Mariners pitching (sorted by xwOBA) ──")
        sea_pit = get_mariners_pitchers(pitchers)
        if not sea_pit.empty:
            cols = [c for c in ["Name","BF","xwOBA_against","wOBA_against",
                                 "K%","BB%","Whiff%",
                                 "HardHit%_against","Barrel%_against"]
                    if c in sea_pit.columns]
            print(sea_pit[cols].sort_values("xwOBA_against")
                  .to_string(index=False))

        print("\n── Mariners pitching luck ──")
        pluck = get_luck_analysis(sea_pit, is_pitcher=True)
        print(pluck.to_string(index=False) if not pluck.empty else "  no data")

        print("\n── MLB xwOBA leaders batters (min 50 PA) ──")
        print(get_rankings(batters, "xwOBA", min_pa=50, top=15,
                           ascending=False).to_string(index=False))

        print("\n── MLB xwOBA leaders pitchers (min 50 BF) ──")
        print(get_rankings(pitchers, "xwOBA_against", min_pa=50, top=15,
                           ascending=True).to_string(index=False))