"""
Load the project's existing data (CSV / parquet / cached API results) into MySQL.

    python load_mysql.py                 # loads 2025 and 2026
    python load_mysql.py 2026            # one season

Connection settings come from environment variables so no password is ever
stored in the repo:
    MLB_DB_HOST      (default localhost)
    MLB_DB_PORT      (default 3306)
    MLB_DB_USER      (default root)
    MLB_DB_PASSWORD  (asked for in the terminal if not set)
    MLB_DB_NAME      (default mlb_project; created if missing)

Re-running is safe: each table's rows for the season being loaded are
replaced, never duplicated.
Needs:  pip install sqlalchemy pymysql
"""
import getpass
import os
import re
import sys

import pandas as pd
from sqlalchemy import create_engine, text

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(HERE, "data", "exports")
CACHE = os.path.join(HERE, "data", "cache")
DATA = os.path.join(HERE, "data")
SCHEMA = os.path.join(HERE, "sql", "schema.sql")
SEASONS = [2025, 2026]


# ── connection ───────────────────────────────────────────────────────────────
def get_engine(echo=False):
    host = os.environ.get("MLB_DB_HOST", "localhost")
    port = os.environ.get("MLB_DB_PORT", "3306")
    user = os.environ.get("MLB_DB_USER", "root")
    pw = os.environ.get("MLB_DB_PASSWORD") or getpass.getpass(f"MySQL password for {user}: ")
    name = os.environ.get("MLB_DB_NAME", "mlb_project")
    from urllib.parse import quote_plus
    base = f"mysql+pymysql://{quote_plus(user)}:{quote_plus(pw)}@{host}:{port}"
    boot = create_engine(base + "/?charset=utf8mb4")
    with boot.begin() as c:
        c.execute(text(f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    return create_engine(f"{base}/{name}?charset=utf8mb4", echo=echo)


def run_schema(engine, path=SCHEMA):
    sql = open(path, encoding="utf-8").read()
    sql = re.sub(r"--[^\n]*", "", sql)
    with engine.begin() as c:
        for stmt in (s.strip() for s in sql.split(";")):
            if stmt:
                c.execute(text(stmt))


# ── helpers ──────────────────────────────────────────────────────────────────
def _replace_rows(engine, table, df, season, extra_where=""):
    """Delete this season's rows, then append df. Keeps the schema/indexes intact."""
    if df is None or df.empty:
        print(f"  [skip] {table} {season}: no data")
        return 0
    with engine.begin() as c:
        c.execute(text(f"DELETE FROM {table} WHERE season = :s {extra_where}"), {"s": season})
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=1000, method="multi")
    print(f"  [ok]   {table} {season}: {len(df):,} rows")
    return len(df)


def _bool_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def _num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _read(path):
    if not os.path.exists(path):
        print(f"  [warn] not found: {os.path.relpath(path, HERE)}")
        return None
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


# ── loaders (one per table) ──────────────────────────────────────────────────
def load_games(engine, season, get_games=None):
    if get_games is None:
        import mlb_api
        get_games = mlb_api.get_season_games
    g = get_games(season)
    if g is None or g.empty:
        return _replace_rows(engine, "games", None, season)
    df = pd.DataFrame({
        "game_pk": g["game_pk"], "season": season, "game_date": g["date"],
        "home_id": g["home_id"], "away_id": g["away_id"],
        "home_runs": g["home_runs"], "away_runs": g["away_runs"],
        "innings": g["innings"], "day_night": g["day_night"], "status": g["status"],
        "played": g["played"].astype(int),
    }).drop_duplicates("game_pk", keep="last")
    return _replace_rows(engine, "games", df, season)


def load_plate_appearances(engine, season):
    df = _read(os.path.join(EXPORTS, f"pitch_count_pas_{season}.csv"))
    if df is None:
        return 0
    df = df.rename(columns={"date": "game_date", "event": "event_type"})
    df.insert(0, "season", season)
    df = _bool_cols(df, ["is_starter", "is_ab", "is_walk", "is_hbp", "is_sac_fly", "is_k"])
    return _replace_rows(engine, "plate_appearances", df, season)


def load_pitches(engine, season):
    df = _read(os.path.join(EXPORTS, f"pitch_velocity_pitches_{season}.csv"))
    if df is None:
        return 0
    df = df.rename(columns={"date": "game_date", "call": "call_code"})
    df.insert(0, "season", season)
    df = _bool_cols(df, ["is_starter"])
    return _replace_rows(engine, "pitches", df, season)


def load_batting_players(engine, season):
    d = _read(os.path.join(CACHE, f"batting_all_players_{season}.parquet"))
    if d is None:
        return 0
    df = pd.DataFrame({
        "season": season, "player_id": d["player_id"], "name": d["Name"], "age": d.get("Age"),
        "team": d["Tm"], "league": d.get("Lg"), "pos": d.get("Pos"),
        "g": d["G"], "pa": d["PA"], "ab": d["AB"], "h": d["H"],
        "doubles": d["2B"], "triples": d["3B"], "hr": d["HR"], "rbi": d["RBI"],
        "sb": d["SB"], "bb": d["BB"], "so": d["SO"],
        "ba": d["BA"], "obp": d["OBP"], "slg": d["SLG"], "ops": d["OPS"],
        "ops_plus": d["OPS+"], "war": d["WAR"],
    })
    df = _num(df, ["age", "ba", "obp", "slg", "ops", "ops_plus", "war"])
    df = df.dropna(subset=["player_id"]).drop_duplicates(["season", "player_id", "team"])
    return _replace_rows(engine, "batting_players", df, season)


def load_pitching_players(engine, season):
    d = _read(os.path.join(CACHE, f"pitching_all_players_{season}.parquet"))
    if d is None:
        return 0
    df = pd.DataFrame({
        "season": season, "player_id": d["player_id"], "name": d["Name"], "age": d.get("Age"),
        "team": d["Tm"], "league": d.get("Lg"),
        "w": d["W"], "l": d["L"], "g": d["G"], "gs": d["GS"], "sv": d["SV"],
        "ip": d["IP"], "h": d["H"], "r": d["R"], "er": d["ER"], "hr": d["HR"],
        "bb": d["BB"], "so": d["SO"], "bf": d["BF"],
        "era": d["ERA"], "whip": d["WHIP"], "fip": d["FIP"], "era_plus": d["ERA+"],
        "war": d["WAR"],
    })
    df = _num(df, ["age", "ip", "era", "whip", "fip", "era_plus", "war"])
    df = df.dropna(subset=["player_id"]).drop_duplicates(["season", "player_id", "team"])
    return _replace_rows(engine, "pitching_players", df, season)


def load_statcast(engine, season):
    total = 0
    frames = []
    for kind in ("batter", "pitcher"):
        d = _read(os.path.join(DATA, f"statcast_{kind}s_{season}.csv"))
        if d is None:
            continue
        df = d.rename(columns={"last_name, first_name": "name"}).drop(columns=["year"], errors="ignore")
        df.insert(0, "player_type", kind)
        df.insert(0, "season", season)
        frames.append(df)
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "player_type", "player_id"])
    return _replace_rows(engine, "statcast_players", df, season)


LOADERS = [load_games, load_plate_appearances, load_pitches,
           load_batting_players, load_pitching_players, load_statcast]


def main(seasons):
    engine = get_engine()
    run_schema(engine)
    for season in seasons:
        print(f"\nSeason {season}")
        for fn in LOADERS:
            try:
                fn(engine, season)
            except Exception as e:
                print(f"  [warn] {fn.__name__} {season} failed: {type(e).__name__}: {e}")
    print("\nDone. Open MySQL Workbench and run:  USE mlb_project;  SHOW TABLES;")


if __name__ == "__main__":
    seasons = [int(a) for a in sys.argv[1:]] or SEASONS
    main(seasons)