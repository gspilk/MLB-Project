"""
mlb_free_agents.py
Builds data/inputs/free_agents.csv (the free-agent list the pipeline reads)
from the public MLB Stats API instead of scraping a website.

HOW IT WORKS
  1. Pulls MLB transactions for a date window (default: Oct 1 of this year
     through today) from  /transactions .
  2. Keeps players whose transaction text says they declared / elected free
     agency, and drops anyone who has since signed or been re-signed.
  3. Looks each player up in  /people  for age, position and (for pitchers)
     games started, then maps that to the position groups the rest of the
     project uses ("Catchers", "First Basemen", ... "Right-Handed Relievers").
  4. Writes name, age_2027, position_group, option_detail.

READ THIS BEFORE TRUSTING IT
  - TIMING: MLB players only "declare free agency" after the World Series
    ends, so before then this finds few or no players. That is not a bug:
    the Stats API has no list of players whose contracts merely EXPIRE.
    Run it again in November, or fill the CSV by hand until then.
  - Not yet run against the live API (written from the documented
    endpoints). It prints the transaction type codes it saw, so if the
    wording differs you can see it and adjust FREE_AGENCY_WORDS below.
  - Pitchers: the API has one 'P' position. Starters vs relievers is decided
    by 2026 games started (>= 10 = Starting Pitcher); pitchers with no stats
    are labelled 'Unclassified Pitchers' rather than guessed.
  - It will NOT overwrite a CSV that already has players unless you pass
    --overwrite, so a hand-typed list is never lost.

Usage:
    python mlb_free_agents.py
    python mlb_free_agents.py --since 2026-10-01 --season 2026 --overwrite
"""
import argparse
import datetime as dt
import os

import pandas as pd
import requests

BASE = "https://statsapi.mlb.com/api/v1"
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "inputs")
OUT_PATH = os.path.join(INPUT_DIR, "free_agents.csv")

FREE_AGENCY_WORDS = ("declared free agency", "elected free agency")
SIGNED_WORDS = ("signed", "re-signed", "agreed to terms", "contract extension")
STARTER_GS = 10

POSITION_GROUPS = {
    "C": "Catchers", "1B": "First Basemen", "2B": "Second Basemen", "SS": "Shortstops",
    "3B": "Third Basemen", "LF": "Left Fielders", "CF": "Center Fielders",
    "RF": "Right Fielders", "DH": "Designated Hitters",
}
OUT_COLS = ["name", "age_2027", "position_group", "option_detail"]


def _get_json(path, params=None, fetch=None):
    if fetch is not None:
        return fetch(path, params or {})
    url = f"{BASE}/{path.lstrip('/')}"
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"MLB Stats API request failed: {url} {params or ''} -> {e}") from e


def get_transactions(start: dt.date, end: dt.date, fetch=None) -> list:
    data = _get_json("transactions", {"startDate": start.strftime("%m/%d/%Y"),
                                      "endDate": end.strftime("%m/%d/%Y"), "sportId": 1}, fetch)
    tx = data.get("transactions")
    if tx is None:
        raise RuntimeError(f"No 'transactions' key in the response (keys: {list(data)}) -- "
                           f"the endpoint shape may have changed.")
    return tx


def declared_free_agents(transactions: list) -> pd.DataFrame:
    """One row per player who declared free agency and has not signed since.
    Transactions are processed in date order so a later signing cancels it."""
    rows = {}
    for t in sorted(transactions, key=lambda x: x.get("date") or ""):
        person = t.get("person") or {}
        pid, text = person.get("id"), (t.get("description") or "")
        if not pid:
            continue
        low = text.lower()
        if any(w in low for w in FREE_AGENCY_WORDS):
            rows[pid] = {"player_id": pid, "name": person.get("fullName"), "option_detail": text,
                         "date": t.get("date")}
        elif pid in rows and any(w in low for w in SIGNED_WORDS):
            del rows[pid]
    return pd.DataFrame(list(rows.values()), columns=["player_id", "name", "option_detail", "date"])


def get_people(ids: list, season: int, fetch=None) -> dict:
    """{id: {age, abbr, hand, gs}} for each id, in chunks of 50."""
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = _get_json("people", {"personIds": ",".join(map(str, chunk)),
                                    "hydrate": f"stats(group=[pitching],type=[season],season={season})"}, fetch)
        for p in data.get("people", []):
            gs = None
            for block in p.get("stats", []) or []:
                for split in block.get("splits", []) or []:
                    gs = (split.get("stat") or {}).get("gamesStarted", gs)
            out[p.get("id")] = {
                "age": p.get("currentAge"),
                "abbr": (p.get("primaryPosition") or {}).get("abbreviation"),
                "hand": (p.get("pitchHand") or {}).get("code"),
                "gs": int(gs) if gs is not None else None,
            }
    return out


def position_group(info: dict) -> str:
    abbr = info.get("abbr")
    if abbr in POSITION_GROUPS:
        return POSITION_GROUPS[abbr]
    if abbr in ("P", "SP", "RP", "TWP"):
        gs = info.get("gs")
        if abbr == "SP" or (gs is not None and gs >= STARTER_GS):
            return "Starting Pitchers"
        if abbr == "RP" or gs is not None:
            return "Left-Handed Relievers" if info.get("hand") == "L" else "Right-Handed Relievers"
        return "Unclassified Pitchers"
    return "Unclassified"


def build(start: dt.date, end: dt.date, season: int, fetch=None) -> pd.DataFrame:
    tx = get_transactions(start, end, fetch)
    seen = {}
    for t in tx:
        key = (t.get("typeCode"), t.get("typeDesc"))
        seen[key] = seen.get(key, 0) + 1
    print(f"Saw {len(tx)} transactions; types: {dict(sorted(seen.items(), key=lambda kv: -kv[1])[:8])}")
    fa = declared_free_agents(tx)
    if fa.empty:
        return pd.DataFrame(columns=OUT_COLS)
    info = get_people(fa["player_id"].tolist(), season, fetch)
    fa["position_group"] = fa["player_id"].map(lambda i: position_group(info.get(i, {})))
    # age_2027: age during the 2027 season (currentAge today + 1 once this season is over)
    fa["age_2027"] = fa["player_id"].map(lambda i: (info.get(i, {}).get("age") or float("nan")) + 1)
    return fa.drop_duplicates("name")[OUT_COLS].sort_values(["position_group", "name"]).reset_index(drop=True)


def run(since=None, season=None, overwrite=False, fetch=None, out_path=None) -> pd.DataFrame:
    out_path = out_path or OUT_PATH
    today = dt.date.today()
    start = dt.date.fromisoformat(since) if since else dt.date(today.year, 10, 1)
    season = season or today.year
    if os.path.exists(out_path) and not overwrite:
        try:
            if len(pd.read_csv(out_path)) > 0:
                print(f"{out_path} already has players -- not overwriting (use --overwrite).")
                return pd.read_csv(out_path)
        except pd.errors.EmptyDataError:
            pass
    df = build(start, today, season, fetch)
    if df.empty:
        print("No players have declared free agency in that window yet. "
              "Free agency opens after the World Series -- try again then, or fill the CSV by hand.")
        return df
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    bad = df[df["position_group"].str.startswith("Unclassified")]
    print(f"Wrote {len(df)} free agents -> {out_path}")
    if len(bad):
        print(f"  {len(bad)} could not be placed in a position group -- check these by hand: "
              f"{', '.join(bad['name'])}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the free-agent input list from the MLB Stats API.")
    ap.add_argument("--since", help="start date, YYYY-MM-DD (default: Oct 1 of this year)")
    ap.add_argument("--season", type=int, help="season used for games-started stats")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    run(a.since, a.season, a.overwrite)