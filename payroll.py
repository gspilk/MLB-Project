"""
payroll.py
Team payroll tracking -- replaces the old payroll scraper.

Salary data isn't in the MLB Stats API, and salary sites don't allow
scraping, so payroll comes from two small CSV files you keep by hand
(contract figures are widely reported public facts; enter them yourself):

    data/inputs/payroll_players.csv     one row per player on a team's books
        Team, Player, Status, Pos, Age, Salary
        Team    full team name, e.g. "Seattle Mariners"
        Status  active | injured | retained | minors
        Salary  this season's salary in dollars, e.g. 12500000

    data/inputs/payroll_league.csv      one row per team (league comparison)
        Team, Active, Injured, Retained, Total
        dollars; leave a cell blank if you don't have it

Blank templates are created the first time this runs.

Usage:
    from payroll import get_mariners_payroll, get_league_cash_totals
    payroll = get_mariners_payroll()
"""

import pandas as pd

import mlb_api

PLAYER_COLS = ["Team", "Player", "Status", "Pos", "Age", "Salary"]
LEAGUE_COLS = ["Team", "Active", "Injured", "Retained", "Total"]
STATUS_KEYS = {"active": "active_payroll", "injured": "injured_list",
               "retained": "retained", "minors": "minors"}


def _players() -> pd.DataFrame:
    df = mlb_api.read_input_csv("payroll_players.csv", PLAYER_COLS, "player payroll")
    if not df.empty:
        df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
        df["Status"] = df["Status"].astype(str).str.strip().str.lower()
    return df


def get_team_payroll(team_name: str = "Seattle Mariners", force_refresh: bool = False) -> dict:
    """
    Returns:
        {
            "active_payroll":  DataFrame -- active roster salaries
            "injured_list":    DataFrame -- IL players, salary still counts
            "retained":        DataFrame -- salary owed to traded/released players
            "totals":          DataFrame -- payroll summary (category, value)
            "minors":          DataFrame -- 40-man/minor-league salaries
        }
    force_refresh is accepted for compatibility; the CSV is read fresh.
    """
    df = _players()
    team = df[df["Team"] == team_name] if not df.empty else df
    if team.empty:
        print(f"  [info] no payroll rows for {team_name} in data/inputs/payroll_players.csv")
    result = {}
    for status, key in STATUS_KEYS.items():
        part = team[team["Status"] == status] if not team.empty else team
        result[key] = part[["Player", "Pos", "Age", "Salary"]].sort_values(
            "Salary", ascending=False).reset_index(drop=True) if not part.empty else pd.DataFrame(
            columns=["Player", "Pos", "Age", "Salary"])
    totals = [{"category": key.replace("_", " ").title(), "value": result[key]["Salary"].sum()}
              for key in STATUS_KEYS.values()]
    totals.append({"category": "Total", "value": sum(t["value"] for t in totals)})
    result["totals"] = pd.DataFrame(totals)
    return result


def get_league_cash_totals(force_refresh: bool = False) -> pd.DataFrame:
    """Every team's payroll totals, from payroll_league.csv -- or, if that's
    empty, summed from payroll_players.csv for whatever teams it covers."""
    df = mlb_api.read_input_csv("payroll_league.csv", LEAGUE_COLS, "league payroll totals")
    if not df.empty:
        for c in LEAGUE_COLS[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("Total", ascending=False).reset_index(drop=True)
    players = _players()
    if players.empty:
        return pd.DataFrame(columns=LEAGUE_COLS)
    pivot = players.pivot_table(index="Team", columns="Status", values="Salary",
                                aggfunc="sum", fill_value=0)
    out = pd.DataFrame({"Team": pivot.index,
                        "Active": pivot.get("active", 0),
                        "Injured": pivot.get("injured", 0),
                        "Retained": pivot.get("retained", 0)}).reset_index(drop=True)
    out["Total"] = out[["Active", "Injured", "Retained"]].sum(axis=1)
    return out.sort_values("Total", ascending=False).reset_index(drop=True)


def get_mariners_payroll(force_refresh: bool = False) -> dict:
    """Same as get_team_payroll('Seattle Mariners')."""
    return get_team_payroll("Seattle Mariners", force_refresh)


def get_all_teams_payroll(force_refresh: bool = False) -> dict:
    """{team_name: payroll_dict} for every team in payroll_players.csv."""
    players = _players()
    teams = sorted(players["Team"].dropna().unique()) if not players.empty else []
    return {t: get_team_payroll(t) for t in teams}


if __name__ == "__main__":
    payroll = get_mariners_payroll()
    for label, df in payroll.items():
        print(f"\n-- {label.upper()} --")
        print(df.to_string(index=False) if not df.empty else "  (empty)")
