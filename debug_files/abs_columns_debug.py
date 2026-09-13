"""
abs_columns_debug.py
Prints the REAL column names and a few sample rows for the by_team and
player_stats tables, so abs_challenge_scraper.py's Mariners-filtering
logic can be fixed against the real structure.

Usage:
    python abs_columns_debug.py
"""

from abs_challenges_scraper import get_abs_challenges

data = get_abs_challenges()

print("\n--- BY_TEAM: real columns ---")
print(data["by_team"].columns.tolist())
print("\n--- BY_TEAM: first 3 rows ---")
print(data["by_team"].head(3).to_string(index=False))

print("\n\n--- BY_PLAYER: real columns ---")
print(data["by_player"].columns.tolist())
print("\n--- BY_PLAYER: first 3 rows ---")
print(data["by_player"].head(3).to_string(index=False))