"""
platoon_debug.py
Verifies the real bbref splits page URL and table structure before
trusting platoon_lineup_optimizer.py's numbers -- same discipline as
every other scraper tonight. Uses Cal Raleigh as the real test case.

Usage:
    python platoon_debug.py
"""

from platoon_lineup_optimizer import get_roster_bbref_ids, get_platoon_splits

print("Step 1: getting real bbref IDs from the current roster...")
ids = get_roster_bbref_ids()
print(f"\nFound {len(ids)} real player IDs. Sample:")
for name, pid in list(ids.items())[:5]:
    print(f"  {name}: {pid}")

raleigh_id = ids.get("Cal Raleigh")
if raleigh_id:
    print(f"\nStep 2: testing real splits fetch for Cal Raleigh ({raleigh_id})...")
    splits = get_platoon_splits("Cal Raleigh", raleigh_id, player_type="batter", force_refresh=True)
    print(f"\nResult: {splits}")
else:
    print("\nCal Raleigh not found in the real roster IDs -- check Step 1's output above")