"""
mlbtr_debug.py
Shows exactly what _fetch_lines() is actually extracting from the real
page -- the live run found 0 matches, meaning the content-container
guess (soup.find("article") or a class regex) likely isn't matching
this site's real HTML structure. Prints a sample of real extracted
lines so the parser can be fixed against what's actually there.

Usage:
    python mlbtr_debug.py
"""

from free_agent import _fetch_lines, POSITION_HEADERS

lines = _fetch_lines()
print(f"\n{len(lines)} total lines extracted\n")

print("-- First 30 lines --")
for line in lines[:30]:
    print(f"  {line!r}")

print("\n-- Checking for real position headers as exact matches --")
for header in POSITION_HEADERS:
    found = header in lines
    print(f"  {header!r}: {'FOUND' if found else 'not found as exact match'}")

print("\n-- Lines containing 'Catchers' or 'Crawford' (partial match) --")
for line in lines:
    if "Catchers" in line or "Crawford" in line:
        print(f"  {line!r}")