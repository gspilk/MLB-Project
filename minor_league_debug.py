"""
minor_league_debug.py

Diagnostic script -- confirms the REAL table ids on bbref's minor league
affiliate page before minor_league_scraper.py is trusted to parse them.

Same discipline used earlier for position_debug.py, data_structure_debug.py,
fielding_debug.py, abs_debug.py, platoon_table_debug.py: don't guess ids,
confirm them from the live page first.

Run:
    python minor_league_debug.py
"""

import re
import requests
from bs4 import BeautifulSoup, Comment

URL = "https://www.baseball-reference.com/register/affiliate.cgi?id=SEA&year=2026"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def list_live_tables(soup: BeautifulSoup):
    print("\n--- Tables found directly in live HTML ---")
    tables = soup.find_all("table")
    if not tables:
        print("  (none)")
    for t in tables:
        tid = t.get("id", "<no id>")
        caption = t.find("caption")
        cap_text = caption.get_text(strip=True) if caption else ""
        ncols = len(t.find_all("th")[:1])
        print(f"  id={tid!r:30} caption={cap_text!r}")


def list_commented_tables(soup: BeautifulSoup):
    print("\n--- Tables hidden inside HTML comments ---")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    found_any = False
    for c in comments:
        if "<table" not in c:
            continue
        inner = BeautifulSoup(c, "html.parser")
        tables = inner.find_all("table")
        for t in tables:
            found_any = True
            tid = t.get("id", "<no id>")
            caption = t.find("caption")
            cap_text = caption.get_text(strip=True) if caption else ""
            print(f"  id={tid!r:30} caption={cap_text!r}")
    if not found_any:
        print("  (none)")


def list_divs_with_ids(soup: BeautifulSoup):
    # Top Prospects and similar sections are sometimes plain divs, not tables
    print("\n--- div/section wrappers with an id containing 'prospect' or 'top' ---")
    divs = soup.find_all(["div", "section"], id=True)
    hits = [d for d in divs if re.search(r"prospect|top", d["id"], re.I)]
    if not hits:
        print("  (none found by id keyword -- will need a manual look at headings below)")
    for d in hits:
        print(f"  id={d['id']!r} tag={d.name}")


def list_headings(soup: BeautifulSoup):
    print("\n--- Section headings on the page (h2/h3) ---")
    for h in soup.find_all(["h2", "h3"]):
        text = h.get_text(strip=True)
        if text:
            print(f"  {h.name}: {text!r}")


def main():
    print(f"Fetching: {URL}")
    soup = fetch_soup(URL)

    list_headings(soup)
    list_live_tables(soup)
    list_commented_tables(soup)
    list_divs_with_ids(soup)

    print(
        "\nDone. Match the real ids/captions printed above against what "
        "minor_league_scraper.py expects (aff_batting, aff_pitching, "
        "team_batting, team_pitching, top_prospects) and fix any mismatch "
        "before trusting the scraper's output."
    )


if __name__ == "__main__":
    main()