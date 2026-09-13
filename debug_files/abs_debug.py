"""
abs_debug.py
One-off diagnostic: lists every real table id found on bbref's ABS
challenge page (both live in the HTML and hidden inside comments,
same pattern as fielding/40-man tables elsewhere in this project) --
so abs_challenge_scraper.py can target real ids instead of a guess.

Usage:
    python abs_debug.py
"""

import time
import requests
from bs4 import BeautifulSoup, Comment

URL = "https://www.baseball-reference.com/friv/abs-challenges.shtml"


def _make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


print(f"  [GET] {URL}")
session = _make_session()
resp = session.get(URL, timeout=15)
resp.raise_for_status()
resp.encoding = "utf-8"
time.sleep(3)

soup = BeautifulSoup(resp.text, "html.parser")

print("\n--- LIVE tables (directly in the HTML) ---")
for tag in soup.find_all("table"):
    tid = tag.get("id", "(no id)")
    print(f"  id={tid}")

print("\n--- Tables hidden inside HTML comments ---")
for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
    if "<table" not in comment:
        continue
    c_soup = BeautifulSoup(comment, "html.parser")
    for tag in c_soup.find_all("table"):
        tid = tag.get("id", "(no id)")
        print(f"  id={tid}")