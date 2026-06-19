import time
import requests
from bs4 import BeautifulSoup, Comment


def _make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.baseball-reference.com/",
    })
    return session


def debug_tables(url: str):
    print(f"[GET] {url}\n")
    session = _make_session()
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    time.sleep(3)
    soup = BeautifulSoup(resp.text, "html.parser")

    print("── tables in live HTML ──")
    live = soup.find_all("table")
    if live:
        for t in live:
            print(f"  id='{t.get('id','(none)')}' class={t.get('class','')}")
    else:
        print("  (none)")

    print("\n── tables inside HTML comments ──")
    comments = soup.find_all(string=lambda t: isinstance(t, Comment))
    found = False
    for comment in comments:
        c_soup = BeautifulSoup(comment, "html.parser")
        for t in c_soup.find_all("table"):
            tid = t.get("id", "(none)")
            if tid != "(none)":
                print(f"  id='{tid}'")
                found = True
    if not found:
        print("  (none)")


if __name__ == "__main__":
    urls = [
        "https://www.baseball-reference.com/leagues/majors/2026.shtml",
    ]
    for url in urls:
        print(f"\n{'='*60}")
        debug_tables(url)
        time.sleep(4)