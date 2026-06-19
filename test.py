# Standard libraries
import requests
from bs4 import BeautifulSoup

def decode_secret_message(url):
    # Fetch and parse the published Google Doc
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract (x, y, character) rows from the table
    table = soup.find("table")
    rows = table.find_all("tr")

    datapoints = []
    for row in rows[1:]:  # skip header row
        cells = row.find_all("td")
        x = int(cells[0].text)
        char = cells[1].text
        y = int(cells[2].text)
        datapoints.append((x, y, char))

    # Determine grid size
    max_x = max(item[0] for item in datapoints)
    max_y = max(item[1] for item in datapoints)

    # Build grid filled with spaces, then place characters
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    for x, y, char in datapoints:
        grid[y][x] = char

    # Print grid (reversed so y=0 is at the top)
    for row in grid[::-1]:
        print(''.join(row))

# calling decode_secret_message to decode and build the message in the url
decode_secret_message("https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub")