"""
Refresh holdings_static.json from Fidelity's public ETF research pages.

Runs in GitHub Actions on a weekly schedule (see .github/workflows/refresh-holdings.yml).
Defensive by design: a ticker is only updated when the fetched table passes sanity
checks; otherwise its previous snapshot is kept. If nothing succeeds, the file is
left untouched and the job still exits 0 (no broken commits, ever).
"""

import json
import re
import sys
import time
from html.parser import HTMLParser

from curl_cffi import requests

TICKERS = [
    "QQQ", "QQQM", "FTEC", "SMH", "VOO", "SPY", "IVV", "VTI",
    "VGT", "XLK", "SOXX", "IXN", "IYW", "SCHG", "VUG", "MGK",
    "SCHD", "VYM", "VIG", "JEPQ",
]

URL = (
    "https://research2.fidelity.com/fidelity/screeners/etf/public/"
    "etfholdings.asp?symbol={ticker}&view=Holdings"
)

OUT_FILE = "holdings_static.json"


class TableParser(HTMLParser):
    """Minimal stdlib HTML table extractor (same logic as app.py)."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self._depth = 0
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self.tables.append([])
        elif self._depth and tag == "tr":
            self._row = []
        elif self._depth and tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self._depth:
            self._depth -= 1
        elif self._depth and tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None
        elif self._depth and tag in ("td", "th") and self._cell is not None:
            text = " ".join("".join(self._cell).split())
            if self._row is not None:
                self._row.append(text)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
BAD_SYMBOLS = {"NAN", "-", "--", "CASH", "USD", "N/A", "NA"}


def parse_holdings(html):
    """Return (rows, asof) where rows = [[symbol, weight_fraction], ...]; ([], None) on failure."""
    asof_match = re.search(r"AS OF\s*([0-9/]{8,10})", html, re.IGNORECASE)
    asof = asof_match.group(1) if asof_match else None

    parser = TableParser()
    parser.feed(html)

    for table in parser.tables:
        if len(table) < 6:
            continue

        header = [str(c).lower() for c in table[0]]
        sym_idx = [i for i, c in enumerate(header) if "symbol" in c]
        wt_idx = [i for i, c in enumerate(header) if "weight" in c]

        if not sym_idx or not wt_idx:
            continue

        si, wi = sym_idx[0], wt_idx[0]
        rows = []

        for row in table[1:]:
            if len(row) <= max(si, wi):
                continue

            symbol = str(row[si]).strip().upper()

            if not symbol or symbol in BAD_SYMBOLS or not SYMBOL_RE.match(symbol):
                continue

            try:
                weight = float(str(row[wi]).replace(",", "")) / 100.0
            except ValueError:
                continue

            if weight > 0:
                rows.append([symbol, round(weight, 6)])

        total = sum(w for _, w in rows)

        # Sanity: a real holdings table has >= 5 names and sums to roughly 100%.
        if len(rows) >= 5 and 0.4 <= total <= 1.6:
            return rows, asof

    return [], None


def fetch_ticker(ticker):
    response = requests.get(URL.format(ticker=ticker), impersonate="chrome", timeout=25)

    if response.status_code != 200 or not response.text:
        raise RuntimeError(f"HTTP {response.status_code}")

    return parse_holdings(response.text)


def main():
    try:
        with open(OUT_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        data = {}

    updated, kept = [], []

    for ticker in TICKERS:
        try:
            rows, asof = fetch_ticker(ticker)
        except Exception as err:
            print(f"{ticker}: fetch failed ({err}) - keeping previous snapshot")
            kept.append(ticker)
            time.sleep(2)
            continue

        if rows:
            data[ticker] = {"asof": asof or "", "rows": rows}
            updated.append(ticker)
            print(f"{ticker}: {len(rows)} holdings, sum {sum(w for _, w in rows):.4f}, as of {asof}")
        else:
            kept.append(ticker)
            print(f"{ticker}: parse failed sanity checks - keeping previous snapshot")

        time.sleep(2)

    print(f"\nUpdated: {', '.join(updated) if updated else 'none'}")
    print(f"Kept previous: {', '.join(kept) if kept else 'none'}")

    if not updated:
        print("No successful fetches - leaving holdings_static.json untouched.")
        sys.exit(0)

    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))

    print(f"Wrote {OUT_FILE} with {len(data)} tickers.")


if __name__ == "__main__":
    main()
