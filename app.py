import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from html.parser import HTMLParser

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", page_icon="📊", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

APP_VERSION = "2.2 (Sep 5, 2026)"

# Your actual portfolio. Every tab starts with these tickers at these percentages.
DEFAULT_PORT = "SPMO, QNDX, FTEC, SMH"
DEFAULT_WEIGHTS = {"SPMO": 25, "QNDX": 25, "FTEC": 25, "SMH": 25}
DEFAULT_BENCH = "VOO, QQQ"
# Watchlist = funds you might consider, shown next to a "MY PORTFOLIO" comparison row.
# QQQ and VGT are the higher-fee twins of QNDX and FTEC, kept here for side-by-side checks.
DEFAULT_WATCH = "QQQ, VGT, QQQM, SCHG, VUG, XLK"

ALL_NUM_COLS = [
    "Yield (TTM)", "Yield (Fwd)", "1D", "1W", "1M", "YTD",
    "1Y Total", "3Y Total", "3Y CAGR", "5Y Total", "5Y CAGR",
    "10Y Total", "10Y CAGR", "15Y Total", "15Y CAGR",
    "20Y Total", "20Y CAGR", "25Y Total", "25Y CAGR",
    "Max Total", "Max CAGR",
    "3Y Div CAGR", "5Y Div CAGR", "10Y Div CAGR", "15Y Div CAGR",
    "20Y Div CAGR", "Max Div CAGR"
]

PERF_PERIODS = ["1M", "YTD", "1Y Total", "3Y CAGR", "5Y CAGR", "10Y CAGR", "15Y CAGR", "20Y CAGR"]

# Shown as percentages in tables. "Expense" is averaged into the portfolio row; "Max DD" is not.
EXTRA_PCT_COLS = ["Expense", "Max DD"]

TOTAL_YEARS = [1, 3, 5, 10, 15, 20, 25]
DIV_YEARS = [3, 5, 10, 15, 20]

# How far history extends past the fund's own inception.
# Each entry: ordered fallback chain of (symbol, kind).
#   "fund"     = older fund tracking the same/equivalent index (total return, incl. dividends)
#   "tr_index" = total-return index (includes dividends)
#   "pr_index" = price-return index (NO dividends -> slightly understates total return)
EXTEND_CHAIN = {
    "QQQ":  [("^XNDX", "tr_index"), ("^NDX", "pr_index")],
    "QQQM": [("QQQ", "fund"), ("^XNDX", "tr_index"), ("^NDX", "pr_index")],
    "QNDX": [("QQQ", "fund"), ("^XNDX", "tr_index"), ("^NDX", "pr_index")],
    "FTEC": [("VGT", "fund"), ("XLK", "fund")],
    "VGT":  [("XLK", "fund")],
    "IYW":  [("XLK", "fund")],
    "SMH":  [("SOXX", "fund"), ("^SOX", "pr_index")],
    "SOXX": [("^SOX", "pr_index")],
    "VOO":  [("^SP500TR", "tr_index"), ("VFINX", "fund"), ("^GSPC", "pr_index")],
    "IVV":  [("^SP500TR", "tr_index"), ("VFINX", "fund"), ("^GSPC", "pr_index")],
    "SPLG": [("^SP500TR", "tr_index"), ("VFINX", "fund"), ("^GSPC", "pr_index")],
    "SPY":  [("^SP500TR", "tr_index"), ("VFINX", "fund"), ("^GSPC", "pr_index")],
    "VTI":  [("VTSMX", "fund"), ("^GSPC", "pr_index")],
    "SCHG": [("VUG", "fund"), ("IWF", "fund")],
    "MGK":  [("VUG", "fund"), ("IWF", "fund")],
    "VUG":  [("IWF", "fund")],
}

# Human-readable description of what the extension is based on (shown in Deep Dive).
INDEX_NOTE = {
    "QQQ":  "Tracks the Nasdaq-100. Extended with ^XNDX (total return) and ^NDX (price-only, index since 1985).",
    "QQQM": "Tracks the Nasdaq-100. Extended with QQQ, then Nasdaq-100 index history.",
    "QNDX": "State Street SPDR Portfolio Nasdaq-100 ETF (launched Jun 23, 2026, 0.10%/yr). Tracks the same "
            "Nasdaq-100 Index as QQQ, so everything before its launch is QQQ's history and, before 1999, "
            "the index itself. Holdings and dividend figures borrow QQQ's until the fund has its own.",
    "SPMO": "Tracks the S&P 500 Momentum Index (about 100 S&P 500 stocks with the strongest recent "
            "price momentum, rebalanced twice a year). Fund launched Oct 9, 2015; there is no older fund "
            "on the same index, so its history is NOT extended - 15Y/20Y figures are blank on purpose.",
    "FTEC": "Tracks MSCI USA IMI Info Tech 25/50. Extended with VGT (same index family, 2004) and XLK (1998).",
    "VGT":  "Tracks MSCI US IMI Info Tech 25/50. Extended with XLK (1998).",
    "SMH":  "Tracks MVIS US Listed Semiconductor 25. Extended with SOXX (2001) and ^SOX index (price-only).",
    "SOXX": "Tracks ICE Semiconductor (formerly PHLX SOX). Extended with ^SOX index (price-only).",
    "VOO":  "Tracks S&P 500. Extended with ^SP500TR (total return, 1988), VFINX (1980), ^GSPC (price-only, 1927).",
    "SPY":  "Tracks S&P 500. Extended with ^SP500TR (total return, 1988), VFINX (1980), ^GSPC (price-only, 1927).",
    "SCHG": "Tracks DJ US Large-Cap Growth. Extended with VUG and IWF (similar large-growth indexes).",
}

IND_MAP = {
    "NVDA": "Semi - GPU/AI Logic", "AMD": "Semi - CPU/GPU", "INTC": "Semi - IDM (Mfg)",
    "TSM": "Semi - Foundry (Mfg)", "AVGO": "Semi - Networking/RF", "QCOM": "Semi - Mobile/Comms",
    "MU": "Semi - Memory (DRAM/NAND)", "TXN": "Semi - Analog/Embedded",
    "ADI": "Semi - Analog/Mixed Signal", "NXPI": "Semi - Auto/IoT", "ON": "Semi - Power/Sensors",
    "MCHP": "Semi - Microcontrollers", "MPWR": "Semi - Power Management",
    "ASML": "Semi Equip - Lithography", "AMAT": "Semi Equip - Materials Eng",
    "LRCX": "Semi Equip - Etch/Deposition", "KLAC": "Semi Equip - Process Control",
    "TER": "Semi Equip - Test/Measurement", "ENTG": "Semi Equip - Adv Materials",
    "SNPS": "Software - Chip Design (EDA)", "CDNS": "Software - Chip Design (EDA)",
    "ARM": "Semi - IP/Architecture", "MSFT": "Cloud & OS Infrastructure",
    "ORCL": "Cloud/DB", "ADBE": "Software - Creative", "CRM": "Software - Enterprise",
    "AAPL": "Consumer Electronics", "CSCO": "Networking Hardware", "GOOG": "Search/Ads",
    "GOOGL": "Search/Ads", "META": "Social Media", "AMZN": "E-Commerce/Cloud",
    "TSLA": "Auto Mfr - EV", "HD": "Home Improvement", "WMT": "Big Box Retail",
    "LLY": "Pharma", "UNH": "Health Ins", "JPM": "Bank", "V": "Payments",
    "MA": "Payments", "COST": "Warehouse Club", "NFLX": "Media - Streaming",
    "PEP": "Beverages", "KO": "Beverages", "PANW": "Cybersecurity",
    "CRWD": "Cybersecurity", "NOW": "Software - IT Services", "PLTR": "Data Analytics",
    "INTU": "Financial Soft", "ISRG": "Medical Devices", "AMGN": "Biotech",
    "QQQM": "Tech / Growth ETF", "QQQ": "Tech / Growth ETF", "QNDX": "Tech / Growth ETF",
    "SPMO": "S&P 500 Momentum ETF", "MTUM": "Momentum ETF", "XLK": "Technology ETF",
    "IXN": "Global Technology ETF", "VGT": "Technology ETF", "FTEC": "Technology ETF",
    "SMH": "Semiconductor ETF", "SOXX": "Semiconductor ETF", "IYW": "Technology ETF",
    "SCHG": "Growth ETF", "VUG": "Growth ETF", "MGK": "Mega-Cap Growth ETF",
    "VOO": "S&P 500 ETF", "SPY": "S&P 500 ETF", "IVV": "S&P 500 ETF",
    "SPLG": "S&P 500 ETF", "VTI": "Total Market ETF", "SCHD": "Dividend ETF",
    "VYM": "Dividend ETF", "VIG": "Dividend ETF", "JEPQ": "Covered-Call ETF"
}

B_INCEPT = {
    "XLK": "1998-12-16", "IXN": "2001-11-12", "SMH": "2011-12-20",
    "QQQ": "1999-03-10", "QQQM": "2020-10-13", "MGK": "2007-12-17",
    "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07",
    "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10",
    "SCHD": "2011-10-20", "VIG": "2006-04-21", "JEPQ": "2022-05-03",
    "SOXX": "2001-07-10", "IYW": "2000-05-15", "VUG": "2004-01-26",
    "IVV": "2000-05-15", "VTI": "2001-05-24", "SPMO": "2015-10-09", "QNDX": "2026-06-23"
}

# Verified annual expense ratios (as a fraction: 0.0013 = 0.13% per year).
# Anything not listed here is looked up from Yahoo Finance at run time.
EXPENSE_RATIO = {
    "SPMO": 0.0013, "QQQ": 0.0020, "QQQM": 0.0015, "QNDX": 0.0010, "VGT": 0.0009, "SMH": 0.0035,
    "VOO": 0.0003, "VTI": 0.0003,
}


# --- HELPERS ---
def parse_tickers(text):
    tickers = []
    seen = set()

    for raw in str(text).replace("\n", ",").split(","):
        ticker = raw.strip().upper()

        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)

    return tickers


def pct_or_na(value):
    return f"{value:.2%}" if value is not None and pd.notnull(value) else "N/A"


def normalize_ratio(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default

        value = float(value)

        if value > 1:
            value = value / 100

        return value
    except Exception:
        return default


def tz_naive(series):
    """Strip timezone so fund / index series from Yahoo can be spliced together."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    series = series.copy()

    try:
        series.index = series.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass

    return series


def format_dataframe(df):
    df = df.copy()

    for col in ALL_NUM_COLS + EXTRA_PCT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").apply(
                lambda x: f"{x:.2%}" if pd.notnull(x) else "-"
            )

    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").apply(
            lambda x: f"${x:.2f}" if pd.notnull(x) else "-"
        )

    return df


def return_by_sessions(close, sessions):
    if len(close) <= sessions:
        return None

    start = float(close.iloc[-sessions - 1])
    end = float(close.iloc[-1])

    return (end / start) - 1 if start > 0 else None


def ytd_return(close):
    if close.empty:
        return None

    year = pd.Timestamp.now().year
    start_ts = pd.Timestamp(f"{year}-01-01")

    before_start = close[close.index < start_ts]
    after_start = close[close.index >= start_ts]

    if not before_start.empty:
        start = float(before_start.iloc[-1])
    elif not after_start.empty:
        start = float(after_start.iloc[0])
    else:
        return None

    end = float(close.iloc[-1])
    return (end / start) - 1 if start > 0 else None


def period_total_return(close, years):
    if close.empty:
        return None

    target = close.index[-1] - pd.DateOffset(years=years)

    if target < close.index[0]:
        return None

    try:
        idx = close.index.get_indexer([target], method="nearest")[0]
    except Exception:
        return None

    if idx < 0:
        return None

    gap_days = abs((close.index[idx] - target).days)
    max_gap_days = 45 if years == 1 else 100

    if gap_days > max_gap_days:
        return None

    start = float(close.iloc[idx])
    end = float(close.iloc[-1])

    return (end / start) - 1 if start > 0 else None


def dividend_growth_streak(annual_div):
    annual_div = annual_div[annual_div > 0].sort_index()

    if len(annual_div) < 2:
        return 0

    streak = 0
    values = annual_div.values

    for i in range(len(values) - 1, 0, -1):
        if values[i] > values[i - 1]:
            streak += 1
        else:
            break

    return streak


def coerce_weight(value):
    try:
        if value is None or pd.isna(value):
            return None

        if isinstance(value, str):
            has_percent = "%" in value
            value = value.replace("%", "").replace(",", "").strip()

            if value in {"", "-", "--", "N/A", "NA"}:
                return None

            value = float(value)

            if has_percent:
                value = value / 100
        else:
            value = float(value)

        if value > 1:
            value = value / 100

        return value if 0 <= value <= 1 else None
    except Exception:
        return None


def ticker_like_score(series):
    sample = series.dropna().astype(str).str.strip().head(50)

    if sample.empty:
        return 0

    return sample.str.match(r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,12}$").mean()


def normalize_holdings_frame(raw):
    if raw is None:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    try:
        if isinstance(raw, pd.Series):
            df = raw.to_frame(name="Raw_Weight").reset_index()
        else:
            df = raw.copy()

            index_series = pd.Series(df.index.astype(str), dtype="object")
            use_index_as_symbol = (
                not isinstance(df.index, pd.RangeIndex)
                or ticker_like_score(index_series) >= 0.35
            )

            if use_index_as_symbol:
                df["_IndexSymbol"] = df.index.astype(str)
                cols = ["_IndexSymbol"] + [col for col in df.columns if col != "_IndexSymbol"]
                df = df[cols]

            df = df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    if df.empty:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    df.columns = [
        " ".join(map(str, col)).strip() if isinstance(col, tuple) else str(col).strip()
        for col in df.columns
    ]

    symbol_col = None

    for col in df.columns:
        col_lower = col.lower()

        if (
            col_lower in {"symbol", "ticker", "holdingticker", "_indexsymbol"}
            or "ticker" in col_lower
            or "symbol" in col_lower
        ):
            if ticker_like_score(df[col]) >= 0.20:
                symbol_col = col
                break

    if symbol_col is None:
        scored_symbols = sorted(
            [(ticker_like_score(df[col]), col) for col in df.columns],
            reverse=True
        )

        if scored_symbols and scored_symbols[0][0] >= 0.35:
            symbol_col = scored_symbols[0][1]

    if symbol_col is None:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    weight_col = None

    for col in df.columns:
        if col == symbol_col:
            continue

        col_lower = col.lower()

        if (
            "holding percent" in col_lower
            or "weight" in col_lower
            or "percent" in col_lower
            or "% of fund" in col_lower
            or "% of net" in col_lower
        ):
            parsed = df[col].map(coerce_weight)

            if parsed.notna().sum() > 0:
                weight_col = col
                break

    if weight_col is None:
        weight_candidates = []

        for col in df.columns:
            if col == symbol_col:
                continue

            parsed = df[col].map(coerce_weight)
            parse_score = parsed.notna().mean()
            name_bonus = 0.5 if any(
                key in col.lower()
                for key in ["weight", "percent", "%", "holding"]
            ) else 0

            if parse_score > 0:
                weight_candidates.append((parse_score + name_bonus, col))

        if weight_candidates:
            weight_col = max(weight_candidates)[1]

    if weight_col is None:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    out = df[[symbol_col, weight_col]].copy()
    out.columns = ["Symbol", "Raw_Weight"]

    out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
    out["Raw_Weight"] = out["Raw_Weight"].map(coerce_weight)

    out = out.dropna(subset=["Symbol", "Raw_Weight"])
    out = out[out["Raw_Weight"] > 0]
    out = out[out["Symbol"].str.match(r"^[A-Z0-9][A-Z0-9.\-]{0,12}$", na=False)]
    out = out[~out["Symbol"].isin({"-", "--", "CASH", "USD", "N/A", "NA"})]

    return out[["Symbol", "Raw_Weight"]]


def merge_goog(df):
    if df.empty:
        return pd.DataFrame(columns=["Symbol", "Weight"])

    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper().replace({"GOOGL": "GOOG"})
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    return (
        df.dropna(subset=["Symbol", "Weight"])
        .groupby("Symbol", as_index=False)["Weight"]
        .sum()
        .sort_values(by="Weight", ascending=False)
    )


# --- FULL-BASKET HOLDINGS (Fidelity public ETF research, works for most US ETFs) ---
FIDELITY_HOLDINGS_URL = (
    "https://research2.fidelity.com/fidelity/screeners/etf/public/"
    "etfholdings.asp?symbol={ticker}&view=Holdings"
)


def _http_get_text(url, timeout=15):
    """Fetch with browser impersonation (curl_cffi ships with yfinance) to avoid
    bot-blocking of plain urllib from cloud servers; falls back to urllib."""
    try:
        from curl_cffi import requests as curl_requests

        response = curl_requests.get(url, impersonate="chrome", timeout=timeout)

        if getattr(response, "status_code", 0) == 200 and response.text:
            return response.text
    except Exception:
        pass

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("latin-1", errors="replace")


class _TableParser(HTMLParser):
    """Minimal stdlib HTML table extractor (no pandas.read_html dependency)."""

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


def fetch_fidelity_holdings(ticker):
    """
    Full basket holdings (every position, not just top 10) from Fidelity's public
    ETF research pages. Works for most US-listed ETFs regardless of issuer.
    Returns (DataFrame[Symbol, Raw_Weight], as_of_date_or_None).
    """
    empty = pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    try:
        html = _http_get_text(FIDELITY_HOLDINGS_URL.format(ticker=urllib.parse.quote(ticker)))
    except Exception:
        return empty, None

    asof_match = re.search(r"AS OF\s*([0-9/]{8,10})", html, re.IGNORECASE)
    asof = asof_match.group(1) if asof_match else None

    try:
        parser = _TableParser()
        parser.feed(html)
    except Exception:
        return empty, None

    for rows in parser.tables:
        if len(rows) < 6:
            continue

        header = [str(c).lower() for c in rows[0]]
        sym_idx = [i for i, c in enumerate(header) if "symbol" in c]
        wt_idx = [i for i, c in enumerate(header) if "weight" in c]

        if not sym_idx or not wt_idx:
            continue

        si, wi = sym_idx[0], wt_idx[0]
        records = [
            (row[si], row[wi])
            for row in rows[1:]
            if len(row) > max(si, wi)
        ]

        if not records:
            continue

        out = pd.DataFrame(records, columns=["Symbol", "Raw_Weight"])

        out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
        out["Raw_Weight"] = pd.to_numeric(
            out["Raw_Weight"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        ) / 100.0

        out = out.dropna(subset=["Raw_Weight"])
        out = out[out["Raw_Weight"] > 0]
        out = out[out["Symbol"].str.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", na=False)]
        out = out[~out["Symbol"].isin({"NAN", "-", "--", "CASH", "USD", "N/A", "NA"})]

        total = out["Raw_Weight"].sum()

        # Sanity check: a real holdings table sums to roughly 100%.
        if len(out) >= 5 and 0.4 <= total <= 1.6:
            return out.reset_index(drop=True), asof

    return empty, None


# --- DATA ENGINE ---
class _DataUnavailable(Exception):
    """Raised inside cached fetchers so failures are NOT cached and retries work."""


def _with_retries(fetch_fn, tries=3, base_delay=2.0):
    """Retry wrapper: Yahoo rate-limits cloud IPs (HTTP 429), one retry usually clears it."""
    last_err = None

    for attempt in range(tries):
        try:
            result = fetch_fn()

            if result is not None:
                return result
        except Exception as err:
            last_err = err

        time.sleep(base_delay * (attempt + 1))

    if last_err is not None:
        return None

    return None


@st.cache_data(ttl=21600, show_spinner=False)
def _history_bundle_cached(symbol):
    """Max-history adjusted close (total return) + dividends. Cached 6h on success;
    raises on failure so empty results are never cached."""
    def fetch():
        tk = yf.Ticker(symbol)
        hist = tk.history(period="max", auto_adjust=True)

        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].dropna()

        if close.empty:
            return None

        try:
            div = tk.dividends
            if div is None:
                div = pd.Series(dtype=float)
        except Exception:
            div = pd.Series(dtype=float)

        return tz_naive(close), tz_naive(div)

    result = _with_retries(fetch)

    if result is None:
        raise _DataUnavailable(symbol)

    return result


def get_history_bundle(symbol):
    symbol = symbol.strip().upper()

    if not symbol:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        return _history_bundle_cached(symbol)
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)


@st.cache_data(ttl=86400, show_spinner=False)
def get_info(symbol):
    """Ticker metadata (yield, category). Non-critical, cached 24h."""
    def fetch():
        info = yf.Ticker(symbol).info
        return info if info else None

    return _with_retries(fetch, tries=2, base_delay=1.5) or {}


@st.cache_data(ttl=86400, show_spinner=False)
def get_expense_ratio(ticker):
    """Annual expense ratio as a fraction. Verified table first, then Yahoo's fund data.
    Returns None when unknown (shown as '-')."""
    ticker = ticker.strip().upper()

    if ticker in EXPENSE_RATIO:
        return EXPENSE_RATIO[ticker]

    def as_fraction(value):
        try:
            if value is None or pd.isna(value):
                return None

            value = float(value)

            # Yahoo sometimes reports 0.35 meaning 0.35%; no equity ETF charges 5%+.
            if value > 0.05:
                value = value / 100

            return value if 0 <= value <= 0.05 else None
        except Exception:
            return None

    try:
        ops = yf.Ticker(ticker).funds_data.fund_operations

        if ops is not None and "Annual Report Expense Ratio" in ops.index:
            row = ops.loc["Annual Report Expense Ratio"]
            value = row[ticker] if ticker in getattr(row, "index", []) else row.iloc[0]
            parsed = as_fraction(value)

            if parsed is not None:
                return parsed
    except Exception:
        pass

    try:
        info = get_info(ticker)

        for key in ("netExpenseRatio", "annualReportExpenseRatio", "expenseRatio"):
            parsed = as_fraction(info.get(key))

            if parsed is not None:
                return parsed
    except Exception:
        pass

    return None


@st.cache_data(ttl=21600, show_spinner=False)
def build_extended_close(ticker):
    """
    Splice the fund's adjusted close with older same-index funds / index data so
    returns reach back to (or toward) the index's inception, not just the ETF's.
    Returns (series, source_labels, has_price_only_era, has_proxy_era).
    """
    close, _ = get_history_bundle(ticker)

    if close.empty:
        return close, [], False, False

    sources = [f"{ticker} {close.index[0].year}→now"]
    has_price_only = False
    has_proxy = False
    ext = close

    for symbol, kind in EXTEND_CHAIN.get(ticker, []):
        proxy_close, _ = get_history_bundle(symbol)

        if proxy_close.empty:
            continue

        first = ext.index[0]
        prior = proxy_close[proxy_close.index < first]

        if prior.empty:
            continue

        anchor = proxy_close[proxy_close.index <= first]

        if anchor.empty:
            continue

        anchor_val = float(anchor.iloc[-1])

        # Refuse to splice across a data gap bigger than ~2 weeks.
        if anchor_val <= 0 or (first - anchor.index[-1]).days > 14:
            continue

        scale = float(ext.iloc[0]) / anchor_val
        ext = pd.concat([prior * scale, ext])
        sources.append(f"{symbol} {prior.index[0].year}–{first.year}")

        if kind == "pr_index":
            has_price_only = True
        else:
            has_proxy = True

    return ext, sources, has_price_only, has_proxy


def annual_dividends(symbol):
    _, div = get_history_bundle(symbol)

    if div.empty:
        return pd.Series(dtype=float)

    annual = div.groupby(div.index.year).sum()
    return annual[annual > 0]


@st.cache_data(ttl=21600, show_spinner=False)
def build_div_growth_factors(ticker):
    """
    Year -> dividend growth factor (div[y] / div[y-1]), preferring the fund's own
    history, then filling earlier years from same-index proxy funds in the chain.
    First partial year of each source is excluded. Returns (factors, proxy_symbols_used).
    """
    current_year = pd.Timestamp.now().year
    factors = {}
    proxies_used = []

    chain = [ticker] + [s for s, kind in EXTEND_CHAIN.get(ticker, []) if kind == "fund"]

    for symbol in chain:
        annual = annual_dividends(symbol)

        if len(annual) < 3:
            continue

        first_full = int(annual.index.min()) + 1
        used_here = False

        for year in range(first_full + 1, current_year):
            if year in factors:
                continue

            if year in annual.index and (year - 1) in annual.index and annual[year - 1] > 0:
                factors[year] = float(annual[year] / annual[year - 1])
                used_here = True

        if used_here and symbol != ticker:
            proxies_used.append(symbol)

    return pd.Series(factors).sort_index(), proxies_used


def div_cagr_from_factors(factors, years):
    """Geometric-mean dividend growth over the trailing `years` completed years."""
    if factors.empty:
        return None

    last_year = pd.Timestamp.now().year - 1
    needed = list(range(last_year - years + 1, last_year + 1))

    if not all(year in factors.index for year in needed):
        return None

    product = 1.0

    for year in needed:
        product *= factors[year]

    return product ** (1 / years) - 1 if product > 0 else None


def max_div_cagr(factors):
    """Dividend CAGR over the longest unbroken run of yearly data ending last year."""
    if factors.empty:
        return None, 0

    last_year = pd.Timestamp.now().year - 1

    if last_year not in factors.index:
        return None, 0

    run = 0
    year = last_year

    while year in factors.index:
        run += 1
        year -= 1

    product = 1.0

    for y in range(last_year - run + 1, last_year + 1):
        product *= factors[y]

    if run < 1 or product <= 0:
        return None, 0

    return product ** (1 / run) - 1, run


@st.cache_data(ttl=21600, show_spinner=False)
def get_full_stats(ticker):
    ticker = ticker.strip().upper()

    if not ticker:
        return None

    own_close, div = get_history_bundle(ticker)

    if own_close.empty:
        return None

    ext_close, sources, has_price_only, has_proxy = build_extended_close(ticker)

    if ext_close.empty:
        ext_close = own_close

    price = float(own_close.iloc[-1])

    if price <= 0:
        return None

    # A fund younger than a year has no full year of payouts yet. Borrow the dividend record
    # (and matching price) of the older same-index fund so yield, streak and frequency are real.
    div_price = price
    div_proxy = None
    own_days = (own_close.index[-1] - own_close.index[0]).days

    if own_days < 365:
        for symbol, kind in EXTEND_CHAIN.get(ticker, []):
            if kind != "fund":
                continue

            proxy_close, proxy_div = get_history_bundle(symbol)

            if not proxy_close.empty and not proxy_div.empty:
                div = proxy_div
                div_price = float(proxy_close.iloc[-1])
                div_proxy = symbol
                break

    if not div.empty and div_price > 0:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
        y_ttm = float(div[div.index >= cutoff].sum()) / div_price
    else:
        y_ttm = 0.0

    info = get_info(ticker)
    quote_type = str(info.get("quoteType", "")).upper()
    fwd_yield = normalize_ratio(info.get("dividendYield"), default=None)

    if "ETF" in quote_type or "FUND" in quote_type or fwd_yield is None:
        y_fwd = y_ttm
    else:
        y_fwd = fwd_yield

    industry = IND_MAP.get(ticker) or info.get("industry") or info.get("category") or "ETF/Fund"

    m = {
        "Ticker": ticker,
        "Price": price,
        "Industry": industry,
        "Inception": B_INCEPT.get(ticker, str(own_close.index[0].date())),
        "Hist From": str(ext_close.index[0].date()),
        "Yield (Fwd)": y_fwd,
        "Yield (TTM)": y_ttm,
        "Expense": get_expense_ratio(ticker),
    }

    # Dividend streak / frequency from the fund's own payments.
    if not div.empty:
        current_year = pd.Timestamp.now().year
        annual_div = div.groupby(div.index.year).sum()
        completed = annual_div[annual_div.index < current_year]
        m["Streak"] = dividend_growth_streak(completed)
        div_count = int(div[div.index.year == current_year - 1].count())
    else:
        m["Streak"] = 0
        div_count = 0

    m["Freq"] = "Mo" if div_count >= 11 else "Qr" if div_count >= 3 else "Yr" if div_count >= 1 else "-"

    # Short-term moves (extended series == own series for the recent end).
    m["1D"] = return_by_sessions(ext_close, 1)
    m["1W"] = return_by_sessions(ext_close, 5)
    m["1M"] = return_by_sessions(ext_close, 21)
    m["YTD"] = ytd_return(ext_close)

    # Long-term totals / CAGRs on the index-extended series.
    for years in TOTAL_YEARS:
        total = period_total_return(ext_close, years)
        m[f"{years}Y Total"] = total
        m[f"{years}Y CAGR"] = (
            ((1 + total) ** (1 / years)) - 1
            if years > 1 and total is not None and total > -1
            else None
        )

    # Since-index-inception ("Max") on the extended series.
    span_years = (ext_close.index[-1] - ext_close.index[0]).days / 365.25
    start_val = float(ext_close.iloc[0])

    if span_years >= 1 and start_val > 0:
        max_total = float(ext_close.iloc[-1]) / start_val - 1
        m["Max Total"] = max_total
        m["Max CAGR"] = (1 + max_total) ** (1 / span_years) - 1 if max_total > -1 else None
    else:
        m["Max Total"] = None
        m["Max CAGR"] = None

    # Worst peak-to-trough drop on the index-extended history, and when the bottom was.
    try:
        drawdown = ext_close / ext_close.cummax() - 1
        m["Max DD"] = float(drawdown.min())
        m["DD Date"] = str(drawdown.idxmin().date())[:7]
    except Exception:
        m["Max DD"] = None
        m["DD Date"] = "-"

    # Dividend growth, extended through same-index proxy funds where needed.
    factors, div_proxies = build_div_growth_factors(ticker)

    for years in DIV_YEARS:
        m[f"{years}Y Div CAGR"] = div_cagr_from_factors(factors, years)

    max_dcagr, div_run = max_div_cagr(factors)
    m["Max Div CAGR"] = max_dcagr

    # Transparency notes.
    notes = []

    if len(sources) > 1:
        notes.append("hist: " + " | ".join(sources[1:]))

    if has_price_only:
        notes.append("oldest era is price-only index (excl. dividends)")

    if div_proxies:
        notes.append(f"div growth pre-{ticker} via {'/'.join(div_proxies)}")

    if div_proxy:
        notes.append(f"yield/streak/freq via {div_proxy} (fund younger than 1 year)")

    if div_run:
        notes.append(f"max div span {div_run}y")

    m["Hist Notes"] = "; ".join(notes) if notes else "-"

    return m


@st.cache_data(ttl=86400, show_spinner=False)
def load_bundled_holdings():
    """Full-basket snapshots committed to the repo (holdings_static.json). Captured
    from Fidelity's public ETF research pages, which block cloud-server IPs - so the
    snapshots are bundled with the app and refreshed by committing a new file."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdings_static.json")

        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(ttl=43200, show_spinner=False)
def _holdings_cached(ticker):
    """Cached 12h on success; raises on failure so failures are retried, not cached.
    Order: bundled full-basket snapshot -> live Fidelity research -> Yahoo top-10."""
    bundled = load_bundled_holdings().get(ticker)

    if bundled and bundled.get("rows"):
        df = pd.DataFrame(bundled["rows"], columns=["Symbol", "Raw_Weight"])
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Raw_Weight"] = pd.to_numeric(df["Raw_Weight"], errors="coerce")
        df = df.dropna(subset=["Raw_Weight"])
        df = df[df["Raw_Weight"] > 0]

        if len(df) >= 5 and 0.4 <= df["Raw_Weight"].sum() <= 1.6:
            asof = bundled.get("asof", "n/a")
            return df.reset_index(drop=True), f"full basket snapshot, as of {asof}"

    full_df, asof = fetch_fidelity_holdings(ticker)

    if not full_df.empty:
        label = "full basket" + (f", as of {asof}" if asof else "")
        return full_df, label

    def fetch():
        raw = yf.Ticker(ticker).funds_data.top_holdings

        if raw is None:
            return None

        normalized = normalize_holdings_frame(raw)
        return normalized if not normalized.empty else None

    result = _with_retries(fetch, tries=2)

    if result is None:
        raise _DataUnavailable(ticker)

    return result, "top 10 only (Yahoo)"


def get_holdings(ticker):
    """
    Holdings with source label. Tries the full basket from Fidelity research first
    (every position, any issuer), then falls back to Yahoo's top-10 list.
    Returns (DataFrame[Symbol, Raw_Weight], source_label).
    """
    ticker = ticker.strip().upper()
    empty = pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    if not ticker:
        return empty, "none"

    try:
        own_df, own_source = _holdings_cached(ticker)
    except Exception:
        own_df, own_source = empty, "unavailable"

    if own_source.startswith("full basket"):
        return own_df, own_source

    # Only a top-10 list (or nothing) for this fund. If an older fund tracks the same index
    # and we have its full basket, that basket is a far better picture than 10 names.
    for symbol, kind in EXTEND_CHAIN.get(ticker, []):
        if kind != "fund":
            continue

        try:
            proxy_df, proxy_source = _holdings_cached(symbol)
        except Exception:
            continue

        if proxy_source.startswith("full basket") and not proxy_df.empty:
            return proxy_df, f"{symbol}'s {proxy_source} used as same-index stand-in"

    return own_df, own_source


def get_stats_for_tickers(tickers):
    data = []
    failed = []

    for ticker in tickers:
        stats = get_full_stats(ticker)

        if stats:
            data.append(stats)
        else:
            failed.append(ticker)

    if failed:
        st.warning(
            f"No usable market data for: {', '.join(failed)}. "
            "This is usually temporary Yahoo rate-limiting - wait ~1 minute and press the button again."
        )

    return data


def build_blended_holdings(etfs, weights):
    dfs = []
    source_counts = {}

    for ticker in etfs:
        weight = weights.get(ticker, 0)

        if weight <= 0:
            continue

        df, source = get_holdings(ticker)
        source_counts[ticker] = f"{len(df)} holdings ({source})"

        if not df.empty:
            df = df.copy()
            df["Weight"] = df["Raw_Weight"] * weight
            dfs.append(df[["Symbol", "Weight"]])

    if not dfs:
        return pd.DataFrame(columns=["Symbol", "Weight", "Weight %", "Industry"]), source_counts

    full = merge_goog(pd.concat(dfs, ignore_index=True))
    full["Weight %"] = (full["Weight"] * 100).round(2)
    full["Industry"] = full["Symbol"].apply(lambda x: IND_MAP.get(x, "Diversified / Other"))

    return full, source_counts


def calculate_blended_performance(weights):
    stats_by_ticker = {}

    for ticker, weight in weights.items():
        if weight <= 0:
            continue

        stats = get_full_stats(ticker)

        if stats:
            stats_by_ticker[ticker] = stats

    period_sources = {
        "1M": ("1M", None),
        "YTD": ("YTD", None),
        "1Y Total": ("1Y Total", None),
        "3Y CAGR": ("3Y Total", 3),
        "5Y CAGR": ("5Y Total", 5),
        "10Y CAGR": ("10Y Total", 10),
        "15Y CAGR": ("15Y Total", 15),
        "20Y CAGR": ("20Y Total", 20),
    }

    results = {}
    excluded = {}

    for label, (source_col, years) in period_sources.items():
        weighted_total = 0.0
        valid_weight = 0.0
        missing = []

        for ticker, stats in stats_by_ticker.items():
            value = stats.get(source_col)
            weight = weights.get(ticker, 0)

            if value is not None and pd.notnull(value):
                weighted_total += value * weight
                valid_weight += weight
            elif weight > 0:
                missing.append(ticker)

        excluded[label] = missing

        if valid_weight <= 0:
            results[label] = None
            continue

        total_return = weighted_total / valid_weight

        if years:
            results[label] = ((1 + total_return) ** (1 / years)) - 1 if total_return > -1 else None
        else:
            results[label] = total_return

    return results, excluded


def current_weights(tickers):
    """Weights for a ticker list as fractions summing to 1: the X-Ray percentage boxes if
    set this session, otherwise DEFAULT_WEIGHTS, otherwise equal weight."""
    tickers = [t for t in tickers if t]

    if not tickers:
        return {}

    equal = 100 / len(tickers)
    raw = {}

    for ticker in tickers:
        value = st.session_state.get(f"weight_{ticker}")

        if value is None:
            value = DEFAULT_WEIGHTS.get(ticker, equal)

        try:
            raw[ticker] = max(float(value), 0.0)
        except Exception:
            raw[ticker] = equal

    total = sum(raw.values())

    if total <= 0:
        return {t: 1 / len(tickers) for t in tickers}

    return {t: v / total for t, v in raw.items()}


def weighted_portfolio_row(stats_list, weights, label="MY PORTFOLIO"):
    """One table row for the blended portfolio. Yields, expense and N-year totals are
    weighted averages; each N-year CAGR is re-derived from the weighted total so the row
    matches the X-Ray blended numbers. Max DD is left blank (a blend's worst drop is not
    an average of the funds' worst drops)."""
    row = {
        "Ticker": label, "Price": None, "Industry": "-", "Inception": "-", "Hist From": "-",
        "Streak": None, "Freq": "-", "Max DD": None, "DD Date": "-", "Hist Notes": "-",
    }
    by_ticker = {s["Ticker"]: s for s in stats_list if s}

    def wavg(col):
        num = 0.0
        den = 0.0

        for ticker, stats in by_ticker.items():
            value = stats.get(col)
            weight = weights.get(ticker, 0)

            if value is not None and pd.notnull(value) and weight > 0:
                num += float(value) * weight
                den += weight

        return num / den if den > 0 else None

    for col in ALL_NUM_COLS + ["Expense"]:
        row[col] = wavg(col)

    for years in TOTAL_YEARS:
        if years > 1:
            total = row.get(f"{years}Y Total")
            row[f"{years}Y CAGR"] = (
                ((1 + total) ** (1 / years)) - 1
                if total is not None and total > -1
                else None
            )

    return row


def holdings_overlap(etfs):
    """(unique stocks, stocks held by 2+ funds, stocks held by every fund) across the ETFs."""
    sets = []

    for ticker in etfs:
        df, _ = get_holdings(ticker)

        if not df.empty:
            sets.append(set(df["Symbol"].astype(str).str.upper().replace({"GOOGL": "GOOG"})))

    if not sets:
        return 0, 0, 0

    counts = {}

    for symbols in sets:
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1

    n_two = sum(1 for c in counts.values() if c >= 2)
    n_all = sum(1 for c in counts.values() if c == len(sets))

    return len(counts), n_two, n_all


# --- LIVE NEWS ENGINE ---
def _parse_news_time(value):
    try:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)

        text = str(value).strip()

        # ISO style: 2026-09-04T12:00:00Z / 2026-09-04T12:00:00+00:00
        if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"

            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        # RSS style: Tue, 01 Sep 2026 22:40:00 GMT  (note: "Tue" contains a T - do not
        # route these through fromisoformat, that bug dropped Tue/Thu/Sat timestamps in v1)
        dt = parsedate_to_datetime(text)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _yahoo_news_for(ticker):
    items = []

    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        raw = []

    for entry in raw:
        try:
            content = entry.get("content", entry)
            title = content.get("title")

            if not title:
                continue

            link = None

            for key in ("canonicalUrl", "clickThroughUrl"):
                target = content.get(key)

                if isinstance(target, dict) and target.get("url"):
                    link = target["url"]
                    break

            if link is None:
                link = entry.get("link") or content.get("link")

            provider = content.get("provider")

            if isinstance(provider, dict):
                source = provider.get("displayName", "Yahoo Finance")
            else:
                source = entry.get("publisher", "Yahoo Finance")

            when = _parse_news_time(
                content.get("pubDate")
                or content.get("displayTime")
                or entry.get("providerPublishTime")
            )

            items.append({
                "Ticker": ticker,
                "Title": str(title).strip(),
                "Link": link,
                "Source": source,
                "Time": when,
            })
        except Exception:
            continue

    return items


def _google_news_for(query, label):
    items = []

    try:
        url = (
            "https://news.google.com/rss/search?q="
            + urllib.parse.quote(query)
            + "&hl=en-US&gl=US&ceid=US:en"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(request, timeout=10) as response:
            root = ET.fromstring(response.read())

        for item in root.iter("item"):
            title = item.findtext("title")

            if not title:
                continue

            source = item.find("source")

            items.append({
                "Ticker": label,
                "Title": title.strip(),
                "Link": item.findtext("link"),
                "Source": source.text if source is not None else "Google News",
                "Time": _parse_news_time(item.findtext("pubDate")),
            })

            if len(items) >= 8:
                break
    except Exception:
        pass

    return items


@st.cache_data(ttl=900, show_spinner=False)
def get_live_news(tickers_key):
    """Fresh headlines per ticker: Yahoo first, Google News RSS as backup. Cached 15 min."""
    rows = []

    for ticker in tickers_key:
        yahoo_items = _yahoo_news_for(ticker)

        if len(yahoo_items) < 3:
            yahoo_items += _google_news_for(f"{ticker} ETF stock", ticker)

        rows.extend(yahoo_items[:10])

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Title", "Link", "Source", "Time"]), datetime.now(timezone.utc)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Title"])
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df.sort_values("Time", ascending=False, na_position="last").reset_index(drop=True)

    return df, datetime.now(timezone.utc)


def relative_age(when):
    if when is None or pd.isnull(when):
        return ""

    delta = datetime.now(timezone.utc) - when.to_pydatetime()
    minutes = int(delta.total_seconds() // 60)

    if minutes < 1:
        return "just now"

    if minutes < 60:
        return f"{minutes}m ago"

    hours = minutes // 60

    if hours < 24:
        return f"{hours}h ago"

    days = hours // 24
    return f"{days}d ago"


@st.cache_data(ttl=900, show_spinner=False)
def get_pulse(tickers_key):
    """Fast 1-year batch download (single request) for the live snapshot table. Cached 15 min."""
    tickers = list(tickers_key)

    def fetch():
        data = yf.download(
            tickers=" ".join(tickers),
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        return data if data is not None and not data.empty else None

    data = _with_retries(fetch, tries=2)
    rows = []

    if data is None:
        return pd.DataFrame(), datetime.now(timezone.utc)

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[ticker]["Close"].dropna()
            else:
                close = data["Close"].dropna()

            close = tz_naive(close)

            if close.empty:
                continue

            last = float(close.iloc[-1])
            high_52w = float(close.max())
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

            rows.append({
                "Ticker": ticker,
                "Price": last,
                "1D": return_by_sessions(close, 1),
                "1W": return_by_sessions(close, 5),
                "1M": return_by_sessions(close, 21),
                "YTD": ytd_return(close),
                "vs 52W High": (last / high_52w) - 1 if high_52w > 0 else None,
                "200DMA": ("Above" if last >= ma200 else "Below") if ma200 else "-",
            })
        except Exception:
            continue

    return pd.DataFrame(rows), datetime.now(timezone.utc)


PULSE_PCT_COLS = ["1D", "1W", "1M", "YTD", "vs 52W High"]


def format_pulse(df):
    df = df.copy()

    for col in PULSE_PCT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").apply(
                lambda x: f"{x:.2%}" if pd.notnull(x) else "-"
            )

    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").apply(
            lambda x: f"${x:.2f}" if pd.notnull(x) else "-"
        )

    return df


def build_growth_frame(tickers, years):
    """Growth of $10,000 using index-extended history. years=None means max common history."""
    series_map = {}

    for ticker in tickers:
        ext_close, _, _, _ = build_extended_close(ticker)

        if not ext_close.empty:
            series_map[ticker] = ext_close

    if not series_map:
        return pd.DataFrame(), None, []

    if years is not None:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    else:
        cutoff = max(s.index[0] for s in series_map.values())

    frames = []
    start_used = None
    skipped = []

    for ticker, close in series_map.items():
        # A fund whose (extended) history starts after the window would begin its line
        # at $10,000 on a later date, which makes the chart lie. Skip it and say so.
        if close.index[0] > cutoff + pd.Timedelta(days=45):
            skipped.append(f"{ticker} (history starts {close.index[0].year})")
            continue

        window = close[close.index >= cutoff]

        if len(window) < 10:
            continue

        weekly = window.resample("W").last().dropna()

        if weekly.empty or float(weekly.iloc[0]) <= 0:
            continue

        growth = (weekly / float(weekly.iloc[0])) * 10000
        start_used = weekly.index[0] if start_used is None else min(start_used, weekly.index[0])

        frames.append(pd.DataFrame({
            "Date": weekly.index,
            "Growth of $10K": growth.values,
            "Ticker": ticker,
        }))

    if not frames:
        return pd.DataFrame(), None, skipped

    return pd.concat(frames, ignore_index=True), start_used, skipped


STATS_TABLE_COLS = (
    ["Ticker", "Price", "Industry", "Inception", "Hist From", "Streak", "Freq",
     "Expense", "Max DD", "DD Date"]
    + ALL_NUM_COLS + ["Hist Notes"]
)


# --- TAX PROFILES (general U.S. tax rules for 2026; education, not personal tax advice) ---
# The exact "qualified" share of any fund's payouts is published by its issuer every January and
# shows up on your own 1099-DIV (box 1b). The shares below are typical values used for estimates.
TAX_PROFILES = {
    "us_equity_index": {
        "name": "U.S. stock index ETF",
        "qualified_share": 0.95,
        "payouts": "Dividends, and essentially all of them are qualified (issuers report 95% to 100% each January).",
        "payout_tax": "Qualified dividends: 0%, 15% or 20% federal depending on your bracket, plus 3.8% only if income is above $200k single / $250k married.",
        "gains_payouts": "Capital-gains payouts are rare. ETFs hand appreciated stock to market makers instead of selling it, so they almost never pass gains to you.",
        "selling": "Shares held more than 1 year: long-term rate (0% / 15% / 20%). One year or less: your ordinary bracket. Each weekly buy is its own lot with its own one-year clock.",
        "best_home": "Works in both Roth and taxable. A low-yield growth index fund is about the most tax-efficient thing you can hold in a taxable account.",
        "upsides": ["Little annual tax because the yield is low", "The big tax bill is deferred until you sell, and you never have to sell", "Heirs receive a stepped-up basis"],
        "downsides": ["Large built-in gains make switching funds later expensive (lock-in)", "Dividends are taxed every year even when reinvested"],
        "form": "1099-DIV: box 1a total dividends, box 1b the qualified part (most of 1a), box 2a capital-gain payouts (usually 0).",
    },
    "us_dividend_equity": {
        "name": "U.S. dividend-focused stock ETF",
        "qualified_share": 0.95,
        "payouts": "Dividends, mostly qualified, but two to four times more of them per dollar than a growth index fund.",
        "payout_tax": "Qualified dividends: 0%, 15% or 20% federal depending on your bracket, plus 3.8% only if income is above $200k single / $250k married.",
        "gains_payouts": "Rare, same in-kind mechanism as any stock ETF.",
        "selling": "More than 1 year: long-term rate. One year or less: ordinary bracket.",
        "best_home": "Fine in either account. In taxable, the higher yield means a bigger yearly tax bill than a growth fund would create.",
        "upsides": ["Qualified treatment", "Steady cash if you ever want income"],
        "downsides": ["More tax every year than a low-yield fund", "Income you did not need still gets taxed"],
        "form": "1099-DIV boxes 1a and 1b.",
    },
    "international_equity": {
        "name": "International stock ETF",
        "qualified_share": 0.70,
        "payouts": "Dividends, roughly 60% to 80% qualified depending on the countries inside. Foreign governments withhold tax before you see the money.",
        "payout_tax": "Qualified part at 0/15/20%, the rest at your ordinary bracket. You can claim the foreign tax withheld as a credit (Form 1116, or directly if it is $300 or less single / $600 married).",
        "gains_payouts": "Rare.",
        "selling": "More than 1 year: long-term rate. One year or less: ordinary bracket.",
        "best_home": "Taxable is usually better than Roth: inside a Roth the foreign tax withheld is simply lost, in taxable you get it back as a credit.",
        "upsides": ["Foreign tax credit recovers most withholding", "Diversification away from U.S. tax law changes"],
        "downsides": ["Lower qualified share", "Extra tax-form work"],
        "form": "1099-DIV boxes 1a, 1b and box 7 (foreign tax paid).",
    },
    "covered_call": {
        "name": "Covered-call / option-income ETF",
        "qualified_share": 0.10,
        "payouts": "Big monthly payouts (often 8% to 12% a year) made mostly of option premium. For JEPI/JEPQ that is ordinary income; for QYLD-style funds much of it is return of capital, which is not taxed now but lowers your cost basis.",
        "payout_tax": "Mostly your ordinary bracket, not the qualified rate. Return-of-capital pieces are deferred until you sell.",
        "gains_payouts": "Possible; the option activity can create short-term gains.",
        "selling": "More than 1 year: long-term rate on the gain (a larger gain if return of capital lowered your basis).",
        "best_home": "Roth or IRA only, if at all. In a taxable account the yearly tax drag can exceed 3% of the position.",
        "upsides": ["High cash yield", "Lower volatility than the plain index"],
        "downsides": ["Caps the upside you are investing for", "Worst tax profile of any stock fund in taxable", "Return of capital hides the true tax bill until sale"],
        "form": "1099-DIV box 1a (mostly not in 1b), box 3 (nondividend distributions = return of capital).",
    },
    "reit": {
        "name": "Real estate (REIT) fund or stock",
        "qualified_share": 0.05,
        "payouts": "Large dividends that are NOT qualified. A portion is return of capital and a small piece can be capital gain.",
        "payout_tax": "Ordinary bracket, but with the 20% Section 199A deduction (made permanent in 2025), so effectively 80% of your ordinary rate.",
        "gains_payouts": "Some, from property sales inside the fund.",
        "selling": "More than 1 year: long-term rate.",
        "best_home": "Roth or IRA. In taxable, most of the payout is taxed at your working-income rate.",
        "upsides": ["199A deduction softens the blow", "Real income stream"],
        "downsides": ["No qualified treatment", "Return of capital lowers basis and grows the eventual gain"],
        "form": "1099-DIV box 1a, box 5 (Section 199A dividends), box 3 (return of capital), box 2a.",
    },
    "taxable_bond": {
        "name": "Bond ETF (corporate / total bond market)",
        "qualified_share": 0.0,
        "payouts": "Monthly interest. None of it is qualified.",
        "payout_tax": "Ordinary bracket, federal and state.",
        "gains_payouts": "Occasional small ones.",
        "selling": "Long-term or short-term rate on any price gain; bonds usually have little.",
        "best_home": "Roth or IRA. Interest at ordinary rates is the classic thing to shelter.",
        "upsides": ["Predictable cash", "Lower volatility"],
        "downsides": ["Ordinary rates on everything", "Fully taxed by states that have an income tax"],
        "form": "1099-DIV box 1a (interest from a bond ETF still arrives as an ordinary dividend), box 1b is 0.",
    },
    "treasury": {
        "name": "U.S. Treasury / T-bill ETF",
        "qualified_share": 0.0,
        "payouts": "Interest from U.S. government debt. Not qualified.",
        "payout_tax": "Ordinary federal bracket, but exempt from state income tax (the fund reports the government-interest share each January).",
        "gains_payouts": "Rare.",
        "selling": "Long-term or short-term rate on any price gain; T-bill funds have almost none.",
        "best_home": "Either. If your state rate is 0% anyway there is no state advantage to capture.",
        "upsides": ["State-tax exempt", "Safest cash-like holding"],
        "downsides": ["Ordinary federal rate on the interest"],
        "form": "1099-DIV box 1a; the issuer's January letter tells you the state-exempt percentage.",
    },
    "muni_bond": {
        "name": "Municipal bond ETF",
        "qualified_share": 0.0,
        "payouts": "Interest that is exempt from federal income tax. Exempt from state tax only for bonds from your own state.",
        "payout_tax": "0% federal. Your state may tax the out-of-state portion.",
        "gains_payouts": "Rare, and those are taxable.",
        "selling": "Price gains are taxed normally; only the interest is exempt.",
        "best_home": "Taxable only. Never inside a Roth or IRA (you would give up the exemption for nothing).",
        "upsides": ["Federal tax-free income"],
        "downsides": ["Lower yield than taxable bonds; only worth it in higher brackets", "Some interest can count for the alternative minimum tax"],
        "form": "1099-DIV box 12 (exempt-interest dividends).",
    },
    "gold_physical": {
        "name": "Physical gold / silver ETF (trust)",
        "qualified_share": 0.0,
        "payouts": "None. (Tiny amounts may be reported for fees the trust pays by selling metal.)",
        "payout_tax": "Nothing yearly.",
        "gains_payouts": "None.",
        "selling": "Taxed as a collectible: up to 28% federal on gains held more than 1 year (not the 15%/20% stock rate). One year or less: ordinary bracket.",
        "best_home": "Roth or IRA, which avoids the 28% collectibles rate entirely.",
        "upsides": ["No annual tax at all", "Simple 1099-B on sale"],
        "downsides": ["28% maximum long-term rate instead of 20%", "Yearly grantor-trust statement to keep for basis"],
        "form": "1099-B on sale plus the trust's annual tax letter.",
    },
    "commodity_futures": {
        "name": "Commodity futures ETF (partnership)",
        "qualified_share": 0.0,
        "payouts": "Usually none, but you are taxed anyway.",
        "payout_tax": "Futures are marked to market every December 31: 60% of the gain is long-term and 40% short-term whether or not you sold. Reported on a Schedule K-1 that arrives in March.",
        "gains_payouts": "Not applicable; gains flow through on the K-1.",
        "selling": "60/40 treatment again on sale.",
        "best_home": "Avoid in taxable unless you accept K-1 filing every year. In an IRA, large positions can trigger unrelated business income tax forms.",
        "upsides": ["60/40 rule is favorable versus pure short-term"],
        "downsides": ["K-1 paperwork, late arrival, sometimes state filings", "Taxed on paper gains you never received"],
        "form": "Schedule K-1 (Form 1065).",
    },
    "leveraged_inverse": {
        "name": "Leveraged or inverse ETF",
        "qualified_share": 0.50,
        "payouts": "Irregular; can include short-term gains from the daily rebalancing.",
        "payout_tax": "Mixed: any qualified part at 0/15/20%, short-term pieces at your ordinary bracket.",
        "gains_payouts": "More likely than a plain index fund.",
        "selling": "Normal long-term / short-term rules on the shares.",
        "best_home": "Roth or IRA if held at all; these funds reset daily and are designed for short holds.",
        "upsides": ["Standard share-sale treatment", "No K-1 for the common equity ones (TQQQ, SOXL, UPRO)"],
        "downsides": ["Daily reset makes multi-year results unpredictable (volatility decay)", "Payouts can be short-term gains"],
        "form": "1099-DIV boxes 1a, 1b, 2a.",
    },
    "us_stock": {
        "name": "Individual U.S. stock",
        "qualified_share": 0.95,
        "payouts": "Dividends from a U.S. corporation are qualified if you hold the shares more than 60 days in the 121-day window around each ex-dividend date. A never-sell holder meets that automatically.",
        "payout_tax": "Qualified: 0/15/20% federal, plus 3.8% only above the high-income line.",
        "gains_payouts": "Not applicable.",
        "selling": "More than 1 year: long-term rate. One year or less: ordinary bracket.",
        "best_home": "Either.",
        "upsides": ["Qualified dividends", "You control exactly when gains are realized"],
        "downsides": ["Single-company risk", "Special dividends and spin-offs can complicate basis"],
        "form": "1099-DIV boxes 1a and 1b; 1099-B on sale.",
    },
    "foreign_stock": {
        "name": "Foreign stock / ADR",
        "qualified_share": 0.80,
        "payouts": "Dividends, qualified only if the company's country has a tax treaty with the U.S. (or the stock trades on a U.S. exchange). Foreign tax is withheld first.",
        "payout_tax": "Qualified part at 0/15/20%; foreign withholding recoverable as a credit in a taxable account.",
        "gains_payouts": "Not applicable.",
        "selling": "Normal long-term / short-term rules.",
        "best_home": "Taxable, to keep the foreign tax credit.",
        "upsides": ["Credit for foreign tax", "Usually qualified"],
        "downsides": ["Withholding is lost inside a Roth", "Treaty details vary by country"],
        "form": "1099-DIV boxes 1a, 1b, 7.",
    },
    "mlp": {
        "name": "Master limited partnership (MLP)",
        "qualified_share": 0.0,
        "payouts": "Quarterly distributions that are mostly return of capital (not taxed now, lowers your basis).",
        "payout_tax": "Reported on a Schedule K-1; the taxable slice is ordinary income.",
        "gains_payouts": "Not applicable.",
        "selling": "Part of the gain is recaptured as ordinary income.",
        "best_home": "Taxable. Inside an IRA, more than $1,000 of partnership income triggers a separate tax return for the account.",
        "upsides": ["Tax-deferred cash flow"],
        "downsides": ["K-1 every year", "Ordinary-income recapture on sale"],
        "form": "Schedule K-1.",
    },
}

# Known funds. Anything not listed is classified from Yahoo's category text (see tax_profile_key).
TAX_OVERRIDES = {
    "SPMO": "us_equity_index", "QNDX": "us_equity_index", "QQQ": "us_equity_index", "QQQM": "us_equity_index",
    "FTEC": "us_equity_index", "VGT": "us_equity_index", "SMH": "us_equity_index", "SOXX": "us_equity_index",
    "XLK": "us_equity_index", "IYW": "us_equity_index", "VOO": "us_equity_index", "SPY": "us_equity_index",
    "IVV": "us_equity_index", "SPLG": "us_equity_index", "VTI": "us_equity_index", "ITOT": "us_equity_index",
    "SCHG": "us_equity_index", "VUG": "us_equity_index", "MGK": "us_equity_index", "IWF": "us_equity_index",
    "SCHD": "us_dividend_equity", "VYM": "us_dividend_equity", "VIG": "us_dividend_equity", "DGRO": "us_dividend_equity",
    "JEPQ": "covered_call", "JEPI": "covered_call", "QYLD": "covered_call", "XYLD": "covered_call",
    "VNQ": "reit", "SCHH": "reit", "XLRE": "reit", "IYR": "reit",
    "BND": "taxable_bond", "AGG": "taxable_bond", "LQD": "taxable_bond", "BNDX": "taxable_bond",
    "TLT": "treasury", "IEF": "treasury", "SHY": "treasury", "SGOV": "treasury", "BIL": "treasury", "VGSH": "treasury",
    "MUB": "muni_bond", "VTEB": "muni_bond",
    "GLD": "gold_physical", "IAU": "gold_physical", "GLDM": "gold_physical", "SGOL": "gold_physical", "SLV": "gold_physical",
    "USO": "commodity_futures", "DBC": "commodity_futures", "UNG": "commodity_futures",
    "TQQQ": "leveraged_inverse", "SOXL": "leveraged_inverse", "UPRO": "leveraged_inverse", "SQQQ": "leveraged_inverse",
    "VXUS": "international_equity", "IXUS": "international_equity", "VEA": "international_equity",
    "VWO": "international_equity", "EFA": "international_equity", "EEM": "international_equity",
    "IXN": "international_equity", "VT": "international_equity", "ACWI": "international_equity",
}

INTERNATIONAL_WORDS = ["foreign", "world", "emerging", "global", "europe", "pacific", "china", "japan", "india", "international", "ex-us", "ex us"]


@st.cache_data(ttl=86400, show_spinner=False)
def tax_profile_key(ticker):
    """Which tax profile fits this ticker. Returns (profile_key, how_it_was_decided)."""
    ticker = ticker.strip().upper()

    if ticker in TAX_OVERRIDES:
        return TAX_OVERRIDES[ticker], "known fund"

    info = get_info(ticker)
    quote_type = str(info.get("quoteType", "")).upper()
    category = str(info.get("category") or "").lower()
    name = str(info.get("longName") or info.get("shortName") or "").lower()
    text = category + " " + name

    if quote_type == "EQUITY":
        industry = str(info.get("industry") or "").lower()
        country = str(info.get("country") or "").lower()

        if "reit" in industry or "reit" in name:
            return "reit", "individual REIT"

        if "midstream" in industry and (" lp" in name or "l.p." in name or "partners" in name):
            return "mlp", "partnership"

        if country and country != "united states":
            return "foreign_stock", f"company based in {info.get('country')}"

        return "us_stock", "U.S. company"

    if "derivative income" in category or "covered call" in text or "buywrite" in text or "premium income" in text:
        return "covered_call", f"Yahoo category: {category or 'n/a'}"

    if "real estate" in category or "reit" in text:
        return "reit", f"Yahoo category: {category or 'n/a'}"

    if "muni" in text:
        return "muni_bond", f"Yahoo category: {category or 'n/a'}"

    if "treasury" in text or "government" in category or "t-bill" in text:
        return "treasury", f"Yahoo category: {category or 'n/a'}"

    if "bond" in category or "fixed income" in text or "core-plus" in category:
        return "taxable_bond", f"Yahoo category: {category or 'n/a'}"

    if "leveraged" in category or "inverse" in category or "trading--" in category:
        return "leveraged_inverse", f"Yahoo category: {category or 'n/a'}"

    if "commodit" in category or "commodit" in name:
        if any(word in text for word in ["gold", "silver", "precious", "platinum"]):
            return "gold_physical", f"Yahoo category: {category or 'n/a'}"

        return "commodity_futures", f"Yahoo category: {category or 'n/a'}"

    if any(word in category for word in INTERNATIONAL_WORDS):
        return "international_equity", f"Yahoo category: {category or 'n/a'}"

    if "dividend" in text or "high yield" in text or "income" in category:
        return "us_dividend_equity", f"Yahoo category: {category or 'n/a'}"

    return "us_equity_index", f"Yahoo category: {category or 'n/a'}"


def tax_rates(bracket_pct, high_income, state_pct):
    """Federal + state rates as fractions. Qualified/long-term: 0% in the 10-12% brackets,
    15% in 22-35%, 20% in 37%. The 3.8% net investment income tax applies above $200k single /
    $250k married."""
    qualified = 0.0 if bracket_pct <= 12 else (0.20 if bracket_pct >= 37 else 0.15)
    niit = 0.038 if high_income else 0.0
    state = max(float(state_pct), 0.0) / 100

    return {
        "ordinary": bracket_pct / 100 + niit + state,
        "qualified": qualified + niit + state,
        "ordinary_fed": bracket_pct / 100 + niit,
        "qualified_fed": qualified + niit,
        "state": state,
    }


def payout_tax_rate(profile_key, rates):
    """Effective tax rate on one dollar of payouts from this kind of holding."""
    profile = TAX_PROFILES[profile_key]
    q = profile["qualified_share"]

    if profile_key == "muni_bond":
        return rates["state"]

    if profile_key == "treasury":
        return rates["ordinary_fed"]

    if profile_key == "gold_physical":
        return 0.0

    if profile_key == "reit":
        return q * rates["qualified"] + (1 - q) * (rates["ordinary_fed"] * 0.8 + rates["state"])

    return q * rates["qualified"] + (1 - q) * rates["ordinary"]


def blended_tax_estimate(tickers, weights, value, taxable_share, rates):
    """Per-fund and total yearly tax on payouts. Only the taxable-account share is taxed."""
    rows = []
    total_payout = 0.0
    total_tax = 0.0

    for ticker in tickers:
        weight = weights.get(ticker, 0)

        if weight <= 0:
            continue

        stats = get_full_stats(ticker)
        yield_ttm = (stats or {}).get("Yield (TTM)") or 0.0
        key, _ = tax_profile_key(ticker)
        rate = payout_tax_rate(key, rates)
        payout = value * weight * yield_ttm
        tax = payout * taxable_share * rate

        total_payout += payout
        total_tax += tax

        rows.append({
            "Fund": ticker,
            "Weight": f"{weight:.0%}",
            "Yield (TTM)": f"{yield_ttm:.2%}",
            "Payout type": TAX_PROFILES[key]["name"],
            "Payouts / yr": f"${payout:,.0f}",
            "Tax rate on payouts": f"{rate:.1%}",
            "Tax / yr (taxable part)": f"${tax:,.0f}",
        })

    return rows, total_payout, total_tax


def holdings_overlap_pct(df_a, df_b):
    """Share of the two funds that is the same stocks (sum of the smaller weight per shared stock)."""
    if df_a.empty or df_b.empty:
        return None

    a = merge_goog(df_a.rename(columns={"Raw_Weight": "Weight"}))
    b = merge_goog(df_b.rename(columns={"Raw_Weight": "Weight"}))
    a_w = dict(zip(a["Symbol"], a["Weight"] / a["Weight"].sum()))
    b_w = dict(zip(b["Symbol"], b["Weight"] / b["Weight"].sum()))

    return sum(min(a_w[s], b_w[s]) for s in a_w if s in b_w)


def describe_fund(ticker, info):
    if ticker in INDEX_NOTE:
        return INDEX_NOTE[ticker].split(". ")[0] + "."

    return IND_MAP.get(ticker) or info.get("category") or info.get("longName") or "-"


def compare_sections(tickers):
    """Side-by-side tables (metrics as rows, funds as columns) plus computed one-line verdicts."""
    stats = {}
    holdings = {}

    for ticker in tickers:
        s = get_full_stats(ticker)

        if s:
            stats[ticker] = s
            holdings[ticker] = get_holdings(ticker)

    if not stats:
        return None, [], []

    def pct(v):
        return pct_or_na(v) if v is not None and pd.notnull(v) else "-"

    def top3(ticker):
        df, _ = holdings[ticker]

        if df.empty:
            return "-"

        top = merge_goog(df.rename(columns={"Raw_Weight": "Weight"})).head(3)
        return ", ".join(f"{r.Symbol} {r.Weight:.1%}" for r in top.itertuples())

    def top10(ticker):
        df, _ = holdings[ticker]

        if df.empty:
            return "-"

        top = merge_goog(df.rename(columns={"Raw_Weight": "Weight"})).head(10)
        return f"{top['Weight'].sum():.1%}"

    cols = list(stats.keys())
    sections = []

    what = {
        "What it holds": {t: describe_fund(t, get_info(t)) for t in cols},
        "Fund since": {t: stats[t]["Inception"] for t in cols},
        "History used from": {t: stats[t]["Hist From"] for t in cols},
        "Cost per year": {t: pct(stats[t].get("Expense")) for t in cols},
        "Number of holdings": {t: (len(holdings[t][0]) if not holdings[t][0].empty else "-") for t in cols},
        "Top 3 holdings": {t: top3(t) for t in cols},
        "Top 10 share": {t: top10(t) for t in cols},
        "Price": {t: f"${stats[t]['Price']:.2f}" for t in cols},
    }
    growth = {
        "1 year": {t: pct(stats[t].get("1Y Total")) for t in cols},
        "3 years (per year)": {t: pct(stats[t].get("3Y CAGR")) for t in cols},
        "5 years (per year)": {t: pct(stats[t].get("5Y CAGR")) for t in cols},
        "10 years (per year)": {t: pct(stats[t].get("10Y CAGR")) for t in cols},
        "15 years (per year)": {t: pct(stats[t].get("15Y CAGR")) for t in cols},
        "20 years (per year)": {t: pct(stats[t].get("20Y CAGR")) for t in cols},
        "Since data start (per year)": {t: pct(stats[t].get("Max CAGR")) for t in cols},
    }
    risk = {
        "Worst drop ever": {t: pct(stats[t].get("Max DD")) for t in cols},
        "Bottom was hit": {t: stats[t].get("DD Date", "-") for t in cols},
        "This year so far": {t: pct(stats[t].get("YTD")) for t in cols},
        "Last month": {t: pct(stats[t].get("1M")) for t in cols},
    }
    income = {
        "Dividend yield (last 12 months)": {t: pct(stats[t].get("Yield (TTM)")) for t in cols},
        "Pays": {t: {"Qr": "Quarterly", "Mo": "Monthly", "Yr": "Yearly"}.get(stats[t].get("Freq"), "-") for t in cols},
        "Years of rising payouts": {t: stats[t].get("Streak", "-") for t in cols},
        "Payout growth, 5 yrs (per year)": {t: pct(stats[t].get("5Y Div CAGR")) for t in cols},
        "Payout growth, 10 yrs (per year)": {t: pct(stats[t].get("10Y Div CAGR")) for t in cols},
    }
    tax = {}

    for t in cols:
        key, _ = tax_profile_key(t)
        tax.setdefault("Payout type", {})[t] = TAX_PROFILES[key]["name"]
        tax.setdefault("Typical qualified share", {})[t] = f"{TAX_PROFILES[key]['qualified_share']:.0%}"
        tax.setdefault("Best home", {})[t] = TAX_PROFILES[key]["best_home"].split(". ")[0] + "."

    for title, block in [("What it is", what), ("Growth", growth), ("Risk", risk), ("Income", income), ("Taxes", tax)]:
        df = pd.DataFrame(block).T.astype(str)   # all text: mixed numbers/strings break the table renderer
        df.index.name = "Metric"
        sections.append((title, df.reset_index()))

    # Pairwise overlap
    overlaps = []

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            share = holdings_overlap_pct(holdings[a][0], holdings[b][0])

            if share is not None:
                overlaps.append({"Pair": f"{a} vs {b}", "Same stocks by weight": f"{share:.0%}"})

    # Verdicts
    verdicts = []
    exp = {t: stats[t].get("Expense") for t in cols if stats[t].get("Expense") is not None}

    if len(exp) >= 2:
        cheapest = min(exp, key=exp.get)
        verdicts.append(f"Cheapest: {cheapest} at {exp[cheapest]:.2%} a year.")

    for years in [20, 15, 10, 5, 3]:
        col = f"{years}Y CAGR"
        vals = {t: stats[t].get(col) for t in cols}

        if all(v is not None and pd.notnull(v) for v in vals.values()) and len(vals) >= 2:
            best = max(vals, key=vals.get)
            worst = min(vals, key=vals.get)
            verdicts.append(
                f"Fastest growth over the longest period all of them share ({years} years): {best} at "
                f"{vals[best]:.1%} a year; slowest {worst} at {vals[worst]:.1%}."
            )
            break

    dd = {t: stats[t].get("Max DD") for t in cols if stats[t].get("Max DD") is not None}

    if len(dd) >= 2:
        roughest = min(dd, key=dd.get)
        smoothest = max(dd, key=dd.get)
        verdicts.append(
            f"Roughest ride: {roughest} once fell {abs(dd[roughest]):.0%} (bottom {stats[roughest].get('DD Date')}); "
            f"smoothest: {smoothest} at {abs(dd[smoothest]):.0%}."
        )

    yl = {t: stats[t].get("Yield (TTM)") for t in cols if stats[t].get("Yield (TTM)") is not None}

    if len(yl) >= 2:
        highest = max(yl, key=yl.get)
        verdicts.append(f"Most income: {highest} yields {yl[highest]:.2%}; the rest pay less and are taxed less each year.")

    if overlaps:
        top_pair = max(overlaps, key=lambda o: float(o["Same stocks by weight"].rstrip("%")))
        verdicts.append(f"Most overlap: {top_pair['Pair']} are {top_pair['Same stocks by weight']} the same stocks by weight.")

    return sections, overlaps, verdicts


# --- SHOWS & VOICES (podcast RSS feeds + YouTube channels) ---
# To add a show: tell Claude the show name or paste its YouTube link and you get one line to paste here.
#   kind "rss"     = a podcast or news feed address
#   kind "youtube" = a YouTube channel handle (the @name from the channel's page)
SHOW_FEEDS = [
    ("CNBC Top News", "rss", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Television (YouTube)", "youtube", "@CNBCtelevision"),
    ("The Compound and Friends (podcast)", "rss", "https://feeds.megaphone.fm/TCP4771071679"),
    ("The Compound (YouTube)", "youtube", "@TheCompoundNews"),
    ("The Real Eisman Playbook (podcast)", "rss", "https://feed.podbean.com/realeismanplaybook/feed.xml"),
    ("The Real Eisman Playbook (YouTube)", "youtube", "@RealEismanPlaybook"),
    ("FINAiUS (YouTube)", "youtube", "@FINAiUS"),
]

# If a direct feed is blocked from the cloud server, use Google News for that outlet instead.
SHOW_FALLBACK_QUERY = {
    "CNBC Top News": "site:cnbc.com",
}

ATOM_NS = "{http://www.w3.org/2005/Atom}"
YT_NS = "{http://www.youtube.com/xml/schemas/2015}"


def _http_get_bytes(url, timeout=15):
    """Raw bytes with browser impersonation first (curl_cffi ships with yfinance), urllib second."""
    try:
        from curl_cffi import requests as curl_requests

        response = curl_requests.get(url, impersonate="chrome", timeout=timeout)

        if getattr(response, "status_code", 0) == 200 and response.content:
            return response.content
    except Exception:
        pass

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _as_utc(when):
    if when is None:
        return None

    try:
        return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_youtube_channel_id(handle):
    """'@SomeChannel' -> 'UC...' channel id by reading the channel page once a day.
    Raises on failure so a miss is retried next time instead of being cached."""
    handle = str(handle).strip()

    if handle.startswith("UC") and len(handle) == 24:
        return handle

    if not handle.startswith("@"):
        handle = "@" + handle

    html = _http_get_bytes(f"https://www.youtube.com/{handle}").decode("utf-8", errors="replace")

    for pattern in (
        r'"externalId":"(UC[0-9A-Za-z_-]{22})"',
        r'youtube\.com/channel/(UC[0-9A-Za-z_-]{22})',
        r'"channelId":"(UC[0-9A-Za-z_-]{22})"',
    ):
        match = re.search(pattern, html)

        if match:
            return match.group(1)

    raise _DataUnavailable(handle)


def _parse_feed_items(xml_bytes, show):
    """Read RSS 2.0 (podcasts, news) or Atom (YouTube) into simple rows."""
    root = ET.fromstring(xml_bytes)
    items = []

    # RSS 2.0: <rss><channel><item>...
    channel_link = root.findtext("channel/link")

    for item in root.iter("item"):
        title = item.findtext("title")

        if not title:
            continue

        link = item.findtext("link")

        if not link:
            enclosure = item.find("enclosure")
            link = enclosure.get("url") if enclosure is not None else channel_link

        items.append({
            "Show": show,
            "Title": " ".join(title.split()),
            "Link": link,
            "Time": _as_utc(_parse_news_time(item.findtext("pubDate"))),
            "VideoId": None,
        })

    # Atom: <feed><entry>... (YouTube channel feeds)
    for entry in root.iter(ATOM_NS + "entry"):
        title = entry.findtext(ATOM_NS + "title")

        if not title:
            continue

        link_el = entry.find(ATOM_NS + "link")
        link = link_el.get("href") if link_el is not None else None
        video_id = entry.findtext(YT_NS + "videoId")

        if not link and video_id:
            link = f"https://www.youtube.com/watch?v={video_id}"

        items.append({
            "Show": show,
            "Title": " ".join(title.split()),
            "Link": link,
            "Time": _as_utc(_parse_news_time(entry.findtext(ATOM_NS + "published"))),
            "VideoId": video_id,
        })

    return items


@st.cache_data(ttl=1800, show_spinner=False)
def get_show_items(feeds_key, per_show=8):
    """Latest episodes/videos per show. Cached 30 min. Returns (DataFrame, status_by_show, fetched_at)."""
    rows = []
    status = {}

    for show, kind, target in feeds_key:
        try:
            if kind == "youtube":
                channel_id = resolve_youtube_channel_id(target)
                url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            else:
                url = target

            items = _parse_feed_items(_http_get_bytes(url), show)

            if not items:
                raise _DataUnavailable(show)

            items.sort(key=lambda r: r["Time"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
            rows.extend(items[:per_show])
            status[show] = f"{min(len(items), per_show)} new"
            continue
        except Exception:
            pass

        query = SHOW_FALLBACK_QUERY.get(show)
        fallback = _google_news_for(query, show) if query else []

        if fallback:
            for entry in fallback[:per_show]:
                rows.append({
                    "Show": show,
                    "Title": entry["Title"],
                    "Link": entry["Link"],
                    "Time": _as_utc(entry["Time"]),
                    "VideoId": None,
                })

            status[show] = f"{min(len(fallback), per_show)} via Google News"
        else:
            status[show] = "unavailable right now"

    columns = ["Show", "Title", "Link", "Time", "VideoId"]

    if not rows:
        return pd.DataFrame(columns=columns), status, datetime.now(timezone.utc)

    df = pd.DataFrame(rows, columns=columns)
    df = df.drop_duplicates(subset=["Title"])
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df.sort_values("Time", ascending=False, na_position="last").reset_index(drop=True)

    return df, status, datetime.now(timezone.utc)


def show_chart(fig):
    """Full-width chart on both new and old Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# --- UI ---
def main():
    tab1, tab2, tab3, tab4, tab5, tab_cmp, tab_tax, tab6, tab7 = st.tabs([
        "🚀 X-Ray",
        "🆚 Benchmark",
        "📈 Dividends",
        "🔍 Deep Dive",
        "👀 Watchlist",
        "⚖️ Compare",
        "🧾 Taxes",
        "📰 Insights & Updates",
        "🎧 Shows & Voices",
    ])

    # --- TAB 1: X-RAY ---
    with tab1:
        st.header("Portfolio X-Ray")

        etfs = parse_tickers(st.text_input("ETFs to Blend:", DEFAULT_PORT, key="xray_tickers"))

        if not etfs:
            st.info("Enter at least one ETF ticker.")
        else:
            cols = st.columns(len(etfs))
            raw_weights = {}
            default_weight = int(round(100 / len(etfs)))

            for i, ticker in enumerate(etfs):
                with cols[i]:
                    raw_weights[ticker] = st.number_input(
                        f"{ticker} %",
                        min_value=0,
                        max_value=100,
                        value=int(DEFAULT_WEIGHTS.get(ticker, default_weight)),
                        step=5,
                        key=f"weight_{ticker}"
                    ) / 100.0

            total_w = sum(raw_weights.values())

            if total_w > 0:
                weights = {ticker: weight / total_w for ticker, weight in raw_weights.items()}

                if abs(total_w - 1.0) > 0.005:
                    st.caption(f"Percentages sum to {total_w * 100:.0f}% - normalized to 100% for the math below.")
            else:
                weights = raw_weights
                st.warning("Set at least one ETF weight above 0.")

            portfolio_value = st.number_input(
                "Portfolio value in $ (optional - adds a dollars-per-stock column):",
                min_value=0,
                value=0,
                step=1000,
                key="xray_value"
            )

            if st.button("Analyze Blended Holdings", key="analyze_blended"):
                st.session_state["xray_go"] = True

            if st.session_state.get("xray_go") and total_w > 0:
                st.subheader("📈 Blended Performance")
                st.caption(
                    "Long-period figures use index-extended history (older same-index funds / "
                    "indexes spliced in before each ETF's inception)."
                )

                blended_stats, excluded = calculate_blended_performance(weights)
                metric_cols = st.columns(len(PERF_PERIODS))

                for i, period in enumerate(PERF_PERIODS):
                    metric_cols[i].metric(period, pct_or_na(blended_stats.get(period)))

                notes = []

                for period, missing in excluded.items():
                    if missing:
                        notes.append(f"{period} excludes {', '.join(missing)}")

                if notes:
                    st.caption(
                        "Where a fund is younger than the window it is left out and the figure is "
                        "re-weighted across the rest: " + "; ".join(notes) + "."
                    )

                expense_bits = []
                blended_expense = 0.0
                expense_weight = 0.0

                for ticker, weight in weights.items():
                    ratio = get_expense_ratio(ticker)

                    if ratio is not None:
                        blended_expense += ratio * weight
                        expense_weight += weight
                        expense_bits.append(f"{ticker} {ratio:.2%}")

                if expense_weight > 0:
                    st.caption(
                        f"Blended expense ratio: **{blended_expense / expense_weight:.2%}** per year "
                        f"({' | '.join(expense_bits)})."
                    )

                st.markdown("---")

                full, source_counts = build_blended_holdings(etfs, weights)

                if full.empty:
                    st.warning("Could not load holdings data for the selected ETFs.")
                    st.caption(
                        "Both holdings sources are unreachable right now. "
                        "Wait ~1 minute and press the button again - results are cached once loaded."
                    )
                else:
                    n_unique, n_two, n_all = holdings_overlap(etfs)
                    top10_weight = full.head(10)["Weight"].sum()

                    overlap_text = (
                        f" {n_two} stocks sit in two or more of your funds and {n_all} sit in all {len(etfs)}."
                        if len(etfs) > 1 else ""
                    )
                    st.caption(
                        f"Showing all {len(full)} blended underlying positions, weighted by your percentages. "
                        f"Top 10 stocks = {top10_weight:.1%} of the portfolio." + overlap_text
                    )
                    st.caption(
                        "Sources - "
                        + " | ".join(f"{ticker}: {info}" for ticker, info in source_counts.items())
                    )

                    table = full[["Symbol", "Industry", "Weight %"]].copy()

                    if portfolio_value and portfolio_value > 0:
                        table["$ Exposure"] = (full["Weight"] * float(portfolio_value)).apply(
                            lambda x: f"${x:,.0f}"
                        )

                    c1, c2 = st.columns([2, 1])

                    with c1:
                        fig = px.treemap(
                            full.head(40),
                            path=[px.Constant("Portfolio"), "Symbol"],
                            values="Weight %",
                            custom_data=["Industry"],
                            title="Top Holdings"
                        )
                        fig.update_traces(
                            textinfo="label+value",
                            texttemplate="%{label}<br>%{customdata[0]}<br>%{value:.2f}%"
                        )
                        show_chart(fig)

                    with c2:
                        st.dataframe(table, height=500, hide_index=True)

    # --- TAB 2: BENCHMARK ---
    with tab2:
        st.header("Portfolio vs. Benchmark")

        cp, cm = st.columns(2)

        with cp:
            p_list = parse_tickers(st.text_input("Your Portfolio (Editable):", DEFAULT_PORT, key="bench_port"))

        with cm:
            m_list = parse_tickers(st.text_input("Benchmark(s):", DEFAULT_BENCH, key="bench_market"))

        chart_choice = st.selectbox(
            "Growth chart window:",
            ["10Y", "15Y", "20Y", "25Y", "Max common history"],
            index=1,
            key="bench_window"
        )
        log_scale = st.checkbox("Log scale (recommended for 15Y+)", value=True, key="bench_log")

        if st.button("Compare", key="compare"):
            st.session_state["bench_go"] = True

        if st.session_state.get("bench_go"):
            if not p_list or not m_list:
                st.info("Enter at least one portfolio ticker and one benchmark ticker.")
            else:
                p_stats = get_stats_for_tickers(p_list)
                m_stats = get_stats_for_tickers(m_list)

                if p_stats and m_stats:
                    bench_weights = current_weights(p_list)
                    portfolio_row = weighted_portfolio_row(p_stats, bench_weights, label="MY PORTFOLIO")

                    final = pd.DataFrame([portfolio_row] + m_stats)
                    final_cols = (
                        ["Ticker", "Inception", "Hist From", "Expense", "Max DD", "DD Date"]
                        + [col for col in ALL_NUM_COLS if col in final.columns]
                        + (["Hist Notes"] if "Hist Notes" in final.columns else [])
                    )
                    final = final[[col for col in final_cols if col in final.columns]]
                    final = final.dropna(axis=1, how="all")

                    st.dataframe(format_dataframe(final), hide_index=True)
                    st.caption(
                        "MY PORTFOLIO is weighted by your X-Ray percentages ("
                        + ", ".join(f"{t} {w:.0%}" for t, w in bench_weights.items())
                        + "). Each N-year CAGR is computed from the blended N-year total return, "
                        "so it matches the X-Ray tab."
                    )

                    st.subheader("💰 Growth of $10,000")

                    years_map = {"10Y": 10, "15Y": 15, "20Y": 20, "25Y": 25, "Max common history": None}
                    growth_df, start_used, skipped = build_growth_frame(
                        list(dict.fromkeys(p_list + m_list)),
                        years_map[chart_choice]
                    )

                    if growth_df.empty:
                        st.info("Not enough overlapping history for the selected window.")
                    else:
                        fig = px.line(
                            growth_df,
                            x="Date",
                            y="Growth of $10K",
                            color="Ticker",
                            title=f"Growth of $10,000 - {chart_choice} (index-extended history)"
                        )

                        if log_scale:
                            fig.update_yaxes(type="log")

                        show_chart(fig)

                        if start_used is not None:
                            st.caption(
                                f"Chart starts {start_used.date()}. Pre-inception eras use spliced "
                                "same-index fund / index data; price-only index eras exclude dividends."
                            )

                    if skipped:
                        st.caption(
                            "Left off this chart because their history starts after the window: "
                            + ", ".join(skipped) + ". Pick a shorter window to include them."
                        )

    # --- TAB 3: DIVIDENDS ---
    with tab3:
        st.header("Dividend & Growth Data")

        t_list = parse_tickers(st.text_area("Tickers", DEFAULT_PORT, key="div_tickers"))

        if st.button("Load Dividends", key="load_dividends"):
            st.session_state["div_go"] = True

        if st.session_state.get("div_go"):
            if not t_list:
                st.info("Enter at least one ticker.")
            else:
                data = get_stats_for_tickers(t_list)

                if data:
                    div_weights = current_weights(t_list)
                    portfolio_row = weighted_portfolio_row(data, div_weights, label="MY PORTFOLIO")

                    final = pd.concat([pd.DataFrame(data), pd.DataFrame([portfolio_row])], ignore_index=True)
                    final = final[[col for col in STATS_TABLE_COLS if col in final.columns]]

                    st.dataframe(format_dataframe(final), height=500, hide_index=True)
                    st.caption(
                        "Div CAGR uses completed calendar years. Where an ETF is younger than the "
                        "window, growth is chained from older same-index funds (see Hist Notes). "
                        "Max Div CAGR covers the longest unbroken yearly run available. "
                        "MY PORTFOLIO is weighted by your X-Ray percentages."
                    )

    # --- TAB 4: DEEP DIVE ---
    with tab4:
        st.header("Multi-ETF Deep Dive")

        deep_tickers = parse_tickers(st.text_input("Tickers:", DEFAULT_PORT, key="deep_dive_tickers"))

        if st.button("Inspect ETFs", key="inspect_etfs"):
            st.session_state["deep_go"] = True

        if st.session_state.get("deep_go"):
            if not deep_tickers:
                st.info("Enter at least one ETF ticker.")
            else:
                for ticker in deep_tickers:
                    stats = get_full_stats(ticker)
                    hist_from = stats.get("Hist From", "N/A") if stats else "N/A"
                    inception = stats.get("Inception", "N/A") if stats else B_INCEPT.get(ticker, "N/A")

                    st.subheader(
                        f"Analysis for {ticker} | Fund inception: {inception} "
                        f"| Usable history from: {hist_from}"
                    )

                    if ticker in INDEX_NOTE:
                        st.caption(INDEX_NOTE[ticker])

                    if stats and stats.get("Hist Notes") not in (None, "-"):
                        st.caption(f"History lineage: {stats['Hist Notes']}")

                    if stats:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Max CAGR (since index data)", pct_or_na(stats.get("Max CAGR")))
                        c2.metric("Max Total Return", pct_or_na(stats.get("Max Total")))
                        c3.metric("15Y CAGR", pct_or_na(stats.get("15Y CAGR")))
                        c4.metric("Max Div CAGR", pct_or_na(stats.get("Max Div CAGR")))

                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("Expense ratio / yr", pct_or_na(stats.get("Expense")))
                        d2.metric(
                            "Worst drop (peak to bottom)",
                            pct_or_na(stats.get("Max DD")),
                            help="Deepest decline on the index-extended history. The date is when the bottom was reached."
                        )
                        d3.metric("Bottom reached", stats.get("DD Date", "-"))
                        d4.metric("10Y CAGR", pct_or_na(stats.get("10Y CAGR")))

                    df, holdings_source = get_holdings(ticker)

                    if df.empty:
                        st.warning(
                            f"Could not load holdings for {ticker}. Both holdings sources are "
                            "unreachable right now - wait ~1 minute and press the button again."
                        )
                    else:
                        display_df = merge_goog(df.rename(columns={"Raw_Weight": "Weight"}))
                        display_df["Weight %"] = (display_df["Weight"] * 100).round(2)
                        display_df["Industry"] = display_df["Symbol"].apply(
                            lambda x: IND_MAP.get(x, "Diversified / Other")
                        )

                        st.caption(f"{len(display_df)} holdings - source: {holdings_source}.")
                        st.dataframe(
                            display_df[["Symbol", "Industry", "Weight %"]],
                            height=400,
                            hide_index=True
                        )

    # --- TAB 5: WATCHLIST (candidates vs my portfolio) ---
    with tab5:
        st.header("Watchlist: candidates vs. my portfolio")
        st.caption(
            "Funds you might consider, lined up against a MY PORTFOLIO row built from your actual blend. "
            "A candidate earns a look only if it beats that row on the things that matter to you: "
            "long-run CAGR, cost, and worst drop."
        )

        watch_tickers = parse_tickers(st.text_input("Candidates:", DEFAULT_WATCH, key="watch_tickers"))
        base_tickers = parse_tickers(
            st.text_input("My portfolio (for the comparison row):", DEFAULT_PORT, key="watch_base")
        )

        if st.button("Update Watchlist", key="update_watchlist"):
            st.session_state["watch_go"] = True

        if st.session_state.get("watch_go"):
            if not watch_tickers:
                st.info("Enter at least one ticker.")
            else:
                data = get_stats_for_tickers(watch_tickers)
                base = get_stats_for_tickers(base_tickers) if base_tickers else []
                rows = []

                if base:
                    rows.append(weighted_portfolio_row(base, current_weights(base_tickers), label="MY PORTFOLIO"))

                rows.extend(data)

                if rows:
                    df = pd.DataFrame(rows)
                    final_cols = [col for col in STATS_TABLE_COLS if col in df.columns]

                    st.dataframe(format_dataframe(df[final_cols]), height=500, hide_index=True)

    # --- TAB: COMPARE (side by side, metrics as rows) ---
    with tab_cmp:
        st.header("⚖️ Compare funds side by side")
        st.caption(
            "Type two to four tickers. Metrics run down the page in plain words, funds run across, "
            "so it reads well on a phone. Your portfolio's four are the default; try QNDX, QQQ to see twins."
        )

        cmp_tickers = parse_tickers(st.text_input("Tickers to compare:", DEFAULT_PORT, key="cmp_tickers"))
        cmp_window = st.selectbox(
            "Growth chart window:",
            ["5Y", "10Y", "15Y", "20Y", "Max common history"],
            index=1,
            key="cmp_window"
        )

        if st.button("Compare in detail", key="compare_detail"):
            st.session_state["cmp_go"] = True

        if st.session_state.get("cmp_go"):
            if len(cmp_tickers) < 2:
                st.info("Enter at least two tickers.")
            else:
                sections, overlaps, verdicts = compare_sections(cmp_tickers[:6])

                if not sections:
                    st.warning("No usable data for those tickers right now. Wait a minute and press the button again.")
                else:
                    if verdicts:
                        st.markdown("### 🧭 The short version")
                        st.markdown("\n\n".join(f"- {v}" for v in verdicts))

                    for title, df in sections:
                        st.markdown(f"### {title}")
                        st.dataframe(df, hide_index=True)

                    if overlaps:
                        st.markdown("### 🔁 How much they overlap")
                        st.dataframe(pd.DataFrame(overlaps), hide_index=True)
                        st.caption(
                            "\"Same stocks by weight\" adds up, stock by stock, the smaller of the two funds' "
                            "weights. 100% would mean identical portfolios."
                        )

                    st.markdown("### 💰 Growth of $10,000")
                    years_map = {"5Y": 5, "10Y": 10, "15Y": 15, "20Y": 20, "Max common history": None}
                    growth_df, start_used, skipped = build_growth_frame(cmp_tickers[:6], years_map[cmp_window])

                    if growth_df.empty:
                        st.info("Not enough overlapping history for the selected window.")
                    else:
                        fig = px.line(
                            growth_df, x="Date", y="Growth of $10K", color="Ticker",
                            title=f"Growth of $10,000 - {cmp_window} (index-extended history)"
                        )
                        fig.update_yaxes(type="log")
                        show_chart(fig)

                        if start_used is not None:
                            st.caption(f"Chart starts {start_used.date()}. Log scale, so equal slopes mean equal growth rates.")

                    if skipped:
                        st.caption("Left off the chart (history starts after the window): " + ", ".join(skipped) + ".")

    # --- TAB: TAXES ---
    with tab_tax:
        st.header("🧾 Taxes: what each fund does to your tax bill")
        st.caption(
            "General U.S. rules for 2026 in plain words, plus an estimate for your own blend. Not personal tax "
            "advice. The exact qualified share of any fund is on your M1 1099-DIV (box 1b) each year."
        )

        # --- Part 1: profile of any ticker ---
        st.markdown("### 1. Look up any ticker")
        tax_tickers = parse_tickers(st.text_input("Tickers:", DEFAULT_PORT, key="tax_tickers"))

        for ticker in tax_tickers[:8]:
            key, how = tax_profile_key(ticker)
            profile = TAX_PROFILES[key]

            with st.expander(f"{ticker} - {profile['name']}", expanded=len(tax_tickers) <= 4):
                st.markdown(f"**What it pays you:** {profile['payouts']}")
                st.markdown(f"**Tax on those payouts:** {profile['payout_tax']}")
                st.markdown(f"**Capital-gain payouts from the fund:** {profile['gains_payouts']}")
                st.markdown(f"**When you sell shares:** {profile['selling']}")
                st.markdown(f"**Best account to hold it in:** {profile['best_home']}")
                st.markdown("**Upsides:** " + "; ".join(profile["upsides"]) + ".")
                st.markdown("**Downsides:** " + "; ".join(profile["downsides"]) + ".")
                st.markdown(f"**On your tax forms:** {profile['form']}")
                st.caption(f"Classified as: {how}.")

        st.markdown("---")

        # --- Part 2: blended estimate ---
        st.markdown("### 2. Estimate for my blend")
        st.caption(
            "Uses the X-Ray tickers and percentages. Only the share held in a taxable brokerage is taxed; "
            "the Roth share is $0 every year and $0 on qualified withdrawals."
        )

        x_tickers = parse_tickers(st.session_state.get("xray_tickers", DEFAULT_PORT))
        tax_weights = current_weights(x_tickers)

        t1, t2 = st.columns(2)

        with t1:
            tax_value = st.number_input(
                "Total portfolio value ($):",
                min_value=0, value=int(st.session_state.get("xray_value", 0) or 0), step=1000, key="tax_value"
            )
            taxable_pct = st.number_input(
                "Share held in a taxable brokerage (%):", min_value=0, max_value=100, value=50, step=5, key="tax_taxable_pct",
                help="The rest is assumed to be Roth IRA. 100 = everything is in taxable."
            )

        with t2:
            bracket = st.selectbox(
                "Federal tax bracket (%):", [10, 12, 22, 24, 32, 35, 37], index=2, key="tax_bracket",
                help="Your marginal federal rate on ordinary income. Housing and food allowances are not taxed, so company-grade officer pay usually lands in the 22% bracket."
            )
            state_pct = st.number_input(
                "State income tax on investments (%):", min_value=0.0, max_value=15.0, value=0.0, step=0.5, key="tax_state",
                help="0 if your state of legal residence has no income tax (Texas, Florida, and others)."
            )
            high_income = st.checkbox(
                "Income above $200k single / $250k married (adds the 3.8% net investment income tax)",
                value=False, key="tax_niit"
            )

        rates = tax_rates(bracket, high_income, state_pct)
        taxable_share = taxable_pct / 100

        st.caption(
            f"Rates used: qualified dividends and long-term gains {rates['qualified']:.1%}, "
            f"ordinary income {rates['ordinary']:.1%} (federal plus state)."
        )

        if tax_value <= 0 or not x_tickers:
            st.info("Enter a portfolio value above to see dollar estimates (nothing is saved).")
        else:
            rows, total_payout, total_tax = blended_tax_estimate(x_tickers, tax_weights, float(tax_value), taxable_share, rates)

            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)

                drag = total_tax / tax_value if tax_value else 0.0
                blended_expense = 0.0
                expense_weight = 0.0

                for ticker, weight in tax_weights.items():
                    ratio = get_expense_ratio(ticker)

                    if ratio is not None:
                        blended_expense += ratio * weight
                        expense_weight += weight

                expense_text = (
                    f" For scale, fund fees cost you about {blended_expense / expense_weight:.2%} a year."
                    if expense_weight > 0 else ""
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Payouts per year (whole portfolio)", f"${total_payout:,.0f}")
                m2.metric("Tax on them per year", f"${total_tax:,.0f}")
                m3.metric("Tax drag on the portfolio", f"{drag:.2%} / yr")

                st.markdown(
                    f"With {taxable_pct}% of the money in taxable and {100 - taxable_pct}% in Roth, your funds' payouts "
                    f"cost about **${total_tax:,.0f} a year** in tax, which is **{drag:.2%}** of the portfolio.{expense_text} "
                    "Nothing else is taxed until you sell, and you never have to."
                )

                # Projection
                st.markdown("#### If this grows for years")
                blended_now, _ = calculate_blended_performance(tax_weights)
                default_growth = blended_now.get("10Y CAGR")
                growth_pct = st.number_input(
                    "Assumed growth per year (%):", min_value=0.0, max_value=40.0,
                    value=float(round((default_growth or 0.10) * 100, 1)), step=0.5, key="tax_growth",
                    help="Starts at your blend's 10-year growth rate. Yields and tax rates are held constant."
                )
                g = growth_pct / 100
                proj_rows = []
                cumulative = 0.0

                for year in range(1, 31):
                    value_t = tax_value * (1 + g) ** (year - 1)
                    cumulative += value_t * drag

                    if year in (1, 5, 10, 20, 30):
                        proj_rows.append({
                            "After": f"{year} yr" if year == 1 else f"{year} yrs",
                            "Portfolio value": f"${tax_value * (1 + g) ** year:,.0f}",
                            "Tax paid on payouts so far": f"${cumulative:,.0f}",
                        })

                st.dataframe(pd.DataFrame(proj_rows), hide_index=True)
                st.caption(
                    "Assumes no new contributions, payouts reinvested, and today's yields, weights and tax rates "
                    "throughout. It is a scale check, not a forecast."
                )

        st.markdown("---")

        # --- Part 3: if you ever sold ---
        st.markdown("### 3. If you ever sold")
        basis = st.number_input(
            "What you paid in total for the taxable shares ($, optional):", min_value=0, value=0, step=1000, key="tax_basis"
        )

        if basis > 0 and tax_value > 0 and taxable_share > 0:
            taxable_value = float(tax_value) * taxable_share
            gain = taxable_value - basis

            if gain > 0:
                st.markdown(
                    f"Taxable-account value about **${taxable_value:,.0f}** on **${basis:,.0f}** paid in: gain of "
                    f"**${gain:,.0f}**. Sold after holding more than 1 year: about **${gain * rates['qualified']:,.0f}** "
                    f"in tax ({rates['qualified']:.1%}). Sold within a year: about **${gain * rates['ordinary']:,.0f}** "
                    f"({rates['ordinary']:.1%}). Roth shares: $0 on a qualified withdrawal."
                )
            else:
                st.markdown("No gain on those numbers, so no tax on a sale (a loss could offset other gains or up to $3,000 of income a year).")

        st.markdown(
            "Rules that matter for a weekly buyer: every purchase is its own lot with its own one-year clock; "
            "dividends on a lot are qualified once you have held it more than 60 days around the ex-dividend date, "
            "which a never-sell holder always meets; M1 sells specific lots in a set order when you do sell, so "
            "check that setting before any sale; and holding until you pass it on gives heirs a stepped-up basis."
        )

    # --- TAB 6: LIVE INSIGHTS & UPDATES ---
    with tab6:
        st.header("📰 Live Insights & Updates")

        insight_tickers = parse_tickers(
            st.text_input("Tickers to monitor:", DEFAULT_PORT, key="insights_tickers")
        )
        tickers_key = tuple(insight_tickers)

        top_l, top_m, top_r = st.columns([1, 1, 2])

        with top_l:
            if st.button("🔄 Refresh now", key="refresh_insights"):
                get_live_news.clear()
                get_pulse.clear()
                st.rerun()

        with top_m:
            if st.button("🧹 Clear all cached data", key="clear_all_cache",
                         help="Use this if a number looks stuck or wrong. Everything reloads fresh (about a minute)."):
                st.cache_data.clear()
                st.rerun()

        if not insight_tickers:
            st.info("Enter at least one ticker.")
        else:
            # --- Live market snapshot ---
            st.markdown("### ⏱️ Market Pulse")

            pulse_df, pulse_time = get_pulse(tickers_key)

            with top_r:
                st.caption(
                    f"Snapshot loaded {pulse_time.astimezone().strftime('%Y-%m-%d %H:%M %Z')} - "
                    "auto-refreshes every 15 minutes when the app is opened or rerun."
                )

            if pulse_df.empty:
                st.warning("Could not load the live snapshot (likely temporary rate-limiting). Press Refresh now.")
            else:
                st.dataframe(format_pulse(pulse_df), hide_index=True)

                # Computed, data-driven callouts (these change every day).
                callouts = []

                for _, row in pulse_df.iterrows():
                    bits = []

                    if pd.notnull(row.get("vs 52W High")):
                        gap = row["vs 52W High"]
                        bits.append(
                            "at/near its 52-week high" if gap > -0.02
                            else f"{abs(gap):.1%} below its 52-week high"
                        )

                    if row.get("200DMA") in ("Above", "Below"):
                        bits.append(f"{row['200DMA'].lower()} its 200-day average")

                    if pd.notnull(row.get("1M")):
                        bits.append(f"{row['1M']:+.1%} over the past month")

                    if bits:
                        callouts.append(f"**{row['Ticker']}** is " + ", ".join(bits) + ".")

                if callouts:
                    st.markdown("\n\n".join(callouts))

            st.markdown("---")

            # --- Live headlines ---
            st.markdown("### 🗞️ Latest Headlines")

            news_df, news_time = get_live_news(tickers_key)

            st.caption(
                f"Headlines fetched {news_time.astimezone().strftime('%Y-%m-%d %H:%M %Z')} "
                "(Yahoo Finance + Google News). Auto-refreshes every 15 minutes."
            )

            if news_df.empty:
                st.info("No headlines returned right now - press Refresh now to retry.")
            else:
                pick = st.multiselect(
                    "Filter by ticker:",
                    options=insight_tickers,
                    default=insight_tickers,
                    key="news_filter"
                )
                shown = news_df[news_df["Ticker"].isin(pick)].head(25)

                for _, item in shown.iterrows():
                    age = relative_age(item["Time"])
                    meta = " · ".join(x for x in [item["Ticker"], str(item["Source"]), age] if x)

                    if item["Link"]:
                        st.markdown(f"**[{item['Title']}]({item['Link']})**  \n{meta}")
                    else:
                        st.markdown(f"**{item['Title']}**  \n{meta}")

            st.markdown("---")

            # --- Computed portfolio read ---
            st.markdown("### 🧠 Portfolio Read (computed live)")

            read_weights = current_weights(insight_tickers)
            core_holdings, _counts = build_blended_holdings(insight_tickers, read_weights)

            if not core_holdings.empty:
                top1 = core_holdings.iloc[0]
                top3 = core_holdings.head(3)
                top3_weight = top3["Weight"].sum()
                top3_names = ", ".join(top3["Symbol"].tolist())
                top10_weight = core_holdings.head(10)["Weight"].sum()
                n_unique, n_two, n_all = holdings_overlap(insight_tickers)
                mix = ", ".join(f"{t} {w:.0%}" for t, w in read_weights.items())

                overlap_text = (
                    f" {n_two} stocks appear in two or more of your funds and {n_all} appear in all "
                    f"{len(insight_tickers)}, which is why these funds move together more than the fund "
                    "count suggests."
                    if len(insight_tickers) > 1 else ""
                )
                st.markdown(
                    f"Blending {mix} gives you {n_unique} underlying stocks. The top 3 ({top3_names}) are an "
                    f"estimated **{top3_weight:.1%}** of the portfolio and the top 10 are **{top10_weight:.1%}**. "
                    f"**{top1['Symbol']}** alone is ~**{top1['Weight']:.1%}**." + overlap_text
                )
                st.caption(
                    "Computed from each fund's published holdings (full basket where available): "
                    + " | ".join(f"{t}: {info}" for t, info in _counts.items())
                )

                st.dataframe(
                    core_holdings[["Symbol", "Industry", "Weight %"]].head(10),
                    hide_index=True
                )
            else:
                st.info("Holdings overlap unavailable right now (rate-limited) - press Refresh now to retry.")

            st.markdown("---")

            # --- AI prompt with live numbers (changes every day) ---
            st.markdown("### 🤖 1-Click Claude Intelligence Prompt")

            pulse_bits = []

            if not pulse_df.empty:
                for _, row in pulse_df.iterrows():
                    if pd.notnull(row.get("YTD")) and pd.notnull(row.get("1M")):
                        pulse_bits.append(f"{row['Ticker']} (YTD {row['YTD']:+.1%}, 1M {row['1M']:+.1%})")

            mix_text = ", ".join(f"{t} {w:.0%}" for t, w in read_weights.items())
            today_str = datetime.now().strftime("%B %d, %Y")
            prompt_text = (
                f"Today is {today_str}. Give me a comprehensive news and analysis update on my "
                f"portfolio: {', '.join(pulse_bits) if pulse_bits else ', '.join(insight_tickers)}, "
                f"weighted {mix_text}. Cover: (1) this week's key news per holding, (2) semiconductor, "
                "tech and momentum-factor dynamics, (3) macro factors (rates, AI capex) affecting these "
                "funds, and (4) anything a long-term buy-and-hold investor should monitor. Cite recent sources."
            )

            st.code(prompt_text, language="text")

            claude_url = "https://claude.ai/new?q=" + urllib.parse.quote(prompt_text)
            st.markdown(f"[🔗 Open in Claude (prompt pre-filled)]({claude_url})")
            st.caption(
                "Prompt embeds today's live numbers, so it changes every day. "
                "The link opens Claude with the prompt ready to send - or copy the box above."
            )

    # --- TAB 7: SHOWS & VOICES ---
    with tab7:
        st.header("🎧 Shows & Voices")
        st.caption(
            "Newest episodes and videos from the shows you follow. Refreshes every 30 minutes. "
            "To add a show, tell Claude the name or paste its YouTube link."
        )

        if st.button("🔄 Refresh shows", key="refresh_shows"):
            get_show_items.clear()
            st.rerun()

        shows_df, show_status, shows_time = get_show_items(tuple(SHOW_FEEDS))

        st.caption(
            f"Fetched {shows_time.astimezone().strftime('%Y-%m-%d %H:%M %Z')} - "
            + " | ".join(f"{name}: {state}" for name, state in show_status.items())
        )

        if shows_df.empty:
            st.info("No episodes came back right now - press Refresh shows to retry.")
        else:
            show_names = [name for name, _, _ in SHOW_FEEDS]
            show_pick = st.multiselect(
                "Filter by show:",
                options=show_names,
                default=show_names,
                key="show_filter"
            )
            shown_shows = shows_df[shows_df["Show"].isin(show_pick)].head(40)

            for _, item in shown_shows.iterrows():
                age = relative_age(item["Time"])
                meta = " · ".join(x for x in [str(item["Show"]), age] if x)

                if item["Link"]:
                    st.markdown(f"**[{item['Title']}]({item['Link']})**  \n{meta}")
                else:
                    st.markdown(f"**{item['Title']}**  \n{meta}")

            videos = shows_df[shows_df["VideoId"].notna()]

            if not videos.empty:
                st.markdown("---")
                st.markdown("### ▶️ Watch here")

                channels = videos.drop_duplicates(subset=["Show"])["Show"].tolist()
                watch_pick = st.selectbox("Latest video from:", channels, key="watch_pick")
                latest = videos[videos["Show"] == watch_pick].iloc[0]

                st.markdown(f"**{latest['Title']}**")
                st.video(f"https://www.youtube.com/watch?v={latest['VideoId']}")

    st.markdown("---")
    st.caption(
        f"Master Portfolio v{APP_VERSION}. Data: Yahoo Finance via yfinance (free, ~15-min delayed quotes). "
        "Prices cached 15 min, full history 6h, holdings 12h, shows 30 min. Long-period returns use "
        "index-extended history - see Hist Notes."
    )


if not os.environ.get("APP_TESTING"):
    main()
