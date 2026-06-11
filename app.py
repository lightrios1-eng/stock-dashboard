import io
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

DEFAULT_PORT = "QQQ, FTEC, SMH"
DEFAULT_BENCH = "VOO, QQQ"
DEFAULT_WATCH = "QQQ, FTEC, SMH"

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
    "QQQM": "Tech / Growth ETF", "QQQ": "Tech / Growth ETF", "XLK": "Technology ETF",
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
    "IVV": "2000-05-15", "VTI": "2001-05-24"
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

    for col in ALL_NUM_COLS:
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
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("latin-1", errors="replace")


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
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return empty, None

    for table in tables:
        cols = [str(c).strip().lower() for c in table.columns]

        sym_idx = [i for i, c in enumerate(cols) if "symbol" in c]
        wt_idx = [i for i, c in enumerate(cols) if "weight" in c]

        if not sym_idx or not wt_idx:
            continue

        out = table.iloc[:, [sym_idx[0], wt_idx[0]]].copy()
        out.columns = ["Symbol", "Raw_Weight"]

        out = out.dropna(subset=["Symbol"])
        out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
        out["Raw_Weight"] = pd.to_numeric(out["Raw_Weight"], errors="coerce") / 100.0

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
def get_history_bundle(symbol):
    """Max-history adjusted close (total return) + dividends for one symbol. Cached 6h."""
    symbol = symbol.strip().upper()

    if not symbol:
        return pd.Series(dtype=float), pd.Series(dtype=float)

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
        return pd.Series(dtype=float), pd.Series(dtype=float)

    return result


@st.cache_data(ttl=86400, show_spinner=False)
def get_info(symbol):
    """Ticker metadata (yield, category). Non-critical, cached 24h."""
    def fetch():
        info = yf.Ticker(symbol).info
        return info if info else None

    return _with_retries(fetch, tries=2, base_delay=1.5) or {}


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

    if not div.empty:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
        y_ttm = float(div[div.index >= cutoff].sum()) / price
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

    if div_run:
        notes.append(f"max div span {div_run}y")

    m["Hist Notes"] = "; ".join(notes) if notes else "-"

    return m


@st.cache_data(ttl=43200, show_spinner=False)
def get_holdings(ticker):
    """
    Holdings with source label. Tries the full basket from Fidelity research first
    (every position), then falls back to Yahoo's top-10 list.
    Returns (DataFrame[Symbol, Raw_Weight], source_label).
    """
    ticker = ticker.strip().upper()
    empty = pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    if not ticker:
        return empty, "none"

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
        return empty, "unavailable"

    return result, "top 10 only (Yahoo)"


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

    for label, (source_col, years) in period_sources.items():
        weighted_total = 0.0
        valid_weight = 0.0

        for ticker, stats in stats_by_ticker.items():
            value = stats.get(source_col)
            weight = weights.get(ticker, 0)

            if value is not None and pd.notnull(value):
                weighted_total += value * weight
                valid_weight += weight

        if valid_weight <= 0:
            results[label] = None
            continue

        total_return = weighted_total / valid_weight

        if years:
            results[label] = ((1 + total_return) ** (1 / years)) - 1 if total_return > -1 else None
        else:
            results[label] = total_return

    return results


# --- LIVE NEWS ENGINE ---
def _parse_news_time(value):
    try:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)

        text = str(value).strip()

        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))

        if "T" in text:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

        return parsedate_to_datetime(text)
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
        return pd.DataFrame(), None

    if years is not None:
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
    else:
        cutoff = max(s.index[0] for s in series_map.values())

    frames = []
    start_used = None

    for ticker, close in series_map.items():
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
        return pd.DataFrame(), None

    return pd.concat(frames, ignore_index=True), start_used


STATS_TABLE_COLS = (
    ["Ticker", "Price", "Industry", "Inception", "Hist From", "Streak", "Freq"]
    + ALL_NUM_COLS + ["Hist Notes"]
)


# --- UI ---
def main():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 X-Ray",
        "🆚 Benchmark",
        "📈 Dividends",
        "🔍 Deep Dive",
        "👀 Watchlist",
        "📰 Insights & Updates"
    ])

    # --- TAB 1: X-RAY ---
    with tab1:
        st.header("Portfolio X-Ray")

        etfs = parse_tickers(st.text_input("ETFs to Blend:", DEFAULT_PORT))

        if not etfs:
            st.info("Enter at least one ETF ticker.")
        else:
            cols = st.columns(len(etfs))
            raw_weights = {}
            default_weight = int(round(100 / len(etfs)))

            for i, ticker in enumerate(etfs):
                with cols[i]:
                    raw_weights[ticker] = st.slider(
                        f"{ticker} %",
                        min_value=0,
                        max_value=100,
                        value=default_weight,
                        key=f"weight_{ticker}"
                    ) / 100.0

            total_w = sum(raw_weights.values())

            if total_w > 0:
                weights = {ticker: weight / total_w for ticker, weight in raw_weights.items()}
                st.caption(f"Sliders sum to {total_w * 100:.0f}% - normalized to 100% for the math below.")
            else:
                weights = raw_weights
                st.warning("Set at least one ETF weight above 0.")

            if st.button("Analyze Blended Holdings", key="analyze_blended") and total_w > 0:
                st.subheader("📈 Blended Performance")
                st.caption(
                    "Long-period figures use index-extended history (older same-index funds / "
                    "indexes spliced in before each ETF's inception)."
                )

                blended_stats = calculate_blended_performance(weights)
                metric_cols = st.columns(len(PERF_PERIODS))

                for i, period in enumerate(PERF_PERIODS):
                    metric_cols[i].metric(period, pct_or_na(blended_stats.get(period)))

                st.markdown("---")

                full, source_counts = build_blended_holdings(etfs, weights)

                if full.empty:
                    st.warning("Could not load holdings data for the selected ETFs.")
                    st.caption(
                        "Both holdings sources are unreachable right now. "
                        "Wait ~1 minute and press the button again - results are cached once loaded."
                    )
                else:
                    st.caption(
                        f"Showing all {len(full)} blended underlying positions, weighted by your slider mix."
                    )
                    st.caption(
                        "Sources - "
                        + " | ".join(f"{ticker}: {info}" for ticker, info in source_counts.items())
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
                        st.plotly_chart(fig, use_container_width=True)

                    with c2:
                        st.dataframe(
                            full[["Symbol", "Industry", "Weight %"]],
                            height=500,
                            hide_index=True
                        )

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
            if not p_list or not m_list:
                st.info("Enter at least one portfolio ticker and one benchmark ticker.")
            else:
                p_stats = get_stats_for_tickers(p_list)
                m_stats = get_stats_for_tickers(m_list)

                if p_stats and m_stats:
                    df_p = pd.DataFrame(p_stats)

                    avg_p = {
                        col: pd.to_numeric(df_p[col], errors="coerce").mean()
                        for col in ALL_NUM_COLS
                        if col in df_p.columns
                    }
                    avg_p["Ticker"] = "PORTFOLIO AVG"
                    avg_p["Inception"] = "-"
                    avg_p["Hist From"] = "-"
                    avg_p["Hist Notes"] = "-"

                    final = pd.DataFrame([avg_p] + m_stats)
                    final_cols = (
                        ["Ticker", "Inception", "Hist From"]
                        + [col for col in ALL_NUM_COLS if col in final.columns]
                        + (["Hist Notes"] if "Hist Notes" in final.columns else [])
                    )
                    final = final[[col for col in final_cols if col in final.columns]]
                    final = final.dropna(axis=1, how="all")

                    st.dataframe(format_dataframe(final), hide_index=True)

                    st.subheader("💰 Growth of $10,000")

                    years_map = {"10Y": 10, "15Y": 15, "20Y": 20, "25Y": 25, "Max common history": None}
                    growth_df, start_used = build_growth_frame(
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

                        st.plotly_chart(fig, use_container_width=True)

                        if start_used is not None:
                            st.caption(
                                f"Chart starts {start_used.date()}. Pre-inception eras use spliced "
                                "same-index fund / index data; price-only index eras exclude dividends."
                            )

    # --- TAB 3: DIVIDENDS ---
    with tab3:
        st.header("Dividend & Growth Data")

        t_list = parse_tickers(st.text_area("Tickers", DEFAULT_PORT))

        if st.button("Load Dividends", key="load_dividends"):
            if not t_list:
                st.info("Enter at least one ticker.")
            else:
                data = get_stats_for_tickers(t_list)

                if data:
                    df = pd.DataFrame(data)

                    avg_data = {
                        col: pd.to_numeric(df[col], errors="coerce").mean()
                        for col in ALL_NUM_COLS
                        if col in df.columns
                    }
                    avg_data.update({
                        "Ticker": "AVERAGE",
                        "Inception": "-",
                        "Hist From": "-",
                        "Industry": "-",
                        "Price": None,
                        "Streak": None,
                        "Freq": "-",
                        "Hist Notes": "-"
                    })

                    final = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)
                    final = final[[col for col in STATS_TABLE_COLS if col in final.columns]]

                    st.dataframe(format_dataframe(final), height=500, hide_index=True)
                    st.caption(
                        "Div CAGR uses completed calendar years. Where an ETF is younger than the "
                        "window, growth is chained from older same-index funds (see Hist Notes). "
                        "Max Div CAGR covers the longest unbroken yearly run available."
                    )

    # --- TAB 4: DEEP DIVE ---
    with tab4:
        st.header("Multi-ETF Deep Dive")

        deep_tickers = parse_tickers(st.text_input("Tickers:", DEFAULT_PORT, key="deep_dive_tickers"))

        if st.button("Inspect ETFs", key="inspect_etfs"):
            if not deep_tickers:
                st.info("Enter at least one ETF ticker.")
            else:
                for ticker in deep_tickers:
                    stats = get_full_stats(ticker)
                    hist_from = stats.get("Hist From", "N/A") if stats else "N/A"

                    st.subheader(
                        f"Analysis for {ticker} | Fund inception: {B_INCEPT.get(ticker, 'N/A')} "
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

    # --- TAB 5: WATCHLIST ---
    with tab5:
        st.header("Watchlist & Consideration")

        watch_tickers = parse_tickers(st.text_input("Tickers:", DEFAULT_WATCH, key="watch_tickers"))

        if st.button("Update Watchlist", key="update_watchlist"):
            if not watch_tickers:
                st.info("Enter at least one ticker.")
            else:
                data = get_stats_for_tickers(watch_tickers)

                if data:
                    df = pd.DataFrame(data)
                    final_cols = [col for col in STATS_TABLE_COLS if col in df.columns]

                    st.dataframe(format_dataframe(df[final_cols]), height=500, hide_index=True)

    # --- TAB 6: LIVE INSIGHTS & UPDATES ---
    with tab6:
        st.header("📰 Live Insights & Updates")

        insight_tickers = parse_tickers(
            st.text_input("Tickers to monitor:", DEFAULT_PORT, key="insights_tickers")
        )
        tickers_key = tuple(insight_tickers)

        top_l, top_r = st.columns([1, 3])

        with top_l:
            if st.button("🔄 Refresh now", key="refresh_insights"):
                get_live_news.clear()
                get_pulse.clear()
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

            equal_weights = {t: 1 / len(insight_tickers) for t in insight_tickers}
            core_holdings, _counts = build_blended_holdings(insight_tickers, equal_weights)

            if not core_holdings.empty:
                nvda_weight = core_holdings.loc[core_holdings["Symbol"] == "NVDA", "Weight"].sum()
                msft_aapl_weight = core_holdings.loc[
                    core_holdings["Symbol"].isin(["MSFT", "AAPL"]),
                    "Weight"
                ].sum()
                top3 = core_holdings.head(3)
                top3_weight = top3["Weight"].sum()
                top3_names = ", ".join(top3["Symbol"].tolist())

                st.markdown(
                    f"Blending {', '.join(insight_tickers)} equally, the top 3 underlying names "
                    f"({top3_names}) are an estimated **{top3_weight:.1%}** of the portfolio. "
                    f"NVDA alone is ~**{nvda_weight:.1%}**; MSFT + AAPL are ~**{msft_aapl_weight:.1%}**. "
                    "Overlap across these ETFs means the portfolio behaves more like a concentrated "
                    "tech basket than the fund count suggests."
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

            today_str = datetime.now().strftime("%B %d, %Y")
            prompt_text = (
                f"Today is {today_str}. Give me a comprehensive news and analysis update on my "
                f"portfolio: {', '.join(pulse_bits) if pulse_bits else ', '.join(insight_tickers)}, "
                "blended equally. Cover: (1) this week's key news per holding, (2) semiconductor and "
                "tech sector dynamics, (3) macro factors (rates, AI capex) affecting these funds, and "
                "(4) anything a long-term buy-and-hold investor should monitor. Cite recent sources."
            )

            st.code(prompt_text, language="text")

            claude_url = "https://claude.ai/new?q=" + urllib.parse.quote(prompt_text)
            st.markdown(f"[🔗 Open in Claude (prompt pre-filled)]({claude_url})")
            st.caption(
                "Prompt embeds today's live numbers, so it changes every day. "
                "The link opens Claude with the prompt ready to send - or copy the box above."
            )

        st.markdown("---")
        st.caption(
            "Data: Yahoo Finance via yfinance (free, ~15-min delayed quotes). Prices cached 15 min, "
            "full history 6h, holdings 12h. Long-period returns use index-extended history - see Hist Notes."
        )


if not os.environ.get("APP_TESTING"):
    main()
