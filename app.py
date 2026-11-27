import inspect
import io
import json
import math
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

# Streamlit treats text between two dollar signs as a math formula. Every "$" in this app is money,
# so all markdown and captions go through these two wrappers, which escape the sign.
_st_markdown, _st_caption = st.markdown, st.caption


def _money_safe(text):
    return text.replace("$", "\\$") if isinstance(text, str) else text


def md(body, **kwargs):
    return _st_markdown(_money_safe(body), **kwargs)


def cap(body, **kwargs):
    return _st_caption(_money_safe(body), **kwargs)

APP_VERSION = "3.0 (Nov 26, 2026)"

# Your actual portfolio. Every tab starts with these tickers at these percentages.
DEFAULT_PORT = "QNDX, VGT, SPMO"
DEFAULT_WEIGHTS = {"QNDX": 34, "VGT": 33, "SPMO": 33}
DEFAULT_BENCH = "VOO, QQQ"
# Watchlist = funds you used to hold or have weighed, shown next to a "MY PORTFOLIO" comparison row.
DEFAULT_WATCH = "MGK, FTEC, SPYM, SMH, QQQM, SCHG"

ALL_NUM_COLS = [
    "Yield (TTM)", "Yield (Fwd)", "1D", "1W", "1M", "YTD",
    "1Y Total", "3Y Total", "3Y CAGR", "5Y Total", "5Y CAGR",
    "10Y Total", "10Y CAGR", "15Y Total", "15Y CAGR",
    "20Y Total", "20Y CAGR", "25Y Total", "25Y CAGR",
    "Max Total", "Max CAGR",
    "3Y Div CAGR", "5Y Div CAGR", "10Y Div CAGR", "15Y Div CAGR",
    "20Y Div CAGR", "Max Div CAGR"
]

PERF_PERIODS = ["1D", "1W", "1M", "YTD", "1Y Total", "3Y CAGR", "5Y CAGR", "10Y CAGR", "15Y CAGR", "20Y CAGR"]

# Plain-word tile labels for the periods above, and what each one means.
PERF_LABELS = {
    "1D": "1 day", "1W": "1 week", "1M": "1 month", "YTD": "This year", "1Y Total": "1 year",
    "3Y CAGR": "3 yrs / yr", "5Y CAGR": "5 yrs / yr", "10Y CAGR": "10 yrs / yr",
    "15Y CAGR": "15 yrs / yr", "20Y CAGR": "20 yrs / yr",
}
PERF_HELP = {
    "1D": "Change from the previous market close to the latest price.",
    "1W": "Change over the last 5 market days.",
    "1M": "Change over the last 21 market days, about one month.",
    "YTD": "Change since the last close of last year.",
    "1Y Total": "Total return over the last 12 months, dividends included.",
    "3Y CAGR": "Average growth per year over the last 3 years, dividends included.",
    "5Y CAGR": "Average growth per year over the last 5 years, dividends included.",
    "10Y CAGR": "Average growth per year over the last 10 years, dividends included.",
    "15Y CAGR": "Average growth per year over the last 15 years, dividends included.",
    "20Y CAGR": "Average growth per year over the last 20 years, dividends included.",
}

# Width of one number tile in pixels. Two fit side by side on a phone.
TILE_WIDTH = 158

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
    "SPYM": [("VOO", "fund"), ("^SP500TR", "tr_index"), ("VFINX", "fund"), ("^GSPC", "pr_index")],
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
    "MGK":  "Tracks CRSP US Mega Cap Growth. Extended with VUG and IWF (similar large-growth indexes).",
    "SPYM": "State Street SPDR Portfolio S&P 500 ETF (0.02%/yr). The same fund that used to trade as SPLG; "
            "the ticker changed in late 2025. Tracks the S&P 500. Extended with VOO and S&P 500 index history.",
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
    "VYM": "Dividend ETF", "VIG": "Dividend ETF", "JEPQ": "Covered-Call ETF",
    "SPYM": "S&P 500 ETF",
    "JNJ": "Pharma & MedTech", "MRK": "Pharma", "ABBV": "Pharma", "PFE": "Pharma", "BMY": "Pharma",
    "AZN": "Pharma", "TMO": "Life Science Tools", "ABT": "Medical Devices", "VRTX": "Biotech",
    "GILD": "Biotech", "REGN": "Biotech", "CVS": "Pharmacy/Health Ins",
    "XOM": "Oil & Gas", "CVX": "Oil & Gas", "COP": "Oil & Gas", "VLO": "Oil Refining",
    "MPC": "Oil Refining", "PSX": "Oil Refining",
    "BRK.B": "Insurance/Conglomerate", "BAC": "Bank", "WFC": "Bank", "C": "Bank",
    "GS": "Investment Bank", "MS": "Investment Bank", "AXP": "Payments/Cards", "SCHW": "Brokerage",
    "BLK": "Asset Manager", "SPGI": "Financial Data", "PGR": "Insurance", "CB": "Insurance",
    "CAT": "Heavy Machinery", "GE": "Aerospace", "RTX": "Aerospace & Defense", "LMT": "Defense",
    "BA": "Aerospace", "HON": "Industrial Conglomerate", "UNP": "Railroad", "CSX": "Railroad",
    "GEV": "Power Equipment", "ETN": "Electrical Equipment", "UBER": "Ride-Hailing",
    "BKNG": "Online Travel", "MCD": "Restaurants", "SBUX": "Restaurants", "NKE": "Apparel",
    "LOW": "Home Improvement", "TJX": "Off-Price Retail", "MELI": "E-Commerce (Latin America)",
    "PDD": "E-Commerce (China)", "SHOP": "E-Commerce Software",
    "PG": "Household Products", "PM": "Tobacco", "MO": "Tobacco",
    "T": "Telecom", "VZ": "Telecom", "TMUS": "Telecom", "CMCSA": "Cable/Media", "DIS": "Media & Parks",
    "IBM": "IT Services", "ACN": "IT Services", "ADP": "Payroll Software", "APP": "Ad Software",
    "ANET": "Networking Hardware", "APH": "Electronic Components", "DELL": "PC/Server Hardware",
    "HPE": "Server Hardware", "STX": "Data Storage", "WDC": "Data Storage", "SNDK": "Data Storage (Flash)",
    "GLW": "Glass/Fiber Optics", "LITE": "Optical Components", "COHR": "Optical Components",
    "CIEN": "Optical Networking", "MRVL": "Semi - Data Center", "MSTR": "Bitcoin Treasury/Software",
    "LIN": "Industrial Gases", "NEE": "Utility", "WELL": "Health Care REIT", "PLD": "Industrial REIT",
}

# Company names for the stocks most likely to top a blend. Anything not listed shows its symbol.
NAME_MAP = {
    "NVDA": "Nvidia", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "GOOG": "Alphabet (both share classes)", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "AVGO": "Broadcom", "TSLA": "Tesla", "BRK.B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase", "LLY": "Eli Lilly", "V": "Visa", "MA": "Mastercard", "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson", "WMT": "Walmart", "COST": "Costco", "NFLX": "Netflix", "ORCL": "Oracle",
    "AMD": "Advanced Micro Devices", "MU": "Micron Technology", "INTC": "Intel", "CSCO": "Cisco Systems",
    "PLTR": "Palantir", "CRM": "Salesforce", "ADBE": "Adobe", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "AMAT": "Applied Materials", "LRCX": "Lam Research", "KLAC": "KLA",
    "ADI": "Analog Devices", "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike", "NOW": "ServiceNow",
    "INTU": "Intuit", "IBM": "IBM", "ACN": "Accenture", "ANET": "Arista Networks", "APH": "Amphenol",
    "MRK": "Merck", "UNH": "UnitedHealth", "ABBV": "AbbVie", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "PG": "Procter & Gamble", "HD": "Home Depot", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "CAT": "Caterpillar", "RTX": "RTX", "GE": "GE Aerospace",
    "TMUS": "T-Mobile US", "ISRG": "Intuitive Surgical", "AMGN": "Amgen", "BKNG": "Booking Holdings",
    "LIN": "Linde", "SHOP": "Shopify", "ASML": "ASML", "ARM": "Arm Holdings", "PDD": "PDD (Temu)",
    "APP": "AppLovin", "MELI": "MercadoLibre", "STX": "Seagate", "WDC": "Western Digital",
    "SNDK": "Sandisk", "DELL": "Dell", "MRVL": "Marvell", "SNPS": "Synopsys", "CDNS": "Cadence Design",
    "TSM": "Taiwan Semiconductor", "UBER": "Uber", "CVX": "Chevron", "WFC": "Wells Fargo",
    "C": "Citigroup", "VLO": "Valero", "WELL": "Welltower", "LITE": "Lumentum",
    "MPC": "Marathon Petroleum", "HON": "Honeywell", "GILD": "Gilead", "VRTX": "Vertex",
    "MSTR": "Strategy (MicroStrategy)", "ADP": "ADP", "SBUX": "Starbucks", "CMCSA": "Comcast",
    "PM": "Philip Morris", "T": "AT&T", "VZ": "Verizon", "DIS": "Disney", "MCD": "McDonald's",
    "ABT": "Abbott", "TMO": "Thermo Fisher", "AXP": "American Express", "NKE": "Nike", "BA": "Boeing",
    "LMT": "Lockheed Martin", "MPWR": "Monolithic Power", "NXPI": "NXP", "MCHP": "Microchip",
    "TER": "Teradyne", "GLW": "Corning", "CIEN": "Ciena", "COHR": "Coherent", "HPE": "Hewlett Packard Enterprise",
    "BMY": "Bristol-Myers Squibb", "CVS": "CVS Health", "PSX": "Phillips 66", "PLD": "Prologis",
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
# QNDX, VGT, SPMO, QQQ, FTEC, MGK, SPYM and VOO were re-checked Nov 24-26, 2026. SMH, QQQM and VTI were
# last checked Sep 5, 2026. QQQ fell from 0.20% to 0.18% on Dec 22, 2025; versions 2.0 to 2.4 still
# carried the old 0.20%. SPLG is the old ticker of SPYM.
FEES_VERIFIED = "Nov 26, 2026"
EXPENSE_RATIO = {
    "QNDX": 0.0010, "VGT": 0.0009, "SPMO": 0.0013, "QQQ": 0.0018, "QQQM": 0.0015, "FTEC": 0.00084,
    "MGK": 0.0005, "SPYM": 0.0002, "SPLG": 0.0002, "SMH": 0.0035, "VOO": 0.0003, "VTI": 0.0003,
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
            if col == "Expense":
                df[col] = pd.to_numeric(df[col], errors="coerce").apply(fmt_expense)
            else:
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


class _PartialData(Exception):
    """Carries a usable but incomplete result out of a cached function, so it is shown now but not
    kept: the next screen refresh tries again for the complete version."""

    def __init__(self, result):
        super().__init__("partial data")
        self.result = result


@st.cache_resource(show_spinner=False)
def _failure_log():
    """Downloads that just failed, with the time to try each one again. Stops a failing symbol from
    being retried (slowly) on every screen refresh, without ever remembering the failure for hours."""
    return {}


def _recently_failed(key):
    until = _failure_log().get(key)
    return bool(until and time.time() < until)


def _mark_failed(key, seconds=600):
    _failure_log()[key] = time.time() + seconds


def _with_retries(fetch_fn, tries=3, base_delay=2.0):
    """Retry wrapper: Yahoo rate-limits cloud IPs (HTTP 429), one retry usually clears it."""
    for attempt in range(tries):
        try:
            result = fetch_fn()

            if result is not None:
                return result
        except Exception:
            pass

        if attempt < tries - 1:
            time.sleep(base_delay * (attempt + 1))

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
    symbol = str(symbol).strip().upper()
    empty = (pd.Series(dtype=float), pd.Series(dtype=float))

    # A symbol that failed in the last 10 minutes is skipped, so one bad ticker cannot slow every refresh.
    if not symbol or _recently_failed("hist:" + symbol):
        return empty

    try:
        return _history_bundle_cached(symbol)
    except Exception:
        _mark_failed("hist:" + symbol)
        return empty


@st.cache_data(ttl=86400, show_spinner=False)
def _info_cached(symbol):
    """Ticker metadata from Yahoo. Raises when nothing comes back so a miss is not kept for a day."""
    def fetch():
        info = yf.Ticker(symbol).info
        return info if info else None

    result = _with_retries(fetch, tries=2, base_delay=1.5)

    if not result:
        raise _DataUnavailable(symbol)

    return result


def get_info(symbol):
    """Ticker metadata (yield, category, sector). Non-critical. Kept 24h when it loads; a failed
    lookup is tried again after 10 minutes instead of being remembered for a day."""
    symbol = str(symbol).strip().upper()

    if not symbol or _recently_failed("info:" + symbol):
        return {}

    try:
        return _info_cached(symbol)
    except Exception:
        _mark_failed("info:" + symbol)
        return {}


@st.cache_data(ttl=86400, show_spinner=False)
def _expense_ratio_lookup(ticker):
    """Expense ratio from Yahoo's fund data, as a fraction. Raises when Yahoo has nothing, so a miss is
    not kept for a day (in v2.4 a single failed lookup left the fund blank for 24 hours)."""
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

    raise _DataUnavailable(ticker)


def get_expense_ratio(ticker):
    """Annual expense ratio as a fraction. Verified table first, then Yahoo's fund data. A single
    stock has no fund fee, so it is 0. Returns None when unknown (shown as '-')."""
    ticker = str(ticker).strip().upper()

    if ticker in EXPENSE_RATIO:
        return EXPENSE_RATIO[ticker]

    if is_single_stock(ticker):
        return 0.0

    if _recently_failed("expense:" + ticker):
        return None

    try:
        return _expense_ratio_lookup(ticker)
    except Exception:
        _mark_failed("expense:" + ticker)
        return None


@st.cache_data(ttl=21600, show_spinner=False)
def _extended_close_cached(ticker):
    """Does the splicing for build_extended_close. Raises when the fund itself has no data, and hands
    back a partial result (shown, not kept) when an older fund or index in its chain did not download."""
    close, _ = get_history_bundle(ticker)

    if close.empty:
        raise _DataUnavailable(ticker)

    sources = [f"{ticker} {close.index[0].year}→now"]
    has_price_only = False
    has_proxy = False
    incomplete = False
    ext = close

    for symbol, kind in EXTEND_CHAIN.get(ticker, []):
        proxy_close, _ = get_history_bundle(symbol)

        if proxy_close.empty:
            incomplete = True
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

    result = (ext, sources, has_price_only, has_proxy)

    if incomplete:
        raise _PartialData(result)

    return result


def build_extended_close(ticker):
    """
    Splice the fund's adjusted close with older same-index funds / index data so
    returns reach back to (or toward) the index's inception, not just the ETF's.
    Returns (series, source_labels, has_price_only_era, has_proxy_era).
    """
    try:
        return _extended_close_cached(str(ticker).strip().upper())
    except _PartialData as partial:
        return partial.result
    except Exception:
        return pd.Series(dtype=float), [], False, False


def annual_dividends(symbol):
    _, div = get_history_bundle(symbol)

    if div.empty:
        return pd.Series(dtype=float)

    annual = div.groupby(div.index.year).sum()
    return annual[annual > 0]


@st.cache_data(ttl=21600, show_spinner=False)
def _div_growth_factors_cached(ticker):
    """Does the work for build_div_growth_factors. Hands back a partial result (shown, not kept) when
    an older same-index fund in the chain did not download."""
    current_year = pd.Timestamp.now().year
    factors = {}
    proxies_used = []
    incomplete = False

    chain = [ticker] + [s for s, kind in EXTEND_CHAIN.get(ticker, []) if kind == "fund"]

    for symbol in chain:
        close, _ = get_history_bundle(symbol)

        if close.empty:
            incomplete = True
            continue

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

    result = (pd.Series(factors, dtype=float).sort_index(), proxies_used)

    if incomplete:
        raise _PartialData(result)

    return result


def build_div_growth_factors(ticker):
    """
    Year -> dividend growth factor (div[y] / div[y-1]), preferring the fund's own
    history, then filling earlier years from same-index proxy funds in the chain.
    First partial year of each source is excluded. Returns (factors, proxy_symbols_used).
    """
    try:
        return _div_growth_factors_cached(ticker)
    except _PartialData as partial:
        return partial.result
    except Exception:
        return pd.Series(dtype=float), []


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
def _full_stats_cached(ticker):
    """Every long-history number for one ticker. Kept 6 hours when complete. Raises when there is no
    price data, and hands back a partial result (shown, not kept) when part of the older history did not
    download, so a bad minute at Yahoo never sticks for hours."""
    own_close, div = get_history_bundle(ticker)

    if own_close.empty:
        raise _DataUnavailable(ticker)

    complete = True

    try:
        ext_close, sources, has_price_only, has_proxy = _extended_close_cached(ticker)
    except _PartialData as partial:
        ext_close, sources, has_price_only, has_proxy = partial.result
        complete = False
    except Exception:
        ext_close, sources, has_price_only, has_proxy = own_close, [], False, False
        complete = False

    if ext_close.empty:
        ext_close = own_close

    price = float(own_close.iloc[-1])

    if price <= 0:
        raise _DataUnavailable(ticker)

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

    # The label and forward yield of a ticker this file does not already know come from Yahoo's
    # metadata. If that did not load, show what we have but do not keep it.
    if not info and ticker not in IND_MAP:
        complete = False

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
        "As of": str(own_close.index[-1].date()),
        "Industry": industry,
        "Inception": B_INCEPT.get(ticker, str(own_close.index[0].date())),
        "Hist From": str(ext_close.index[0].date()),
        "Yield (Fwd)": y_fwd,
        "Yield (TTM)": y_ttm,
        "Expense": None,   # filled in fresh by get_full_stats so a fee is never 6 hours stale
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
    m["Span Years"] = span_years

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
    try:
        factors, div_proxies = _div_growth_factors_cached(ticker)
    except _PartialData as partial:
        factors, div_proxies = partial.result
        complete = False
    except Exception:
        factors, div_proxies = pd.Series(dtype=float), []
        complete = False

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

    if not complete:
        raise _PartialData(m)

    return m


def get_full_stats(ticker):
    """Stats for one ticker, or None when there is no price data. The fee is looked up fresh each time."""
    ticker = str(ticker).strip().upper()

    if not ticker:
        return None

    try:
        stats = dict(_full_stats_cached(ticker))
    except _PartialData as partial:
        stats = dict(partial.result)
    except Exception:
        return None

    stats["Expense"] = get_expense_ratio(ticker)
    return stats


SHORT_MOVE_COLS = ("1D", "1W", "1M")


def freshen_stats(stats, pulse_row):
    """Fold the newest price into one ticker's stats. The long history is cached for 6 hours, but the
    price snapshot is 15 minutes fresh, so the day, week and month moves come from the snapshot and
    every longer figure has its end point moved to the newest price."""
    stats = dict(stats)

    if pulse_row is None:
        return stats

    try:
        new_price = float(pulse_row["Price"])
        old_price = float(stats["Price"])
        as_of = str(pulse_row["As of"])
    except Exception:
        return stats

    if not (new_price > 0 and old_price > 0) or as_of < str(stats.get("As of") or ""):
        return stats

    factor = new_price / old_price

    # A jump this size between two looks at the same fund is a bad tick or a split the two feeds
    # disagree on, not a market move. Leave the cached figures alone.
    if not 0.5 < factor < 2.0:
        return stats

    for col in SHORT_MOVE_COLS:
        fresh = pulse_row.get(col)

        if fresh is not None and pd.notnull(fresh):
            stats[col] = float(fresh)

    # Two facts only the price snapshot has: distance from the 52-week high and the 200-day trend.
    for col in ("vs 52W High", "200DMA"):
        fresh = pulse_row.get(col)

        if fresh is not None and pd.notnull(fresh):
            stats[col] = fresh

    if abs(factor - 1) > 1e-9:
        for col in ["YTD", "Max Total"] + [f"{years}Y Total" for years in TOTAL_YEARS]:
            old = stats.get(col)

            if old is not None and pd.notnull(old):
                stats[col] = (1 + old) * factor - 1

        for years in TOTAL_YEARS:
            if years > 1:
                total = stats.get(f"{years}Y Total")
                stats[f"{years}Y CAGR"] = (
                    ((1 + total) ** (1 / years)) - 1
                    if total is not None and pd.notnull(total) and total > -1
                    else None
                )

        span = stats.get("Span Years")
        max_total = stats.get("Max Total")

        if span and span >= 1 and max_total is not None and pd.notnull(max_total) and max_total > -1:
            stats["Max CAGR"] = (1 + max_total) ** (1 / span) - 1

        for col in ("Yield (TTM)", "Yield (Fwd)"):
            old = stats.get(col)

            if old is not None and pd.notnull(old):
                stats[col] = old / factor

    stats["Price"] = new_price
    stats["As of"] = as_of
    return stats


def fresh_stats(tickers):
    """{ticker: stats} with the newest price folded in. Tickers with no data are simply left out."""
    tickers = [t for t in dict.fromkeys(tickers) if t]

    if not tickers:
        return {}

    try:
        pulse_df, _ = get_pulse(tuple(tickers))
    except Exception:
        pulse_df = pd.DataFrame()

    pulse = {}

    if pulse_df is not None and not pulse_df.empty and "Ticker" in pulse_df.columns:
        pulse = {row["Ticker"]: row for _, row in pulse_df.iterrows()}

    out = {}

    for ticker in tickers:
        stats = get_full_stats(ticker)

        if stats:
            out[ticker] = freshen_stats(stats, pulse.get(ticker))

    return out


@st.cache_resource(ttl=86400, show_spinner=False)
def load_bundled_holdings():
    """Full-basket snapshots committed to the repo (holdings_static.json). Captured
    from Fidelity's public ETF research pages, which block cloud-server IPs - so the
    snapshots are bundled with the app and refreshed by committing a new file.
    Read-only: nothing in the app changes this dictionary."""
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdings_static.json")

        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(ttl=43200, show_spinner=False)
def _holdings_cached(ticker):
    """Cached 12h on success; raises on failure so failures are retried, not cached.
    Order: bundled full-basket snapshot -> live Fidelity research -> Yahoo top-10. A cut-off list
    (far fewer names than the fund really holds) never wins over a complete one: if the bundled
    list is cut off the live page is still tried, and the longer of the two is used."""
    bundled = load_bundled_holdings().get(ticker)
    cut_off = None

    if bundled and bundled.get("rows"):
        df = pd.DataFrame(bundled["rows"], columns=["Symbol", "Raw_Weight"])
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Raw_Weight"] = pd.to_numeric(df["Raw_Weight"], errors="coerce")
        df = df.dropna(subset=["Raw_Weight"])
        df = df[df["Raw_Weight"] > 0]

        if len(df) >= 5 and 0.4 <= df["Raw_Weight"].sum() <= 1.6:
            asof = bundled.get("asof", "n/a")

            if holdings_completeness(ticker, len(df)) != "partial":
                return df.reset_index(drop=True), f"full basket snapshot, as of {asof}"

            cut_off = (
                df.reset_index(drop=True),
                f"partial list: {len(df)} of ~{EXPECTED_HOLDINGS[ticker]} (snapshot as of {asof})",
            )

    full_df, asof = fetch_fidelity_holdings(ticker)

    if not full_df.empty:
        if holdings_completeness(ticker, len(full_df)) != "partial":
            label = "full basket" + (f", as of {asof}" if asof else "")
            return full_df, label

        if cut_off is None or len(full_df) > len(cut_off[0]):
            cut_off = (full_df, f"partial list: {len(full_df)} of ~{EXPECTED_HOLDINGS[ticker]} (Fidelity page cut off)")

    if cut_off is not None:
        return cut_off

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


def is_single_stock(ticker):
    """True for one company's own shares (NVDA), False for a fund. A fund is recognised from the
    bundled holdings file, the fee table and the history table; after that Yahoo's own label decides.
    If Yahoo does not answer, a ticker that sits inside the sector funds' baskets is a stock."""
    ticker = str(ticker).strip().upper()

    if not ticker or ticker in EXPENSE_RATIO or ticker in EXTEND_CHAIN or ticker in load_bundled_holdings():
        return False

    quote_type = str(get_info(ticker).get("quoteType") or "").upper()

    if quote_type:
        return quote_type == "EQUITY"

    mapping = sector_map()
    return ticker in mapping or ticker.replace("-", ".") in mapping


def get_holdings(ticker):
    """
    Holdings with source label. Tries the full basket from Fidelity research first
    (every position, any issuer), then falls back to Yahoo's top-10 list. A single stock
    is one holding at 100%, so stocks can be blended with funds.
    Returns (DataFrame[Symbol, Raw_Weight], source_label).
    """
    ticker = str(ticker).strip().upper()
    empty = pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    if not ticker:
        return empty, "none"

    if is_single_stock(ticker):
        return pd.DataFrame({"Symbol": [ticker.replace("-", ".")], "Raw_Weight": [1.0]}), "single stock"

    if _recently_failed("holdings:" + ticker):
        own_df, own_source = empty, "unavailable"
    else:
        try:
            own_df, own_source = _holdings_cached(ticker)
        except Exception:
            _mark_failed("holdings:" + ticker)
            own_df, own_source = empty, "unavailable"

    if own_source.startswith("full basket"):
        return own_df, own_source

    # Only a top-10 or cut-off list (or nothing) for this fund. If an older fund tracks the same index
    # and we have its full basket, that basket is a far better picture than 10 names.
    for symbol, kind in EXTEND_CHAIN.get(ticker, []):
        if kind != "fund":
            continue

        if _recently_failed("holdings:" + symbol):
            continue

        try:
            proxy_df, proxy_source = _holdings_cached(symbol)
        except Exception:
            _mark_failed("holdings:" + symbol)
            continue

        if proxy_source.startswith("full basket") and not proxy_df.empty:
            return proxy_df, f"{symbol}'s {proxy_source} used as same-index stand-in"

    return own_df, own_source


# --- SECTORS (look-through: which part of the economy each underlying stock belongs to) ---
# A stock's sector is read from which sector fund holds it. Sector funds are built on the standard
# sector system the index companies use (GICS), so a stock inside Vanguard's technology fund is a
# technology stock by definition. Vanguard's funds are read first, then Fidelity's, then State
# Street's, because Vanguard's reach the most companies. The two chip funds at the end of the
# technology list pick up foreign chip makers (Taiwan Semiconductor, ASML) that U.S. sector funds skip.
SECTOR_FUNDS = {
    "Technology": ["VGT", "FTEC", "XLK", "SMH", "SOXX"],
    "Health Care": ["VHT", "FHLC", "XLV"],
    "Financials": ["VFH", "FNCL", "XLF"],
    "Consumer Discretionary": ["VCR", "FDIS", "XLY"],
    "Communication Services": ["VOX", "FCOM", "XLC"],
    "Industrials": ["VIS", "FIDU", "XLI"],
    "Consumer Staples": ["VDC", "FSTA", "XLP"],
    "Energy": ["VDE", "FENY", "XLE"],
    "Utilities": ["VPU", "FUTY", "XLU"],
    "Real Estate": ["VNQ", "FREL", "XLRE"],
    "Materials": ["VAW", "FMAT", "XLB"],
}

# Well-known companies no U.S. sector fund holds, mostly because they are based abroad.
SECTOR_MANUAL = {
    "TSM": "Technology", "ASML": "Technology", "ARM": "Technology", "SHOP": "Technology",
    "CRWV": "Technology", "SAP": "Technology",
    "PDD": "Consumer Discretionary", "BABA": "Consumer Discretionary", "CPNG": "Consumer Discretionary",
    "MELI": "Consumer Discretionary", "TM": "Consumer Discretionary",
    "AZN": "Health Care", "NVO": "Health Care", "NVS": "Health Care",
    "TRI": "Industrials", "FER": "Industrials", "RBA": "Industrials",
    "CCEP": "Consumer Staples", "BRK.A": "Financials", "BRK.B": "Financials",
    "VG": "Energy", "SPOT": "Communication Services",
}

# Yahoo's sector names, translated to the names used above (only for the few leftovers).
YAHOO_SECTOR = {
    "Technology": "Technology", "Healthcare": "Health Care", "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary", "Consumer Defensive": "Consumer Staples",
    "Communication Services": "Communication Services", "Industrials": "Industrials",
    "Energy": "Energy", "Utilities": "Utilities", "Real Estate": "Real Estate",
    "Basic Materials": "Materials",
}

CASH_LABEL = "Cash and other"
UNCLASSIFIED = "Not classified"
_CASH_SYMBOLS = {"XTSLA", "CASH", "USD", "CASH_USD", "USDOLLAR"}
_MONEY_FUND_RE = re.compile(r"^[A-Z]{3}XX$")
_FUTURES_RE = re.compile(r"^[A-Z0-9]{2,3}[FGHJKMNQUVXZ](2[5-9]|3[0-9])$")


def is_cash_like(symbol):
    """Money-market sweep funds (five letters ending in XX) and index futures (ESZ26) are a fund's
    cash drawer, not stocks."""
    symbol = str(symbol).strip().upper()
    return symbol in _CASH_SYMBOLS or bool(_MONEY_FUND_RE.match(symbol)) or bool(_FUTURES_RE.match(symbol))


@st.cache_resource(ttl=86400, show_spinner=False)
def sector_map():
    """{stock symbol: sector}, read from the sector funds' baskets in holdings_static.json.
    Read-only: nothing in the app changes this dictionary."""
    bundled = load_bundled_holdings()
    mapping = {}
    depth = max(len(funds) for funds in SECTOR_FUNDS.values())

    for rank in range(depth):
        for sector, funds in SECTOR_FUNDS.items():
            if rank >= len(funds):
                continue

            for row in (bundled.get(funds[rank]) or {}).get("rows", []):
                symbol = str(row[0]).strip().upper()

                if symbol and not is_cash_like(symbol):
                    mapping.setdefault(symbol, sector)

    for symbol, sector in SECTOR_MANUAL.items():
        mapping.setdefault(symbol, sector)

    if "GOOGL" in mapping:
        mapping.setdefault("GOOG", mapping["GOOGL"])

    return mapping


def sector_of(symbol, lookup=False):
    """Sector for one stock symbol. With lookup=True, a symbol no sector fund holds is asked of Yahoo."""
    symbol = str(symbol).strip().upper()

    if is_cash_like(symbol):
        return CASH_LABEL

    mapping = sector_map()

    if symbol in mapping:
        return mapping[symbol]

    dotted = symbol.replace("-", ".")

    if dotted in mapping:
        return mapping[dotted]

    if lookup:
        yahoo_name = str(get_info(symbol.replace(".", "-")).get("sector") or "")

        if yahoo_name in YAHOO_SECTOR:
            return YAHOO_SECTOR[yahoo_name]

    return UNCLASSIFIED


def assign_sectors(frame, lookups=8, min_weight=0.0005):
    """Sector for every row of a holdings frame (columns Symbol and Weight, heaviest first). Up to
    `lookups` of the heaviest names no sector fund holds are asked of Yahoo; the rest stay unlabelled."""
    left = int(lookups)
    sectors = []

    for symbol, weight in zip(frame["Symbol"], frame["Weight"]):
        sector = sector_of(symbol)

        if sector == UNCLASSIFIED and left > 0 and weight >= min_weight:
            left -= 1
            sector = sector_of(symbol, lookup=True)

        sectors.append(sector)

    return sectors


def sector_shares(frame):
    """{sector: share} for a holdings frame that has Sector and Weight. What a fund keeps in cash,
    futures or does not list is counted as 'Cash and other', so the shares add up to 100%."""
    if frame is None or frame.empty:
        return {}

    total = float(frame["Weight"].sum())

    if total <= 0:
        return {}

    base = max(total, 1.0)
    shares = {sector: float(weight) / base for sector, weight in frame.groupby("Sector")["Weight"].sum().items()}
    remainder = 1.0 - sum(shares.values())

    if remainder > 0.00005:
        shares[CASH_LABEL] = shares.get(CASH_LABEL, 0.0) + remainder

    return shares


def fund_sector_shares(ticker, labels=None):
    """Sector split of one fund (or one stock) on its own. `labels` is an optional {symbol: sector}
    that wins over the built-in map, so a table can use one set of labels in every column."""
    df, _ = get_holdings(ticker)

    if df.empty:
        return {}

    frame = merge_goog(df.rename(columns={"Raw_Weight": "Weight"})).reset_index(drop=True)
    labels = labels or {}
    frame["Sector"] = [labels.get(symbol) or sector_of(symbol) for symbol in frame["Symbol"]]
    return sector_shares(frame)


def get_stats_for_tickers(tickers):
    stats = fresh_stats(tickers)
    data = [stats[ticker] for ticker in tickers if ticker in stats]
    failed = [ticker for ticker in tickers if ticker not in stats]

    if failed:
        st.warning(
            f"No usable market data for: {', '.join(failed)}. "
            "This is usually temporary Yahoo rate-limiting - wait ~1 minute and press the button again."
        )

    return data


def build_blended_holdings(etfs, weights, sector_lookups=8):
    """Every underlying position of the blend, heaviest first, with its sector, a plain-words industry
    label, the company name where known and which of your funds hold it."""
    columns = ["Symbol", "Weight", "Weight %", "Industry", "Sector", "Company", "In funds"]
    dfs = []
    source_counts = {}
    held_by = {}

    for ticker in etfs:
        weight = weights.get(ticker, 0)

        if weight <= 0:
            continue

        df, source = get_holdings(ticker)
        source_counts[ticker] = "single stock" if source == "single stock" else f"{len(df)} holdings ({source})"

        if not df.empty:
            df = df.copy()
            df["Weight"] = df["Raw_Weight"] * weight
            dfs.append(df[["Symbol", "Weight"]])

            for symbol in set(df["Symbol"].astype(str).str.strip().str.upper().replace({"GOOGL": "GOOG"})):
                held_by.setdefault(symbol, []).append(ticker)

    if not dfs:
        return pd.DataFrame(columns=columns), source_counts

    full = merge_goog(pd.concat(dfs, ignore_index=True)).reset_index(drop=True)
    full["Weight %"] = (full["Weight"] * 100).round(2)
    full["Sector"] = assign_sectors(full, lookups=sector_lookups)
    full["Industry"] = [IND_MAP.get(symbol) or sector for symbol, sector in zip(full["Symbol"], full["Sector"])]
    full["Company"] = [NAME_MAP.get(symbol, symbol) for symbol in full["Symbol"]]
    full["In funds"] = [", ".join(held_by.get(symbol, [])) for symbol in full["Symbol"]]

    return full[columns], source_counts


def calculate_blended_performance(weights, stats_by_ticker=None):
    """Blended return for each period on the X-Ray tiles. Each figure is the weighted average of the
    funds' total returns over that period (your percentages bought at the start and left alone);
    multi-year figures are then turned into a per-year rate."""
    if stats_by_ticker is None:
        stats_by_ticker = fresh_stats([t for t, w in weights.items() if w > 0])

    stats_by_ticker = {t: s for t, s in stats_by_ticker.items() if weights.get(t, 0) > 0 and s}

    period_sources = {
        "1D": ("1D", None),
        "1W": ("1W", None),
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
                "As of": str(close.index[-1].date()),
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


# --- BLEND ENGINE (the whole portfolio as one line: growth, worst drop, year by year) ---
def fresh_close(ticker, stats=None):
    """One ticker's index-extended price history with the newest price folded in. The long history
    is cached for 6 hours; `stats` (from fresh_stats) carries the 15-minute price, which replaces
    today's point or is added as a new last point. This keeps the chart, the yearly table and the
    tiles on the same closing price."""
    close = build_extended_close(ticker)[0]

    if close.empty:
        return close

    close = close[~close.index.duplicated(keep="last")]
    row = (stats or {}).get(ticker)

    if not row:
        return close

    try:
        stamp = pd.Timestamp(str(row["As of"]))
        price = float(row["Price"])
        last_day = close.index[-1].normalize()
    except Exception:
        return close

    if not price > 0 or not 0.5 < price / float(close.iloc[-1]) < 2.0:
        return close

    if stamp > last_day:
        return pd.concat([close, pd.Series([price], index=[stamp])])

    if stamp == last_day:
        close = close.copy()
        close.iloc[-1] = price

    return close


def blend_series(weights, stats=None):
    """(daily growth of the blend, the holding with the shortest record, the day that record starts).
    The line starts at 1.0 on the first day every holding has data and is put back to the target
    percentages at the end of each calendar year. Built on each holding's index-extended history.
    An empty series means a holding had no price data."""
    held = {t: float(w) for t, w in weights.items() if w and w > 0}
    nothing = (pd.Series(dtype=float), None, None)

    if not held:
        return nothing

    closes = {}

    for ticker in held:
        close = fresh_close(ticker, stats)

        if close.empty:
            return nothing

        closes[ticker] = close

    frame = pd.DataFrame(closes).sort_index().dropna()

    if len(frame) < 30:
        return nothing

    target = pd.Series(held)
    target = (target / target.sum())[list(frame.columns)]

    pieces = []
    value = 1.0
    base = frame.iloc[0]

    for _year, chunk in frame.groupby(frame.index.year):
        growth = chunk.div(base, axis=1).mul(target, axis=1).sum(axis=1) * value
        pieces.append(growth)
        value = float(growth.iloc[-1])
        base = chunk.iloc[-1]

    limiting = max(closes, key=lambda ticker: closes[ticker].index[0])
    return pd.concat(pieces), limiting, str(closes[limiting].index[0].date())


def worst_drop(series):
    """(deepest fall from a high, month the bottom was reached) for any price or growth series."""
    if series is None or len(series) < 2:
        return None, "-"

    try:
        drawdown = series / series.cummax() - 1
        return float(drawdown.min()), str(drawdown.idxmin().date())[:7]
    except Exception:
        return None, "-"


def calendar_year_returns(close):
    """Total return for each calendar year, indexed by year. A year counts only when there is a
    closing price from the year before, so a fund's first partial year is left out. The last entry
    is the current year so far."""
    if close is None or len(close) < 2:
        return pd.Series(dtype=float)

    year_end = close.groupby(close.index.year).last()
    return (year_end / year_end.shift(1) - 1).dropna()


def year_by_year(weights, extra_tickers, stats=None, max_years=12):
    """Calendar-year returns, newest year first: MY BLEND (your percentages at the start of each
    year), then each holding, then each extra ticker. Returns (text table, MY BLEND's full years as
    numbers, the current year)."""
    held = [t for t, w in weights.items() if w and w > 0]
    order = held + [t for t in extra_tickers if t not in held]
    columns = {}

    for ticker in order:
        returns = calendar_year_returns(fresh_close(ticker, stats))

        if not returns.empty:
            columns[ticker] = returns

    if not columns:
        return pd.DataFrame(), pd.Series(dtype=float), None

    frame = pd.DataFrame(columns).sort_index()
    current_year = int(frame.index.max())
    have = [t for t in held if t in frame.columns]

    if have and len(have) == len(held):
        target = pd.Series({t: float(weights[t]) for t in have})
        target = target / target.sum()
        blend = frame[have].mul(target, axis=1).sum(axis=1, skipna=False)
    else:
        blend = pd.Series(float("nan"), index=frame.index)

    frame.insert(0, "MY BLEND", blend)
    shown = frame.tail(max_years).iloc[::-1]

    table = pd.DataFrame({"Year": [f"{int(y)} so far" if int(y) == current_year else str(int(y)) for y in shown.index]})

    for col in shown.columns:
        table[col] = [f"{v:+.1%}" if pd.notnull(v) else "-" for v in shown[col]]

    full_years = blend[blend.index < current_year].dropna()
    return table, full_years, current_year


def blend_growth_frame(weights, extra_tickers, years, stats=None):
    """Growth of $10,000 for MY BLEND and each extra ticker, every line starting on the same day.
    years=None means as far back as the blend goes. Returns (long frame for the chart, facts)."""
    series, limiting, limit_date = blend_series(weights, stats)
    facts = {
        "start": None, "end": None, "limiting": limiting, "limit_date": limit_date, "shortened": False,
        "skipped": [], "ends": {}, "left_alone": None, "years": None,
    }

    if series.empty:
        return pd.DataFrame(), facts

    cutoff = series.index[-1] - pd.DateOffset(years=years) if years else series.index[0]
    window = series[series.index >= cutoff]

    if len(window) < 10:
        return pd.DataFrame(), facts

    start, end = window.index[0], window.index[-1]
    facts.update({
        "start": start, "end": end,
        "shortened": bool(years) and (start - cutoff).days > 45,
        "years": (end - start).days / 365.25,
    })

    lines = {"MY BLEND": window}

    for ticker in extra_tickers:
        close = fresh_close(ticker, stats)

        if close.empty:
            continue

        # A line that starts later than the others would begin at $10,000 on a later day and make
        # the chart lie, so it is left off and named.
        if close.index[0] > start + pd.Timedelta(days=10):
            facts["skipped"].append(f"{ticker} (history starts {close.index[0].year})")
            continue

        cut = close[close.index >= start]

        if len(cut) >= 10:
            lines[ticker] = cut

    frames = []

    for name, line in lines.items():
        growth = line / float(line.iloc[0]) * 10000
        weekly = growth.resample("W").last().dropna()
        points = pd.concat([growth.iloc[:1], weekly.iloc[:-1], growth.iloc[-1:]])
        points = points[~points.index.duplicated(keep="last")].sort_index()
        facts["ends"][name] = float(growth.iloc[-1])
        frames.append(pd.DataFrame({"Date": points.index, "Growth of $10K": points.values, "Line": name}))

    # The same money bought once at your percentages and never put back, for comparison.
    left_alone = 0.0
    total_weight = 0.0

    for ticker, weight in weights.items():
        if not weight or weight <= 0:
            continue

        close = fresh_close(ticker, stats)
        cut = close[(close.index >= start) & (close.index <= end)]

        if len(cut) >= 2 and float(cut.iloc[0]) > 0:
            left_alone += weight * float(cut.iloc[-1]) / float(cut.iloc[0])
            total_weight += weight

    if total_weight > 0:
        facts["left_alone"] = left_alone / total_weight * 10000

    return pd.concat(frames, ignore_index=True), facts


def blend_scorecard(held, weights, bench, stats, full, blended, excluded):
    """One table, facts down the side, MY BLEND then each holding then each benchmark across the top.
    Everything is text so the table renders the same on every screen."""
    columns = ["MY BLEND"] + [t for t in held + bench if t in stats]
    row_for_blend = weighted_portfolio_row([stats[t] for t in held if t in stats], weights, label="MY BLEND")
    series, limiting, limit_date = blend_series({t: weights[t] for t in held if t in stats}, stats)
    blend_dd, blend_dd_date = worst_drop(series)
    holdings = {t: get_holdings(t) for t in columns[1:]}
    details = {t: (dividend_detail(t) or {}) for t in columns[1:]}

    def pct(value, signed=False):
        if value is None or pd.isnull(value):
            return "-"

        return f"{value:+.2%}" if signed else f"{value:.2%}"

    def each(fn, blend_value):
        row = {"MY BLEND": blend_value}

        for ticker in columns[1:]:
            try:
                row[ticker] = fn(ticker)
            except Exception:
                row[ticker] = "-"

        return row

    def perf_row(col, period):
        mark = "*" if excluded.get(period) else ""
        blend_value = pct(blended.get(period), signed=True)
        return each(lambda t: pct(stats[t].get(col), signed=True), blend_value + (mark if blend_value != "-" else ""))

    def top_share(ticker, count):
        df, _ = holdings[ticker]

        if df.empty:
            return "-"

        merged = merge_goog(df.rename(columns={"Raw_Weight": "Weight"}))
        merged = merged[~merged["Symbol"].map(is_cash_like)]
        return f"{merged.head(count)['Weight'].sum():.1%}"

    def biggest_sector(shares):
        real = {k: v for k, v in shares.items() if k not in (CASH_LABEL, UNCLASSIFIED)}

        if not real:
            return "-"

        name = max(real, key=real.get)
        return f"{name} {real[name]:.0%}"

    def count_holdings(ticker):
        df, source = holdings[ticker]

        if df.empty:
            return "-"

        return "1 (a single stock)" if source == "single stock" else str(len(df))

    def borrowed(ticker):
        """' (QQQ's)' when a young fund is showing its same-index twin's payout record."""
        source = details[ticker].get("source")
        return f" ({source}'s)" if source and source != ticker else ""

    def money(ticker, key):
        value = details[ticker].get(key)
        return f"${value:.4f}{borrowed(ticker)}" if value is not None else "-"

    def payout_date(ticker):
        value = details[ticker].get("last_date")
        return f"{value}{borrowed(ticker)}" if value else "-"

    def from_high(value):
        if value is None or pd.isnull(value):
            return "-"

        return "at its high" if value > -0.0005 else f"{value:.1%}"

    def qualified_share(ticker):
        key, _how = tax_profile_key(ticker)
        return TAX_PROFILES[key]["qualified_share"]

    # The blend's own distance from its 52-week high and its 200-day trend, from the blend line.
    blend_high, blend_trend = None, "-"

    if len(series) >= 30:
        blend_high = float(series.iloc[-1] / series.iloc[-252:].max() - 1)

        if len(series) >= 200:
            blend_trend = "Above" if float(series.iloc[-1]) >= float(series.iloc[-200:].mean()) else "Below"

    # Share of the blend's dividends that is usually "qualified" (taxed at the lower rates).
    paid = {t: weights[t] * (stats[t].get("Yield (TTM)") or 0) for t in held if t in stats}
    blend_qualified = (
        f"{sum(paid[t] * qualified_share(t) for t in paid) / sum(paid.values()):.0%}"
        if sum(paid.values()) > 0 else "-"
    )

    stocks = full[full["Sector"] != CASH_LABEL] if not full.empty else full
    blend_expense = row_for_blend.get("Expense")
    rows = {}

    rows["Share of my blend"] = each(lambda t: f"{weights[t]:.0%}" if t in held else "benchmark", "100%")
    rows["Price"] = each(lambda t: f"${stats[t]['Price']:,.2f}", "-")
    rows["1 day"] = perf_row("1D", "1D")
    rows["1 week"] = perf_row("1W", "1W")
    rows["1 month"] = perf_row("1M", "1M")
    rows["From its 52-week high"] = each(lambda t: from_high(stats[t].get("vs 52W High")), from_high(blend_high))
    rows["Versus its 200-day average"] = each(lambda t: str(stats[t].get("200DMA") or "-"), blend_trend)
    rows["This year so far"] = perf_row("YTD", "YTD")
    rows["1 year"] = perf_row("1Y Total", "1Y Total")

    for years in (3, 5, 10, 15, 20):
        rows[f"{years} years (per year)"] = perf_row(f"{years}Y CAGR", f"{years}Y CAGR")

    rows["Cost per year"] = each(
        lambda t: "no fund fee" if holdings[t][1] == "single stock" else fmt_expense(stats[t].get("Expense")),
        fmt_expense(blend_expense),
    )
    rows["Dividend yield (last 12 months)"] = each(lambda t: pct(stats[t].get("Yield (TTM)")), pct(row_for_blend.get("Yield (TTM)")))
    rows["Pays"] = each(lambda t: {"Qr": "Quarterly", "Mo": "Monthly", "Yr": "Yearly"}.get(stats[t].get("Freq"), "-"), "-")
    rows["Last payout per share"] = each(lambda t: money(t, "last_amount"), "-")
    rows["Last ex-dividend date"] = each(payout_date, "-")
    rows["Payout growth, 5 yrs (per year)"] = each(lambda t: pct(stats[t].get("5Y Div CAGR")), pct(row_for_blend.get("5Y Div CAGR")))
    rows["Years of rising payouts"] = each(lambda t: str(stats[t].get("Streak", "-")), "-")
    rows["Usual qualified share of payouts"] = each(lambda t: f"{qualified_share(t):.0%}", blend_qualified)
    rows["Worst drop on record"] = each(lambda t: pct(stats[t].get("Max DD")), pct(blend_dd))
    rows["Bottom was hit"] = each(lambda t: str(stats[t].get("DD Date") or "-"), blend_dd_date)
    rows["Record starts"] = each(lambda t: str(stats[t].get("Hist From") or "-"), str(series.index[0].date()) if not series.empty else "-")
    rows["Number of holdings"] = each(count_holdings, str(len(stocks)) if not stocks.empty else "-")
    rows["Top 10 share"] = each(lambda t: top_share(t, 10), f"{stocks.head(10)['Weight'].sum():.1%}" if not stocks.empty else "-")
    rows["Biggest sector"] = each(lambda t: biggest_sector(fund_sector_shares(t)), biggest_sector(sector_shares(full)))
    rows["Fund since"] = each(lambda t: str(stats[t].get("Inception") or "-"), "-")

    table = pd.DataFrame(rows).T.astype(str)
    table.index.name = "Fact"
    return table.reset_index()[["Fact"] + columns]


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
    "IVV": "us_equity_index", "SPLG": "us_equity_index", "SPYM": "us_equity_index",
    "VTI": "us_equity_index", "ITOT": "us_equity_index",
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
    if ticker in SHORT_DESC:
        return SHORT_DESC[ticker]

    if ticker in INDEX_NOTE:
        return INDEX_NOTE[ticker].split(". ")[0][:90]

    return IND_MAP.get(ticker) or info.get("category") or info.get("longName") or "-"


def compare_sections(tickers):
    """Side-by-side tables (metrics as rows, funds as columns) plus computed one-line verdicts."""
    stats = {}
    holdings = {}
    fresh = fresh_stats(tickers)

    for ticker in tickers:
        s = fresh.get(ticker)

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
        "Cost per year": {t: fmt_expense(stats[t].get("Expense")) for t in cols},
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
    details = {t: (dividend_detail(t) or {}) for t in cols}

    def dv(t, key, money=False, pct_fmt=False):
        value = details[t].get(key)

        if value is None:
            return "-"

        if money:
            return f"${value:.4f}"

        if pct_fmt:
            return f"{value:+.1%}"

        return str(value)

    def source_note(t):
        src_t = details[t].get("source")
        return f" ({src_t}'s record)" if src_t and src_t != t else ""

    income = {
        "Dividend yield (last 12 months)": {t: pct(stats[t].get("Yield (TTM)")) for t in cols},
        "Pays": {t: {"Qr": "Quarterly", "Mo": "Monthly", "Yr": "Yearly"}.get(stats[t].get("Freq"), "-") for t in cols},
        "Usual payout months": {t: dv(t, "usual_months") + source_note(t) for t in cols},
        "Last payout per share": {t: dv(t, "last_amount", money=True) for t in cols},
        "Last ex-dividend date": {t: dv(t, "last_date") for t in cols},
        "Same payout a year earlier": {t: dv(t, "prev_amount", money=True) for t in cols},
        "Change vs. a year earlier": {t: dv(t, "yoy", pct_fmt=True) for t in cols},
        "Last 12 months per share": {t: dv(t, "ttm", money=True) for t in cols},
        "Prior 12 months per share": {t: dv(t, "prior_ttm", money=True) for t in cols},
        "Change, 12-month total": {t: dv(t, "ttm_change", pct_fmt=True) for t in cols},
        "Next/last pay date (Yahoo)": {t: dv(t, "pay_date_yahoo") for t in cols},
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
        verdicts.append(f"Cheapest: {cheapest} at {fmt_expense(exp[cheapest])} a year.")

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


# --- EXPECTED HOLDINGS COUNTS (to catch a cut-off list, e.g. 36 rows for a 100-stock fund) ---
EXPECTED_HOLDINGS = {
    "SPMO": 100, "QQQ": 101, "QQQM": 101, "QNDX": 101, "VGT": 300, "FTEC": 290, "SMH": 26,
    "VOO": 500, "SPY": 500, "IVV": 500, "SPLG": 500, "SPYM": 500, "VTI": 3500, "SCHG": 200, "VUG": 150,
    "XLK": 70, "SOXX": 30, "IYW": 140, "MGK": 56, "MTUM": 125,
}


def holdings_completeness(ticker, count):
    """'ok', 'partial' or 'unknown' given how many rows we have versus what the fund really holds."""
    expected = EXPECTED_HOLDINGS.get(ticker.strip().upper())

    if not expected:
        return "unknown"

    return "ok" if count >= 0.6 * expected else "partial"


def fmt_expense(value):
    """0.0013 -> '0.13%', 0.00084 -> '0.084%'."""
    if value is None or pd.isna(value):
        return "-"

    text = f"{float(value):.3%}"
    return text[:-2] + "%" if text.endswith("0%") else text


SHORT_DESC = {
    "SPMO": "S&P 500 Momentum Index: the ~100 S&P 500 stocks with the strongest recent price momentum",
    "QNDX": "Nasdaq-100 Index (same index as QQQ), State Street, 0.10%",
    "QQQ": "Nasdaq-100 Index, Invesco",
    "QQQM": "Nasdaq-100 Index, Invesco (cheaper QQQ)",
    "FTEC": "MSCI USA IMI Information Technology 25/50: essentially all U.S. tech stocks",
    "VGT": "MSCI US IMI Information Technology 25/50: essentially all U.S. tech stocks",
    "SMH": "MVIS US Listed Semiconductor 25: the 25 biggest U.S.-listed chip companies",
    "SOXX": "ICE Semiconductor Index: ~30 U.S.-listed chip companies",
    "VOO": "S&P 500 Index",
    "VTI": "Total U.S. stock market",
    "SCHG": "Dow Jones U.S. Large-Cap Growth",
    "VUG": "CRSP US Large Cap Growth",
    "XLK": "S&P 500 technology sector",
    "MGK": "CRSP US Mega Cap Growth: the biggest U.S. growth companies, Vanguard, 0.05%",
    "SPYM": "S&P 500 Index, State Street, 0.02% (used to trade as SPLG)",
}


# --- DIVIDEND DETAIL (per share, dates, growth) ---
def dividend_detail(ticker):
    """Per-share payout facts. A fund under a year old borrows its same-index twin's record and says so."""
    ticker = ticker.strip().upper()
    close, div = get_history_bundle(ticker)

    if close.empty:
        return None

    source = ticker
    own_days = (close.index[-1] - close.index[0]).days

    if div.empty or own_days < 365:
        for symbol, kind in EXTEND_CHAIN.get(ticker, []):
            if kind != "fund":
                continue

            _, proxy_div = get_history_bundle(symbol)

            if not proxy_div.empty:
                div = proxy_div
                source = symbol
                break

    info = get_info(ticker)
    dates = {}

    for key, label in (("exDividendDate", "ex_date_yahoo"), ("dividendDate", "pay_date_yahoo")):
        try:
            raw = info.get(key)

            if raw:
                dates[label] = str(datetime.fromtimestamp(float(raw), tz=timezone.utc).date())
        except Exception:
            pass

    result = {"source": source, "rows": [], "usual_months": "-", "last_amount": None, "last_date": None,
              "prev_amount": None, "yoy": None, "ttm": None, "prior_ttm": None, "ttm_change": None}
    result.update(dates)

    div = div[div > 0].sort_index()

    if div.empty:
        return result

    now = pd.Timestamp.now()
    last_date = div.index[-1]
    last_amount = float(div.iloc[-1])

    def year_earlier(date):
        target = date - pd.DateOffset(years=1)
        window = div[(div.index >= target - pd.Timedelta(days=45)) & (div.index <= target + pd.Timedelta(days=45))]
        return float(window.iloc[-1]) if not window.empty else None

    prev_amount = year_earlier(last_date)
    ttm = float(div[div.index > now - pd.DateOffset(years=1)].sum())
    prior = float(div[(div.index > now - pd.DateOffset(years=2)) & (div.index <= now - pd.DateOffset(years=1))].sum())

    result.update({
        "last_amount": last_amount,
        "last_date": str(last_date.date()),
        "prev_amount": prev_amount,
        "yoy": (last_amount / prev_amount - 1) if prev_amount else None,
        "ttm": ttm,
        "prior_ttm": prior if prior > 0 else None,
        "ttm_change": (ttm / prior - 1) if prior > 0 else None,
    })

    import calendar as _cal
    months = sorted({d.month for d in div.index[-4:]})
    result["usual_months"] = ", ".join(_cal.month_abbr[m] for m in months) if months else "-"

    for date, amount in list(div.tail(8).items())[::-1]:
        earlier = year_earlier(date)
        result["rows"].append({
            "Ex-date": str(date.date()),
            "$ per share": f"${float(amount):.4f}",
            "Same payout a year earlier": f"${earlier:.4f}" if earlier else "-",
            "Change": f"{float(amount) / earlier - 1:+.1%}" if earlier else "-",
        })

    return result


# --- MILITARY PAY, TAX AND VETERAN NUMBERS (2026, verified Sep 5, 2026) ---
STATIC_FACTS_VERIFIED = "Sep 5, 2026"

# Monthly basic pay, 2026 (DFAS table, 3.8% raise). Columns = years of service thresholds below.
YOS_COLS = [0, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
PAY_2026 = {
    "O-1": [4150.32, 4320.18, 5222.55, 5222.55, 5222.55, 5222.55, 5222.55, 5222.55, 5222.55, 5222.55, 5222.55, 5222.55],
    "O-2": [4781.94, 5446.11, 6272.66, 6484.50, 6617.60, 6617.60, 6617.60, 6617.60, 6617.60, 6617.60, 6617.60, 6617.60],
    "O-3": [5534.30, 6273.64, 6770.54, 7382.64, 7737.02, 8125.24, 8375.81, 8788.43, 9004.18, 9004.18, 9004.18, 9004.18],
    "O-4": [6294.79, 7286.32, 7773.47, 7880.85, 8332.20, 8816.41, 9419.73, 9888.32, 10214.39, 10401.83, 10509.86, 10509.86],
    "O-5": [7295.43, 8218.31, 8787.13, 8894.19, 9249.54, 9461.38, 9928.35, 10271.99, 10714.87, 11391.41, 11713.89, 12032.80],
    "O-6": [8751.33, 9614.00, 10244.98, 10244.98, 10284.35, 10724.96, 10783.54, 10783.54, 11396.29, 12479.59, 13115.45, 13750.98],
}
BAS_2026 = {"Officer": 328.48, "Enlisted": 476.95}
TSP_LIMIT_2026 = 24500
SS_WAGE_BASE_2026 = 184500
STD_DEDUCTION_2026 = {"Single": 16100, "Married filing jointly": 32200}
BRACKETS_2026 = {
    "Single": [(12400, 0.10), (50400, 0.12), (105700, 0.22), (201775, 0.24), (256225, 0.32), (640600, 0.35), (float("inf"), 0.37)],
    "Married filing jointly": [(24800, 0.10), (100800, 0.12), (211400, 0.22), (403550, 0.24), (512450, 0.32), (768700, 0.35), (float("inf"), 0.37)],
}
# VA disability compensation, monthly, effective Dec 1, 2025 (2.8% COLA): (veteran alone, veteran with spouse)
VA_RATES_2026 = {
    10: (180.42, 180.42), 20: (356.66, 356.66), 30: (552.47, 618.26), 40: (795.84, 883.22),
    50: (1132.90, 1241.87), 60: (1435.02, 1566.60), 70: (1808.45, 1961.62), 80: (2102.15, 2276.91),
    90: (2362.30, 2559.34), 100: (3938.58, 4158.17),
}


# --- BAH BY ZIP CODE (reads the DoD "All BAH Rates" ASCII files if they are in the repo) ---
# Get them once a year: travel.dod.mil > Allowances > Basic Allowance for Housing > BAH Rate Lookup >
# "All BAH Rates" > File Type ASCII > Year > Download. Unzip and upload the three .txt files
# (with-dependents rates, without-dependents rates, ZIP-to-area list) to the GitHub repository.
BAH_GRADES = ["E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "W01", "W02", "W03", "W04", "W05",
              "O01E", "O02E", "O03E", "O01", "O02", "O03", "O04", "O05", "O06", "O07"]
BAH_FILE_PATTERNS = {
    "with": re.compile(r"^bahw(\d\d)\.txt$", re.I),
    "without": re.compile(r"^bahwo(\d\d)\.txt$", re.I),
    "zip": re.compile(r"^sorted_zipmha(\d\d)\.txt$", re.I),
    "names": re.compile(r"^mhanames(\d\d)?\.txt$", re.I),
}
MHA_RE = re.compile(r"^[A-Z]{2}\d{3}$")


def _bah_file_paths():
    """Newest set of DoD BAH files sitting next to app.py, by two-digit year."""
    folder = os.path.dirname(os.path.abspath(__file__))
    found = {}

    try:
        names = os.listdir(folder)
    except Exception:
        return {}

    for name in names:
        for key, pattern in BAH_FILE_PATTERNS.items():
            match = pattern.match(name)

            if match:
                year = match.group(1) or "00"
                found.setdefault(key, []).append((year, os.path.join(folder, name)))

    if "with" not in found or "without" not in found or "zip" not in found:
        return {}

    year = max(y for y, _ in found["with"])
    picked = {"year": "20" + year}

    for key in ("with", "without", "zip", "names"):
        candidates = [p for y, p in found.get(key, []) if y == year] or [p for _, p in found.get(key, [])]

        if candidates:
            picked[key] = sorted(candidates)[-1]

    return picked


def _parse_bah_rates(path):
    rates = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in re.split(r"[,\s]+", line.strip()) if p.strip()]

            if len(parts) < 25 or not MHA_RE.match(parts[0].upper()):
                continue

            numbers = []

            for token in parts[1:]:
                try:
                    numbers.append(float(token.replace("$", "")))
                except ValueError:
                    break

            if len(numbers) >= len(BAH_GRADES):
                rates[parts[0].upper()] = dict(zip(BAH_GRADES, numbers[:len(BAH_GRADES)]))

    return rates


def _parse_zip_mha(path):
    zips = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in re.split(r"[,\s]+", line.strip()) if p.strip()]
            zip_code = next((p for p in parts if re.fullmatch(r"\d{5}", p)), None)
            mha = next((p.upper() for p in parts if MHA_RE.match(p.upper())), None)

            if zip_code and mha:
                zips[zip_code] = mha

    return zips


def _parse_mha_names(path):
    names = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]

            if len(parts) >= 2 and MHA_RE.match(parts[0].upper()):
                names[parts[0].upper()] = ", ".join(p for p in parts[1:] if p)

    return names


@st.cache_data(ttl=86400, show_spinner=False)
def load_bah_tables():
    """None when the DoD files are not in the repo; otherwise the parsed tables."""
    paths = _bah_file_paths()

    if not paths:
        return None

    try:
        tables = {
            "year": paths["year"],
            "with": _parse_bah_rates(paths["with"]),
            "without": _parse_bah_rates(paths["without"]),
            "zip": _parse_zip_mha(paths["zip"]),
            "names": _parse_mha_names(paths["names"]) if paths.get("names") else {},
        }
    except Exception:
        return None

    if not tables["with"] or not tables["zip"]:
        return None

    return tables


def bah_grade_code(pay_grade, prior_enlisted=False):
    """'O-2' -> 'O02' (or 'O02E' for a prior-enlisted officer in O-1 to O-3)."""
    code = pay_grade.replace("-", "").upper()

    if len(code) == 2:
        code = code[0] + "0" + code[1]

    if prior_enlisted and code in ("O01", "O02", "O03"):
        code += "E"

    return code


def bah_lookup(zip_code, pay_grade, with_dependents, prior_enlisted=False):
    """Monthly BAH for a ZIP, grade and dependency status from the DoD tables. None if unavailable."""
    tables = load_bah_tables()

    if not tables:
        return None

    zip_code = "".join(ch for ch in str(zip_code) if ch.isdigit())[:5].zfill(5)
    mha = tables["zip"].get(zip_code)

    if not mha:
        return {"error": f"ZIP {zip_code} is not in the DoD list"}

    table = tables["with"] if with_dependents else tables["without"]
    grade_code = bah_grade_code(pay_grade, prior_enlisted)
    rate = (table.get(mha) or {}).get(grade_code)

    if rate is None:
        return {"error": f"no {grade_code} rate for area {mha}"}

    return {"rate": rate, "mha": mha, "name": tables["names"].get(mha, mha), "year": tables["year"], "grade": grade_code}


def todays_dollars(amount, years, inflation):
    """What a future amount buys in today's money at a steady inflation rate."""
    return amount / ((1 + inflation) ** years) if years > 0 else amount


def basic_pay_2026(grade, years):
    row = PAY_2026.get(grade)

    if not row:
        return 0.0

    index = 0

    for i, threshold in enumerate(YOS_COLS):
        if years >= threshold:
            index = i

    return row[index]


def federal_tax_2026(taxable_income, status):
    """Tax from the 2026 brackets on income after the standard deduction; also returns the marginal rate."""
    remaining = max(float(taxable_income), 0.0)
    tax = 0.0
    lower = 0.0
    marginal = 0.10

    for upper, rate in BRACKETS_2026[status]:
        if remaining <= 0:
            break

        slice_amount = min(remaining, upper - lower)
        tax += slice_amount * rate
        marginal = rate
        remaining -= slice_amount
        lower = upper

    return tax, marginal


def military_pay_summary(base, bah, bas, other_taxable, roth_pct, trad_pct, status, state_pct):
    """Monthly figures. Allowances (BAH, BAS) are tax-free. Agency TSP is 1% automatic plus a match
    of 100% on the first 3% you put in and 50% on the next 2% (max 4%), always traditional."""
    contrib_pct = roth_pct + trad_pct
    match = min(contrib_pct, 3) * 1.0 + max(min(contrib_pct, 5) - 3, 0) * 0.5
    agency = base * (1 + match) / 100
    employee = base * contrib_pct / 100
    trad = base * trad_pct / 100

    annual_wages = (base + other_taxable) * 12
    annual_taxable = max(annual_wages - trad * 12 - STD_DEDUCTION_2026[status], 0)
    fed_year, marginal = federal_tax_2026(annual_taxable, status)
    fed = fed_year / 12
    ss = min(annual_wages, SS_WAGE_BASE_2026) * 0.062 / 12
    medicare = annual_wages * 0.0145 / 12
    state = max(annual_wages - trad * 12, 0) * state_pct / 100 / 12
    allowances = bah + bas
    gross_cash = base + other_taxable + allowances
    take_home = gross_cash - fed - ss - medicare - state - employee
    tax_advantage = allowances * marginal

    return {
        "base": base, "allowances": allowances, "other": other_taxable, "gross_cash": gross_cash,
        "taxable_wages": base + other_taxable - trad, "fed": fed, "ss": ss, "medicare": medicare, "state": state,
        "employee_tsp": employee, "agency_tsp": agency, "take_home": take_home, "marginal": marginal,
        "tax_advantage": tax_advantage, "total_comp": gross_cash + agency + tax_advantage,
        "annual_taxable": annual_taxable, "match_pct": match,
    }


def va_combined_rating(ratings):
    """VA 'whole person' math: each rating applies to what is left, then round to the nearest 10."""
    ratings = sorted([int(r) for r in ratings if 0 < int(r) <= 100], reverse=True)

    if not ratings:
        return 0, 0.0

    remaining = 100.0

    for rating in ratings:
        remaining -= remaining * rating / 100

    exact = 100 - remaining
    rounded = int(math.floor(exact / 10.0 + 0.5)) * 10   # VA rounds 5s up (Python's round() would not)

    return min(rounded, 100), exact


def va_monthly_2026(rating, with_spouse):
    rates = VA_RATES_2026.get(rating)

    if not rates:
        return 0.0

    return rates[1] if with_spouse else rates[0]


def tsp_projection(monthly_employee, monthly_agency, years, annual_growth):
    """End balance after `years` with monthly contributions and a steady growth rate."""
    r = annual_growth / 12
    n = years * 12
    total = monthly_employee + monthly_agency

    if abs(r) < 1e-12:
        balance = total * n
    else:
        balance = total * (((1 + r) ** n - 1) / r)

    split = monthly_employee / total if total > 0 else 0

    return balance, balance * split, balance * (1 - split)


# --- DATA HEALTH (runs on every load; cheap checks on cached data) ---
def data_health(tickers, deep=False):
    checks = []

    try:
        today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    except Exception:
        today = pd.Timestamp.now().normalize()

    # 1. Prices
    try:
        pulse_df, _ = get_pulse(tuple(tickers))
    except Exception:
        pulse_df = pd.DataFrame()

    if pulse_df.empty or "As of" not in pulse_df.columns:
        checks.append(("Prices (Yahoo)", "Problem", "No price data came back. Press 'Refresh now' on Insights, then 'Clear all cached data' if it persists."))
    else:
        latest = pd.to_datetime(pulse_df["As of"]).max()
        age = (today - latest.normalize()).days
        status = "OK" if age <= 5 else "Check"
        checks.append(("Prices (Yahoo)", status, f"Last close {latest.date()} ({age} days ago). Weekends and holidays are normal; more than 5 days is not."))

    # 2. Holdings snapshot age and completeness
    bundled = load_bundled_holdings()
    asof_dates = []

    for ticker in tickers:
        entry = bundled.get(ticker) or {}

        try:
            asof_dates.append(pd.to_datetime(entry.get("asof")))
        except Exception:
            pass

    asof_dates = [d for d in asof_dates if pd.notnull(d)]

    if asof_dates:
        oldest = min(asof_dates)
        age = (today - oldest.normalize()).days
        checks.append(("Holdings snapshot age", "OK" if age <= 45 else "Check",
                        f"Oldest fund list is as of {oldest.date()} ({age} days). Fidelity publishes monthly; the Monday job picks it up."))

    for ticker in tickers:
        df, source = get_holdings(ticker)
        count = len(df)
        completeness = holdings_completeness(ticker, count)
        expected = EXPECTED_HOLDINGS.get(ticker)

        if df.empty:
            checks.append((f"{ticker} holdings", "Problem", "No holdings loaded. Press 'Clear all cached data'."))
        elif source == "single stock":
            checks.append((f"{ticker} holdings", "OK", "A single stock, counted as one holding at 100%."))
        elif "stand-in" in source:
            checks.append((f"{ticker} holdings", "Check", f"{count} rows, {source}. Fine until the fund publishes its own list."))
        elif completeness == "partial" or "top 10" in source or "partial" in source:
            checks.append((f"{ticker} holdings", "Problem",
                            f"Only {count} rows (fund holds about {expected or '?'}). Source: {source}. Run 'Refresh ETF holdings' on GitHub; if it stays short, tell Claude."))
        else:
            checks.append((f"{ticker} holdings", "OK", f"{count} rows, {source}."))

    # 3. History extension and expense ratio (only once the stats are already computed)
    if deep:
        for ticker in tickers:
            stats = get_full_stats(ticker)

            if not stats:
                checks.append((f"{ticker} history", "Problem", "No price history. Yahoo may be rate-limiting; press 'Clear all cached data'."))
                continue

            if ticker in EXTEND_CHAIN and "hist:" not in str(stats.get("Hist Notes", "")):
                checks.append((f"{ticker} history", "Check", f"Long-period history did not extend past {stats['Hist From']}. Press 'Clear all cached data' to retry."))
            else:
                checks.append((f"{ticker} history", "OK", f"Usable history from {stats['Hist From']}."))

            if stats.get("Expense") is None:
                checks.append((f"{ticker} expense ratio", "Check", "Unknown right now. It is looked up again every 10 minutes; if it stays blank, ask Claude to add the verified figure."))

    # 3b. Sector labels for the blend
    try:
        blend, _ = build_blended_holdings(tickers, current_weights(tickers), sector_lookups=0)

        if not blend.empty and float(blend["Weight"].sum()) > 0:
            unlabelled = float(blend.loc[blend["Sector"] == UNCLASSIFIED, "Weight"].sum()) / float(blend["Weight"].sum())
            checks.append((
                "Sector labels", "OK" if unlabelled <= 0.02 else "Check",
                f"{1 - unlabelled:.1%} of the blend is labelled straight from the sector funds' baskets "
                f"({len(sector_map()):,} stocks on file). The X-Ray asks Yahoo about the heaviest of the rest."
            ))
    except Exception:
        pass

    # 4. Shows feeds
    try:
        _, show_status, shows_time = get_show_items(tuple(SHOW_FEEDS))
        bad = [name for name, state in show_status.items() if "unavailable" in state]
        checks.append(("Shows & Voices feeds", "Check" if bad else "OK",
                        ("Not loading: " + ", ".join(bad)) if bad else f"All {len(show_status)} feeds loaded at {shows_time.astimezone().strftime('%H:%M')}."))
    except Exception:
        checks.append(("Shows & Voices feeds", "Check", "Feed check did not run."))

    # 5. BAH tables for the Military tab
    tables = load_bah_tables()

    if tables:
        checks.append(("BAH rate tables (Military tab)", "OK" if tables["year"] == str(today.year) else "Check",
                        f"DoD {tables['year']} files loaded: {len(tables['with'])} housing areas, {len(tables['zip']):,} ZIP codes."
                        + ("" if tables["year"] == str(today.year) else " A newer year is out; upload the new files.")))
    else:
        checks.append(("BAH rate tables (Military tab)", "Check",
                        "Not uploaded yet, so the ZIP lookup is off and BAH must be typed. See the tracker document for the 2-minute upload."))

    # 6. Hand-entered facts and the day each set was last checked. These flip to "Check" on their own
    #    once they are due, so an old figure cannot sit here looking fine.
    fees_age = (today - pd.to_datetime(FEES_VERIFIED)).days
    checks.append((
        "Fund fees (checked " + FEES_VERIFIED + ")", "OK" if fees_age <= 365 else "Check",
        "QNDX 0.10%, VGT 0.09%, SPMO 0.13%, QQQ 0.18%, FTEC 0.084%, MGK 0.05%, SPYM 0.02%, VOO 0.03%. "
        "SMH, QQQM and VTI were last checked Sep 5, 2026. Any other fund's fee comes from Yahoo."
        + (" More than a year old; ask Claude to re-check." if fees_age > 365 else "")
    ))

    verified = pd.to_datetime(STATIC_FACTS_VERIFIED)
    yearly_due = today.year > verified.year and (today.month, today.day) >= (1, 15)
    checks.append((
        "Tax, pay and VA tables (verified " + STATIC_FACTS_VERIFIED + ")", "Check" if yearly_due else "OK",
        f"These are {verified.year} figures and the {today.year} ones are out. Ask Claude to update them."
        if yearly_due else
        "Tax rules, military pay tables, BAS, the TSP limit and VA rates for 2026. Due for a re-check every January; ask Claude."
    ))

    return checks


# --- SHOWS & VOICES (podcast RSS feeds + YouTube channels) ---
# To add a show: tell Claude the show name or paste its YouTube link and you get one line to paste here.
#   kind "rss"     = a podcast or news feed address
#   kind "youtube" = a YouTube channel handle (the @name from the channel's page)
SHOW_FEEDS = [
    ("CNBC Top News", "rss", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Television (YouTube)", "youtube", "@CNBCtelevision"),
    ("The Compound and Friends (podcast)", "rss", "https://feeds.megaphone.fm/TCP4771071679"),
    ("The Compound (YouTube)", "youtube", "@TheCompoundNews"),
    ("The Real Eisman Playbook (podcast)", "rss", "https://rss.libsyn.com/shows/633424/destinations/5592589.xml"),
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


def render_health(tickers):
    """Top-of-page panel: what is fresh, what is stale, what is hand-entered and when it was verified."""
    deep = bool(st.session_state.get("xray_go") or st.session_state.get("bench_go") or st.session_state.get("cmp_go"))
    checks = data_health(tickers, deep=deep)
    flagged = [c for c in checks if c[1] != "OK"]
    stamp = datetime.now().strftime("%H:%M")
    label = "🩺 Data health: all clear" if not flagged else f"🩺 Data health: {len(flagged)} item(s) to look at"

    with st.expander(f"{label} (checked {stamp})", expanded=bool([c for c in flagged if c[1] == "Problem"])):
        st.dataframe(pd.DataFrame(checks, columns=["Item", "Status", "Detail"]), hide_index=True)

        if not deep:
            cap("History and expense checks run once you press Analyze, Compare or Compare in detail.")

        cap(
            "Runs on every load. Live data (prices, holdings, headlines, shows) is checked for age and completeness. "
            "Rules and rates that live in text (tax law, pay tables, VA rates) cannot be pulled from a feed, so the "
            "panel shows when they were verified and when they are due."
        )


def military_tab():
    st.header("🎖️ Military: pay, TSP, retirement, separation, VA")
    cap(
        f"2026 figures: DFAS pay table (3.8% raise), BAS ${BAS_2026['Officer']:.2f} officer, TSP limit "
        f"${TSP_LIMIT_2026:,}, IRS 2026 brackets and standard deduction, VA rates effective Dec 1, 2025. "
        f"Verified {STATIC_FACTS_VERIFIED}. Copy exact numbers from your LES; nothing typed here is saved."
    )

    # ---- 1. Pay and taxes now
    md("### 1. My pay and taxes now")
    a1, a2, a3 = st.columns(3)

    with a1:
        grade = st.selectbox("Pay grade:", list(PAY_2026.keys()), index=1, key="mil_grade")
        yos = st.number_input("Years of service (LES 'YRS SVC'):", min_value=0, max_value=30, value=4, step=1, key="mil_yos")
        status = st.selectbox("Filing status:", list(STD_DEDUCTION_2026.keys()), index=0, key="mil_status")

    with a2:
        zip_code = st.text_input("Duty station ZIP code (for BAH):", "40121", key="mil_zip",
                                 help="Fort Knox is 40121. BAH is set by the duty station's area, not where you rent.")
        dependents = st.selectbox("Dependents:", ["With dependents", "Without dependents"], index=0, key="mil_dep")
        prior_enlisted = st.checkbox("Prior enlisted (O-1E/O-2E/O-3E rate)", value=False, key="mil_prior")
        bah_override = st.number_input("BAH override per month ($; 0 = use the ZIP lookup):", min_value=0, value=0, step=50, key="mil_bah",
                                       help="Type your LES figure here if it differs (rate protection after a move, for example).")
        bas = st.number_input("BAS per month ($):", min_value=0.0, value=float(BAS_2026["Officer"]), step=1.0, key="mil_bas")
        other = st.number_input("Other taxable pay per month ($, special pays, bonuses):", min_value=0, value=0, step=50, key="mil_other")

    looked_up = bah_lookup(zip_code, grade, dependents == "With dependents", prior_enlisted)

    if bah_override > 0:
        bah = float(bah_override)
        bah_note = f"BAH: using your override of ${bah:,.0f} a month."
    elif looked_up and "rate" in looked_up:
        bah = float(looked_up["rate"])
        bah_note = (f"BAH: ${bah:,.0f} a month for {looked_up['grade']} {dependents.lower()} in area {looked_up['mha']} "
                    f"({looked_up['name']}), DoD {looked_up['year']} table.")
    elif looked_up and "error" in looked_up:
        bah = 0.0
        bah_note = f"BAH lookup: {looked_up['error']}. Type your LES figure in the override box."
    else:
        bah = 0.0
        bah_note = ("BAH lookup is off because the DoD rate files are not in the repository yet. Type your LES figure in the "
                    "override box for now; the tracker document explains the 2-minute upload that turns the ZIP lookup on.")

    with a3:
        roth_pct = st.number_input("Roth TSP contribution (% of base pay):", min_value=0, max_value=100, value=5, step=1, key="mil_roth")
        trad_pct = st.number_input("Traditional TSP contribution (% of base pay):", min_value=0, max_value=100, value=0, step=1, key="mil_trad")
        state_pct = st.number_input("State income tax rate (%; 0 for Texas):", min_value=0.0, max_value=15.0, value=0.0, step=0.5, key="mil_state")

    base = basic_pay_2026(grade, yos)
    pay = military_pay_summary(base, float(bah), float(bas), float(other), roth_pct, trad_pct, status, state_pct)

    cap(f"2026 basic pay for {grade} at {yos} years: ${base:,.2f} a month. {bah_note} The LES is the truth; this table is the reference. "
        "Everything in this section is today's dollars: it is what the money is worth now.")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Cash per month (before tax)", f"${pay['gross_cash']:,.0f}")
    p2.metric("Take-home per month", f"${pay['take_home']:,.0f}", help="After federal tax, Social Security, Medicare, state tax and your TSP contribution.")
    p3.metric("Agency TSP per month", f"${pay['agency_tsp']:,.0f}", help="1% automatic plus the match, always traditional (pre-tax) even when your own money is Roth.")
    p4.metric("Total compensation per month", f"${pay['total_comp']:,.0f}", help="Cash plus agency TSP plus what the tax-free allowances would have cost you in federal tax. Health care (Tricare) is on top and not counted.")

    st.dataframe(pd.DataFrame([
        {"Line": "Basic pay (taxable)", "Per month": f"${pay['base']:,.2f}", "Per year": f"${pay['base'] * 12:,.0f}"},
        {"Line": "Other taxable pay", "Per month": f"${pay['other']:,.2f}", "Per year": f"${pay['other'] * 12:,.0f}"},
        {"Line": "BAH + BAS (tax-free)", "Per month": f"${pay['allowances']:,.2f}", "Per year": f"${pay['allowances'] * 12:,.0f}"},
        {"Line": "Taxable wages after traditional TSP", "Per month": f"${pay['taxable_wages']:,.2f}", "Per year": f"${pay['taxable_wages'] * 12:,.0f}"},
        {"Line": f"Federal income tax (marginal {pay['marginal']:.0%})", "Per month": f"-${pay['fed']:,.2f}", "Per year": f"-${pay['fed'] * 12:,.0f}"},
        {"Line": "Social Security 6.2% + Medicare 1.45%", "Per month": f"-${pay['ss'] + pay['medicare']:,.2f}", "Per year": f"-${(pay['ss'] + pay['medicare']) * 12:,.0f}"},
        {"Line": "State income tax", "Per month": f"-${pay['state']:,.2f}", "Per year": f"-${pay['state'] * 12:,.0f}"},
        {"Line": f"Your TSP ({roth_pct}% Roth + {trad_pct}% traditional)", "Per month": f"-${pay['employee_tsp']:,.2f}", "Per year": f"-${pay['employee_tsp'] * 12:,.0f}"},
        {"Line": "Take-home", "Per month": f"${pay['take_home']:,.2f}", "Per year": f"${pay['take_home'] * 12:,.0f}"},
        {"Line": "Agency TSP (not in your check)", "Per month": f"${pay['agency_tsp']:,.2f}", "Per year": f"${pay['agency_tsp'] * 12:,.0f}"},
    ]), hide_index=True)
    cap(
        f"Federal tax uses the 2026 standard deduction (${STD_DEDUCTION_2026[status]:,}) and brackets with only this pay as income. "
        "BAH and BAS never show up in W-2 box 1. Roth TSP money is taxed now and never again; traditional TSP is the opposite. "
        f"A deployment to a designated combat zone makes basic pay tax-free too (not the case for most European rotations)."
    )

    # ---- 2. TSP
    md("### 2. TSP: where 5% Roth in the C Fund ends up")
    cap(
        "The C Fund tracks the S&P 500, so the growth assumption starts at the S&P 500's own 10-year rate from this app's "
        "index-extended history (VOO). Agency money vests after 2 years of service for the automatic 1%; the match is yours immediately."
    )
    md(
        "**Where the money lands.** Your 5% goes into your Roth balance (taxed now, never again). The agency's 1% automatic and "
        "4% match are deposited into your traditional balance even though you elected Roth; the TSP has not turned on Roth "
        "treatment of agency money as of 2026, so that part is taxed when you withdraw it. You can see the split on tsp.gov under "
        "My Account, where Traditional and Roth balances are listed separately; the agency dollars sit under Traditional."
    )
    voo = get_full_stats("VOO")
    default_growth = (voo or {}).get("10Y CAGR") or 0.10
    t1, t2 = st.columns(2)

    with t1:
        tsp_growth = st.number_input("Assumed C Fund growth per year (%):", min_value=0.0, max_value=25.0,
                                     value=float(min(max(round(default_growth * 100, 1), 0.0), 25.0)), step=0.5, key="mil_tsp_growth")
        tsp_now = st.number_input("TSP balance today ($, optional):", min_value=0, value=0, step=1000, key="mil_tsp_now")

    with t2:
        years_more = st.number_input("Years you will keep contributing:", min_value=1, max_value=40, value=16, step=1, key="mil_tsp_years")
        inflation = st.number_input("Inflation assumption for 'today's dollars' (% per year):", min_value=0.0, max_value=10.0,
                                    value=2.5, step=0.5, key="mil_inflation",
                                    help="Used everywhere a future amount is shown in today's dollars. 2.5% is close to the long-run average.")

    rows = []

    for label, emp_pct in [(f"Your plan ({roth_pct + trad_pct}%)", roth_pct + trad_pct), ("If you did 10%", 10), ("If you did 15%", 15)]:
        emp = base * emp_pct / 100
        match = min(emp_pct, 3) * 1.0 + max(min(emp_pct, 5) - 3, 0) * 0.5
        agency = base * (1 + match) / 100
        end, yours, theirs = tsp_projection(emp, agency, int(years_more), tsp_growth / 100)
        end += tsp_now * (1 + tsp_growth / 100) ** years_more
        rows.append({
            "Contribution": label,
            "You put in / month": f"${emp:,.0f}",
            "Agency adds / month (traditional)": f"${agency:,.0f}",
            f"Balance after {years_more} yrs (future dollars)": f"${end:,.0f}",
            "Same balance in today's dollars": f"${todays_dollars(end, int(years_more), inflation / 100):,.0f}",
            "Of which agency money (future dollars)": f"${theirs:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True)
    cap(
        f"Held at today's base pay (promotions and raises make the real number bigger). 2026 limit on your own contributions: "
        f"${TSP_LIMIT_2026:,}. 5% is the floor that captures the full match; every dollar above it is unmatched but still Roth. "
        f"'Today's dollars' divides the future balance by {inflation:.1f}% inflation compounded, so it is comparable to prices you know now."
    )

    # ---- 3. Retirement at 20
    md("### 3. If you stay to 20 years (Blended Retirement System)")
    r1, r2 = st.columns(2)

    with r1:
        ret_grade = st.selectbox("Grade at retirement:", ["O-4", "O-5", "O-6"], index=1, key="mil_ret_grade")
        ret_years = st.number_input("Years of service at retirement:", min_value=20, max_value=30, value=20, step=1, key="mil_ret_years")

    with r2:
        high3_default = (basic_pay_2026(ret_grade, 18) + basic_pay_2026(ret_grade, 20)) / 2
        high3 = st.number_input("Average monthly base pay over your last 36 months ($):", min_value=0.0,
                                value=float(round(high3_default, 2)), step=100.0, key="mil_high3",
                                help="Starts at the 2026 pay for that grade averaged across years 18-20. Future raises will push it up.")

    pension_month = 0.02 * ret_years * high3
    pension_fed, pension_marginal = federal_tax_2026(max(pension_month * 12 - STD_DEDUCTION_2026[status], 0), status)
    q1, q2, q3 = st.columns(3)
    q1.metric("Pension per month (today's dollars)", f"${pension_month:,.0f}",
              help="Built from the 2026 pay table, so it is in today's money. The real check will be larger in future dollars because pay tables and the yearly cost-of-living increase both track inflation.")
    q2.metric("Per year", f"${pension_month * 12:,.0f}")
    q3.metric("Federal tax if it were your only income", f"${pension_fed:,.0f} / yr")
    md(
        f"Blended Retirement System math: 2.0% x {ret_years} years x ${high3:,.0f} = **${pension_month:,.0f} a month**, "
        "paid for life with yearly cost-of-living increases, starting the day after retirement. It is taxed as ordinary income "
        "federally (no Social Security tax), and not at all in Texas. Retirees keep Tricare (small enrollment fees), the "
        "TSP with the agency money on top, and continuation pay along the way (between years 8 and 12 the Army pays at least "
        "2.5 months of base pay for a 3-year commitment). At retirement you can also take 25% or 50% of the pension as a lump "
        "sum in exchange for a smaller check until age 67; on the numbers that is almost always a bad trade."
    )
    md(
        "**Pension plus VA disability:** with 20 years and a VA rating of 50% or higher you receive both in full (CRDP). "
        "Below 50% the VA amount is subtracted from the pension, but that portion becomes tax-free, so it still helps. "
        "Combat-related conditions can qualify for CRSC at any rating."
    )

    # ---- 4. Separation before 20
    md("### 4. If you leave before 20 years")
    isp_full = 0.10 * yos * base * 12
    md(
        f"All figures here are today's dollars. No pension. What you keep: the TSP (your money always; the agency money once vested), the Roth TSP can stay put or roll "
        "to a Roth IRA with no tax, and every VA benefit below. If the separation is involuntary and honorable with 6 or more "
        f"years of service, involuntary separation pay is 10% x years x annual base pay (about **${isp_full:,.0f}** at {grade} "
        f"with {yos} years, taxed as income, and later recouped dollar-for-dollar from any VA disability pay until repaid; "
        "the half rate is 5%). Unemployment insurance for ex-service members (UCX) is paid by the state you live in."
    )
    st.dataframe(pd.DataFrame([
        {"Benefit": "Post-9/11 GI Bill", "What it is": "36 months of tuition at public-school rates (full after 36 months of service), a monthly housing allowance at the E-5-with-dependents BAH rate of the school's ZIP, and $1,000 a year for books.", "Watch-out": "Transferring it to a spouse or child must be done WHILE STILL SERVING (needs 6 years served and a 4-year further commitment)."},
        {"Benefit": "Texas Hazlewood Act", "What it is": "Up to 150 credit hours tuition-free at Texas public colleges for veterans who entered service in Texas or had Texas as home of record, honorably discharged, 181+ days active duty.", "Watch-out": "Applies when federal GI Bill benefits are used up or do not cover the tuition for that term."},
        {"Benefit": "VA home loan", "What it is": "No down payment, no mortgage insurance; funding fee 2.15% on a first use with nothing down.", "Watch-out": "The funding fee is waived entirely with a VA rating of 10% or more."},
        {"Benefit": "VR&E (Chapter 31)", "What it is": "Retraining, degree or certification paid in full with a monthly stipend, for a service-connected rating of 20% or more (10% with a serious employment handicap).", "Watch-out": "Separate from the GI Bill and can be used first, saving the GI Bill."},
        {"Benefit": "VA health care", "What it is": "Enrollment priority rises with rating; 50%+ means no copays for anything.", "Watch-out": "Apply right after separation; Tricare ends (TAMP gives 180 days of Tricare only for involuntary separations and some other cases)."},
        {"Benefit": "Texas property tax", "What it is": "Homestead exemption of $5,000 to $12,000 by rating; a 100% rating (or unemployability) exempts the whole homestead.", "Watch-out": "Applies to the Texas home you live in, not a rental."},
        {"Benefit": "SGLI to VGLI", "What it is": "The $500k SGLI life insurance ends 120 days after separation; VGLI continues it.", "Watch-out": "Apply within 240 days for no health questions; the deadline is 1 year and 120 days."},
        {"Benefit": "SkillBridge and TAP", "What it is": "Up to the last 180 days of service working for a civilian employer while still paid; the Transition Assistance Program is required starting a year out.", "Watch-out": "Needs command approval; apply early."},
        {"Benefit": "Reserve or National Guard", "What it is": "Keep accruing points toward a reserve retirement paid from about age 60, keep Tricare Reserve Select, keep the TSP match.", "Watch-out": "Reserve retirement is smaller and starts later, but it is not zero."},
    ]), hide_index=True)

    # ---- 5. VA disability
    md("### 5. VA disability: combined rating and monthly pay")
    v1, v2 = st.columns(2)

    with v1:
        ratings_text = st.text_input("Ratings for each condition (%):", "30, 10, 10", key="mil_va_ratings",
                                     help="VA does not add them. Each one takes its share of what is left, then it rounds to the nearest 10.")

    with v2:
        with_spouse = st.checkbox("Married (adds spouse pay at 30%+)", value=False, key="mil_va_spouse")

    try:
        ratings = [int(float(x)) for x in ratings_text.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        ratings = []

    combined, exact = va_combined_rating(ratings)
    monthly = va_monthly_2026(combined, with_spouse)
    w1, w2, w3 = st.columns(3)
    w1.metric("Combined rating", f"{combined}%", help=f"Exact before rounding: {exact:.1f}%")
    w2.metric("Per month (tax-free, today's dollars)", f"${monthly:,.2f}")
    w3.metric("Per year (tax-free, today's dollars)", f"${monthly * 12:,.0f}", help="VA rates get a cost-of-living increase every December, so the real amount keeps pace with prices.")
    st.dataframe(pd.DataFrame([
        {"Rating": f"{r}%", "Veteran alone": f"${a:,.2f}", "With spouse": f"${s:,.2f}"} for r, (a, s) in VA_RATES_2026.items()
    ]), hide_index=True)
    cap(
        "Monthly rates effective Dec 1, 2025 (2.8% cost-of-living increase); children and dependent parents add more from 30% up. "
        "VA pay is tax-free at every level, stacks on top of civilian pay, and never reduces Social Security. "
        "File a Benefits Delivery at Discharge claim 180 to 90 days before separation so the rating decision arrives with the DD-214."
    )


def show_chart(fig):
    """Full-width chart on both new and old Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def _supports(fn, *names):
    """True when this Streamlit version's function accepts every named option."""
    try:
        params = inspect.signature(fn).parameters
        return all(name in params for name in names)
    except Exception:
        return False


_FLEX_TILES = _supports(st.container, "horizontal") and _supports(st.metric, "width", "border")


def metric_grid(items, per_row=5):
    """One group of number tiles. On a computer they sit side by side. On a phone they wrap two to a
    line instead of stacking one per line, which halves the scrolling. Older Streamlit versions fall
    back to plain columns, per_row to a line. Each item is (label, value, help text or None)."""
    if _FLEX_TILES:
        with st.container(horizontal=True):
            for label, value, help_text in items:
                st.metric(label, value, help=help_text, border=True, width=TILE_WIDTH)

        return

    for first in range(0, len(items), per_row):
        cols = st.columns(per_row)

        for col, (label, value, help_text) in zip(cols, items[first:first + per_row]):
            col.metric(label, value, help=help_text)


def full_height(rows, row_px=35, cap=None):
    """Table height in pixels that shows every row without an inner scroll bar."""
    height = int(row_px * (rows + 1) + 3)
    return min(height, cap) if cap else height


def render_xray_results(etfs, weights, bench, portfolio_value):
    """Everything about the blend on one page: performance, costs, sectors, top holdings, a
    scorecard against each fund and the benchmarks, growth, year-by-year returns, overlap and headlines."""
    held = [t for t in etfs if weights.get(t, 0) > 0]
    bench = [t for t in bench if t not in held]
    value = float(portfolio_value or 0)
    mix_text = ", ".join(f"{t} {weights[t]:.0%}" for t in held)

    with st.spinner("Loading prices and history for your blend. The first load of the day takes the longest."):
        stats = fresh_stats(held + bench)

    missing = [t for t in held if t not in stats]

    if missing:
        st.warning(
            f"No usable market data for: {', '.join(missing)}. This is usually temporary Yahoo "
            "rate-limiting - wait about a minute and press the button again. Until then the blend "
            "below is worked out from the rest."
            + (" For a share class, Yahoo wants a dash instead of a dot: BRK-B, not BRK.B."
               if any("." in t for t in missing) else "")
        )

    if not any(t in stats for t in held):
        return

    # ---------- 1. Performance ----------
    md(
        "Jump to: [Sectors](#sectors) · [Top 15](#top-15) · [All positions](#all-positions) · "
        "[Scorecard](#scorecard) · [Growth](#growth) · [Year by year](#year-by-year) · "
        "[Overlap](#overlap) · [Headlines](#headlines)"
    )
    st.subheader("📈 Blended Performance", anchor="performance")
    blended, excluded = calculate_blended_performance(weights, stats)
    tiles = [(PERF_LABELS[p], pct_or_na(blended.get(p)), PERF_HELP[p]) for p in PERF_PERIODS]
    metric_grid(tiles[:4], per_row=4)     # the recent moves: day, week, month, this year
    metric_grid(tiles[4:], per_row=6)     # the long view: 1 year, then 3 to 20 years per year

    dates = sorted({str(stats[t].get("As of")) for t in held if t in stats and stats[t].get("As of")})
    as_of_text = f"Prices as of {dates[-1]}" if dates else "Prices from Yahoo"

    if len(dates) > 1:
        lagging = [t for t in held if t in stats and str(stats[t].get("As of")) != dates[-1]]
        as_of_text += f" ({', '.join(lagging)} last priced earlier)"

    cap(
        f"{mix_text}. {as_of_text}, refreshed every 15 minutes while the market is open. "
        "\"/ yr\" is the average growth per year over that stretch. Long stretches use index-extended "
        "history (older funds and indexes on the same index, spliced in before each fund existed)."
    )

    notes = [f"{PERF_LABELS[period]} leaves out {', '.join(names)}" for period, names in excluded.items() if names]

    if notes:
        cap(
            "Where a fund is younger than the stretch it is left out and the figure is re-weighted "
            "across the rest: " + "; ".join(notes) + "."
        )

    # ---------- 2. Holdings behind the blend (needed by several sections) ----------
    with st.spinner("Opening each fund and adding up what is inside..."):
        full, source_counts = build_blended_holdings(etfs, weights)

    stocks = full[full["Sector"] != CASH_LABEL].reset_index(drop=True) if not full.empty else full

    # ---------- 3. At a glance ----------
    st.subheader("🧾 At a glance", anchor="at-a-glance")
    blend_row = weighted_portfolio_row([stats[t] for t in held if t in stats], weights, label="MY BLEND")
    blend_expense = blend_row.get("Expense")
    blend_yield = blend_row.get("Yield (TTM)")
    tiles = [
        ("Cost per year", fmt_expense(blend_expense), "The yearly fund fee on the whole blend, weighted by your percentages."),
        ("Dividend yield", pct_or_na(blend_yield), "Cash paid out over the last 12 months as a share of today's price, weighted by your percentages."),
    ]

    if not stocks.empty:
        top = stocks.iloc[0]
        tiles += [
            ("Stocks held", f"{len(stocks):,}", "How many different companies you own once the funds are opened up."),
            (f"Biggest: {top['Symbol']}", f"{top['Weight']:.2%}", "Your largest single company and its share of the blend."),
            ("Top 10 share", f"{stocks.head(10)['Weight'].sum():.2%}", "Your 10 largest companies added together."),
            ("Top 15 share", f"{stocks.head(15)['Weight'].sum():.2%}", "Your 15 largest companies added together."),
        ]

    metric_grid(tiles, per_row=6)

    fee_bits = [f"{t} {fmt_expense(stats[t].get('Expense'))}" for t in held if t in stats and stats[t].get("Expense") is not None]
    fee_missing = [t for t in held if t in stats and stats[t].get("Expense") is None]
    basis = value if value > 0 else 10000.0
    money_bits = []

    if blend_expense is not None and pd.notnull(blend_expense):
        money_bits.append(f"about ${basis * blend_expense:,.0f} a year in fund fees")

    if blend_yield is not None and pd.notnull(blend_yield):
        money_bits.append(f"about ${basis * blend_yield:,.0f} a year in dividends")

    line = "Fees by fund: " + " | ".join(fee_bits) + "." if fee_bits else ""

    if fee_missing:
        line += f" No fee on file for {', '.join(fee_missing)}, so the blended cost leaves it out."

    if money_bits:
        line += f" On ${basis:,.0f}: " + " and ".join(money_bits) + "."

    young = [t for t in held if t in stats and "yield/streak/freq via" in str(stats[t].get("Hist Notes", ""))]

    if young:
        line += (
            f" {', '.join(young)} is under a year old, so its yield uses the payouts of the older fund "
            "on the same index."
        )

    if value > 0:
        taxable_pct = float(st.session_state.get("tax_taxable_pct", 50))
        rates = tax_rates(
            st.session_state.get("tax_bracket", 22),
            bool(st.session_state.get("tax_niit", False)),
            float(st.session_state.get("tax_state", 0.0) or 0.0),
        )
        _rows, _payout, tax_total = blended_tax_estimate(held, weights, value, taxable_pct / 100, rates)
        line += (
            f" Estimated tax on those dividends: about ${tax_total:,.0f} a year, with {taxable_pct:.0f}% of the money "
            f"in a taxable account at {rates['qualified']:.0%} on qualified dividends (change both on the Taxes tab)."
        )

    if line:
        cap(line.strip())

    if full.empty:
        st.warning("Could not load holdings data for the selected tickers.")
        cap(
            "Both holdings sources are unreachable right now. Wait about a minute and press the button "
            "again - results are kept once loaded. Sectors and top holdings need this list."
        )
    else:
        # ---------- 4. Sectors ----------
        st.subheader("🧩 Sectors", anchor="sectors")
        shares = sector_shares(full)
        # Every column uses the labels the blend settled on, so a stock is never one sector in the
        # blend column and unlabelled in its own fund's column.
        labels = dict(zip(full["Symbol"], full["Sector"]))
        others = {t: fund_sector_shares(t, labels) for t in held + bench}
        real = sorted((s for s in shares if s not in (CASH_LABEL, UNCLASSIFIED)), key=lambda s: -shares[s])
        tail = [
            s for s in (UNCLASSIFIED, CASH_LABEL)
            if shares.get(s, 0) > 0.00005 or any(split.get(s, 0) > 0.00005 for split in others.values())
        ]
        seen = set(real) | set(tail)
        extra = sorted({s for split in others.values() for s in split if s not in seen and split[s] > 0.00005
                        and s not in (CASH_LABEL, UNCLASSIFIED)})
        order = real + extra + tail

        sector_rows = []

        for sector in order:
            row = {"Sector": sector, "MY BLEND": f"{shares.get(sector, 0):.2%}"}

            if value > 0:
                row["$ of mine"] = f"${shares.get(sector, 0) * value:,.0f}"

            for ticker in held + bench:
                row[ticker] = f"{others[ticker].get(sector, 0):.2%}" if others[ticker] else "-"

            sector_rows.append(row)

        total_row = {"Sector": "Total", "MY BLEND": f"{sum(shares.values()):.2%}"}

        if value > 0:
            total_row["$ of mine"] = f"${value:,.0f}"

        for ticker in held + bench:
            total_row[ticker] = f"{sum(others[ticker].values()):.2%}" if others[ticker] else "-"

        sector_rows.append(total_row)
        sector_df = pd.DataFrame(sector_rows)

        if real:
            lead = real[0]
            md(
                f"**{lead}** is **{shares[lead]:.1%}** of your blend. "
                + (f"The next two are {real[1]} at {shares[real[1]]:.1%} and {real[2]} at {shares[real[2]]:.1%}."
                   if len(real) >= 3 else "")
            )

        s1, s2 = st.columns([1, 1])

        with s1:
            st.dataframe(sector_df, hide_index=True, height=full_height(len(sector_df)))

        with s2:
            chart_df = pd.DataFrame({
                "Sector": order,
                "Share of my blend (%)": [round(shares.get(s, 0) * 100, 2) for s in order],
            })
            fig = px.bar(
                chart_df.iloc[::-1], x="Share of my blend (%)", y="Sector", orientation="h",
                text="Share of my blend (%)", title="My blend by sector",
            )
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
            fig.update_layout(
                height=max(320, 34 * len(chart_df) + 110), margin=dict(l=10, r=40, t=50, b=10),
                yaxis_title=None, xaxis_title=None,
            )
            show_chart(fig)

        unlabelled = shares.get(UNCLASSIFIED, 0)
        cap(
            "Worked out stock by stock from each fund's published holdings, then added up with your "
            "percentages. The other columns show each fund and each benchmark on its own, so you can see "
            "what each one adds. Sector names follow the standard system the index companies use, so a "
            "line can differ by a hair from Seeking Alpha. \"Cash and other\" is what a fund keeps in cash "
            "or does not list."
            + (f" {unlabelled:.2%} of the blend has no sector label (companies no U.S. sector fund holds)."
               if unlabelled > 0.00005 else "")
        )

        # ---------- 5. Top 15 holdings ----------
        st.subheader("🏆 Top 15 holdings", anchor="top-15")
        top15 = stocks.head(15)
        top10_share = stocks.head(10)["Weight"].sum()
        top15_share = top15["Weight"].sum()
        top25_share = stocks.head(25)["Weight"].sum()

        md(
            f"Your 15 largest holdings are **{top15_share:.2%}** of the blend. "
            f"The top 10 are {top10_share:.2%} and the top 25 are {top25_share:.2%}."
        )

        top_rows = []

        for rank, row in enumerate(top15.to_dict("records"), start=1):
            entry = {"#": str(rank), "Symbol": row["Symbol"], "Weight %": f"{row['Weight']:.2%}"}

            if value > 0:
                entry["$ of mine"] = f"${row['Weight'] * value:,.0f}"

            entry.update({"Company": row["Company"], "Sector": row["Sector"], "Held through": row["In funds"]})
            top_rows.append(entry)

        total_entry = {"#": "", "Symbol": "Total", "Weight %": f"{top15_share:.2%}"}

        if value > 0:
            total_entry["$ of mine"] = f"${top15_share * value:,.0f}"

        total_entry.update({"Company": f"Top {len(top15)} together", "Sector": "", "Held through": ""})
        top_rows.append(total_entry)
        st.dataframe(pd.DataFrame(top_rows), hide_index=True, height=full_height(len(top_rows)))
        cap(
            "Each stock's weight is added up across every fund that holds it. Alphabet's two share "
            "classes (GOOG and GOOGL) are counted together as one company, so this list can differ "
            "from one that shows them on two lines."
        )

        # ---------- 6. All positions ----------
        st.subheader("🗺️ All positions", anchor="all-positions")
        n_unique, n_two, n_all = holdings_overlap(held)
        overlap_text = (
            f" {n_two} of them sit in two or more of your funds and {n_all} sit in all {len(held)}."
            if len(held) > 1 else ""
        )
        cap(
            f"All {len(stocks):,} companies in the blend, weighted by your percentages." + overlap_text
            + " Sources - " + " | ".join(f"{ticker}: {info}" for ticker, info in source_counts.items())
        )

        table = full[["Symbol", "Weight %"]].copy()

        if value > 0:
            table["$ Exposure"] = (full["Weight"] * value).apply(lambda x: f"${x:,.0f}")

        table["Sector"] = full["Sector"]
        table["Industry"] = full["Industry"]

        c1, c2 = st.columns([2, 1])

        with c1:
            fig = px.treemap(
                stocks.head(40),
                path=[px.Constant("My blend"), "Sector", "Symbol"],
                values="Weight %",
                custom_data=["Industry"],
                title="Top 40 holdings, grouped by sector",
            )
            fig.update_traces(
                textinfo="label+value",
                texttemplate="%{label}<br>%{value:.2f}%",
                hovertemplate="%{label}<br>%{customdata[0]}<br>%{value:.2f}% of the blend<extra></extra>",
            )
            fig.update_layout(margin=dict(l=5, r=5, t=45, b=5))
            show_chart(fig)

        with c2:
            st.dataframe(table, height=500, hide_index=True)

        try:
            st.download_button(
                "⬇️ Download the full list (CSV)",
                data=full[["Symbol", "Company", "Sector", "Industry", "Weight %", "In funds"]].to_csv(index=False),
                file_name="blended_holdings.csv",
                mime="text/csv",
                key="xray_download",
            )
        except Exception:
            pass

    # ---------- 7. Scorecard ----------
    st.subheader("📊 Scorecard vs. benchmarks", anchor="scorecard")
    scorecard = blend_scorecard(held, weights, bench, stats, full, blended, excluded)
    st.dataframe(scorecard, hide_index=True, height=full_height(len(scorecard)))
    starred = sorted({name for names in excluded.values() for name in names})
    cap(
        "Facts run down the side; your blend, then each fund, then each benchmark run across. "
        "Swipe sideways on a phone. "
        + (f"* = {', '.join(starred)} is younger than that stretch, so the blend figure leaves it out and "
           "re-weights the rest. " if starred else "")
        + "The blend's worst drop is measured on the blend itself over the years every fund has data; "
        "each fund's own column reaches back further through its index. \"Qualified\" payouts are the "
        "dividends taxed at the lower long-term rates; the Taxes tab has the detail."
    )

    # ---------- 8. Growth of $10,000 ----------
    st.subheader("💰 Growth of $10,000", anchor="growth")
    g1, g2, g3 = st.columns([1, 1, 1])

    with g1:
        window_choice = st.selectbox(
            "How far back:", ["3Y", "5Y", "10Y", "15Y", "20Y", "As far back as the blend goes"],
            index=2, key="xray_window",
        )

    with g2:
        show_funds = st.checkbox("Draw each of my funds too", value=False, key="xray_show_funds")

    with g3:
        log_scale = st.checkbox("Log scale (fairer over long stretches)", value=False, key="xray_log")

    years_map = {"3Y": 3, "5Y": 5, "10Y": 10, "15Y": 15, "20Y": 20, "As far back as the blend goes": None}
    blend_weights = {t: weights[t] for t in held if t in stats}
    extra_lines = bench + ([t for t in held if t in stats] if show_funds and len(held) > 1 else [])
    growth_df, facts = blend_growth_frame(blend_weights, extra_lines, years_map[window_choice], stats)

    if growth_df.empty:
        st.info("Not enough shared price history to draw the blend as one line yet.")
    else:
        fig = px.line(
            growth_df, x="Date", y="Growth of $10K", color="Line",
            title=f"Growth of $10,000 from {facts['start'].date()} to {facts['end'].date()}",
        )
        fig.update_traces(line=dict(width=1.6))
        fig.update_traces(selector=dict(name="MY BLEND"), line=dict(width=3.2))
        fig.update_layout(legend_title_text=None, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None, xaxis_title=None)

        if log_scale:
            fig.update_yaxes(type="log")

        show_chart(fig)

        span = facts["years"] or 0
        end_value = facts["ends"].get("MY BLEND")
        blend_series_full, _limit, _limit_date = blend_series(blend_weights, stats)
        dd_value, dd_month = worst_drop(blend_series_full)
        growth_tiles = [("$10,000 became", f"${end_value:,.0f}", "MY BLEND over the stretch drawn above.")]

        if span >= 1 and end_value and end_value > 0:
            growth_tiles.append((
                "Per year", f"{(end_value / 10000) ** (1 / span) - 1:.2%}",
                "The steady yearly rate that turns $10,000 into that amount.",
            ))

        for name in bench:
            if name in facts["ends"]:
                growth_tiles.append((f"{name} became", f"${facts['ends'][name]:,.0f}", f"$10,000 in {name} over the same stretch."))

        if dd_value is not None:
            growth_tiles.append((
                "Worst drop", f"{dd_value:.1%}",
                f"The blend's deepest fall from a high, measured since {blend_series_full.index[0].date()}.",
            ))
            growth_tiles.append(("Bottom reached", dd_month, "The month that worst drop hit its low."))

        metric_grid(growth_tiles, per_row=6)

        text = (
            "MY BLEND is one line for the whole portfolio: your percentages, put back to target at the end of "
            "each calendar year, with dividends reinvested."
        )

        if facts.get("left_alone"):
            text += (
                f" Bought once and never put back, the same $10,000 would be ${facts['left_alone']:,.0f}. "
                "That bought-once method is what the tiles at the top use, which is why the two can differ a little."
            )

        if facts["shortened"] and facts.get("limiting"):
            text += (
                f" The chart starts on {facts['start'].date()} instead of going back the full {window_choice[:-1]} years "
                f"because {facts['limiting']} has no history before {facts['limit_date']}."
            )

        if dd_value is not None:
            text += f" The worst drop is measured from {blend_series_full.index[0].date()}; the bottom came in {dd_month}."

        text += " Dollar values are of their own time, not adjusted to today's dollars."
        cap(text)

        if facts["skipped"]:
            cap("Left off the chart because their history starts after the chart does: " + ", ".join(facts["skipped"]) + ".")

    # ---------- 9. Year by year ----------
    st.subheader("📅 Year by year", anchor="year-by-year")
    year_table, full_years, current_year = year_by_year(blend_weights, bench, stats)

    if year_table.empty:
        st.info("Yearly returns are not available right now.")
    else:
        if len(full_years) >= 2:
            best, worst = full_years.idxmax(), full_years.idxmin()
            up_years = int((full_years > 0).sum())
            md(
                f"In the {len(full_years)} full years every fund has data, your blend's best year was "
                f"**{int(best)} ({full_years[best]:+.1%})**, its worst was **{int(worst)} ({full_years[worst]:+.1%})**, "
                f"and it finished up in {up_years} of {len(full_years)}."
            )

        st.dataframe(year_table, hide_index=True, height=full_height(len(year_table)))
        cap(
            "Total return for each calendar year, dividends included. MY BLEND is your percentages at the start "
            "of each year. A dash means a fund was not around for that whole year, so the blend cannot be "
            "worked out for it. Fund columns before a fund existed use its index's older history."
        )

    # ---------- 10. Overlap ----------
    if len(held) > 1:
        st.subheader("🔁 How much my funds overlap", anchor="overlap")
        pairs = []

        for i, a in enumerate(held):
            for b in held[i + 1:]:
                share = holdings_overlap_pct(get_holdings(a)[0], get_holdings(b)[0])

                if share is not None:
                    pairs.append({"Pair": f"{a} and {b}", "Same stocks by weight": f"{share:.0%}"})

        if pairs:
            st.dataframe(pd.DataFrame(pairs), hide_index=True)
            cap(
                "Stock by stock, this adds up the smaller of the two funds' weights. 100% would mean two "
                "identical funds and 0% would mean nothing in common. High overlap is why funds move together."
            )

    # ---------- 11. Headlines ----------
    st.subheader("🗞️ Latest headlines", anchor="headlines")

    with st.expander("Show the newest headlines for these tickers"):
        try:
            news_df, news_time = get_live_news(tuple(held))
        except Exception:
            news_df, news_time = pd.DataFrame(), None

        if news_df is None or news_df.empty:
            cap("No headlines came back right now. The Insights & Updates tab has a Refresh button.")
        else:
            for _, item in news_df.head(8).iterrows():
                meta = " · ".join(x for x in [item["Ticker"], str(item["Source"]), relative_age(item["Time"])] if x)

                if item["Link"]:
                    md(f"**[{item['Title']}]({item['Link']})**  \n{meta}")
                else:
                    md(f"**{item['Title']}**  \n{meta}")

            cap("Yahoo Finance and Google News, refreshed every 15 minutes. The full feed is on the Insights & Updates tab.")


# --- UI ---
def main():
    render_health(parse_tickers(st.session_state.get("xray_tickers", DEFAULT_PORT)) or parse_tickers(DEFAULT_PORT))

    tab1, tab2, tab3, tab4, tab5, tab_cmp, tab_tax, tab_mil, tab6, tab7 = st.tabs([
        "🚀 X-Ray",
        "🆚 Benchmark",
        "📈 Dividends",
        "🔍 Deep Dive",
        "👀 Watchlist",
        "⚖️ Compare",
        "🧾 Taxes",
        "🎖️ Military",
        "📰 Insights & Updates",
        "🎧 Shows & Voices",
    ])

    # --- TAB 1: X-RAY (everything about the blend on one page) ---
    with tab1:
        st.header("Portfolio X-Ray")
        cap(
            "Type the funds and stocks to blend, set each percentage, then press the button. Everything "
            "about the blend shows on this one page: performance, cost, sectors, top holdings, a scorecard "
            "against the benchmarks, growth, yearly returns, overlap and headlines."
        )

        etfs = parse_tickers(st.text_input("ETFs and stocks to blend:", DEFAULT_PORT, key="xray_tickers"))

        if not etfs:
            st.info("Enter at least one ticker.")
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
                        step=1,
                        key=f"weight_{ticker}"
                    ) / 100.0

            total_w = sum(raw_weights.values())

            if total_w > 0:
                weights = {ticker: weight / total_w for ticker, weight in raw_weights.items()}

                if abs(total_w - 1.0) > 0.005:
                    cap(f"Percentages sum to {total_w * 100:.0f}% - normalized to 100% for the math below.")
            else:
                weights = raw_weights
                st.warning("Set at least one weight above 0.")

            b1, b2 = st.columns(2)

            with b1:
                bench = parse_tickers(
                    st.text_input("Compare against (benchmarks):", DEFAULT_BENCH, key="xray_bench")
                )

            with b2:
                portfolio_value = st.number_input(
                    "Portfolio value in $ (optional - adds dollar figures):",
                    min_value=0,
                    value=0,
                    step=1000,
                    key="xray_value"
                )

            if st.button("Analyze Blended Holdings", key="analyze_blended", type="primary"):
                st.session_state["xray_go"] = True

            if st.session_state.get("xray_go") and total_w > 0:
                render_xray_results(etfs, weights, bench[:4], portfolio_value)

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
                    cap(
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
                            cap(
                                f"Chart starts {start_used.date()}. Pre-inception eras use spliced "
                                "same-index fund / index data; price-only index eras exclude dividends. "
                                "Dollar values are of their own time, not adjusted to today's dollars."
                            )

                    if skipped:
                        cap(
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
                    cap(
                        "Div CAGR uses completed calendar years. Where an ETF is younger than the "
                        "window, growth is chained from older same-index funds (see Hist Notes). "
                        "Max Div CAGR covers the longest unbroken yearly run available. "
                        "MY PORTFOLIO is weighted by your X-Ray percentages."
                    )

                    md("### 💵 Payout history, per share")

                    for ticker in t_list:
                        detail = dividend_detail(ticker)

                        if not detail or not detail["rows"]:
                            continue

                        note = f" (showing {detail['source']}'s record, same index)" if detail["source"] != ticker else ""

                        with st.expander(f"{ticker}: last payout ${detail['last_amount']:.4f} on {detail['last_date']}{note}", expanded=len(t_list) <= 4):
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Last payout / share", f"${detail['last_amount']:.4f}")
                            m2.metric("Change vs. a year earlier", f"{detail['yoy']:+.1%}" if detail["yoy"] is not None else "-")
                            m3.metric("Last 12 months / share", f"${detail['ttm']:.4f}" if detail["ttm"] is not None else "-")
                            m4.metric("12-month total change", f"{detail['ttm_change']:+.1%}" if detail["ttm_change"] is not None else "-")
                            st.dataframe(pd.DataFrame(detail["rows"]), hide_index=True)
                            pay_note = (
                                f"Yahoo lists the pay date as {detail['pay_date_yahoo']}."
                                if detail.get("pay_date_yahoo")
                                else "Yahoo does not publish ETF pay dates; they usually fall 2 to 7 days after the ex-date (the issuer's site has the exact day)."
                            )
                            cap(f"Usual payout months: {detail['usual_months']}. {pay_note} Ex-date = the day you must already own shares to get that payout.")

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
                deep_stats = fresh_stats(deep_tickers)

                for ticker in deep_tickers:
                    stats = deep_stats.get(ticker)
                    hist_from = stats.get("Hist From", "N/A") if stats else "N/A"
                    inception = stats.get("Inception", "N/A") if stats else B_INCEPT.get(ticker, "N/A")

                    st.subheader(
                        f"Analysis for {ticker} | Fund inception: {inception} "
                        f"| Usable history from: {hist_from}"
                    )

                    if ticker in INDEX_NOTE:
                        cap(INDEX_NOTE[ticker])

                    if stats and stats.get("Hist Notes") not in (None, "-"):
                        cap(f"History lineage: {stats['Hist Notes']}")

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

                        detail = dividend_detail(ticker)

                        if detail and detail.get("last_amount") is not None:
                            e1, e2, e3, e4 = st.columns(4)
                            e1.metric("Last payout / share", f"${detail['last_amount']:.4f}")
                            e2.metric("Ex-date", detail["last_date"])
                            e3.metric("Change vs. a year earlier", f"{detail['yoy']:+.1%}" if detail["yoy"] is not None else "-")
                            e4.metric("Last 12 months / share", f"${detail['ttm']:.4f}")
                            note = f" Showing {detail['source']}'s record (same index) until {ticker} has a full year." if detail["source"] != ticker else ""
                            pay_note = f" Pay date per Yahoo: {detail['pay_date_yahoo']}." if detail.get("pay_date_yahoo") else " Pay date: usually 2 to 7 days after the ex-date (issuer site has the exact day)."
                            cap(f"Usual payout months: {detail['usual_months']}.{pay_note}{note}")

                    df, holdings_source = get_holdings(ticker)

                    if df.empty:
                        st.warning(
                            f"Could not load holdings for {ticker}. Both holdings sources are "
                            "unreachable right now - wait ~1 minute and press the button again."
                        )
                    else:
                        display_df = merge_goog(df.rename(columns={"Raw_Weight": "Weight"})).reset_index(drop=True)
                        display_df["Weight %"] = (display_df["Weight"] * 100).round(2)
                        display_df["Sector"] = assign_sectors(display_df, lookups=0)
                        display_df["Industry"] = [
                            IND_MAP.get(symbol) or sector
                            for symbol, sector in zip(display_df["Symbol"], display_df["Sector"])
                        ]

                        cap(f"{len(display_df)} holdings - source: {holdings_source}.")
                        st.dataframe(
                            display_df[["Symbol", "Sector", "Industry", "Weight %"]],
                            height=400,
                            hide_index=True
                        )

    # --- TAB 5: WATCHLIST (candidates vs my portfolio) ---
    with tab5:
        st.header("Watchlist: candidates vs. my portfolio")
        cap(
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
        cap(
            "Type two to four tickers. Metrics run down the page in plain words, funds run across, "
            "so it reads well on a phone. Your portfolio's funds are the default; try QNDX, QQQ to see twins."
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
                        md("### 🧭 The short version")
                        md("\n\n".join(f"- {v}" for v in verdicts))

                    for title, df in sections:
                        md(f"### {title}")
                        st.dataframe(df, hide_index=True)

                    if overlaps:
                        md("### 🔁 How much they overlap")
                        st.dataframe(pd.DataFrame(overlaps), hide_index=True)
                        cap(
                            "\"Same stocks by weight\" adds up, stock by stock, the smaller of the two funds' "
                            "weights. 100% would mean identical portfolios."
                        )

                    md("### 💰 Growth of $10,000")
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
                            cap(f"Chart starts {start_used.date()}. Log scale, so equal slopes mean equal growth rates.")

                    if skipped:
                        cap("Left off the chart (history starts after the window): " + ", ".join(skipped) + ".")

    # --- TAB: TAXES ---
    with tab_tax:
        st.header("🧾 Taxes: what each fund does to your tax bill")
        cap(
            "General U.S. rules for 2026 in plain words, plus an estimate for your own blend. Not personal tax "
            "advice. The exact qualified share of any fund is on your M1 1099-DIV (box 1b) each year."
        )

        # --- Part 1: profile of any ticker ---
        md("### 1. Look up any ticker")
        tax_tickers = parse_tickers(st.text_input("Tickers:", DEFAULT_PORT, key="tax_tickers"))

        for ticker in tax_tickers[:8]:
            key, how = tax_profile_key(ticker)
            profile = TAX_PROFILES[key]

            with st.expander(f"{ticker} - {profile['name']}", expanded=len(tax_tickers) <= 4):
                md(f"**What it pays you:** {profile['payouts']}")
                md(f"**Tax on those payouts:** {profile['payout_tax']}")
                md(f"**Capital-gain payouts from the fund:** {profile['gains_payouts']}")
                md(f"**When you sell shares:** {profile['selling']}")
                md(f"**Best account to hold it in:** {profile['best_home']}")
                md("**Upsides:** " + "; ".join(profile["upsides"]) + ".")
                md("**Downsides:** " + "; ".join(profile["downsides"]) + ".")
                md(f"**On your tax forms:** {profile['form']}")
                cap(f"Classified as: {how}.")

        md("---")

        # --- Part 2: blended estimate ---
        md("### 2. Estimate for my blend")
        cap(
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

        cap(
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

                md(
                    f"With {taxable_pct}% of the money in taxable and {100 - taxable_pct}% in Roth, your funds' payouts "
                    f"cost about **${total_tax:,.0f} a year** in tax, which is **{drag:.2%}** of the portfolio.{expense_text} "
                    "Nothing else is taxed until you sell, and you never have to."
                )

                # Projection
                md("#### If this grows for years")
                blended_now, _ = calculate_blended_performance(tax_weights)
                default_growth = blended_now.get("10Y CAGR")
                growth_pct = st.number_input(
                    "Assumed growth per year (%):", min_value=0.0, max_value=40.0,
                    value=float(min(max(round((default_growth or 0.10) * 100, 1), 0.0), 40.0)), step=0.5, key="tax_growth",
                    help="Starts at your blend's 10-year growth rate. Yields and tax rates are held constant."
                )
                tax_inflation = st.number_input("Inflation assumption for 'today's dollars' (% per year):", min_value=0.0, max_value=10.0,
                                                value=2.5, step=0.5, key="tax_inflation")
                g = growth_pct / 100
                infl = tax_inflation / 100
                proj_rows = []
                cumulative = 0.0
                cumulative_today = 0.0

                for year in range(1, 31):
                    value_t = tax_value * (1 + g) ** (year - 1)
                    tax_t = value_t * drag
                    cumulative += tax_t
                    cumulative_today += todays_dollars(tax_t, year, infl)

                    if year in (1, 5, 10, 20, 30):
                        future_value = tax_value * (1 + g) ** year
                        proj_rows.append({
                            "After": f"{year} yr" if year == 1 else f"{year} yrs",
                            "Portfolio value (future dollars)": f"${future_value:,.0f}",
                            "Portfolio value (today's dollars)": f"${todays_dollars(future_value, year, infl):,.0f}",
                            "Tax paid on payouts so far (future dollars)": f"${cumulative:,.0f}",
                            "Same tax in today's dollars": f"${cumulative_today:,.0f}",
                        })

                st.dataframe(pd.DataFrame(proj_rows), hide_index=True)
                cap(
                    "Assumes no new contributions, payouts reinvested, and today's yields, weights and tax rates "
                    f"throughout. 'Today's dollars' removes {tax_inflation:.1f}% yearly inflation. It is a scale check, not a forecast."
                )

        md("---")

        # --- Part 3: if you ever sold ---
        md("### 3. If you ever sold")
        basis = st.number_input(
            "What you paid in total for the taxable shares ($, optional):", min_value=0, value=0, step=1000, key="tax_basis"
        )

        if basis > 0 and tax_value > 0 and taxable_share > 0:
            taxable_value = float(tax_value) * taxable_share
            gain = taxable_value - basis

            if gain > 0:
                md(
                    f"Taxable-account value about **${taxable_value:,.0f}** on **${basis:,.0f}** paid in: gain of "
                    f"**${gain:,.0f}**. Sold after holding more than 1 year: about **${gain * rates['qualified']:,.0f}** "
                    f"in tax ({rates['qualified']:.1%}). Sold within a year: about **${gain * rates['ordinary']:,.0f}** "
                    f"({rates['ordinary']:.1%}). Roth shares: $0 on a qualified withdrawal."
                )
            else:
                md("No gain on those numbers, so no tax on a sale (a loss could offset other gains or up to $3,000 of income a year).")

        md(
            "Rules that matter for a weekly buyer: every purchase is its own lot with its own one-year clock; "
            "dividends on a lot are qualified once you have held it more than 60 days around the ex-dividend date, "
            "which a never-sell holder always meets; M1 sells specific lots in a set order when you do sell, so "
            "check that setting before any sale; and holding until you pass it on gives heirs a stepped-up basis."
        )

    # --- TAB: MILITARY ---
    with tab_mil:
        military_tab()

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
                _failure_log().clear()
                st.rerun()

        with top_m:
            if st.button("🧹 Clear all cached data", key="clear_all_cache",
                         help="Use this if a number looks stuck or wrong. Everything reloads fresh (about a minute)."):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        if not insight_tickers:
            st.info("Enter at least one ticker.")
        else:
            # --- Live market snapshot ---
            md("### ⏱️ Market Pulse")

            pulse_df, pulse_time = get_pulse(tickers_key)

            with top_r:
                cap(
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
                    md("\n\n".join(callouts))

            md("---")

            # --- Live headlines ---
            md("### 🗞️ Latest Headlines")

            news_df, news_time = get_live_news(tickers_key)

            cap(
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
                        md(f"**[{item['Title']}]({item['Link']})**  \n{meta}")
                    else:
                        md(f"**{item['Title']}**  \n{meta}")

            md("---")

            # --- Computed portfolio read ---
            md("### 🧠 Portfolio Read (computed live)")

            read_weights = current_weights(insight_tickers)
            core_holdings, _counts = build_blended_holdings(insight_tickers, read_weights, sector_lookups=0)

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
                md(
                    f"Blending {mix} gives you {n_unique} underlying stocks. The top 3 ({top3_names}) are an "
                    f"estimated **{top3_weight:.1%}** of the portfolio and the top 10 are **{top10_weight:.1%}**. "
                    f"**{top1['Symbol']}** alone is ~**{top1['Weight']:.1%}**." + overlap_text
                )
                cap(
                    "Computed from each fund's published holdings (full basket where available): "
                    + " | ".join(f"{t}: {info}" for t, info in _counts.items())
                )

                st.dataframe(
                    core_holdings[["Symbol", "Sector", "Industry", "Weight %"]].head(10),
                    hide_index=True
                )
            else:
                st.info("Holdings overlap unavailable right now (rate-limited) - press Refresh now to retry.")

            md("---")

            # --- AI prompt with live numbers (changes every day) ---
            md("### 🤖 1-Click Claude Intelligence Prompt")

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
                f"weighted {mix_text}. Cover: (1) this week's key news per holding, (2) Nasdaq-100, "
                "technology-sector and momentum-factor dynamics, (3) macro factors (rates, AI capex) affecting these "
                "funds, and (4) anything a long-term buy-and-hold investor should monitor. Cite recent sources."
            )

            st.code(prompt_text, language="text")

            claude_url = "https://claude.ai/new?q=" + urllib.parse.quote(prompt_text)
            md(f"[🔗 Open in Claude (prompt pre-filled)]({claude_url})")
            cap(
                "Prompt embeds today's live numbers, so it changes every day. "
                "The link opens Claude with the prompt ready to send - or copy the box above."
            )

    # --- TAB 7: SHOWS & VOICES ---
    with tab7:
        st.header("🎧 Shows & Voices")
        cap(
            "Newest episodes and videos from the shows you follow. Refreshes every 30 minutes. "
            "To add a show, tell Claude the name or paste its YouTube link."
        )

        if st.button("🔄 Refresh shows", key="refresh_shows"):
            get_show_items.clear()
            st.rerun()

        shows_df, show_status, shows_time = get_show_items(tuple(SHOW_FEEDS))

        cap(
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
                    md(f"**[{item['Title']}]({item['Link']})**  \n{meta}")
                else:
                    md(f"**{item['Title']}**  \n{meta}")

            videos = shows_df[shows_df["VideoId"].notna()]

            if not videos.empty:
                md("---")
                md("### ▶️ Watch here")

                channels = videos.drop_duplicates(subset=["Show"])["Show"].tolist()
                watch_pick = st.selectbox("Latest video from:", channels, key="watch_pick")
                latest = videos[videos["Show"] == watch_pick].iloc[0]

                md(f"**{latest['Title']}**")
                st.video(f"https://www.youtube.com/watch?v={latest['VideoId']}")

    md("---")
    cap(
        f"Master Portfolio v{APP_VERSION}. Data: Yahoo Finance via yfinance (free, ~15-min delayed quotes). "
        "Prices cached 15 min, full history 6h, holdings 12h, shows 30 min. Long-period returns use "
        "index-extended history - see Hist Notes. "
        f"Running on Streamlit {st.__version__}, pandas {pd.__version__}, yfinance {getattr(yf, '__version__', '?')}."
    )


if not os.environ.get("APP_TESTING"):
    main()
