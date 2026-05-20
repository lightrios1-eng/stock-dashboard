Yes, that makes sense. The issue is not just `.head(20)`: `yfinance` often only returns top holdings. This version removes the table cap and adds broader holdings sources, then falls back to yfinance when full holdings are not available.

Sources used for provider holdings behavior: [State Street ETF holdings](https://www.ssga.com/us/en/intermediary/etfs/funds/the-technology-select-sector-spdr-fund-xlk), [iShares IXN page](https://www.ishares.com/us/products/239750/ishares-global-tech-etf), [VanEck SMH holdings](https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/holdings/).

```python
import os
import re
import xml.etree.ElementTree as ET
from io import BytesIO

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

DEFAULT_PORT = "XLK, IXN, SMH, QQQ"
DEFAULT_BENCH = "VOO, QQQ"
DEFAULT_WATCH = "XLK, IXN, SMH, QQQ"

ALL_NUM_COLS = [
    "Yield (TTM)", "Yield (Fwd)", "1D", "1W", "1M", "YTD",
    "1Y Total", "3Y Total", "3Y CAGR", "5Y Total", "5Y CAGR",
    "10Y Total", "10Y CAGR", "15Y Total", "15Y CAGR",
    "3Y Div CAGR", "5Y Div CAGR", "10Y Div CAGR", "15Y Div CAGR"
]

PERF_PERIODS = ["1W", "1M", "YTD", "1Y Total", "3Y CAGR", "5Y CAGR", "10Y CAGR", "15Y CAGR"]

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
    "IXN": "Global Technology ETF", "VGT": "Technology ETF", "SMH": "Semiconductor ETF",
    "SCHG": "Growth ETF", "VOO": "S&P 500 ETF", "SCHD": "Dividend ETF",
    "VYM": "Dividend ETF", "VIG": "Dividend ETF"
}

B_INCEPT = {
    "XLK": "1998-12-16", "IXN": "2001-11-12", "SMH": "2011-12-20",
    "QQQ": "1999-03-10", "QQQM": "2020-10-13", "MGK": "2007-12-17",
    "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07",
    "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10",
    "SCHD": "2011-10-20", "VIG": "2006-04-21", "JEPQ": "2022-05-03"
}

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 portfolio research app",
    "Accept": "*/*",
}

VANECK_DATASET_URLS = {
    "SMH": "https://www.vaneck.com/Main/HoldingsBlock/GetDataset/?blockId=144458&pageId=233107&ticker=SMH"
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


def index_tz(index):
    return getattr(index, "tz", None)


def now_for_index(index):
    tz = index_tz(index)
    return pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()


def timestamp_for_index(value, index):
    ts = pd.Timestamp(value)
    tz = index_tz(index)

    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


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

    year = now_for_index(close.index).year
    start_ts = timestamp_for_index(f"{year}-01-01", close.index)

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


def clean_symbol(value):
    symbol = str(value).strip().upper()
    symbol = re.sub(r"\s+", "", symbol)
    return symbol


def coerce_weight(value):
    try:
        if value is None or pd.isna(value):
            return None

        if isinstance(value, str):
            has_percent = "%" in value
            value = value.replace("%", "").replace(",", "").strip()
            if value in {"", "-", "--"}:
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

    return sample.str.match(r"^[A-Za-z0-9][A-Za-z0-9.\-:]{0,20}$").mean()


def unique_column_names(names):
    used = {}
    clean = []

    for i, name in enumerate(names):
        base = str(name).strip()
        if not base or base.lower() in {"nan", "none"}:
            base = f"col_{i}"

        if base in used:
            used[base] += 1
            clean.append(f"{base}_{used[base]}")
        else:
            used[base] = 0
            clean.append(base)

    return clean


def normalize_table_by_header(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    df = df.dropna(how="all").copy()

    for row_idx, row in df.iterrows():
        labels = [str(x).strip() for x in row.tolist()]
        lowered = [x.lower() for x in labels]

        has_symbol = any(x in {"ticker", "symbol"} or "ticker" in x or "symbol" in x for x in lowered)
        has_weight = any("weight" in x or "% of fund" in x or "% of net" in x or "net assets" in x for x in lowered)

        if has_symbol and has_weight:
            data = df.loc[row_idx + 1:].copy()
            data.columns = unique_column_names(labels)
            return normalize_holdings_frame(data)

    return normalize_holdings_frame(df)


def normalize_holdings_frame(raw):
    if raw is None:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    if isinstance(raw, pd.Series):
        df = raw.reset_index()
    else:
        df = raw.copy().reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    df.columns = [
        "_".join(map(str, col)).strip() if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]

    symbol_candidates = [
        col for col in df.columns
        if any(key in col.lower() for key in ["ticker", "symbol"])
        and "cusip" not in col.lower()
        and "isin" not in col.lower()
    ]

    weight_candidates = [
        col for col in df.columns
        if any(key in col.lower() for key in ["weight", "% of fund", "% of net", "net assets"])
    ]

    symbol_col = None
    weight_col = None

    if symbol_candidates and weight_candidates:
        symbol_col = symbol_candidates[0]
        weight_col = weight_candidates[0]
    else:
        scored_symbols = sorted(
            [(ticker_like_score(df[col]), col) for col in df.columns],
            reverse=True
        )
        symbol_col = scored_symbols[0][1] if scored_symbols and scored_symbols[0][0] >= 0.35 else None

        weight_scores = []
        for col in df.columns:
            if col == symbol_col:
                continue

            parsed = df[col].map(coerce_weight)
            parse_score = parsed.notna().mean()
            name_bonus = 0.5 if any(x in col.lower() for x in ["weight", "percent", "%", "holding"]) else 0

            if parse_score > 0:
                weight_scores.append((parse_score + name_bonus, col))

        weight_col = max(weight_scores)[1] if weight_scores else None

    if symbol_col is None or weight_col is None:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    out = df[[symbol_col, weight_col]].copy()
    out.columns = ["Symbol", "Raw_Weight"]

    out["Symbol"] = out["Symbol"].map(clean_symbol)
    out["Raw_Weight"] = out["Raw_Weight"].map(coerce_weight)

    out = out.dropna(subset=["Symbol", "Raw_Weight"])
    out = out[out["Raw_Weight"] > 0]
    out = out[out["Symbol"].str.match(r"^[A-Z0-9][A-Z0-9.\-:]{0,20}$")]
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


def get_secret_or_env(name):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    return value or os.environ.get(name)


def cell_value(value):
    if isinstance(value, dict):
        return value.get("d", value.get("r"))
    return value


def parse_xml_spreadsheet(content):
    frames = []

    try:
        root = ET.fromstring(content)
    except Exception:
        return frames

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    index_attr = "{urn:schemas-microsoft-com:office:spreadsheet}Index"

    for worksheet in root.findall(".//ss:Worksheet", ns):
        rows = []

        for row in worksheet.findall(".//ss:Row", ns):
            values = []
            col_index = 1

            for cell in row.findall("ss:Cell", ns):
                idx = cell.attrib.get(index_attr)

                if idx:
                    idx = int(idx)
                    while col_index < idx:
                        values.append(None)
                        col_index += 1

                data = cell.find("ss:Data", ns)
                values.append(data.text if data is not None else None)
                col_index += 1

            if any(v is not None and str(v).strip() for v in values):
                rows.append(values)

        if rows:
            frames.append(pd.DataFrame(rows))

    return frames


# --- HOLDINGS SOURCES ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_holdings_api(ticker):
    api_key = get_secret_or_env("ETF_HOLDINGS_API_KEY")

    if not api_key:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    urls = [
        f"https://etf-holdings.com/api/v1/holdings?symbol={ticker}",
        f"https://etf-holdings.com/api/v1/holdings?ticker={ticker}",
    ]

    headers = dict(REQUEST_HEADERS)
    headers["Authorization"] = f"Bearer {api_key}"

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=20)

            if response.status_code >= 400:
                continue

            payload = response.json()
            rows = payload.get("holdings") if isinstance(payload, dict) else payload

            if not rows:
                continue

            df = pd.DataFrame(rows)
            normalized = normalize_holdings_frame(df)

            if not normalized.empty:
                return normalized
        except Exception:
            continue

    return pd.DataFrame(columns=["Symbol", "Raw_Weight"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_state_street_holdings(ticker):
    url = f"https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx"

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)

        if response.status_code != 200 or len(response.content) < 1000:
            return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        sheets = pd.read_excel(BytesIO(response.content), sheet_name=None, header=None)

        best = pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        for raw in sheets.values():
            normalized = normalize_table_by_header(raw)

            if len(normalized) > len(best):
                best = normalized

        return best
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ishares_product_id(ticker):
    url = (
        "https://www.ishares.com/us/product-screener/product-screener-v3.jsn"
        "?dcrPath=/templatedata/config/product-screener-v3/data/en/us-ishares/"
        "ishares-product-screener-backend-config"
    )

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=25)

        if response.status_code != 200:
            return None

        payload = response.json()
        table = payload.get("data", {}).get("tableData", {})
        columns = [col.get("name") for col in table.get("columns", [])]
        rows = table.get("data", [])

        ticker_idx = columns.index("localExchangeTicker")
        portfolio_idx = columns.index("portfolioId")

        for row in rows:
            values = row.get("value", row) if isinstance(row, dict) else row

            if not isinstance(values, list) or len(values) <= max(ticker_idx, portfolio_idx):
                continue

            row_ticker = str(cell_value(values[ticker_idx])).upper()

            if row_ticker == ticker:
                return cell_value(values[portfolio_idx])
    except Exception:
        return None

    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ishares_holdings(ticker):
    portfolio_id = fetch_ishares_product_id(ticker)

    if not portfolio_id:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    url = (
        "https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/get-fund-document"
        f"?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US"
        f"&portfolioId={portfolio_id}&component=fundDownload&userType=individual"
    )

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)

        if response.status_code != 200:
            return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        frames = parse_xml_spreadsheet(response.content)
        best = pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        for raw in frames:
            normalized = normalize_table_by_header(raw)

            if len(normalized) > len(best):
                best = normalized

        return best
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_vaneck_holdings(ticker):
    url = VANECK_DATASET_URLS.get(ticker)

    if not url:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)

        if response.status_code != 200:
            return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        payload = response.json()
        rows = payload.get("Holdings", [])

        if not rows:
            return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        df = pd.DataFrame(rows)

        symbol_col = "HoldingTicker" if "HoldingTicker" in df.columns else "Label"
        weight_col = "Weight"

        out = df[[symbol_col, weight_col]].copy()
        out.columns = ["Symbol", "Raw_Weight"]

        return normalize_holdings_frame(out)
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yfinance_holdings(ticker):
    try:
        funds_data = yf.Ticker(ticker).funds_data
        raw = getattr(funds_data, "top_holdings", None)

        if callable(raw):
            raw = raw()

        return normalize_holdings_frame(raw)
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_holdings_with_source(ticker):
    ticker = ticker.strip().upper()

    sources = [
        ("ETF Holdings API", fetch_etf_holdings_api),
        ("State Street provider file", fetch_state_street_holdings),
        ("iShares/BlackRock provider file", fetch_ishares_holdings),
        ("VanEck provider file", fetch_vaneck_holdings),
        ("yfinance top_holdings", fetch_yfinance_holdings),
    ]

    for source_name, fetcher in sources:
        df = fetcher(ticker)

        if df is not None and not df.empty:
            df = df.copy()
            df["Source"] = source_name
            return df, source_name

    return pd.DataFrame(columns=["Symbol", "Raw_Weight", "Source"]), "No holdings source"


def get_holdings(ticker):
    df, _source = get_holdings_with_source(ticker)
    return df[["Symbol", "Raw_Weight"]].copy() if not df.empty else pd.DataFrame(columns=["Symbol", "Raw_Weight"])


# --- DATA ENGINE ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_full_stats(ticker):
    ticker = ticker.strip().upper()

    if not ticker:
        return None

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max", auto_adjust=True)

        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].dropna()

        if close.empty:
            return None

        try:
            div = stock.dividends
            if div is None:
                div = pd.Series(dtype=float)
        except Exception:
            div = pd.Series(dtype=float)

        try:
            info = stock.info or {}
        except Exception:
            info = {}

    except Exception:
        return None

    price = float(close.iloc[-1])
    if price <= 0:
        return None

    if not div.empty:
        cutoff = now_for_index(div.index) - pd.Timedelta(days=365)
        y_ttm = float(div[div.index >= cutoff].sum()) / price
    else:
        y_ttm = 0.0

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
        "Inception": B_INCEPT.get(ticker, "N/A"),
        "Yield (Fwd)": y_fwd,
        "Yield (TTM)": y_ttm,
    }

    if not div.empty:
        current_year = now_for_index(div.index).year
        annual_div = div.groupby(div.index.year).sum()
        completed = annual_div[annual_div.index < current_year]
        m["Streak"] = dividend_growth_streak(completed)

        last_year = current_year - 1
        div_count = int(div[div.index.year == last_year].count())
    else:
        annual_div = pd.Series(dtype=float)
        current_year = pd.Timestamp.now().year
        m["Streak"] = 0
        div_count = 0

    m["Freq"] = "Mo" if div_count >= 11 else "Qr" if div_count >= 3 else "Yr" if div_count >= 1 else "-"

    m["1D"] = return_by_sessions(close, 1)
    m["1W"] = return_by_sessions(close, 5)
    m["1M"] = return_by_sessions(close, 21)
    m["YTD"] = ytd_return(close)

    for years in [1, 3, 5, 10, 15]:
        total = period_total_return(close, years)
        m[f"{years}Y Total"] = total
        m[f"{years}Y CAGR"] = (
            ((1 + total) ** (1 / years)) - 1
            if years > 1 and total is not None and total > -1
            else None
        )

    if not div.empty:
        last_completed_year = current_year - 1

        for years in [3, 5, 10, 15]:
            try:
                start_year = last_completed_year - years

                if (
                    last_completed_year in annual_div.index
                    and start_year in annual_div.index
                    and annual_div.loc[start_year] > 0
                ):
                    m[f"{years}Y Div CAGR"] = (
                        (annual_div.loc[last_completed_year] / annual_div.loc[start_year]) ** (1 / years)
                    ) - 1
                else:
                    m[f"{years}Y Div CAGR"] = None
            except Exception:
                m[f"{years}Y Div CAGR"] = None
    else:
        for years in [3, 5, 10, 15]:
            m[f"{years}Y Div CAGR"] = None

    return m


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
        st.warning(f"No usable market data for: {', '.join(failed)}")

    return data


def build_blended_holdings(etfs, weights):
    dfs = []

    for ticker in etfs:
        weight = weights.get(ticker, 0)

        if weight <= 0:
            continue

        df = get_holdings(ticker)

        if not df.empty:
            df = df.copy()
            df["Weight"] = df["Raw_Weight"] * weight
            dfs.append(df[["Symbol", "Weight"]])

    if not dfs:
        return pd.DataFrame(columns=["Symbol", "Weight", "Weight %", "Industry"])

    full = merge_goog(pd.concat(dfs, ignore_index=True))
    full["Weight %"] = (full["Weight"] * 100).round(2)
    full["Industry"] = full["Symbol"].apply(lambda x: IND_MAP.get(x, "Diversified / Other"))

    return full


def calculate_blended_performance(weights):
    stats_by_ticker = {}

    for ticker, weight in weights.items():
        if weight <= 0:
            continue

        stats = get_full_stats(ticker)
        if stats:
            stats_by_ticker[ticker] = stats

    period_sources = {
        "1W": ("1W", None),
        "1M": ("1M", None),
        "YTD": ("YTD", None),
        "1Y Total": ("1Y Total", None),
        "3Y CAGR": ("3Y Total", 3),
        "5Y CAGR": ("5Y Total", 5),
        "10Y CAGR": ("10Y Total", 10),
        "15Y CAGR": ("15Y Total", 15),
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


# --- UI TABS ---
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

            blended_stats = calculate_blended_performance(weights)
            metric_cols = st.columns(len(PERF_PERIODS))

            for i, period in enumerate(PERF_PERIODS):
                metric_cols[i].metric(period, pct_or_na(blended_stats.get(period)))

            st.markdown("---")

            full = build_blended_holdings(etfs, weights)

            if full.empty:
                st.warning("Could not load holdings data for the selected ETFs.")
            else:
                source_notes = []
                limited_sources = []

                for ticker in etfs:
                    df_src, source_name = get_holdings_with_source(ticker)
                    row_count = len(df_src)
                    source_notes.append(f"{ticker}: {row_count} rows from {source_name}")

                    if source_name == "yfinance top_holdings":
                        limited_sources.append(ticker)

                st.caption(
                    f"Showing all {len(full)} blended holdings loaded by the app. "
                    "Scroll the holdings table to see the full list."
                )
                st.caption(" | ".join(source_notes))

                if limited_sources:
                    st.info(
                        "Some ETFs are source-limited because yfinance only provides top holdings for them: "
                        f"{', '.join(limited_sources)}. The table still shows every holding row the app could retrieve."
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

                final = pd.DataFrame([avg_p] + m_stats)
                final_cols = ["Ticker", "Inception"] + [col for col in ALL_NUM_COLS if col in final.columns]
                final = final[final_cols].dropna(axis=1, how="all")

                st.dataframe(format_dataframe(final), hide_index=True)


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
                    "Industry": "-",
                    "Price": None,
                    "Streak": None,
                    "Freq": "-"
                })

                final = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)

                cols = ["Ticker", "Price", "Industry", "Inception", "Streak", "Freq"] + ALL_NUM_COLS
                final = final[[col for col in cols if col in final.columns]]

                st.dataframe(format_dataframe(final), height=500, hide_index=True)


# --- TAB 4: DEEP DIVE ---
with tab4:
    st.header("Multi-ETF Deep Dive")

    deep_tickers = parse_tickers(st.text_input("Tickers:", DEFAULT_PORT, key="deep_dive_tickers"))

    if st.button("Inspect ETFs", key="inspect_etfs"):
        if not deep_tickers:
            st.info("Enter at least one ETF ticker.")
        else:
            for ticker in deep_tickers:
                st.subheader(f"Analysis for {ticker} | Inception: {B_INCEPT.get(ticker, 'N/A')}")

                df = get_holdings(ticker)

                if df.empty:
                    st.warning(f"Could not load holdings for {ticker}.")
                else:
                    display_df = merge_goog(df.rename(columns={"Raw_Weight": "Weight"}))
                    display_df["Weight"] = (display_df["Weight"] * 100).map("{:.2f}%".format)
                    st.table(display_df.head(15))


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
                cols = ["Ticker", "Price", "Industry", "Inception"] + ALL_NUM_COLS
                final_cols = [col for col in cols if col in df.columns]

                st.dataframe(format_dataframe(df[final_cols]), height=500, hide_index=True)


# --- TAB 6: AI NEWS & INSIGHTS ---
with tab6:
    st.header("📰 Deep-Dive Portfolio Insights")

    st.markdown("### ⏱️ Market Pulse")
    st.caption("Cached for one hour. Market data may be delayed depending on yfinance availability.")

    pulse_tickers = parse_tickers(DEFAULT_PORT)
    pulse_data = [stats for stats in (get_full_stats(ticker) for ticker in pulse_tickers) if stats]

    if pulse_data:
        df_pulse = pd.DataFrame(pulse_data)
        pulse_cols = ["Ticker", "Price", "1D", "1W", "1M", "YTD"]
        df_pulse = df_pulse[[col for col in pulse_cols if col in df_pulse.columns]]
        st.dataframe(format_dataframe(df_pulse), hide_index=True)
    else:
        st.warning("Could not load market data.")

    st.markdown("---")

    st.markdown("### 🤖 1-Click Gemini Intelligence Prompt")
    st.markdown("Use this payload to generate a broader news and portfolio review.")

    prompt_text = (
        "Give me a comprehensive news update about my investment portfolio. "
        f"It is blended equally among: {', '.join(pulse_tickers)}. "
        "Format with headings for Daily/Weekly Short-Term Dynamics, "
        "Monthly Medium-Term Trends, and Annual Long-Term Fundamentals."
    )

    st.code(prompt_text, language="text")
    st.markdown("[🔗 Open Gemini](https://gemini.google.com/app)")

    st.markdown("---")

    st.markdown("### 🧠 Executive Summary")

    equal_weights = {ticker: 1 / len(pulse_tickers) for ticker in pulse_tickers} if pulse_tickers else {}
    core_holdings = build_blended_holdings(pulse_tickers, equal_weights) if equal_weights else pd.DataFrame()

    st.markdown("#### 📊 Asset Allocation & Overlap Analysis")

    if not core_holdings.empty:
        nvda_weight = core_holdings.loc[core_holdings["Symbol"] == "NVDA", "Weight"].sum()
        msft_aapl_weight = core_holdings.loc[
            core_holdings["Symbol"].isin(["MSFT", "AAPL"]),
            "Weight"
        ].sum()
        top3_weight = core_holdings.head(3)["Weight"].sum()

        st.markdown(f"""
This portfolio remains an aggressive, growth-oriented allocation concentrated in U.S. mega-cap technology and semiconductors.

Based on the latest available holdings data:

* **NVDA estimated exposure:** {nvda_weight:.2%}
* **MSFT + AAPL estimated exposure:** {msft_aapl_weight:.2%}
* **Top 3 holdings estimated exposure:** {top3_weight:.2%}

Because the selected ETFs can overlap heavily, the portfolio may move more like a concentrated technology basket than a broadly diversified ETF portfolio.
""")

        st.dataframe(
            core_holdings[["Symbol", "Industry", "Weight %"]].head(10),
            hide_index=True
        )
    else:
        st.markdown("""
This portfolio is an aggressive, technology-heavy allocation.

The biggest structural risk is overlap. Multiple technology and growth ETFs can hold many of the same large companies, which can make the real underlying portfolio more concentrated than the ETF count suggests.
""")

    st.markdown("---")

    st.markdown("#### 🚀 Primary Performance Drivers")
    st.markdown("""
* **AI infrastructure spending:** Semiconductor and technology ETFs benefit when cloud providers and enterprises keep spending on chips, data centers, and software infrastructure.
* **Large-cap growth momentum:** QQQ and technology-heavy funds are tied to earnings strength and valuation multiples in mega-cap growth.
* **Rate sensitivity:** Growth-heavy portfolios can benefit when interest-rate expectations fall, but they can also reprice sharply when rates rise.
""")

    st.markdown("---")

    st.markdown("#### 📉 Risk Management & Vulnerabilities")
    st.markdown("""
* **Sector concentration:** The portfolio may have limited defensive exposure compared with the S&P 500.
* **Valuation risk:** If AI or mega-cap tech expectations cool, the drawdown can be larger than a broad-market benchmark.
* **Low income:** Dividend yield is likely modest, so returns depend mainly on capital appreciation.
""")

    st.markdown("---")

    st.markdown("#### 🎯 Strategic Considerations")
    st.markdown("""
1. **Add ballast if volatility becomes uncomfortable:** SCHD, VYM, VIG, or VOO can reduce single-theme dependence.
2. **Watch overlap:** Technology and growth ETFs can duplicate the same mega-cap exposure.
3. **Review concentration periodically:** If one or two names become too large, rebalance rules can help keep risk intentional.
""")

st.markdown("---")
```
