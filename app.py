import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_holdings(ticker):
    ticker = ticker.strip().upper()

    if not ticker:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

    try:
        stock = yf.Ticker(ticker)
        raw = stock.funds_data.top_holdings

        if raw is None:
            return pd.DataFrame(columns=["Symbol", "Raw_Weight"])

        return normalize_holdings_frame(raw)
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Raw_Weight"])


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
    source_counts = {}

    for ticker in etfs:
        weight = weights.get(ticker, 0)

        if weight <= 0:
            continue

        df = get_holdings(ticker)
        source_counts[ticker] = len(df)

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

            full, source_counts = build_blended_holdings(etfs, weights)

            if full.empty:
                st.warning("Could not load holdings data for the selected ETFs.")
                st.caption("yfinance sometimes does not provide holdings for every ETF ticker.")
            else:
                st.caption(
                    f"Showing all {len(full)} blended holdings returned by yfinance. "
                    "Some ETFs only expose top holdings through yfinance."
                )
                st.caption(
                    "Holdings loaded: "
                    + " | ".join(f"{ticker}: {count}" for ticker, count in source_counts.items())
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
                    display_df["Weight %"] = (display_df["Weight"] * 100).round(2)
                    display_df["Industry"] = display_df["Symbol"].apply(
                        lambda x: IND_MAP.get(x, "Diversified / Other")
                    )

                    st.caption(f"Showing all {len(display_df)} holdings returned by yfinance for {ticker}.")
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
    core_holdings, _source_counts = build_blended_holdings(pulse_tickers, equal_weights) if equal_weights else (
        pd.DataFrame(),
        {}
    )

    st.markdown("#### 📊 Asset Allocation & Overlap Analysis")

    if not core_holdings.empty:
        nvda_weight = core_holdings.loc[core_holdings["Symbol"] == "NVDA", "Weight"].sum()
        msft_aapl_weight = core_holdings.loc[
            core_holdings["Symbol"].isin(["MSFT", "AAPL"]),
            "Weight"
        ].sum()
        top3_weight = core_holdings.head(3)["Weight"].sum()

        st.markdown(f"""
This portfolio remains an aggressive, growth-oriented allocation concentrated in technology and semiconductors.

Based on the latest available holdings data from yfinance:

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
This portfolio is a technology-heavy allocation.

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
