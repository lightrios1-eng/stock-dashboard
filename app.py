import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

DEFAULT_PORT = "SCHG, QQQ, VGT, SMH"
DEFAULT_BENCH = "VOO, SCHD"
DEFAULT_WATCH = "SCHD, VYM, VIG, VOO, SCHG, QQQ, VGT, SMH"

ALL_NUM_COLS = [
    'Yield (TTM)', 'Yield (Fwd)', '1D', '1W', '1M', 'YTD', 
    '1Y Total', '3Y Total', '3Y CAGR', '5Y Total', '5Y CAGR', 
    '10Y Total', '10Y CAGR', '15Y Total', '15Y CAGR', 
    '3Y Div CAGR', '5Y Div CAGR', '10Y Div CAGR', '15Y Div CAGR'
]

# --- COMPACT DATA MAPS ---
IND_MAP = {"NVDA": "Semi - GPU/AI", "AMD": "Semi - CPU/GPU", "INTC": "Semi - IDM", "TSM": "Semi - Foundry", "AVGO": "Semi - Network", "QCOM": "Semi - Mobile", "MU": "Semi - Memory", "TXN": "Semi - Analog", "ASML": "Semi Equip", "AMAT": "Semi Equip", "LRCX": "Semi Equip", "MSFT": "Cloud/OS", "ORCL": "Cloud/DB", "ADBE": "Creative Soft", "CRM": "Enterprise Soft", "AAPL": "Consumer Elec", "CSCO": "Network HW", "GOOG": "Search/Ads", "GOOGL": "Search/Ads", "META": "Social Media", "AMZN": "E-Commerce/Cloud", "TSLA": "EV Auto", "HD": "Home Improv", "WMT": "Big Box", "LLY": "Pharma", "UNH": "Health Ins", "JPM": "Bank", "V": "Payments", "LMT": "Defense", "XOM": "Oil/Gas", "PLD": "REIT", "QQQM": "Tech / Growth", "QQQ": "Tech / Growth", "VGT": "Technology"}
B_INCEPT = {"SMH": "2011-12-20", "QQQ": "1999-03-10", "QQQM": "2020-10-13", "MGK": "2007-12-17", "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07", "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10", "SCHD": "2011-10-20", "JEPQ": "2022-05-03"}

# --- CORE ENGINE ---
def merge_goog(df):
    df['Symbol'] = df['Symbol'].replace({'GOOGL': 'GOOG'})
    return df.groupby('Symbol', as_index=False)['Weight'].sum().sort_values(by='Weight', ascending=False)

def format_dataframe(df):
    """Safely formats floats to strings to prevent Pandas/PyArrow TypeErrors."""
    df = df.copy()
    for col in ALL_NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
    if 'Price' in df.columns:
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce').apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "-")
    return df

@st.cache_data(ttl=3600)
def get_full_stats(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max", auto_adjust=True)
        div = stock.dividends
        info = stock.info
    except: return None
    if hist.empty: return None

    p = hist['Close'].iloc[-1]
    y_ttm = div[div.index >= (pd.Timestamp.now().tz_localize(div.index.dtype.tz) - pd.Timedelta(days=365))].sum() / p if not div.empty else 0
    y_fwd = y_ttm if 'ETF' in info.get('quoteType','').upper() else info.get('dividendYield', 0)

    m = {'Ticker': ticker, 'Price': p, 'Industry': IND_MAP.get(ticker, "ETF/Fund"), 'Inception': B_INCEPT.get(ticker, "N/A"), 'Yield (Fwd)': y_fwd, 'Yield (TTM)': y_ttm}

    if not div.empty:
        annual_div = div.groupby(div.index.year).sum()
        completed = annual_div[annual_div.index < datetime.now().year].sort_index(ascending=False)
        streak = next((i for i in range(len(completed)-1) if completed.iloc[i] <= completed.iloc[i+1]), len(completed)-1) if len(completed) > 1 else 0
    else: streak = 0
    
    cnt = div[div.index.year == (datetime.now().year - 1)].count() if not div.empty else 0
    m['Streak'] = streak
    m['Freq'] = "Mo" if cnt >= 11 else "Qr" if cnt >= 3 else "Yr" if cnt >= 1 else "-"

    m['1D'] = (p - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] if len(hist)>1 else None
    m['1W'] = (p - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] if len(hist)>5 else None
    m['1M'] = (p - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22] if len(hist)>21 else None
    ytd = hist[hist.index >= pd.Timestamp(f"{datetime.now().year}-01-01").tz_localize(hist.index.dtype.tz)]
    m['YTD'] = (p - ytd['Open'].iloc[0]) / ytd['Open'].iloc[0] if not ytd.empty else None

    for y in [1, 3, 5, 10, 15]:
        target = hist.index[-1] - timedelta(days=y*365)
        try:
            idx = hist.index.get_indexer([target], method='nearest')[0]
            if idx >= 0 and abs((hist.index[idx] - target).days) < 100:
                sp = hist['Close'].iloc[idx]
                m[f'{y}Y Total'] = (p - sp) / sp
                m[f'{y}Y CAGR'] = (p / sp) ** (1/y) - 1 if y > 1 else None
            else: m[f'{y}Y Total'], m[f'{y}Y CAGR'] = None, None
        except: m[f'{y}Y Total'], m[f'{y}Y CAGR'] = None, None

    if not div.empty:
        ly = datetime.now().year - 1
        for y in [3, 5, 10, 15]:
            try:
                if ly in annual_div.index and (ly-y) in annual_div.index and annual_div.loc[ly-y] > 0:
                    m[f'{y}Y Div CAGR'] = (annual_div.loc[ly] / annual_div.loc[ly-y]) ** (1/y) - 1
                else: m[f'{y}Y Div CAGR'] = None
            except: m[f'{y}Y Div CAGR'] = None
    else:
        for y in [3, 5, 10, 15]: m[f'{y}Y Div CAGR'] = None

    return m

def get_holdings(ticker):
    try:
        df = yf.Ticker(ticker).funds_data.top_holdings
        if not df.empty:
            df = df.to_frame() if isinstance(df, pd.Series) else df
            df = df.reset_index().iloc[:, [0, -1]]
            df.columns = ['Symbol', 'Raw_Weight']
            return df
    except: pass
    return pd.DataFrame()

# --- UI TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 X-Ray", "🆚 Benchmark", "📈 Dividends", "🔍 Deep Dive", "👀 Watchlist", "📰 Insights & Updates"])

# --- TAB 1: X-RAY ---
with tab1:
    st.header("Portfolio X-Ray")
    etfs = [x.strip().upper() for x in st.text_input("ETFs to Blend:", DEFAULT_PORT).split(',')]
    cols = st.columns(len(etfs) if len(etfs) > 0 else 1)
    weights = {}
    for i, t in enumerate(etfs):
        with cols[i]: weights[t] = st.slider(f"{t} %", 0, 100, 100//len(etfs)) / 100.0

    if st.button("Analyze Blended Holdings"):
        st.subheader("📈 Blended Performance (Weighted Average)")
        periods = ['1W', '1M', 'YTD', '1Y Total', '3Y CAGR', '5Y CAGR', '10Y CAGR', '15Y CAGR']
        b_stats = {p: 0.0 for p in periods}
        v_weights = {p: 0.0 for p in periods}
        
        for t in etfs:
            if weights[t] > 0:
                stats = get_full_stats(t)
                if stats:
                    for p in periods:
                        if stats.get(p) is not None:
                            b_stats[p] += stats[p] * weights[t]
                            v_weights[p] += weights[t]
        
        m_cols = st.columns(len(periods))
        for i, p in enumerate(periods):
            vw = v_weights[p]
            m_cols[i].metric(p, f"{(b_stats[p] / vw):.2%}" if vw > 0.1 else "N/A")
        st.markdown("---")

        dfs = []
        for t in etfs:
            if weights[t] > 0:
                df = get_holdings(t)
                if not df.empty:
                    df['Weight'] = df['Raw_Weight'] * weights[t]
                    dfs.append(df)
        if dfs:
            full = merge_goog(pd.concat(dfs))
            full['Weight %'] = (full['Weight'] * 100).round(2)
            full['Industry'] = full['Symbol'].apply(lambda x: IND_MAP.get(x, "ETF/Fund"))
            
            c1, c2 = st.columns([2,1])
            with c1: 
                fig = px.treemap(full.head(40), path=[px.Constant("Portfolio"), 'Symbol'], values='Weight %', custom_data=['Industry'], title="Top Holdings")
                fig.update_traces(textinfo="label+value", texttemplate="%{label}<br>%{customdata[0]}<br>%{value:.2f}%")
                st.plotly_chart(fig, use_container_width=True)
            with c2: st.dataframe(full[['Symbol', 'Industry', 'Weight %']].head(20), height=500)

# --- TAB 2: BENCHMARK ---
with tab2:
    st.header("Portfolio vs. Benchmark")
    cp, cm = st.columns(2)
    with cp: p_list = [x.strip().upper() for x in st.text_input("Your Portfolio (Editable):", DEFAULT_PORT).split(',')]
    with cm: m_list = [x.strip().upper() for x in st.text_input("Benchmark(s):", DEFAULT_BENCH).split(',')]
    
    if st.button("Compare"):
        p_stats = [s for s in (get_full_stats(t) for t in p_list) if s]
        m_stats = [s for s in (get_full_stats(t) for t in m_list) if s]
        if p_stats and m_stats:
            df_p = pd.DataFrame(p_stats)
            avg_p = {c: df_p[c].mean() for c in ALL_NUM_COLS if c in df_p.columns}
            avg_p['Ticker'] = "PORTFOLIO AVG"
            avg_p['Inception'] = "-"
            
            final = pd.DataFrame([avg_p] + m_stats)[['Ticker', 'Inception'] + ALL_NUM_COLS].dropna(axis=1, how='all')
            final_formatted = format_dataframe(final)
            st.dataframe(final_formatted, hide_index=True)

# --- TAB 3: DIVIDENDS ---
with tab3:
    st.header("Dividend & Growth Data")
    if st.button("Load Dividends"):
        t_list = [x.strip().upper() for x in st.text_area("Tickers", DEFAULT_PORT).split(',')]
        data = [s for s in (get_full_stats(t) for t in t_list) if s]
        if data:
            df = pd.DataFrame(data)
            avg_data = {c: df[c].mean() for c in ALL_NUM_COLS if c in df.columns}
            avg_data.update({'Ticker': "AVERAGE", 'Inception': "-", 'Industry': "-", 'Price': None, 'Streak': None, 'Freq': "-"})
            
            final = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)
            cols = ['Ticker', 'Price', 'Industry', 'Inception', 'Streak', 'Freq'] + ALL_NUM_COLS
            final = final[[c for c in cols if c in final.columns]]
            
            final_formatted = format_dataframe(final)
            st.dataframe(final_formatted, height=500)

# --- TAB 4: DEEP DIVE ---
with tab4:
    st.header("Multi-ETF Deep Dive")
    if st.button("Inspect ETFs"):
        for t in [x.strip().upper() for x in st.text_input("Tickers:", DEFAULT_PORT).split(',')]:
            st.subheader(f"Analysis for {t} | Inception: {B_INCEPT.get(t, 'N/A')}")
            df = get_holdings(t)
            if not df.empty:
                df = merge_goog(df.rename(columns={'Symbol': 'Symbol', 'Raw_Weight': 'Weight'}))
                df['Weight'] = (df['Weight'] * 100).map('{:.2f}%'.format)
                st.table(df.head(15))

# --- TAB 5: WATCHLIST ---
with tab5:
    st.header("Watchlist & Consideration")
    if st.button("Update Watchlist"):
        tickers = [x.strip().upper() for x in st.text_input("Tickers:", DEFAULT_WATCH).split(',')]
        data = [s for s in (get_full_stats(t) for t in tickers) if s]
        if data:
            df = pd.DataFrame(data)
            cols = ['Ticker', 'Price', 'Industry', 'Inception'] + ALL_NUM_COLS
            final_cols = [c for c in cols if c in df.columns]
            
            final_formatted = format_dataframe(df[final_cols])
            st.dataframe(final_formatted, height=500)

# --- TAB 6: AI NEWS & INSIGHTS ---
with tab6:
    st.header("📰 Deep-Dive Portfolio Insights")
    
    st.markdown("### ⏱️ Dynamic Market Pulse (Updates Instantly)")
    st.caption("Live performance metrics for your core holdings across Daily, Weekly, Monthly, and Annual timeframes.")
    
    pulse_tickers = [x.strip().upper() for x in DEFAULT_PORT.split(',')]
    pulse_data = [s for s in (get_full_stats(t) for t in pulse_tickers) if s]
    
    if pulse_data:
        df_pulse = pd.DataFrame(pulse_data)
        pulse_cols = ['Ticker', 'Price', '1D', '1W', '1M', 'YTD']
        df_pulse = df_pulse[[c for c in pulse_cols if c in df_pulse.columns]]
        st.dataframe(format_dataframe(df_pulse), hide_index=True)
    else:
        st.warning("Could not load real-time market data.")

    st.markdown("---")

    st.markdown("### 🤖 1-Click Gemini Ultra Intelligence Update")
    st.markdown("Use this payload to instantly command Gemini to generate a comprehensive Daily, Weekly, Monthly, and Annual structural review of your live holdings.")
    
    prompt_text = f"Give me a comprehensive news update about my investment portfolio. It is blended equally among: {DEFAULT_PORT}. Format with headings for Daily/Weekly (Short-Term Dynamics), Monthly (Medium-Term Trends), and Annual Outlook (Long-Term Fundamentals)."
    
    st.code(prompt_text, language="text")
    st.markdown("[🔗 **Open Gemini Ultra (Click Here)**](https://gemini.google.com/app)")

    st.markdown("---")

    st.markdown("### 🧠 Executive Summary: Your Portfolio (SCHG, QQQ, VGT, SMH)")
    
    st.markdown("#### 📊 Asset Allocation & Overlap Analysis")
    st.markdown("""
    This portfolio is an aggressive, hyper-concentrated bet on **U.S. Mega-Cap Technology and Semiconductors**. 
    
    * **The Overlap Effect:** Because VGT (Information Technology), SCHG (Large-Cap Growth), and QQQ (Nasdaq 100) utilize market-cap weighting, they overwhelmingly hold the exact same top companies. 
    * **The Semiconductor Tilt:** By dedicating 25% of your portfolio directly to SMH (Semiconductors), you are actively layering semiconductor exposure *on top* of the semiconductor exposure already embedded in VGT, QQQ, and SCHG.
    
    **⚠️ Concentration Vulnerability**
    If you break down the actual underlying holdings across these four ETFs, your true exposure is extraordinarily top-heavy:
    * **Nvidia (NVDA):** You have massive exposure to NVDA. It is the #1 holding in SMH (~20%), a top 3 holding in VGT (~13%), a top 3 holding in QQQ (~7%), and a top 3 holding in SCHG (~11%). **Estimated pure portfolio exposure: ~12-13%.**
    * **Microsoft (MSFT) & Apple (AAPL):** These two companies make up roughly 32% of VGT, 16% of QQQ, and 22% of SCHG. **Estimated pure portfolio exposure to just these two companies: ~17-18%.**
    
    **Bottom Line:** Roughly **1/3 of your entire portfolio** is dictated by the daily price movements of just three companies (NVDA, MSFT, AAPL).
    """)

    st.markdown("---")

    st.markdown("#### 🚀 Primary Performance Drivers (The Bull Case)")
    st.markdown("""
    * **The AI Infrastructure Supercycle:** This portfolio is perfectly positioned to capture the ongoing capital expenditure (CapEx) boom in Artificial Intelligence. As long as hyperscalers (Meta, Google, Microsoft, Amazon) continue pouring billions into data centers and hardware, SMH (providing the chips) and VGT/QQQ (providing the cloud software infrastructure) will structurally outperform the broader market.
    * **Interest Rate Sensitivity:** High-growth technology companies rely heavily on future cash flow valuations. A macroeconomic environment featuring declining inflation and Federal Reserve rate cuts acts as a tailwind, reducing the discount rate and expanding the P/E multiples of SCHG and QQQ.
    * **Margin Expansion:** Unlike standard S&P 500 companies (VOO) which include lower-margin retail and manufacturing, your holdings represent the highest-margin software and hardware monopolies in the global economy.
    """)

    st.markdown("---")

    st.markdown("#### 📉 Risk Management & Vulnerabilities (The Bear Case)")
    st.markdown("""
    * **Zero Defensive Capabilities:** This portfolio contains effectively 0% exposure to defensive sectors (Utilities, Consumer Staples, Healthcare). In a classic recessionary environment or a tech-led market correction (similar to 2022), this portfolio will experience drawdowns significantly deeper than the S&P 500.
    * **Valuation Risk:** Growth stocks are currently priced for perfection. Any sign of slowing AI adoption, delayed hardware rollouts (e.g., Blackwell chip delays), or regulatory antitrust actions against Mega-Cap tech will trigger immediate algorithmic sell-offs.
    * **The Income Gap:** The dividend yield on this portfolio is negligible (sub-0.50%). It does not generate meaningful cash flow to reinvest during market downturns, relying entirely on capital appreciation for Total Return.
    """)

    st.markdown("---")

    st.markdown("#### 🎯 Strategic Considerations for the Future")
    st.markdown("""
    Given your stated financial goal of **Fat FIRE ($8M+ Net Worth)** and your long time horizon, this aggressive posture is mathematically justified, provided you can stomach high volatility. However, consider the following tactical adjustments as your portfolio scales:

    1.  **Introduce Yield/Value (SCHD/VYM):** As your balance grows, mitigating volatility sequence-of-returns risk becomes vital. Allocating 10-15% to a high-quality dividend growth fund (SCHD) provides a stabilizing anchor that performs well during tech corrections.
    2.  **Consolidate Redundancy (VGT vs QQQ):** VGT and QQQ track very similar metrics. You could simplify the portfolio by dropping one and reallocating that 25% into a broader S&P 500 fund (VOO) to capture non-tech growth (Financials, Healthcare, Industrials) without sacrificing your aggressive edge.
    """)

st.markdown("---")
