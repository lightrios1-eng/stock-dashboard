import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

# --- DEFAULT CONSTANTS ---
DEFAULT_PORTFOLIO = "SCHG, QQQ, VGT, SMH"
DEFAULT_BENCHMARK = "VOO, SCHD"
DEFAULT_WATCHLIST = "SCHD, VYM, VIG, VOO, SCHG, QQQ, VGT, SMH, VIG"

# --- COMPACT DATA BANKS (Failsafes) ---
BACKUP_HOLDINGS = {
    "SMH": [["NVDA", 0.20], ["TSM", 0.12], ["AVGO", 0.08], ["AMD", 0.05], ["ASML", 0.05], ["LRCX", 0.04], ["MU", 0.04], ["AMAT", 0.04], ["TXN", 0.04], ["INTC", 0.03]],
    "QQQ": [["AAPL", 0.08], ["MSFT", 0.08], ["NVDA", 0.07], ["AMZN", 0.05], ["META", 0.04], ["AVGO", 0.04], ["GOOGL", 0.03], ["GOOG", 0.03], ["TSLA", 0.02], ["COST", 0.02]],
    "MGK": [["AAPL", 0.13], ["MSFT", 0.12], ["NVDA", 0.11], ["AMZN", 0.06], ["META", 0.04], ["GOOGL", 0.04], ["GOOG", 0.03], ["TSLA", 0.03], ["AVGO", 0.03], ["LLY", 0.02]],
    "SCHG": [["NVDA", 0.11], ["MSFT", 0.11], ["AAPL", 0.10], ["AMZN", 0.06], ["META", 0.04], ["GOOGL", 0.04], ["GOOG", 0.03], ["AVGO", 0.03], ["TSLA", 0.03], ["LLY", 0.02]],
    "FTEC": [["AAPL", 0.21], ["MSFT", 0.19], ["NVDA", 0.15], ["AVGO", 0.05], ["CRM", 0.02], ["ADBE", 0.02], ["ORCL", 0.02], ["AMD", 0.02], ["ACN", 0.02], ["CSCO", 0.01]],
    "VOO": [["MSFT", 0.07], ["AAPL", 0.06], ["NVDA", 0.06], ["AMZN", 0.03], ["META", 0.02], ["GOOGL", 0.02], ["GOOG", 0.01], ["AVGO", 0.01], ["LLY", 0.01], ["TSLA", 0.01]],
    "SPY": [["MSFT", 0.07], ["AAPL", 0.06], ["NVDA", 0.06], ["AMZN", 0.03], ["META", 0.02], ["GOOGL", 0.02], ["GOOG", 0.01], ["AVGO", 0.01], ["LLY", 0.01], ["TSLA", 0.01]],
    "VGT": [["AAPL", 0.16], ["MSFT", 0.16], ["NVDA", 0.13], ["AVGO", 0.04], ["CRM", 0.02], ["ACN", 0.02], ["ADBE", 0.02], ["AMD", 0.02], ["CSCO", 0.01], ["INTC", 0.01]],
    "VYM": [["JPM", 0.03], ["AVGO", 0.03], ["XOM", 0.03], ["JNJ", 0.02], ["HD", 0.02], ["PG", 0.02], ["COST", 0.02], ["ABBV", 0.02], ["MRK", 0.02], ["CVX", 0.01]],
    "SCHD": [["ABBV", 0.04], ["AVGO", 0.04], ["CVX", 0.04], ["KO", 0.04], ["PEP", 0.04], ["MRK", 0.04], ["HD", 0.04], ["TXN", 0.04], ["CSCO", 0.04], ["AMGN", 0.04]]
}

BACKUP_CAGR_10Y = {"SMH": 0.285, "QQQ": 0.182, "MGK": 0.195, "SCHG": 0.188, "FTEC": 0.205, "VOO": 0.128, "SPY": 0.128, "VGT": 0.192, "VYM": 0.095, "SCHD": 0.115, "JEPQ": 0.105}
BACKUP_INCEPTION = {"SMH": "2011-12-20", "QQQ": "1999-03-10", "MGK": "2007-12-17", "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07", "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10", "SCHD": "2011-10-20", "JEPQ": "2022-05-03"}

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Portfolio X-Ray", "🆚 Portfolio vs. Benchmark", "📈 Dividend & Growth", 
    "🔍 Multi-ETF Deep Dive", "👀 Watchlist", "📰 AI News & Insights"
])

# --- CORE LOGIC ENGINE ---
def merge_google(df):
    df = df.copy()
    df['Symbol'] = df['Symbol'].replace({'GOOG': 'GOOG/L', 'GOOGL': 'GOOG/L'})
    return df.groupby('Symbol', as_index=False)['Weight'].sum().sort_values(by='Weight', ascending=False).reset_index(drop=True)

def get_inception_date(ticker, info=None):
    if info and info.get('fundInceptionDate'): return datetime.fromtimestamp(info.get('fundInceptionDate')).strftime('%Y-%m-%d')
    try:
        stock = yf.Ticker(ticker)
        if stock.info.get('fundInceptionDate'): return datetime.fromtimestamp(stock.info.get('fundInceptionDate')).strftime('%Y-%m-%d')
    except: pass
    return BACKUP_INCEPTION.get(ticker, "N/A")

def get_cagr_div(end, start, years):
    if start == 0 or pd.isna(start) or pd.isna(end): return None
    return (end / start) ** (1 / years) - 1

def get_streak_and_freq(div_hist):
    if div_hist.empty: return 0, "-"
    annual = div_hist.groupby(div_hist.index.year).sum().sort_index(ascending=False)
    curr_year = datetime.now().year
    streak = 0
    completed = annual[annual.index < curr_year]
    if len(completed) < 2: return 0, "-"
    for i in range(len(completed) - 1):
        if completed.iloc[i] > completed.iloc[i+1]: streak += 1
        else: break
    last_yr_count = div_hist[div_hist.index.year == (curr_year - 1)].count()
    if last_yr_count >= 11: freq = "Mo"
    elif last_yr_count >= 3: freq = "Qr"
    elif last_yr_count >= 1: freq = "Yr"
    else: freq = "-"
    return streak, freq

def get_full_stats(ticker):
    """Fetches Live Metrics, Returns, CAGR, and Dynamic Sector/Industry Data."""
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="max", auto_adjust=True)
        div_hist = stock.dividends
        info = stock.info
    except: return None
    
    if hist.empty: return None
    price = hist['Close'].iloc[-1]
    
    # Dynamic Sector & Industry Fetching (Solves token limits)
    sector = info.get('sector', 'ETF / Fund')
    industry = info.get('industry', 'Diversified')
    if 'ETF' in info.get('quoteType', '').upper() or 'MUTUALFUND' in info.get('quoteType', '').upper():
        sector = "ETF / Fund"
        industry = "ETF / Fund"
    
    # Dividends
    yield_ttm = div_hist[div_hist.index >= (pd.Timestamp.now().tz_localize(div_hist.index.dtype.tz) - pd.Timedelta(days=365))].sum() / price if not div_hist.empty else 0
    yield_fwd = yield_ttm if 'ETF' in info.get('quoteType', '').upper() else (info.get('dividendRate', 0) / price if info.get('dividendRate', 0) > 0 else (info.get('dividendYield', 0)/100 if info.get('dividendYield', 0) > 0.25 else info.get('dividendYield', 0)))

    streak, freq = get_streak_and_freq(div_hist)
    metrics = {
        'Ticker': ticker, 'Price': price, 'Sector': sector, 'Industry': industry,
        'Inception': get_inception_date(ticker, info),
        'Yield (TTM)': yield_ttm, 'Yield (Fwd)': yield_fwd,
        'Streak': streak, 'Freq': freq
    }
    
    # Short Term
    metrics['1D'] = (price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] if len(hist) > 1 else None
    metrics['1W'] = (price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] if len(hist) > 5 else None
    metrics['1M'] = (price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22] if len(hist) > 21 else None
    hist_ytd = hist[hist.index >= pd.Timestamp(f"{datetime.now().year}-01-01").tz_localize(hist.index.dtype.tz)]
    metrics['YTD'] = (price - hist_ytd['Open'].iloc[0]) / hist_ytd['Open'].iloc[0] if not hist_ytd.empty else None

    # Long Term Total Return & CAGR
    for y in [1, 3, 5, 10, 15]:
        target = hist.index[-1] - timedelta(days=y*365)
        idx_loc = hist.index.get_indexer([target], method='nearest')
        if len(idx_loc) > 0 and abs((hist.index[idx_loc[0]] - target).days) < 100:
            sp = hist['Close'].iloc[idx_loc[0]]
            metrics[f'{y}Y Total'] = (price - sp) / sp
            metrics[f'{y}Y CAGR'] = (price / sp) ** (1 / y) - 1 if y > 1 else None
        else:
            metrics[f'{y}Y Total'], metrics[f'{y}Y CAGR'] = None, None

    # Dividend Growth
    if not div_hist.empty:
        annual = div_hist.groupby(div_hist.index.year).sum()
        ly = datetime.now().year - 1
        for y in [3, 5, 10, 15]:
            if ly in annual.index and (ly - y) in annual.index:
                metrics[f'{y}Y Div CAGR'] = get_cagr_div(annual.loc[ly], annual.loc[ly - y], y)
            else: metrics[f'{y}Y Div CAGR'] = None
    else:
        for y in [3, 5, 10, 15]: metrics[f'{y}Y Div CAGR'] = None
        
    return metrics

def get_holdings_robust(ticker):
    try:
        df = yf.Ticker(ticker).funds_data.top_holdings
        if not df.empty:
            df = df.to_frame() if isinstance(df, pd.Series) else df
            df = df.reset_index().iloc[:, [0, -1]]
            df.columns = ['Symbol', 'Raw_Weight']
            return df, "Live Data"
    except: pass
    if ticker in BACKUP_HOLDINGS: return pd.DataFrame(BACKUP_HOLDINGS[ticker], columns=['Symbol', 'Raw_Weight']), "Backup Data"
    return pd.DataFrame(), "Failed"

# ==========================================
# TAB 1: PORTFOLIO X-RAY
# ==========================================
with tab1:
    st.header("See what you actually own")
    etf_list = [x.strip().upper() for x in st.text_input("Enter ETFs to Blend (comma separated):", value=DEFAULT_PORTFOLIO).split(',')]

    cols = st.columns(3)
    weights = {}
    for i, ticker in enumerate(etf_list):
        with cols[i % 3]:
            w = st.slider(f"{ticker} %", 0, 100, 100 // len(etf_list), key=f"s_{ticker}")
            weights[ticker] = w / 100.0

    if st.button("Analyze Blended Holdings"):
        st.markdown("### 📈 Blended Performance (Annualized)")
        timeframes = [1, 3, 5, 10, 15]
        blended_stats = {y: 0.0 for y in timeframes}
        valid_weights = {y: 0.0 for y in timeframes}
        
        for t in etf_list:
            if weights[t] > 0:
                stats = get_full_stats(t)
                if stats:
                    for y in timeframes:
                        val = stats.get(f'{y}Y CAGR' if y > 1 else '1Y Total')
                        if val is not None:
                            blended_stats[y] += val * weights[t]
                            valid_weights[y] += weights[t]
        
        m_cols = st.columns(5)
        for i, y in enumerate(timeframes):
            vw = valid_weights[y]
            m_cols[i].metric(f"{y}-Year", f"{(blended_stats[y] / vw):.2%}" if vw > 0.1 else "N/A")
        st.markdown("---")

        all_holdings = []
        for t in etf_list:
            if weights[t] > 0:
                df, _ = get_holdings_robust(t)
                if not df.empty:
                    df['Weight'] = df['Raw_Weight'] * weights[t]
                    all_holdings.append(df)

        if all_holdings:
            full_df = pd.concat(all_holdings)
            grouped = merge_google(full_df)
            
            # Fetch dynamic sector/industry for top 40 overlapping holdings to save time
            top_holdings = grouped.head(40)['Symbol'].tolist()
            sec_ind_map = {t: get_full_stats(t) for t in top_holdings}
            
            grouped['Sector'] = grouped['Symbol'].apply(lambda x: sec_ind_map.get(x, {}).get('Sector', 'Other') if x in sec_ind_map else 'Other')
            grouped['Industry'] = grouped['Symbol'].apply(lambda x: sec_ind_map.get(x, {}).get('Industry', 'Other') if x in sec_ind_map else 'Other')
            grouped['Weight %'] = (grouped['Weight'] * 100).round(2)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.treemap(grouped.head(40), path=[px.Constant("Portfolio"), 'Sector', 'Symbol'], values='Weight %', title="Portfolio Composition by Sector", color='Sector')
                fig.update_traces(textinfo="label+value", texttemplate="%{label}<br>%{value}%")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(grouped[['Symbol', 'Sector', 'Industry', 'Weight %']].head(20), height=500)

# ==========================================
# TAB 2: PORTFOLIO vs BENCHMARK
# ==========================================
with tab2:
    st.header("🆚 Portfolio vs. Benchmark")
    col_p, col_m = st.columns(2)
    with col_p: p_list = [x.strip().upper() for x in st.text_input("Your Portfolio (Editable):", value=DEFAULT_PORTFOLIO).split(',')]
    with col_m: m_list = [x.strip().upper() for x in st.text_input("Benchmark(s):", value=DEFAULT_BENCHMARK).split(',')]
        
    if st.button("Compare Performance"):
        p_stats = [s for s in (get_full_stats(t) for t in p_list) if s]
        m_stats = [s for s in (get_full_stats(t) for t in m_list) if s]
        
        if p_stats and m_stats:
            df_p = pd.DataFrame(p_stats)
            numeric_cols = ['Yield (TTM)', 'Yield (Fwd)', '1D', '1W', '1M', 'YTD', '1Y Total', '3Y Total', '3Y CAGR', '5Y Total', '5Y CAGR', '10Y Total', '10Y CAGR', '15Y Total', '15Y CAGR', '3Y Div CAGR', '5Y Div CAGR', '10Y Div CAGR', '15Y Div CAGR']
            
            avg_p = {col: df_p[col].mean() for col in numeric_cols if col in df_p.columns}
            avg_p['Ticker'] = "YOUR PORTFOLIO (Avg)"
            
            df_final = pd.DataFrame([avg_p] + m_stats)[['Ticker'] + numeric_cols].dropna(axis=1, how='all')
            st.dataframe(df_final.style.format({c: '{:.2%}' for c in numeric_cols}, na_rep="-"), hide_index=True)
            
            st.markdown("### 📊 Annualized Return (CAGR) Comparison")
            chart_data = [{'Period': c, 'Return': avg_p[c], 'Type': 'Your Portfolio'} for c in ['1Y Total', '3Y CAGR', '5Y CAGR', '10Y CAGR', '15Y CAGR'] if c in avg_p]
            chart_data += [{'Period': c, 'Return': m[c], 'Type': m['Ticker']} for m in m_stats for c in ['1Y Total', '3Y CAGR', '5Y CAGR', '10Y CAGR', '15Y CAGR'] if c in m]
            st.plotly_chart(px.bar(pd.DataFrame(chart_data), x='Period', y='Return', color='Type', barmode='group'), use_container_width=True)

# ==========================================
# TAB 3: DIVIDENDS
# ==========================================
with tab3:
    st.header("📈 Dividend & Growth Data")
    if st.button("Update Dividends"):
        t_list = [x.strip().upper() for x in st.text_area("Tickers", DEFAULT_PORTFOLIO).split(',')]
        data, progress = [], st.progress(0)
        for i, t in enumerate(t_list):
            if res := get_full_stats(t): data.append(res)
            progress.progress((i + 1) / len(t_list))
        
        if data:
            df = pd.DataFrame(data)
            numeric_cols = ['Yield (TTM)', 'Yield (Fwd)', '1D', '1W', '1M', 'YTD', '1Y Total', '3Y Total', '3Y CAGR', '5Y Total', '5Y CAGR', '10Y Total', '10Y CAGR', '15Y Total', '15Y CAGR', '3Y Div CAGR', '5Y Div CAGR', '10Y Div CAGR', '15Y Div CAGR']
            avg_data = {col: df[col].mean() for col in numeric_cols if col in df.columns}
            avg_data.update({'Ticker': "AVERAGE", 'Industry': "-", 'Inception': "-", 'Price': None})
            
            df_final = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)
            cols = ['Ticker', 'Price', 'Industry', 'Inception', 'Streak', 'Freq'] + numeric_cols
            fmt = {c: '{:.2%}' for c in numeric_cols}
            fmt['Price'] = '${:.2f}'
            st.dataframe(df_final[[c for c in cols if c in df_final.columns]].style.format(fmt, na_rep="-"), height=600)

# ==========================================
# TAB 4: DEEP DIVE
# ==========================================
with tab
