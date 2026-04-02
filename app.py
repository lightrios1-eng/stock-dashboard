import streamlit as st, yfinance as yf, pandas as pd, plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

DEFAULT_PORT = "SCHG, QQQ, VGT, SMH"
DEFAULT_BENCH = "VOO, SCHD"
DEFAULT_WATCH = "SCHD, VYM, VIG, VOO, SCHG, QQQ, VGT, SMH, VIG"

# Compacted Backup Holdings
BACKUP_HOLDINGS = {
    "SMH": [["NVDA", 0.20], ["TSM", 0.12], ["AVGO", 0.08], ["AMD", 0.05], ["ASML", 0.05], ["LRCX", 0.04], ["MU", 0.04], ["AMAT", 0.04], ["TXN", 0.04], ["INTC", 0.03]],
    "QQQ": [["AAPL", 0.08], ["MSFT", 0.08], ["NVDA", 0.07], ["AMZN", 0.05], ["META", 0.04], ["AVGO", 0.04], ["GOOGL", 0.03], ["GOOG", 0.03], ["TSLA", 0.02], ["COST", 0.02]],
    "SCHG": [["NVDA", 0.11], ["MSFT", 0.11], ["AAPL", 0.10], ["AMZN", 0.06], ["META", 0.04], ["GOOGL", 0.04], ["GOOG", 0.03], ["AVGO", 0.03], ["TSLA", 0.03], ["LLY", 0.02]],
    "VGT": [["AAPL", 0.16], ["MSFT", 0.16], ["NVDA", 0.13], ["AVGO", 0.04], ["CRM", 0.02], ["ACN", 0.02], ["ADBE", 0.02], ["AMD", 0.02], ["CSCO", 0.01], ["INTC", 0.01]]
}

# --- CORE ENGINE ---
@st.cache_data(ttl=3600)
def get_stats(ticker):
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

    annual_div = div.groupby(div.index.year).sum() if not div.empty else pd.Series()
    
    m = {'Ticker': ticker, 'Price': p, 'Yield (Fwd)': y_fwd, 'Yield (TTM)': y_ttm}
    m['1D'] = (p - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] if len(hist)>1 else None
    m['1W'] = (p - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] if len(hist)>5 else None
    
    ytd_data = hist[hist.index >= pd.Timestamp(f"{datetime.now().year}-01-01").tz_localize(hist.index.dtype.tz)]
    m['YTD'] = (p - ytd_data['Open'].iloc[0]) / ytd_data['Open'].iloc[0] if not ytd_data.empty else None

    for y in [1, 3, 5, 10, 15]:
        target = hist.index[-1] - timedelta(days=y*365)
        idx = hist.index.get_indexer([target], method='nearest')[0]
        if idx >= 0 and abs((hist.index[idx] - target).days) < 100:
            sp = hist['Close'].iloc[idx]
            m[f'{y}Y Total'] = (p - sp) / sp
            m[f'{y}Y CAGR'] = (p / sp) ** (1/y) - 1 if y > 1 else None
        else: m[f'{y}Y Total'], m[f'{y}Y CAGR'] = None, None

        if not annual_div.empty:
            ly = datetime.now().year - 1
            m[f'{y}Y Div CAGR'] = ((annual_div.loc[ly] / annual_div.loc[ly-y]) ** (1/y) - 1) if (ly in annual_div.index and (ly-y) in annual_div.index and annual_div.loc[ly-y]>0) else None
        else: m[f'{y}Y Div CAGR'] = None

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
    if ticker in BACKUP_HOLDINGS: return pd.DataFrame(BACKUP_HOLDINGS[ticker], columns=['Symbol', 'Raw_Weight'])
    return pd.DataFrame()

# --- UI TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 X-Ray", "🆚 Benchmark", "📈 Dividends", "🔍 Deep Dive", "👀 Watchlist", "📰 News Feed"])

with tab1:
    st.header("Portfolio X-Ray")
    etfs = [x.strip().upper() for x in st.text_input("ETFs to Blend:", DEFAULT_PORT).split(',')]
    cols = st.columns(3)
    weights = {}
    for i, t in enumerate(etfs):
        with cols[i%3]: weights[t] = st.slider(f"{t} %", 0, 100, 100//len(etfs)) / 100.0

    if st.button("Analyze Blended Holdings"):
        dfs = []
        for t in etfs:
            if weights[t] > 0:
                df = get_holdings(t)
                if not df.empty:
                    df['Weight'] = df['Raw_Weight'] * weights[t]
                    dfs.append(df)
        if dfs:
            full = pd.concat(dfs).groupby('Symbol', as_index=False)['Weight'].sum().sort_values(by='Weight', ascending=False)
            full['Weight %'] = (full['Weight'] * 100).round(2)
            c1, c2 = st.columns([2,1])
            with c1: st.plotly_chart(px.treemap(full.head(30), path=[px.Constant("Portfolio"), 'Symbol'], values='Weight %', title="Top Holdings"), use_container_width=True)
            with c2: st.dataframe(full[['Symbol', 'Weight %']].head(20), height=500)

with tab2:
    st.header("Portfolio vs. Benchmark")
    cp, cm = st.columns(2)
    with cp: p_list = [x.strip().upper() for x in st.text_input("Your Portfolio:", DEFAULT_PORT).split(',')]
    with cm: m_list = [x.strip().upper() for x in st.text_input("Benchmark(s):", DEFAULT_BENCH).split(',')]
    
    if st.button("Compare"):
        p_stats = [s for s in (get_stats(t) for t in p_list) if s]
        m_stats = [s for s in (get_stats(t) for t in m_list) if s]
        if p_stats and m_stats:
            df_p = pd.DataFrame(p_stats)
            num_cols = ['1D', '1W', 'YTD', '1Y Total', '3Y Total', '3Y CAGR', '5Y Total', '5Y CAGR', '10Y Total', '10Y CAGR', '15Y Total', '15Y CAGR']
            avg_p = {c: df_p[c].mean() for c in num_cols if c in df_p.columns}
            avg_p['Ticker'] = "PORTFOLIO AVG"
            final = pd.DataFrame([avg_p] + m_stats)[['Ticker'] + num_cols].dropna(axis=1, how='all')
            st.dataframe(final.style.format({c: '{:.2%}' for c in num_cols}, na_rep="-"), hide_index=True)

with tab3:
    st.header("Dividend & Growth Data")
    if st.button("Load Dividends"):
        t_list = [x.strip().upper() for x in st.text_area("Tickers", DEFAULT_PORT).split(',')]
        data = [s for s in (get_stats(t) for t in t_list) if s]
        if data:
            df = pd.DataFrame(data)
            cols = ['Ticker', 'Price', 'Yield (TTM)', 'Yield (Fwd)', '1Y Total', '3Y CAGR', '5Y CAGR', '10Y CAGR', '15Y CAGR', '3Y Div CAGR', '10Y Div CAGR']
            fmt = {c: '{:.2%}' for c in cols if 'Yield' in c or 'CAGR' in c or 'Total' in c}
            fmt['Price'] = '${:.2f}'
            st.dataframe(df[[c for c in cols if c in df.columns]].style.format(fmt, na_rep="-"))

with tab4:
    st.header("Multi-ETF Deep Dive")
    if st.button("Inspect ETFs"):
        for t in [x.strip().upper() for x in st.text_input("Tickers:", DEFAULT_PORT).split(',')]:
            st.subheader(f"Analysis for {t}")
            df = get_holdings(t)
            if not df.empty:
                df['Weight'] = (df['Raw_Weight'] * 100).map('{:.2f}%'.format)
                st.table(df[['Symbol', 'Weight']].head(10))

with tab5:
    st.header("Watchlist")
    if st.button("Update Watchlist"):
        tickers = [x.strip().upper() for x in st.text_input("Tickers:", DEFAULT_WATCH).split(',')]
        data = [s for s in (get_stats(t) for t in tickers) if s]
        if data:
            df = pd.DataFrame(data)
            cols = ['Ticker', 'Price', 'Yield (Fwd)', '1D', '1W', 'YTD', '1Y Total', '3Y Total', '3Y CAGR', '5Y Total', '5Y CAGR', '10Y Total', '10Y CAGR', '15Y Total', '15Y CAGR']
            fmt = {c: '{:.2%}' for c in cols if c not in ['Ticker', 'Price']}
            fmt['Price'] = '${:.2f}'
            st.dataframe(df[[c for c in cols if c in df.columns]].style.format(fmt, na_rep="-"))

with tab6:
    st.header("📰 News Feed")
    st.info("Pulling latest headlines directly from Yahoo Finance.")
    
    all_t = list(set(DEFAULT_PORT.split(',') + DEFAULT_WATCH.split(',')))
    all_t = [x.strip().upper() for x in all_t if x.strip()]
    
    news_feed = []
    bar = st.progress(0)
    for i, t in enumerate(all_t):
        try:
            n = yf.Ticker(t).news
            if n:
                for a in n:
                    news_feed.append({'Ticker': t, 'Title': a.get('title'), 'Publisher': a.get('publisher'), 'Time': datetime.fromtimestamp(a.get('providerPublishTime', 0))})
        except: pass
        bar.progress((i+1)/len(all_t))
    bar.empty()
    
    if news_feed:
        df_news = pd.DataFrame(news_feed).sort_values(by='Time', ascending=False).drop_duplicates(subset=['Title'])
        today = datetime.now().date()
        
        def show_news(title, df):
            if not df.empty:
                st.subheader(title)
                for _, r in df.iterrows():
                    st.markdown(f"**{r['Ticker']}** | {r['Title']} _({r['Publisher']})_")
                    
        show_news("🌞 Today", df_news[df_news['Time'].dt.date == today])
        show_news("📅 This Week", df_news[(df_news['Time'].dt.date < today) & (df_news['Time'].dt.date >= today - timedelta(days=7))])
        show_news("🗓️ This Month", df_news[(df_news['Time'].dt.date < today - timedelta(days=7)) & (df_news['Time'].dt.date >= today - timedelta(days=30))])
    else:
        st.warning("No news fetched. Provider may be blocking access.")

st.markdown("---")
