import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 X-Ray", "🆚 Benchmark", "📈 Dividends", "🔍 Deep Dive", "👀 Watchlist", "📰 AI News & Insights"])

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
            c1, c2 = st.columns([2,1])
            with c1: 
                fig = px.treemap(full.head(40), path=[px.Constant("Portfolio"), 'Symbol'], values='Weight %', title="Top Holdings")
                fig.update_traces(textinfo="label+value", texttemplate="%{label}<br>%{value:.2f}%")
                st.plotly_chart(fig, use_container_width=True)
            with c2: st.dataframe(full[['Symbol', 'Weight %']].head(20), height=500)

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

# --- TAB 6: AI NEWS (LIVE FETCH) ---
with tab6:
    st.header("📰 Live Market News")
    st.info("Aggregating live, free news from Yahoo Finance, Google News, CNBC, and other financial outlets...")
    
    all_tickers = list(set(DEFAULT_PORT.split(',') + DEFAULT_WATCH.split(',')))
    all_tickers = [x.strip().upper() for x in all_tickers if x.strip()]
    
    news_feed = []
    my_bar = st.progress(0, text="Scanning for breaking news...")
    
    for i, t in enumerate(all_tickers):
        # 1. Attempt Yahoo Finance News
        try:
            stock = yf.Ticker(t)
            y_news = stock.news
            if y_news:
                for a in y_news[:5]:
                    dt = datetime.fromtimestamp(a.get('providerPublishTime', 0))
                    news_feed.append({'Ticker': t, 'Title': a.get('title'), 'Publisher': a.get('publisher', 'Yahoo Finance'), 'Link': a.get('link'), 'Time': dt})
        except: pass

        # 2. Attempt Google News RSS (Aggregates CNBC, Reuters, Bloomberg, etc.)
        try:
            url = f"https://news.google.com/rss/search?q={t}+stock&hl=en-US&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//channel/item')[:5]:
                pub_date_str = item.find('pubDate').text
                dt = parsedate_to_datetime(pub_date_str)
                if dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
                source = item.find('source').text if item.find('source') is not None else 'Google News'
                news_feed.append({'Ticker': t, 'Title': item.find('title').text, 'Publisher': source, 'Link': item.find('link').text, 'Time': dt})
        except: pass
        
        my_bar.progress((i + 1) / len(all_tickers), text=f"Fetching latest alerts for {t}...")
        
    my_bar.empty()
    
    if news_feed:
        df_news = pd.DataFrame(news_feed)
        # Clean data and remove duplicate titles from crossing sources
        df_news = df_news.sort_values(by='Time', ascending=False)
        df_news = df_news.drop_duplicates(subset=['Title'])
        
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        daily = df_news[df_news['Time'].dt.date == today]
        weekly = df_news[(df_news['Time'].dt.date < today) & (df_news['Time'].dt.date >= week_ago)]
        monthly = df_news[(df_news['Time'].dt.date < week_ago) & (df_news['Time'].dt.date >= month_ago)]
        
        def display_news_section(title, df):
            if not df.empty:
                st.subheader(title)
                for index, row in df.iterrows():
                    st.markdown(f"**{row['Ticker']}** | [{row['Title']}]({row['Link']})")
                    st.caption(f"Source: {row['Publisher']} • {row['Time'].strftime('%Y-%m-%d %I:%M %p')}")
                    st.markdown("---")

        display_news_section("🔥 Breaking Today", daily)
        display_news_section("📅 This Week", weekly)
        display_news_section("🗓️ This Month", monthly)
    else:
        st.warning("No recent news found. The cloud server may be temporarily blocked from accessing news feeds.")

st.markdown("---")
