import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: X-Ray, Dividends & Holdings")

# --- BACKUP DATA BANKS (Failsafe) ---
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

# 10Y Backup only.
BACKUP_CAGR_10Y = {
    "SMH": 0.285, "QQQ": 0.182, "MGK": 0.195, "SCHG": 0.188, 
    "FTEC": 0.205, "VOO": 0.128, "SPY": 0.128, "VGT": 0.192, 
    "VYM": 0.095, "SCHD": 0.115, "JEPQ": 0.105
}

# Inception Date Backup
BACKUP_INCEPTION = {
    "SMH": "2011-12-20", "QQQ": "1999-03-10", "MGK": "2007-12-17", 
    "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07", 
    "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10", 
    "SCHD": "2011-10-20", "JEPQ": "2022-05-03"
}

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Portfolio X-Ray", "📈 Dividend & Growth Data", "🔍 Multi-ETF Deep Dive"])

# --- SHARED HELPERS ---
def merge_google(df, symbol_col='Symbol', weight_col='Weight'):
    """Combines GOOG and GOOGL into a single entry."""
    df = df.copy()
    df[symbol_col] = df[symbol_col].replace({'GOOG': 'GOOG/L', 'GOOGL': 'GOOG/L'})
    df = df.groupby(symbol_col, as_index=False)[weight_col].sum()
    df = df.sort_values(by=weight_col, ascending=False).reset_index(drop=True)
    return df

def get_inception_date(ticker, info=None):
    """Fetches inception date from Info or Backup."""
    if info:
        ts = info.get('fundInceptionDate')
        if ts: return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    try:
        if not info:
            stock = yf.Ticker(ticker)
            ts = stock.info.get('fundInceptionDate')
            if ts: return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    except: pass
    if ticker in BACKUP_INCEPTION: return BACKUP_INCEPTION[ticker]
    return "N/A"

def get_cagr_for_year(ticker, years):
    """Calculates CAGR for a specific year count."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="max", auto_adjust=True)
        if hist.empty: return None
        
        end_date = hist.index[-1]
        start_date_target = end_date - timedelta(days=years*365)
        first_available_date = hist.index[0]
        
        if first_available_date > (start_date_target + timedelta(days=60)):
            if years == 10 and ticker in BACKUP_CAGR_10Y: return BACKUP_CAGR_10Y[ticker]
            return None

        idx_loc = hist.index.get_indexer([start_date_target], method='nearest')
        if len(idx_loc) > 0 and idx_loc[0] >= 0:
            idx = idx_loc[0]
            start_price = hist['Close'].iloc[idx]
            end_price = hist['Close'].iloc[-1]
            return (end_price / start_price) ** (1 / years) - 1
    except: pass
    if years == 10 and ticker in BACKUP_CAGR_10Y: return BACKUP_CAGR_10Y[ticker]
    return None

def get_holdings_robust(ticker):
    stock = yf.Ticker(ticker)
    try:
        df = stock.funds_data.top_holdings
        if not df.empty:
            if isinstance(df, pd.Series): df = df.to_frame()
            df = df.reset_index()
            df = df.iloc[:, [0, -1]]
            df.columns = ['Symbol', 'Raw_Weight']
            return df, "Live Data"
    except: pass
    if ticker in BACKUP_HOLDINGS:
        data = BACKUP_HOLDINGS[ticker]
        df = pd.DataFrame(data, columns=['Symbol', 'Raw_Weight'])
        return df, "Backup Data"
    return pd.DataFrame(), "Failed"

# ==========================================
# TAB 1: PORTFOLIO X-RAY
# ==========================================
with tab1:
    st.header("See what you actually own")
    default_etfs = "SMH, QQQ, MGK, SCHG, FTEC"
    etf_input = st.text_input("Enter ETFs to Blend (comma separated):", value=default_etfs)
    etf_list = [x.strip().upper() for x in etf_input.split(',')]

    cols = st.columns(3)
    weights = {}
    total_weight = 0
    for i, ticker in enumerate(etf_list):
        with cols[i % 3]:
            default_w = 100 // len(etf_list)
            w = st.slider(f"{ticker} %", 0, 100, default_w, key=f"slider_{ticker}")
            weights[ticker] = w / 100.0
            total_weight += w

    if st.button("Analyze Blended Holdings"):
        st.markdown("### 📈 Blended Performance (Annualized)")
        
        timeframes = [1, 3, 5, 10, 15]
        blended_stats = {year: 0.0 for year in timeframes}
        valid_weights = {year: 0.0 for year in timeframes}
        
        perf_table_data = []
        for ticker in etf_list:
            weight = weights[ticker]
            if weight > 0:
                inc_date = get_inception_date(ticker)
                cagr_10 = get_cagr_for_year(ticker, 10)
                perf_table_data.append({
                    "Ticker": ticker,
                    "Allocation": f"{weight*100:.0f}%",
                    "Inception": inc_date,
                    "10Y CAGR": f"{cagr_10:.2%}" if cagr_10 else "N/A"
                })

                for year in timeframes:
                    cagr = get_cagr_for_year(ticker, year)
                    if cagr is not None:
                        blended_stats[year] += cagr * weight
                        valid_weights[year] += weight
        
        final_display = {}
        for year in timeframes:
            vw = valid_weights[year]
            if vw > 0.10: final_display[year] = (blended_stats[year] / vw)
            else: final_display[year] = None
        
        cols = st.columns(5)
        labels = ["1-Year", "3-Year", "5-Year", "10-Year", "15-Year"]
        for i, year in enumerate(timeframes):
            val = final_display[year]
            cols[i].metric(labels[i], f"{val:.2%}" if val is not None else "N/A")
        
        st.caption("*Weighted Average CAGR. ETFs too young for a timeframe are excluded from that average.*")
        st.dataframe(pd.DataFrame(perf_table_data), hide_index=True)
        st.markdown("---")

        all_holdings = []
        status_text = []
        for ticker in etf_list:
            if weights[ticker] > 0:
                df, source = get_holdings_robust(ticker)
                if not df.empty:
                    df['Portfolio_Weight'] = df['Raw_Weight'] * weights[ticker]
                    all_holdings.append(df)
                    if source == "Backup Data": status_text.append(f"⚠️ {ticker}: Used backup data.")
                else: status_text.append(f"❌ {ticker}: No data found.")

        if status_text: st.caption(" | ".join(status_text))

        if all_holdings:
            full_df = pd.concat(all_holdings)
            full_df = merge_google(full_df, symbol_col='Symbol', weight_col='Portfolio_Weight')
            grouped = full_df.groupby('Symbol')['Portfolio_Weight'].sum().reset_index()
            grouped = grouped.sort_values(by='Portfolio_Weight', ascending=False)
            grouped['Weight %'] = (grouped['Portfolio_Weight'] * 100).round(2)
            
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.treemap(grouped.head(30), path=['Symbol'], values='Portfolio_Weight', 
                                 title="Top Overlapping Holdings", color='Portfolio_Weight', color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(grouped[['Symbol', 'Weight %']].head(20), height=500)
        else: st.warning("Could not calculate holdings. Check spelling.")

# ==========================================
# TAB 2: DIVIDEND DATA (WITH SHORT-TERM PERF)
# ==========================================
with tab2:
    def get_cagr_div(end, start, years):
        if start == 0 or pd.isna(start) or pd.isna(end): return None
        return (end / start) ** (1 / years) - 1

    def get_streak_and_freq(div_hist):
        if div_hist.empty: return 0, "-"
        annual = div_hist.resample('Y').sum().sort_index(ascending=False)
        curr_year = datetime.now().year
        streak = 0
        completed = annual[annual.index.year < curr_year]
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

    def get_div_data(ticker):
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period="20y", auto_adjust=True)
            div_hist = stock.dividends
            info = stock.info
        except: return None
        
        if hist.empty: return None
        price = hist['Close'].iloc[-1]
        
        if not div_hist.empty:
            now = pd.Timestamp.now().tz_localize(div_hist.index.dtype.tz)
            ttm_divs = div_hist[div_hist.index >= (now - pd.Timedelta(days=365))].sum()
            yield_ttm = ttm_divs / price
        else: yield_ttm = 0
        
        quote_type = info.get('quoteType', '').upper()
        if 'ETF' in quote_type: yield_fwd = yield_ttm
        else:
            rate = info.get('dividendRate', 0)
            if rate and rate > 0: yield_fwd = rate / price
            else:
                raw = info.get('dividendYield', 0)
                if raw > 0.25: yield_fwd = raw / 100
                else: yield_fwd = raw

        ex_div = info.get('exDividendDate', None)
        ex_div = datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d') if ex_div else "-"
        payout = info.get('payoutDate', None)
        payout = datetime.fromtimestamp(payout).strftime('%Y-%m-%d') if payout else "-"
        
        streak, freq = get_streak_and_freq(div_hist)
        inc_date = get_inception_date(ticker, info)

        metrics = {
            'Ticker': ticker, 'Price': price, 
            'Inception': inc_date,
            'Yield (TTM)': yield_ttm, 'Yield (Fwd)': yield_fwd,
            'Streak': streak, 'Freq': freq,
            'Ex-Div': ex_div, 'Payout': payout
        }
        
        # --- SHORT TERM PERFORMANCE ---
        # 1-Day
        if len(hist) > 1:
            metrics['1D'] = (price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
        else: metrics['1D'] = None
        
        # 1-Week (5 trading days)
        if len(hist) > 5:
            metrics['1W'] = (price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]
        else: metrics['1W'] = None
        
        # 1-Month (21 trading days)
        if len(hist) > 21:
            metrics['1M'] = (price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]
        else: metrics['1M'] = None

        # YTD
        current_year = datetime.now().year
        ytd_start = f"{current_year}-01-01"
        hist_ytd = hist[hist.index >= pd.Timestamp(ytd_start).tz_localize(hist.index.dtype.tz)]
        if not hist_ytd.empty:
            start_price_ytd = hist_ytd['Open'].iloc[0] # Use first open of year
            metrics['YTD'] = (price - start_price_ytd) / start_price_ytd
        else:
            metrics['YTD'] = None

        # --- LONG TERM PERFORMANCE ---
        curr_date = hist.index[-1]
        for y in [1, 3, 5, 10, 15]:
            target = curr_date - timedelta(days=y*365)
            idx_loc = hist.index.get_indexer([target], method='nearest')
            if len(idx_loc) > 0:
                idx = idx_loc[0]
                if abs((hist.index[idx] - target).days) < 100:
                    start_price = hist['Close'].iloc[idx]
                    end_price = price
                    total_ret = (end_price - start_price) / start_price
                    metrics[f'{y}Y Total'] = total_ret
                    if y > 1:
                        cagr = (end_price / start_price) ** (1 / y) - 1
                        metrics[f'{y}Y CAGR'] = cagr
                else: 
                    metrics[f'{y}Y Total'] = None
                    if y > 1: metrics[f'{y}Y CAGR'] = None
            else: 
                metrics[f'{y}Y Total'] = None
                if y > 1: metrics[f'{y}Y CAGR'] = None

        if not div_hist.empty:
            annual = div_hist.resample('Y').sum()
            last_year = datetime.now().year - 1
            try:
                curr_div = annual[annual.index.year == last_year].iloc[0]
                for y in [3, 5, 10, 15]:
                    target = last_year - y
                    try:
                        past_div = annual[annual.index.year == target].iloc[0]
                        metrics[f'{y}Y Div CAGR'] = get_cagr_div(curr_div, past_div, y)
                    except: metrics[f'{y}Y Div CAGR'] = None
            except: pass
        else:
            for y in [3, 5, 10, 15]: metrics[f'{y}Y Div CAGR'] = None
            
        return metrics

    if st.button("Update Dividends"):
        tickers = st.text_area("Tickers", "SMH, MSFT, VYM, SCHD, VOO, QQQ, JEPQ")
        t_list = [x.strip().upper() for x in tickers.split(',')]
        data = []
        progress = st.progress(0)
        for i, t in enumerate(t_list):
            res = get_div_data(t)
            if res: data.append(res)
            progress.progress((i + 1) / len(t_list))
        
        if data:
            df = pd.DataFrame(data)
            
            # --- AVERAGE ROW ---
            numeric_cols = [
                'Yield (TTM)', 'Yield (Fwd)', 
                '1D', '1W', '1M', 'YTD',
                '1Y Total', '3Y Total', '3Y CAGR', 
                '5Y Total', '5Y CAGR', '10Y Total', '10Y CAGR', 
                '15Y Total', '15Y CAGR',
                '3Y Div CAGR', '5Y Div CAGR', '10Y Div CAGR', '15Y Div CAGR'
            ]
            
            avg_data = {col: df[col].mean() for col in numeric_cols if col in df.columns}
            avg_data['Ticker'] = "AVERAGE"
            avg_data['Inception'] = "-"
            avg_data['Price'] = None 
            
            df_avg = pd.DataFrame([avg_data])
            df_final = pd.concat([df, df_avg], ignore_index=True)
            
            cols = [
                'Ticker', 'Price', 'Inception', 'Streak', 'Freq', 'Yield (TTM)', 'Yield (Fwd)', 'Ex-Div', 'Payout',
                '1D', '1W', '1M', 'YTD',
                '1Y Total', 
                '3Y Total', '3Y CAGR', 
                '5Y Total', '5Y CAGR', 
                '10Y Total', '10Y CAGR', 
                '15Y Total', '15Y CAGR',
                '3Y Div CAGR', '5Y Div CAGR', '10Y Div CAGR', '15Y Div CAGR'
            ]
            final_cols = [c for c in cols if c in df_final.columns]
            df_final = df_final[final_cols]
            
            fmt = {
                'Price':'${:.2f}', 'Yield (TTM)':'{:.2%}', 'Yield (Fwd)':'{:.2%}', 
                '1D':'{:.2%}', '1W':'{:.2%}', '1M':'{:.2%}', 'YTD':'{:.2%}',
                '1Y Total':'{:.2%}', 
                '3Y Total':'{:.2%}', '3Y CAGR':'{:.2%}',
                '5Y Total':'{:.2%}', '5Y CAGR':'{:.2%}',
                '10Y Total':'{:.2%}', '10Y CAGR':'{:.2%}',
                '15Y Total':'{:.2%}', '15Y CAGR':'{:.2%}',
                '3Y Div CAGR':'{:.2%}', '5Y Div CAGR':'{:.2%}', '10Y Div CAGR':'{:.2%}', '15Y Div CAGR':'{:.2%}'
            }
            st.dataframe(df_final.style.format(fmt, na_rep="-"), height=600)
            st.caption("* 'Total' = Total percent return over the period. 'CAGR' = Annualized return per year. *")

# ==========================================
# TAB 3: MULTI-ETF INSPECTION
# ==========================================
with tab3:
    st.header("🔍 Multi-ETF Inspection")
    target_input = st.text_input("Enter ETF Tickers (comma separated):", value="SMH, MGK, VOO")
    
    if st.button("Analyze ETFs"):
        targets = [x.strip().upper() for x in target_input.split(',')]
        
        for target in targets:
            st.markdown(f"---")
            st.markdown(f"### 🔎 Analysis for **{target}**")
            
            inc_date = get_inception_date(target)
            st.write(f"**Inception Date:** {inc_date}")
            
            df, source = get_holdings_robust(target)
            
            if not df.empty:
                df.columns = ['Holding', 'Weight']
                df = merge_google(df, symbol_col='Holding', weight_col='Weight')
                df['Weight'] = (df['Weight'] * 100).map('{:.2f}%'.format)
                
                if source == "Backup Data":
                    st.caption(f"⚠️ Yahoo blocked the connection. Using cached backup data.")
                else:
                    st.caption(f"✅ Live data fetched from Yahoo Finance.")
                    
                st.write(f"**Top Holdings (GOOG merged):**")
                st.table(df)
            else:
                st.error(f"Could not fetch holdings for {target}. (Not in backup list)")
