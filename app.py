import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: X-Ray, Dividends & Holdings")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["🚀 Portfolio X-Ray", "📈 Dividend & Growth Data", "🔍 Multi-ETF Deep Dive"])

# --- SHARED HELPER: GOOG MERGER ---
def merge_google(df, symbol_col='Symbol', weight_col='Weight'):
    """Combines GOOG and GOOGL into a single entry."""
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Replace ticker symbols
    # We use regex=False to exact match
    df[symbol_col] = df[symbol_col].replace({'GOOG': 'GOOG/L', 'GOOGL': 'GOOG/L'})
    
    # Group by the Symbol column and sum the weights
    # We use as_index=False to keep Symbol as a column
    df = df.groupby(symbol_col, as_index=False)[weight_col].sum()
    
    # Re-sort so the new combined GOOG/L goes to its correct rank
    df = df.sort_values(by=weight_col, ascending=False).reset_index(drop=True)
    
    return df

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
        all_holdings = []
        status = st.empty()
        
        for ticker in etf_list:
            status.text(f"Fetching {ticker}...")
            try:
                etf = yf.Ticker(ticker)
                try:
                    raw_holdings = etf.funds_data.top_holdings
                except:
                    raw_holdings = etf.holdings
                
                if not raw_holdings.empty:
                    if isinstance(raw_holdings, pd.Series):
                        df = raw_holdings.to_frame()
                    else:
                        df = raw_holdings.copy()
                    
                    df = df.reset_index()
                    # FORCE 2 COLUMNS
                    df = df.iloc[:, [0, -1]]
                    df.columns = ['Symbol', 'Raw_Weight']
                    
                    # Apply Weight
                    df['Portfolio_Weight'] = df['Raw_Weight'] * weights[ticker]
                    all_holdings.append(df)
            except Exception as e:
                st.error(f"Could not load {ticker}: {e}")

        status.empty()

        if all_holdings:
            full_df = pd.concat(all_holdings)
            
            # --- FIX: MERGE GOOG BEFORE GROUPING ---
            # This ensures GOOG and GOOGL from different ETFs combine correctly
            full_df['Symbol'] = full_df['Symbol'].replace({'GOOG': 'GOOG/L', 'GOOGL': 'GOOG/L'})
            
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
        else:
            st.warning("No holdings data found. Yahoo might be blocking requests temporarily.")

# ==========================================
# TAB 2: DIVIDEND DATA
# ==========================================
with tab2:
    def get_cagr(end, start, years):
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
        
        rate = info.get('dividendRate', 0)
        if rate and rate > 0: 
            yield_fwd = rate / price
        else:
            raw = info.get('dividendYield', 0)
            yield_fwd = raw / 100 if raw > 1 else raw

        ex_div = info.get('exDividendDate', None)
        ex_div = datetime.fromtimestamp(ex_div).strftime('%Y-%m-%d') if ex_div else "-"
        payout = info.get('payoutDate', None)
        payout = datetime.fromtimestamp(payout).strftime('%Y-%m-%d') if payout else "-"
        
        streak, freq = get_streak_and_freq(div_hist)

        metrics = {
            'Ticker': ticker, 'Price': price, 
            'Yield (TTM)': yield_ttm, 'Yield (Fwd)': yield_fwd,
            'Streak': streak, 'Freq': freq,
            'Ex-Div': ex_div, 'Payout': payout
        }
        
        curr_date = hist.index[-1]
        for y in [1, 3, 5, 10, 15]:
            target = curr_date - timedelta(days=y*365)
            idx_loc = hist.index.get_indexer([target], method='nearest')
            if len(idx_loc) > 0:
                idx = idx_loc[0]
                if abs((hist.index[idx] - target).days) < 100:
                    metrics[f'{y}Y Return'] = (price - hist['Close'].iloc[idx]) / hist['Close'].iloc[idx]
                else: metrics[f'{y}Y Return'] = None
            else: metrics[f'{y}Y Return'] = None

        if not div_hist.empty:
            annual = div_hist.resample('Y').sum()
            last_year = datetime.now().year - 1
            try:
                curr_div = annual[annual.index.year == last_year].iloc[0]
                for y in [3, 5, 10, 15]:
                    target = last_year - y
                    try:
                        past_div = annual[annual.index.year == target].iloc[0]
                        metrics[f'{y}Y Div CAGR'] = get_cagr(curr_div, past_div, y)
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
            cols = [
                'Ticker', 'Price', 'Streak', 'Freq', 'Yield (TTM)', 'Yield (Fwd)', 'Ex-Div', 'Payout',
                '1Y Return', '3Y Return', '5Y Return', '10Y Return', '15Y Return',
                '3Y Div CAGR', '5Y Div CAGR', '10Y Div CAGR', '15Y Div CAGR'
            ]
            final_cols = [c for c in cols if c in df.columns]
            df = df[final_cols]
            
            fmt = {
                'Price':'${:.2f}', 'Yield (TTM)':'{:.2%}', 'Yield (Fwd)':'{:.2%}', 
                '1Y Return':'{:.2%}', '3Y Return':'{:.2%}', '5Y Return':'{:.2%}', '10Y Return':'{:.2%}', '15Y Return':'{:.2%}',
                '3Y Div CAGR':'{:.2%}', '5Y Div CAGR':'{:.2%}', '10Y Div CAGR':'{:.2%}', '15Y Div CAGR':'{:.2%}'
            }
            st.dataframe(df.style.format(fmt, na_rep="-"), height=600)

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
            
            stock = yf.Ticker(target)
            
            # 1. HOLDINGS (With Percentage Formatting)
            try:
                try:
                    raw = stock.funds_data.top_holdings
                except:
                    raw = stock.holdings
                    
                if not raw.empty:
                    # Clean Data
                    if isinstance(raw, pd.Series): df = raw.to_frame()
                    else: df = raw.copy()
                    
                    df = df.reset_index()
                    df = df.iloc[:, [0, -1]]
                    df.columns = ['Holding', 'Weight']
                    
                    # --- FIX: MERGE GOOG BEFORE DISPLAY ---
                    # Combine GOOG + GOOGL weights
                    df = merge_google(df, symbol_col='Holding', weight_col='Weight')
                    
                    # FORMAT TO PERCENTAGE
                    df['Weight'] = (df['Weight'] * 100).map('{:.2f}%'.format)
                    
                    st.write(f"**Top Holdings (GOOG merged):**")
                    st.table(df)
                else:
                    st.warning(f"No holdings found for {target}.")
            except Exception as e:
                st.error(f"Error fetching holdings for {target}: {e}")
                
            # 2. SECTORS
            try:
                sect = stock.funds_data.sector_weightings
                if sect:
                    sdf = pd.DataFrame(list(sect.items()), columns=['Sector', 'Weight'])
                    
                    col_chart, col_data = st.columns([1, 1])
                    with col_chart:
                        fig = px.pie(sdf, values='Weight', names='Sector', title=f"{target} Sector Allocation")
                        st.plotly_chart(fig, use_container_width=True)
            except: pass