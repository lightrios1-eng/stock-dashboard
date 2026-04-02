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

# --- DATA BANKS ---
SECTOR_MAP = {
    "NVDA": "Technology", "MSFT": "Technology", "AAPL": "Technology", "AVGO": "Technology",
    "ORCL": "Technology", "ADBE": "Technology", "CRM": "Technology", "AMD": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "INTC": "Technology", "AMAT": "Technology",
    "IBM": "Technology", "MU": "Technology", "LRCX": "Technology", "ADI": "Technology",
    "KLAC": "Technology", "SNPS": "Technology", "CDNS": "Technology", "PANW": "Technology",
    "NOW": "Technology", "PLTR": "Technology", "ASML": "Technology", "TSM": "Technology",
    "NXPI": "Technology", "MPH": "Technology", "ON": "Technology", "MCHP": "Technology",
    "FTEC": "Technology", "VGT": "Technology", "SMH": "Technology", "XLK": "Technology",
    "CSCO": "Technology", "ANET": "Technology", "HPQ": "Technology", "DELL": "Technology",
    "CRWD": "Technology", "FTNT": "Technology", "ZS": "Technology", "NET": "Technology",
    "GOOG": "Communication", "GOOGL": "Communication", "GOOG/L": "Communication",
    "META": "Communication", "NFLX": "Communication", "DIS": "Communication",
    "CMCSA": "Communication", "TMUS": "Communication", "VZ": "Communication",
    "T": "Communication", "CHTR": "Communication", "EA": "Communication",
    "TTWO": "Communication", "WBD": "Communication", "FOX": "Communication",
    "AMZN": "Cons. Cyclical", "TSLA": "Cons. Cyclical", "HD": "Cons. Cyclical",
    "MCD": "Cons. Cyclical", "NKE": "Cons. Cyclical", "SBUX": "Cons. Cyclical",
    "LOW": "Cons. Cyclical", "BKNG": "Cons. Cyclical", "TJX": "Cons. Cyclical",
    "F": "Cons. Cyclical", "GM": "Cons. Cyclical", "TGT": "Cons. Cyclical",
    "MAR": "Cons. Cyclical", "HLT": "Cons. Cyclical", "ABNB": "Cons. Cyclical",
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "TMO": "Healthcare", "PFE": "Healthcare", "AMGN": "Healthcare",
    "DHR": "Healthcare", "ISRG": "Healthcare", "ELV": "Healthcare", "BMY": "Healthcare",
    "CVS": "Healthcare", "CI": "Healthcare", "SYK": "Healthcare", "ZTS": "Healthcare",
    "JPM": "Financial", "V": "Financial", "MA": "Financial", "BAC": "Financial",
    "WFC": "Financial", "MS": "Financial", "GS": "Financial", "BLK": "Financial",
    "SPGI": "Financial", "AXP": "Financial", "C": "Financial", "BRK.B": "Financial",
    "USB": "Financial", "PNC": "Financial", "CB": "Financial", "MMC": "Financial",
    "WMT": "Cons. Defensive", "PG": "Cons. Defensive", "COST": "Cons. Defensive",
    "KO": "Cons. Defensive", "PEP": "Cons. Defensive", "PM": "Cons. Defensive",
    "MO": "Cons. Defensive", "CL": "Cons. Defensive", "KMB": "Cons. Defensive",
    "EL": "Cons. Defensive", "GIS": "Cons. Defensive", "K": "Cons. Defensive",
    "CAT": "Industrials", "UNP": "Industrials", "HON": "Industrials", "GE": "Industrials",
    "UPS": "Industrials", "BA": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "DE": "Industrials", "MMM": "Industrials", "ADP": "Industrials", "GD": "Industrials",
    "NOC": "Industrials", "LHX": "Industrials", "WM": "Industrials", "ETN": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "MPC": "Energy", "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate", "O": "Real Estate",
    "SPG": "Real Estate", "VICI": "Real Estate", "DLR": "Real Estate", "EQIX": "Real Estate",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities",
    "D": "Utilities", "EXC": "Utilities", "SRE": "Utilities"
}

INDUSTRY_MAP = {
    "NVDA": "Semi - GPU/AI Logic", "AMD": "Semi - CPU/GPU Logic",
    "INTC": "Semi - Manufacturing (IDM)", "TSM": "Semi - Foundry (Mfg)",
    "AVGO": "Semi - Networking/Radio", "QCOM": "Semi - Mobile/Communications",
    "MU": "Semi - Memory (DRAM/NAND)", "TXN": "Semi - Analog/Embedded",
    "ADI": "Semi - Analog", "NXPI": "Semi - Automotive",
    "ON": "Semi - Power/Auto", "STM": "Semi - Microcontrollers",
    "MCHP": "Semi - Microcontrollers", "MPWR": "Semi - Power Management",
    "ASML": "Semi Equip - Lithography", "AMAT": "Semi Equip - Materials Eng",
    "LRCX": "Semi Equip - Etch/Deposition", "KLAC": "Semi Equip - Process Control",
    "TER": "Semi Equip - Testing", "ENTG": "Semi Equip - Materials",
    "MSFT": "Cloud & OS Infrastructure", "ORCL": "Cloud & Database",
    "ADBE": "Creative Content Software", "CRM": "Enterprise CRM",
    "NOW": "Enterprise Workflow", "PLTR": "Data Analytics/Defense",
    "SNPS": "Chip Design Software (EDA)", "CDNS": "Chip Design Software (EDA)",
    "PANW": "Cybersecurity", "CRWD": "Cybersecurity", "FTNT": "Cybersecurity",
    "ZS": "Cybersecurity - Cloud", "NET": "Cloud Networking/Security",
    "DDOG": "Cloud Monitoring", "MDB": "Database Software",
    "AAPL": "Consumer Electronics", "CSCO": "Networking Hardware",
    "ANET": "Cloud Networking", "HPQ": "PC/Printers", "DELL": "Server/Storage",
    "IBM": "IT Services & Consulting", "STX": "Data Storage", "WDC": "Data Storage",
    "GOOG": "Search & Digital Ads", "GOOGL": "Search & Digital Ads",
    "GOOG/L": "Search & Digital Ads", "META": "Social Media & Ads",
    "NFLX": "Streaming Entertainment", "DIS": "Media Conglomerate",
    "WBD": "Media & Streaming", "CMCSA": "Cable & Broadband",
    "TMUS": "Wireless Telecom", "VZ": "Telecom Services", "T": "Telecom Services",
    "AMZN": "E-Commerce & Cloud (AWS)", "BKNG": "Online Travel",
    "ABNB": "Lodging Platform", "UBER": "Rideshare & Logistics",
    "DASH": "Food Delivery", "EBAY": "Online Marketplace",
    "TSLA": "EV Manufacturer", "F": "Legacy Auto Manufacturer",
    "GM": "Legacy Auto Manufacturer", "RIVN": "EV Manufacturer",
    "HD": "Home Improvement Retail", "LOW": "Home Improvement Retail",
    "WMT": "Big Box Retail", "TGT": "Big Box Retail", "COST": "Membership Warehouses",
    "MCD": "Fast Food Restaurants", "SBUX": "Coffee Chain",
    "NKE": "Athletic Footwear", "LULU": "Apparel", "TJX": "Off-Price Retail",
    "PG": "Household Basics", "KO": "Non-Alcoholic Bev", "PEP": "Bev & Snacks",
    "PM": "Tobacco", "MO": "Tobacco",
    "LLY": "Pharma (Diabetes/Weight)", "NVO": "Pharma (Diabetes/Weight)",
    "JNJ": "Pharma & MedTech", "MRK": "Pharma (Oncology)",
    "PFE": "Pharma (Vaccines)", "ABBV": "Biopharma (Immunology)",
    "AMGN": "Biotech", "VRTX": "Biotech", "GILD": "Biotech",
    "UNH": "Health Insurance", "ELV": "Health Insurance", "CVS": "Pharmacy & Insurance",
    "ISRG": "Robotic Surgery", "SYK": "Medical Devices", "MDT": "Medical Devices",
    "TMO": "Lab Instruments", "DHR": "Life Sciences",
    "JPM": "Global Universal Bank", "BAC": "Consumer Banking",
    "WFC": "Consumer Banking", "C": "Global Banking",
    "MS": "Investment Banking", "GS": "Investment Banking",
    "BLK": "Asset Management (ETFs)", "V": "Payment Network",
    "MA": "Payment Network", "AXP": "Credit Card Issuer",
    "PYPL": "Digital Payments", "SQ": "Fintech / Payments",
    "SPGI": "Financial Data/Ratings", "MCO": "Financial Data/Ratings",
    "BRK.B": "Conglomerate / Insurance",
    "LMT": "Aerospace & Defense", "RTX": "Aerospace & Defense",
    "NOC": "Aerospace & Defense", "GD": "Aerospace & Defense",
    "BA": "Aerospace (Commercial)", "GE": "Jet Engines / Power",
    "CAT": "Construction Machinery", "DE": "Agricultural Machinery",
    "UNP": "Railroad", "CSX": "Railroad", "UPS": "Package Delivery",
    "FDX": "Package Delivery", "HON": "Industrial Conglomerate",
    "WM": "Waste Management",
    "XOM": "Integrated Oil & Gas", "CVX": "Integrated Oil & Gas",
    "COP": "Oil Exploration", "EOG": "Oil Exploration",
    "SLB": "Oilfield Services", "HAL": "Oilfield Services",
    "MPC": "Oil Refining", "PSX": "Oil Refining",
    "PLD": "REIT - Industrial", "AMT": "Cell Tower REIT", "CCI": "Cell Tower REIT",
    "EQIX": "Data Center REIT", "DLR": "Data Center REIT",
    "O": "Retail REIT", "VICI": "Casino/Gaming REIT", "SPG": "Mall REIT"
}

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

BACKUP_CAGR_10Y = {
    "SMH": 0.285, "QQQ": 0.182, "MGK": 0.195, "SCHG": 0.188, 
    "FTEC": 0.205, "VOO": 0.128, "SPY": 0.128, "VGT": 0.192, 
    "VYM": 0.095, "SCHD": 0.115, "JEPQ": 0.105
}

BACKUP_INCEPTION = {
    "SMH": "2011-12-20", "QQQ": "1999-03-10", "MGK": "2007-12-17", 
    "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07", 
    "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10", 
    "SCHD": "2011-10-20", "JEPQ": "2022-05-03"
}

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Portfolio X-Ray", 
    "🆚 Portfolio vs. Benchmark", 
    "📈 Dividend & Growth", 
    "🔍 Multi-ETF Deep Dive", 
    "👀 Watchlist",
    "📰 AI News & Insights"
])

# --- SHARED HELPERS (THE CORE ENGINE) ---
def merge_google(df, symbol_col='Symbol', weight_col='Weight'):
    df = df.copy()
    df[symbol_col] = df[symbol_col].replace({'GOOG': 'GOOG/L', 'GOOGL': 'GOOG/L'})
    df = df.groupby(symbol_col, as_index=False)[weight_col].sum()
    df = df.sort_values(by=weight_col, ascending=False).reset_index(drop=True)
    return df

def get_sector(ticker):
    return SECTOR_MAP.get(ticker, "Other / Diversified")

def get_industry(ticker):
    if ticker in INDUSTRY_MAP: return INDUSTRY_MAP[ticker]
    return SECTOR_MAP.get(ticker, "ETF / Fund")

def get_inception_date(ticker, info=None):
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

def get_cagr_div(end, start, years):
    if start == 0 or pd.isna(start) or pd.isna(end): return None
    return (end / start) ** (1 / years) - 1

def get_streak_and_freq(div_hist):
    if div_hist.empty: return 0, "-"
    # FIXED: Replaced resample() with groupby() to avoid version conflicts
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
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="max", auto_adjust=True)
        div_hist = stock.dividends
        info = stock.info
    except: return None
    
    if hist.empty: return None
    price = hist['Close'].iloc[-1]
    
    # --- DIVIDENDS ---
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
    industry = get_industry(ticker)

    metrics = {
        'Ticker': ticker, 'Price': price, 
        'Industry': industry,
        'Inception': inc_date,
        'Yield (TTM)': yield_ttm, 'Yield (Fwd)': yield_fwd,
        'Streak': streak, 'Freq': freq,
        'Ex-Div': ex_div, 'Payout': payout
    }
    
    # --- SHORT TERM RETURNS ---
    if len(hist) > 1: metrics['1D'] = (price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
    else: metrics['1D'] = None
    if len(hist) > 5: metrics['1W'] = (price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]
    else: metrics['1W'] = None
    if len(hist) > 21: metrics['1M'] = (price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22]
    else: metrics['1M'] = None

    current_year = datetime.now().year
    ytd_start = f"{current_year}-01-01"
    hist_ytd = hist[hist.index >= pd.Timestamp(ytd_start).tz_localize(hist.index.dtype.tz)]
    if not hist_ytd.empty:
        metrics['YTD'] = (price - hist_ytd['Open'].iloc[0]) / hist_ytd['Open'].iloc[0]
    else: metrics['YTD'] = None

    # --- LONG TERM TOTAL RETURN & CAGR ---
    curr_date = hist.index[-1]
    for y in [1, 3, 5, 10, 15]:
        target = curr_date - timedelta(days=y*365)
        idx_loc = hist.index.get_indexer([target], method='nearest')
        if len(idx_loc) > 0:
            idx = idx_loc[0]
            if abs((hist.index[idx] - target).days) < 100:
                start_price = hist['Close'].iloc[idx]
                end_price = price
                metrics[f'{y}Y Total'] = (end_price - start_price) / start_price
                if y > 1:
                    metrics[f'{y}Y CAGR'] = (end_price / start_price) ** (1 / y) - 1
            else: 
                metrics[f'{y}Y Total'] = None
                if y > 1: metrics[f'{y}Y CAGR'] = None
        else: 
            metrics[f'{y}Y Total'] = None
            if y > 1: metrics[f'{y}Y CAGR'] = None

    # --- DIVIDEND GROWTH CAGR (FIXED FOR VERSION COMPATIBILITY) ---
    if not div_hist.empty:
        annual = div_hist.groupby(div_hist.index.year).sum()
        last_year = datetime.now().year - 1
        try:
            if last_year in annual.index:
                curr_div = annual.loc[last_year]
                for y in [3, 5, 10, 15]:
                    target = last_year - y
                    try:
                        if target in annual.index:
                            past_div = annual.loc[target]
                            metrics[f'{y}Y Div CAGR'] = get_cagr_div(curr_div, past_div, y)
                        else:
                            metrics[f'{y}Y Div CAGR'] = None
                    except: metrics[f'{y}Y Div CAGR'] = None
            else:
                for y in [3, 5, 10, 15]: metrics[f'{y}Y Div CAGR'] = None
        except: pass
    else:
        for y in [3, 5, 10, 15]: metrics[f'{y}Y Div CAGR'] = None
        
    return metrics

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

def get_cagr_for_year(ticker, years):
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

# ==========================================
# TAB 1: PORTFOLIO X-RAY
# ==========================================
with tab1:
    st.header("See what you actually own")
    etf_input = st.text_input("Enter ETFs to Blend (comma separated):", value=DEFAULT_PORTFOLIO)
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
        
        for ticker in etf_list:
            weight = weights[ticker]
            if weight > 0:
                for year in timeframes:
                    cagr = get_cagr_for_year(ticker, year)
                    if cagr is not None:
                        blended_stats[year] += cagr * weight
                        valid_weights[year] += weight
        
        cols = st.columns(5)
        labels = ["1-Year", "3-Year", "5-Year", "10-Year", "15-Year"]
        for i, year in enumerate(timeframes):
            vw = valid_weights[year]
            if vw > 0.10: 
                val = (blended_stats[year] / vw)
                cols[i].metric(labels[i], f"{val:.2%}")
            else:
                cols[i].metric(labels[i], "N/A")
        
        st.markdown("---")

        all_holdings = []
        for ticker in etf_list:
            if weights[ticker] > 0:
                df, source = get_holdings_robust(ticker)
                if not df.empty:
                    df['Portfolio_Weight'] = df['Raw_Weight'] * weights[ticker]
                    all_holdings.append(df)

        if all_holdings:
            full_df = pd.
