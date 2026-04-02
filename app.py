import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Master Portfolio", layout="wide")
st.title("📊 Master Portfolio: Light Rios Edition")

DEFAULT_PORTFOLIO = "SCHG, QQQ, VGT, SMH"
DEFAULT_BENCHMARK = "VOO, SCHD"
DEFAULT_WATCHLIST = "SCHD, VYM, VIG, VOO, SCHG, QQQ, VGT, SMH, VIG"

SECTOR_MAP = {
    "NVDA": "Technology", "MSFT": "Technology", "AAPL": "Technology", "AVGO": "Technology", "ORCL": "Technology", "ADBE": "Technology", "CRM": "Technology", "AMD": "Technology", "QCOM": "Technology", "TXN": "Technology", "INTC": "Technology", "AMAT": "Technology", "IBM": "Technology", "MU": "Technology", "LRCX": "Technology", "ADI": "Technology", "KLAC": "Technology", "SNPS": "Technology", "CDNS": "Technology", "PANW": "Technology", "NOW": "Technology", "PLTR": "Technology", "ASML": "Technology", "TSM": "Technology", "NXPI": "Technology", "MPH": "Technology", "ON": "Technology", "MCHP": "Technology", "FTEC": "Technology", "VGT": "Technology", "SMH": "Technology", "XLK": "Technology", "CSCO": "Technology", "ANET": "Technology", "HPQ": "Technology", "DELL": "Technology", "CRWD": "Technology", "FTNT": "Technology", "ZS": "Technology", "NET": "Technology",
    "GOOG": "Communication", "GOOGL": "Communication", "GOOG/L": "Communication", "META": "Communication", "NFLX": "Communication", "DIS": "Communication", "CMCSA": "Communication", "TMUS": "Communication", "VZ": "Communication", "T": "Communication", "CHTR": "Communication", "EA": "Communication", "TTWO": "Communication", "WBD": "Communication", "FOX": "Communication",
    "AMZN": "Cons. Cyclical", "TSLA": "Cons. Cyclical", "HD": "Cons. Cyclical", "MCD": "Cons. Cyclical", "NKE": "Cons. Cyclical", "SBUX": "Cons. Cyclical", "LOW": "Cons. Cyclical", "BKNG": "Cons. Cyclical", "TJX": "Cons. Cyclical", "F": "Cons. Cyclical", "GM": "Cons. Cyclical", "TGT": "Cons. Cyclical", "MAR": "Cons. Cyclical", "HLT": "Cons. Cyclical", "ABNB": "Cons. Cyclical",
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare", "TMO": "Healthcare", "PFE": "Healthcare", "AMGN": "Healthcare", "DHR": "Healthcare", "ISRG": "Healthcare", "ELV": "Healthcare", "BMY": "Healthcare", "CVS": "Healthcare", "CI": "Healthcare", "SYK": "Healthcare", "ZTS": "Healthcare",
    "JPM": "Financial", "V": "Financial", "MA": "Financial", "BAC": "Financial", "WFC": "Financial", "MS": "Financial", "GS": "Financial", "BLK": "Financial", "SPGI": "Financial", "AXP": "Financial", "C": "Financial", "BRK.B": "Financial", "USB": "Financial", "PNC": "Financial", "CB": "Financial", "MMC": "Financial",
    "WMT": "Cons. Defensive", "PG": "Cons. Defensive", "COST": "Cons. Defensive", "KO": "Cons. Defensive", "PEP": "Cons. Defensive", "PM": "Cons. Defensive", "MO": "Cons. Defensive", "CL": "Cons. Defensive", "KMB": "Cons. Defensive", "EL": "Cons. Defensive", "GIS": "Cons. Defensive", "K": "Cons. Defensive",
    "CAT": "Industrials", "UNP": "Industrials", "HON": "Industrials", "GE": "Industrials", "UPS": "Industrials", "BA": "Industrials", "LMT": "Industrials", "RTX": "Industrials", "DE": "Industrials", "MMM": "Industrials", "ADP": "Industrials", "GD": "Industrials", "NOC": "Industrials", "LHX": "Industrials", "WM": "Industrials", "ETN": "Industrials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate", "O": "Real Estate", "SPG": "Real Estate", "VICI": "Real Estate", "DLR": "Real Estate", "EQIX": "Real Estate",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities", "D": "Utilities", "EXC": "Utilities", "SRE": "Utilities"
}

INDUSTRY_MAP = {
    "NVDA": "Semi - GPU/AI Logic", "AMD": "Semi - CPU/GPU Logic", "INTC": "Semi - Manufacturing (IDM)", "TSM": "Semi - Foundry (Mfg)", "AVGO": "Semi - Networking/Radio", "QCOM": "Semi - Mobile/Communications", "MU": "Semi - Memory (DRAM/NAND)", "TXN": "Semi - Analog/Embedded", "ADI": "Semi - Analog", "NXPI": "Semi - Automotive", "ON": "Semi - Power/Auto", "STM": "Semi - Microcontrollers", "MCHP": "Semi - Microcontrollers", "MPWR": "Semi - Power Management",
    "ASML": "Semi Equip - Lithography", "AMAT": "Semi Equip - Materials Eng", "LRCX": "Semi Equip - Etch/Deposition", "KLAC": "Semi Equip - Process Control", "TER": "Semi Equip - Testing", "ENTG": "Semi Equip - Materials",
    "MSFT": "Cloud & OS Infrastructure", "ORCL": "Cloud & Database", "ADBE": "Creative Content Software", "CRM": "Enterprise CRM", "NOW": "Enterprise Workflow", "PLTR": "Data Analytics/Defense", "SNPS": "Chip Design Software (EDA)", "CDNS": "Chip Design Software (EDA)", "PANW": "Cybersecurity", "CRWD": "Cybersecurity", "FTNT": "Cybersecurity", "ZS": "Cybersecurity - Cloud", "NET": "Cloud Networking/Security", "DDOG": "Cloud Monitoring", "MDB": "Database Software",
    "AAPL": "Consumer Electronics", "CSCO": "Networking Hardware", "ANET": "Cloud Networking", "HPQ": "PC/Printers", "DELL": "Server/Storage", "IBM": "IT Services & Consulting", "STX": "Data Storage", "WDC": "Data Storage",
    "GOOG": "Search & Digital Ads", "GOOGL": "Search & Digital Ads", "GOOG/L": "Search & Digital Ads", "META": "Social Media & Ads", "NFLX": "Streaming Entertainment", "DIS": "Media Conglomerate", "WBD": "Media & Streaming", "CMCSA": "Cable & Broadband", "TMUS": "Wireless Telecom", "VZ": "Telecom Services", "T": "Telecom Services",
    "AMZN": "E-Commerce & Cloud (AWS)", "BKNG": "Online Travel", "ABNB": "Lodging Platform", "UBER": "Rideshare & Logistics", "DASH": "Food Delivery", "EBAY": "Online Marketplace",
    "TSLA": "EV Manufacturer", "F": "Legacy Auto Manufacturer", "GM": "Legacy Auto Manufacturer", "RIVN": "EV Manufacturer",
    "HD": "Home Improvement Retail", "LOW": "Home Improvement Retail", "WMT": "Big Box Retail", "TGT": "Big Box Retail", "COST": "Membership Warehouses", "MCD": "Fast Food Restaurants", "SBUX": "Coffee Chain", "NKE": "Athletic Footwear", "LULU": "Apparel", "TJX": "Off-Price Retail", "PG": "Household Basics", "KO": "Non-Alcoholic Bev", "PEP": "Bev & Snacks", "PM": "Tobacco", "MO": "Tobacco",
    "LLY": "Pharma (Diabetes/Weight)", "NVO": "Pharma (Diabetes/Weight)", "JNJ": "Pharma & MedTech", "MRK": "Pharma (Oncology)", "PFE": "Pharma (Vaccines)", "ABBV": "Biopharma (Immunology)", "AMGN": "Biotech", "VRTX": "Biotech", "GILD": "Biotech", "UNH": "Health Insurance", "ELV": "Health Insurance", "CVS": "Pharmacy & Insurance", "ISRG": "Robotic Surgery", "SYK": "Medical Devices", "MDT": "Medical Devices", "TMO": "Lab Instruments", "DHR": "Life Sciences",
    "JPM": "Global Universal Bank", "BAC": "Consumer Banking", "WFC": "Consumer Banking", "C": "Global Banking", "MS": "Investment Banking", "GS": "Investment Banking", "BLK": "Asset Management (ETFs)", "V": "Payment Network", "MA": "Payment Network", "AXP": "Credit Card Issuer", "PYPL": "Digital Payments", "SQ": "Fintech / Payments", "SPGI": "Financial Data/Ratings", "MCO": "Financial Data/Ratings", "BRK.B": "Conglomerate / Insurance",
    "LMT": "Aerospace & Defense", "RTX": "Aerospace & Defense", "NOC": "Aerospace & Defense", "GD": "Aerospace & Defense", "BA": "Aerospace (Commercial)", "GE": "Jet Engines / Power", "CAT": "Construction Machinery", "DE": "Agricultural Machinery", "UNP": "Railroad", "CSX": "Railroad", "UPS": "Package Delivery", "FDX": "Package Delivery", "HON": "Industrial Conglomerate", "WM": "Waste Management",
    "XOM": "Integrated Oil & Gas", "CVX": "Integrated Oil & Gas", "COP": "Oil Exploration", "EOG": "Oil Exploration", "SLB": "Oilfield Services", "HAL": "Oilfield Services", "MPC": "Oil Refining", "PSX": "Oil Refining",
    "PLD": "REIT - Industrial", "AMT": "Cell Tower REIT", "CCI": "Cell Tower REIT", "EQIX": "Data Center REIT", "DLR": "Data Center REIT", "O": "Retail REIT", "VICI": "Casino/Gaming REIT", "SPG": "Mall REIT"
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

BACKUP_CAGR_10Y = {"SMH": 0.285, "QQQ": 0.182, "MGK": 0.195, "SCHG": 0.188, "FTEC": 0.205, "VOO": 0.128, "SPY": 0.128, "VGT": 0.192, "VYM": 0.095, "SCHD": 0.115, "JEPQ": 0.105}
BACKUP_INCEPTION = {"SMH": "2011-12-20", "QQQ": "1999-03-10", "MGK": "2007-12-17", "SCHG": "2009-12-11", "FTEC": "2013-10-21", "VOO": "2010-09-07", "SPY": "1993-01-22", "VGT": "2004-01-26", "VYM": "2006-11-10", "SCHD": "2011-10-20", "JEPQ": "2022-05-03"}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🚀 Portfolio X-Ray", "🆚 Portfolio vs. Benchmark", "📈 Dividend & Growth", "🔍 Multi-ETF Deep Dive", "👀 Watchlist", "📰 AI News & Insights"])

def merge_google(df):
    df = df.copy()
    df['Symbol'] = df['Symbol'].replace({'GOOG': 'GOOG/L', 'GOOGL': 'GOOG/L'})
    return df.groupby('Symbol', as_index=False)['Weight'].sum().sort_values(by='Weight', ascending=False).reset_index(drop=True)

def get_sector(ticker): return SECTOR_MAP.get(ticker, "Other / Diversified")
def get_industry(ticker): return INDUSTRY_MAP.get(ticker, SECTOR_MAP.get(ticker, "ETF / Fund"))

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

def get_full_stats(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="max", auto_adjust=True)
        div_hist = stock.dividends
        info = stock.info
    except: return None
    if hist.empty: return None
    price = hist['Close'].iloc[-1]
    
    yield_ttm = div_hist[div_hist.index >= (pd.Timestamp.now().tz_localize(div_hist.index.dtype.tz) - pd.Timedelta(days=365))].sum() / price if not div_hist.empty else 0
    yield_fwd = yield_ttm if 'ETF' in info.get('quoteType', '').upper() else (info.get('dividendRate', 0) / price if info.get('dividendRate', 0) > 0 else (info.get('dividendYield', 0)/100 if info.get('dividendYield', 0) > 0.25 else info.get('dividendYield', 0)))
    
    annual_divs = div_hist.groupby(div_hist.index.year).sum() if not div_hist.empty else pd.Series()
    streak = 0
    if not annual_divs.empty:
        completed = annual_divs[annual_divs.index < datetime.now().year].sort_index(ascending=False)
        for i in range(len(completed) - 1):
            if completed.iloc[i] > completed.iloc[i+1]: streak += 1
            else: break
    
    freq = "Mo" if (div_hist[div_hist.index.year == (datetime.now().year - 1)].count() >= 11) else ("Qr" if (div_hist[div_hist.index.year == (datetime.now().year - 1)].count() >= 3) else ("Yr" if (div_hist[div_hist.index.year == (datetime.now().year - 1)].count() >= 1) else "-"))

    metrics = {'Ticker': ticker, 'Price': price, 'Industry': get_industry(ticker), 'Inception': get_inception_date(ticker, info), 'Yield (TTM)': yield_ttm, 'Yield (Fwd)': yield_fwd, 'Streak': streak, 'Freq': freq}
    
    metrics['1D'] = (price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] if len(hist) > 1 else None
    metrics['1W'] = (price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6] if len(hist) > 5 else None
    metrics['1M'] = (price - hist['Close'].iloc[-22]) / hist['Close'].iloc[-22] if len(hist) > 21 else None
    
    hist_ytd = hist[hist.index >= pd.Timestamp(f"{datetime.now().year}-01-01").tz_localize(hist.index.dtype.tz)]
    metrics['YTD'] = (price - hist_ytd['Open'].iloc[0]) / hist_ytd['Open'].iloc[0] if not hist_ytd.empty else None

    for y in [1, 3, 5, 10, 15]:
        target = hist.index[-1] - timedelta(days=y*365)
        idx_loc = hist.index.get_indexer([target], method='nearest')
        if len(idx_loc) > 0 and abs((hist.index[idx_loc[0]] - target).days) < 100:
            start_price = hist['Close'].iloc[idx_loc[0]]
            metrics[f'{y}Y Total'] = (price - start_price) / start_price
            metrics[f'{y}Y CAGR'] = (price / start_price) ** (1 / y) - 1 if y > 1 else None
        else:
            metrics[f'{y}Y Total'], metrics[f'{y}Y CAGR'] = None, None

    last_year = datetime.now().year - 1
    for y in [3, 5, 10, 15]:
        if not annual_divs.empty and last_year in annual_divs.index and (last_year - y) in annual_divs.index:
            metrics[f'{y}Y Div CAGR'] = get_cagr_div(annual_divs.loc[last_year], annual_divs.loc[last_year - y], y)
        else: metrics[f'{y}Y Div CAGR'] = None
        
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

def get_cagr_for_year(ticker, years):
    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if not hist.empty:
            target = hist.index[-1] - timedelta(days=years*365)
            if hist.index[0] <= target + timedelta(days=60):
                idx = hist.index.get_indexer([target], method='nearest')[0]
                if idx >= 0: return (hist['Close'].iloc[-1] / hist['Close'].iloc[idx]) ** (1 / years) - 1
    except: pass
    return BACKUP_CAGR_10Y.get(ticker, None) if years == 10 else None

# === TAB 1: X-RAY ===
with tab1:
    st.header("See what you actually own")
    etf_list = [x.strip().upper() for x in st.text_input("Enter ETFs to Blend:", value=DEFAULT_PORTFOLIO).split(',')]
    cols, weights, total_weight = st.columns(3), {}, 0
    for i, ticker in enumerate(etf_list):
        with cols[i % 3]:
            w = st.slider(f"{ticker} %", 0, 100, 100 // len(etf_list), key=f"s_{ticker}")
            weights[ticker] = w / 100.0
            total_weight += w

    if st.button("Analyze Blended Holdings"):
        st.markdown("### 📈 Blended Performance (Annualized)")
        timeframes = [1, 3, 5, 10, 15]
        blended_stats, valid_weights = {y: 0.0 for y in timeframes}, {y: 0.0 for y in timeframes}
        
        for t in etf_list:
            if weights[t] > 0:
                for y in timeframes:
                    cagr = get_cagr_for_year(t, y)
                    if cagr is not None:
                        blended_stats[y] += cagr * weights[t]
                        valid_weights[y] += weights[t]
        
        m_cols = st.columns(5)
        for i, y in enumerate(timeframes):
            m_cols[i].metric(f"{y}-Year", f"{(blended_stats[y] / valid_weights[y]):.2%}" if valid_weights[y] > 0.1 else "N/A")
        st.markdown("---")

        all_holdings = [get_holdings_robust(t)[0].assign(Portfolio_Weight=lambda x: x['Raw_Weight'] * weights[t]) for t in etf_list if weights[t] > 0 and not get_holdings_robust(t)[0].empty]

        if all_holdings:
            full_df = pd.concat(all_holdings).rename(columns={'Symbol': 'Symbol', 'Portfolio_Weight': 'Weight'})
            grouped = merge_google(full_df)
            grouped['Sector'] = grouped['Symbol'].apply(get_sector)
            grouped['Industry'] = grouped['Symbol'].apply(get_industry)
            grouped['Weight %'] = (grouped['Weight'] * 100).round(2)
            
            c1, c2 = st.columns(
