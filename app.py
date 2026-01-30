import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import datetime as dt
from io import BytesIO 
# NOTE: reportlab is imported at the bottom, but the modules must be installed
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
 
# ------------- PAGE CONFIG & INITIAL SETUP ------------- #
 
st.set_page_config(
    page_title="Macro-Micro Hedge Fund Dashboard",
    layout="wide",
)
 
st.title("📊 Macro-Micro Investment Dashboard (US Focus)")
st.caption("Layer 1: Sector Comparison | Layer 2: High-Convicton Subsector Analysis.")
 
# --- DATE SETUP ---
END_DATE = dt.date.today()
END_DATETIME = dt.datetime.combine(END_DATE, dt.time.min)
 
START_DATE_6Y_DATE = END_DATE - dt.timedelta(days=365 * 6)
START_DATE_3Y = dt.datetime.combine(START_DATE_6Y_DATE, dt.time.min) # 6 Years
 
START_DATE_2Y_DATE = END_DATE - dt.timedelta(days=365 * 2)
START_DATE_2Y = dt.datetime.combine(START_DATE_2Y_DATE, dt.time.min) 
 
# =======================================================
# US CONFIG: MAIN SECTORS + SUBSECTORS (FIXED: Added Communication & Staples)
# =======================================================
 
MAIN_SECTOR_ETFS = {
    "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI", 
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", 
    "Healthcare": "XLV", "Financials": "XLF", "Technology": "XLK", 
    "Communication": "XLC", "Utilities": "XLU", "Real Estate": "XLRE",
}
 
SUBSECTOR_MAP = {
    "Technology": {
        "Semiconductors & Equip.": "SMH", "Cybersecurity": "CIBR", 
        "Application Software": "IGV", "Cloud Services": "SKYY",
    },
    # --- NEW: Added Communication Services Subsectors ---
    "Communication Services": {
        "Interactive Media (Ads)": "VOX",       # Vanguard Communication Services ETF
        "Media & Streaming": "PBS",             # Public Broadcasting ETF (closest liquid proxy)
        "Gaming & Esports": "HERO",             # Gaming/Esports ETF (high growth theme)
        "Telecom & Infrastructure": "IYZ",      # iShares U.S. Telecommunications ETF
    },
    # --- FIX: Added Consumer Staples Subsectors ---
    "Consumer Staples": {
        "Packaged Food & Beverages": "PSL",     # PowerShares Food & Beverage 
        "Household Products": "KXI",            # iShares Global Consumer Staples
        "Beverages (Global)": "FXG",            # First Trust Consumer Staples AlphaDEX Fund
        "Staples (Broad)": "XLP",               # Broad XLP included for comparison
    },
    "Healthcare": {
        "Biotechnology": "XBI", "Medical Equipment": "IHI", 
        "Managed Care / Providers": "IHF", "Pharmaceuticals": "IHE",
    },
    "Financials": {
        "Regional Banks": "KRE", "Capital Markets / IB": "IAI", "Insurance": "KIE",
    },
    "Energy": {
        "Integrated Oil & Gas": "XLE", "Drilling & Services": "OIH", 
        "Pipelines (Midstream)": "AMLP", "Exploration & Production": "XOP",
    },
    "Materials": {
        "Lithium & Battery Metals": "LIT", "Copper Miners": "COPX", 
        "Gold Miners": "GDX", "Timber / Forestry": "WOOD",
    },
    "Industrials": {
        "Aerospace & Defense": "ITA", "Air Freight & Logistics": "IYT", 
        "Construction & Engineering": "PAVE",
    },
    "Consumer Discretionary": {
        "Automobiles": "CARZ", "Homebuilding": "XHB", "Internet Retail": "ONLN",
    },
    "Utilities": {
        "Renewable Energy": "ICLN", "Electric / Gas Utilities": "XLU",
    },
    "Real Estate": {
        "Data Center REITs": "SRVR", "Industrial REITs": "INDS",
    }
}
 
# Global Ticker Lists
main_etf_list = list(MAIN_SECTOR_ETFS.values())
ticker_to_sector_name = {v: k for k, v in MAIN_SECTOR_ETFS.items()}
all_sub_etfs = [etf for sectors in SUBSECTOR_MAP.values() for etf in sectors.values()]
all_tickers_to_fetch = sorted(set(main_etf_list + all_sub_etfs))
 
# =======================================================
# DATA FETCH HELPERS
# =======================================================
 
@st.cache_data(show_spinner="Fetching Data from Yahoo Finance...", ttl=dt.timedelta(days=7))
def get_price_history(etfs: list[str]) -> pd.DataFrame:
    """Download adjusted prices for ALL ETFs over the given period."""
    try:
        data = yf.download(etfs, start=START_DATE_3Y, end=END_DATETIME, interval="1d", auto_adjust=True, progress=False)
        if data.empty:
            return pd.DataFrame()
        
        # Handle single vs. multi-ticker download structure
        if isinstance(data.columns, pd.MultiIndex):
            field = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
            prices = data[field]
        else:
            field = "Adj Close" if "Adj Close" in data.columns else "Close"
            prices = data[[field]].copy()
            prices.columns = etfs[:1]
            
        prices = prices[[c for c in etfs if c in prices.columns]]
        st.session_state['last_update'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return prices
        
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()
 
 
# =======================================================
# PERIOD AGGREGATION & RETURN CALCULATIONS
# =======================================================
 
def build_period_data(prices: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate daily prices to period-end (D, W, M, Q, Y), compute returns & rolling averages.
    """
    if prices.empty: return pd.DataFrame()
    
    # --- Frequency Mapping ---
    if freq == "D": 
        period_prices = prices.copy()
        period_returns = period_prices.pct_change()
        avg_returns = period_returns.rolling(20).mean() # 20 periods for daily
    elif freq == "W":
        period_prices = prices.resample('W').last().dropna(how="all")
        period_returns = period_prices.pct_change()
        avg_returns = period_returns.rolling(4).mean() # 4 periods
    else: # M, Q, Y
        period_prices = prices.resample(freq).last().dropna(how="all")
        period_returns = period_prices.pct_change()
        avg_returns = period_returns.rolling(4).mean() # 4 periods
    
    
    # --- Long Form Merge ---
    df = period_prices.reset_index().melt(id_vars="Date", var_name="ETF", value_name="Price")
    
    returns_long = period_returns.reset_index().melt(id_vars="Date", var_name="ETF", value_name="PeriodReturn")
    avg_long = avg_returns.reset_index().melt(id_vars="Date", var_name="ETF", value_name="AvgLast4")
 
    df = df.merge(returns_long, on=["Date", "ETF"]).merge(avg_long, on=["Date", "ETF"])
 
    # Map Ticker to Sector/Subsector Name
    def map_to_industry(ticker):
        if ticker in ticker_to_sector_name: return ticker_to_sector_name[ticker]
        for sector, subsectors in SUBSECTOR_MAP.items():
            if ticker in subsectors.values():
                return next((name for name, t in subsectors.items() if t == ticker), "N/A")
        return "N/A"
    
    df["Industry"] = df["ETF"].map(map_to_industry)
 
    # Period label for display
    if freq == "D": df["PeriodLabel"] = df["Date"].dt.strftime("%Y-%m-%d")
    elif freq == "W": df["PeriodLabel"] = df["Date"].dt.to_period("W").astype(str)
    elif freq == "M": df["PeriodLabel"] = df["Date"].dt.to_period("M").astype(str)
    elif freq == "Q": df["PeriodLabel"] = df["Date"].dt.to_period("Q").astype(str)
    else: df["PeriodLabel"] = df["Date"].dt.year.astype(str)
 
    df["PeriodReturn"] = df["PeriodReturn"].astype(float)
    df["AvgLast4"] = df["AvgLast4"].astype(float)
    return df.dropna(subset=["Price"])
 
 
# =======================================================
# GROWTH SUMMARY HELPERS (With Momentum)
# =======================================================
 
def get_quarterly_summary_metrics(prices: pd.DataFrame, ticker: str):
    """Calculates Quarterly QoQ growth, momentum delta, and status."""
    if ticker not in prices.columns:
        return None, "Ticker data not found in main DataFrame."
 
    series = prices[ticker].dropna()
    if len(series) < 252 * 1.5: # Need at least 1.5 years of data
        return None, "Insufficient history for full analysis."
 
    q_data = series.resample('Q').last().dropna()
    
    if len(q_data) < 5:
        return None, "Need at least 5 quarters of data for momentum."
        
    q_points = q_data.iloc[-5:] 
    
    # --- STABLE SCALAR EXTRACTION ---
    try:
        latest_q_close = q_points.iloc[-1].item()
        q_ago_close = q_points.iloc[-2].item()
        q2_ago_close = q_points.iloc[-3].item()
        
        # Get Avg Last 4 Quarters from the built data frame 
        df_q_metrics = build_period_data(prices[[ticker]], freq="Q")
        avg_last_4 = df_q_metrics['AvgLast4'].iloc[-1] if not df_q_metrics.empty else np.nan
 
        latest_qoq = (latest_q_close / q_ago_close - 1) * 100
        prev_qoq = (q_ago_close / q2_ago_close - 1) * 100
        
        # Momentum Calculation
        momentum_delta = latest_qoq - prev_qoq
        momentum_status = "Accelerating" if momentum_delta > 0.01 else (
            "Decelerating" if momentum_delta < -0.01 else "Stable"
        )
        
    except (ValueError, IndexError):
        return None, "Data shape or indexing error. Cannot calculate."
 
    summary = {
        "Latest QoQ Growth": latest_qoq,
        "Momentum Delta (%)": momentum_delta,
        "Momentum Status": momentum_status,
        "Avg QoQ Growth (Last 4 Quarters)": avg_last_4,
    }
    
    return summary, None
 
 
# =======================================================
# PDF GENERATION FUNCTION (Stabilized)
# =======================================================
# NOTE: To use this, you must run: pip install reportlab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
 
def generate_pdf_report(sector_df, df_master_sub_pdf) -> BytesIO:
    """Generates a PDF report containing all sector and subsector tables."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
 
    # --- Title ---
    story.append(Paragraph("Macro-Micro Investment Report", styles['Title']))
    story.append(Paragraph(f"Generated on: {dt.date.today().strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 12))
 
    # --- 1. Sector Summary Table (Stabilized column renaming) ---
    story.append(Paragraph("1. Major Sector Momentum Summary (Quarterly)", styles['Heading2']))
    
    # Prepare data for ReportLab
    df_sector_pdf = sector_df.copy()
    df_sector_pdf.columns = [
        "Industry", "ETF", "Latest QoQ %", "Momentum Delta %", "Momentum Status", "Avg 4Q Growth %"
    ]
    data_sector = [df_sector_pdf.columns.tolist()] + df_sector_pdf.values.tolist()
    
    # Format floating point numbers in the data structure for PDF table
    for i in range(1, len(data_sector)):
        data_sector[i][2] = f"{data_sector[i][2]:+.2f}%" # Latest QoQ
        data_sector[i][3] = f"{data_sector[i][3]:+.2f}%" # Momentum Delta
        data_sector[i][5] = f"{data_sector[i][5]:.2%}"   # Avg QoQ
 
    table_sector = Table(data_sector, repeatRows=1)
    table_sector.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table_sector)
    story.append(Spacer(1, 24))
 
    # --- 2. Subsector Detailed Analysis ---
    story.append(Paragraph("2. Subsector Deep Dive Analysis (Quarterly Momentum)", styles['Heading2']))
 
    # Group by Sector for cleaner presentation
    for sector_name, df_sub in df_master_sub_pdf.groupby('Sector'):
        story.append(Paragraph(f"Theme: {sector_name}", styles['h3']))
        
        # Prepare data for ReportLab
        df_sub_clean = df_sub[['Subsector', 'ETF', 'Latest QoQ Growth (%)', 'Momentum Delta (%)', 'Momentum Status', 'Avg 4Q Growth (%)']].copy()
        
        # Rename columns for presentation
        df_sub_clean.columns = ['Subsector', 'ETF', 'Latest QoQ %', 'Momentum Delta %', 'Momentum Status', 'Avg 4Q %']
        
        data_sub = [df_sub_clean.columns.tolist()] + df_sub_clean.values.tolist()
 
        # Format floating point numbers
        for i in range(1, len(data_sub)):
            data_sub[i][2] = f"{data_sub[i][2]:+.2f}%" # Latest QoQ
            data_sub[i][3] = f"{data_sub[i][3]:+.2f}%" # Momentum Delta
            data_sub[i][5] = f"{data_sub[i][5]:.2%}"   # Avg 4Q
 
        table_sub = Table(data_sub, repeatRows=1)
        table_sub.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ]))
        story.append(table_sub)
        story.append(Spacer(1, 12))
 
    doc.build(story)
    buffer.seek(0)
    return buffer
 
 
# =======================================================
# LOAD ALL DATA AND SETUP TABS
# =======================================================
 
# Fetch all data (6y history for granularity and Q/Y analysis)
prices_all = get_price_history(all_tickers_to_fetch)
 
 
tab_sector, tab_subsector = st.tabs(["🏛️ Sector Comparison (Tab 1)", "🔬 Subsector Deep Dive (Tab 2)"])
 
# =======================================================
# TAB 1: SECTOR COMPARISON 
# =======================================================
 
with tab_sector:
    st.subheader("Layer 1: Macro Allocation — Cross-Sector Analysis (Last 6 Years)")
 
    # --- UI: FREQUENCY SELECTOR ---
    agg_choice = st.radio("Aggregation level", ["Monthly", "Quarterly", "Yearly"], horizontal=True, key='sector_agg')
    if agg_choice == "Monthly": freq, period_label, metric_label = "M", "Month-end", "MoM"
    elif agg_choice == "Quarterly": freq, period_label, metric_label = "Q", "Quarter-end", "QoQ"
    else: freq, period_label, metric_label = "Y", "Year-end", "YoY"
    
    # Filter prices to just the main sector ETFs
    sector_prices = prices_all[[c for c in main_etf_list if c in prices_all.columns]].copy()
    df_period = build_period_data(sector_prices, freq=freq) 
    
    
    # --- PLOTLY CHART ---
    st.markdown("---")
    st.subheader(f"📈 Sector ETF {period_label} Prices")
    
    if df_period.empty:
        st.info("No data available for plotting.")
    else:
        fig = go.Figure()
        
        # Order ETFs by latest price high -> low
        latest_points = (df_period.sort_values("Date").groupby("ETF").tail(1).set_index("ETF"))
        etfs_with_data = [etf for etf in main_etf_list if etf in latest_points.index]
        ordered_etfs = (latest_points.loc[etfs_with_data].sort_values("Price", ascending=False).index.tolist())
 
        for ticker in ordered_etfs:
            df_t = df_period[df_period["ETF"] == ticker]
            if df_t.empty: continue
            
            customdata = np.stack(
                [df_t["Industry"].values, df_t["ETF"].values, df_t["PeriodLabel"].values,
                 df_t["PeriodReturn"].values, df_t["AvgLast4"].values], axis=-1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_t["Date"], y=df_t["Price"], mode="lines+markers",
                    name=f"{ticker} ({df_t['Industry'].iloc[0]})", customdata=customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]} (%{customdata[1]})</b><br>"
                        f"{metric_label}: %{{customdata[3]:.2%}} | Avg4: %{{customdata[4]:.2%}}<extra></extra>"
                    ),
                )
            )
        
        fig.update_layout(hovermode="x unified", xaxis_title=period_label, yaxis_title="Price", 
                            legend_title="ETF (Industry)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- QUARTERLY SUMMARY TABLE (With Momentum) ---
    st.markdown("---")
    st.subheader("📋 Quarterly Growth and Momentum Summary")
    
    rows_sector = []
    for etf in main_etf_list:
        # Use full quarterly metrics function
        summary, error = get_quarterly_summary_metrics(prices_all, etf)
        
        if summary:
            rows_sector.append({
                "Industry": ticker_to_sector_name[etf],
                "ETF": etf,
                "Latest QoQ Growth": summary['Latest QoQ Growth'],
                "Momentum Delta (%)": summary['Momentum Delta (%)'],
                "Momentum Status": summary['Momentum Status'],
                "Avg QoQ Growth (Last 4 Quarters)": summary['Avg QoQ Growth (Last 4 Quarters)'],
            })
        elif error:
            rows_sector.append({"Industry": ticker_to_sector_name[etf], "ETF": etf, 
                                 "Latest QoQ Growth": np.nan, "Momentum Delta (%)": np.nan,
                                 "Momentum Status": "N/A", "Avg QoQ Growth (Last 4 Quarters)": np.nan})
 
 
    df_sector_table = pd.DataFrame(rows_sector).sort_values("Industry")
    
    if df_sector_table.empty: st.info("Not enough quarterly data.")
    else:
        def color_momentum_tab1(val):
            if val == 'Accelerating': return 'background-color: #d4edda; color: green'
            elif val == 'Decelerating': return 'background-color: #f8d7da; color: red'
            elif val == 'Stable': return 'background-color: #fff3cd; color: #856404'
            return ''
            
        st.dataframe(df_sector_table.style.format({
            "Latest QoQ Growth": "{:+.2f}%", 
            "Momentum Delta (%)": "{:+.2f}%",
            "Avg QoQ Growth (Last 4 Quarters)": "{:.2%}",
        }).applymap(color_momentum_tab1, subset=['Momentum Status']), hide_index=True, use_container_width=True)
 
    st.caption(f"Last Data Update: **{st.session_state.get('last_update', 'N/A')}**")
 
 
# =======================================================
# TAB 2: SUBSECTOR DEEP DIVE
# =======================================================
 
# Prepare master subsector PDF table list (Quarterly)
pdf_subsector_table_list_raw = []
for sector_name, subsector_data in SUBSECTOR_MAP.items():
    for sub_name, ticker in subsector_data.items():
        summary, error = get_quarterly_summary_metrics(prices_all, ticker)
        if summary:
            pdf_subsector_table_list_raw.append({
                "Sector": sector_name,
                "Subsector": sub_name,
                "ETF": ticker,
                "Latest QoQ Growth (%)": summary['Latest QoQ Growth'],
                "Momentum Delta (%)": summary['Momentum Delta (%)'],
                "Momentum Status": summary['Momentum Status'],
                "Avg 4Q Growth (%)": summary['Avg QoQ Growth (Last 4 Quarters)'],
            })
 
df_master_sub_pdf = pd.DataFrame(pdf_subsector_table_list_raw)
df_master_sub_pdf = df_master_sub_pdf.sort_values(by=['Sector', 'Subsector'])
 
 
with tab_subsector:
    st.header("🔬 Layer 2: Subsector Comparative Analysis")
    st.markdown("Analyze high-conviction thematic subsectors by comparing growth rates.")
 
    st.info(f"Data Granularity: **Daily/Weekly/Monthly/Quarterly/Yearly** | Last Data Update: **{st.session_state.get('last_update', 'N/A')}**")
    
    # 1. Sector Selector Control
    sector_choice = st.selectbox(
        "Select Main Sector to Analyze Subsectors:",
        list(SUBSECTOR_MAP.keys()),
        key='subsector_selector'
    )
    
    st.markdown(f"### Thematic Deep Dive: {sector_choice}")
 
    # 2. Aggregation Control
    sub_agg_choice = st.radio(
        "Select Aggregation for Chart and Table:",
        ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
        horizontal=True,
        key="subsector_agg",
    )
 
    if sub_agg_choice == "Daily": freq_sub, x_title_sub, metric_label_sub = "D", "Date (Daily)", "DoD"
    elif sub_agg_choice == "Weekly": freq_sub, x_title_sub, metric_label_sub = "W", "Week End Date", "WoW"
    elif sub_agg_choice == "Monthly": freq_sub, x_title_sub, metric_label_sub = "M", "Month End Date", "MoM"
    elif sub_agg_choice == "Quarterly": freq_sub, x_title_sub, metric_label_sub = "Q", "Quarter End Date", "QoQ"
    else: freq_sub, x_title_sub, metric_label_sub = "Y", "Year End Date", "YoY"
    
    # 3. Data Preparation
    subsectors = SUBSECTOR_MAP[sector_choice]
    sub_tickers = [t for t in subsectors.values() if t in prices_all.columns]
    
    prices_filtered = prices_all.loc[:, [t for t in sub_tickers if t in prices_all.columns]]
    df_sub_period = build_period_data(prices_filtered, freq=freq_sub)
 
    # --- CHART AND TABLE STRUCTURE ---
    
    st.markdown("---")
    st.subheader(f"📈 Price Comparison: {sector_choice} Subsectors ({sub_agg_choice})")
    
    if df_sub_period.empty:
        st.info("No data to plot for these subsectors.")
    else:
        fig_sub_price = go.Figure()
        
        latest_sub_points = (df_sub_period.sort_values("Date").groupby("ETF").tail(1).set_index("ETF"))
        ordered_sub_etfs = (latest_sub_points.sort_values("Price", ascending=False).index.tolist())
        
        for ticker in ordered_sub_etfs:
            df_t = df_sub_period[df_sub_period["ETF"] == ticker]
            if df_t.empty: continue
            
            # Use the subsector name from the SUBSECTOR_MAP, falling back to ticker
            sub_name = next((name for name, t in subsectors.items() if t == ticker), ticker) 
            
            # FIX: Ensure sub_name is in customdata[0] for the tooltip
            customdata = np.stack(
                [np.full_like(df_t["Industry"].values, sub_name, dtype=object), # Use the robustly determined sub_name
                 df_t["ETF"].values, df_t["PeriodLabel"].values,
                 df_t["PeriodReturn"].values, df_t["AvgLast4"].values], axis=-1
            )
            
            fig_sub_price.add_trace(
                go.Scatter(
                    x=df_t["Date"], y=df_t["Price"], mode="lines+markers",
                    name=f"{ticker} ({sub_name})", 
                    customdata=customdata,
                    # Tooltip uses customdata[0] for Subsector Name and customdata[1] for Ticker
                    hovertemplate=f"<b>%{{customdata[0]}} (%{{customdata[1]}})</b><br>{metric_label_sub}: %{{customdata[3]:.2%}} | Avg4: %{{customdata[4]:.2%}}<extra></extra>",
                )
            )
        
        fig_sub_price.update_layout(hovermode="x unified", xaxis_title=x_title_sub, yaxis_title="Price", 
                                    legend_title="Subsector ETF", height=500)
        
        st.plotly_chart(fig_sub_price, use_container_width=True)
 
    # ------------------ Summary Table (Final Unified Structure) ------------------
    st.markdown("---")
    st.subheader(f"📋 {sub_agg_choice} Growth Summary ({metric_label_sub})")
 
    table_rows = []
    
    # Logic to populate the table dynamically based on chosen frequency
    df_latest_sub_period = df_sub_period[df_sub_period["Date"] == df_sub_period["Date"].max()]
 
    for sub_name, ticker in subsectors.items():
        d = df_latest_sub_period[df_latest_sub_period["ETF"] == ticker]
        
        if d.empty: continue
        
        row = {
            "Subsector": sub_name,
            "ETF": ticker,
            f"Latest {metric_label_sub} Return": d["PeriodReturn"].iloc[0] * 100,
            "Avg Return (Last 4 Periods)": d["AvgLast4"].iloc[0],
        }
 
        # Add Momentum Columns ONLY IF Frequency is Quarterly
        if freq_sub == 'Q':
            summary, error = get_quarterly_summary_metrics(prices_all, ticker)
            if summary:
                row["Momentum Delta (%)"] = summary['Momentum Delta (%)']
                row["Momentum Status"] = summary['Momentum Status']
            else:
                row["Momentum Delta (%)"] = np.nan
                row["Momentum Status"] = "N/A"
        
        table_rows.append(row)
    
    df_sub_table = pd.DataFrame(table_rows).sort_values("Subsector")
 
    if not df_sub_table.empty:
        
        format_dict = {f"Latest {metric_label_sub} Return": "{:+.2f}%", 
                        "Avg Return (Last 4 Periods)": "{:.2%}"}
        
        # Conditional formatting logic for Momentum
        if freq_sub == 'Q':
            format_dict["Momentum Delta (%)"] = "{:+.2f}%"
            
            def color_momentum_final(val):
                if val == 'Accelerating': return 'background-color: #d4edda; color: green'
                elif val == 'Decelerating': return 'background-color: #f8d7da; color: red'
                elif val == 'Stable': return 'background-color: #fff3cd; color: #856404'
                return ''
            
            style_df = df_sub_table.style.format(format_dict).applymap(color_momentum_final, subset=['Momentum Status'])
        else:
            style_df = df_sub_table.style.format(format_dict)
            
        st.dataframe(
            style_df,
            hide_index=True,
            use_container_width=True
        )
 
        st.caption(f"Returns are calculated based on the selected **{sub_agg_choice}** period. Momentum metrics are only available in Quarterly mode.")
 
# =======================================================
# ---- SIDEBAR: ABOUT THE CREATOR (Minimal) ----
# =======================================================
st.sidebar.markdown("---") 
st.sidebar.header("👋 Creator")

st.sidebar.markdown(f"**Sahith Krishna Kesina**")

st.sidebar.markdown(
    """
    I created this dashboard because I was once the young investor constantly reading but struggling to find a **well-structured plan**. 
    
    This tool is the guidance I wished I had: a clear, step-by-step roadmap showing you *where* to look next, starting with the big picture (Macro) and working down to the specific sectors and subsectors.
    
    It’s built for anyone who wants to start disciplined analysis but doesn't know where to begin. Happy to connect and share the journey.
    """
)

st.sidebar.markdown(
    """
    * [**Connect on LinkedIn**](https://www.linkedin.com/in/sahith-krishna-kesina-09634b260/)
    """
) 


# =======================================================
# DOWNLOAD BUTTON (PDF Export)
# =======================================================
 
st.sidebar.markdown("---")
st.sidebar.subheader("⬇️ Download Full Report")
st.sidebar.markdown("Generate a PDF summarizing all Sector and Subsector Quarterly Momentum.")
 
 
# --- PDF GENERATION BUTTON ---
try:
    # Requires Reportlab to be installed
    pdf_buffer = generate_pdf_report(df_sector_table, df_master_sub_pdf) # Using df_sector_table from Tab 1 scope
    
    st.sidebar.download_button(
        label="Download Investment PDF",
        data=pdf_buffer,
        file_name="Macro_Momentum_Report.pdf",
        mime="application/pdf",
    )
 
except ImportError:
    st.sidebar.warning("Install 'reportlab' to enable PDF export (pip install reportlab).")