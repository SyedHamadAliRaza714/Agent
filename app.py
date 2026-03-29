import os
import json
import re
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Type

if "GEMINI_API_KEY" not in os.environ:
    raise RuntimeError("Please add GEMINI_API_KEY in Space Secrets")

GEMINI_MODEL = "gemini-2.5-flash-lite" 

class FinanceArgs(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. 'AAPL'")
    period: str = Field(..., description="yfinance period: 1d,5d,1mo,…")

class NewsArgs(BaseModel):
    query: str = Field(..., description="Ticker for news")

period_map = {
    "today": "1d", "1day": "1d", "day": "1d",
    "last week": "5d", "week": "5d", "7days": "5d",
    "past month": "1mo", "month": "1mo", "1month": "1mo",
    "last month": "1mo",
    "3months": "3mo", "quarter": "3mo",
    "6months": "6mo", "half year": "6mo",
    "last year": "1y", "year": "1y", "1year": "1y",
    "2years": "2y", "5years": "5y", "10years": "10y",
    "ytd": "ytd", "max": "max"
}

@tool("normalize_period")
def normalize_period_tool(user_period: str) -> str:
    """Convert human period → yfinance format."""
    key = user_period.strip().lower()
    if key in period_map:
        return period_map[key]
    valid = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    if key in valid:
        return key
    raise ValueError(f"Invalid period: '{user_period}'")

class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceTool"
    description: str = "Fetches price + fundamentals."
    args_schema: Type[BaseModel] = FinanceArgs

    def _run(self, ticker: str, period: str) -> str:
        try:
            stock = yf.Ticker(ticker.upper())
            df = stock.history(period=period)
            if df.empty or len(df) < 2:
                return json.dumps({"error": "No data"})
            df = df.tail(30)
            latest, prev = df.iloc[-1], df.iloc[-2]
            change = latest['Close'] - prev['Close']
            pct_change = (change / prev['Close']) * 100
            volatility = df['Close'].pct_change().std() * 100
            info = stock.info
            summary = {
                "ticker": ticker.upper(),
                "period": period,
                "latest_price": round(latest['Close'], 2),
                "pct_change": round(pct_change, 2),
                "volatility_pct": round(volatility, 2),
                "trend": "up" if latest['Close'] > df['Close'].mean() else "down",
                "revenue_b": round(info.get("totalRevenue",0)/1e9, 2),
                "eps": round(info.get("trailingEps",0), 2),
                "pe_ratio": round(info.get("trailingPE",0), 2),
                "market_cap_b": round(info.get("marketCap",0)/1e9, 2)
            }
            return json.dumps({"stock_data": summary})
        except Exception as e:
            return json.dumps({"error": str(e)})

class NewsScraperTool(BaseTool):
    name: str = "NewsScraperTool"
    description: str = "Top 3 Yahoo Finance headlines."
    args_schema: Type[BaseModel] = NewsArgs

    def _run(self, query: str) -> str:
        try:
            url = f"https://finance.yahoo.com/quote/{query.upper()}/news"
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            headlines = [h.text.strip() for h in soup.find_all('h3', class_='Mb(5px)')][:3]
            return json.dumps({"headlines": headlines or ["No headlines"]})
        except:
            return json.dumps({"headlines": ["Unavailable"]})

orchestrator = Agent(
    role="Orchestrator",
    goal="Synthesize research & analysis into a concise report.",
    backstory="Lead analyst – combine data & insights.",
    llm=GEMINI_MODEL, verbose=False, allow_delegation=True, max_iter=3,
    memory=False, cache=True)

researcher = Agent(
    role="Research Agent",
    goal="Collect stock summary + 3 news headlines.",
    backstory="Fetch only structured data – no analysis.",
    tools=[YahooFinanceTool(), NewsScraperTool(), normalize_period_tool],
    llm=GEMINI_MODEL, verbose=False, allow_delegation=False, max_iter=3,
    memory=False, cache=True)

analyst = Agent(
    role="Analysis Agent",
    goal="Give sentiment + one-line strategy.",
    backstory="Concise, actionable insights.",
    llm=GEMINI_MODEL, verbose=False, allow_delegation=False, max_iter=3,
    memory=False, cache=True)

task_research = Task(
    description=(
        "From user query: '{query}',\n"
        "1. Extract **ticker** and **period**.\n"
        "2. Call **normalize_period** → valid yfinance period.\n"
        "3. Call **YahooFinanceTool** and **NewsScraperTool**.\n"
        "4. Return ONLY JSON:\n"
        "```json\n"
        "{{\"stock_data\":{{...}},\"news\":[\"...\",\"...\",\"...\"]}}\n"
        "```"
    ),
    expected_output="Valid JSON",
    agent=researcher
)

task_analyze = Task(
    description=(
        "Using ONLY the JSON from Research:\n"
        "Analyze price, volatility, trend, news.\n"
        "Output exactly:\n"
        "Sentiment: Positive / Neutral / Negative\n"
        "Strategy: [One short sentence]"
    ),
    expected_output="Two lines",
    agent=analyst
)

orchestrator_task = Task(
    description=(
        "FINAL agent. Take Research JSON + Analysis text.\n"
        "Write 3–5 sentence report.\n"
        "Start with `Final Answer:` and nothing else.\n"
        "Example:\n"
        "Final Answer: TSLA fell 0.86% to $424.19. Revenue: $20.9B. Sentiment: Positive. Strategy: Buy on momentum."
    ),
    expected_output="Text starting with 'Final Answer:'",
    agent=orchestrator
)

crew = Crew(
    agents=[orchestrator, researcher, analyst],
    tasks=[task_research, task_analyze, orchestrator_task],
    process=Process.sequential,
    verbose=False,
    max_rpm=60, 
    cache=True
)

st.set_page_config(page_title="FinAgent Pro", layout="wide", page_icon="gem")

st.markdown("""
<style>
    .main-header {
        font-family: 'Georgia', serif;
        font-size: 5.2rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #14B8A6, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2.5rem;
        font-style: italic;
    }
    .metric-card {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        color: black !important;
        padding: 1.4rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(124,58,237,0.35),
                    0 0 20px rgba(124,58,237,0.25);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .metric-card:hover { transform: translateY(-6px); }
    .metric-card.positive { background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%); }
    .metric-card.negative { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
    .news-item, .report-box {
        background: #FAF5FF;
        padding: 1.2rem;
        margin: 0.7rem 0;
        border-radius: 14px;
        border-left: 6px solid #7C3AED;
        box-shadow: 0 4px 12px rgba(124,58,237,0.15);
    }
    .report-box pre {
        color: #7C3AED !important;
        white-space: pre-wrap;
        margin: 0;
        font-weight: 500;
        line-height: 1.7;
    }
    .section-title {
        font-size: 2rem;
        color: #5B21B6;
        font-weight: 700;
        border-bottom: 4px solid #14B8A6;
        padding-bottom: 0.7rem;
        margin-top: 2.5rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED, #6D28D9);
        color: white;
        border-radius: 14px;
        padding: 0.8rem 1.8rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #6D28D9, #5B21B6);
        transform: translateY(-3px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">FinAgent Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Equity Intelligence • Real-Time • Institutional Grade • Powered by Gemini</p>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3.5, 1, 1])
with c1: query = st.text_input("", placeholder="e.g., Analyze Tesla last month", label_visibility="collapsed")
with c2: analyze_btn = st.button("Analyze", use_container_width=True)
with c3: st.button("Clear", use_container_width=True)

if analyze_btn and query:
    with st.spinner("Gemini Agents Analyzing…"):
        result = crew.kickoff(inputs={"query": query})

    raw_str = str(result)
    final_report = raw_str.split("Final Answer:")[-1].strip() if "Final Answer:" in raw_str else "Report unavailable."

    try:
        m = re.search(r'\{.*"stock_data".*?\}', raw_str, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        stock = data.get("stock_data", {})
        news = data.get("news", [])
    except:
        stock, news = {}, []

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<h3 class="section-title">Stock Intelligence</h3>', unsafe_allow_html=True)
        if stock and "latest_price" in stock:
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><h3>${stock["latest_price"]}</h3><p>Price</p></div>', unsafe_allow_html=True)
            with c2:
                cls = "positive" if stock.get("pct_change",0)>0 else "negative"
                st.markdown(f'<div class="metric-card {cls}"><h3>{stock.get("pct_change",0):+.2f}%</h3><p>Change</p></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-card"><h3>{stock.get("volatility_pct",0):.1f}%</h3><p>Volatility</p></div>', unsafe_allow_html=True)
            with c4:
                trend = "Up" if stock.get("trend")=="up" else "Down"
                colr = "#14B8A6" if trend=="Up" else "#EF4444"
                st.markdown(f'<div class="metric-card" style="background:{colr}"><h3>{trend}</h3><p>Trend</p></div>', unsafe_allow_html=True)

            st.markdown("**Key Financials**")
            f1,f2,f3,f4 = st.columns(4)
            with f1: st.markdown(f'<div class="metric-card"><h4>${stock.get("revenue_b",0)}B</h4><p>Revenue</p></div>', unsafe_allow_html=True)
            with f2: st.markdown(f'<div class="metric-card"><h4>${stock.get("eps",0)}</h4><p>EPS</p></div>', unsafe_allow_html=True)
            with f3: st.markdown(f'<div class="metric-card"><h4>{stock.get("pe_ratio",0)}</h4><p>PE</p></div>', unsafe_allow_html=True)
            with f4: st.markdown(f'<div class="metric-card"><h4>${stock.get("market_cap_b",0)}B</h4><p>Market Cap</p></div>', unsafe_allow_html=True)
        else:
            st.info("No stock data.")

    with col_b:
        st.markdown('<h3 class="section-title">Market Pulse</h3>', unsafe_allow_html=True)
        if news:
            for item in news[:3]:
                st.markdown(f'<div class="news-item">• {item}</div>', unsafe_allow_html=True)
        else:
            st.info("No news.")

    st.markdown('<h3 class="section-title">Institutional Report</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-box"><pre>{final_report}</pre></div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem;background:linear-gradient(135deg,#FAF5FF 0%,#F0FDF4 100%);border-radius:20px;margin:2rem 0;">
        <h2 style="color:#5B21B6;margin-bottom:1rem;">Real-Time AI Financial Research</h2>
        <p style="color:#64748B;font-size:1.1rem;">Enter any stock + time period. Get institutional-grade insights in seconds.</p>
        <p style="margin-top:1.5rem;color:#7C3AED;font-weight:600;">Powered by Gemini • CrewAI • Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)
