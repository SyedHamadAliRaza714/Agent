# 💎 FinAgent Pro: Multi-Agent AI Financial Analyst

An institutional-grade, AI-powered equity intelligence platform that leverages a multi-agent AI system to fetch, analyze, and report on real-time stock market data in seconds.

## 📋 Overview

FinAgent Pro transforms complex financial research into a seamless experience. By utilizing **CrewAI** and the lightning-fast **Groq** inference engine (Llama 3.1), the platform orchestrates a team of specialized AI agents. You simply type a natural language query (e.g., *"Analyze Tesla's performance last month"*), and the AI automatically fetches ticker data, scrapes the latest news, analyzes market sentiment, and synthesizes an actionable financial report.

## ✨ Key Features

### 🤖 Multi-Agent AI Architecture
* **Research Agent** - Autonomously figures out the correct stock ticker, normalizes the timeframe, and scrapes Yahoo Finance for exact pricing and headlines.
* **Analysis Agent** - Ingests the raw data to calculate volatility, identify trends, gauge news sentiment, and formulate a one-line institutional strategy.
* **Orchestrator Agent** - Synthesizes all findings into a concise, professional final report.

### 📈 Real-Time Market Intelligence
* **Live Financial Data** - Extracts live pricing, Revenue, EPS, P/E Ratios, and Market Cap using the `yfinance` API.
* **Automated News Scraping** - Uses `BeautifulSoup` to bypass standard blocks and scrape the top 3 real-time headlines for your requested stock.
* **Dynamic Timeframes** - Understands natural language periods (e.g., "last week", "ytd", "6 months") and converts them into valid financial data queries.

### 🎨 Premium UI/UX
* **Modern Dashboard** - Built with a custom-styled Streamlit interface featuring a sleek Purple/Teal gradient theme.
* **Interactive Metric Cards** - Automatically color-coded (Green/Red) KPI cards for quick visual analysis of price changes and trends.
* **Transparent Agent Logging** - View the raw "thought process" and outputs of each individual AI agent in a beautifully formatted report box.

## 🚀 Technology Stack

* **Frontend:** Streamlit, HTML/CSS (Custom Styling)
* **AI Framework:** CrewAI, LangChain
* **LLM Provider:** Gemini (`llama-3.1-8b-instant` for ultra-fast, free inference)
* **Financial Data:** Yahoo Finance (`yfinance`)
* **Web Scraping:** BeautifulSoup4, Requests
* **Deployment:** Google Colab, PyNgrok

## 📦 Installation & Setup

### Prerequisites
* Python 3.8+
* **Groq API Key** (Get a free, high-speed API key at [console.groq.com](https://console.groq.com/))
* **Ngrok Auth Token** (For Colab deployment, via [ngrok.com](https://ngrok.com/))

### Local Setup
Clone the repository:
```bash
git clone [https://github.com/HamadAliRaza/FinAgent-Pro.git](https://github.com/HamadAliRaza/FinAgent-Pro.git)
cd FinAgent-Pro
```
Install the required dependencies:

```Bash
pip install crewai[tools] langchain-huggingface huggingface_hub yfinance beautifulsoup4 requests streamlit pandas litellm langchain-groq
```
Set your API key as an environment variable:

Linux/Mac:

```Bash
export GROQ_API_KEY="your_groq_api_key"
```
Windows (Command Prompt):

```DOS
set GROQ_API_KEY="your_groq_api_key"
```
(Alternatively, you can create a .env file in the root directory and add your key there).

Run the application:

```Bash
streamlit run app.py
```
## 📖 Usage
Launch the App: Open the Streamlit web URL provided in your terminal.

Enter a Query: Use natural, human language in the search bar.

Example 1: "Analyze Apple over the last month."

Example 2: "Give me a breakdown of NVDA for the past week."

Wait for the Crew: Watch the spinner as the Orchestrator, Researcher, and Analyst agents collaborate.

Review the Intelligence: Read the final institutional report, view the live stock metrics, and check the latest news pulse.

## 📁 Project Structure
Below is the organized directory layout for the FinAgent Pro application.

```Plaintext
FinAgent-Pro/
├── app.py                      # Main Streamlit UI, CrewAI setup, and Tool definitions
├── requirements.txt            # Python dependencies list
├── FinAgent_Colab.ipynb        # Ready-to-run Jupyter Notebook for Colab deployment
├── .env                        # Environment variables (Add your GROQ_API_KEY here)
└── README.md                   # Project documentation
```
## 🎓 Learning Outcomes
This project demonstrates proficiency in:

Multi-Agent Orchestration - Successfully building and delegating tasks to autonomous AI agents using the CrewAI framework.

Fast LLM Integration - Utilizing Groq's LPU infrastructure for near-instantaneous language model inference.

Custom Tool Creation - Writing highly specific, error-handled Python classes (BaseTool) for AI agents to interact with external APIs.

Frontend Design - Transforming standard Streamlit components into a premium, responsive web dashboard using raw CSS injection.

## 🔮 Future Enhancements
Add a Charting Tool for the Research Agent to generate and display candlestick graphs.

Integrate SEC Filings parsing (10-K, 10-Q) for deep fundamental analysis.

Allow users to input a portfolio of multiple stocks for a comparative multi-agent report.

Export the final institutional report as a downloadable PDF.

## 👨‍💻 Author
Hamad Ali Raza

GitHub: [@HamadAliRaza](https://github.com/SyedHamadAliRaza714)

Project Link: [[FinAgent Pro Repository](https://github.com/SyedHamadAliRaza714/FinAgent-Pro-)(

## 📄 License
This project is open source and available under the MIT License.

🙏 Acknowledgments
Market data generously provided by Yahoo Finance.
