# Price Attribution Dashboard

A Streamlit-based web application for analyzing healthcare sector price movements using AI-powered web search and attribution analysis.

## Features

- 📊 **Interactive Visualizations**: Heatmaps, bar charts, and distribution plots
- 🔍 **Automated Web Search**: Real-time market intelligence for significant price movers
- 🤖 **AI-Generated Insights**: Executive summaries powered by Claude
- 📥 **PDF Export**: Professional reports ready for sharing
- 🎯 **Subsector Analysis**: Drill-down into Provider, Pharma, Devices, Payors, and Tech sectors

## Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd price_attribution_dashboard

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

You need an Anthropic API key for web search and AI features.

**Option A: Environment Variable**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Option B: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

### 3. Run the App

```bash
streamlit run price_attribution_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Step 1: Upload Data

1. Click **"Upload movers.csv"** in the sidebar
2. Select your CSV file with price movement data
3. Verify the file loads successfully

### Step 2: Run Analysis

1. Adjust the **"Minimum move threshold"** if desired (default: 0.5%)
2. Click **"Run Attribution Analysis"**
3. Wait for web searches and AI processing to complete

### Step 3: Review Results

The analysis page displays:
- Executive summary with key insights
- Market overview metrics
- Interactive visualizations
- Detailed subsector breakdowns with web context

### Step 4: Export PDF

1. Click **"Generate PDF"** in the sidebar
2. Download the formatted report

## CSV Format

Your CSV file must include these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `ticker` | String | Company name/ticker | "ACME HEALTHCARE INC" |
| `subsector` | String | Sector classification | "Provider", "Pharma", "Devices" |
| `rating` | String | Credit rating | "Ba2", "B1", "Caa1" |
| `return_1w` | Float | 1-week return (%) | 1.5 |
| `return_1d` | Float | 1-day return (%) | 0.3 |
| `date` | String | Observation date | "2024-11-02" |

### Example CSV:

```csv
ticker,subsector,rating,return_1w,return_1d,date
RADIOLOGY PARTNERS INC,Provider,B3,-1.21,-0.20,11-02-2024
IQVIA INC,Pharma,Ba1,-0.58,0.05,11-02-2024
SERVICE CORP INTL,Provider,Ba2,0.54,0.27,11-02-2024
```

## Troubleshooting

### "Validation failed: Missing required columns"
**Solution**: Verify your CSV has all 6 required columns with exact names

### "ANTHROPIC_API_KEY not found"
**Solution**: Set API key in environment variable or `.streamlit/secrets.toml`

### "Search failed for [subsector]"
**Solution**: Temporary API issue - try running analysis again

### Analysis is slow
**Solution**: Increase minimum move threshold to reduce web searches

---

**Built with Streamlit, Claude, and Plotly**
