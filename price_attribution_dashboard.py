#!/usr/bin/env python3
"""
Price Attribution Dashboard
================================
Streamlit-based interactive dashboard for healthcare price attribution analysis.
"""

import os
import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key from multiple sources in priority order:
    1. Streamlit secrets (.streamlit/secrets.toml)
    2. Environment variables
    3. Returns None if not found
    """
    key_name = f"{provider.upper()}_API_KEY"
    
    # Try Streamlit secrets first
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    # Fall back to environment variable
    return os.getenv(key_name)


SUBSECTOR_COLORS = {
    "Provider": "#3b82f6",
    "Devices": "#f59e0b", 
    "Pharma": "#10b981",
    "Payors": "#8b5cf6",
    "Tech": "#ef4444",
    "REIT": "#ec4899"
}

RATING_ORDER = ["Baa3", "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca", "WR"]

REQUIRED_COLUMNS = {
    'ticker': 'string',
    'subsector': 'string', 
    'rating': 'string',
    'return_1w': 'float',
    'return_1d': 'float',
    'date': 'string'
}

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Price Attribution Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stat-box {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #dcfce7;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .error-box {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    .subsector-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .explanation-box {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #f59e0b;
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WEB SEARCH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class WebSearchEngine:
    """Searches web to explain subsector movements."""
    
    def __init__(self, provider: str = 'anthropic', api_key: str = None):
        self.provider = provider.lower()
        
        # Get API key from provided value, secrets, or environment
        if api_key is None:
            api_key = get_api_key(self.provider)
        
        if self.provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = "claude-sonnet-4-20250514"
            except Exception as e:
                st.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
        elif self.provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                self.model = "gpt-4o"
            except Exception as e:
                st.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def search_subsector(self, subsector: str, avg_return: float, tickers: List[str], date_str: str) -> str:
        """Search web for why a subsector moved."""
        
        if not self.client:
            return ""
        
        ticker_str = " ".join(tickers[:5]) if tickers else ""
        direction = "increase" if avg_return > 0 else "decline"
        
        search_query = f"healthcare {subsector} sector {direction} {ticker_str} {date_str}"
        
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    tools=[{
                        "type": "web_search_20250305",
                        "name": "web_search"
                    }],
                    messages=[{
                        "role": "user",
                        "content": f"""Search the web for: {search_query}

Then explain in 3-4 sentences why the {subsector} healthcare subsector moved {avg_return:+.1f}% during the week ending {date_str}.

CRITICAL INSTRUCTIONS:
- ONLY reference events from the week ending {date_str}
- DO NOT mention any specific dates unless you found them in the search results
- If you don't know the exact date of an event, say "recently" or "this week" instead of making up a date
- DO NOT say "November 2024" or any other made-up date

Focus on:
- Regulatory or policy changes
- Major earnings reports or guidance updates  
- M&A activity or restructuring
- Industry-wide trends or market conditions
- Specific company news

Be specific about companies and events, but do NOT invent dates."""
                    }]
                )
                
                explanation = []
                for block in response.content:
                    if hasattr(block, 'text') and block.text.strip():
                        explanation.append(block.text.strip())
                
                return " ".join(explanation)
            
            elif self.provider == 'openai':
                prompt = f"""Based on recent healthcare market developments, explain why the {subsector} subsector moved {avg_return:+.1f}% around {date_str}.

Key tickers: {ticker_str}

Provide a 3-4 sentence explanation focusing on:
- Recent regulatory or policy changes
- Major earnings or guidance updates
- M&A activity or restructuring  
- Industry trends or market conditions

Be specific about companies and events."""
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a healthcare sector analyst providing market context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                
                explanation = response.choices[0].message.content.strip()
                return f"[Based on market knowledge] {explanation}"
        
        except Exception as e:
            st.warning(f"Search failed for {subsector}: {str(e)[:100]}")
            return ""
        
        return ""
    
    def search_all_subsectors(self, subsector_data: Dict, date_str: str, min_move: float = 0.5) -> Dict[str, str]:
        """Search web for all subsectors that moved significantly."""
        
        explanations = {}
        
        # Sort by absolute movement
        sorted_subsectors = sorted(
            subsector_data.items(),
            key=lambda x: abs(x[1]['avg_return']),
            reverse=True
        )
        
        # Find subsectors with significant moves
        eligible = [(s, d) for s, d in sorted_subsectors if abs(d['avg_return']) >= min_move]
        
        if not eligible:
            return {}
        
        # Progress bar for web searches
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (subsector, data) in enumerate(eligible):
            status_text.text(f"Searching web for {subsector}... ({idx+1}/{len(eligible)})")
            
            tickers = data.get('tickers', [])
            
            explanation = self.search_subsector(
                subsector=subsector,
                avg_return=data['avg_return'],
                tickers=tickers,
                date_str=date_str
            )
            
            if explanation:
                explanations[subsector] = explanation
            
            progress_bar.progress((idx + 1) / len(eligible))
        
        progress_bar.empty()
        status_text.empty()
        
        return explanations


def generate_narrative(subsector_data: Dict, subsector_explanations: Dict, provider: str, api_key: str = None, date_str: str = None) -> str:
    """Generate overall narrative combining subsector data and web explanations."""
    
    # Calculate overall stats
    all_returns = [data['avg_return'] for data in subsector_data.values()]
    overall_avg = np.mean(all_returns)
    
    # Build prompt
    prompt = f"""You are a senior credit analyst writing a market commentary.

OVERALL PERFORMANCE:
- Average healthcare sector return: {overall_avg:+.1f}%

SUBSECTOR BREAKDOWN:
"""
    
    for subsector, data in sorted(subsector_data.items(), key=lambda x: x[1]['avg_return'], reverse=True):
        prompt += f"\n{subsector}:"
        prompt += f"\n  - Average return: {data['avg_return']:+.1f}%"
        prompt += f"\n  - Number of names: {data['count']}"
        prompt += f"\n  - Key tickers: {', '.join(data['tickers'][:5])}"
        
        if subsector in subsector_explanations:
            prompt += f"\n  - Web research: {subsector_explanations[subsector]}"
    
    prompt += f"""\n\nWrite a 2-3 sentence opening paragraph that:
1. Summarizes the overall market performance
2. Highlights key subsector movements
3. Explains the main drivers based on the web research

CRITICAL: Do NOT mention any specific dates unless they appear in the web research above. Do NOT say "In early November 2024" or make up any dates. If you need to reference time, say "this week" or "recently".

Be specific about companies and events. Write in past tense."""
    
    try:
        if provider == 'anthropic':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key or get_api_key('anthropic'))
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.content[0].text.strip()
        
        elif provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key or get_api_key('openai'))
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior credit analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.warning(f"Narrative generation failed: {e}")
        return f"Healthcare sector moved {overall_avg:+.1f}% on average this week."


# ══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate uploaded CSV file."""
    
    if df is None or df.empty:
        return False, "CSV file is empty"
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # Check data types
    for col, dtype in REQUIRED_COLUMNS.items():
        if dtype == 'float':
            try:
                pd.to_numeric(df[col], errors='coerce')
            except:
                return False, f"Column '{col}' must contain numeric values"
        elif dtype == 'string':
            if not df[col].dtype == 'object':
                return False, f"Column '{col}' must contain text values"
    
    # Check for null values in critical columns
    critical_nulls = df[['ticker', 'subsector', 'return_1w']].isnull().sum()
    if critical_nulls.any():
        return False, f"Found null values in critical columns: {critical_nulls[critical_nulls > 0].to_dict()}"
    
    return True, "Validation successful"


@st.cache_data
def aggregate_by_subsector(movers_df: pd.DataFrame) -> Dict:
    """Aggregate movers by subsector."""
    
    subsector_data = {}
    
    for subsector in movers_df['subsector'].unique():
        sub_df = movers_df[movers_df['subsector'] == subsector]
        
        tickers = sub_df['ticker'].tolist()
        
        subsector_data[subsector] = {
            'avg_return': float(sub_df['return_1w'].mean()),
            'median_return': float(sub_df['return_1w'].median()),
            'count': len(sub_df),
            'tickers': tickers,
            'min_return': float(sub_df['return_1w'].min()),
            'max_return': float(sub_df['return_1w'].max())
        }
    
    return subsector_data


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def create_heatmap(movers_df: pd.DataFrame) -> go.Figure:
    """Create interactive subsector x rating heatmap."""
    
    # Normalize ratings
    def normalize_rating(r):
        if pd.isna(r) or str(r).upper() in ['NR', 'UNKNOWN', '']:
            return 'WR'
        return str(r).upper()
    
    movers_df['rating_norm'] = movers_df['rating'].apply(normalize_rating)
    
    # Aggregate
    pivot = movers_df.pivot_table(
        index='subsector',
        columns='rating_norm',
        values='return_1w',
        aggfunc='mean'
    )
    
    # Reorder columns
    available_ratings = [r for r in RATING_ORDER if r in pivot.columns]
    pivot = pivot[[r for r in available_ratings]]
    
    # Prepare hover text
    hover_text = []
    for subsector in pivot.index:
        hover_row = []
        for rating in pivot.columns:
            mask = (movers_df['subsector'] == subsector) & (movers_df['rating_norm'] == rating)
            bucket_tickers = movers_df[mask]
            
            if len(bucket_tickers) > 0:
                avg_return = pivot.loc[subsector, rating]
                ticker_list = [f"{row['ticker']}: {row['return_1w']:+.1f}%" 
                              for _, row in bucket_tickers.iterrows()]
                
                hover_info = f"<b>{subsector} {rating}</b><br>"
                hover_info += f"Avg Return: {avg_return:+.1f}%<br>"
                hover_info += f"Count: {len(bucket_tickers)}<br><br>"
                hover_info += "<b>Tickers:</b><br>"
                hover_info += "<br>".join(ticker_list[:10])
                if len(ticker_list) > 10:
                    hover_info += f"<br>... and {len(ticker_list) - 10} more"
            else:
                hover_info = f"<b>{subsector} {rating}</b><br>No data"
            
            hover_row.append(hover_info)
        hover_text.append(hover_row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        hovertext=hover_text,
        hoverinfo='text',
        colorbar=dict(title="Return (%)")
    ))
    
    fig.update_layout(
        title=dict(
            text='Healthcare Price Movers (1W)',
            font=dict(size=18, color='#1e293b')
        ),
        xaxis=dict(
            title="Rating",
            side='bottom',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title="Subsector",
            tickfont=dict(size=11)
        ),
        height=500,
        margin=dict(l=100, r=100, t=80, b=80)
    )
    
    return fig


def create_subsector_bar_chart(subsector_data: Dict) -> go.Figure:
    """Create bar chart of subsector returns."""
    
    subsectors = list(subsector_data.keys())
    returns = [subsector_data[s]['avg_return'] for s in subsectors]
    colors = [SUBSECTOR_COLORS.get(s, '#64748b') for s in subsectors]
    
    # Sort by return
    sorted_pairs = sorted(zip(subsectors, returns, colors), key=lambda x: x[1], reverse=True)
    subsectors, returns, colors = zip(*sorted_pairs)
    
    fig = go.Figure(data=[
        go.Bar(
            x=subsectors,
            y=returns,
            marker_color=colors,
            text=[f"{r:+.1f}%" for r in returns],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Subsector Performance (1W)",
        xaxis_title="Subsector",
        yaxis_title="Average Return (%)",
        height=400,
        showlegend=False,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#cbd5e1')
    )
    
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    movers_df: pd.DataFrame,
    subsector_data: Dict,
    subsector_explanations: Dict,
    narrative: str,
    date_str: str
) -> bytes:
    """Generate PDF report."""
    
    if not REPORTLAB_AVAILABLE:
        st.error("ReportLab not available. Cannot generate PDF.")
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Price Attribution Report", title_style))
    story.append(Paragraph(f"Period: {date_str}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(narrative, styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Overall Statistics
    all_returns = [data['avg_return'] for data in subsector_data.values()]
    overall_avg = np.mean(all_returns)
    
    stats_data = [
        ['Metric', 'Value'],
        ['Average Return', f"{overall_avg:+.2f}%"],
        ['Number of Subsectors', str(len(subsector_data))],
        ['Total Names', str(len(movers_df))]
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Subsector Analysis
    story.append(Paragraph("Subsector Analysis", heading_style))
    
    for subsector in sorted(subsector_data.keys(), key=lambda s: subsector_data[s]['avg_return'], reverse=True):
        data = subsector_data[subsector]
        
        story.append(Paragraph(f"<b>{subsector}</b>", styles['Heading3']))
        
        subsector_stats = [
            ['Average Return', f"{data['avg_return']:+.2f}%"],
            ['Median Return', f"{data['median_return']:+.2f}%"],
            ['Number of Names', str(data['count'])],
            ['Range', f"{data['min_return']:+.2f}% to {data['max_return']:+.2f}%"]
        ]
        
        sub_table = Table(subsector_stats, colWidths=[2*inch, 2*inch])
        sub_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(sub_table)
        story.append(Spacer(1, 0.1*inch))
        
        # Key tickers
        story.append(Paragraph(f"<i>Key tickers: {', '.join(data['tickers'][:8])}</i>", styles['Normal']))
        
        # Explanation
        if subsector in subsector_explanations:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("<b>Market Context:</b>", styles['Normal']))
            story.append(Paragraph(subsector_explanations[subsector], styles['BodyText']))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'subsector_data' not in st.session_state:
    st.session_state.subsector_data = None
if 'subsector_explanations' not in st.session_state:
    st.session_state.subsector_explanations = None
if 'narrative' not in st.session_state:
    st.session_state.narrative = None
if 'date_str' not in st.session_state:
    st.session_state.date_str = None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1e40af/ffffff?text=Price+Attribution", use_container_width=True)
        st.title("Controls")
        
        # CSV Upload
        st.subheader("1. Upload Data")
        uploaded_file = st.file_uploader(
            "Upload movers.csv",
            type=['csv'],
            help="Upload a CSV file containing price mover data"
        )
        
        if uploaded_file is not None:
            # Check filename
            if uploaded_file.name != "movers.csv":
                st.markdown(
                    '<div class="warning-box"><b>Warning:</b> Expected filename is "movers.csv"</div>',
                    unsafe_allow_html=True
                )
            
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate
                is_valid, message = validate_csv(df)
                
                if is_valid:
                    st.session_state.uploaded_df = df
                    
                    st.markdown(
                        f'<div class="success-box"><b>Loaded successfully!</b><br>'
                        f'Rows: {len(df):,}<br>'
                        f'Columns: {len(df.columns)}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Extract date
                    if 'date' in df.columns:
                        st.session_state.date_str = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                    else:
                        st.session_state.date_str = datetime.now().strftime('%Y-%m-%d')
                    
                else:
                    st.markdown(
                        f'<div class="error-box"><b>Validation Failed</b><br>{message}</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.uploaded_df = None
                    
            except Exception as e:
                st.markdown(
                    f'<div class="error-box"><b>Error loading file</b><br>{str(e)}</div>',
                    unsafe_allow_html=True
                )
                st.session_state.uploaded_df = None
        
        st.divider()
        
        # Run Analysis
        st.subheader("2. Run Analysis")
        
        # API Configuration
        with st.expander("API Settings", expanded=False):
            provider = st.selectbox(
                "LLM Provider",
                ["anthropic", "openai"],
                help="Select which LLM provider to use for web search"
            )
            
            api_key = st.text_input(
                "API Key (optional)",
                type="password",
                help="Leave empty to use environment variable"
            )
            
            min_move = st.slider(
                "Min Movement (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="Minimum price movement to trigger web search"
            )
        
        run_button = st.button(
            "Run Attribution Analysis",
            type="primary",
            disabled=(st.session_state.uploaded_df is None),
            use_container_width=True
        )
        
        if run_button and st.session_state.uploaded_df is not None:
            with st.spinner("Running attribution analysis..."):
                try:
                    # Aggregate data
                    st.session_state.subsector_data = aggregate_by_subsector(st.session_state.uploaded_df)
                    
                    # Web search
                    searcher = WebSearchEngine(provider, api_key)
                    st.session_state.subsector_explanations = searcher.search_all_subsectors(
                        st.session_state.subsector_data,
                        st.session_state.date_str,
                        min_move
                    )
                    
                    # Generate narrative
                    st.session_state.narrative = generate_narrative(
                        st.session_state.subsector_data,
                        st.session_state.subsector_explanations,
                        provider,
                        api_key,
                        st.session_state.date_str
                    )
                    
                    st.session_state.analysis_complete = True
                    st.success(" Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state.analysis_complete = False
        
        st.divider()
        
        # PDF Download
        st.subheader("3. Export")
        
        if st.session_state.analysis_complete:
            pdf_data = generate_pdf_report(
                st.session_state.uploaded_df,
                st.session_state.subsector_data,
                st.session_state.subsector_explanations,
                st.session_state.narrative,
                st.session_state.date_str
            )
            
            if pdf_data:
                st.download_button(
                    label=" Download PDF Report",
                    data=pdf_data,
                    file_name=f"price_attribution_{st.session_state.date_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button(
                " Download PDF Report",
                disabled=True,
                use_container_width=True,
                help="Run analysis first to enable PDF download"
            )
    
    # Main content area
    tab1, tab2 = st.tabs([" Price Attribution Analysis", "📖 Instructions"])
    
    with tab1:
        if st.session_state.analysis_complete:
            # Header
            st.markdown('<div class="main-header">Price Attribution Analysis</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sub-header">Period ending: {st.session_state.date_str}</div>', unsafe_allow_html=True)
            
            # Executive Summary
            st.markdown("### Executive Summary")
            st.info(st.session_state.narrative)
            
            # Overall Metrics
            st.markdown("### Overall Performance")
            
            all_returns = [data['avg_return'] for data in st.session_state.subsector_data.values()]
            overall_avg = np.mean(all_returns)
            overall_median = np.median(all_returns)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Return", f"{overall_avg:+.2f}%")
            with col2:
                st.metric("Median Return", f"{overall_median:+.2f}%")
            with col3:
                st.metric("Subsectors", len(st.session_state.subsector_data))
            with col4:
                st.metric("Total Names", len(st.session_state.uploaded_df))
            
            # Visualizations
            st.markdown("### Performance Heatmap")
            heatmap_fig = create_heatmap(st.session_state.uploaded_df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.markdown("### Subsector Performance")
            bar_fig = create_subsector_bar_chart(st.session_state.subsector_data)
            st.plotly_chart(bar_fig, use_container_width=True)
            
            # Detailed Subsector Analysis
            st.markdown("### Detailed Subsector Analysis")
            
            for subsector in sorted(st.session_state.subsector_data.keys(), 
                                   key=lambda s: st.session_state.subsector_data[s]['avg_return'], 
                                   reverse=True):
                data = st.session_state.subsector_data[subsector]
                color = SUBSECTOR_COLORS.get(subsector, '#64748b')
                
                with st.container():
                    st.markdown(f"#### {subsector}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Return", f"{data['avg_return']:+.2f}%")
                    with col2:
                        st.metric("Median Return", f"{data['median_return']:+.2f}%")
                    with col3:
                        st.metric("Count", data['count'])
                    with col4:
                        st.metric("Range", f"{data['max_return']-data['min_return']:.2f}%")
                    
                    st.caption(f"**Key tickers:** {', '.join(data['tickers'][:10])}")
                    
                    if subsector in st.session_state.subsector_explanations:
                        st.markdown(
                            f'<div class="explanation-box">'
                            f'<b> Market Context:</b><br>{st.session_state.subsector_explanations[subsector]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.divider()
        
        else:
            # Welcome screen
            st.markdown('<div class="main-header">Price Attribution Dashboard</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Analyze healthcare sector price movements with AI-powered attribution</div>', unsafe_allow_html=True)
            
            st.markdown("### Get Started")
            st.markdown("""
            1. **Upload** your movers.csv file using the sidebar
            2. **Configure** API settings if needed (optional)
            3. **Run** the attribution analysis
            4. **Review** results and download PDF report
            """)
            
            st.info("Upload a CSV file in the sidebar to begin")
            
            # Show sample data format
            with st.expander(" View Sample Data Format"):
                sample_data = {
                    'ticker': ['COMPANY A', 'COMPANY B', 'COMPANY C'],
                    'subsector': ['Provider', 'Pharma', 'Payors'],
                    'rating': ['Ba2', 'B1', 'B3'],
                    'return_1w': [1.5, -0.8, 2.3],
                    'return_1d': [0.3, -0.1, 0.5],
                    'date': ['2024-11-02', '2024-11-02', '2024-11-02']
                }
                st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
    
    with tab2:
        st.markdown('<div class="main-header"> Instructions</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## What This Tool Does
        
        The **Price Attribution Dashboard** analyzes healthcare sector price movements and provides AI-powered 
        attribution using real-time web search. It aggregates price mover data by subsector, searches the web 
        for market context, and generates a comprehensive analysis report with visualizations.
        
        This tool helps credit analysts quickly understand:
        - Which subsectors moved and by how much
        - Why specific subsectors experienced price changes
        - Market drivers and catalysts behind movements
        - Detailed performance breakdowns by rating and subsector
        
        ---
        
        ## How to Use
        
        ### Step 1: Upload movers.csv
        
        1. Click the **"Browse files"** button in the sidebar
        2. Select your CSV file (preferably named `movers.csv`)
        3. Wait for validation to complete
        4. If successful, you'll see a green success message with row/column counts
        
        ### Step 2: Configure Settings (Optional)
        
        Expand **" API Settings"** to customize:
        
        - **LLM Provider**: Choose between Anthropic (Claude) or OpenAI (GPT-4)
        - **API Key**: Provide your API key or use environment variable
        - **Min Movement**: Set threshold for web search (default: 0.5%)
        
        ### Step 3: Run Analysis
        
        1. Click **" Run Attribution Analysis"** button
        2. The tool will:
           - Aggregate data by subsector
           - Search the web for market context
           - Generate AI-powered narrative
           - Create interactive visualizations
        3. Progress indicators will show current status
        
        ### Step 4: Review Results
        
        The analysis includes:
        
        - **Executive Summary**: AI-generated overview of market movements
        - **Overall Metrics**: Sector-wide performance statistics
        - **Interactive Heatmap**: Subsector × Rating performance grid
        - **Bar Chart**: Subsector performance comparison
        - **Detailed Analysis**: Deep dive into each subsector with web context
        
        ### Step 5: Download PDF
        
        1. Click **"📥 Download PDF Report"** in the sidebar
        2. Save the comprehensive PDF report for offline viewing or sharing
        
        ---
        
        ## CSV Formatting Requirements
        
        Your CSV file **must** include these columns:
        
        | Column | Type | Description | Example |
        |--------|------|-------------|---------|
        | `ticker` | string | Company ticker or name | `DAVITA INC` |
        | `subsector` | string | Healthcare subsector classification | `Provider`, `Pharma`, `Payors`, `Tech`, `Devices` |
        | `rating` | string | Credit rating | `Ba2`, `B1`, `Caa1`, `WR` |
        | `return_1w` | float | 1-week return (decimal) | `0.0129` (for 1.29%) |
        | `return_1d` | float | 1-day return (decimal) | `-0.0045` (for -0.45%) |
        | `date` | string | Date of observation | `2024-11-02` or `11-02-2024` |
        
        ### Important Notes:
        
        - **No missing values** allowed in `ticker`, `subsector`, or `return_1w`
        - **Returns should be decimals**, not percentages (0.01 = 1%)
        - **Date format** can be YYYY-MM-DD or MM-DD-YYYY
        - **Rating** can include standard Moody's ratings or `WR` (withdrawn)
        - File should have **header row** with exact column names
        
        ### Example CSV:
        
        ```csv
        ticker,subsector,rating,return_1w,return_1d,date
        DAVITA INC,Provider,Ba2,0.9067,-0.0322,11-02-2024
        BIOMARIN PHARMACEUTICAL,Pharma,Ba2,0.6862,0.0000,11-02-2024
        SEDGWICK CMS INC,Payors,B2,-1.4640,-0.1273,11-02-2024
        ```
        
        ---
        
        ## Troubleshooting
        
        **CSV Upload Failed**
        - Verify all required columns are present
        - Check for typos in column names (case-sensitive)
        - Ensure no completely empty rows
        
        **Analysis Failed**
        - Confirm API key is valid (if not using environment variable)
        - Check internet connection for web search
        - Verify subsectors have sufficient data (at least 1 ticker)
        
        **PDF Download Not Available**
        - Make sure analysis has been run successfully
        - Check that ReportLab is installed
        - Try running the analysis again
        
        ---
        
        ## API Requirements
        
        This tool requires an API key for either:
        
        - **Anthropic (Claude)**: Set `ANTHROPIC_API_KEY` environment variable or provide in settings
        - **OpenAI (GPT-4)**: Set `OPENAI_API_KEY` environment variable or provide in settings
        
        The LLM is used for:
        1. Web search to find market context
        2. Generating subsector explanations
        3. Creating executive summary narrative
        
        ---
        
        ## Support
        
        For questions or issues:
        - Check CSV formatting matches requirements exactly
        - Ensure API credentials are configured correctly
        - Review error messages in the sidebar for specific issues
        """)


if __name__ == "__main__":
    main()
