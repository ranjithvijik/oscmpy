"""
OSCM Simulator - Complete Streamlit Application
Operations and Supply Chain Management Simulator
Based on Jacobs & Chase (2024). Operations and Supply Chain Management, 17th ed. McGraw-Hill.

All 40 modules with theory, calculators, and practice problems.
Organized by chapter order (Ch 1 → Ch 22).
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import factorial
import math

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="OSCM Simulator v3.5",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME MANAGEMENT
# ============================================================
def init_theme():
    """Initialize theme in session state."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

def toggle_theme():
    """Toggle between light and dark themes."""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

init_theme()

# ============================================================
# DYNAMIC CSS BASED ON THEME
# ============================================================
def get_theme_css():
    """Generate CSS based on current theme."""
    is_dark = st.session_state.theme == "dark"
    
    if is_dark:
        # Dark Mode Colors
        bg_primary = "#0f172a"
        bg_secondary = "#1e293b"
        bg_card = "#1e293b"
        bg_input = "#334155"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        border_color = "#334155"
        accent_primary = "#818cf8"
        accent_secondary = "#a78bfa"
        success_bg = "#064e3b"
        success_text = "#6ee7b7"
        success_border = "#10b981"
        warning_bg = "#78350f"
        warning_text = "#fcd34d"
        warning_border = "#f59e0b"
        danger_bg = "#7f1d1d"
        danger_text = "#fca5a5"
        danger_border = "#ef4444"
        info_bg = "#1e3a5f"
        info_text = "#93c5fd"
        info_border = "#3b82f6"
        citation_bg = "#422006"
        citation_text = "#fef3c7"
        citation_border = "#d97706"
        equation_bg = "#1e3a5f"
        equation_border = "#60a5fa"
        equation_text = "#e0f2fe"
        theory_bg = "#1e293b"
        theory_border = "#818cf8"
        insight_bg = "#064e3b"
        insight_border = "#34d399"
        insight_title = "#6ee7b7"
        practice_bg = "#1e3a5f"
        practice_border = "#60a5fa"
        metric_highlight_bg = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)"
        metric_highlight_text = "#ffffff"
        metric_normal_bg = "#334155"
        metric_normal_text = "#f1f5f9"
        table_header_bg = "#334155"
        table_row_alt = "#1e293b"
        link_color = "#93c5fd"
        code_bg = "#0f172a"
    else:
        # Light Mode Colors
        bg_primary = "#ffffff"
        bg_secondary = "#f8fafc"
        bg_card = "#ffffff"
        bg_input = "#ffffff"
        text_primary = "#1e293b"
        text_secondary = "#475569"
        text_muted = "#94a3b8"
        border_color = "#e2e8f0"
        accent_primary = "#6366f1"
        accent_secondary = "#8b5cf6"
        success_bg = "#f0fdf4"
        success_text = "#166534"
        success_border = "#86efac"
        warning_bg = "#fffbeb"
        warning_text = "#92400e"
        warning_border = "#fcd34d"
        danger_bg = "#fef2f2"
        danger_text = "#991b1b"
        danger_border = "#fca5a5"
        info_bg = "#eff6ff"
        info_text = "#1e40af"
        info_border = "#93c5fd"
        citation_bg = "#fefce8"
        citation_text = "#854d0e"
        citation_border = "#eab308"
        equation_bg = "#f0f9ff"
        equation_border = "#bae6fd"
        equation_text = "#0c4a6e"
        theory_bg = "#f8fafc"
        theory_border = "#6366f1"
        insight_bg = "#ecfdf5"
        insight_border = "#6ee7b7"
        insight_title = "#047857"
        practice_bg = "#eff6ff"
        practice_border = "#93c5fd"
        metric_highlight_bg = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)"
        metric_highlight_text = "#ffffff"
        metric_normal_bg = "#f1f5f9"
        metric_normal_text = "#1e293b"
        table_header_bg = "#f1f5f9"
        table_row_alt = "#f8fafc"
        link_color = "#2563eb"
        code_bg = "#f1f5f9"
    
    return f"""
    <style>
        /* Global Styles */
        .stApp {{
            background-color: {bg_primary};
            color: {text_primary};
        }}
        
        /* Main Header */
        .main-header {{
            background: {metric_highlight_bg};
            padding: 1.5rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .main-header h1 {{
            margin: 0;
            font-size: 1.8rem;
            font-weight: 700;
            color: white !important;
        }}
        .main-header p {{
            margin: 0.5rem 0 0 0;
            opacity: 0.95;
            color: white !important;
        }}
        .chapter-badge {{
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: {metric_normal_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            margin: 0.5rem 0;
        }}
        .metric-card.highlight {{
            background: {metric_highlight_bg};
            color: {metric_highlight_text};
            border: none;
        }}
        .metric-card.success {{
            background: {success_bg};
            border-color: {success_border};
        }}
        .metric-card.danger {{
            background: {danger_bg};
            border-color: {danger_border};
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            color: {text_primary};
            line-height: 1.2;
        }}
        .metric-card.highlight .metric-value {{
            color: white;
        }}
        .metric-card.success .metric-value {{
            color: {success_text};
        }}
        .metric-card.danger .metric-value {{
            color: {danger_text};
        }}
        .metric-label {{
            font-size: 0.85rem;
            color: {text_secondary};
            margin-top: 0.4rem;
        }}
        .metric-card.highlight .metric-label {{
            color: rgba(255,255,255,0.9);
        }}
        
        /* Theory Box */
        .theory-box {{
            background: {theory_bg};
            border-left: 4px solid {theory_border};
            padding: 1.2rem 1.5rem;
            border-radius: 0 12px 12px 0;
            margin: 1rem 0;
            color: {text_primary};
        }}
        .theory-box h3 {{
            color: {accent_primary};
            margin-top: 0;
        }}
        
        /* Citation Box */
        .citation-box {{
            background: {citation_bg};
            border-left: 4px solid {citation_border};
            padding: 1.2rem 1.5rem;
            border-radius: 0 12px 12px 0;
            margin: 1rem 0;
            font-style: italic;
            color: {citation_text};
        }}
        .citation-source {{
            display: block;
            margin-top: 0.75rem;
            font-style: normal;
            font-weight: 600;
            color: {citation_text};
        }}
        
        /* Equation Box */
        .equation-box {{
            background: {equation_bg};
            border: 1px solid {equation_border};
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1rem 0;
            text-align: center;
        }}
        .equation-label {{
            font-size: 0.85rem;
            font-weight: 600;
            color: {accent_primary};
            margin-bottom: 0.5rem;
        }}
        .equation-description {{
            font-size: 0.85rem;
            color: {text_secondary};
            margin-top: 0.75rem;
            text-align: left;
        }}
        
        /* Key Insight Box */
        .key-insight {{
            background: {insight_bg};
            border: 1px solid {insight_border};
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1rem 0;
        }}
        .key-insight-title {{
            font-weight: 700;
            color: {insight_title};
            margin-bottom: 0.5rem;
            font-size: 1rem;
        }}
        .key-insight-text {{
            color: {text_primary};
        }}
        
        /* Practice Box */
        .practice-box {{
            background: {practice_bg};
            border: 1px solid {practice_border};
            border-radius: 12px;
            padding: 1.2rem;
            margin: 1rem 0;
        }}
        .practice-badge {{
            background: {accent_primary};
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .practice-question {{
            color: {text_primary};
            font-weight: 500;
            margin: 0.75rem 0;
        }}
        
        /* Alert Boxes */
        .alert {{
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin: 0.75rem 0;
        }}
        .alert-success {{
            background: {success_bg};
            border: 1px solid {success_border};
            color: {success_text};
        }}
        .alert-warning {{
            background: {warning_bg};
            border: 1px solid {warning_border};
            color: {warning_text};
        }}
        .alert-danger {{
            background: {danger_bg};
            border: 1px solid {danger_border};
            color: {danger_text};
        }}
        .alert-info {{
            background: {info_bg};
            border: 1px solid {info_border};
            color: {info_text};
        }}
        
        /* Tables */
        .styled-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        .styled-table th {{
            background: {table_header_bg};
            color: {text_primary};
            padding: 0.75rem;
            text-align: left;
            border-bottom: 2px solid {border_color};
        }}
        .styled-table td {{
            padding: 0.75rem;
            border-bottom: 1px solid {border_color};
            color: {text_primary};
        }}
        .styled-table tr:nth-child(even) {{
            background: {table_row_alt};
        }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {bg_secondary};
        }}
        section[data-testid="stSidebar"] .stMarkdown {{
            color: {text_primary};
        }}
        
        /* Theme Toggle Button */
        .theme-toggle {{
            position: fixed;
            top: 0.75rem;
            right: 1rem;
            z-index: 9999;
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 50%;
            width: 45px;
            height: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 1.3rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        }}
        .theme-toggle:hover {{
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        /* Concept Cards */
        .concept-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            transition: all 0.2s ease;
        }}
        .concept-card:hover {{
            border-color: {accent_primary};
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
        }}
        .concept-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        .concept-title {{
            font-weight: 600;
            color: {text_primary};
            margin-bottom: 0.3rem;
        }}
        .concept-desc {{
            font-size: 0.85rem;
            color: {text_secondary};
        }}
        
        /* Step-by-Step Solution */
        .solution-step {{
            background: {bg_secondary};
            border-left: 3px solid {accent_primary};
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }}
        .step-number {{
            background: {accent_primary};
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}
        
        /* Formula Reference Card */
        .formula-card {{
            background: {equation_bg};
            border: 1px solid {equation_border};
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        .formula-title {{
            font-weight: 600;
            color: {accent_primary};
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background: {bg_secondary};
            color: {text_primary};
            border-radius: 8px;
        }}
        
        /* Input fields */
        .stTextInput input, .stNumberInput input, .stSelectbox select {{
            background: {bg_input};
            color: {text_primary};
            border-color: {border_color};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {bg_secondary};
            color: {text_primary};
            border-radius: 8px 8px 0 0;
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent_primary};
            color: white;
        }}
        
        /* Code blocks */
        code {{
            background: {code_bg};
            color: {text_primary};
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }}
        
        /* Links */
        a {{
            color: {link_color};
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: {bg_secondary};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {border_color};
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {text_muted};
        }}
    </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_header(icon, chapter, title, subtitle):
    """Display module header with consistent styling."""
    st.markdown(f"""
    <div class="main-header">
        <h1>{icon} {title}</h1>
        <p><span class="chapter-badge">{chapter}</span> {subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def display_theory(content):
    """Display theory content in styled box."""
    st.markdown(f'<div class="theory-box">{content}</div>', unsafe_allow_html=True)

def display_citation(quote, source):
    """Display citation with source."""
    st.markdown(f"""
    <div class="citation-box">
        "{quote}"
        <span class="citation-source">— {source}</span>
    </div>
    """, unsafe_allow_html=True)

def display_key_insight(title, content):
    """Display key insight box."""
    st.markdown(f"""
    <div class="key-insight">
        <div class="key-insight-title">💡 {title}</div>
        <div class="key-insight-text">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_equation(label, latex_eq, description=""):
    """Display equation in styled box."""
    st.markdown(f"""
    <div class="equation-box">
        <div class="equation-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(latex_eq)
    if description:
        st.markdown(f"<p style='font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem;'>{description}</p>", unsafe_allow_html=True)

def display_metric_card(value, label, card_type="normal"):
    """Display a metric card."""
    css_class = f"metric-card {card_type}" if card_type != "normal" else "metric-card"
    st.markdown(f"""
    <div class="{css_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def display_concept_card(icon, title, description):
    """Display a concept card."""
    st.markdown(f"""
    <div class="concept-card">
        <div class="concept-icon">{icon}</div>
        <div class="concept-title">{title}</div>
        <div class="concept-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def display_solution_step(step_num, content):
    """Display a solution step."""
    st.markdown(f"""
    <div class="solution-step">
        <span class="step-number">{step_num}</span>
        {content}
    </div>
    """, unsafe_allow_html=True)

def display_alert(content, alert_type="info"):
    """Display an alert box."""
    st.markdown(f'<div class="alert alert-{alert_type}">{content}</div>', unsafe_allow_html=True)

def check_answer(user_answer, correct_answer, tolerance=0.05):
    """Check if user answer is within tolerance of correct answer."""
    if user_answer is None or correct_answer is None:
        return False
    try:
        return abs(float(user_answer) - float(correct_answer)) <= abs(float(correct_answer) * tolerance) + 0.01
    except:
        return False

def normal_cdf(z):
    """Standard normal cumulative distribution function."""
    return stats.norm.cdf(z)

def normal_ppf(p):
    """Standard normal percent point function (inverse CDF)."""
    return stats.norm.ppf(p)

def poisson_pmf(k, lam):
    """Poisson probability mass function."""
    return stats.poisson.pmf(k, lam)

def format_currency(value):
    """Format number as currency."""
    return f"${value:,.2f}"

def format_number(value, decimals=2):
    """Format number with specified decimals."""
    return f"{value:,.{decimals}f}"

# ============================================================
# THEME TOGGLE IN SIDEBAR
# ============================================================
def render_theme_toggle():
    """Render theme toggle button."""
    theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
    theme_text = "Dark Mode" if st.session_state.theme == "light" else "Light Mode"
    
    if st.sidebar.button(f"{theme_icon} {theme_text}", key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()

# ============================================================
# Z-TABLE DATA
# ============================================================
Z_TABLE = {}
for z_int in range(-30, 40):
    z_base = z_int / 10
    for z_dec in range(10):
        z = z_base + z_dec / 100
        Z_TABLE[round(z, 2)] = round(normal_cdf(z), 4)

# ============================================================
# MODULE 1: SUPPLY CHAIN RISK (Chapter 1)
# ============================================================
def module_risk():
    display_header("🛡️", "Chapter 1", "Supply Chain Risk Assessment", 
                   "Probability and Impact Matrix (Exhibit 1.4)")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### Supply Chain Risk Management")
        
        st.write("""
        **Supply chain risk** is the likelihood of a disruption that would impact the ability of a 
        company to continuously supply products or services. Effective risk management involves 
        systematic identification, assessment, and mitigation of potential threats.
        """)
        
        display_citation(
            "Supply chain risk management involves the identification of potential sources of risk "
            "and implementation of appropriate strategies through a coordinated approach among "
            "supply chain members to reduce supply chain vulnerability.",
            "Jacobs & Chase (2024, p. 12)"
        )
        
        st.markdown("#### Risk Assessment Framework")
        st.latex(r"\text{Risk Score} = \text{Probability} \times \text{Impact}")
        
        st.write("""
        Each risk event is scored on a scale (typically 1-5 or 1-10). Events with high scores 
        require immediate mitigation strategies such as:
        - **Redundancy** - Multiple suppliers, backup facilities
        - **Insurance** - Financial protection against losses
        - **Process Changes** - Redesigning vulnerable processes
        - **Inventory Buffers** - Safety stock for critical items
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("⚠️", "Risk Identification", 
                "Systematically identify all potential sources of supply chain disruption")
        with col2:
            display_concept_card("📊", "Risk Assessment", 
                "Evaluate probability and impact of each identified risk")
        
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("🛡️", "Risk Mitigation", 
                "Develop and implement strategies to reduce risk exposure")
        with col2:
            display_concept_card("📈", "Risk Monitoring", 
                "Continuously track risk indicators and update assessments")
        
        display_key_insight(
            "What-If Analysis",
            "Some companies call this 'what if' analysis. Answering these 'what if' questions can be "
            "useful for understanding how sensitive an analysis is to cost and profit assumptions. "
            "Consider scenarios like: 25% increase in development time, 25% change in sales volume, "
            "$1 change in price or cost. (Jacobs & Chase, 2024, p. 60)"
        )
        
        st.markdown("#### Common Supply Chain Risk Categories")
        
        risk_categories = pd.DataFrame({
            "Category": ["Operational", "Financial", "Strategic", "Hazard", "Demand", "Supply"],
            "Examples": [
                "Equipment failure, quality issues, capacity constraints",
                "Currency fluctuation, supplier bankruptcy, credit risk",
                "Competitor actions, market changes, technology shifts",
                "Natural disasters, accidents, terrorism",
                "Forecast errors, demand volatility, bullwhip effect",
                "Supplier failure, logistics disruption, material shortage"
            ],
            "Mitigation": [
                "Preventive maintenance, quality systems, flexible capacity",
                "Hedging, supplier financial monitoring, diversification",
                "Market intelligence, scenario planning, agility",
                "Insurance, business continuity planning, geographic spread",
                "Demand sensing, collaborative forecasting, postponement",
                "Multi-sourcing, safety stock, supplier development"
            ]
        })
        st.dataframe(risk_categories, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Risk Assessment Matrix Calculator")
        st.write("Score each risk event from 1 (Low) to 5 (High) for both Probability and Impact")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Risk Event Scoring")
            
            risk_names = [
                "Supplier Failure (Financial)",
                "Natural Disaster / Weather",
                "Quality Issue / Product Recall",
                "Logistics / Customs Delay",
                "Demand Volatility",
                "Cybersecurity Breach",
                "Regulatory Changes",
                "Key Personnel Loss"
            ]
            
            risks = []
            for i, name in enumerate(risk_names):
                with st.expander(f"📌 {name}", expanded=(i < 3)):
                    c1, c2 = st.columns(2)
                    with c1:
                        prob = st.slider(f"Probability", 1, 5, 3, key=f"risk_p_{i}",
                                        help="1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain")
                    with c2:
                        impact = st.slider(f"Impact", 1, 5, 4, key=f"risk_i_{i}",
                                          help="1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Catastrophic")
                    risks.append({"name": name, "prob": prob, "impact": impact, "score": prob * impact})
        
        with col2:
            st.markdown("#### Risk Analysis Results")
            
            df = pd.DataFrame(risks)
            df.columns = ["Risk Event", "Probability", "Impact", "Risk Score"]
            df = df.sort_values("Risk Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            total_score = sum(r["score"] for r in risks)
            max_score = len(risks) * 25
            risk_percentage = (total_score / max_score) * 100
            
            col_a, col_b = st.columns(2)
            with col_a:
                display_metric_card(f"{total_score}", "Total Risk Score", "highlight")
            with col_b:
                display_metric_card(f"{risk_percentage:.0f}%", "Risk Exposure Level", 
                                   "danger" if risk_percentage > 60 else "success" if risk_percentage < 40 else "normal")
            
            # Risk Priority Classification
            st.markdown("#### Risk Priority Classification")
            high_risks = [r for r in risks if r["score"] >= 15]
            med_risks = [r for r in risks if 8 <= r["score"] < 15]
            low_risks = [r for r in risks if r["score"] < 8]
            
            if high_risks:
                display_alert(f"🔴 <strong>HIGH PRIORITY ({len(high_risks)}):</strong> {', '.join([r['name'] for r in high_risks])}<br><em>Immediate action required</em>", "danger")
            if med_risks:
                display_alert(f"🟡 <strong>MEDIUM PRIORITY ({len(med_risks)}):</strong> {', '.join([r['name'] for r in med_risks])}<br><em>Monitor closely and develop contingency plans</em>", "warning")
            if low_risks:
                display_alert(f"🟢 <strong>LOW PRIORITY ({len(low_risks)}):</strong> {', '.join([r['name'] for r in low_risks])}<br><em>Periodic review sufficient</em>", "success")
    
    with tab3:
        st.markdown("### Practice Problems")
        
        # Problem 1
        with st.expander("📝 Problem 1: Triple Bottom Line", expanded=True):
            st.markdown("""
            **Question:** What is the "Triple Bottom Line" and why is it important for modern supply chain management?
            """)
            
            user_answer_1 = st.text_area("Your Answer:", key="risk_p1_ans", height=100,
                                         placeholder="Enter your answer here...")
            
            if st.button("Check Answer", key="risk_p1_btn"):
                st.markdown("---")
                st.markdown("#### ✅ Model Answer:")
                display_solution_step(1, "<strong>Definition:</strong> The Triple Bottom Line (TBL) evaluates a firm against three criteria:")
                display_solution_step(2, "<strong>Social (People):</strong> Impact on employees, communities, and society - fair labor practices, community engagement, human rights")
                display_solution_step(3, "<strong>Economic (Profit):</strong> Financial performance and long-term economic sustainability - not just short-term profits")
                display_solution_step(4, "<strong>Environmental (Planet):</strong> Ecological footprint and environmental sustainability - carbon emissions, waste reduction, resource conservation")
                
                display_key_insight("Why It Matters",
                    "Modern consumers and investors increasingly demand that companies demonstrate responsibility "
                    "across all three dimensions. Supply chains that ignore social or environmental factors face "
                    "reputational risks, regulatory penalties, and loss of market share.")
        
        # Problem 2
        with st.expander("📝 Problem 2: Efficiency vs. Effectiveness"):
            st.markdown("""
            **Question:** Distinguish between "Efficiency" and "Effectiveness" in operations management. 
            Provide an example where a company might be efficient but not effective.
            """)
            
            if st.button("Show Solution", key="risk_p2_btn"):
                st.markdown("#### ✅ Solution:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Efficiency**")
                    st.write("Doing something at the **lowest possible cost** (doing things right)")
                    st.write("*Focus: Resource utilization*")
                with col2:
                    st.markdown("**Effectiveness**")
                    st.write("Doing the **right things** to create the most value for the customer")
                    st.write("*Focus: Goal achievement*")
                
                display_alert(
                    "<strong>Example:</strong> A factory produces widgets at the lowest cost per unit (efficient), "
                    "but the widgets don't meet customer quality expectations (not effective). The company saves "
                    "money on production but loses customers due to poor quality.",
                    "info"
                )
        
        # Problem 3
        with st.expander("📝 Problem 3: Risk Score Calculation"):
            st.markdown("""
            **Question:** A company identifies the following risks:
            - Supplier bankruptcy: Probability = 2, Impact = 5
            - Equipment failure: Probability = 4, Impact = 3
            - Demand surge: Probability = 3, Impact = 4
            
            Calculate the risk score for each and determine which should be addressed first.
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ans_1 = st.number_input("Supplier bankruptcy score:", key="risk_p3_1")
            with col2:
                ans_2 = st.number_input("Equipment failure score:", key="risk_p3_2")
            with col3:
                ans_3 = st.number_input("Demand surge score:", key="risk_p3_3")
            
            if st.button("Check Answers", key="risk_p3_btn"):
                correct_1, correct_2, correct_3 = 10, 12, 12
                
                results = []
                if check_answer(ans_1, correct_1): results.append("✅ Supplier bankruptcy correct")
                else: results.append(f"❌ Supplier bankruptcy: 2 × 5 = {correct_1}")
                
                if check_answer(ans_2, correct_2): results.append("✅ Equipment failure correct")
                else: results.append(f"❌ Equipment failure: 4 × 3 = {correct_2}")
                
                if check_answer(ans_3, correct_3): results.append("✅ Demand surge correct")
                else: results.append(f"❌ Demand surge: 3 × 4 = {correct_3}")
                
                for r in results:
                    st.write(r)
                
                display_alert(
                    "<strong>Priority:</strong> Equipment failure and Demand surge (both score 12) should be "
                    "addressed first, followed by Supplier bankruptcy (score 10). However, the high impact (5) "
                    "of supplier bankruptcy means it may warrant special attention despite lower probability.",
                    "info"
                )
        
        # Problem 4
        with st.expander("📝 Problem 4: What-If Sensitivity Analysis"):
            st.markdown("""
            **Question:** Based on the textbook's guidance on sensitivity analysis, explain what happens 
            to project profitability if:
            1. Development time increases by 25%
            2. Sales volume decreases by 25%
            3. Product cost increases by $1 per unit
            """)
            
            if st.button("Show Analysis", key="risk_p4_btn"):
                st.markdown("#### ✅ Sensitivity Analysis:")
                
                display_solution_step(1, 
                    "<strong>25% increase in development time:</strong> Delays production ramp-up, marketing efforts, "
                    "and product sales. This pushes revenue further into the future, reducing its present value. "
                    "Also increases development costs and may allow competitors to enter first.")
                
                display_solution_step(2,
                    "<strong>25% decrease in sales volume:</strong> Directly reduces revenue while fixed costs remain "
                    "constant. This can turn a profitable project into a loss. The impact is magnified by operating "
                    "leverage (high fixed costs relative to variable costs).")
                
                display_solution_step(3,
                    "<strong>$1 increase in product cost:</strong> Reduces profit by $1 per unit sold. For high-volume "
                    "products, this can significantly impact total profitability. Consider: 100,000 units × $1 = $100,000 "
                    "reduction in profit.")
                
                display_citation(
                    "A dollar spent or saved on development cost is worth the present value of that dollar to the "
                    "value of the project.",
                    "Jacobs & Chase (2024, p. 60)"
                )

# ============================================================
# MODULE 2: PERT NETWORK (Chapter 4)
# ============================================================
def module_pert():
    display_header("🔗", "Chapter 4", "PERT Network Diagram & Completion Probability", 
                   "Critical path identification, slack calculation & Z-score probability (Exhibits 4.8–4.9)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Activity Estimator", "📊 Probability Calculator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### PERT Network Analysis")
        
        st.write("""
        **PERT (Program Evaluation and Review Technique)** is a project management tool that uses 
        probabilistic time estimates to account for uncertainty in activity durations. It was developed 
        by the U.S. Navy in 1958 for the Polaris missile project.
        """)
        
        display_citation(
            "A conservative approach dictates using the critical path with the largest total variance "
            "to focus management's attention on the activities most likely to exhibit broad variations.",
            "Jacobs & Chase (2024, p. 99)"
        )
        
        st.markdown("#### PERT Time Estimates")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🟢", "Optimistic (a)", 
                "Best-case scenario - everything goes perfectly. Probability ≈ 1%")
        with col2:
            display_concept_card("🔵", "Most Likely (m)", 
                "Normal conditions - most frequent outcome if repeated many times")
        with col3:
            display_concept_card("🔴", "Pessimistic (b)", 
                "Worst-case scenario - everything goes wrong. Probability ≈ 1%")
        
        st.markdown("#### Key Formulas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Expected Time (Tₑ)")
            st.latex(r"T_E = \frac{a + 4m + b}{6}")
            st.write("Weighted average giving 4× weight to most likely estimate (based on Beta distribution)")
        
        with col2:
            st.markdown("##### Variance (σ²)")
            st.latex(r"\sigma^2 = \left(\frac{b - a}{6}\right)^2")
            st.write("Measures uncertainty - larger spread between a and b means higher variance")
        
        st.markdown("##### Standard Deviation (σ)")
        st.latex(r"\sigma = \frac{b - a}{6}")
        
        st.markdown("#### Project Completion Probability")
        st.latex(r"Z = \frac{D - T_E}{\sqrt{\sum \sigma^2_{cp}}}")
        
        st.write("""
        Where:
        - **D** = Desired (target) completion time
        - **Tₑ** = Expected project duration (sum of critical path expected times)
        - **Σσ²cp** = Sum of variances on the critical path
        - **Z** = Standard normal deviate (look up in Z-table for probability)
        """)
        
        display_key_insight(
            "Critical Path Selection with Multiple Critical Paths",
            "When there are two or more critical paths of equal length, use the one with the "
            "<strong>largest total variance</strong> for probability calculations. This conservative "
            "approach focuses attention on activities most likely to cause schedule problems."
        )
        
        st.markdown("#### Example from Textbook (Exhibit 4.9)")
        
        example_data = pd.DataFrame({
            "Path": ["A-B-E-G-I", "A-C-F-G-I", "A-D-F-G-I", "A-D-H-I"],
            "Length (days)": [22, 17, 21, 21],
            "Status": ["CRITICAL PATH", "Slack = 5", "Slack = 1", "Slack = 1"]
        })
        st.dataframe(example_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Activity Time Estimator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Estimates")
            a = st.slider("Optimistic Time (a)", 1, 20, 4, help="Best-case duration")
            m = st.slider("Most Likely Time (m)", 1, 30, 8, help="Most probable duration")
            b = st.slider("Pessimistic Time (b)", 1, 40, 16, help="Worst-case duration")
            
            if not (a <= m <= b):
                display_alert("⚠️ PERT estimates should satisfy a ≤ m ≤ b", "warning")
            else:
                display_alert("✅ Valid PERT estimates", "success")
        
        with col2:
            te = (a + 4*m + b) / 6
            variance = ((b - a) / 6) ** 2
            std_dev = math.sqrt(variance)
            
            st.markdown("#### Calculated Results")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                display_metric_card(f"{te:.2f}", "Expected Time (Tₑ)", "highlight")
            with col_b:
                display_metric_card(f"{variance:.2f}", "Variance (σ²)", "normal")
            with col_c:
                display_metric_card(f"{std_dev:.2f}", "Std Dev (σ)", "normal")
            
            st.markdown("#### Step-by-Step Calculation")
            display_solution_step(1, f"Expected Time: Tₑ = ({a} + 4×{m} + {b}) / 6 = {a + 4*m + b} / 6 = <strong>{te:.2f}</strong>")
            display_solution_step(2, f"Variance: σ² = (({b} - {a}) / 6)² = ({b-a} / 6)² = ({(b-a)/6:.2f})² = <strong>{variance:.2f}</strong>")
            display_solution_step(3, f"Std Dev: σ = √{variance:.2f} = <strong>{std_dev:.2f}</strong>")
            
            st.markdown("#### Probability Ranges")
            st.write(f"- 68% chance: {te - std_dev:.1f} to {te + std_dev:.1f} days")
            st.write(f"- 95% chance: {te - 2*std_dev:.1f} to {te + 2*std_dev:.1f} days")
            st.write(f"- 99.7% chance: {te - 3*std_dev:.1f} to {te + 3*std_dev:.1f} days")
    
    with tab3:
        st.markdown("### Project Completion Probability Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Project Parameters")
            te_project = st.number_input("Expected Project Duration (Tₑ)", value=38.0, step=0.5,
                                         help="Sum of expected times on critical path")
            d_target = st.number_input("Desired Completion (D)", value=35.0, step=0.5,
                                       help="Target completion date")
            sum_variance = st.number_input("Sum of CP Variances (Σσ²)", value=11.89, step=0.1,
                                           help="Sum of variances for all critical path activities")
        
        with col2:
            if sum_variance > 0:
                z_score = (d_target - te_project) / math.sqrt(sum_variance)
                prob = normal_cdf(z_score)
                
                st.markdown("#### Results")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"{z_score:.2f}", "Z-Score", "normal")
                with col_b:
                    card_type = "danger" if prob < 0.5 else "success"
                    display_metric_card(f"{prob*100:.1f}%", "P(Complete by D)", card_type)
                
                st.markdown("#### Interpretation")
                if prob < 0.25:
                    display_alert(
                        f"🔴 <strong>Very Low Probability ({prob*100:.1f}%)</strong><br>"
                        f"Only a {prob*100:.1f}% chance of completing by day {d_target}. "
                        f"Consider crashing critical activities or extending the deadline to {te_project}+ days.",
                        "danger"
                    )
                elif prob < 0.5:
                    display_alert(
                        f"🟡 <strong>Below Average Probability ({prob*100:.1f}%)</strong><br>"
                        f"Less than 50% chance of meeting the deadline. Risk mitigation recommended.",
                        "warning"
                    )
                else:
                    display_alert(
                        f"🟢 <strong>Good Probability ({prob*100:.1f}%)</strong><br>"
                        f"Reasonable chance of meeting the deadline.",
                        "success"
                    )
                
                st.markdown("#### Calculation")
                st.latex(rf"Z = \frac{{{d_target} - {te_project}}}{{\sqrt{{{sum_variance}}}}} = \frac{{{d_target - te_project:.2f}}}{{{math.sqrt(sum_variance):.2f}}} = {z_score:.2f}")
        
        # Variance Builder
        st.markdown("---")
        st.markdown("### Critical Path Variance Builder")
        st.write("Enter activity estimates to calculate total path variance")
        
        num_activities = st.number_input("Number of CP Activities", 1, 10, 5, key="pert_num_act")
        
        activities = []
        total_te = 0
        total_var = 0
        
        cols_header = st.columns([1, 1, 1, 1, 1, 1])
        cols_header[0].write("**Activity**")
        cols_header[1].write("**a**")
        cols_header[2].write("**m**")
        cols_header[3].write("**b**")
        cols_header[4].write("**Tₑ**")
        cols_header[5].write("**σ²**")
        
        for i in range(int(num_activities)):
            cols = st.columns([1, 1, 1, 1, 1, 1])
            with cols[0]:
                st.write(f"Activity {chr(65+i)}")
            with cols[1]:
                a_i = st.number_input(f"a_{i}", value=2+i, key=f"pert_a_{i}", label_visibility="collapsed")
            with cols[2]:
                m_i = st.number_input(f"m_{i}", value=4+i, key=f"pert_m_{i}", label_visibility="collapsed")
            with cols[3]:
                b_i = st.number_input(f"b_{i}", value=8+i*2, key=f"pert_b_{i}", label_visibility="collapsed")
            
            te_i = (a_i + 4*m_i + b_i) / 6
            var_i = ((b_i - a_i) / 6) ** 2
            total_te += te_i
            total_var += var_i
            
            with cols[4]:
                st.write(f"{te_i:.2f}")
            with cols[5]:
                st.write(f"{var_i:.3f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card(f"{total_te:.2f}", "Total Path Duration", "highlight")
        with col2:
            display_metric_card(f"{total_var:.3f}", "Total Path Variance", "normal")
        with col3:
            display_metric_card(f"{math.sqrt(total_var):.3f}", "Path Std Deviation", "normal")
    
    with tab4:
        st.markdown("### Practice Problems")
        
        # Problem 1
        with st.expander("📝 Problem 1: Calculate Expected Time (Easy)", expanded=True):
            st.markdown("""
            **Given:** Activity X has the following time estimates:
            - Optimistic (a) = 5 days
            - Most Likely (m) = 8 days
            - Pessimistic (b) = 17 days
            
            **Calculate:** Expected time (Tₑ) and variance (σ²)
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                ans_te = st.number_input("Expected Time (Tₑ):", key="pert_p1_te", format="%.2f")
            with col2:
                ans_var = st.number_input("Variance (σ²):", key="pert_p1_var", format="%.2f")
            
            if st.button("Check Answer", key="pert_p1_btn"):
                correct_te = (5 + 4*8 + 17) / 6
                correct_var = ((17 - 5) / 6) ** 2
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Tₑ = (a + 4m + b) / 6 = (5 + 4×8 + 17) / 6 = (5 + 32 + 17) / 6 = 54 / 6 = <strong>9.00 days</strong>")
                display_solution_step(2, f"σ² = ((b - a) / 6)² = ((17 - 5) / 6)² = (12 / 6)² = 2² = <strong>4.00</strong>")
                
                if check_answer(ans_te, correct_te) and check_answer(ans_var, correct_var):
                    display_alert("✅ Both answers correct!", "success")
                else:
                    if not check_answer(ans_te, correct_te):
                        display_alert(f"❌ Expected time incorrect. Correct answer: {correct_te:.2f}", "danger")
                    if not check_answer(ans_var, correct_var):
                        display_alert(f"❌ Variance incorrect. Correct answer: {correct_var:.2f}", "danger")
        
        # Problem 2
        with st.expander("📝 Problem 2: Project Completion Probability (Medium)"):
            st.markdown("""
            **Given:** A project has:
            - Expected duration (Tₑ) = 45 weeks
            - Sum of critical path variances = 16 weeks²
            - Target completion = 41 weeks
            
            **Calculate:** 
            1. The Z-score
            2. The probability of completing by week 41
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                ans_z = st.number_input("Z-Score:", key="pert_p2_z", format="%.2f")
            with col2:
                ans_prob = st.number_input("Probability (%):", key="pert_p2_prob", format="%.1f")
            
            if st.button("Check Answer", key="pert_p2_btn"):
                correct_z = (41 - 45) / math.sqrt(16)
                correct_prob = normal_cdf(correct_z) * 100
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Z = (D - Tₑ) / √(Σσ²) = (41 - 45) / √16 = -4 / 4 = <strong>-1.00</strong>")
                display_solution_step(2, f"Look up Z = -1.00 in standard normal table")
                display_solution_step(3, f"P(Z ≤ -1.00) = <strong>{correct_prob:.1f}%</strong>")
                
                display_alert(
                    f"<strong>Interpretation:</strong> There is only a {correct_prob:.1f}% chance of completing "
                    f"the project by week 41. The project manager should either extend the deadline or crash "
                    f"critical path activities.",
                    "info"
                )
        
        # Problem 3
        with st.expander("📝 Problem 3: Critical Path Analysis (Hard)"):
            st.markdown("""
            **Given:** A project has the following activities on the critical path:
            
            | Activity | a | m | b |
            |----------|---|---|---|
            | A | 2 | 4 | 6 |
            | B | 3 | 5 | 13 |
            | C | 4 | 6 | 8 |
            | D | 2 | 3 | 10 |
            
            **Calculate:**
            1. Expected time for each activity
            2. Total expected project duration
            3. Total variance
            4. Probability of completing in 20 days or less
            """)
            
            if st.button("Show Complete Solution", key="pert_p3_btn"):
                st.markdown("#### Solution:")
                
                activities = [
                    ("A", 2, 4, 6),
                    ("B", 3, 5, 13),
                    ("C", 4, 6, 8),
                    ("D", 2, 3, 10)
                ]
                
                results = []
                total_te = 0
                total_var = 0
                
                for name, a, m, b in activities:
                    te = (a + 4*m + b) / 6
                    var = ((b - a) / 6) ** 2
                    total_te += te
                    total_var += var
                    results.append({"Activity": name, "a": a, "m": m, "b": b, "Tₑ": f"{te:.2f}", "σ²": f"{var:.3f}"})
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
                display_solution_step(1, f"Total Expected Duration: Tₑ = {total_te:.2f} days")
                display_solution_step(2, f"Total Variance: Σσ² = {total_var:.3f}")
                display_solution_step(3, f"Standard Deviation: σ = √{total_var:.3f} = {math.sqrt(total_var):.3f}")
                
                z = (20 - total_te) / math.sqrt(total_var)
                prob = normal_cdf(z) * 100
                
                display_solution_step(4, f"Z = (20 - {total_te:.2f}) / {math.sqrt(total_var):.3f} = {z:.2f}")
                display_solution_step(5, f"P(Complete ≤ 20 days) = {prob:.1f}%")

# ============================================================
# MODULE 3: PROJECT CRASHING (Chapter 4)
# ============================================================
def module_crashing():
    display_header("⚡", "Chapter 4", "Project Crashing", "Time-cost trade-off analysis")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### Project Crashing Theory")
        
        st.write("""
        **Project crashing** (also called project compression) involves reducing project duration 
        by adding resources to critical path activities. The goal is to achieve the desired 
        completion date at minimum additional cost.
        """)
        
        st.markdown("#### Crash Cost per Day Formula")
        st.latex(r"\text{Crash Cost per Day} = \frac{\text{Crash Cost} - \text{Normal Cost}}{\text{Normal Time} - \text{Crash Time}}")
        
        st.markdown("#### Key Concepts")
        
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("⏱️", "Normal Time", 
                "Standard duration using normal resources and methods")
            display_concept_card("💰", "Normal Cost", 
                "Cost to complete activity in normal time")
        with col2:
            display_concept_card("⚡", "Crash Time", 
                "Minimum possible duration with maximum resources")
            display_concept_card("💸", "Crash Cost", 
                "Cost to complete activity in crash time (always higher)")
        
        display_key_insight(
            "Crashing Strategy",
            "Always crash the activity on the critical path with the <strong>lowest crash cost per day</strong> first. "
            "Continue until: (1) target date is reached, (2) critical path changes, or (3) no more crashing is possible. "
            "When the critical path changes, you may need to crash multiple paths simultaneously."
        )
        
        st.markdown("#### Crashing Procedure")
        st.write("""
        1. **Identify** the critical path
        2. **Calculate** crash cost per day for each critical activity
        3. **Select** the activity with lowest crash cost per day
        4. **Crash** that activity by one day (or until it reaches crash time or path changes)
        5. **Recalculate** critical path and repeat until target is met
        """)
    
    with tab2:
        st.markdown("### Crash Cost Calculator")
        
        num_activities = st.number_input("Number of Activities", 2, 10, 4, key="crash_num")
        
        activities = []
        
        st.markdown("#### Activity Data")
        
        cols_header = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1])
        headers = ["Activity", "Normal Time", "Crash Time", "Normal Cost", "Crash Cost", "Max Crash", "Cost/Day"]
        for i, h in enumerate(headers):
            cols_header[i].write(f"**{h}**")
        
        for i in range(int(num_activities)):
            cols = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1])
            
            with cols[0]:
                st.write(f"**{chr(65+i)}**")
            with cols[1]:
                nt = st.number_input(f"NT_{i}", value=5+i, key=f"crash_nt_{i}", label_visibility="collapsed")
            with cols[2]:
                ct = st.number_input(f"CT_{i}", value=3+i//2, key=f"crash_ct_{i}", label_visibility="collapsed")
            with cols[3]:
                nc = st.number_input(f"NC_{i}", value=1000+i*200, key=f"crash_nc_{i}", label_visibility="collapsed")
            with cols[4]:
                cc = st.number_input(f"CC_{i}", value=1800+i*400, key=f"crash_cc_{i}", label_visibility="collapsed")
            
            max_crash = nt - ct
            if max_crash > 0:
                cpd = (cc - nc) / max_crash
            else:
                cpd = float('inf')
            
            with cols[5]:
                st.write(f"{max_crash} days")
            with cols[6]:
                if cpd != float('inf'):
                    st.write(f"${cpd:.0f}")
                else:
                    st.write("N/A")
            
            activities.append({
                "Activity": chr(65+i),
                "Normal Time": nt,
                "Crash Time": ct,
                "Normal Cost": nc,
                "Crash Cost": cc,
                "Max Crash Days": max_crash,
                "Cost/Day": cpd if cpd != float('inf') else None
            })
        
        # Summary
        st.markdown("---")
        st.markdown("#### Summary")
        
        total_normal_time = sum(a["Normal Time"] for a in activities)
        total_crash_time = sum(a["Crash Time"] for a in activities)
        total_normal_cost = sum(a["Normal Cost"] for a in activities)
        total_crash_cost = sum(a["Crash Cost"] for a in activities)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric_card(f"{total_normal_time}", "Normal Duration", "normal")
        with col2:
            display_metric_card(f"{total_crash_time}", "Crash Duration", "highlight")
        with col3:
            display_metric_card(f"${total_normal_cost:,}", "Normal Cost", "normal")
        with col4:
            display_metric_card(f"${total_crash_cost:,}", "Full Crash Cost", "danger")
        
        # Crashing priority
        crashable = [a for a in activities if a["Cost/Day"] is not None and a["Max Crash Days"] > 0]
        if crashable:
            crashable_sorted = sorted(crashable, key=lambda x: x["Cost/Day"])
            
            st.markdown("#### Crashing Priority (Lowest Cost First)")
            for i, a in enumerate(crashable_sorted):
                st.write(f"{i+1}. **Activity {a['Activity']}**: ${a['Cost/Day']:.0f}/day (can crash {a['Max Crash Days']} days)")
    
    with tab3:
        st.markdown("### Practice Problems")
        
        with st.expander("📝 Problem 1: Calculate Crash Cost per Day"):
            st.markdown("""
            **Given:** Activity X has:
            - Normal Time = 10 days, Crash Time = 6 days
            - Normal Cost = $5,000, Crash Cost = $9,000
            
            **Calculate:** Crash cost per day
            """)
            
            ans = st.number_input("Crash Cost per Day ($):", key="crash_p1")
            
            if st.button("Check Answer", key="crash_p1_btn"):
                correct = (9000 - 5000) / (10 - 6)
                
                display_solution_step(1, "Crash Cost per Day = (Crash Cost - Normal Cost) / (Normal Time - Crash Time)")
                display_solution_step(2, f"= ($9,000 - $5,000) / (10 - 6)")
                display_solution_step(3, f"= $4,000 / 4 days = <strong>${correct:.0f}/day</strong>")
                
                if check_answer(ans, correct):
                    display_alert("✅ Correct!", "success")
                else:
                    display_alert(f"❌ Incorrect. Correct answer: ${correct:.0f}/day", "danger")

# ============================================================
# MODULE 4: BREAK-EVEN ANALYSIS (Chapter 5)
# ============================================================
def module_breakeven():
    display_header("📈", "Chapter 5", "Break-Even Analysis", "Cost-Volume-Profit (CVP) Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Theory", "🔬 Simulator", "📊 Sensitivity", "⚖️ Comparison", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Cost-Volume-Profit Analysis")
        
        st.write("""
        **Break-even analysis** determines the point at which total revenue equals total cost. 
        It establishes the relationship between fixed costs, variable costs, selling price, 
        and volume to identify the minimum output needed to cover all costs.
        """)
        
        display_citation(
            "Break-even analysis is a standard approach to determine the volume of output at which "
            "total revenue equals total cost. It is useful for comparing capacity alternatives and "
            "for determining the volume needed to achieve a target profit.",
            "Jacobs & Chase (2024, p. 155)"
        )
        
        st.markdown("#### Cost Structure Components")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🏢", "Fixed Costs (F)", 
                "Costs that remain constant regardless of output: rent, depreciation, insurance, salaries")
        with col2:
            display_concept_card("📦", "Variable Costs (V)", 
                "Costs that vary with output: raw materials, direct labor, packaging, shipping per unit")
        with col3:
            display_concept_card("💵", "Contribution Margin", 
                "Price minus Variable Cost (P - V). Each unit contributes this toward covering fixed costs")
        
        st.markdown("#### Key Formulas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Break-Even Point (Units)")
            st.latex(r"BEP_{units} = \frac{F}{P - V}")
            
            st.markdown("##### Break-Even Point (Revenue)")
            st.latex(r"BEP_{\$} = \frac{F}{1 - \frac{V}{P}} = BEP_{units} \times P")
        
        with col2:
            st.markdown("##### Volume for Target Profit")
            st.latex(r"Q_{target} = \frac{F + \text{Target Profit}}{P - V}")
            
            st.markdown("##### Total Cost & Revenue")
            st.latex(r"TC = F + V \cdot Q \quad ; \quad TR = P \cdot Q")
        
        display_key_insight(
            "Comparing Capacity Alternatives",
            "When comparing two process alternatives (e.g., manual vs. automated), the option with "
            "higher fixed costs but lower variable costs will have a higher BEP but becomes more "
            "profitable at higher volumes. The <strong>indifference point</strong> where both options "
            "yield equal total cost is: Q* = (F₂ - F₁) / (V₁ - V₂)"
        )
        
        st.markdown("#### Assumptions & Limitations")
        st.write("""
        - Revenue and costs are linear functions of volume
        - Fixed costs remain constant within the relevant range
        - All units produced are sold (no inventory buildup)
        - Single product analysis (or constant product mix)
        - Price and variable cost per unit are constant
        """)
    
    with tab2:
        st.markdown("### Break-Even Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cost Parameters")
            fixed_cost = st.slider("Fixed Costs ($)", 10000, 200000, 50000, 5000)
            price = st.slider("Price per Unit ($)", 10, 500, 100, 5)
            variable_cost = st.slider("Variable Cost per Unit ($)", 5, 400, 60, 5)
            
            if price <= variable_cost:
                display_alert("⚠️ Price must be greater than Variable Cost!", "danger")
        
        with col2:
            if price > variable_cost:
                bep_units = fixed_cost / (price - variable_cost)
                bep_revenue = bep_units * price
                contribution_margin = price - variable_cost
                cm_ratio = contribution_margin / price
                
                st.markdown("#### Results")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"{bep_units:,.0f}", "BEP (Units)", "highlight")
                with col_b:
                    display_metric_card(f"${bep_revenue:,.0f}", "BEP (Revenue)", "highlight")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"${contribution_margin:.2f}", "Contribution Margin", "normal")
                with col_b:
                    display_metric_card(f"{cm_ratio:.1%}", "CM Ratio", "normal")
                
                st.markdown("#### Calculation")
                st.latex(rf"BEP = \frac{{{fixed_cost:,}}}{{{price} - {variable_cost}}} = \frac{{{fixed_cost:,}}}{{{contribution_margin}}} = {bep_units:,.0f} \text{{ units}}")
        
        # Target Profit Calculator
        if price > variable_cost:
            st.markdown("---")
            st.markdown("### Target Profit Analysis")
            
            target_profit = st.number_input("Target Annual Profit ($)", value=25000, step=5000)
            
            target_units = (fixed_cost + target_profit) / (price - variable_cost)
            target_revenue = target_units * price
            
            col1, col2 = st.columns(2)
            with col1:
                display_metric_card(f"{target_units:,.0f}", "Required Units", "success")
            with col2:
                display_metric_card(f"${target_revenue:,.0f}", "Required Revenue", "success")
            
            st.latex(rf"Q_{{target}} = \frac{{{fixed_cost:,} + {target_profit:,}}}{{{price} - {variable_cost}}} = {target_units:,.0f} \text{{ units}}")
    
    with tab3:
        st.markdown("### Sensitivity Analysis (What-If)")
        
        display_citation(
            "Some companies call this 'what if' analysis. Answering these 'what if' questions can be "
            "useful for understanding how sensitive an analysis is to cost and profit assumptions.",
            "Jacobs & Chase (2024, p. 60)"
        )
        
        st.markdown("#### Base Case")
        col1, col2, col3 = st.columns(3)
        with col1:
            base_fc = st.number_input("Base Fixed Cost ($)", value=50000, key="sens_fc")
        with col2:
            base_price = st.number_input("Base Price ($)", value=100, key="sens_p")
        with col3:
            base_vc = st.number_input("Base Variable Cost ($)", value=60, key="sens_vc")
        
        if base_price > base_vc:
            base_bep = base_fc / (base_price - base_vc)
            
            st.markdown("#### Sensitivity Scenarios")
            
            scenarios = [
                ("Base Case", base_fc, base_price, base_vc),
                ("+25% Fixed Costs", base_fc * 1.25, base_price, base_vc),
                ("-25% Fixed Costs", base_fc * 0.75, base_price, base_vc),
                ("+$10 Price", base_fc, base_price + 10, base_vc),
                ("-$10 Price", base_fc, base_price - 10, base_vc),
                ("+$5 Variable Cost", base_fc, base_price, base_vc + 5),
                ("-$5 Variable Cost", base_fc, base_price, base_vc - 5),
            ]
            
            results = []
            for name, fc, p, vc in scenarios:
                if p > vc:
                    bep = fc / (p - vc)
                    change = ((bep - base_bep) / base_bep) * 100
                    results.append({
                        "Scenario": name,
                        "Fixed Cost": f"${fc:,.0f}",
                        "Price": f"${p:.0f}",
                        "Var Cost": f"${vc:.0f}",
                        "BEP (units)": f"{bep:,.0f}",
                        "Change": f"{change:+.1f}%"
                    })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            display_key_insight(
                "Sensitivity Insights",
                "Notice that BEP is most sensitive to changes in contribution margin (P - V). "
                "A $10 price increase has a larger impact than a 25% change in fixed costs."
            )
    
    with tab4:
        st.markdown("### Scenario Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Scenario A (Current)")
            fc_a = st.number_input("Fixed Costs A ($)", value=50000, key="be_fc_a")
            p_a = st.number_input("Price A ($)", value=100, key="be_p_a")
            vc_a = st.number_input("Variable Cost A ($)", value=60, key="be_vc_a")
            
            if p_a > vc_a:
                bep_a = fc_a / (p_a - vc_a)
                display_metric_card(f"{bep_a:,.0f}", "BEP A (units)", "highlight")
        
        with col2:
            st.markdown("#### Scenario B (Alternative)")
            fc_b = st.number_input("Fixed Costs B ($)", value=80000, key="be_fc_b")
            p_b = st.number_input("Price B ($)", value=100, key="be_p_b")
            vc_b = st.number_input("Variable Cost B ($)", value=45, key="be_vc_b")
            
            if p_b > vc_b:
                bep_b = fc_b / (p_b - vc_b)
                display_metric_card(f"{bep_b:,.0f}", "BEP B (units)", "highlight")
        
        # Indifference Point
        if p_a > vc_a and p_b > vc_b and vc_a != vc_b:
            st.markdown("---")
            st.markdown("### Indifference Analysis")
            
            indiff_point = (fc_b - fc_a) / (vc_a - vc_b)
            
            if indiff_point > 0:
                display_metric_card(f"{indiff_point:,.0f}", "Indifference Point (units)", "highlight")
                
                st.latex(rf"Q^* = \frac{{F_B - F_A}}{{V_A - V_B}} = \frac{{{fc_b:,} - {fc_a:,}}}{{{vc_a} - {vc_b}}} = {indiff_point:,.0f}")
                
                lower_fc = "A" if fc_a < fc_b else "B"
                lower_vc = "A" if vc_a < vc_b else "B"
                
                display_alert(
                    f"📊 <strong>Decision Rule:</strong><br>"
                    f"• Below {indiff_point:,.0f} units: Choose Scenario {lower_fc} (lower fixed costs)<br>"
                    f"• Above {indiff_point:,.0f} units: Choose Scenario {lower_vc} (lower variable costs)",
                    "info"
                )
    
    with tab5:
        st.markdown("### Practice Problems")
        
        # Problem 1
        with st.expander("📝 Problem 1: Basic BEP Calculation (Easy)", expanded=True):
            st.markdown("""
            **Given:**
            - Fixed Costs = $40,000
            - Selling Price = $120 per unit
            - Variable Cost = $80 per unit
            
            **Calculate:** Break-even point in units
            """)
            
            ans = st.number_input("BEP (units):", key="be_p1")
            
            if st.button("Check Answer", key="be_p1_btn"):
                correct = 40000 / (120 - 80)
                
                st.markdown("#### Solution:")
                display_solution_step(1, "Contribution Margin = Price - Variable Cost = $120 - $80 = $40")
                display_solution_step(2, "BEP = Fixed Costs / Contribution Margin")
                display_solution_step(3, f"BEP = $40,000 / $40 = <strong>{correct:,.0f} units</strong>")
                
                if check_answer(ans, correct):
                    display_alert("✅ Correct!", "success")
                else:
                    display_alert(f"❌ Incorrect. Correct answer: {correct:,.0f} units", "danger")
        
        # Problem 2
        with st.expander("📝 Problem 2: Target Profit (Medium)"):
            st.markdown("""
            **Given:**
            - Fixed Costs = $60,000
            - Selling Price = $50 per unit
            - Variable Cost = $30 per unit
            - Target Profit = $20,000
            
            **Calculate:** Units needed to achieve target profit
            """)
            
            ans = st.number_input("Required units:", key="be_p2")
            
            if st.button("Check Answer", key="be_p2_btn"):
                correct = (60000 + 20000) / (50 - 30)
                
                st.markdown("#### Solution:")
                display_solution_step(1, "Contribution Margin = $50 - $30 = $20")
                display_solution_step(2, "Q = (Fixed Costs + Target Profit) / CM")
                display_solution_step(3, f"Q = ($60,000 + $20,000) / $20 = $80,000 / $20 = <strong>{correct:,.0f} units</strong>")
                
                if check_answer(ans, correct):
                    display_alert("✅ Correct!", "success")
                else:
                    display_alert(f"❌ Incorrect. Correct answer: {correct:,.0f} units", "danger")
        
        # Problem 3
        with st.expander("📝 Problem 3: Indifference Point (Hard)"):
            st.markdown("""
            **Given:** Two manufacturing options:
            
            | Option | Fixed Costs | Variable Cost/Unit |
            |--------|-------------|-------------------|
            | Manual | $20,000 | $15 |
            | Automated | $80,000 | $5 |
            
            **Calculate:**
            1. Indifference point (where both options have equal total cost)
            2. Which option is better at 5,000 units?
            3. Which option is better at 8,000 units?
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ans_indiff = st.number_input("Indifference point:", key="be_p3_1")
            with col2:
                ans_5000 = st.selectbox("Better at 5,000:", ["Manual", "Automated"], key="be_p3_2")
            with col3:
                ans_8000 = st.selectbox("Better at 8,000:", ["Manual", "Automated"], key="be_p3_3")
            
            if st.button("Check Answer", key="be_p3_btn"):
                correct_indiff = (80000 - 20000) / (15 - 5)
                
                tc_manual_5000 = 20000 + 15 * 5000
                tc_auto_5000 = 80000 + 5 * 5000
                tc_manual_8000 = 20000 + 15 * 8000
                tc_auto_8000 = 80000 + 5 * 8000
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Indifference Point = (F₂ - F₁) / (V₁ - V₂) = ($80,000 - $20,000) / ($15 - $5) = $60,000 / $10 = <strong>{correct_indiff:,.0f} units</strong>")
                
                display_solution_step(2, f"At 5,000 units:<br>• Manual: $20,000 + $15×5,000 = ${tc_manual_5000:,}<br>• Automated: $80,000 + $5×5,000 = ${tc_auto_5000:,}<br><strong>Manual is better</strong>")
                
                display_solution_step(3, f"At 8,000 units:<br>• Manual: $20,000 + $15×8,000 = ${tc_manual_8000:,}<br>• Automated: $80,000 + $5×8,000 = ${tc_auto_8000:,}<br><strong>Automated is better</strong>")

# ============================================================
# MODULE 5: DECISION TREES (Chapter 5)
# ============================================================
def module_decision():
    display_header("🌳", "Chapter 5", "Decision Trees & Expected Monetary Value", 
                   "Structured decision-making under uncertainty")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### Expected Monetary Value (EMV) Analysis")
        
        st.write("""
        **Decision tree analysis** is a quantitative approach for evaluating alternatives that 
        involve sequential decisions and chance events. It provides a visual framework for 
        analyzing decisions under uncertainty.
        """)
        
        display_citation(
            "A decision tree is a schematic model of alternatives available to the decision maker, "
            "along with their possible consequences. The term gets its name from the tree-like "
            "appearance of the diagram.",
            "Jacobs & Chase (2024, p. 148)"
        )
        
        st.markdown("#### Decision Tree Components")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("◻️", "Decision Node", 
                "Point where decision maker chooses between alternatives (square symbol)")
        with col2:
            display_concept_card("⭕", "Chance Node", 
                "Point where chance determines outcome - probabilities must sum to 1.0 (circle symbol)")
        with col3:
            display_concept_card("🔺", "Terminal Node", 
                "Final payoff at end of branch - the monetary outcome (triangle symbol)")
        
        st.markdown("#### Key Formulas")
        
        st.markdown("##### Expected Monetary Value (EMV)")
        st.latex(r"EMV = \sum_{i=1}^{n} (P_i \times V_i)")
        st.write("Where Pᵢ = probability of outcome i, Vᵢ = monetary value of outcome i")
        st.write("**Decision Rule:** Select the alternative with the highest EMV")
        
        st.markdown("##### Expected Value of Perfect Information (EVPI)")
        st.latex(r"EVPI = EV_{with\ PI} - EV_{without\ PI}")
        st.write("EVPI represents the maximum amount you should pay for perfect information")
        
        display_key_insight(
            "Roll Back Method",
            "Decision trees are solved from <strong>right to left</strong> (backward induction). "
            "At each chance node, calculate the EMV. At each decision node, select the alternative "
            "with the highest EMV. This process 'rolls back' the tree to determine the optimal initial decision."
        )
    
    with tab2:
        st.markdown("### EMV Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏭 Large Facility Option")
            st.write("Higher risk, higher potential reward")
            
            prob_high = st.slider("P(High Demand)", 0, 100, 60, help="Probability of high demand scenario") / 100
            payoff_high_large = st.number_input("High Demand Payoff ($)", value=200000, key="dt_h_l")
            payoff_low_large = st.number_input("Low Demand Payoff ($)", value=-50000, key="dt_l_l")
            
            emv_large = prob_high * payoff_high_large + (1 - prob_high) * payoff_low_large
            
            st.markdown("##### Calculation:")
            st.latex(rf"EMV_{{Large}} = {prob_high:.2f} \times \${payoff_high_large:,} + {1-prob_high:.2f} \times \${payoff_low_large:,}")
            display_metric_card(f"${emv_large:,.0f}", "EMV (Large Facility)", "highlight")
        
        with col2:
            st.markdown("#### 🏠 Small Facility Option")
            st.write("Lower risk, more conservative")
            
            st.write(f"P(High Demand) = {prob_high:.0%} (same as above)")
            payoff_high_small = st.number_input("High Demand Payoff ($)", value=90000, key="dt_h_s")
            payoff_low_small = st.number_input("Low Demand Payoff ($)", value=25000, key="dt_l_s")
            
            emv_small = prob_high * payoff_high_small + (1 - prob_high) * payoff_low_small
            
            st.markdown("##### Calculation:")
            st.latex(rf"EMV_{{Small}} = {prob_high:.2f} \times \${payoff_high_small:,} + {1-prob_high:.2f} \times \${payoff_low_small:,}")
            display_metric_card(f"${emv_small:,.0f}", "EMV (Small Facility)", "normal")
        
        st.markdown("---")
        st.markdown("### Decision Recommendation")
        
        if emv_large > emv_small:
            display_alert(f"✅ <strong>Choose Large Facility</strong><br>EMV = ${emv_large:,.0f} > ${emv_small:,.0f}", "success")
        elif emv_small > emv_large:
            display_alert(f"✅ <strong>Choose Small Facility</strong><br>EMV = ${emv_small:,.0f} > ${emv_large:,.0f}", "success")
        else:
            display_alert(f"⚖️ <strong>Indifferent</strong><br>Both options have EMV = ${emv_large:,.0f}", "info")
        
        # EVPI Calculation
        st.markdown("---")
        st.markdown("### Expected Value of Perfect Information (EVPI)")
        
        # EV with perfect information = weighted average of best outcomes in each state
        best_high = max(payoff_high_large, payoff_high_small)
        best_low = max(payoff_low_large, payoff_low_small)
        ev_with_pi = prob_high * best_high + (1 - prob_high) * best_low
        ev_without_pi = max(emv_large, emv_small)
        evpi = ev_with_pi - ev_without_pi
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card(f"${ev_with_pi:,.0f}", "EV with Perfect Info", "normal")
        with col2:
            display_metric_card(f"${ev_without_pi:,.0f}", "EV without Perfect Info", "normal")
        with col3:
            display_metric_card(f"${evpi:,.0f}", "EVPI", "highlight")
        
        display_alert(
            f"💡 <strong>Interpretation:</strong> You should pay at most <strong>${evpi:,.0f}</strong> for perfect "
            f"market information (e.g., a market research study that perfectly predicts demand).",
            "info"
        )
    
    with tab3:
        st.markdown("### Practice Problems")
        
        with st.expander("📝 Problem 1: Calculate EMV (Easy)", expanded=True):
            st.markdown("""
            **Given:** A company is deciding between two options:
            - **Option A:** 40% chance of $100,000, 60% chance of $20,000
            - **Option B:** 50% chance of $80,000, 50% chance of $30,000
            
            **Calculate:** EMV for each option and determine which to choose
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                ans_a = st.number_input("EMV(A) ($):", key="dt_p1_a")
            with col2:
                ans_b = st.number_input("EMV(B) ($):", key="dt_p1_b")
            
            if st.button("Check Answer", key="dt_p1_btn"):
                emv_a = 0.4 * 100000 + 0.6 * 20000
                emv_b = 0.5 * 80000 + 0.5 * 30000
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"EMV(A) = 0.40 × $100,000 + 0.60 × $20,000 = $40,000 + $12,000 = <strong>${emv_a:,.0f}</strong>")
                display_solution_step(2, f"EMV(B) = 0.50 × $80,000 + 0.50 × $30,000 = $40,000 + $15,000 = <strong>${emv_b:,.0f}</strong>")
                display_solution_step(3, f"<strong>Choose Option {'A' if emv_a > emv_b else 'B'}</strong> (higher EMV)")
                
                if check_answer(ans_a, emv_a) and check_answer(ans_b, emv_b):
                    display_alert("✅ Both answers correct!", "success")

# ============================================================
# MODULE 6: LEARNING CURVES (Chapter 6)
# ============================================================
def module_learning():
    display_header("📉", "Chapter 6", "Learning Curves", "Experience curve cost reduction")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Learning Curve Model")
        st.write("""
        The **learning curve** describes the systematic reduction in production time (or cost) 
        as cumulative output doubles. It reflects the phenomenon that workers become more 
        efficient through repetition, process improvements, and organizational learning.
        """)
        
        display_citation(
            "The learning curve theory is based on three assumptions: (1) the amount of time required "
            "to complete a given task will be less each time the task is undertaken, (2) the unit time "
            "will decrease at a decreasing rate, and (3) the reduction in time will follow a specific "
            "and predictable pattern.",
            "Jacobs & Chase (2024, p. 168)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Unit Time Formula")
            st.latex(r"Y_x = K \cdot x^n")
            st.write("Where Yₓ = time for unit x, K = time for first unit, n = learning exponent")
        
        with col2:
            st.markdown("#### Learning Exponent")
            st.latex(r"n = \frac{\log(b)}{\log(2)}")
            st.write("Where b = learning rate (e.g., 0.80 for 80% curve)")
        
        display_key_insight(
            "Interpreting the Learning Rate",
            "An '80% learning curve' means that every time cumulative production doubles, the per-unit "
            "time drops to 80% of its previous level. Unit 1 = 100 hrs → Unit 2 = 80 hrs → Unit 4 = 64 hrs"
        )
    
    with tab2:
        st.markdown("### Learning Curve Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            k = st.slider("First Unit Time (K)", 10, 500, 100)
            learning_rate = st.slider("Learning Rate (%)", 50, 100, 80)
            
            b = learning_rate / 100
            n = math.log(b) / math.log(2)
            
            st.metric("Learning Exponent (n)", f"{n:.3f}")
            st.latex(rf"n = \frac{{\log({b})}}{{\log(2)}} = {n:.3f}")
        
        with col2:
            st.markdown("#### Unit Times Table")
            units = [1, 2, 4, 8, 16, 32, 64, 128]
            times = [k * (u ** n) for u in units]
            
            df = pd.DataFrame({
                "Unit": units,
                "Time (hrs)": [f"{t:.1f}" for t in times]
            })
            st.dataframe(df, use_container_width=True)
        
        # Specific Unit Calculator
        st.markdown("---")
        st.markdown("### Calculate for Specific Unit")
        target_unit = st.number_input("Calculate time for unit #", value=10, min_value=1)
        
        unit_time = k * (target_unit ** n)
        
        # Cumulative time (approximation using integral)
        cumulative_time = k * (target_unit ** (n + 1)) / (n + 1) if n != -1 else k * math.log(target_unit)
        avg_time = cumulative_time / target_unit if target_unit > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Time for Unit {target_unit}", f"{unit_time:.1f} hrs")
        with col2:
            st.metric("Cumulative Time (1 to X)", f"{cumulative_time:.1f} hrs")
        with col3:
            st.metric("Cumulative Average", f"{avg_time:.1f} hrs")
    
    with tab3:
        st.markdown("### Practice Problems")
        
        with st.expander("Problem: Calculate Unit Time"):
            st.write("First unit takes 100 hours. With an 80% learning curve, how long will unit 8 take?")
            user_ans = st.number_input("Your Answer (hours):", key="lc_p1")
            if st.button("Check Answer", key="lc_p1_btn"):
                n = math.log(0.8) / math.log(2)
                correct = 100 * (8 ** n)
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Y₈ = 100 × 8^{n:.3f} = {correct:.1f} hours")
                else:
                    st.error(f"❌ Incorrect. Y₈ = 100 × 8^{n:.3f} = {correct:.1f} hours")

# ============================================================
# MODULE 7: DECOUPLING POINT (Chapter 7)
# ============================================================
def module_decoupling():
    display_header("🔀", "Chapter 7", "Customer Order Decoupling Point", 
                   "MTS vs. MTO vs. ATO vs. ETO strategies")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Configuration Calculator"])
    
    with tab1:
        st.markdown("### Customer Order Decoupling")
        st.write("""
        The **customer order decoupling point** is where inventory is positioned to allow 
        processes in the supply chain to operate independently. Upstream operations are 
        forecast-driven; downstream operations are customer-order-driven.
        """)
        
        st.markdown("#### Product Configurations Formula")
        st.latex(r"\text{Configurations} = \prod_{i=1}^{n} N_i = N_1 \times N_2 \times \cdots \times N_n")
        
        st.markdown("#### Manufacturing Strategies")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📦 Make-to-Stock (MTS)**
            - Produce to forecast, sell from inventory
            - Fast delivery, high inventory risk
            - Example: Consumer goods
            
            **🔧 Assemble-to-Order (ATO)**
            - Build from pre-made modules when order received
            - Balance of speed and customization
            - Example: Dell computers
            """)
        
        with col2:
            st.markdown("""
            **📐 Make-to-Order (MTO)**
            - Manufacture after order received
            - Longer lead time, lower inventory
            - Example: Boeing aircraft
            
            **✏️ Engineer-to-Order (ETO)**
            - Design and build from scratch
            - Longest lead time
            - Example: Custom machinery
            """)
    
    with tab2:
        st.markdown("### Configuration Calculator")
        st.write("Calculate total product configurations from modular options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            opt1 = st.number_input("Option 1 (e.g., Processors)", value=3, min_value=1)
            opt2 = st.number_input("Option 2 (e.g., Memory)", value=3, min_value=1)
            opt3 = st.number_input("Option 3 (e.g., Storage)", value=4, min_value=1)
            opt4 = st.number_input("Option 4 (e.g., Display)", value=2, min_value=1)
            opt5 = st.number_input("Option 5 (e.g., Color)", value=4, min_value=1)
        
        with col2:
            total_configs = opt1 * opt2 * opt3 * opt4 * opt5
            total_components = opt1 + opt2 + opt3 + opt4 + opt5
            
            st.metric("Total Configurations", f"{total_configs:,}")
            st.metric("Total Components", f"{total_components}")
            
            st.success(f"""
            💡 Only **{total_components}** components create **{total_configs:,}** unique products — 
            the power of modular design!
            """)

# ============================================================
# MODULE 8: LINE BALANCING (Chapter 8)
# ============================================================
def module_linebalance():
    display_header("⚖️", "Chapter 8", "Assembly Line Balancing", 
                   "Assigning tasks to workstations to minimize idle time")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Simulator"])
    
    with tab1:
        st.markdown("### Line Balancing Fundamentals")
        st.write("""
        **Assembly line balancing** assigns tasks to workstations so that each station has 
        approximately the same amount of work. The goal is to minimize the number of 
        workstations while respecting precedence constraints and cycle time limits.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Cycle Time")
            st.latex(r"C = \frac{\text{Production Time per Day}}{\text{Required Output per Day}}")
        
        with col2:
            st.markdown("#### Minimum Workstations")
            st.latex(r"N_{min} = \left\lceil \frac{\sum t_i}{C} \right\rceil")
        
        st.markdown("#### Line Efficiency")
        st.latex(r"\text{Efficiency} = \frac{\sum t_i}{N_{actual} \times C} \times 100\%")
        
        display_key_insight(
            "Balance Delay",
            "Balance Delay = 100% - Efficiency. It represents the percentage of idle time across all workstations."
        )
    
    with tab2:
        st.markdown("### Line Balancing Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prod_time = st.number_input("Production Time per Day (seconds)", value=28800)
            output_req = st.number_input("Required Output per Day (units)", value=480)
            sum_task = st.number_input("Sum of All Task Times (seconds)", value=195)
            num_stations = st.number_input("Actual Number of Workstations", value=4)
        
        with col2:
            if output_req > 0:
                cycle_time = prod_time / output_req
                n_min = math.ceil(sum_task / cycle_time)
                efficiency = (sum_task / (num_stations * cycle_time)) * 100
                balance_delay = 100 - efficiency
                total_idle = (num_stations * cycle_time) - sum_task
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Cycle Time", f"{cycle_time:.0f}s")
                    st.metric("Efficiency", f"{efficiency:.1f}%")
                with col_b:
                    st.metric("Min Workstations", f"{n_min}")
                    st.metric("Balance Delay", f"{balance_delay:.1f}%")
                
                st.metric("Total Idle Time/Cycle", f"{total_idle:.0f}s")
                
                # Visual representation
                st.markdown("#### Workstation Utilization")
                avg_util = efficiency
                for i in range(int(num_stations)):
                    st.progress(min(avg_util/100, 1.0), text=f"Station {i+1}: {avg_util:.0f}%")

# ============================================================
# MODULE 9: SERVICE DESIGN (Chapter 9)
# ============================================================
def module_service():
    display_header("🎯", "Chapter 9", "Service Process Design", "Designing customer-centric delivery systems")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔺 Service Triangle", "📋 Blueprinting"])
    
    with tab1:
        st.markdown("### Service Design Theory")
        st.write("""
        **Service design** differs from product design because the process IS the product. 
        Key factors include simultaneity, lack of legal protection, and the service package.
        """)
        
        display_citation(
            "The process and the product must be developed simultaneously; indeed, in services, "
            "the process is the product.",
            "Jacobs & Chase (2024, p. 229)"
        )
        
        st.markdown("#### Service-System Design Matrix")
        df = pd.DataFrame({
            "Contact Level": ["Low", "Medium", "High"],
            "Example": ["Internet/Technology", "Phone Contact", "Face-to-Face"],
            "Worker Skills": ["Clerical", "Procedural", "Diagnostic"],
            "Efficiency": ["High", "Medium", "Low"]
        })
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### The Service Triangle")
        st.write("""
        The Service Triangle illustrates the relationships between four key elements:
        - **Service Strategy** - The organization's service vision
        - **Systems** - Procedures and equipment
        - **People** - Employees who deliver service
        - **Customer** - At the center of everything
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Strategy")
            st.write("Service vision and positioning")
        with col2:
            st.markdown("### 👤 Customer")
            st.write("Central focus of all elements")
        with col3:
            st.markdown("### ⚙️ Systems")
            st.write("Procedures and technology")
    
    with tab3:
        st.markdown("### Service Blueprinting")
        st.write("""
        A **service blueprint** is a visual mapping tool that shows the service process, 
        points of customer contact, and the evidence of service from the customer's viewpoint.
        """)
        
        st.markdown("#### Key Feature: Line of Visibility")
        st.info("The **Line of Visibility** separates activities visible to customers (onstage) from those that are not (backstage).")
        
        st.markdown("#### Example: Restaurant Service Blueprint")
        
        df = pd.DataFrame({
            "Layer": ["Customer Actions", "Onstage (Visible)", "Backstage (Invisible)", "Support Processes"],
            "Step 1": ["Arrive", "Greet", "Prep Table", "Inventory"],
            "Step 2": ["Order", "Take Order", "Cook Food", "Purchasing"],
            "Step 3": ["Eat", "Serve Food", "Plate Food", "Scheduling"],
            "Step 4": ["Pay", "Process Payment", "Clean", "Accounting"]
        })
        st.dataframe(df, use_container_width=True)

# ============================================================
# MODULE 10: QUEUING THEORY (Chapter 10)
# ============================================================
def module_queuing():
    display_header("👥", "Chapter 10", "Queuing Theory (Waiting Line Models)", 
                   "Analyzing the trade-off between service capacity and customer waiting")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Theory", "🔬 M/M/1", "👥 M/M/s", "💰 Cost Analysis", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Waiting Line Models")
        st.write("""
        **Queuing theory** provides mathematical models to analyze waiting lines. It helps 
        managers balance the cost of providing service capacity against the cost of customers waiting.
        """)
        
        display_citation(
            "Queuing theory enables us to analyze the relationship between demand on a service system "
            "and the delays suffered by users of that system. It helps managers make capacity decisions "
            "by quantifying the relationship between arrival rates, service rates, and system performance.",
            "Jacobs & Chase (2024, p. 286)"
        )
        
        st.markdown("#### Kendall's Notation: A/B/s")
        st.write("""
        - **A** = Arrival distribution (M = Markovian/Poisson, D = Deterministic)
        - **B** = Service time distribution (M = Exponential, D = Deterministic)
        - **s** = Number of servers
        """)
        
        st.markdown("#### M/M/1 Formulas")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"\rho = \frac{\lambda}{\mu}")
            st.write("Utilization (must be < 1)")
            
            st.latex(r"L_s = \frac{\lambda}{\mu - \lambda}")
            st.write("Average number in system")
        
        with col2:
            st.latex(r"L_q = \frac{\lambda^2}{\mu(\mu - \lambda)}")
            st.write("Average number in queue")
            
            st.latex(r"W_s = \frac{1}{\mu - \lambda}")
            st.write("Average time in system")
        
        display_key_insight(
            "Little's Law Connection",
            "Little's Law (L = λW) connects all queuing metrics. Ls = λWs and Lq = λWq"
        )
    
    with tab2:
        st.markdown("### M/M/1 Single Server Queue")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lam = st.slider("Arrival Rate (λ) per hour", 1, 50, 10)
            mu = st.slider("Service Rate (μ) per hour", 1, 60, 15)
        
        with col2:
            if lam < mu:
                rho = lam / mu
                Ls = lam / (mu - lam)
                Lq = (lam ** 2) / (mu * (mu - lam))
                Ws = 1 / (mu - lam)
                Wq = lam / (mu * (mu - lam))
                P0 = 1 - rho
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Utilization (ρ)", f"{rho:.1%}")
                    st.metric("Avg in System (Ls)", f"{Ls:.2f}")
                    st.metric("Avg in Queue (Lq)", f"{Lq:.2f}")
                with col_b:
                    st.metric("P(System Empty)", f"{P0:.1%}")
                    st.metric("Avg Time in System (Ws)", f"{Ws:.2f} hrs")
                    st.metric("Avg Wait in Queue (Wq)", f"{Wq:.2f} hrs")
            else:
                st.error("⚠️ Unstable system! λ must be < μ")
    
    with tab3:
        st.markdown("### M/M/s Multi-Server Queue")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lam_s = st.number_input("Arrival Rate (λ)", value=15, key="mms_lam")
            mu_s = st.number_input("Service Rate per Server (μ)", value=6, key="mms_mu")
            s = st.number_input("Number of Servers (s)", value=3, min_value=1, key="mms_s")
        
        with col2:
            if lam_s < s * mu_s:
                rho_s = lam_s / (s * mu_s)
                
                # Calculate P0 for M/M/s
                sum_term = sum([(lam_s/mu_s)**n / math.factorial(n) for n in range(int(s))])
                last_term = ((lam_s/mu_s)**s / math.factorial(int(s))) * (s*mu_s / (s*mu_s - lam_s))
                P0_s = 1 / (sum_term + last_term)
                
                # Calculate Lq
                Lq_s = (P0_s * (lam_s/mu_s)**s * rho_s) / (math.factorial(int(s)) * (1 - rho_s)**2)
                Wq_s = Lq_s / lam_s
                Ws_s = Wq_s + 1/mu_s
                Ls_s = lam_s * Ws_s
                
                st.metric("Utilization (ρ)", f"{rho_s:.1%}")
                st.metric("Avg in Queue (Lq)", f"{Lq_s:.3f}")
                st.metric("Avg Wait in Queue (Wq)", f"{Wq_s:.3f} hrs")
                st.metric("Avg in System (Ls)", f"{Ls_s:.3f}")
            else:
                st.error("⚠️ Unstable! λ must be < s×μ")
    
    with tab4:
        st.markdown("### Queue Cost Analysis")
        st.write("Balance service costs against waiting costs")
        
        st.latex(r"TC = L_s \times C_w + S \times C_s")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lam_c = st.number_input("Arrival Rate (λ)", value=3, key="qc_lam")
            Cw = st.number_input("Waiting Cost ($/hr/customer)", value=25, key="qc_cw")
            Cs = st.number_input("Service Cost ($/hr/server)", value=16, key="qc_cs")
        
        with col2:
            st.markdown("#### Compare Scenarios")
            
            scenarios = []
            for i, (workers, mu_val) in enumerate([(1, 4), (1, 7), (2, 4)]):
                if lam_c < workers * mu_val:
                    if workers == 1:
                        Ls_c = lam_c / (mu_val - lam_c)
                    else:
                        # Simplified M/M/s calculation
                        rho_c = lam_c / (workers * mu_val)
                        Ls_c = lam_c / (mu_val - lam_c/workers) if rho_c < 1 else float('inf')
                    
                    wait_cost = Ls_c * Cw
                    labor_cost = workers * Cs
                    total_cost = wait_cost + labor_cost
                    
                    scenarios.append({
                        "Scenario": f"Case {i+1}",
                        "Workers": workers,
                        "μ": mu_val,
                        "Ls": f"{Ls_c:.2f}",
                        "Wait Cost": f"${wait_cost:.2f}",
                        "Labor Cost": f"${labor_cost:.2f}",
                        "Total Cost": f"${total_cost:.2f}"
                    })
            
            df = pd.DataFrame(scenarios)
            st.dataframe(df, use_container_width=True)
    
    with tab5:
        st.markdown("### Practice Problems")
        
        with st.expander("Problem: Calculate Lq"):
            st.write("λ = 12/hr, μ = 18/hr. Calculate Lq for M/M/1.")
            user_ans = st.number_input("Your Answer:", key="q_p1", format="%.2f")
            if st.button("Check Answer", key="q_p1_btn"):
                correct = (12**2) / (18 * (18 - 12))
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Lq = 12²/(18×6) = {correct:.2f}")
                else:
                    st.error(f"❌ Incorrect. Lq = 12²/(18×6) = {correct:.2f}")

# ============================================================
# MODULE 11: DISTRIBUTIONS (Chapter 10)
# ============================================================
def module_distributions():
    display_header("📐", "Chapter 10", "Exponential & Poisson Distributions", 
                   "Probability distributions for queuing analysis")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "📊 Exponential", "🎲 Poisson"])
    
    with tab1:
        st.markdown("### Probability Distributions for Queuing")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Exponential Distribution")
            st.latex(r"f(t) = \lambda e^{-\lambda t}")
            st.latex(r"P(t \leq T) = 1 - e^{-\lambda T}")
            st.write("Models inter-arrival times and service times")
        
        with col2:
            st.markdown("#### Poisson Distribution")
            st.latex(r"P_T(n) = \frac{(\lambda T)^n \cdot e^{-\lambda T}}{n!}")
            st.write("Models number of arrivals in time period T")
        
        display_key_insight(
            "Relationship",
            "If arrivals follow a Poisson process with rate λ, then inter-arrival times follow "
            "an exponential distribution with parameter λ. Mean inter-arrival time = 1/λ."
        )
    
    with tab2:
        st.markdown("### Exponential Distribution Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            exp_lambda = st.number_input("Rate (λ per time unit)", value=1.0, min_value=0.1, step=0.1)
            exp_t = st.number_input("Time (t)", value=2.0, min_value=0.0, step=0.1)
        
        with col2:
            p_within = 1 - math.exp(-exp_lambda * exp_t)
            p_beyond = math.exp(-exp_lambda * exp_t)
            
            st.metric("P(within t)", f"{p_within:.1%}")
            st.metric("P(more than t)", f"{p_beyond:.1%}")
        
        # Table
        st.markdown("#### Probability Table")
        times = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
        data = []
        for t in times:
            p_more = math.exp(-exp_lambda * t)
            p_less = 1 - p_more
            data.append({"t": t, "P(more than t)": f"{p_more:.4f}", "P(within t)": f"{p_less:.4f}"})
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    
    with tab3:
        st.markdown("### Poisson Distribution Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            poi_lambda = st.number_input("Arrival Rate (λ)", value=3.0, min_value=0.1, step=0.5)
            poi_T = st.number_input("Time Period (T)", value=1.0, min_value=0.1, step=0.5)
            poi_n = st.number_input("Number of Arrivals (n)", value=2, min_value=0)
        
        with col2:
            lambda_T = poi_lambda * poi_T
            p_n = poisson_pmf(poi_n, lambda_T)
            
            st.metric(f"P(exactly {poi_n} arrivals)", f"{p_n:.1%}")
            st.metric("λT", f"{lambda_T:.1f}")
        
        # Distribution table
        st.markdown("#### Distribution Table")
        data = []
        cumulative = 0
        for n in range(10):
            p = poisson_pmf(n, lambda_T)
            cumulative += p
            data.append({"n": n, "P(n)": f"{p:.4f}", "P(≤n)": f"{cumulative:.4f}"})
        st.dataframe(pd.DataFrame(data), use_container_width=True)

# ============================================================
# MODULE 12: LITTLE'S LAW (Chapter 11)
# ============================================================
def module_littles():
    display_header("🔄", "Chapter 11", "Little's Law", "The fundamental law of process analysis")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Simulator"])
    
    with tab1:
        st.markdown("### Little's Law: I = R × T")
        st.write("""
        **Little's Law** states that the long-run average number of items in a stable system 
        equals the long-run average arrival rate multiplied by the average time an item spends 
        in the system.
        """)
        
        display_citation(
            "Little's law states a mathematical relationship between throughput rate, flow time, "
            "and the amount of work-in-process inventory. It applies to any stable system, "
            "regardless of the distribution of arrivals or service times.",
            "Jacobs & Chase (2024, p. 312)"
        )
        
        st.latex(r"I = R \times T")
        st.latex(r"T = \frac{I}{R} \quad ; \quad R = \frac{I}{T}")
        
        st.markdown("#### Applications")
        st.write("""
        - **Manufacturing:** If R = 100 units/day and WIP = 500 units, then T = 5 days
        - **Hospital:** If R = 20 patients/day and T = 4 days, then I = 80 patients
        - **Call Center:** If R = 60 calls/hour and T = 5 min, then I = 5 calls
        """)
        
        display_key_insight(
            "Reducing Flow Time",
            "To reduce flow time (T) without reducing throughput (R), you must reduce WIP (I). "
            "This is the theoretical foundation for Lean Manufacturing and JIT systems."
        )
    
    with tab2:
        st.markdown("### Little's Law Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            R = st.slider("Throughput Rate (R) units/hr", 1, 50, 10)
            T = st.slider("Flow Time (T) hours", 1, 24, 5)
        
        with col2:
            I = R * T
            
            st.metric("WIP Inventory (I)", f"{I} units")
            st.latex(rf"I = {R} \times {T} = {I}")
            
            st.markdown("#### Pipeline Visualization")
            st.progress(min(I/100, 1.0), text=f"WIP: {I} units in system")

# ============================================================
# MODULE 13: SIX SIGMA / DPMO (Chapter 12)
# ============================================================
def module_dpmo():
    display_header("🎯", "Chapter 12", "DPMO & DMAIC", "Six Sigma quality methodology")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 DPMO Calculator", "🔄 DMAIC"])
    
    with tab1:
        st.markdown("### Six Sigma Fundamentals")
        st.write("""
        **Six Sigma** is a highly disciplined process that helps focus on developing and 
        delivering near-perfect products and services. The term refers to a statistical 
        measure of quality that strives for near perfection.
        """)
        
        st.latex(r"DPMO = \frac{\text{Total Defects}}{\text{Total Opportunities}} \times 1,000,000")
        
        st.markdown("#### Sigma Level Conversion")
        df = pd.DataFrame({
            "Sigma Level": [1, 2, 3, 4, 5, 6],
            "DPMO": ["691,462", "308,538", "66,807", "6,210", "233", "3.4"],
            "Yield": ["30.9%", "69.1%", "93.3%", "99.38%", "99.977%", "99.9997%"]
        })
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### DPMO Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            units = st.number_input("Total Units Inspected", value=2000)
            defects = st.number_input("Total Defects Found", value=33)
            opportunities = st.number_input("Opportunities per Unit", value=5)
        
        with col2:
            if units > 0 and opportunities > 0:
                total_opp = units * opportunities
                dpmo = (defects / total_opp) * 1000000
                dpu = defects / units
                yield_pct = (1 - defects/total_opp) * 100
                
                # Approximate sigma level
                if dpmo > 0:
                    sigma = stats.norm.ppf(1 - dpmo/1000000) + 1.5
                else:
                    sigma = 6
                
                st.metric("DPMO", f"{dpmo:,.0f}")
                st.metric("Sigma Level", f"{sigma:.1f}σ")
                st.metric("DPU", f"{dpu:.4f}")
                st.metric("Yield", f"{yield_pct:.1f}%")
    
    with tab3:
        st.markdown("### DMAIC Methodology")
        
        phases = {
            "Define": ("📋", "Project Charter, SIPOC", "Scope & Problem Statement"),
            "Measure": ("📏", "Process Map, DPMO", "Baseline Performance"),
            "Analyze": ("🔍", "Fishbone, Pareto, 5 Whys", "Root Causes"),
            "Improve": ("⚡", "Poka-yoke, Pilot Testing", "Solutions"),
            "Control": ("🎛️", "SPC, Control Plans", "Sustainability")
        }
        
        for phase, (icon, tools, deliverable) in phases.items():
            with st.expander(f"{icon} {phase}"):
                st.write(f"**Key Tools:** {tools}")
                st.write(f"**Deliverable:** {deliverable}")

# ============================================================
# MODULE 14: FMEA (Chapter 12)
# ============================================================
def module_fmea():
    display_header("⚠️", "Chapter 12", "FMEA Risk Analysis", "Failure Mode and Effects Analysis")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### FMEA Theory")
        st.write("""
        **Failure Mode and Effects Analysis (FMEA)** is a systematic procedure for identifying 
        potential failure modes and their causes. It prioritizes failures based on their 
        Risk Priority Number (RPN).
        """)
        
        st.latex(r"RPN = \text{Severity} \times \text{Occurrence} \times \text{Detection}")
        
        st.markdown("#### Rating Scales (1-10)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Severity (S)**")
            st.write("1 = No effect")
            st.write("10 = Hazardous")
        with col2:
            st.markdown("**Occurrence (O)**")
            st.write("1 = Remote")
            st.write("10 = Very high")
        with col3:
            st.markdown("**Detection (D)**")
            st.write("1 = Almost certain")
            st.write("10 = Cannot detect")
    
    with tab2:
        st.markdown("### FMEA Worksheet")
        
        num_modes = st.number_input("Number of Failure Modes", 2, 10, 3)
        
        modes = []
        for i in range(int(num_modes)):
            cols = st.columns(5)
            with cols[0]:
                step = st.text_input(f"Step {i+1}", value=f"Step {i+1}", key=f"fmea_step_{i}")
            with cols[1]:
                failure = st.text_input(f"Failure {i+1}", value=f"Failure Mode {i+1}", key=f"fmea_fail_{i}")
            with cols[2]:
                s = st.number_input("S", 1, 10, 5+i, key=f"fmea_s_{i}")
            with cols[3]:
                o = st.number_input("O", 1, 10, 4, key=f"fmea_o_{i}")
            with cols[4]:
                d = st.number_input("D", 1, 10, 6, key=f"fmea_d_{i}")
            
            rpn = s * o * d
            modes.append({"Step": step, "Failure Mode": failure, "S": s, "O": o, "D": d, "RPN": rpn})
        
        df = pd.DataFrame(modes)
        df = df.sort_values("RPN", ascending=False)
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total RPN", sum(m["RPN"] for m in modes))
        with col2:
            st.metric("Highest RPN", max(m["RPN"] for m in modes))
        with col3:
            st.metric("Average RPN", f"{sum(m['RPN'] for m in modes)/len(modes):.0f}")

# ============================================================
# MODULE 15: SQC - CONTROL CHARTS (Chapter 13)
# ============================================================
def module_sqc():
    display_header("📉", "Chapter 13", "Statistical Quality Control", "p-Chart & c-Chart calculators")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "📊 p-Chart", "📊 c-Chart"])
    
    with tab1:
        st.markdown("### Attribute Control Charts")
        st.write("""
        When quality is measured as a proportion (defective/not defective), we use **p-charts**. 
        When we count the number of defects per unit, we use **c-charts**.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### p-Chart Formulas")
            st.latex(r"\bar{p} = \frac{\text{Total Defectives}}{\text{Total Inspected}}")
            st.latex(r"S_p = \sqrt{\frac{\bar{p}(1-\bar{p})}{n}}")
            st.latex(r"UCL = \bar{p} + 3S_p")
            st.latex(r"LCL = \bar{p} - 3S_p")
        
        with col2:
            st.markdown("#### c-Chart Formulas")
            st.latex(r"\bar{c} = \frac{\text{Total Defects}}{\text{Number of Units}}")
            st.latex(r"UCL = \bar{c} + 3\sqrt{\bar{c}}")
            st.latex(r"LCL = \bar{c} - 3\sqrt{\bar{c}}")
    
    with tab2:
        st.markdown("### p-Chart Calculator")
        
        n = st.number_input("Sample Size (n)", value=300)
        num_samples = st.number_input("Number of Samples", value=10, min_value=3)
        
        st.markdown("Enter defectives for each sample:")
        defectives = []
        cols = st.columns(5)
        for i in range(int(num_samples)):
            with cols[i % 5]:
                d = st.number_input(f"Sample {i+1}", value=8+i%5, key=f"pc_d_{i}")
                defectives.append(d)
        
        total_def = sum(defectives)
        total_insp = n * num_samples
        p_bar = total_def / total_insp
        sp = math.sqrt(p_bar * (1 - p_bar) / n)
        ucl = p_bar + 3 * sp
        lcl = max(0, p_bar - 3 * sp)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("p̄ (Center)", f"{p_bar:.4f}")
        with col2:
            st.metric("UCL", f"{ucl:.4f}")
        with col3:
            st.metric("LCL", f"{lcl:.4f}")
        
        # Check for out of control
        proportions = [d/n for d in defectives]
        ooc = [i+1 for i, p in enumerate(proportions) if p > ucl or p < lcl]
        
        if ooc:
            st.error(f"⚠️ Out of Control: Samples {ooc}")
        else:
            st.success("✅ All points within control limits")
    
    with tab3:
        st.markdown("### c-Chart Calculator")
        
        num_units = st.number_input("Number of Units", value=10, min_value=3, key="cc_units")
        
        st.markdown("Enter defects for each unit:")
        defects_list = []
        cols = st.columns(5)
        for i in range(int(num_units)):
            with cols[i % 5]:
                c = st.number_input(f"Unit {i+1}", value=4+i%3, key=f"cc_c_{i}")
                defects_list.append(c)
        
        c_bar = sum(defects_list) / num_units
        ucl_c = c_bar + 3 * math.sqrt(c_bar)
        lcl_c = max(0, c_bar - 3 * math.sqrt(c_bar))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("c̄ (Center)", f"{c_bar:.2f}")
        with col2:
            st.metric("UCL", f"{ucl_c:.2f}")
        with col3:
            st.metric("LCL", f"{lcl_c:.2f}")

# ============================================================
# MODULE 16: PROCESS CAPABILITY (Chapter 13)
# ============================================================
def module_capability():
    display_header("🎯", "Chapter 13", "Process Capability — Cp & Cpk", 
                   "Measuring process performance relative to specification limits")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Process Capability Indices")
        st.write("""
        **Process capability** compares the output of a process to the specification limits. 
        **Cp** measures potential capability (if centered), while **Cpk** measures actual 
        capability accounting for any shift from center.
        """)
        
        display_citation(
            "Working with our example in Exhibit 13.4, let's assume our process is centered at 1.251 "
            "and σ = 0.00083. Cpk = 1.6, which is the smaller number. This is a pretty good capability "
            "index because few defects will be produced by this process.",
            "Jacobs & Chase (2024, p. 374)"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Cp (Potential)")
            st.latex(r"C_p = \frac{USL - LSL}{6\sigma}")
        
        with col2:
            st.markdown("#### Cpk (Actual)")
            st.latex(r"C_{pk} = \min\left(\frac{USL - \bar{X}}{3\sigma}, \frac{\bar{X} - LSL}{3\sigma}\right)")
        
        st.markdown("#### Interpretation")
        df = pd.DataFrame({
            "Cpk": ["< 1.0", "1.0 - 1.33", "1.33 - 1.67", "> 1.67"],
            "Assessment": ["Not Capable", "Marginally Capable", "Capable", "Highly Capable"],
            "Action": ["Immediate improvement needed", "Monitor closely", "Acceptable", "Excellent"]
        })
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### Cpk Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            usl = st.number_input("Upper Spec Limit (USL)", value=1.255, format="%.4f")
            lsl = st.number_input("Lower Spec Limit (LSL)", value=1.245, format="%.4f")
            mean = st.number_input("Process Mean (X̄)", value=1.251, format="%.4f")
            sigma = st.number_input("Process Std Dev (σ)", value=0.00083, format="%.5f")
        
        with col2:
            if sigma > 0:
                cp = (usl - lsl) / (6 * sigma)
                cpu = (usl - mean) / (3 * sigma)
                cpl = (mean - lsl) / (3 * sigma)
                cpk = min(cpu, cpl)
                
                # Sigma level approximation
                sigma_level = cpk * 3
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Cp (Potential)", f"{cp:.2f}")
                    st.metric("Cpu (Upper)", f"{cpu:.2f}")
                with col_b:
                    st.metric("Cpk (Actual)", f"{cpk:.2f}")
                    st.metric("Cpl (Lower)", f"{cpl:.2f}")
                
                st.metric("Sigma Level", f"{sigma_level:.1f}σ")
                
                if cpk >= 1.67:
                    st.success("✅ Excellent: Process is highly capable")
                elif cpk >= 1.33:
                    st.success("✅ Good: Process is capable")
                elif cpk >= 1.0:
                    st.warning("⚠️ Marginal: Process needs monitoring")
                else:
                    st.error("❌ Not Capable: Immediate improvement needed")

# ============================================================
# MODULE 17: ACCEPTANCE SAMPLING (Chapter 13)
# ============================================================
def module_sampling():
    display_header("📊", "Chapter 13", "Acceptance Sampling", "Statistical methods for lot acceptance")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Acceptance Sampling Theory")
        st.write("""
        **Acceptance sampling** is performed on goods that already exist to determine what 
        percentage of products conform to specifications. It's used when 100% inspection 
        is impractical or destructive.
        """)
        
        st.markdown("#### Key Terms")
        st.write("""
        - **AQL (Acceptable Quality Level):** Maximum defect rate considered acceptable
        - **LTPD (Lot Tolerance Percent Defective):** Defect rate that should be rejected
        - **Producer's Risk (α):** Probability of rejecting a good lot (Type I error)
        - **Consumer's Risk (β):** Probability of accepting a bad lot (Type II error)
        """)
    
    with tab2:
        st.markdown("### Sampling Plan Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n = st.number_input("Sample Size (n)", value=50, min_value=1)
            c = st.number_input("Acceptance Number (c)", value=2, min_value=0)
            p = st.number_input("Lot Proportion Defective (p)", value=0.05, min_value=0.0, max_value=1.0, format="%.3f")
        
        with col2:
            # Calculate P(Accept) using Poisson approximation
            np_val = n * p
            p_accept = sum([poisson_pmf(k, np_val) for k in range(c + 1)])
            
            st.metric("P(Accept Lot)", f"{p_accept:.3f}")
            
            # Find approximate AQL and LTPD
            # AQL: p where P(accept) ≈ 0.95
            # LTPD: p where P(accept) ≈ 0.10
            
            st.info(f"""
            📊 **Sampling Plan (n={n}, c={c})**
            - At p = {p:.1%} defective, probability of acceptance = {p_accept:.1%}
            """)

# ============================================================
# MODULE 18: PARETO ANALYSIS (Chapter 13)
# ============================================================
def module_pareto():
    display_header("📊", "Chapter 13", "Pareto Analysis", "The 80/20 rule for quality improvement")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Simulator"])
    
    with tab1:
        st.markdown("### Pareto Principle")
        st.write("""
        The **Pareto Principle** (80/20 rule) states that roughly 80% of effects come from 
        20% of causes. In quality management, this means focusing on the "vital few" problems 
        that cause most defects.
        """)
        
        display_key_insight(
            "Application",
            "A Pareto chart displays categories in descending order of frequency with a cumulative "
            "line. Focus improvement efforts on the leftmost bars (highest frequency categories)."
        )
    
    with tab2:
        st.markdown("### Pareto Chart Builder")
        
        num_categories = st.number_input("Number of Categories", 3, 10, 5)
        
        categories = []
        for i in range(int(num_categories)):
            cols = st.columns(2)
            with cols[0]:
                name = st.text_input(f"Category {i+1}", value=f"Category {chr(65+i)}", key=f"par_name_{i}")
            with cols[1]:
                freq = st.number_input(f"Frequency {i+1}", value=50-i*8, min_value=0, key=f"par_freq_{i}")
            categories.append({"Category": name, "Frequency": freq})
        
        # Sort by frequency
        df = pd.DataFrame(categories)
        df = df.sort_values("Frequency", ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        total = df["Frequency"].sum()
        df["Percentage"] = df["Frequency"] / total * 100
        df["Cumulative %"] = df["Percentage"].cumsum()
        
        st.dataframe(df, use_container_width=True)
        
        # Identify vital few (80%)
        vital_few = df[df["Cumulative %"] <= 80]["Category"].tolist()
        if not vital_few:
            vital_few = [df.iloc[0]["Category"]]
        
        st.success(f"🎯 **Vital Few (80% of problems):** {', '.join(vital_few)}")

# ============================================================
# MODULE 19: FISHBONE DIAGRAM (Chapter 13)
# ============================================================
def module_fishbone():
    display_header("🐟", "Chapter 13", "Fishbone Diagram", "Cause-and-effect analysis (Ishikawa)")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Builder"])
    
    with tab1:
        st.markdown("### Cause-and-Effect Diagram")
        st.write("""
        The **Fishbone Diagram** (Ishikawa or Cause-and-Effect diagram) is a visual tool for 
        systematically identifying potential causes of a problem. It organizes causes into 
        major categories.
        """)
        
        st.markdown("#### The 6 M's (Manufacturing)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            - **Man** (People)
            - **Machine** (Equipment)
            - **Method** (Process)
            """)
        with col2:
            st.write("""
            - **Material** (Inputs)
            - **Measurement** (Inspection)
            - **Mother Nature** (Environment)
            """)
    
    with tab2:
        st.markdown("### Fishbone Builder")
        
        problem = st.text_input("Problem (Effect)", value="Product Defects")
        
        st.markdown("#### Enter Causes by Category")
        
        categories = ["Man", "Machine", "Method", "Material", "Measurement", "Environment"]
        causes = {}
        
        cols = st.columns(3)
        for i, cat in enumerate(categories):
            with cols[i % 3]:
                causes[cat] = st.text_area(f"{cat}", value=f"Cause 1\nCause 2", height=100, key=f"fb_{cat}")
        
        st.markdown("---")
        st.markdown(f"### Fishbone Diagram: {problem}")
        
        for cat in categories:
            cause_list = causes[cat].split('\n')
            st.markdown(f"**{cat}:** {', '.join([c for c in cause_list if c.strip()])}")

# ============================================================
# MODULE 20: LEAN SUPPLY CHAINS (Chapter 14)
# ============================================================
def module_lean():
    display_header("🔄", "Chapter 14", "Lean Supply Chains", "Eliminating waste and maximizing value")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Lead Time Calculator", "🗺️ Value Stream"])
    
    with tab1:
        st.markdown("### Lean Production Principles")
        st.write("""
        **Lean production** is an integrated set of activities designed to achieve high-volume 
        production using minimal inventories of raw materials, work-in-process, and finished goods.
        """)
        
        st.markdown("#### The 7 Wastes (TIMWOOD)")
        wastes = {
            "T": ("Transportation", "Unnecessary movement of materials"),
            "I": ("Inventory", "Excess stock beyond immediate needs"),
            "M": ("Motion", "Unnecessary movement of people"),
            "W": ("Waiting", "Idle time between operations"),
            "O": ("Overproduction", "Making more than needed"),
            "O": ("Overprocessing", "Doing more work than required"),
            "D": ("Defects", "Rework and scrap")
        }
        
        for letter, (name, desc) in wastes.items():
            st.write(f"**{letter} - {name}:** {desc}")
    
    with tab2:
        st.markdown("### Lead Time Reduction Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rm_days = st.number_input("RM Inventory (days)", value=10)
            wip_days = st.number_input("WIP Inventory (days)", value=14)
            fg_days = st.number_input("FG Inventory (days)", value=10)
            
            current_lt = rm_days + wip_days + fg_days
            st.metric("Current Lead Time", f"{current_lt} days")
        
        with col2:
            rm_red = st.slider("RM Reduction %", 0, 100, 85)
            wip_red = st.slider("WIP Reduction %", 0, 100, 85)
            fg_red = st.slider("FG Reduction %", 0, 100, 85)
            
            future_rm = rm_days * (1 - rm_red/100)
            future_wip = wip_days * (1 - wip_red/100)
            future_fg = fg_days * (1 - fg_red/100)
            future_lt = future_rm + future_wip + future_fg
            
            reduction_pct = (1 - future_lt/current_lt) * 100
            
            st.metric("Future Lead Time", f"{future_lt:.1f} days")
            st.metric("Lead Time Reduction", f"{reduction_pct:.0f}%")
    
    with tab3:
        st.markdown("### Value Stream Mapping")
        st.write("""
        A **Value Stream Map** visualizes the flow of materials and information from supplier 
        to customer. It identifies value-added vs. non-value-added activities.
        """)
        
        st.markdown("#### Process Flow")
        flow = ["Supplier", "→", "RM Inventory", "→", "Process 1", "→", "WIP", "→", "Process 2", "→", "FG", "→", "Customer"]
        st.write(" ".join(flow))

# ============================================================
# MODULE 21: CENTROID METHOD (Chapter 15)
# ============================================================
def module_centroid():
    display_header("📍", "Chapter 15", "Centroid Method", "Weighted center of gravity for facility location")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Weighted Center of Gravity")
        st.write("""
        The **Centroid Method** is a mathematical technique for finding the optimal location 
        for a single distribution center that will serve multiple destinations. It minimizes 
        the distance-weighted volume of shipments.
        """)
        
        display_citation(
            "The centroid method considers existing facilities, the distances between them, "
            "and the volumes of goods to be shipped.",
            "Jacobs & Chase (2024, p. 456)"
        )
        
        st.latex(r"C_x = \frac{\sum(d_{ix} \times V_i)}{\sum V_i}")
        st.latex(r"C_y = \frac{\sum(d_{iy} \times V_i)}{\sum V_i}")
    
    with tab2:
        st.markdown("### Centroid Calculator")
        
        num_locations = st.number_input("Number of Locations", 2, 10, 3)
        
        locations = []
        for i in range(int(num_locations)):
            cols = st.columns(4)
            with cols[0]:
                name = st.text_input(f"Location {i+1}", value=chr(65+i), key=f"cent_name_{i}")
            with cols[1]:
                x = st.number_input(f"X {i+1}", value=20+i*30, key=f"cent_x_{i}")
            with cols[2]:
                y = st.number_input(f"Y {i+1}", value=80-i*30, key=f"cent_y_{i}")
            with cols[3]:
                v = st.number_input(f"Volume {i+1}", value=1000-i*200, key=f"cent_v_{i}")
            locations.append({"name": name, "x": x, "y": y, "v": v})
        
        total_v = sum(loc["v"] for loc in locations)
        cx = sum(loc["x"] * loc["v"] for loc in locations) / total_v
        cy = sum(loc["y"] * loc["v"] for loc in locations) / total_v
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Optimal X (Cx)", f"{cx:.1f}")
        with col2:
            st.metric("Optimal Y (Cy)", f"{cy:.1f}")

# ============================================================
# MODULE 22: FACTOR RATING (Chapter 15)
# ============================================================
def module_factor():
    display_header("⚖️", "Chapter 15", "Factor Rating Method", "Multi-criteria location analysis")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Factor Rating Method")
        st.write("""
        The **Factor Rating Method** assigns weights to relevant factors and scores each 
        potential location. The location with the highest weighted score is preferred.
        """)
        
        st.latex(r"\text{Score} = \sum_{i=1}^{n} (w_i \times s_i)")
    
    with tab2:
        st.markdown("### Factor Rating Calculator")
        
        factors = ["Labor Cost", "Proximity to Market", "Tax Environment", "Infrastructure"]
        
        st.markdown("#### Enter Weights and Scores")
        
        data = []
        for i, factor in enumerate(factors):
            cols = st.columns(4)
            with cols[0]:
                st.write(factor)
            with cols[1]:
                w = st.number_input(f"Weight {i}", value=0.25, min_value=0.0, max_value=1.0, key=f"fr_w_{i}")
            with cols[2]:
                a = st.number_input(f"Loc A {i}", value=70+i*5, min_value=0, max_value=100, key=f"fr_a_{i}")
            with cols[3]:
                b = st.number_input(f"Loc B {i}", value=75-i*3, min_value=0, max_value=100, key=f"fr_b_{i}")
            data.append({"factor": factor, "weight": w, "a": a, "b": b})
        
        total_w = sum(d["weight"] for d in data)
        score_a = sum(d["weight"] * d["a"] for d in data)
        score_b = sum(d["weight"] * d["b"] for d in data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Weight", f"{total_w:.2f}")
        with col2:
            st.metric("Location A Score", f"{score_a:.1f}")
        with col3:
            st.metric("Location B Score", f"{score_b:.1f}")
        
        if score_a > score_b:
            st.success("📍 **Recommendation:** Location A is preferred")
        else:
            st.success("📍 **Recommendation:** Location B is preferred")

# ============================================================
# MODULE 23: TRANSPORTATION METHOD (Chapter 15)
# ============================================================
def module_transportation():
    display_header("🚚", "Chapter 15", "Transportation Method", "Optimal allocation of supply to demand")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Transportation Problem")
        st.write("""
        The **transportation method** finds the minimum-cost allocation of goods from multiple 
        sources to multiple destinations. It can be solved using the Northwest Corner Method 
        for an initial solution, then optimized.
        """)
        
        st.latex(r"\text{Minimize } Z = \sum_{i}\sum_{j} c_{ij} \cdot x_{ij}")
        
        st.write("""
        Subject to:
        - Supply constraints: Σⱼ xᵢⱼ = Sᵢ
        - Demand constraints: Σᵢ xᵢⱼ = Dⱼ
        - Non-negativity: xᵢⱼ ≥ 0
        """)
    
    with tab2:
        st.markdown("### Transportation Tableau (3×3)")
        
        st.markdown("#### Unit Costs")
        costs = []
        for i in range(3):
            cols = st.columns(4)
            with cols[0]:
                st.write(f"Source {chr(65+i)}")
            row = []
            for j in range(3):
                with cols[j+1]:
                    c = st.number_input(f"C{i}{j}", value=25+i*10+j*5, key=f"tr_c_{i}_{j}", label_visibility="collapsed")
                    row.append(c)
            costs.append(row)
        
        st.markdown("#### Allocations")
        allocs = []
        for i in range(3):
            cols = st.columns(4)
            with cols[0]:
                st.write(f"Source {chr(65+i)}")
            row = []
            for j in range(3):
                with cols[j+1]:
                    a = st.number_input(f"A{i}{j}", value=0, key=f"tr_a_{i}_{j}", label_visibility="collapsed")
                    row.append(a)
            allocs.append(row)
        
        # Calculate total cost
        total_cost = sum(costs[i][j] * allocs[i][j] for i in range(3) for j in range(3))
        st.metric("Total Transportation Cost", f"${total_cost:,}")

# ============================================================
# MODULE 24: GLOBAL SOURCING (Chapter 16)
# ============================================================
def module_sourcing():
    display_header("🌐", "Chapter 16", "Global Sourcing", "Total cost of ownership analysis")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 TCO Calculator"])
    
    with tab1:
        st.markdown("### Strategic Sourcing")
        st.write("""
        **Strategic sourcing** is the development of supply channels at the lowest total cost, 
        not just the lowest purchase price. Total Cost of Ownership (TCO) includes all costs 
        associated with acquiring and using a product.
        """)
        
        st.markdown("#### Functional vs. Innovative Products")
        df = pd.DataFrame({
            "Characteristic": ["Demand", "Life Cycle", "Margin", "Focus"],
            "Functional": ["Predictable", "Long", "Low", "Efficiency"],
            "Innovative": ["Unpredictable", "Short", "High", "Responsiveness"]
        })
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### Total Cost of Ownership Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Domestic Supplier")
            dom_price = st.number_input("Unit Price ($)", value=100, key="tco_dom_p")
            dom_ship = st.number_input("Shipping ($)", value=5, key="tco_dom_s")
            dom_other = st.number_input("Other Costs ($)", value=0, key="tco_dom_o")
            dom_total = dom_price + dom_ship + dom_other
            st.metric("Total Cost", f"${dom_total:.2f}")
        
        with col2:
            st.markdown("#### Overseas Supplier")
            ovs_price = st.number_input("Unit Price ($)", value=75, key="tco_ovs_p")
            ovs_ship = st.number_input("Shipping ($)", value=15, key="tco_ovs_s")
            ovs_other = st.number_input("Other Costs ($)", value=5, key="tco_ovs_o")
            ovs_total = ovs_price + ovs_ship + ovs_other
            st.metric("Total Cost", f"${ovs_total:.2f}")
        
        if dom_total < ovs_total:
            st.success(f"📍 **Recommendation:** Domestic supplier (saves ${ovs_total - dom_total:.2f}/unit)")
        else:
            st.success(f"📍 **Recommendation:** Overseas supplier (saves ${dom_total - ovs_total:.2f}/unit)")

# ============================================================
# MODULE 25: FORECASTING (Chapter 18)
# ============================================================
def module_forecast():
    display_header("📈", "Chapter 18", "Enhanced Forecasting", "WMA, Holt's, Seasonal & Tracking Signal")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Weighted MA", "📈 Holt's Method", "🌊 Seasonal Index", "📡 Tracking Signal"])
    
    with tab1:
        st.markdown("### Weighted Moving Average")
        st.latex(r"F_t = \frac{\sum_{i=1}^{n} w_i \cdot A_{t-i}}{\sum_{i=1}^{n} w_i}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            w1 = st.number_input("Weight t-1 (most recent)", value=0.5, key="wma_w1")
            w2 = st.number_input("Weight t-2", value=0.3, key="wma_w2")
            w3 = st.number_input("Weight t-3", value=0.2, key="wma_w3")
            
            d1 = st.number_input("Demand t-1", value=120, key="wma_d1")
            d2 = st.number_input("Demand t-2", value=110, key="wma_d2")
            d3 = st.number_input("Demand t-3", value=130, key="wma_d3")
        
        with col2:
            total_w = w1 + w2 + w3
            wma = (w1*d1 + w2*d2 + w3*d3) / total_w if total_w > 0 else 0
            st.metric("WMA Forecast", f"{wma:.1f}")
    
    with tab2:
        st.markdown("### Holt's Trend-Adjusted Exponential Smoothing")
        st.latex(r"L_t = \alpha A_t + (1-\alpha)(L_{t-1} + T_{t-1})")
        st.latex(r"T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}")
        st.latex(r"F_{t+m} = L_t + m \cdot T_t")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alpha = st.slider("Alpha (α)", 0.05, 0.95, 0.3, 0.05)
            beta = st.slider("Beta (β)", 0.05, 0.95, 0.2, 0.05)
            L0 = st.number_input("Initial Level (L₀)", value=100)
            T0 = st.number_input("Initial Trend (T₀)", value=10)
            At = st.number_input("Latest Actual (Aₜ)", value=125)
        
        with col2:
            Lt = alpha * At + (1 - alpha) * (L0 + T0)
            Tt = beta * (Lt - L0) + (1 - beta) * T0
            forecast = Lt + Tt
            
            st.metric("New Level (Lₜ)", f"{Lt:.1f}")
            st.metric("New Trend (Tₜ)", f"{Tt:.1f}")
            st.metric("Forecast (t+1)", f"{forecast:.1f}")
    
    with tab3:
        st.markdown("### Seasonal Index Calculator")
        st.latex(r"SI_i = \frac{\text{Average demand in season } i}{\text{Overall average demand}}")
        
        st.markdown("Enter quarterly demand for 3 years:")
        
        data = []
        for q in range(4):
            cols = st.columns(5)
            with cols[0]:
                st.write(f"Q{q+1}")
            year_data = []
            for y in range(3):
                with cols[y+1]:
                    d = st.number_input(f"Y{y+1}Q{q+1}", value=80+q*20+y*5, key=f"si_{q}_{y}", label_visibility="collapsed")
                    year_data.append(d)
            avg = sum(year_data) / 3
            data.append({"Quarter": f"Q{q+1}", "Y1": year_data[0], "Y2": year_data[1], "Y3": year_data[2], "Avg": avg})
        
        df = pd.DataFrame(data)
        overall_avg = df["Avg"].mean()
        df["Seasonal Index"] = df["Avg"] / overall_avg
        
        st.dataframe(df, use_container_width=True)
        st.metric("Overall Average", f"{overall_avg:.1f}")
    
    with tab4:
        st.markdown("### Tracking Signal Monitor")
        st.latex(r"TS = \frac{RSFE}{MAD}")
        
        st.write("""
        - **RSFE** = Running Sum of Forecast Errors = Σ(Aₜ - Fₜ)
        - **MAD** = Mean Absolute Deviation
        - If |TS| > 4, the forecast model may need adjustment
        """)
        
        # Example data
        periods = 6
        actuals = [100, 110, 105, 115, 120, 125]
        forecasts = [95, 105, 108, 110, 115, 118]
        
        errors = [a - f for a, f in zip(actuals, forecasts)]
        abs_errors = [abs(e) for e in errors]
        rsfe = sum(errors)
        mad = sum(abs_errors) / len(abs_errors)
        ts = rsfe / mad if mad > 0 else 0
        
        df = pd.DataFrame({
            "Period": list(range(1, periods+1)),
            "Actual": actuals,
            "Forecast": forecasts,
            "Error": errors,
            "|Error|": abs_errors
        })
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAD", f"{mad:.2f}")
        with col2:
            st.metric("RSFE", f"{rsfe:.0f}")
        with col3:
            st.metric("Tracking Signal", f"{ts:.2f}")
        
        if abs(ts) > 4:
            st.error("⚠️ Tracking signal exceeds ±4. Review forecast model.")
        else:
            st.success("✅ Tracking signal within acceptable range.")

# ============================================================
# MODULE 26: REGRESSION (Chapter 18)
# ============================================================
def module_regression():
    display_header("📈", "Chapter 18", "Linear Regression Trend Line", "Least squares forecasting")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Linear Regression Theory")
        st.write("""
        **Linear regression** finds the best-fitting straight line through data points. 
        It's used for trend projection and causal forecasting.
        """)
        
        st.latex(r"Y = a + bx")
        st.latex(r"b = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
        st.latex(r"a = \bar{y} - b\bar{x}")
    
    with tab2:
        st.markdown("### Regression Calculator")
        
        st.markdown("Enter 12 periods of data:")
        
        x_vals = list(range(1, 13))
        y_vals = []
        
        cols = st.columns(6)
        for i in range(12):
            with cols[i % 6]:
                y = st.number_input(f"Period {i+1}", value=100+i*5+np.random.randint(-5, 5), key=f"reg_y_{i}")
                y_vals.append(y)
        
        # Calculate regression
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x**2 for x in x_vals)
        
        b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        a = (sum_y - b * sum_x) / n
        
        st.metric("Regression Equation", f"Y = {a:.2f} + {b:.2f}x")
        
        # Forecast
        forecast_x = st.number_input("Forecast for Period", value=13, min_value=1)
        forecast_y = a + b * forecast_x
        st.metric(f"Forecast for Period {forecast_x}", f"{forecast_y:.1f}")

# ============================================================
# MODULE 27: AGGREGATE PLANNING (Chapter 19)
# ============================================================
def module_aggregate():
    display_header("📋", "Chapter 19", "Aggregate Planning & S&OP", "Balancing supply and demand")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Simulator"])
    
    with tab1:
        st.markdown("### Sales & Operations Planning")
        
        display_citation(
            "Sales and operations planning was coined by companies to refer to the process that helps "
            "firms keep demand and supply in balance. In operations management, this process traditionally "
            "was called aggregate planning.",
            "Jacobs & Chase (2024)"
        )
        
        st.markdown("#### Pure Planning Strategies")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🏃 Chase Strategy**")
            st.write("Match production to demand by hiring/firing. Low inventory, high workforce instability.")
        with col2:
            st.markdown("**📏 Level Strategy**")
            st.write("Constant workforce and production. Inventory absorbs demand fluctuations.")
        with col3:
            st.markdown("**🔀 Mixed Strategy**")
            st.write("Combination using overtime, subcontracting, or part-time workers.")
    
    with tab2:
        st.markdown("### Aggregate Planning Calculator")
        
        st.markdown("#### Quarterly Demand Forecast")
        cols = st.columns(4)
        demands = []
        for i in range(4):
            with cols[i]:
                d = st.number_input(f"Q{i+1} Demand", value=1200+i*200, key=f"ag_d_{i}")
                demands.append(d)
        
        st.markdown("#### Cost Parameters")
        col1, col2 = st.columns(2)
        with col1:
            reg_cost = st.number_input("Regular Cost/Unit ($)", value=40)
            hold_cost = st.number_input("Holding Cost/Unit ($)", value=5)
        with col2:
            hire_fire = st.number_input("Hire/Fire Cost/Worker ($)", value=200)
            units_per_worker = st.number_input("Units per Worker/Quarter", value=10)
        
        total_demand = sum(demands)
        avg_demand = total_demand / 4
        
        # Chase Strategy
        chase_cost = sum(demands) * reg_cost
        workers_needed = [d / units_per_worker for d in demands]
        hf_changes = sum(abs(workers_needed[i] - workers_needed[i-1]) for i in range(1, 4))
        chase_hf_cost = hf_changes * hire_fire
        chase_total = chase_cost + chase_hf_cost
        
        # Level Strategy
        level_prod = avg_demand
        level_workers = level_prod / units_per_worker
        level_cost = total_demand * reg_cost
        
        # Calculate inventory
        inventory = []
        inv = 0
        for d in demands:
            inv = inv + level_prod - d
            inventory.append(max(0, inv))
        level_hold_cost = sum(inventory) * hold_cost
        level_total = level_cost + level_hold_cost
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏃 Chase Strategy")
            st.metric("Total Cost", f"${chase_total:,.0f}")
        with col2:
            st.markdown("#### 📏 Level Strategy")
            st.metric("Total Cost", f"${level_total:,.0f}")
        
        if chase_total < level_total:
            st.success(f"📊 **Recommendation:** Chase Strategy saves ${level_total - chase_total:,.0f}")
        else:
            st.success(f"📊 **Recommendation:** Level Strategy saves ${chase_total - level_total:,.0f}")

# ============================================================
# MODULE 28: EOQ (Chapter 20)
# ============================================================
def module_eoq():
    display_header("📦", "Chapter 20", "Economic Order Quantity (EOQ)", "Minimizing total inventory costs")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Simulator", "⚙️ EPQ", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Economic Order Quantity Model")
        
        display_citation(
            "The EOQ model attempts to find the order quantity that minimizes total annual holding "
            "and ordering costs. At the optimal order quantity Q*, the annual ordering cost exactly "
            "equals the annual holding cost.",
            "Jacobs & Chase (2024, p. 598)"
        )
        
        st.latex(r"TC = \frac{D}{Q} \cdot S + \frac{Q}{2} \cdot H")
        st.latex(r"Q^* = \sqrt{\frac{2DS}{H}}")
        st.latex(r"TC_{min} = \sqrt{2DSH}")
        
        display_key_insight(
            "Robustness of EOQ",
            "The total cost curve is relatively flat around Q*. Even if Q deviates by ±25%, "
            "total cost increases by less than 3%."
        )
    
    with tab2:
        st.markdown("### EOQ Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            D = st.slider("Annual Demand (D)", 100, 10000, 1000, 100)
            S = st.slider("Order Cost (S) $", 5, 500, 50, 5)
            H = st.slider("Holding Cost (H) $/unit/yr", 1, 50, 2)
        
        with col2:
            Q_star = math.sqrt(2 * D * S / H)
            TC = (D / Q_star) * S + (Q_star / 2) * H
            orders_per_year = D / Q_star
            
            st.metric("Optimal Q*", f"{Q_star:.0f} units")
            st.metric("Total Cost", f"${TC:.0f}")
            st.metric("Orders/Year", f"{orders_per_year:.1f}")
            
            st.latex(rf"Q^* = \sqrt{{\frac{{2 \times {D} \times {S}}}{{{H}}}}} = {Q_star:.0f}")
    
    with tab3:
        st.markdown("### Production Order Quantity (EPQ)")
        st.latex(r"Q^*_p = \sqrt{\frac{2DS}{H(1-d/p)}}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            p = st.number_input("Production Rate (p)", value=1000)
            d = st.number_input("Demand Rate (d)", value=200)
        
        with col2:
            if p > d:
                Q_epq = math.sqrt(2 * D * S / (H * (1 - d/p)))
                st.metric("Optimal Production Lot", f"{Q_epq:.0f}")
            else:
                st.error("Production rate must exceed demand rate!")
    
    with tab4:
        st.markdown("### Practice Problems")
        
        with st.expander("Problem: Calculate Q*"):
            st.write("D = 5,000, S = $100, H = $4. Calculate optimal Q*.")
            user_ans = st.number_input("Your Answer:", key="eoq_p1")
            if st.button("Check Answer", key="eoq_p1_btn"):
                correct = math.sqrt(2 * 5000 * 100 / 4)
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Q* = √(2×5000×100/4) = {correct:.0f}")
                else:
                    st.error(f"❌ Incorrect. Q* = √(2×5000×100/4) = {correct:.0f}")

# ============================================================
# MODULE 29: SAFETY STOCK (Chapter 20)
# ============================================================
def module_safetystock():
    display_header("🛡️", "Chapter 20", "Safety Stock & Reorder Point", 
                   "Protecting against demand and lead time variability")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Safety Stock Theory")
        
        st.latex(r"SS = z \cdot \sigma_d \cdot \sqrt{LT}")
        st.latex(r"ROP = \bar{d} \times LT + SS")
        
        st.markdown("#### Common Z-Values")
        df = pd.DataFrame({
            "Service Level": ["90%", "95%", "97.5%", "99%", "99.9%"],
            "z-value": [1.28, 1.65, 1.96, 2.33, 3.09]
        })
        st.dataframe(df, use_container_width=True)
        
        display_key_insight(
            "Service Level Trade-off",
            "Increasing service level from 95% to 99% requires 41% more safety stock "
            "(z increases from 1.65 to 2.33)."
        )
    
    with tab2:
        st.markdown("### Safety Stock Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            d_avg = st.slider("Average Daily Demand", 10, 500, 100)
            sigma_d = st.slider("Demand Std Dev (σd)", 1, 100, 20)
            lt = st.slider("Lead Time (days)", 1, 30, 7)
            
            service_level = st.selectbox("Service Level", ["90%", "95%", "99%"])
            z_values = {"90%": 1.28, "95%": 1.65, "99%": 2.33}
            z = z_values[service_level]
        
        with col2:
            sigma_lt = sigma_d * math.sqrt(lt)
            ss = z * sigma_lt
            rop = d_avg * lt + ss
            
            st.metric("Safety Stock", f"{ss:.0f} units")
            st.metric("Reorder Point", f"{rop:.0f} units")
            
            st.latex(rf"SS = {z} \times {sigma_d} \times \sqrt{{{lt}}} = {ss:.0f}")
            st.latex(rf"ROP = {d_avg} \times {lt} + {ss:.0f} = {rop:.0f}")

# ============================================================
# MODULE 30: NEWSVENDOR (Chapter 20)
# ============================================================
def module_newsvendor():
    display_header("📰", "Chapter 20", "Newsvendor Model", "Single-period inventory optimization")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Newsvendor Model Theory")
        st.write("""
        The **Newsvendor Model** is used for perishable or seasonal items with a single 
        ordering opportunity. It balances the cost of understocking against overstocking.
        """)
        
        st.latex(r"C_u = \text{Price} - \text{Cost}")
        st.latex(r"C_o = \text{Cost} - \text{Salvage}")
        st.latex(r"P \leq \frac{C_u}{C_u + C_o}")
        st.latex(r"Q^* = \mu + z\sigma")
    
    with tab2:
        st.markdown("### Newsvendor Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            price = st.number_input("Selling Price ($)", value=100)
            cost = st.number_input("Unit Cost ($)", value=70)
            salvage = st.number_input("Salvage Value ($)", value=20)
            mu = st.number_input("Mean Demand (μ)", value=200)
            sigma = st.number_input("Demand Std Dev (σ)", value=40)
        
        with col2:
            Cu = price - cost
            Co = cost - salvage
            critical_ratio = Cu / (Cu + Co)
            z = normal_ppf(critical_ratio)
            Q_star = mu + z * sigma
            
            st.metric("Cost of Understocking (Cu)", f"${Cu}")
            st.metric("Cost of Overstocking (Co)", f"${Co}")
            st.metric("Critical Ratio", f"{critical_ratio:.3f}")
            st.metric("Z-Score", f"{z:.2f}")
            st.metric("Optimal Q*", f"{Q_star:.0f} units")

# ============================================================
# MODULE 31: MRP (Chapter 21)
# ============================================================
def module_mrp():
    display_header("🏭", "Chapter 21", "Material Requirements Planning", "Scheduling dependent-demand inventory")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 MRP Matrix", "🌳 BOM Explorer"])
    
    with tab1:
        st.markdown("### MRP System Structure")
        
        display_citation(
            "The MRP system uses three sources of information: 1. Demand comes from the master schedule. "
            "2. The bill-of-materials identifies exactly what is needed to make each end item. "
            "3. The current inventory status of the items managed by the system.",
            "Jacobs & Chase (2024)"
        )
        
        st.latex(r"\text{Net} = \text{Gross} - (\text{On Hand} + \text{Scheduled Receipts})")
        st.latex(r"\text{Release Period} = \text{Receipt Period} - \text{Lead Time}")
        
        st.markdown("#### MRP Inputs")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📅 Master Schedule**")
            st.write("What to make and when")
        with col2:
            st.markdown("**🌳 Bill of Materials**")
            st.write("Product structure")
        with col3:
            st.markdown("**📦 Inventory Records**")
            st.write("On-hand and scheduled receipts")
    
    with tab2:
        st.markdown("### MRP Record Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            lead_time = st.number_input("Lead Time (weeks)", value=1, min_value=1, max_value=4)
            beg_inv = st.number_input("Beginning Inventory", value=20)
            lot_rule = st.selectbox("Lot Sizing Rule", ["Lot-for-Lot (L4L)", "Fixed Order Qty (FOQ)"])
            if lot_rule == "Fixed Order Qty (FOQ)":
                lot_size = st.number_input("Lot Size", value=50)
            else:
                lot_size = 0
        
        st.markdown("#### Gross Requirements (6 weeks)")
        gross = []
        cols = st.columns(6)
        for i in range(6):
            with cols[i]:
                g = st.number_input(f"Wk {i+1}", value=[50, 0, 100, 0, 150, 0][i], key=f"mrp_g_{i}")
                gross.append(g)
        
        # Calculate MRP
        on_hand = [beg_inv]
        net_req = []
        planned_receipt = []
        planned_release = [0] * 6
        
        for i in range(6):
            # Net requirement
            net = gross[i] - on_hand[-1]
            net_req.append(max(0, net))
            
            # Planned receipt
            if net > 0:
                if lot_rule == "Lot-for-Lot (L4L)":
                    receipt = net
                else:
                    receipt = math.ceil(net / lot_size) * lot_size
            else:
                receipt = 0
            planned_receipt.append(receipt)
            
            # Update on-hand
            new_oh = on_hand[-1] - gross[i] + receipt
            on_hand.append(max(0, new_oh))
            
            # Planned release (offset by lead time)
            if i + lead_time < 6 and receipt > 0:
                release_period = i - lead_time
                if release_period >= 0:
                    planned_release[release_period] = receipt
        
        # Display MRP table
        df = pd.DataFrame({
            "Week": list(range(1, 7)),
            "Gross Req": gross,
            "On-Hand": on_hand[1:],
            "Net Req": net_req,
            "Planned Receipt": planned_receipt,
            "Planned Release": planned_release
        })
        st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.markdown("### Bill of Materials Explorer")
        
        st.markdown("""
        ```
        (Level 0) Product A
        ├── (Level 1) Sub-B (×2)
        │   └── (Level 2) Part-D (×4)
        └── (Level 1) Sub-C (×1)
            └── (Level 2) Part-E (×2)
        ```
        """)
        
        end_qty = st.number_input("End Items Required", value=100)
        
        st.markdown("#### Component Requirements")
        st.write(f"- Product A: {end_qty}")
        st.write(f"- Sub-B: {end_qty * 2}")
        st.write(f"- Sub-C: {end_qty * 1}")
        st.write(f"- Part-D: {end_qty * 2 * 4}")
        st.write(f"- Part-E: {end_qty * 1 * 2}")

# ============================================================
# MODULE 32: MRP LOT SIZING (Chapter 21)
# ============================================================
def module_mrp_lotsizing():
    display_header("📦", "Chapter 21", "MRP Lot Sizing Comparison", "L4L, EOQ, and POQ techniques")
    
    st.markdown("### Lot Sizing Methods")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Lot-for-Lot (L4L)**")
        st.write("Order exactly what's needed. Minimizes holding cost, maximizes setup cost.")
    with col2:
        st.markdown("**EOQ**")
        st.write("Fixed quantity based on EOQ formula. May not align with requirements.")
    with col3:
        st.markdown("**POQ**")
        st.write("Order for fixed number of periods. T = EOQ / Avg Demand.")
    
    st.markdown("### Case Study (Exhibit 21.16)")
    st.write("Parameters: S = $47, H = $2/unit/week")
    
    requirements = [105, 80, 130, 50, 0, 200, 125, 100]
    
    df = pd.DataFrame({
        "Week": list(range(1, 9)),
        "Net Requirements": requirements
    })
    st.dataframe(df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("L4L Total Cost", "$329")
    with col2:
        st.metric("EOQ Total Cost", "$284")
    with col3:
        st.metric("POQ Total Cost", "$261")
    
    st.success("**Result:** POQ method is most cost-effective for this demand pattern.")

# ============================================================
# MODULE 33: JOB SCHEDULING (Chapter 22)
# ============================================================
def module_scheduling():
    display_header("📅", "Chapter 22", "Job Sequencing & Priority Rules", "Determining job processing order")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Priority Sequencing Rules")
        
        display_citation(
            "Priority rules are simple heuristics used to select the order in which jobs will be "
            "processed. They are especially important in job shop environments where different jobs "
            "compete for the same resources.",
            "Jacobs & Chase (2024, p. 672)"
        )
        
        st.markdown("#### Common Rules")
        rules = {
            "FCFS": "First Come, First Served - Process in order of arrival",
            "SPT": "Shortest Processing Time - Minimizes average flow time",
            "EDD": "Earliest Due Date - Minimizes maximum tardiness",
            "CR": "Critical Ratio - (Due Date - Today) / Processing Time"
        }
        for rule, desc in rules.items():
            st.write(f"**{rule}:** {desc}")
        
        st.latex(r"CR = \frac{\text{Due Date} - \text{Today}}{\text{Processing Time}}")
        
        display_key_insight(
            "SPT Optimality",
            "SPT is provably optimal for minimizing average flow time, average WIP, and average lateness."
        )
    
    with tab2:
        st.markdown("### Job Scheduling Simulator")
        
        rule = st.selectbox("Select Priority Rule", ["FCFS", "SPT", "EDD", "CR"])
        today = st.number_input("Current Day", value=0, min_value=0)
        
        st.markdown("#### Job Data")
        jobs = []
        for i, name in enumerate(["A", "B", "C", "D"]):
            cols = st.columns(3)
            with cols[0]:
                st.write(f"Job {name}")
            with cols[1]:
                pt = st.number_input(f"Proc Time {name}", value=3+i, key=f"sch_pt_{i}")
            with cols[2]:
                dd = st.number_input(f"Due Date {name}", value=5+i*2, key=f"sch_dd_{i}")
            jobs.append({"name": name, "pt": pt, "dd": dd})
        
        # Sort by rule
        if rule == "FCFS":
            sorted_jobs = jobs
        elif rule == "SPT":
            sorted_jobs = sorted(jobs, key=lambda x: x["pt"])
        elif rule == "EDD":
            sorted_jobs = sorted(jobs, key=lambda x: x["dd"])
        elif rule == "CR":
            for j in jobs:
                j["cr"] = (j["dd"] - today) / j["pt"] if j["pt"] > 0 else float('inf')
            sorted_jobs = sorted(jobs, key=lambda x: x["cr"])
        
        # Calculate metrics
        flow_times = []
        tardiness = []
        current_time = 0
        
        results = []
        for job in sorted_jobs:
            current_time += job["pt"]
            flow = current_time
            tardy = max(0, current_time - job["dd"])
            flow_times.append(flow)
            tardiness.append(tardy)
            results.append({
                "Job": job["name"],
                "Proc Time": job["pt"],
                "Flow Time": flow,
                "Due Date": job["dd"],
                "Tardiness": tardy
            })
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Flow Time", f"{sum(flow_times)/len(flow_times):.2f}")
        with col2:
            st.metric("Avg Tardiness", f"{sum(tardiness)/len(tardiness):.2f}")
    
    with tab3:
        st.markdown("### Practice Problems")
        
        with st.expander("Problem: Which rule minimizes average flow time?"):
            if st.button("Show Answer", key="sch_p1"):
                st.success("**SPT (Shortest Processing Time)** minimizes average flow time.")

# ============================================================
# MODULE 34: POKA-YOKE (Chapter 9)
# ============================================================
def module_pokayoke():
    display_header("🛡️", "Chapter 9", "Poka-yoke Database", "Mistake-proofing techniques")
    
    st.markdown("### Poka-yoke Examples")
    
    examples = [
        ("Bill is illegible", "Top copy given to customer (ensures carbon copy legibility)"),
        ("Feedback not obtained", "Postcard included with bill"),
        ("Wrong medication", "Bar-code scanning of patient and medicine"),
        ("Surgical error", "Pre-surgery checklist"),
        ("Card left in ATM", "Beep until card removed"),
        ("Wrong part assembled", "Parts bin with sensors"),
        ("Missing component", "Shadow board for tools"),
        ("Incorrect data entry", "Input validation and dropdown menus")
    ]
    
    df = pd.DataFrame(examples, columns=["Failure Mode", "Poka-yoke Solution"])
    st.dataframe(df, use_container_width=True)
    
    st.markdown("### Types of Poka-yoke")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prevention**")
        st.write("Makes errors impossible (e.g., USB connector shape)")
    with col2:
        st.markdown("**Detection**")
        st.write("Alerts when error occurs (e.g., car door ajar warning)")

# ============================================================
# MODULE 35: SQC PRACTICE (Chapter 13)
# ============================================================
def module_sqc_practice():
    display_header("🎓", "Chapter 13", "SQC Practice Questions", "Statistical Quality Control exam prep")
    
    questions = [
        ("A Six Sigma process running at center expects how many DPMO?", "3.4 defects per million opportunities"),
        ("What does DMAIC stand for?", "Define, Measure, Analyze, Improve, Control"),
        ("A z-value of 3 gives what percent confidence?", "99.7% confidence (3-sigma limits)"),
        ("What is Cpk if USL=1.255, LSL=1.245, mean=1.251, σ=0.00083?", "Cpk = 1.6 (min of Cpu and Cpl)"),
        ("Washers: mean=2.0mm, σ=0.2mm. What fraction > 2.4mm?", "Z = 2.0, P(Z>2) = 2.28%")
    ]
    
    for i, (q, a) in enumerate(questions):
        with st.expander(f"Q{i+1}: {q}"):
            if st.button(f"Show Answer", key=f"sqc_prac_{i}"):
                st.success(a)

# ============================================================
# MODULE 1: SUPPLY CHAIN RISK (Chapter 1)
# ============================================================
def module_risk():
    display_header("🛡️", "Chapter 1", "Supply Chain Risk Assessment", 
                   "Probability and Impact Matrix (Exhibit 1.4)")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### Supply Chain Risk Management")
        
        st.write("""
        **Supply chain risk** is the likelihood of a disruption that would impact the ability of a 
        company to continuously supply products or services. Effective risk management involves 
        systematic identification, assessment, and mitigation of potential threats.
        """)
        
        display_citation(
            "Supply chain risk management involves the identification of potential sources of risk "
            "and implementation of appropriate strategies through a coordinated approach among "
            "supply chain members to reduce supply chain vulnerability.",
            "Jacobs & Chase (2024, p. 12)"
        )
        
        st.markdown("#### Risk Assessment Framework")
        st.latex(r"\text{Risk Score} = \text{Probability} \times \text{Impact}")
        
        st.write("""
        Each risk event is scored on a scale (typically 1-5 or 1-10). Events with high scores 
        require immediate mitigation strategies such as:
        - **Redundancy** - Multiple suppliers, backup facilities
        - **Insurance** - Financial protection against losses
        - **Process Changes** - Redesigning vulnerable processes
        - **Inventory Buffers** - Safety stock for critical items
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("⚠️", "Risk Identification", 
                "Systematically identify all potential sources of supply chain disruption")
        with col2:
            display_concept_card("📊", "Risk Assessment", 
                "Evaluate probability and impact of each identified risk")
        
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("🛡️", "Risk Mitigation", 
                "Develop and implement strategies to reduce risk exposure")
        with col2:
            display_concept_card("📈", "Risk Monitoring", 
                "Continuously track risk indicators and update assessments")
        
        display_key_insight(
            "What-If Analysis",
            "Some companies call this 'what if' analysis. Answering these 'what if' questions can be "
            "useful for understanding how sensitive an analysis is to cost and profit assumptions. "
            "Consider scenarios like: 25% increase in development time, 25% change in sales volume, "
            "$1 change in price or cost. (Jacobs & Chase, 2024, p. 60)"
        )
        
        st.markdown("#### Common Supply Chain Risk Categories")
        
        risk_categories = pd.DataFrame({
            "Category": ["Operational", "Financial", "Strategic", "Hazard", "Demand", "Supply"],
            "Examples": [
                "Equipment failure, quality issues, capacity constraints",
                "Currency fluctuation, supplier bankruptcy, credit risk",
                "Competitor actions, market changes, technology shifts",
                "Natural disasters, accidents, terrorism",
                "Forecast errors, demand volatility, bullwhip effect",
                "Supplier failure, logistics disruption, material shortage"
            ],
            "Mitigation": [
                "Preventive maintenance, quality systems, flexible capacity",
                "Hedging, supplier financial monitoring, diversification",
                "Market intelligence, scenario planning, agility",
                "Insurance, business continuity planning, geographic spread",
                "Demand sensing, collaborative forecasting, postponement",
                "Multi-sourcing, safety stock, supplier development"
            ]
        })
        st.dataframe(risk_categories, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Risk Assessment Matrix Calculator")
        st.write("Score each risk event from 1 (Low) to 5 (High) for both Probability and Impact")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Risk Event Scoring")
            
            risk_names = [
                "Supplier Failure (Financial)",
                "Natural Disaster / Weather",
                "Quality Issue / Product Recall",
                "Logistics / Customs Delay",
                "Demand Volatility",
                "Cybersecurity Breach",
                "Regulatory Changes",
                "Key Personnel Loss"
            ]
            
            risks = []
            for i, name in enumerate(risk_names):
                with st.expander(f"📌 {name}", expanded=(i < 3)):
                    c1, c2 = st.columns(2)
                    with c1:
                        prob = st.slider(f"Probability", 1, 5, 3, key=f"risk_p_{i}",
                                        help="1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain")
                    with c2:
                        impact = st.slider(f"Impact", 1, 5, 4, key=f"risk_i_{i}",
                                          help="1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Catastrophic")
                    risks.append({"name": name, "prob": prob, "impact": impact, "score": prob * impact})
        
        with col2:
            st.markdown("#### Risk Analysis Results")
            
            df = pd.DataFrame(risks)
            df.columns = ["Risk Event", "Probability", "Impact", "Risk Score"]
            df = df.sort_values("Risk Score", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            total_score = sum(r["score"] for r in risks)
            max_score = len(risks) * 25
            risk_percentage = (total_score / max_score) * 100
            
            col_a, col_b = st.columns(2)
            with col_a:
                display_metric_card(f"{total_score}", "Total Risk Score", "highlight")
            with col_b:
                display_metric_card(f"{risk_percentage:.0f}%", "Risk Exposure Level", 
                                   "danger" if risk_percentage > 60 else "success" if risk_percentage < 40 else "normal")
            
            # Risk Priority Classification
            st.markdown("#### Risk Priority Classification")
            high_risks = [r for r in risks if r["score"] >= 15]
            med_risks = [r for r in risks if 8 <= r["score"] < 15]
            low_risks = [r for r in risks if r["score"] < 8]
            
            if high_risks:
                display_alert(f"🔴 <strong>HIGH PRIORITY ({len(high_risks)}):</strong> {', '.join([r['name'] for r in high_risks])}<br><em>Immediate action required</em>", "danger")
            if med_risks:
                display_alert(f"🟡 <strong>MEDIUM PRIORITY ({len(med_risks)}):</strong> {', '.join([r['name'] for r in med_risks])}<br><em>Monitor closely and develop contingency plans</em>", "warning")
            if low_risks:
                display_alert(f"🟢 <strong>LOW PRIORITY ({len(low_risks)}):</strong> {', '.join([r['name'] for r in low_risks])}<br><em>Periodic review sufficient</em>", "success")
    
    with tab3:
        st.markdown("### Practice Problems")
        
        # Problem 1
        with st.expander("📝 Problem 1: Triple Bottom Line", expanded=True):
            st.markdown("""
            **Question:** What is the "Triple Bottom Line" and why is it important for modern supply chain management?
            """)
            
            user_answer_1 = st.text_area("Your Answer:", key="risk_p1_ans", height=100,
                                         placeholder="Enter your answer here...")
            
            if st.button("Check Answer", key="risk_p1_btn"):
                st.markdown("---")
                st.markdown("#### ✅ Model Answer:")
                display_solution_step(1, "<strong>Definition:</strong> The Triple Bottom Line (TBL) evaluates a firm against three criteria:")
                display_solution_step(2, "<strong>Social (People):</strong> Impact on employees, communities, and society - fair labor practices, community engagement, human rights")
                display_solution_step(3, "<strong>Economic (Profit):</strong> Financial performance and long-term economic sustainability - not just short-term profits")
                display_solution_step(4, "<strong>Environmental (Planet):</strong> Ecological footprint and environmental sustainability - carbon emissions, waste reduction, resource conservation")
                
                display_key_insight("Why It Matters",
                    "Modern consumers and investors increasingly demand that companies demonstrate responsibility "
                    "across all three dimensions. Supply chains that ignore social or environmental factors face "
                    "reputational risks, regulatory penalties, and loss of market share.")
        
        # Problem 2
        with st.expander("📝 Problem 2: Efficiency vs. Effectiveness"):
            st.markdown("""
            **Question:** Distinguish between "Efficiency" and "Effectiveness" in operations management. 
            Provide an example where a company might be efficient but not effective.
            """)
            
            if st.button("Show Solution", key="risk_p2_btn"):
                st.markdown("#### ✅ Solution:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Efficiency**")
                    st.write("Doing something at the **lowest possible cost** (doing things right)")
                    st.write("*Focus: Resource utilization*")
                with col2:
                    st.markdown("**Effectiveness**")
                    st.write("Doing the **right things** to create the most value for the customer")
                    st.write("*Focus: Goal achievement*")
                
                display_alert(
                    "<strong>Example:</strong> A factory produces widgets at the lowest cost per unit (efficient), "
                    "but the widgets don't meet customer quality expectations (not effective). The company saves "
                    "money on production but loses customers due to poor quality.",
                    "info"
                )
        
        # Problem 3
        with st.expander("📝 Problem 3: Risk Score Calculation"):
            st.markdown("""
            **Question:** A company identifies the following risks:
            - Supplier bankruptcy: Probability = 2, Impact = 5
            - Equipment failure: Probability = 4, Impact = 3
            - Demand surge: Probability = 3, Impact = 4
            
            Calculate the risk score for each and determine which should be addressed first.
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ans_1 = st.number_input("Supplier bankruptcy score:", key="risk_p3_1")
            with col2:
                ans_2 = st.number_input("Equipment failure score:", key="risk_p3_2")
            with col3:
                ans_3 = st.number_input("Demand surge score:", key="risk_p3_3")
            
            if st.button("Check Answers", key="risk_p3_btn"):
                correct_1, correct_2, correct_3 = 10, 12, 12
                
                results = []
                if check_answer(ans_1, correct_1): results.append("✅ Supplier bankruptcy correct")
                else: results.append(f"❌ Supplier bankruptcy: 2 × 5 = {correct_1}")
                
                if check_answer(ans_2, correct_2): results.append("✅ Equipment failure correct")
                else: results.append(f"❌ Equipment failure: 4 × 3 = {correct_2}")
                
                if check_answer(ans_3, correct_3): results.append("✅ Demand surge correct")
                else: results.append(f"❌ Demand surge: 3 × 4 = {correct_3}")
                
                for r in results:
                    st.write(r)
                
                display_alert(
                    "<strong>Priority:</strong> Equipment failure and Demand surge (both score 12) should be "
                    "addressed first, followed by Supplier bankruptcy (score 10). However, the high impact (5) "
                    "of supplier bankruptcy means it may warrant special attention despite lower probability.",
                    "info"
                )
        
        # Problem 4
        with st.expander("📝 Problem 4: What-If Sensitivity Analysis"):
            st.markdown("""
            **Question:** Based on the textbook's guidance on sensitivity analysis, explain what happens 
            to project profitability if:
            1. Development time increases by 25%
            2. Sales volume decreases by 25%
            3. Product cost increases by $1 per unit
            """)
            
            if st.button("Show Analysis", key="risk_p4_btn"):
                st.markdown("#### ✅ Sensitivity Analysis:")
                
                display_solution_step(1, 
                    "<strong>25% increase in development time:</strong> Delays production ramp-up, marketing efforts, "
                    "and product sales. This pushes revenue further into the future, reducing its present value. "
                    "Also increases development costs and may allow competitors to enter first.")
                
                display_solution_step(2,
                    "<strong>25% decrease in sales volume:</strong> Directly reduces revenue while fixed costs remain "
                    "constant. This can turn a profitable project into a loss. The impact is magnified by operating "
                    "leverage (high fixed costs relative to variable costs).")
                
                display_solution_step(3,
                    "<strong>$1 increase in product cost:</strong> Reduces profit by $1 per unit sold. For high-volume "
                    "products, this can significantly impact total profitability. Consider: 100,000 units × $1 = $100,000 "
                    "reduction in profit.")
                
                display_citation(
                    "A dollar spent or saved on development cost is worth the present value of that dollar to the "
                    "value of the project.",
                    "Jacobs & Chase (2024, p. 60)"
                )

# ============================================================
# MODULE 2: PERT NETWORK (Chapter 4)
# ============================================================
def module_pert():
    display_header("🔗", "Chapter 4", "PERT Network Diagram & Completion Probability", 
                   "Critical path identification, slack calculation & Z-score probability (Exhibits 4.8–4.9)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Activity Estimator", "📊 Probability Calculator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### PERT Network Analysis")
        
        st.write("""
        **PERT (Program Evaluation and Review Technique)** is a project management tool that uses 
        probabilistic time estimates to account for uncertainty in activity durations. It was developed 
        by the U.S. Navy in 1958 for the Polaris missile project.
        """)
        
        display_citation(
            "A conservative approach dictates using the critical path with the largest total variance "
            "to focus management's attention on the activities most likely to exhibit broad variations.",
            "Jacobs & Chase (2024, p. 99)"
        )
        
        st.markdown("#### PERT Time Estimates")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🟢", "Optimistic (a)", 
                "Best-case scenario - everything goes perfectly. Probability ≈ 1%")
        with col2:
            display_concept_card("🔵", "Most Likely (m)", 
                "Normal conditions - most frequent outcome if repeated many times")
        with col3:
            display_concept_card("🔴", "Pessimistic (b)", 
                "Worst-case scenario - everything goes wrong. Probability ≈ 1%")
        
        st.markdown("#### Key Formulas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Expected Time (Tₑ)")
            st.latex(r"T_E = \frac{a + 4m + b}{6}")
            st.write("Weighted average giving 4× weight to most likely estimate (based on Beta distribution)")
        
        with col2:
            st.markdown("##### Variance (σ²)")
            st.latex(r"\sigma^2 = \left(\frac{b - a}{6}\right)^2")
            st.write("Measures uncertainty - larger spread between a and b means higher variance")
        
        st.markdown("##### Standard Deviation (σ)")
        st.latex(r"\sigma = \frac{b - a}{6}")
        
        st.markdown("#### Project Completion Probability")
        st.latex(r"Z = \frac{D - T_E}{\sqrt{\sum \sigma^2_{cp}}}")
        
        st.write("""
        Where:
        - **D** = Desired (target) completion time
        - **Tₑ** = Expected project duration (sum of critical path expected times)
        - **Σσ²cp** = Sum of variances on the critical path
        - **Z** = Standard normal deviate (look up in Z-table for probability)
        """)
        
        display_key_insight(
            "Critical Path Selection with Multiple Critical Paths",
            "When there are two or more critical paths of equal length, use the one with the "
            "<strong>largest total variance</strong> for probability calculations. This conservative "
            "approach focuses attention on activities most likely to cause schedule problems."
        )
        
        st.markdown("#### Example from Textbook (Exhibit 4.9)")
        
        example_data = pd.DataFrame({
            "Path": ["A-B-E-G-I", "A-C-F-G-I", "A-D-F-G-I", "A-D-H-I"],
            "Length (days)": [22, 17, 21, 21],
            "Status": ["CRITICAL PATH", "Slack = 5", "Slack = 1", "Slack = 1"]
        })
        st.dataframe(example_data, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Activity Time Estimator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Estimates")
            a = st.slider("Optimistic Time (a)", 1, 20, 4, help="Best-case duration")
            m = st.slider("Most Likely Time (m)", 1, 30, 8, help="Most probable duration")
            b = st.slider("Pessimistic Time (b)", 1, 40, 16, help="Worst-case duration")
            
            if not (a <= m <= b):
                display_alert("⚠️ PERT estimates should satisfy a ≤ m ≤ b", "warning")
            else:
                display_alert("✅ Valid PERT estimates", "success")
        
        with col2:
            te = (a + 4*m + b) / 6
            variance = ((b - a) / 6) ** 2
            std_dev = math.sqrt(variance)
            
            st.markdown("#### Calculated Results")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                display_metric_card(f"{te:.2f}", "Expected Time (Tₑ)", "highlight")
            with col_b:
                display_metric_card(f"{variance:.2f}", "Variance (σ²)", "normal")
            with col_c:
                display_metric_card(f"{std_dev:.2f}", "Std Dev (σ)", "normal")
            
            st.markdown("#### Step-by-Step Calculation")
            display_solution_step(1, f"Expected Time: Tₑ = ({a} + 4×{m} + {b}) / 6 = {a + 4*m + b} / 6 = <strong>{te:.2f}</strong>")
            display_solution_step(2, f"Variance: σ² = (({b} - {a}) / 6)² = ({b-a} / 6)² = ({(b-a)/6:.2f})² = <strong>{variance:.2f}</strong>")
            display_solution_step(3, f"Std Dev: σ = √{variance:.2f} = <strong>{std_dev:.2f}</strong>")
            
            st.markdown("#### Probability Ranges")
            st.write(f"- 68% chance: {te - std_dev:.1f} to {te + std_dev:.1f} days")
            st.write(f"- 95% chance: {te - 2*std_dev:.1f} to {te + 2*std_dev:.1f} days")
            st.write(f"- 99.7% chance: {te - 3*std_dev:.1f} to {te + 3*std_dev:.1f} days")
    
    with tab3:
        st.markdown("### Project Completion Probability Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Project Parameters")
            te_project = st.number_input("Expected Project Duration (Tₑ)", value=38.0, step=0.5,
                                         help="Sum of expected times on critical path")
            d_target = st.number_input("Desired Completion (D)", value=35.0, step=0.5,
                                       help="Target completion date")
            sum_variance = st.number_input("Sum of CP Variances (Σσ²)", value=11.89, step=0.1,
                                           help="Sum of variances for all critical path activities")
        
        with col2:
            if sum_variance > 0:
                z_score = (d_target - te_project) / math.sqrt(sum_variance)
                prob = normal_cdf(z_score)
                
                st.markdown("#### Results")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"{z_score:.2f}", "Z-Score", "normal")
                with col_b:
                    card_type = "danger" if prob < 0.5 else "success"
                    display_metric_card(f"{prob*100:.1f}%", "P(Complete by D)", card_type)
                
                st.markdown("#### Interpretation")
                if prob < 0.25:
                    display_alert(
                        f"🔴 <strong>Very Low Probability ({prob*100:.1f}%)</strong><br>"
                        f"Only a {prob*100:.1f}% chance of completing by day {d_target}. "
                        f"Consider crashing critical activities or extending the deadline to {te_project}+ days.",
                        "danger"
                    )
                elif prob < 0.5:
                    display_alert(
                        f"🟡 <strong>Below Average Probability ({prob*100:.1f}%)</strong><br>"
                        f"Less than 50% chance of meeting the deadline. Risk mitigation recommended.",
                        "warning"
                    )
                else:
                    display_alert(
                        f"🟢 <strong>Good Probability ({prob*100:.1f}%)</strong><br>"
                        f"Reasonable chance of meeting the deadline.",
                        "success"
                    )
                
                st.markdown("#### Calculation")
                st.latex(rf"Z = \frac{{{d_target} - {te_project}}}{{\sqrt{{{sum_variance}}}}} = \frac{{{d_target - te_project:.2f}}}{{{math.sqrt(sum_variance):.2f}}} = {z_score:.2f}")
        
        # Variance Builder
        st.markdown("---")
        st.markdown("### Critical Path Variance Builder")
        st.write("Enter activity estimates to calculate total path variance")
        
        num_activities = st.number_input("Number of CP Activities", 1, 10, 5, key="pert_num_act")
        
        activities = []
        total_te = 0
        total_var = 0
        
        cols_header = st.columns([1, 1, 1, 1, 1, 1])
        cols_header[0].write("**Activity**")
        cols_header[1].write("**a**")
        cols_header[2].write("**m**")
        cols_header[3].write("**b**")
        cols_header[4].write("**Tₑ**")
        cols_header[5].write("**σ²**")
        
        for i in range(int(num_activities)):
            cols = st.columns([1, 1, 1, 1, 1, 1])
            with cols[0]:
                st.write(f"Activity {chr(65+i)}")
            with cols[1]:
                a_i = st.number_input(f"a_{i}", value=2+i, key=f"pert_a_{i}", label_visibility="collapsed")
            with cols[2]:
                m_i = st.number_input(f"m_{i}", value=4+i, key=f"pert_m_{i}", label_visibility="collapsed")
            with cols[3]:
                b_i = st.number_input(f"b_{i}", value=8+i*2, key=f"pert_b_{i}", label_visibility="collapsed")
            
            te_i = (a_i + 4*m_i + b_i) / 6
            var_i = ((b_i - a_i) / 6) ** 2
            total_te += te_i
            total_var += var_i
            
            with cols[4]:
                st.write(f"{te_i:.2f}")
            with cols[5]:
                st.write(f"{var_i:.3f}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card(f"{total_te:.2f}", "Total Path Duration", "highlight")
        with col2:
            display_metric_card(f"{total_var:.3f}", "Total Path Variance", "normal")
        with col3:
            display_metric_card(f"{math.sqrt(total_var):.3f}", "Path Std Deviation", "normal")
    
    with tab4:
        st.markdown("### Practice Problems")
        
        # Problem 1
        with st.expander("📝 Problem 1: Calculate Expected Time (Easy)", expanded=True):
            st.markdown("""
            **Given:** Activity X has the following time estimates:
            - Optimistic (a) = 5 days
            - Most Likely (m) = 8 days
            - Pessimistic (b) = 17 days
            
            **Calculate:** Expected time (Tₑ) and variance (σ²)
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                ans_te = st.number_input("Expected Time (Tₑ):", key="pert_p1_te", format="%.2f")
            with col2:
                ans_var = st.number_input("Variance (σ²):", key="pert_p1_var", format="%.2f")
            
            if st.button("Check Answer", key="pert_p1_btn"):
                correct_te = (5 + 4*8 + 17) / 6
                correct_var = ((17 - 5) / 6) ** 2
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Tₑ = (a + 4m + b) / 6 = (5 + 4×8 + 17) / 6 = (5 + 32 + 17) / 6 = 54 / 6 = <strong>9.00 days</strong>")
                display_solution_step(2, f"σ² = ((b - a) / 6)² = ((17 - 5) / 6)² = (12 / 6)² = 2² = <strong>4.00</strong>")
                
                if check_answer(ans_te, correct_te) and check_answer(ans_var, correct_var):
                    display_alert("✅ Both answers correct!", "success")
                else:
                    if not check_answer(ans_te, correct_te):
                        display_alert(f"❌ Expected time incorrect. Correct answer: {correct_te:.2f}", "danger")
                    if not check_answer(ans_var, correct_var):
                        display_alert(f"❌ Variance incorrect. Correct answer: {correct_var:.2f}", "danger")
        
        # Problem 2
        with st.expander("📝 Problem 2: Project Completion Probability (Medium)"):
            st.markdown("""
            **Given:** A project has:
            - Expected duration (Tₑ) = 45 weeks
            - Sum of critical path variances = 16 weeks²
            - Target completion = 41 weeks
            
            **Calculate:** 
            1. The Z-score
            2. The probability of completing by week 41
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                ans_z = st.number_input("Z-Score:", key="pert_p2_z", format="%.2f")
            with col2:
                ans_prob = st.number_input("Probability (%):", key="pert_p2_prob", format="%.1f")
            
            if st.button("Check Answer", key="pert_p2_btn"):
                correct_z = (41 - 45) / math.sqrt(16)
                correct_prob = normal_cdf(correct_z) * 100
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Z = (D - Tₑ) / √(Σσ²) = (41 - 45) / √16 = -4 / 4 = <strong>-1.00</strong>")
                display_solution_step(2, f"Look up Z = -1.00 in standard normal table")
                display_solution_step(3, f"P(Z ≤ -1.00) = <strong>{correct_prob:.1f}%</strong>")
                
                display_alert(
                    f"<strong>Interpretation:</strong> There is only a {correct_prob:.1f}% chance of completing "
                    f"the project by week 41. The project manager should either extend the deadline or crash "
                    f"critical path activities.",
                    "info"
                )
        
        # Problem 3
        with st.expander("📝 Problem 3: Critical Path Analysis (Hard)"):
            st.markdown("""
            **Given:** A project has the following activities on the critical path:
            
            | Activity | a | m | b |
            |----------|---|---|---|
            | A | 2 | 4 | 6 |
            | B | 3 | 5 | 13 |
            | C | 4 | 6 | 8 |
            | D | 2 | 3 | 10 |
            
            **Calculate:**
            1. Expected time for each activity
            2. Total expected project duration
            3. Total variance
            4. Probability of completing in 20 days or less
            """)
            
            if st.button("Show Complete Solution", key="pert_p3_btn"):
                st.markdown("#### Solution:")
                
                activities = [
                    ("A", 2, 4, 6),
                    ("B", 3, 5, 13),
                    ("C", 4, 6, 8),
                    ("D", 2, 3, 10)
                ]
                
                results = []
                total_te = 0
                total_var = 0
                
                for name, a, m, b in activities:
                    te = (a + 4*m + b) / 6
                    var = ((b - a) / 6) ** 2
                    total_te += te
                    total_var += var
                    results.append({"Activity": name, "a": a, "m": m, "b": b, "Tₑ": f"{te:.2f}", "σ²": f"{var:.3f}"})
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
                
                display_solution_step(1, f"Total Expected Duration: Tₑ = {total_te:.2f} days")
                display_solution_step(2, f"Total Variance: Σσ² = {total_var:.3f}")
                display_solution_step(3, f"Standard Deviation: σ = √{total_var:.3f} = {math.sqrt(total_var):.3f}")
                
                z = (20 - total_te) / math.sqrt(total_var)
                prob = normal_cdf(z) * 100
                
                display_solution_step(4, f"Z = (20 - {total_te:.2f}) / {math.sqrt(total_var):.3f} = {z:.2f}")
                display_solution_step(5, f"P(Complete ≤ 20 days) = {prob:.1f}%")

# ============================================================
# MODULE 3: PROJECT CRASHING (Chapter 4)
# ============================================================
def module_crashing():
    display_header("⚡", "Chapter 4", "Project Crashing", "Time-cost trade-off analysis")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### Project Crashing Theory")
        
        st.write("""
        **Project crashing** (also called project compression) involves reducing project duration 
        by adding resources to critical path activities. The goal is to achieve the desired 
        completion date at minimum additional cost.
        """)
        
        st.markdown("#### Crash Cost per Day Formula")
        st.latex(r"\text{Crash Cost per Day} = \frac{\text{Crash Cost} - \text{Normal Cost}}{\text{Normal Time} - \text{Crash Time}}")
        
        st.markdown("#### Key Concepts")
        
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("⏱️", "Normal Time", 
                "Standard duration using normal resources and methods")
            display_concept_card("💰", "Normal Cost", 
                "Cost to complete activity in normal time")
        with col2:
            display_concept_card("⚡", "Crash Time", 
                "Minimum possible duration with maximum resources")
            display_concept_card("💸", "Crash Cost", 
                "Cost to complete activity in crash time (always higher)")
        
        display_key_insight(
            "Crashing Strategy",
            "Always crash the activity on the critical path with the <strong>lowest crash cost per day</strong> first. "
            "Continue until: (1) target date is reached, (2) critical path changes, or (3) no more crashing is possible. "
            "When the critical path changes, you may need to crash multiple paths simultaneously."
        )
        
        st.markdown("#### Crashing Procedure")
        st.write("""
        1. **Identify** the critical path
        2. **Calculate** crash cost per day for each critical activity
        3. **Select** the activity with lowest crash cost per day
        4. **Crash** that activity by one day (or until it reaches crash time or path changes)
        5. **Recalculate** critical path and repeat until target is met
        """)
    
    with tab2:
        st.markdown("### Crash Cost Calculator")
        
        num_activities = st.number_input("Number of Activities", 2, 10, 4, key="crash_num")
        
        activities = []
        
        st.markdown("#### Activity Data")
        
        cols_header = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1])
        headers = ["Activity", "Normal Time", "Crash Time", "Normal Cost", "Crash Cost", "Max Crash", "Cost/Day"]
        for i, h in enumerate(headers):
            cols_header[i].write(f"**{h}**")
        
        for i in range(int(num_activities)):
            cols = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1])
            
            with cols[0]:
                st.write(f"**{chr(65+i)}**")
            with cols[1]:
                nt = st.number_input(f"NT_{i}", value=5+i, key=f"crash_nt_{i}", label_visibility="collapsed")
            with cols[2]:
                ct = st.number_input(f"CT_{i}", value=3+i//2, key=f"crash_ct_{i}", label_visibility="collapsed")
            with cols[3]:
                nc = st.number_input(f"NC_{i}", value=1000+i*200, key=f"crash_nc_{i}", label_visibility="collapsed")
            with cols[4]:
                cc = st.number_input(f"CC_{i}", value=1800+i*400, key=f"crash_cc_{i}", label_visibility="collapsed")
            
            max_crash = nt - ct
            if max_crash > 0:
                cpd = (cc - nc) / max_crash
            else:
                cpd = float('inf')
            
            with cols[5]:
                st.write(f"{max_crash} days")
            with cols[6]:
                if cpd != float('inf'):
                    st.write(f"${cpd:.0f}")
                else:
                    st.write("N/A")
            
            activities.append({
                "Activity": chr(65+i),
                "Normal Time": nt,
                "Crash Time": ct,
                "Normal Cost": nc,
                "Crash Cost": cc,
                "Max Crash Days": max_crash,
                "Cost/Day": cpd if cpd != float('inf') else None
            })
        
        # Summary
        st.markdown("---")
        st.markdown("#### Summary")
        
        total_normal_time = sum(a["Normal Time"] for a in activities)
        total_crash_time = sum(a["Crash Time"] for a in activities)
        total_normal_cost = sum(a["Normal Cost"] for a in activities)
        total_crash_cost = sum(a["Crash Cost"] for a in activities)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric_card(f"{total_normal_time}", "Normal Duration", "normal")
        with col2:
            display_metric_card(f"{total_crash_time}", "Crash Duration", "highlight")
        with col3:
            display_metric_card(f"${total_normal_cost:,}", "Normal Cost", "normal")
        with col4:
            display_metric_card(f"${total_crash_cost:,}", "Full Crash Cost", "danger")
        
        # Crashing priority
        crashable = [a for a in activities if a["Cost/Day"] is not None and a["Max Crash Days"] > 0]
        if crashable:
            crashable_sorted = sorted(crashable, key=lambda x: x["Cost/Day"])
            
            st.markdown("#### Crashing Priority (Lowest Cost First)")
            for i, a in enumerate(crashable_sorted):
                st.write(f"{i+1}. **Activity {a['Activity']}**: ${a['Cost/Day']:.0f}/day (can crash {a['Max Crash Days']} days)")
    
    with tab3:
        st.markdown("### Practice Problems")
        
        with st.expander("📝 Problem 1: Calculate Crash Cost per Day"):
            st.markdown("""
            **Given:** Activity X has:
            - Normal Time = 10 days, Crash Time = 6 days
            - Normal Cost = $5,000, Crash Cost = $9,000
            
            **Calculate:** Crash cost per day
            """)
            
            ans = st.number_input("Crash Cost per Day ($):", key="crash_p1")
            
            if st.button("Check Answer", key="crash_p1_btn"):
                correct = (9000 - 5000) / (10 - 6)
                
                display_solution_step(1, "Crash Cost per Day = (Crash Cost - Normal Cost) / (Normal Time - Crash Time)")
                display_solution_step(2, f"= ($9,000 - $5,000) / (10 - 6)")
                display_solution_step(3, f"= $4,000 / 4 days = <strong>${correct:.0f}/day</strong>")
                
                if check_answer(ans, correct):
                    display_alert("✅ Correct!", "success")
                else:
                    display_alert(f"❌ Incorrect. Correct answer: ${correct:.0f}/day", "danger")

# ============================================================
# MODULE 4: BREAK-EVEN ANALYSIS (Chapter 5)
# ============================================================
def module_breakeven():
    display_header("📈", "Chapter 5", "Break-Even Analysis", "Cost-Volume-Profit (CVP) Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Theory", "🔬 Simulator", "📊 Sensitivity", "⚖️ Comparison", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Cost-Volume-Profit Analysis")
        
        st.write("""
        **Break-even analysis** determines the point at which total revenue equals total cost. 
        It establishes the relationship between fixed costs, variable costs, selling price, 
        and volume to identify the minimum output needed to cover all costs.
        """)
        
        display_citation(
            "Break-even analysis is a standard approach to determine the volume of output at which "
            "total revenue equals total cost. It is useful for comparing capacity alternatives and "
            "for determining the volume needed to achieve a target profit.",
            "Jacobs & Chase (2024, p. 155)"
        )
        
        st.markdown("#### Cost Structure Components")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🏢", "Fixed Costs (F)", 
                "Costs that remain constant regardless of output: rent, depreciation, insurance, salaries")
        with col2:
            display_concept_card("📦", "Variable Costs (V)", 
                "Costs that vary with output: raw materials, direct labor, packaging, shipping per unit")
        with col3:
            display_concept_card("💵", "Contribution Margin", 
                "Price minus Variable Cost (P - V). Each unit contributes this toward covering fixed costs")
        
        st.markdown("#### Key Formulas")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Break-Even Point (Units)")
            st.latex(r"BEP_{units} = \frac{F}{P - V}")
            
            st.markdown("##### Break-Even Point (Revenue)")
            st.latex(r"BEP_{\$} = \frac{F}{1 - \frac{V}{P}} = BEP_{units} \times P")
        
        with col2:
            st.markdown("##### Volume for Target Profit")
            st.latex(r"Q_{target} = \frac{F + \text{Target Profit}}{P - V}")
            
            st.markdown("##### Total Cost & Revenue")
            st.latex(r"TC = F + V \cdot Q \quad ; \quad TR = P \cdot Q")
        
        display_key_insight(
            "Comparing Capacity Alternatives",
            "When comparing two process alternatives (e.g., manual vs. automated), the option with "
            "higher fixed costs but lower variable costs will have a higher BEP but becomes more "
            "profitable at higher volumes. The <strong>indifference point</strong> where both options "
            "yield equal total cost is: Q* = (F₂ - F₁) / (V₁ - V₂)"
        )
        
        st.markdown("#### Assumptions & Limitations")
        st.write("""
        - Revenue and costs are linear functions of volume
        - Fixed costs remain constant within the relevant range
        - All units produced are sold (no inventory buildup)
        - Single product analysis (or constant product mix)
        - Price and variable cost per unit are constant
        """)
    
    with tab2:
        st.markdown("### Break-Even Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cost Parameters")
            fixed_cost = st.slider("Fixed Costs ($)", 10000, 200000, 50000, 5000)
            price = st.slider("Price per Unit ($)", 10, 500, 100, 5)
            variable_cost = st.slider("Variable Cost per Unit ($)", 5, 400, 60, 5)
            
            if price <= variable_cost:
                display_alert("⚠️ Price must be greater than Variable Cost!", "danger")
        
        with col2:
            if price > variable_cost:
                bep_units = fixed_cost / (price - variable_cost)
                bep_revenue = bep_units * price
                contribution_margin = price - variable_cost
                cm_ratio = contribution_margin / price
                
                st.markdown("#### Results")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"{bep_units:,.0f}", "BEP (Units)", "highlight")
                with col_b:
                    display_metric_card(f"${bep_revenue:,.0f}", "BEP (Revenue)", "highlight")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"${contribution_margin:.2f}", "Contribution Margin", "normal")
                with col_b:
                    display_metric_card(f"{cm_ratio:.1%}", "CM Ratio", "normal")
                
                st.markdown("#### Calculation")
                st.latex(rf"BEP = \frac{{{fixed_cost:,}}}{{{price} - {variable_cost}}} = \frac{{{fixed_cost:,}}}{{{contribution_margin}}} = {bep_units:,.0f} \text{{ units}}")
        
        # Target Profit Calculator
        if price > variable_cost:
            st.markdown("---")
            st.markdown("### Target Profit Analysis")
            
            target_profit = st.number_input("Target Annual Profit ($)", value=25000, step=5000)
            
            target_units = (fixed_cost + target_profit) / (price - variable_cost)
            target_revenue = target_units * price
            
            col1, col2 = st.columns(2)
            with col1:
                display_metric_card(f"{target_units:,.0f}", "Required Units", "success")
            with col2:
                display_metric_card(f"${target_revenue:,.0f}", "Required Revenue", "success")
            
            st.latex(rf"Q_{{target}} = \frac{{{fixed_cost:,} + {target_profit:,}}}{{{price} - {variable_cost}}} = {target_units:,.0f} \text{{ units}}")
    
    with tab3:
        st.markdown("### Sensitivity Analysis (What-If)")
        
        display_citation(
            "Some companies call this 'what if' analysis. Answering these 'what if' questions can be "
            "useful for understanding how sensitive an analysis is to cost and profit assumptions.",
            "Jacobs & Chase (2024, p. 60)"
        )
        
        st.markdown("#### Base Case")
        col1, col2, col3 = st.columns(3)
        with col1:
            base_fc = st.number_input("Base Fixed Cost ($)", value=50000, key="sens_fc")
        with col2:
            base_price = st.number_input("Base Price ($)", value=100, key="sens_p")
        with col3:
            base_vc = st.number_input("Base Variable Cost ($)", value=60, key="sens_vc")
        
        if base_price > base_vc:
            base_bep = base_fc / (base_price - base_vc)
            
            st.markdown("#### Sensitivity Scenarios")
            
            scenarios = [
                ("Base Case", base_fc, base_price, base_vc),
                ("+25% Fixed Costs", base_fc * 1.25, base_price, base_vc),
                ("-25% Fixed Costs", base_fc * 0.75, base_price, base_vc),
                ("+$10 Price", base_fc, base_price + 10, base_vc),
                ("-$10 Price", base_fc, base_price - 10, base_vc),
                ("+$5 Variable Cost", base_fc, base_price, base_vc + 5),
                ("-$5 Variable Cost", base_fc, base_price, base_vc - 5),
            ]
            
            results = []
            for name, fc, p, vc in scenarios:
                if p > vc:
                    bep = fc / (p - vc)
                    change = ((bep - base_bep) / base_bep) * 100
                    results.append({
                        "Scenario": name,
                        "Fixed Cost": f"${fc:,.0f}",
                        "Price": f"${p:.0f}",
                        "Var Cost": f"${vc:.0f}",
                        "BEP (units)": f"{bep:,.0f}",
                        "Change": f"{change:+.1f}%"
                    })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            display_key_insight(
                "Sensitivity Insights",
                "Notice that BEP is most sensitive to changes in contribution margin (P - V). "
                "A $10 price increase has a larger impact than a 25% change in fixed costs."
            )
    
    with tab4:
        st.markdown("### Scenario Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Scenario A (Current)")
            fc_a = st.number_input("Fixed Costs A ($)", value=50000, key="be_fc_a")
            p_a = st.number_input("Price A ($)", value=100, key="be_p_a")
            vc_a = st.number_input("Variable Cost A ($)", value=60, key="be_vc_a")
            
            if p_a > vc_a:
                bep_a = fc_a / (p_a - vc_a)
                display_metric_card(f"{bep_a:,.0f}", "BEP A (units)", "highlight")
        
        with col2:
            st.markdown("#### Scenario B (Alternative)")
            fc_b = st.number_input("Fixed Costs B ($)", value=80000, key="be_fc_b")
            p_b = st.number_input("Price B ($)", value=100, key="be_p_b")
            vc_b = st.number_input("Variable Cost B ($)", value=45, key="be_vc_b")
            
            if p_b > vc_b:
                bep_b = fc_b / (p_b - vc_b)
                display_metric_card(f"{bep_b:,.0f}", "BEP B (units)", "highlight")
        
        # Indifference Point
        if p_a > vc_a and p_b > vc_b and vc_a != vc_b:
            st.markdown("---")
            st.markdown("### Indifference Analysis")
            
            indiff_point = (fc_b - fc_a) / (vc_a - vc_b)
            
            if indiff_point > 0:
                display_metric_card(f"{indiff_point:,.0f}", "Indifference Point (units)", "highlight")
                
                st.latex(rf"Q^* = \frac{{F_B - F_A}}{{V_A - V_B}} = \frac{{{fc_b:,} - {fc_a:,}}}{{{vc_a} - {vc_b}}} = {indiff_point:,.0f}")
                
                lower_fc = "A" if fc_a < fc_b else "B"
                lower_vc = "A" if vc_a < vc_b else "B"
                
                display_alert(
                    f"📊 <strong>Decision Rule:</strong><br>"
                    f"• Below {indiff_point:,.0f} units: Choose Scenario {lower_fc} (lower fixed costs)<br>"
                    f"• Above {indiff_point:,.0f} units: Choose Scenario {lower_vc} (lower variable costs)",
                    "info"
                )
    
    with tab5:
        st.markdown("### Practice Problems")
        
        # Problem 1
        with st.expander("📝 Problem 1: Basic BEP Calculation (Easy)", expanded=True):
            st.markdown("""
            **Given:**
            - Fixed Costs = $40,000
            - Selling Price = $120 per unit
            - Variable Cost = $80 per unit
            
            **Calculate:** Break-even point in units
            """)
            
            ans = st.number_input("BEP (units):", key="be_p1")
            
            if st.button("Check Answer", key="be_p1_btn"):
                correct = 40000 / (120 - 80)
                
                st.markdown("#### Solution:")
                display_solution_step(1, "Contribution Margin = Price - Variable Cost = $120 - $80 = $40")
                display_solution_step(2, "BEP = Fixed Costs / Contribution Margin")
                display_solution_step(3, f"BEP = $40,000 / $40 = <strong>{correct:,.0f} units</strong>")
                
                if check_answer(ans, correct):
                    display_alert("✅ Correct!", "success")
                else:
                    display_alert(f"❌ Incorrect. Correct answer: {correct:,.0f} units", "danger")
        
        # Problem 2
        with st.expander("📝 Problem 2: Target Profit (Medium)"):
            st.markdown("""
            **Given:**
            - Fixed Costs = $60,000
            - Selling Price = $50 per unit
            - Variable Cost = $30 per unit
            - Target Profit = $20,000
            
            **Calculate:** Units needed to achieve target profit
            """)
            
            ans = st.number_input("Required units:", key="be_p2")
            
            if st.button("Check Answer", key="be_p2_btn"):
                correct = (60000 + 20000) / (50 - 30)
                
                st.markdown("#### Solution:")
                display_solution_step(1, "Contribution Margin = $50 - $30 = $20")
                display_solution_step(2, "Q = (Fixed Costs + Target Profit) / CM")
                display_solution_step(3, f"Q = ($60,000 + $20,000) / $20 = $80,000 / $20 = <strong>{correct:,.0f} units</strong>")
                
                if check_answer(ans, correct):
                    display_alert("✅ Correct!", "success")
                else:
                    display_alert(f"❌ Incorrect. Correct answer: {correct:,.0f} units", "danger")
        
        # Problem 3
        with st.expander("📝 Problem 3: Indifference Point (Hard)"):
            st.markdown("""
            **Given:** Two manufacturing options:
            
            | Option | Fixed Costs | Variable Cost/Unit |
            |--------|-------------|-------------------|
            | Manual | $20,000 | $15 |
            | Automated | $80,000 | $5 |
            
            **Calculate:**
            1. Indifference point (where both options have equal total cost)
            2. Which option is better at 5,000 units?
            3. Which option is better at 8,000 units?
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ans_indiff = st.number_input("Indifference point:", key="be_p3_1")
            with col2:
                ans_5000 = st.selectbox("Better at 5,000:", ["Manual", "Automated"], key="be_p3_2")
            with col3:
                ans_8000 = st.selectbox("Better at 8,000:", ["Manual", "Automated"], key="be_p3_3")
            
            if st.button("Check Answer", key="be_p3_btn"):
                correct_indiff = (80000 - 20000) / (15 - 5)
                
                tc_manual_5000 = 20000 + 15 * 5000
                tc_auto_5000 = 80000 + 5 * 5000
                tc_manual_8000 = 20000 + 15 * 8000
                tc_auto_8000 = 80000 + 5 * 8000
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Indifference Point = (F₂ - F₁) / (V₁ - V₂) = ($80,000 - $20,000) / ($15 - $5) = $60,000 / $10 = <strong>{correct_indiff:,.0f} units</strong>")
                
                display_solution_step(2, f"At 5,000 units:<br>• Manual: $20,000 + $15×5,000 = ${tc_manual_5000:,}<br>• Automated: $80,000 + $5×5,000 = ${tc_auto_5000:,}<br><strong>Manual is better</strong>")
                
                display_solution_step(3, f"At 8,000 units:<br>• Manual: $20,000 + $15×8,000 = ${tc_manual_8000:,}<br>• Automated: $80,000 + $5×8,000 = ${tc_auto_8000:,}<br><strong>Automated is better</strong>")

# ============================================================
# MODULE 5: DECISION TREES (Chapter 5)
# ============================================================
def module_decision():
    display_header("🌳", "Chapter 5", "Decision Trees & Expected Monetary Value", 
                   "Structured decision-making under uncertainty")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice Problems"])
    
    with tab1:
        st.markdown("### Expected Monetary Value (EMV) Analysis")
        
        st.write("""
        **Decision tree analysis** is a quantitative approach for evaluating alternatives that 
        involve sequential decisions and chance events. It provides a visual framework for 
        analyzing decisions under uncertainty.
        """)
        
        display_citation(
            "A decision tree is a schematic model of alternatives available to the decision maker, "
            "along with their possible consequences. The term gets its name from the tree-like "
            "appearance of the diagram.",
            "Jacobs & Chase (2024, p. 148)"
        )
        
        st.markdown("#### Decision Tree Components")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("◻️", "Decision Node", 
                "Point where decision maker chooses between alternatives (square symbol)")
        with col2:
            display_concept_card("⭕", "Chance Node", 
                "Point where chance determines outcome - probabilities must sum to 1.0 (circle symbol)")
        with col3:
            display_concept_card("🔺", "Terminal Node", 
                "Final payoff at end of branch - the monetary outcome (triangle symbol)")
        
        st.markdown("#### Key Formulas")
        
        st.markdown("##### Expected Monetary Value (EMV)")
        st.latex(r"EMV = \sum_{i=1}^{n} (P_i \times V_i)")
        st.write("Where Pᵢ = probability of outcome i, Vᵢ = monetary value of outcome i")
        st.write("**Decision Rule:** Select the alternative with the highest EMV")
        
        st.markdown("##### Expected Value of Perfect Information (EVPI)")
        st.latex(r"EVPI = EV_{with\ PI} - EV_{without\ PI}")
        st.write("EVPI represents the maximum amount you should pay for perfect information")
        
        display_key_insight(
            "Roll Back Method",
            "Decision trees are solved from <strong>right to left</strong> (backward induction). "
            "At each chance node, calculate the EMV. At each decision node, select the alternative "
            "with the highest EMV. This process 'rolls back' the tree to determine the optimal initial decision."
        )
    
    with tab2:
        st.markdown("### EMV Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏭 Large Facility Option")
            st.write("Higher risk, higher potential reward")
            
            prob_high = st.slider("P(High Demand)", 0, 100, 60, help="Probability of high demand scenario") / 100
            payoff_high_large = st.number_input("High Demand Payoff ($)", value=200000, key="dt_h_l")
            payoff_low_large = st.number_input("Low Demand Payoff ($)", value=-50000, key="dt_l_l")
            
            emv_large = prob_high * payoff_high_large + (1 - prob_high) * payoff_low_large
            
            st.markdown("##### Calculation:")
            st.latex(rf"EMV_{{Large}} = {prob_high:.2f} \times \${payoff_high_large:,} + {1-prob_high:.2f} \times \${payoff_low_large:,}")
            display_metric_card(f"${emv_large:,.0f}", "EMV (Large Facility)", "highlight")
        
        with col2:
            st.markdown("#### 🏠 Small Facility Option")
            st.write("Lower risk, more conservative")
            
            st.write(f"P(High Demand) = {prob_high:.0%} (same as above)")
            payoff_high_small = st.number_input("High Demand Payoff ($)", value=90000, key="dt_h_s")
            payoff_low_small = st.number_input("Low Demand Payoff ($)", value=25000, key="dt_l_s")
            
            emv_small = prob_high * payoff_high_small + (1 - prob_high) * payoff_low_small
            
            st.markdown("##### Calculation:")
            st.latex(rf"EMV_{{Small}} = {prob_high:.2f} \times \${payoff_high_small:,} + {1-prob_high:.2f} \times \${payoff_low_small:,}")
            display_metric_card(f"${emv_small:,.0f}", "EMV (Small Facility)", "normal")
        
        st.markdown("---")
        st.markdown("### Decision Recommendation")
        
        if emv_large > emv_small:
            display_alert(f"✅ <strong>Choose Large Facility</strong><br>EMV = ${emv_large:,.0f} > ${emv_small:,.0f}", "success")
        elif emv_small > emv_large:
            display_alert(f"✅ <strong>Choose Small Facility</strong><br>EMV = ${emv_small:,.0f} > ${emv_large:,.0f}", "success")
        else:
            display_alert(f"⚖️ <strong>Indifferent</strong><br>Both options have EMV = ${emv_large:,.0f}", "info")
        
        # EVPI Calculation
        st.markdown("---")
        st.markdown("### Expected Value of Perfect Information (EVPI)")
        
        # EV with perfect information = weighted average of best outcomes in each state
        best_high = max(payoff_high_large, payoff_high_small)
        best_low = max(payoff_low_large, payoff_low_small)
        ev_with_pi = prob_high * best_high + (1 - prob_high) * best_low
        ev_without_pi = max(emv_large, emv_small)
        evpi = ev_with_pi - ev_without_pi
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card(f"${ev_with_pi:,.0f}", "EV with Perfect Info", "normal")
        with col2:
            display_metric_card(f"${ev_without_pi:,.0f}", "EV without Perfect Info", "normal")
        with col3:
            display_metric_card(f"${evpi:,.0f}", "EVPI", "highlight")
        
        display_alert(
            f"💡 <strong>Interpretation:</strong> You should pay at most <strong>${evpi:,.0f}</strong> for perfect "
            f"market information (e.g., a market research study that perfectly predicts demand).",
            "info"
        )
    
    with tab3:
        st.markdown("### Practice Problems")
        
        with st.expander("📝 Problem 1: Calculate EMV (Easy)", expanded=True):
            st.markdown("""
            **Given:** A company is deciding between two options:
            - **Option A:** 40% chance of $100,000, 60% chance of $20,000
            - **Option B:** 50% chance of $80,000, 50% chance of $30,000
            
            **Calculate:** EMV for each option and determine which to choose
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                ans_a = st.number_input("EMV(A) ($):", key="dt_p1_a")
            with col2:
                ans_b = st.number_input("EMV(B) ($):", key="dt_p1_b")
            
            if st.button("Check Answer", key="dt_p1_btn"):
                emv_a = 0.4 * 100000 + 0.6 * 20000
                emv_b = 0.5 * 80000 + 0.5 * 30000
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"EMV(A) = 0.40 × $100,000 + 0.60 × $20,000 = $40,000 + $12,000 = <strong>${emv_a:,.0f}</strong>")
                display_solution_step(2, f"EMV(B) = 0.50 × $80,000 + 0.50 × $30,000 = $40,000 + $15,000 = <strong>${emv_b:,.0f}</strong>")
                display_solution_step(3, f"<strong>Choose Option {'A' if emv_a > emv_b else 'B'}</strong> (higher EMV)")
                
                if check_answer(ans_a, emv_a) and check_answer(ans_b, emv_b):
                    display_alert("✅ Both answers correct!", "success")

# ============================================================
# MODULE 36: PRACTICE PROBLEMS (General)
# ============================================================
def module_practice():
    display_header("🎓", "Exam Prep", "Practice Problems", "Comprehensive review questions")
    
    st.markdown("### Mixed Practice Problems")
    
    problems = [
        ("DPMO Calculation", "2,000 units, 33 defects, 5 opportunities per unit. What is DPMO?", 
         "DPMO = (33 / (2000 × 5)) × 1,000,000 = 3,300"),
        ("Factor Rating", "Loc A: 75.25, Loc B: 74.50. Which location?", 
         "Location A (higher weighted score)"),
        ("EOQ", "D=10,000, S=$50, H=$2. Calculate Q*.", 
         "Q* = √(2×10000×50/2) = 707 units"),
        ("Break-Even", "FC=$100,000, P=$50, VC=$30. Calculate BEP.", 
         "BEP = 100,000/(50-30) = 5,000 units"),
        ("Learning Curve", "First unit=100hrs, 80% curve. Time for unit 4?", 
         "Y₄ = 100 × 4^(-0.322) = 64 hours")
    ]
    
    for i, (title, question, answer) in enumerate(problems):
        with st.expander(f"{title}"):
            st.write(question)
            if st.button(f"Show Solution", key=f"prac_{i}"):
                st.success(answer)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
def main():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem;">📊</div>
        <div style="font-weight: 700; font-size: 1.2rem;">OSCM Simulator</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Statistics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Modules", "40")
    with col2:
        st.metric("Formulas", "75+")
    
    st.sidebar.markdown("""
    <div style="font-size: 0.75rem; padding: 0.5rem; background: #f8fafc; border-radius: 6px; margin: 0.5rem 0;">
        Based on Jacobs & Chase (2024). <em>Operations and Supply Chain Management</em>, 17th ed. McGraw-Hill.
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Module selection organized by chapter
    modules = {
        "Ch 1: Strategy & Risk": {
            "🛡️ SC Risk Assessment": "risk"
        },
        "Ch 4: Project Management": {
            "🔗 PERT Network": "pert",
            "⚡ Project Crashing": "crashing"
        },
        "Ch 5: Capacity Planning": {
            "📈 Break-Even Analysis": "breakeven",
            "🌳 Decision Trees": "decision"
        },
        "Ch 6: Learning Curves": {
            "📉 Learning Curves": "learning"
        },
        "Ch 7: Manufacturing": {
            "🔀 Decoupling Point": "decoupling"
        },
        "Ch 8: Layout": {
            "⚖️ Line Balancing": "linebalance"
        },
        "Ch 9: Service Design": {
            "🎯 Service Design": "service",
            "🛡️ Poka-yoke DB": "pokayoke"
        },
        "Ch 10: Queuing": {
            "👥 Queuing Theory": "queuing",
            "📐 Distributions": "distributions"
        },
        "Ch 11: Process Analysis": {
            "🔄 Little's Law": "littles"
        },
        "Ch 12: Six Sigma": {
            "🎯 DPMO & DMAIC": "dpmo",
            "⚠️ FMEA Risk": "fmea"
        },
        "Ch 13: Quality Control": {
            "📉 p & c Charts": "sqc",
            "🎯 Process Capability": "capability",
            "📊 Acceptance Sampling": "sampling",
            "📊 Pareto Analysis": "pareto",
            "🐟 Fishbone Diagram": "fishbone",
            "🎓 SQC Practice": "sqc_practice"
        },
        "Ch 14: Lean": {
            "🔄 Lean Supply Chains": "lean"
        },
        "Ch 15: Logistics": {
            "📍 Centroid Method": "centroid",
            "⚖️ Factor Rating": "factor",
            "🚚 Transportation": "transportation"
        },
        "Ch 16: Sourcing": {
            "🌐 Global Sourcing": "sourcing"
        },
        "Ch 18: Forecasting": {
            "📈 Enhanced Forecast": "forecast",
            "📈 Regression+": "regression"
        },
        "Ch 19: Aggregate Planning": {
            "📋 Aggregate Planning": "aggregate"
        },
        "Ch 20: Inventory": {
            "📦 EOQ Model": "eoq",
            "🛡️ Safety Stock": "safetystock",
            "📰 Newsvendor Model": "newsvendor"
        },
        "Ch 21: MRP": {
            "🏭 MRP Matrix": "mrp",
            "📦 MRP Lot Sizing": "mrp_lotsizing"
        },
        "Ch 22: Scheduling": {
            "📅 Job Scheduling": "scheduling"
        },
        "Exam Prep": {
            "🎓 Practice Problems": "practice"
        }
    }
    
    # Create navigation
    selected_module = None
    
    for section, section_modules in modules.items():
        with st.sidebar.expander(section, expanded=False):
            for name, key in section_modules.items():
                if st.button(name, key=f"nav_{key}", use_container_width=True):
                    st.session_state.selected_module = key
    
    # Get selected module from session state
    if "selected_module" not in st.session_state:
        st.session_state.selected_module = "risk"
    
    selected = st.session_state.selected_module
    
    # Route to selected module
    module_functions = {
        "risk": module_risk,
        "pert": module_pert,
        "crashing": module_crashing,
        "breakeven": module_breakeven,
        "decision": module_decision,
        "learning": module_learning,
        "decoupling": module_decoupling,
        "linebalance": module_linebalance,
        "service": module_service,
        "queuing": module_queuing,
        "distributions": module_distributions,
        "littles": module_littles,
        "dpmo": module_dpmo,
        "fmea": module_fmea,
        "sqc": module_sqc,
        "capability": module_capability,
        "sampling": module_sampling,
        "pareto": module_pareto,
        "fishbone": module_fishbone,
        "lean": module_lean,
        "centroid": module_centroid,
        "factor": module_factor,
        "transportation": module_transportation,
        "sourcing": module_sourcing,
        "forecast": module_forecast,
        "regression": module_regression,
        "aggregate": module_aggregate,
        "eoq": module_eoq,
        "safetystock": module_safetystock,
        "newsvendor": module_newsvendor,
        "mrp": module_mrp,
        "mrp_lotsizing": module_mrp_lotsizing,
        "scheduling": module_scheduling,
        "pokayoke": module_pokayoke,
        "sqc_practice": module_sqc_practice,
        "practice": module_practice
    }
    
    if selected in module_functions:
        module_functions[selected]()
    else:
        module_risk()  # Default

if __name__ == "__main__":
    main()