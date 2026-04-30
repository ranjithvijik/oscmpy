import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import math

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="OSCM Simulator v4.0 - Enhanced Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME MANAGEMENT
# ============================================================
def init_theme():
    """Initialize theme in session state."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

def toggle_theme():
    """Toggle between dark and light mode."""
    st.session_state.dark_mode = not st.session_state.dark_mode

init_theme()

# ============================================================
# DYNAMIC CSS STYLING (Merged Dark/Light Mode)
# ============================================================
def get_theme_css():
    """Generate CSS based on current theme. Merged from v3.5 and v4.0"""
    is_dark = st.session_state.dark_mode
    
    if is_dark:
        # Dark Mode Colors (Combining v3.5 robust vars with v4.0 improvements)
        bg_primary = "#0f172a"
        bg_secondary = "#1e293b"
        bg_card = "#1e293b"
        bg_input = "#334155"
        text_primary = "#e2e8f0"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        border_color = "#475569"
        accent_primary = "#6366f1"
        success_bg = "#064e3b"
        success_text = "#d1fae5"
        success_border = "#10b981"
        warning_bg = "#422006"
        warning_text = "#fef3c7"
        warning_border = "#f59e0b"
        danger_bg = "#7f1d1d"
        danger_text = "#fecaca"
        danger_border = "#ef4444"
        info_bg = "#1e3a5f"
        info_text = "#93c5fd"
        info_border = "#3b82f6"
        citation_bg = "#422006"
        citation_text = "#fef3c7"
        citation_border = "#f59e0b"
        equation_bg = "#0c4a6e"
        equation_border = "#0ea5e9"
        equation_text = "#e0f2fe"
        theory_bg = "#1e293b"
        theory_border = "#818cf8"
        insight_bg = "linear-gradient(135deg, #064e3b 0%, #065f46 100%)"
        insight_border = "#10b981"
        insight_title = "#34d399"
        practice_bg = "#1e3a5f"
        practice_border = "#3b82f6"
        metric_highlight_bg = "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)"
        metric_highlight_text = "#ffffff"
        metric_normal_bg = "linear-gradient(135deg, #1e293b 0%, #334155 100%)"
        metric_normal_text = "#f1f5f9"
        table_header_bg = "#334155"
        table_row_alt = "#1e293b"
        link_color = "#93c5fd"
        code_bg = "#0f172a"
        
        # New v4.0 components
        textbook_bg = "#1e293b"
        textbook_border = "#8b5cf6"
        textbook_text = "#e2e8f0"
        textbook_h4 = "#c4b5fd"
        practice_prob_bg = "#1e293b"
        practice_prob_border = "#6366f1"
        practice_prob_text = "#e2e8f0"
        practice_prob_h4 = "#a5b4fc"
        solution_bg = "#064e3b"
        solution_border = "#10b981"
        solution_text = "#d1fae5"
        hint_bg = "#422006"
        hint_border = "#f59e0b"
        hint_text = "#fef3c7"

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
        citation_text = "#713f12"
        citation_border = "#eab308"
        equation_bg = "#f0f9ff"
        equation_border = "#bae6fd"
        equation_text = "#0c4a6e"
        theory_bg = "#f8fafc"
        theory_border = "#6366f1"
        insight_bg = "linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)"
        insight_border = "#6ee7b7"
        insight_title = "#047857"
        practice_bg = "#f0f7ff"
        practice_border = "#93c5fd"
        metric_highlight_bg = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)"
        metric_highlight_text = "#ffffff"
        metric_normal_bg = "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)"
        metric_normal_text = "#1e293b"
        table_header_bg = "#f1f5f9"
        table_row_alt = "#f8fafc"
        link_color = "#2563eb"
        code_bg = "#f1f5f9"

        # New v4.0 components
        textbook_bg = "#faf5ff"
        textbook_border = "#8b5cf6"
        textbook_text = "#1e293b"
        textbook_h4 = "#6d28d9"
        practice_prob_bg = "#fafafa"
        practice_prob_border = "#6366f1"
        practice_prob_text = "#1e293b"
        practice_prob_h4 = "#4f46e5"
        solution_bg = "#f0fdf4"
        solution_border = "#86efac"
        solution_text = "#166534"
        hint_bg = "#fffbeb"
        hint_border = "#fcd34d"
        hint_text = "#92400e"
    
    return f"""
    <style>
        /* Global Styles */
        .stApp {{ background-color: {bg_primary}; color: {text_primary}; }}
        
        /* Main Header */
        .main-header {{
            background: {metric_highlight_bg};
            padding: 1.5rem; border-radius: 12px; color: white;
            margin-bottom: 1.5rem; border: 1px solid {accent_primary};
        }}
        .main-header h1 {{ margin: 0; font-size: 1.8rem; font-weight: 700; color: white !important; }}
        .main-header p {{ margin: 0.5rem 0 0 0; opacity: 0.95; color: white !important; }}
        .chapter-badge {{
            background: rgba(255,255,255,0.2); color: white;
            padding: 0.25rem 0.75rem; border-radius: 6px;
            font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem;
        }}
        
        /* Metric Cards */
        .metric-card {{
            background: {metric_normal_bg}; border: 1px solid {border_color};
            border-radius: 12px; padding: 1.2rem; text-align: center; margin: 0.5rem 0;
            color: {metric_normal_text};
        }}
        .metric-card.highlight {{
            background: {metric_highlight_bg}; color: {metric_highlight_text}; border: 1px solid {accent_primary};
        }}
        .metric-card.success {{ background: {success_bg}; border-color: {success_border}; color: {success_text}; }}
        .metric-card.danger {{ background: {danger_bg}; border-color: {danger_border}; color: {danger_text}; }}
        
        .metric-value {{ font-size: 1.8rem; font-weight: 700; line-height: 1.2; }}
        .metric-label {{ font-size: 0.85rem; opacity: 0.9; margin-top: 0.3rem; }}
        
        /* Educational Boxes */
        .theory-box {{ background: {theory_bg}; border-left: 4px solid {theory_border}; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0; }}
        .citation-box {{ background: {citation_bg}; border-left: 4px solid {citation_border}; padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0; font-style: italic; color: {citation_text}; }}
        .citation-source {{ display: block; margin-top: 0.5rem; font-style: normal; font-weight: 600; color: {citation_border}; }}
        .equation-box {{ background: {equation_bg}; border: 1px solid {equation_border}; border-radius: 8px; padding: 1rem; margin: 0.8rem 0; text-align: center; color: {equation_text}; }}
        .key-insight {{ background: {insight_bg}; border: 1px solid {insight_border}; border-radius: 8px; padding: 1rem; margin: 1rem 0; color: {success_text}; }}
        .key-insight-title {{ font-weight: 700; color: {insight_title}; margin-bottom: 0.5rem; }}
        .practice-box {{ background: {practice_bg}; border: 1px solid {practice_border}; border-radius: 8px; padding: 1rem; margin: 0.8rem 0; color: {info_text}; }}
        
        /* v4.0 New Components */
        .textbook-content {{ background: {textbook_bg}; border-left: 4px solid {textbook_border}; padding: 1.2rem; border-radius: 0 8px 8px 0; margin: 1rem 0; color: {textbook_text}; }}
        .textbook-content h4 {{ color: {textbook_h4}; margin-bottom: 0.8rem; }}
        .practice-problem {{ background: {practice_prob_bg}; border: 2px solid {practice_prob_border}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; color: {practice_prob_text}; }}
        .practice-problem h4 {{ color: {practice_prob_h4}; margin-bottom: 1rem; }}
        .solution-box {{ background: {solution_bg}; border: 1px solid {solution_border}; border-radius: 8px; padding: 1rem; margin-top: 1rem; color: {solution_text}; }}
        .hint-box {{ background: {hint_bg}; border: 1px solid {hint_border}; border-radius: 8px; padding: 0.8rem; margin: 0.5rem 0; color: {hint_text}; font-size: 0.9rem; }}
        .formula-card {{ background: {theory_bg}; border: 2px solid {accent_primary}; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; text-align: center; }}
        .formula-title {{ color: {accent_primary}; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; }}
        
        /* Concept Cards & Solution Steps (v3.5 components retained) */
        .concept-card {{ background: {bg_card}; border: 1px solid {border_color}; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; transition: all 0.2s ease; }}
        .concept-card:hover {{ border-color: {accent_primary}; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15); }}
        .concept-icon {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
        .concept-title {{ font-weight: 600; color: {text_primary}; margin-bottom: 0.3rem; }}
        .concept-desc {{ font-size: 0.85rem; color: {text_secondary}; }}
        .solution-step {{ background: {bg_secondary}; border-left: 3px solid {accent_primary}; padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0; }}
        .step-number {{ background: {accent_primary}; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem; }}

        /* Alert Boxes */
        .alert {{ padding: 1rem 1.2rem; border-radius: 10px; margin: 0.75rem 0; }}
        .alert-success {{ background: {success_bg}; border: 1px solid {success_border}; color: {success_text}; }}
        .alert-warning {{ background: {warning_bg}; border: 1px solid {warning_border}; color: {warning_text}; }}
        .alert-danger {{ background: {danger_bg}; border: 1px solid {danger_border}; color: {danger_text}; }}
        .alert-info {{ background: {info_bg}; border: 1px solid {info_border}; color: {info_text}; }}
        
        /* Tables */
        .styled-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
        .styled-table th {{ background: {table_header_bg}; color: {text_primary}; padding: 0.75rem; text-align: left; border-bottom: 2px solid {border_color}; }}
        .styled-table td {{ padding: 0.75rem; border-bottom: 1px solid {border_color}; color: {text_primary}; }}
        .styled-table tr:nth-child(even) {{ background: {table_row_alt}; }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{ background-color: {bg_secondary}; color: {text_primary}; }}
        section[data-testid="stSidebar"] .stMarkdown {{ color: {text_primary}; }}
        
        /* Inputs & Misc */
        .stTextInput input, .stNumberInput input, .stSelectbox select {{ background: {bg_input}; color: {text_primary}; border-color: {border_color}; }}
        .streamlit-expanderHeader {{ background: {bg_secondary}; color: {text_primary}; border-radius: 8px; }}
        .stDataFrame {{ background-color: {bg_card}; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.5rem; color: {text_primary}; }}
        
        .theme-toggle {{
            background: {metric_normal_bg}; border: 1px solid {accent_primary};
            border-radius: 8px; padding: 0.5rem 1rem; color: {accent_primary};
            cursor: pointer; text-align: center; margin: 0.5rem 0;
        }}
    </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS (Combined from v3.5 and v4.0)
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

def display_textbook_content(title, content):
    """Display enhanced textbook content (v4.0)."""
    st.markdown(f"""
    <div class="textbook-content">
        <h4>📖 {title}</h4>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_equation(label, latex_eq, description=""):
    """Display equation in styled box (v3.5)."""
    st.markdown(f"""
    <div class="equation-box">
        <div class="equation-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(latex_eq)
    if description:
        st.markdown(f"<p style='font-size: 0.85rem; opacity: 0.9; margin-top: 0.5rem;'>{description}</p>", unsafe_allow_html=True)

def display_formula_card(title, formula_latex):
    """Display a formula in a styled card (v4.0)."""
    st.markdown(f"""
    <div class="formula-card">
        <div class="formula-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(formula_latex)

def display_metric_card(value, label, card_type="normal"):
    """Display a metric card."""
    # Maps v3.5 card_type ("normal", "highlight", "success", "danger") into classes
    css_class = f"metric-card {card_type}" if card_type != "normal" else "metric-card"
    # To support v4.0's True/False `highlight` argument without breaking v3.5 string argument:
    if highlight := (card_type == True or card_type == "highlight"):
        css_class = "metric-card highlight"
        
    st.markdown(f"""
    <div class="{css_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def display_concept_card(icon, title, description):
    """Display a concept card (v3.5)."""
    st.markdown(f"""
    <div class="concept-card">
        <div class="concept-icon">{icon}</div>
        <div class="concept-title">{title}</div>
        <div class="concept-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def display_solution_step(step_num, content):
    """Display a solution step (v3.5)."""
    st.markdown(f"""
    <div class="solution-step">
        <span class="step-number">{step_num}</span>
        {content}
    </div>
    """, unsafe_allow_html=True)

def display_alert(content, alert_type="info"):
    """Display an alert box (v3.5)."""
    st.markdown(f'<div class="alert alert-{alert_type}">{content}</div>', unsafe_allow_html=True)

def display_practice_problem(problem_num, difficulty, problem_text):
    """Display a practice problem with difficulty indicator (v4.0)."""
    diff_colors = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
    st.markdown(f"""
    <div class="practice-problem">
        <h4>Problem {problem_num} {diff_colors.get(difficulty, "⚪")} {difficulty}</h4>
        <p>{problem_text}</p>
    </div>
    """, unsafe_allow_html=True)

def display_hint(hint_text):
    """Display a hint box (v4.0)."""
    st.markdown(f"""
    <div class="hint-box">
        💡 <strong>Hint:</strong> {hint_text}
    </div>
    """, unsafe_allow_html=True)

def display_solution(solution_text):
    """Display solution box (v4.0)."""
    st.markdown(f"""
    <div class="solution-box">
        ✅ <strong>Solution:</strong><br>{solution_text}
    </div>
    """, unsafe_allow_html=True)

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
# Z-TABLE DATA
# ============================================================
Z_TABLE = {}
for z_int in range(-30, 40):
    z_base = z_int / 10
    for z_dec in range(10):
        z = z_base + z_dec / 100
        Z_TABLE[round(z, 2)] = round(normal_cdf(z), 4)

# ============================================================
# MODULE 1: SUPPLY CHAIN RISK (Chapter 1) - MERGED
# ============================================================
def module_risk():
    display_header("🛡️", "Chapter 1", "Supply Chain Risk Assessment", "Probability and Impact Matrix")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Risk Identification & Assessment")
        
        display_textbook_content(
            "Operations and Supply Chain Management Defined",
            """OSCM is defined as the design, operation, and improvement of the systems that create 
            and deliver the firm's primary products and services. Understanding OSCM is critical 
            regardless of your business major - finance, marketing, or accounting - because all 
            business functions are interconnected through operations and supply chain processes."""
        )
        
        st.write("""
        **Supply chain risk** is the likelihood of a disruption that would impact the ability of a 
        company to continuously supply products or services. Risk management involves three steps:
        1. **Identification** - Recognize potential risk events
        2. **Assessment** - Evaluate probability and impact
        3. **Mitigation** - Develop strategies to reduce risk
        """)
        
        display_citation(
            "Supply chain risk management involves the identification of potential sources of risk and implementation of appropriate strategies through a coordinated approach among supply chain members to reduce supply chain vulnerability.",
            "Jacobs & Chase (2024, p. 12)"
        )
        
        st.markdown("### Core Formula")
        display_formula_card("Risk Score Calculation", r"\text{Risk Score} = \text{Probability} \times \text{Impact}")
        
        st.write("""
        Each risk event is scored on a scale (typically 1-5). Events with high scores require 
        immediate mitigation strategies such as redundancy, insurance, or process changes.
        """)
        
        display_textbook_content(
            "Process Analysis - A Basic Skill",
            """Process analysis is a basic skill needed to understand how a business operates. 
            Great insight is obtained by drawing a simple flowchart showing the flow of materials 
            or information through an enterprise. Often, 90 percent or more of the time required 
            to serve a customer is spent just waiting. Hence, merely eliminating the waiting time 
            can dramatically improve performance."""
        )
        
        display_key_insight(
            "Risk Categories",
            "Common supply chain risks include: supplier failure, natural disasters, quality issues, "
            "logistics delays, demand volatility, geopolitical events, and cybersecurity threats."
        )
        
        st.markdown("### The Three Elements of OSCM Integration")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("📊", "Strategy", "Competitive positioning and service vision")
        with col2:
            display_concept_card("⚙️", "Processes", "Operations that create and deliver value")
        with col3:
            display_concept_card("👥", "People", "Workforce skills and organizational culture")
    
    with tab2:
        st.markdown("### Risk Assessment Matrix Calculator")
        st.write("Score events from 1 (Low) to 5 (High) for both Probability and Impact")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Risk Event Scoring")
            risks = []
            risk_names = [
                "Supplier Failure (Financial)",
                "Natural Disaster / Weather",
                "Quality Issue / Product Recall",
                "Logistics / Customs Delay",
                "Demand Volatility",
                "Cybersecurity Breach"
            ]
            
            for i, name in enumerate(risk_names):
                with st.expander(f"📌 {name}", expanded=(i < 3)):
                    c1, c2 = st.columns(2)
                    with c1:
                        prob = st.slider(f"Probability", 1, 5, 3, key=f"risk_p_{i}")
                    with c2:
                        impact = st.slider(f"Impact", 1, 5, 4, key=f"risk_i_{i}")
                    risks.append({"name": name, "prob": prob, "impact": impact, "score": prob * impact})
        
        with col2:
            st.markdown("#### Risk Analysis Results")
            df = pd.DataFrame(risks)
            df.columns = ["Risk Event", "Probability", "Impact", "Risk Score"]
            # Sorting logic retained from v3.5
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
            
            st.markdown("### Risk Priority Matrix")
            high_risks = [r for r in risks if r["score"] >= 15]
            med_risks = [r for r in risks if 8 <= r["score"] < 15]
            low_risks = [r for r in risks if r["score"] < 8]
            
            if high_risks:
                st.error(f"🔴 **HIGH PRIORITY ({len(high_risks)}):** {', '.join([r['name'] for r in high_risks])}\n*Immediate action required*")
            if med_risks:
                st.warning(f"🟡 **MEDIUM PRIORITY ({len(med_risks)}):** {', '.join([r['name'] for r in med_risks])}\n*Monitor closely and develop contingency plans*")
            if low_risks:
                st.success(f"🟢 **LOW PRIORITY ({len(low_risks)}):** {', '.join([r['name'] for r in low_risks])}\n*Periodic review sufficient*")
    
    with tab3:
        st.markdown("### 📝 Enhanced Practice Problems")
        
        # Problem 1 (v3.5 / v4.0)
        with st.expander("🟢 Problem 1: Triple Bottom Line (Easy)"):
            display_practice_problem(1, "Easy", 
                "What are the three components of the Triple Bottom Line, and why is each important for sustainable business operations?")
            
            show_hint1 = st.checkbox("Show Hint", key="risk_hint1")
            if show_hint1:
                display_hint("Think about the three P's: People, Profit, and Planet.")
            
            if st.button("Show Complete Solution", key="risk_q1"):
                display_solution("""
                <strong>The Triple Bottom Line (3BL)</strong> evaluates a firm against three criteria:<br><br>
                <strong>1. Social (People)</strong><br>
                • Impact on employees, communities, and society<br>
                • Fair labor practices, community engagement<br>
                • Employee well-being and development<br><br>
                <strong>2. Economic (Profit)</strong><br>
                • Financial performance and sustainability<br>
                • Long-term profitability, not just short-term gains<br>
                • Value creation for stakeholders<br><br>
                <strong>3. Environmental (Planet)</strong><br>
                • Ecological footprint and sustainability<br>
                • Resource conservation, emissions reduction<br>
                • Sustainable sourcing and waste management<br><br>
                <em>Key Insight:</em> Companies that balance all three dimensions tend to have 
                more resilient supply chains and better long-term performance.
                """)
        
        # Problem 2 (v3.5 / v4.0)
        with st.expander("🟡 Problem 2: Efficiency vs. Effectiveness (Medium)"):
            display_practice_problem(2, "Medium",
                "A manufacturing company reduced its production costs by 15% but customer complaints increased by 25%. "
                "Analyze this situation using the concepts of efficiency and effectiveness. What went wrong?")
            
            show_hint2 = st.checkbox("Show Hint", key="risk_hint2")
            if show_hint2:
                display_hint("Efficiency = doing things right (low cost). Effectiveness = doing the right things (customer value).")
            
            if st.button("Show Complete Solution", key="risk_q2"):
                display_solution("""
                <strong>Analysis:</strong><br><br>
                <strong>Efficiency (Doing things right):</strong><br>
                • The company improved efficiency by reducing costs 15%<br>
                • This suggests process optimization or cost-cutting measures<br><br>
                <strong>Effectiveness (Doing the right things):</strong><br>
                • Customer complaints increased 25% - effectiveness decreased<br>
                • The cost cuts likely compromised quality or service<br><br>
                <strong>What Went Wrong:</strong><br>
                • The company optimized for efficiency at the expense of effectiveness<br>
                • Cost reductions may have affected: material quality, inspection processes, 
                  customer service staffing, or delivery times<br>
                • This is a classic trade-off failure<br><br>
                <strong>Recommendation:</strong><br>
                • Balance both metrics - find cost savings that don't impact customer value<br>
                • Use customer feedback to identify which cost cuts caused problems<br>
                • Remember: A highly efficient process that produces the wrong output is worthless
                """)
        
        # Problem 3 (v4.0)
        with st.expander("🔴 Problem 3: Risk Quantification (Hard)"):
            display_practice_problem(3, "Hard",
                """A company has identified three major supply chain risks:
                • Risk A: Probability = 0.15, Impact = $2,000,000
                • Risk B: Probability = 0.30, Impact = $500,000
                • Risk C: Probability = 0.05, Impact = $10,000,000
                
                Calculate the Expected Monetary Value (EMV) for each risk and determine which risk 
                should receive the highest mitigation priority. Also calculate the total risk exposure.""")
            
            st.markdown("#### Your Calculations:")
            col1, col2, col3 = st.columns(3)
            with col1:
                user_emv_a = st.number_input("EMV Risk A ($)", value=0, key="risk_emv_a")
            with col2:
                user_emv_b = st.number_input("EMV Risk B ($)", value=0, key="risk_emv_b")
            with col3:
                user_emv_c = st.number_input("EMV Risk C ($)", value=0, key="risk_emv_c")
            
            user_total = st.number_input("Total Risk Exposure ($)", value=0, key="risk_total")
            user_priority = st.selectbox("Highest Priority Risk", ["Select...", "Risk A", "Risk B", "Risk C"], key="risk_priority")
            
            if st.button("Check My Answers", key="risk_q3_check"):
                correct_a = 0.15 * 2000000
                correct_b = 0.30 * 500000
                correct_c = 0.05 * 10000000
                correct_total = correct_a + correct_b + correct_c
                
                results = []
                if check_answer(user_emv_a, correct_a): results.append("✅ EMV A correct")
                else: results.append(f"❌ EMV A: Should be ${correct_a:,.0f}")
                
                if check_answer(user_emv_b, correct_b): results.append("✅ EMV B correct")
                else: results.append(f"❌ EMV B: Should be ${correct_b:,.0f}")
                
                if check_answer(user_emv_c, correct_c): results.append("✅ EMV C correct")
                else: results.append(f"❌ EMV C: Should be ${correct_c:,.0f}")
                
                if check_answer(user_total, correct_total): results.append("✅ Total correct")
                else: results.append(f"❌ Total: Should be ${correct_total:,.0f}")
                
                max_emv = max(correct_a, correct_b, correct_c)
                correct_priority = "Risk C" if max_emv == correct_c else "Risk A" if max_emv == correct_a else "Risk B"
                
                if user_priority == correct_priority: results.append(f"✅ Priority correct: {correct_priority}")
                else: results.append(f"❌ Priority: Should be {correct_priority}")
                
                for r in results:
                    st.write(r)
            
            if st.button("Show Complete Solution", key="risk_q3"):
                display_solution(f"""
                <strong>Step 1: Calculate EMV for each risk</strong><br>
                EMV = Probability × Impact<br><br>
                • EMV(A) = 0.15 × $2,000,000 = <strong>${0.15*2000000:,.0f}</strong><br>
                • EMV(B) = 0.30 × $500,000 = <strong>${0.30*500000:,.0f}</strong><br>
                • EMV(C) = 0.05 × $10,000,000 = <strong>${0.05*10000000:,.0f}</strong><br><br>
                <strong>Step 2: Total Risk Exposure</strong><br>
                Total = $300,000 + $150,000 + $500,000 = <strong>$950,000</strong><br><br>
                <strong>Step 3: Priority Ranking</strong><br>
                1. Risk C ($500,000 EMV) - Highest priority<br>
                2. Risk A ($300,000 EMV)<br>
                3. Risk B ($150,000 EMV)<br><br>
                <strong>Key Insight:</strong> Even though Risk C has the lowest probability (5%), 
                its high impact makes it the highest priority.
                """)
        
        # Problem 4 (v4.0)
        with st.expander("🟡 Problem 4: Straddling Strategy (Medium)"):
            display_practice_problem(4, "Medium",
                "What is 'Straddling' in competitive strategy? Provide an example of a company that failed due to straddling.")
            
            if st.button("Show Complete Solution", key="risk_q4"):
                display_solution("""
                <strong>Straddling Definition:</strong><br>
                Straddling occurs when a company seeks to match the benefits of a successful 
                competitive position while maintaining its existing position. This often leads 
                to failure due to conflicting processes and trade-offs.<br><br>
                <strong>Why Straddling Fails:</strong><br>
                • Conflicting operational requirements<br>
                • Diluted brand positioning<br>
                • Increased complexity without clear benefits<br>
                • "Stuck in the middle" - neither low-cost nor differentiated<br><br>
                <strong>Classic Example:</strong><br>
                Continental Airlines tried to compete with Southwest Airlines by creating 
                "Continental Lite" - a low-cost subsidiary. The result:<br>
                • Confused customers about the brand<br>
                • Operational conflicts between full-service and low-cost models<br>
                • Neither service excelled<br>
                • Eventually abandoned the strategy<br><br>
                <strong>Lesson:</strong> Companies must make clear strategic choices and accept 
                the trade-offs that come with their chosen position.
                """)

        # Problem 5 (v3.5 - Restored to prevent deletion)
        with st.expander("🟡 Problem 5: What-If Sensitivity Analysis (Medium)"):
            display_practice_problem(5, "Medium",
                """Based on the textbook's guidance on sensitivity analysis, explain what happens 
            to project profitability if:
            1. Development time increases by 25%
            2. Sales volume decreases by 25%
            3. Product cost increases by $1 per unit""")
            
            if st.button("Show Analysis", key="risk_q5"):
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

# ============================================================
# MODULE 2: PERT NETWORK (Chapter 4) - MERGED
# ============================================================
def module_pert():
    display_header("🔗", "Chapter 4", "PERT Network Diagram & Completion Probability", 
                   "Critical path identification, slack calculation & Z-score probability")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Activity Estimator", "📊 Probability Calculator", "🎓 Practice"])
    
    with tab1:
        st.markdown("### PERT Network Analysis")
        st.write("""
        **PERT (Program Evaluation and Review Technique)** is a project management tool that uses 
        probabilistic time estimates to account for uncertainty in activity durations.
        """)
        
        display_citation(
            "A conservative approach dictates using the critical path with the largest total variance "
            "to focus management's attention on the activities most likely to exhibit broad variations.",
            "Jacobs & Chase (2024, p. 99)"
        )
        
        st.markdown("### Key Formulas")
        
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("PERT Expected Time", r"T_E = \frac{a + 4m + b}{6}")
            st.write("Where: a = optimistic, m = most likely, b = pessimistic")
        
        with col2:
            display_formula_card("PERT Variance", r"\sigma^2 = \left(\frac{b - a}{6}\right)^2")
            st.write("Variance of each activity's time estimate")
        
        st.markdown("### Project Completion Probability")
        display_formula_card("Z-Score for Project Completion", r"Z = \frac{D - T_E}{\sqrt{\sum \sigma^2_{cp}}}")
        
        st.write("""
        Where:
        - **D** = Desired completion time (target deadline)
        - **T_E** = Expected project duration (sum of critical path activities)
        - **Σσ²_cp** = Sum of variances on the critical path
        """)
        
        display_textbook_content(
            "Understanding the Beta Distribution",
            """PERT uses the beta distribution because it can model asymmetric uncertainty. 
            The formula T_E = (a + 4m + b)/6 is an approximation of the mean of a beta distribution. 
            The weight of 4 on the most likely estimate reflects that this value has the highest 
            probability of occurring."""
        )
        
        display_key_insight(
            "Critical Path Selection",
            "When there are two critical paths of equal length, use the one with the largest "
            "total variance for probability calculations to be conservative."
        )
        
        st.markdown("### Standard Normal Distribution Table (Excerpt)")
        z_data = []
        for z in [0.0, 0.5, 1.0, 1.28, 1.65, 1.96, 2.0, 2.33, 2.5, 3.0]:
            z_data.append({"Z-Score": z, "P(Z ≤ z)": f"{normal_cdf(z):.4f}", "P(Z > z)": f"{1-normal_cdf(z):.4f}"})
        st.dataframe(pd.DataFrame(z_data), use_container_width=True)
    
    with tab2:
        st.markdown("### Activity Time Estimator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            a = st.slider("Optimistic Time (a)", 1, 20, 4)
            m = st.slider("Most Likely Time (m)", 1, 30, 8)
            b = st.slider("Pessimistic Time (b)", 1, 40, 16)
            
            if not (a <= m <= b):
                st.warning("⚠️ PERT estimates should satisfy a ≤ m ≤ b")
        
        with col2:
            te = (a + 4*m + b) / 6
            variance = ((b - a) / 6) ** 2
            std_dev = math.sqrt(variance)
            
            st.markdown("#### Results")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                display_metric_card(f"{te:.2f}", "Expected Time (Tₑ)", "highlight")
            with col_b:
                display_metric_card(f"{variance:.2f}", "Variance (σ²)", "normal")
            with col_c:
                display_metric_card(f"{std_dev:.2f}", "Std Dev (σ)", "normal")
            
            st.markdown("#### Step-by-Step Calculation")
            st.latex(rf"T_E = \frac{{{a} + 4({m}) + {b}}}{{6}} = \frac{{{a + 4*m + b}}}{{6}} = {te:.2f}")
            st.latex(rf"\sigma^2 = \left(\frac{{{b} - {a}}}{{6}}\right)^2 = \left(\frac{{{b-a}}}{{6}}\right)^2 = {variance:.2f}")
    
    with tab3:
        st.markdown("### Project Completion Probability Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            te_project = st.number_input("Expected Project Duration (Tₑ)", value=38.0, step=0.5)
            d_target = st.number_input("Desired Completion (D)", value=35.0, step=0.5)
            sum_variance = st.number_input("Sum of CP Variances (Σσ²)", value=11.89, step=0.1)
        
        with col2:
            if sum_variance > 0:
                z_score = (d_target - te_project) / math.sqrt(sum_variance)
                prob = normal_cdf(z_score)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    display_metric_card(f"{z_score:.2f}", "Z-Score", "normal")
                with col_b:
                    card_type = "danger" if prob < 0.5 else "success"
                    display_metric_card(f"{prob*100:.1f}%", "P(Complete by D)", card_type)
                
                st.markdown("#### Calculation Steps")
                st.latex(rf"Z = \frac{{{d_target} - {te_project}}}{{\sqrt{{{sum_variance}}}}} = \frac{{{d_target - te_project:.2f}}}{{{math.sqrt(sum_variance):.2f}}} = {z_score:.2f}")
                
                if prob < 0.5:
                    st.warning(f"""
                    ⚠️ Only a **{prob*100:.1f}%** chance of completing in {d_target} weeks. 
                    The project manager should plan for {te_project}+ weeks or crash critical activities.
                    """)
                else:
                    st.success(f"✅ Good probability ({prob*100:.1f}%) of meeting the deadline.")
        
        # Variance Builder
        st.markdown("---")
        st.markdown("### Critical Path Variance Builder")
        
        num_activities = st.number_input("Number of CP Activities", 1, 10, 3, key="pert_cp_num")
        
        activities = []
        cols = st.columns(4)
        cols[0].write("**Activity**")
        cols[1].write("**a**")
        cols[2].write("**m**")
        cols[3].write("**b**")
        
        total_te = 0
        total_var = 0
        
        for i in range(int(num_activities)):
            cols = st.columns(4)
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
            activities.append({"Activity": chr(65+i), "a": a_i, "m": m_i, "b": b_i, "Tₑ": round(te_i, 2), "σ²": round(var_i, 3)})
        
        df = pd.DataFrame(activities)
        st.dataframe(df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card(f"{total_te:.2f}", "Total Path Duration", "highlight")
        with col2:
            display_metric_card(f"{total_var:.3f}", "Total Path Variance", "normal")
        with col3:
            display_metric_card(f"{math.sqrt(total_var):.3f}", "Path Standard Deviation", "normal")
    
    with tab4:
        st.markdown("### 📝 Enhanced Practice Problems")
        
        # Problem 1
        with st.expander("🟢 Problem 1: Calculate Expected Time (Easy)"):
            display_practice_problem(1, "Easy", 
                "Given: a = 5 days, m = 8 days, b = 17 days. Calculate the expected time (Tₑ).")
            
            user_te = st.number_input("Your Answer (Tₑ):", key="pert_p1", format="%.2f")
            
            show_hint = st.checkbox("Show Hint", key="pert_hint1")
            if show_hint:
                display_hint("Use the formula: Tₑ = (a + 4m + b) / 6")
            
            if st.button("Check Answer", key="pert_p1_btn"):
                correct = (5 + 4*8 + 17) / 6
                if check_answer(user_te, correct):
                    st.success(f"✅ Correct! Tₑ = (5 + 4×8 + 17)/6 = {correct:.2f} days")
                else:
                    st.error(f"❌ Incorrect. Let's work through it:")
                    display_solution(f"""
                    T_E = (a + 4m + b) / 6<br>
                    T_E = (5 + 4×8 + 17) / 6<br>
                    T_E = (5 + 32 + 17) / 6<br>
                    T_E = 54 / 6 = <strong>{correct:.2f} days</strong>
                    """)
        
        # Problem 2
        with st.expander("🟢 Problem 2: Calculate Variance (Easy)"):
            display_practice_problem(2, "Easy",
                "Given: a = 3 days, b = 15 days. Calculate the variance (σ²).")
            
            user_var = st.number_input("Your Answer (σ²):", key="pert_p2", format="%.2f")
            
            if st.button("Check Answer", key="pert_p2_btn"):
                correct = ((15 - 3) / 6) ** 2
                if check_answer(user_var, correct):
                    st.success(f"✅ Correct! σ² = ((15-3)/6)² = {correct:.2f}")
                else:
                    st.error(f"❌ Incorrect.")
                    display_solution(f"""
                    σ² = ((b - a) / 6)²<br>
                    σ² = ((15 - 3) / 6)²<br>
                    σ² = (12 / 6)²<br>
                    σ² = 2² = <strong>{correct:.2f}</strong>
                    """)
        
        # Problem 3
        with st.expander("🟡 Problem 3: Project Completion Probability (Medium)"):
            display_practice_problem(3, "Medium",
                """A project has the following critical path activities:
                
                | Activity | a | m | b |
                |----------|---|---|---|
                | A | 2 | 4 | 6 |
                | B | 3 | 5 | 13 |
                | C | 4 | 6 | 8 |
                
                Calculate: (a) Expected project duration, (b) Total variance, 
                (c) Probability of completing in 17 days.""")
            
            st.markdown("#### Your Calculations:")
            col1, col2, col3 = st.columns(3)
            with col1:
                user_duration = st.number_input("Expected Duration:", key="pert_p3_dur", format="%.2f")
            with col2:
                user_variance = st.number_input("Total Variance:", key="pert_p3_var", format="%.2f")
            with col3:
                user_prob = st.number_input("P(Complete ≤ 17):", key="pert_p3_prob", format="%.1f")
            
            if st.button("Check All Answers", key="pert_p3_btn"):
                te_a = (2 + 4*4 + 6) / 6
                te_b = (3 + 4*5 + 13) / 6
                te_c = (4 + 4*6 + 8) / 6
                total_te = te_a + te_b + te_c
                
                var_a = ((6-2)/6)**2
                var_b = ((13-3)/6)**2
                var_c = ((8-4)/6)**2
                total_var = var_a + var_b + var_c
                
                z = (17 - total_te) / math.sqrt(total_var)
                prob = normal_cdf(z) * 100
                
                results = []
                if check_answer(user_duration, total_te): results.append(f"✅ Duration correct: {total_te:.2f} days")
                else: results.append(f"❌ Duration: Should be {total_te:.2f} days")
                
                if check_answer(user_variance, total_var): results.append(f"✅ Variance correct: {total_var:.2f}")
                else: results.append(f"❌ Variance: Should be {total_var:.2f}")
                
                if check_answer(user_prob, prob, tolerance=0.02): results.append(f"✅ Probability correct: {prob:.1f}%")
                else: results.append(f"❌ Probability: Should be {prob:.1f}%")
                
                for r in results: st.write(r)
            
            if st.button("Show Complete Solution", key="pert_p3_sol"):
                te_a = (2 + 4*4 + 6) / 6
                te_b = (3 + 4*5 + 13) / 6
                te_c = (4 + 4*6 + 8) / 6
                total_te = te_a + te_b + te_c
                
                var_a = ((6-2)/6)**2
                var_b = ((13-3)/6)**2
                var_c = ((8-4)/6)**2
                total_var = var_a + var_b + var_c
                
                z = (17 - total_te) / math.sqrt(total_var)
                prob = normal_cdf(z) * 100
                
                display_solution(f"""
                <strong>Step 1: Calculate Expected Times</strong><br>
                T_E(A) = (2 + 4×4 + 6)/6 = {te_a:.2f} days<br>
                T_E(B) = (3 + 4×5 + 13)/6 = {te_b:.2f} days<br>
                T_E(C) = (4 + 4×6 + 8)/6 = {te_c:.2f} days<br>
                <strong>Total Duration = {total_te:.2f} days</strong><br><br>
                
                <strong>Step 2: Calculate Variances</strong><br>
                σ²(A) = ((6-2)/6)² = {var_a:.3f}<br>
                σ²(B) = ((13-3)/6)² = {var_b:.3f}<br>
                σ²(C) = ((8-4)/6)² = {var_c:.3f}<br>
                <strong>Total Variance = {total_var:.3f}</strong><br><br>
                
                <strong>Step 3: Calculate Z-Score</strong><br>
                Z = (17 - {total_te:.2f}) / √{total_var:.3f}<br>
                Z = {17 - total_te:.2f} / {math.sqrt(total_var):.3f} = {z:.2f}<br><br>
                
                <strong>Step 4: Find Probability</strong><br>
                P(Z ≤ {z:.2f}) = <strong>{prob:.1f}%</strong>
                """)
        
        # Problem 4
        with st.expander("🔴 Problem 4: Multiple Critical Paths (Hard)"):
            display_practice_problem(4, "Hard",
                """A project has two potential critical paths of equal length (20 days):
                
                Path 1: Activities A-B-C with total variance = 4.0
                Path 2: Activities D-E-F with total variance = 9.0
                
                Management wants to know the probability of completing in 18 days.
                Which path should be used for the calculation and why? Calculate the probability.""")
            
            if st.button("Show Complete Solution", key="pert_p4"):
                z_path2 = (18 - 20) / math.sqrt(9)
                prob = normal_cdf(z_path2) * 100
                
                display_solution(f"""
                <strong>Which Path to Use?</strong><br>
                Use <strong>Path 2</strong> (variance = 9.0) because:<br>
                • When paths have equal expected duration, use the one with LARGER variance<br>
                • This is the conservative approach - it gives the lower probability<br>
                • Higher variance means more uncertainty, so we plan for the worst case<br><br>
                
                <strong>Calculation:</strong><br>
                Z = (D - T_E) / √(Σσ²)<br>
                Z = (18 - 20) / √9<br>
                Z = -2 / 3 = {z_path2:.2f}<br><br>
                
                P(Z ≤ {z_path2:.2f}) = <strong>{prob:.1f}%</strong><br><br>
                
                <strong>Interpretation:</strong><br>
                There is only a {prob:.1f}% chance of completing in 18 days. 
                Management should either extend the deadline or allocate additional 
                resources to crash critical activities.
                """)

# ============================================================
# MODULE 3: PROJECT CRASHING (Chapter 4) - MERGED
# ============================================================
def module_crashing():
    display_header("⚡", "Chapter 4", "Project Crashing", "Time-cost trade-off analysis")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice"])
    
    with tab1:
        st.markdown("### Project Crashing Theory")
        st.write("""
        **Project crashing** involves reducing project duration by adding resources to critical 
        path activities. The goal is to minimize the total cost of crashing while meeting a 
        target completion date.
        """)
        
        display_formula_card("Crash Cost per Time Unit", 
            r"\text{Crash Cost/Day} = \frac{\text{Crash Cost} - \text{Normal Cost}}{\text{Normal Time} - \text{Crash Time}}")
        
        st.markdown("### Crashing Procedure")
        st.write("""
        1. **Identify the critical path** - Only crash activities on the critical path
        2. **Calculate crash cost per day** for each critical activity
        3. **Crash the cheapest activity first** - Select the activity with lowest crash cost/day
        4. **Crash until**:
           - Target duration is reached, OR
           - Activity reaches its crash limit, OR
           - A new critical path emerges (then evaluate both paths)
        5. **Repeat** until target is met or no more crashing is possible
        """)
        
        display_key_insight(
            "Crashing Strategy",
            "Always crash the activity on the critical path with the lowest crash cost per day first. "
            "Continue until the target date is reached or no more crashing is possible. "
            "Watch for new critical paths that may emerge!"
        )
        
        display_textbook_content(
            "When to Crash",
            """Crashing is appropriate when:
            • The project is behind schedule and penalties apply
            • Early completion offers bonuses or competitive advantage
            • Resources need to be freed for other projects
            • Market conditions require faster delivery
            
            Remember: Crashing increases direct costs but may reduce indirect costs (overhead, 
            penalties) and opportunity costs."""
        )
    
    with tab2:
        st.markdown("### Crash Cost Calculator")
        
        num_activities = st.number_input("Number of Activities", 2, 10, 3)
        
        activities = []
        for i in range(int(num_activities)):
            st.markdown(f"**Activity {chr(65+i)}**")
            cols = st.columns(5)
            with cols[0]:
                nt = st.number_input("Normal Time", value=5+i, key=f"crash_nt_{i}")
            with cols[1]:
                ct = st.number_input("Crash Time", value=3+i, key=f"crash_ct_{i}")
            with cols[2]:
                nc = st.number_input("Normal Cost ($)", value=1000+i*200, key=f"crash_nc_{i}")
            with cols[3]:
                cc = st.number_input("Crash Cost ($)", value=1800+i*300, key=f"crash_cc_{i}")
            with cols[4]:
                if nt > ct:
                    cpd = (cc - nc) / (nt - ct)
                    st.metric("Cost/Day", f"${cpd:.0f}")
                else:
                    cpd = float('inf')
                    st.metric("Cost/Day", "N/A")
            
            activities.append({
                "Activity": chr(65+i),
                "Normal Time": nt,
                "Crash Time": ct,
                "Normal Cost": nc,
                "Crash Cost": cc,
                "Max Crash Days": nt - ct,
                "Cost/Day": cpd if nt > ct else None
            })
        
        df = pd.DataFrame(activities)
        st.dataframe(df, use_container_width=True)
        
        # Summary
        total_normal_time = sum(a["Normal Time"] for a in activities)
        total_crash_time = sum(a["Crash Time"] for a in activities)
        total_crash_cost = sum(a["Crash Cost"] - a["Normal Cost"] for a in activities)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            display_metric_card(f"{total_normal_time} days", "Normal Duration", "normal")
        with col2:
            display_metric_card(f"{total_crash_time} days", "Minimum Duration", "highlight")
        with col3:
            display_metric_card(f"${total_crash_cost:,.0f}", "Full Crash Cost", "danger")
        
        # Crashing recommendation
        valid_activities = [a for a in activities if a["Cost/Day"] and a["Cost/Day"] != float('inf')]
        if valid_activities:
            cheapest = min(valid_activities, key=lambda x: x["Cost/Day"])
            st.success(f"💡 **Recommendation:** Crash Activity {cheapest['Activity']} first (${cheapest['Cost/Day']:.0f}/day)")
    
    with tab3:
        st.markdown("### 📝 Enhanced Practice Problems")
        
        with st.expander("🟡 Problem 1: Crash Cost Calculation (Medium)"):
            display_practice_problem(1, "Medium",
                """Activity X has:
                • Normal Time = 10 days, Normal Cost = $5,000
                • Crash Time = 6 days, Crash Cost = $9,000
                
                Calculate the crash cost per day.""")
            
            user_cpd = st.number_input("Crash Cost per Day ($):", key="crash_p1")
            
            if st.button("Check Answer", key="crash_p1_btn"):
                correct = (9000 - 5000) / (10 - 6)
                if check_answer(user_cpd, correct):
                    st.success(f"✅ Correct! Crash Cost/Day = ${correct:,.0f}")
                else:
                    display_solution(f"""
                    Crash Cost/Day = (Crash Cost - Normal Cost) / (Normal Time - Crash Time)<br>
                    = ($9,000 - $5,000) / (10 - 6)<br>
                    = $4,000 / 4 days<br>
                    = <strong>${correct:,.0f}/day</strong>
                    """)
        
        with st.expander("🔴 Problem 2: Crashing Decision (Hard)"):
            display_practice_problem(2, "Hard",
                """A project has a critical path A-B-C with normal duration of 15 days.
                The client offers a $2,000 bonus for each day the project finishes early.
                
                | Activity | Normal Time | Crash Time | Crash Cost/Day |
                |----------|-------------|------------|----------------|
                | A | 5 | 3 | $800 |
                | B | 6 | 4 | $1,500 |
                | C | 4 | 3 | $2,500 |
                
                How many days should you crash, and what is the net benefit?""")
            
            if st.button("Show Complete Solution", key="crash_p2"):
                display_solution("""
                <strong>Analysis:</strong><br>
                Bonus = $2,000/day saved<br><br>
                
                <strong>Step 1: Compare crash costs to bonus</strong><br>
                • Activity A: $800/day < $2,000 bonus → Profitable to crash<br>
                • Activity B: $1,500/day < $2,000 bonus → Profitable to crash<br>
                • Activity C: $2,500/day > $2,000 bonus → NOT profitable<br><br>
                
                <strong>Step 2: Crash in order of lowest cost</strong><br>
                1. Crash A by 2 days: Cost = 2 × $800 = $1,600, Benefit = 2 × $2,000 = $4,000<br>
                   Net = $4,000 - $1,600 = <strong>$2,400</strong><br><br>
                2. Crash B by 2 days: Cost = 2 × $1,500 = $3,000, Benefit = 2 × $2,000 = $4,000<br>
                   Net = $4,000 - $3,000 = <strong>$1,000</strong><br><br>
                3. Don't crash C (cost > benefit)<br><br>
                
                <strong>Total:</strong><br>
                • Days crashed: 4 days (A: 2, B: 2)<br>
                • New duration: 15 - 4 = 11 days<br>
                • Total crash cost: $1,600 + $3,000 = $4,600<br>
                • Total bonus: 4 × $2,000 = $8,000<br>
                • <strong>Net benefit: $8,000 - $4,600 = $3,400</strong>
                """)

# ============================================================
# MODULE 4: BREAK-EVEN ANALYSIS (Chapter 5) - MERGED
# ============================================================
def module_breakeven():
    display_header("📈", "Chapter 5", "Break-Even Analysis", "Cost-Volume-Profit (CVP) Analysis")
    
    # Fully merged tabs including Sensitivity from v3.5
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
        
        st.markdown("### Key Formulas")
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Break-Even Point (Units)", r"BEP_{units} = \frac{F}{P - V}")
            st.write("Where F = Fixed Costs, P = Price, V = Variable Cost per unit")
        with col2:
            display_formula_card("Break-Even Point (Revenue)", r"BEP_{\$} = \frac{F}{1 - \frac{V}{P}}")
        
        display_formula_card("Volume for Target Profit", r"Q_{target} = \frac{F + \text{Target Profit}}{P - V}")
        
        st.markdown("### Additional Formulas")
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Contribution Margin", r"CM = P - V")
        with col2:
            display_formula_card("Contribution Margin Ratio", r"CM\% = \frac{P - V}{P}")
        
        display_key_insight(
            "Comparing Capacity Alternatives",
            "When comparing two process alternatives (e.g., manual vs. automated), the option with "
            "higher fixed costs but lower variable costs will have a higher BEP but becomes more "
            "profitable at higher volumes."
        )
        
        display_formula_card("Indifference Point (Two Alternatives)", r"Q^* = \frac{F_2 - F_1}{V_1 - V_2}")
    
    with tab2:
        st.markdown("### Break-Even Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            fixed_cost = st.slider("Fixed Costs ($)", 10000, 200000, 50000, 5000)
            price = st.slider("Price per Unit ($)", 10, 500, 100, 5)
            variable_cost = st.slider("Variable Cost per Unit ($)", 5, 400, 60, 5)
        
        with col2:
            if price > variable_cost:
                bep_units = fixed_cost / (price - variable_cost)
                bep_revenue = bep_units * price
                contribution_margin = price - variable_cost
                cm_ratio = contribution_margin / price
                
                st.metric("Break-Even Units", f"{bep_units:,.0f}")
                st.metric("Break-Even Revenue", f"${bep_revenue:,.0f}")
                st.metric("Contribution Margin", f"${contribution_margin:.2f}/unit")
                st.metric("CM Ratio", f"{cm_ratio:.1%}")
                
                st.markdown("#### Calculation")
                st.latex(rf"BEP = \frac{{{fixed_cost:,}}}{{{price} - {variable_cost}}} = {bep_units:,.0f} \text{{ units}}")
            else:
                st.error("⚠️ Price must be greater than Variable Cost!")
        
        # Target Profit Calculator
        st.markdown("---")
        st.markdown("### Target Profit Analysis")
        target_profit = st.number_input("Target Annual Profit ($)", value=25000, step=5000)
        
        if price > variable_cost:
            target_units = (fixed_cost + target_profit) / (price - variable_cost)
            target_revenue = target_units * price
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Required Units", f"{target_units:,.0f}")
            with col2:
                st.metric("Required Revenue", f"${target_revenue:,.0f}")
            
            st.latex(rf"Q_{{target}} = \frac{{{fixed_cost:,} + {target_profit:,}}}{{{price} - {variable_cost}}} = {target_units:,.0f}")
            
    with tab3:
        # Restored Sensitivity Analysis from v3.5
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
                display_metric_card(f"{bep_a:,.0f} units", "BEP A", "highlight")
        
        with col2:
            st.markdown("#### Scenario B (Alternative)")
            fc_b = st.number_input("Fixed Costs B ($)", value=70000, key="be_fc_b")
            p_b = st.number_input("Price B ($)", value=100, key="be_p_b")
            vc_b = st.number_input("Variable Cost B ($)", value=50, key="be_vc_b")
            
            if p_b > vc_b:
                bep_b = fc_b / (p_b - vc_b)
                display_metric_card(f"{bep_b:,.0f} units", "BEP B", "highlight")
        
        # Indifference Point
        if p_a > vc_a and p_b > vc_b and vc_a != vc_b:
            indiff_point = (fc_b - fc_a) / (vc_a - vc_b)
            st.markdown("---")
            st.markdown("### Indifference Analysis")
            st.metric("Indifference Point", f"{indiff_point:,.0f} units")
            
            st.latex(rf"Q^* = \frac{{{fc_b:,} - {fc_a:,}}}{{{vc_a} - {vc_b}}} = {indiff_point:,.0f}")
            
            if indiff_point > 0:
                st.info(f"""
                📊 **Analysis:**
                - Below {indiff_point:,.0f} units: Choose {'A' if fc_a < fc_b else 'B'} (lower fixed costs)
                - Above {indiff_point:,.0f} units: Choose {'B' if vc_b < vc_a else 'A'} (lower variable costs)
                """)
    
    with tab5:
        st.markdown("### 📝 Enhanced Practice Problems")
        
        with st.expander("🟢 Problem 1: Basic BEP Calculation (Easy)"):
            display_practice_problem(1, "Easy",
                "Fixed Costs = $40,000, Price = $120/unit, Variable Cost = $80/unit. Calculate the break-even point in units.")
            
            user_bep = st.number_input("Your Answer (units):", key="be_p1")
            
            if st.button("Check Answer", key="be_p1_btn"):
                correct = 40000 / (120 - 80)
                if check_answer(user_bep, correct):
                    st.success(f"✅ Correct! BEP = {correct:,.0f} units")
                else:
                    display_solution(f"""
                    BEP = F / (P - V)<br>
                    BEP = $40,000 / ($120 - $80)<br>
                    BEP = $40,000 / $40<br>
                    BEP = <strong>{correct:,.0f} units</strong>
                    """)
        
        with st.expander("🟡 Problem 2: Target Profit (Medium)"):
            display_practice_problem(2, "Medium",
                """A company has:
                • Fixed Costs = $60,000
                • Price = $50/unit
                • Variable Cost = $30/unit
                • Target Profit = $20,000
                
                How many units must be sold to achieve the target profit?""")
            
            user_target = st.number_input("Your Answer (units):", key="be_p2")
            
            if st.button("Check Answer", key="be_p2_btn"):
                correct = (60000 + 20000) / (50 - 30)
                if check_answer(user_target, correct):
                    st.success(f"✅ Correct! {correct:,.0f} units")
                else:
                    display_solution(f"""
                    Q = (F + Target Profit) / (P - V)<br>
                    Q = ($60,000 + $20,000) / ($50 - $30)<br>
                    Q = $80,000 / $20<br>
                    Q = <strong>{correct:,.0f} units</strong>
                    """)

        # V3.5 Restored Manual vs Auto Indifference Problem
        with st.expander("🔴 Problem 3: Indifference Point (Hard)"):
            display_practice_problem(3, "Hard",
                """Two manufacturing options:
                
                | Option | Fixed Costs | Variable Cost/Unit |
                |--------|-------------|-------------------|
                | Manual | $20,000 | $15 |
                | Automated | $80,000 | $5 |
                
                1. Indifference point (where both options have equal total cost)
                2. Which option is better at 5,000 units?
                3. Which option is better at 8,000 units?""")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                ans_indiff = st.number_input("Indifference point:", key="be_p3_1")
            with col2:
                ans_5000 = st.selectbox("Better at 5,000:", ["Manual", "Automated"], key="be_p3_2")
            with col3:
                ans_8000 = st.selectbox("Better at 8,000:", ["Manual", "Automated"], key="be_p3_3")
            
            if st.button("Check Answer", key="be_p3_btn_orig"):
                correct_indiff = (80000 - 20000) / (15 - 5)
                tc_manual_5000 = 20000 + 15 * 5000
                tc_auto_5000 = 80000 + 5 * 5000
                tc_manual_8000 = 20000 + 15 * 8000
                tc_auto_8000 = 80000 + 5 * 8000
                
                st.markdown("#### Solution:")
                display_solution_step(1, f"Indifference Point = (F₂ - F₁) / (V₁ - V₂) = ($80,000 - $20,000) / ($15 - $5) = $60,000 / $10 = <strong>{correct_indiff:,.0f} units</strong>")
                display_solution_step(2, f"At 5,000 units:<br>• Manual: $20,000 + $15×5,000 = ${tc_manual_5000:,}<br>• Automated: $80,000 + $5×5,000 = ${tc_auto_5000:,}<br><strong>Manual is better</strong>")
                display_solution_step(3, f"At 8,000 units:<br>• Manual: $20,000 + $15×8,000 = ${tc_manual_8000:,}<br>• Automated: $80,000 + $5×8,000 = ${tc_auto_8000:,}<br><strong>Automated is better</strong>")

        # V4.0 Make vs Buy Indifference Problem
        with st.expander("🔴 Problem 4: Make vs. Buy Decision (Hard)"):
            display_practice_problem(4, "Hard",
                """A company is deciding between making a component in-house or buying it:
                
                **Make Option:**
                • Fixed Cost = $100,000/year (equipment)
                • Variable Cost = $15/unit
                
                **Buy Option:**
                • Purchase Price = $25/unit (no fixed costs)
                
                a) At what volume are the two options equivalent?
                b) If expected demand is 8,000 units, which option is better and by how much?""")
            
            col1, col2 = st.columns(2)
            with col1:
                user_indiff = st.number_input("Indifference Point (units):", key="be_p4a")
            with col2:
                user_better = st.selectbox("Better option at 8,000 units:", ["Select...", "Make", "Buy"], key="be_p4b")
            
            user_savings = st.number_input("Savings amount ($):", key="be_p4c")
            
            if st.button("Check All Answers", key="be_p4_btn"):
                correct_indiff = 100000 / (25 - 15)
                make_cost_8000 = 100000 + 15 * 8000
                buy_cost_8000 = 25 * 8000
                correct_better = "Buy" if buy_cost_8000 < make_cost_8000 else "Make"
                correct_savings = abs(make_cost_8000 - buy_cost_8000)
                
                results = []
                if check_answer(user_indiff, correct_indiff): results.append(f"✅ Indifference point correct: {correct_indiff:,.0f} units")
                else: results.append(f"❌ Indifference point: Should be {correct_indiff:,.0f} units")
                
                if user_better == correct_better: results.append(f"✅ Better option correct: {correct_better}")
                else: results.append(f"❌ Better option: Should be {correct_better}")
                
                if check_answer(user_savings, correct_savings): results.append(f"✅ Savings correct: ${correct_savings:,.0f}")
                else: results.append(f"❌ Savings: Should be ${correct_savings:,.0f}")
                
                for r in results:
                    st.write(r)
            
            if st.button("Show Complete Solution", key="be_p4_sol"):
                display_solution("""
                <strong>Part a) Indifference Point</strong><br>
                Set Make Cost = Buy Cost<br>
                $100,000 + $15Q = $25Q<br>
                $100,000 = $10Q<br>
                Q = <strong>10,000 units</strong><br><br>
                
                <strong>Part b) At 8,000 units</strong><br>
                Make Cost = $100,000 + $15(8,000) = $100,000 + $120,000 = $220,000<br>
                Buy Cost = $25(8,000) = $200,000<br><br>
                
                <strong>Buy is better by $220,000 - $200,000 = $20,000</strong><br><br>
                
                <em>Note: Since 8,000 < 10,000 (indifference point), Buy is preferred 
                because it has lower fixed costs.</em>
                """)

# ============================================================
# MODULE 5: DECISION TREES (Chapter 5) - MERGED
# ============================================================
def module_decision():
    display_header("🌳", "Chapter 5", "Decision Trees & Expected Monetary Value", 
                   "Structured decision-making under uncertainty")
    
    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice"])
    
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
            st.markdown("**◻️ Decision Node**")
            st.write("Point where decision maker chooses between alternatives")
        with col2:
            st.markdown("**⭕ Chance Node**")
            st.write("Point where chance determines outcome (probabilities sum to 1)")
        with col3:
            st.markdown("**🔺 Terminal Node**")
            st.write("Final payoff at end of branch")
        
        st.markdown("### Key Formulas")
        display_formula_card("Expected Monetary Value", r"EMV = \sum_{i=1}^{n} (P_i \times V_i)")
        st.write("**Decision rule:** Select the alternative with the highest EMV")
        
        display_formula_card("Expected Value of Perfect Information", r"EVPI = EV_{with\ PI} - EV_{without\ PI}")
        
        display_key_insight(
            "Roll Back Method",
            "Decision trees are solved from right to left (backward induction). At each chance node, "
            "calculate the EMV. At each decision node, select the alternative with the highest EMV."
        )
    
    with tab2:
        st.markdown("### EMV Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Large Facility Option")
            prob_high = st.slider("P(High Demand)", 0, 100, 60) / 100
            payoff_high_large = st.number_input("High Demand Payoff ($)", value=200000, key="dt_h_l")
            payoff_low_large = st.number_input("Low Demand Payoff ($)", value=-50000, key="dt_l_l")
            
            emv_large = prob_high * payoff_high_large + (1 - prob_high) * payoff_low_large
            st.metric("EMV (Large)", f"${emv_large:,.0f}")
            
            st.latex(rf"EMV_L = {prob_high:.2f} \times {payoff_high_large:,} + {1-prob_high:.2f} \times ({payoff_low_large:,})")
        
        with col2:
            st.markdown("#### Small Facility Option")
            payoff_high_small = st.number_input("High Demand Payoff ($)", value=90000, key="dt_h_s")
            payoff_low_small = st.number_input("Low Demand Payoff ($)", value=25000, key="dt_l_s")
            
            emv_small = prob_high * payoff_high_small + (1 - prob_high) * payoff_low_small
            st.metric("EMV (Small)", f"${emv_small:,.0f}")
            
            st.latex(rf"EMV_S = {prob_high:.2f} \times {payoff_high_small:,} + {1-prob_high:.2f} \times {payoff_low_small:,}")
        
        st.markdown("---")
        st.markdown("### Recommendation")
        
        if emv_large > emv_small:
            st.success(f"✅ **Choose Large Facility** (EMV ${emv_large:,.0f} > ${emv_small:,.0f})")
        else:
            st.success(f"✅ **Choose Small Facility** (EMV ${emv_small:,.0f} > ${emv_large:,.0f})")
        
        # EVPI Calculation
        st.markdown("---")
        st.markdown("### Expected Value of Perfect Information (EVPI)")
        
        ev_with_pi = prob_high * max(payoff_high_large, payoff_high_small) + \
                     (1 - prob_high) * max(payoff_low_large, payoff_low_small)
        ev_without_pi = max(emv_large, emv_small)
        evpi = ev_with_pi - ev_without_pi
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("EV with Perfect Info", f"${ev_with_pi:,.0f}")
        with col2:
            st.metric("EV without Perfect Info", f"${ev_without_pi:,.0f}")
        with col3:
            st.metric("EVPI", f"${evpi:,.0f}")
        
        st.info(f"💡 You should pay at most **${evpi:,.0f}** for perfect market information.")
    
    with tab3:
        st.markdown("### 📝 Enhanced Practice Problems")
        
        with st.expander("🟢 Problem 1: Basic EMV (Easy)"):
            display_practice_problem(1, "Easy",
                """Calculate the EMV for Option A:
                • 40% chance of $100,000
                • 60% chance of $20,000""")
            
            user_emv = st.number_input("EMV ($):", key="dt_p1")
            
            if st.button("Check Answer", key="dt_p1_btn"):
                correct = 0.4 * 100000 + 0.6 * 20000
                if check_answer(user_emv, correct):
                    st.success(f"✅ Correct! EMV = ${correct:,.0f}")
                else:
                    display_solution(f"""
                    EMV = P₁ × V₁ + P₂ × V₂<br>
                    EMV = 0.40 × $100,000 + 0.60 × $20,000<br>
                    EMV = $40,000 + $12,000<br>
                    EMV = <strong>${correct:,.0f}</strong>
                    """)
        
        with st.expander("🟡 Problem 2: Decision Comparison (Medium)"):
            display_practice_problem(2, "Medium",
                """Compare two investment options:
                
                **Option A:** 40% chance of $100,000, 60% chance of $20,000
                **Option B:** 50% chance of $80,000, 50% chance of $30,000
                
                Which option has the higher EMV?""")
            
            if st.button("Show Solution", key="dt_p2"):
                emv_a = 0.4 * 100000 + 0.6 * 20000
                emv_b = 0.5 * 80000 + 0.5 * 30000
                display_solution(f"""
                <strong>Option A:</strong><br>
                EMV(A) = 0.4 × $100,000 + 0.6 × $20,000<br>
                EMV(A) = $40,000 + $12,000 = <strong>${emv_a:,.0f}</strong><br><br>
                
                <strong>Option B:</strong><br>
                EMV(B) = 0.5 × $80,000 + 0.5 × $30,000<br>
                EMV(B) = $40,000 + $15,000 = <strong>${emv_b:,.0f}</strong><br><br>
                
                <strong>Decision: Choose Option {'A' if emv_a > emv_b else 'B'}</strong> 
                (${max(emv_a, emv_b):,.0f} > ${min(emv_a, emv_b):,.0f})
                """)
        
        with st.expander("🔴 Problem 3: EVPI Calculation (Hard)"):
            display_practice_problem(3, "Hard",
                """Using the options from Problem 2:
                
                **Option A:** 40% chance of $100,000, 60% chance of $20,000
                **Option B:** 50% chance of $80,000, 50% chance of $30,000
                
                Assume the probabilities represent the same market conditions (high/low demand).
                Calculate the Expected Value of Perfect Information (EVPI).""")
            
            user_evpi = st.number_input("EVPI ($):", key="dt_p3")
            
            if st.button("Check Answer", key="dt_p3_btn"):
                ev_with_pi = 0.4 * 100000 + 0.6 * 30000
                ev_without_pi = max(0.4*100000 + 0.6*20000, 0.5*80000 + 0.5*30000)
                correct_evpi = ev_with_pi - ev_without_pi
                
                if check_answer(user_evpi, correct_evpi):
                    st.success(f"✅ Correct! EVPI = ${correct_evpi:,.0f}")
                else:
                    display_solution(f"""
                    <strong>Step 1: EV with Perfect Information</strong><br>
                    If we knew the market state in advance:<br>
                    • High demand (40%): Choose A ($100,000 > $80,000)<br>
                    • Low demand (60%): Choose B ($30,000 > $20,000)<br><br>
                    EV_PI = 0.40 × $100,000 + 0.60 × $30,000<br>
                    EV_PI = $40,000 + $18,000 = ${ev_with_pi:,.0f}<br><br>
                    
                    <strong>Step 2: EV without Perfect Information</strong><br>
                    EMV(A) = ${0.4*100000 + 0.6*20000:,.0f}<br>
                    EMV(B) = ${0.5*80000 + 0.5*30000:,.0f}<br>
                    Best EMV = ${ev_without_pi:,.0f}<br><br>
                    
                    <strong>Step 3: EVPI</strong><br>
                    EVPI = ${ev_with_pi:,.0f} - ${ev_without_pi:,.0f} = <strong>${correct_evpi:,.0f}</strong>
                    """)

# ============================================================
# MODULE 6: LEARNING CURVES (Chapter 6) - MERGED
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
        
        st.markdown("### Key Formulas")
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Unit Time Formula", r"Y_x = K \cdot x^n")
            st.write("Where Yₓ = time for unit x, K = time for first unit, n = learning exponent")
        
        with col2:
            display_formula_card("Learning Exponent", r"n = \frac{\log(b)}{\log(2)}")
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
        
        # Retain cumulative calculation from v3.5
        cumulative_time = k * (target_unit ** (n + 1)) / (n + 1) if n != -1 else k * math.log(target_unit)
        avg_time = cumulative_time / target_unit if target_unit > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Time for Unit {target_unit}", f"{unit_time:.1f} hrs")
        with col2:
            st.metric("Cumulative Time (1 to X)", f"{cumulative_time:.1f} hrs")
        with col3:
            st.metric("Cumulative Average", f"{avg_time:.1f} hrs")
            
        st.latex(rf"Y_{{{target_unit}}} = {k} \times {target_unit}^{{{n:.3f}}} = {unit_time:.1f}")
    
    with tab3:
        st.markdown("### 📝 Enhanced Practice Problems")
        
        with st.expander("🟢 Problem 1: Calculate Unit Time (Easy)"):
            display_practice_problem(1, "Easy",
                "First unit takes 100 hours. With an 80% learning curve, how long will unit 8 take?")
            
            user_ans = st.number_input("Your Answer (hours):", key="lc_p1")
            
            show_hint = st.checkbox("Show Hint", key="lc_hint1")
            if show_hint:
                display_hint("Unit 8 is the 3rd doubling (1→2→4→8). Each doubling multiplies by 0.80.")
            
            if st.button("Check Answer", key="lc_p1_btn"):
                n = math.log(0.8) / math.log(2)
                correct = 100 * (8 ** n)
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Y₈ = {correct:.1f} hours")
                else:
                    display_solution(f"""
                    <strong>Method 1: Using the formula</strong><br>
                    n = log(0.80) / log(2) = {n:.3f}<br>
                    Y₈ = 100 × 8^{n:.3f} = <strong>{correct:.1f} hours</strong><br><br>
                    
                    <strong>Method 2: Doubling approach</strong><br>
                    Unit 1: 100 hours<br>
                    Unit 2: 100 × 0.80 = 80 hours<br>
                    Unit 4: 80 × 0.80 = 64 hours<br>
                    Unit 8: 64 × 0.80 = <strong>51.2 hours</strong>
                    """)

# ============================================================
# MODULE 7: DECOUPLING POINT (Chapter 7) - FULL V3.5 RESTORE
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
# MODULE 8: LINE BALANCING (Chapter 8) - FULL V3.5 RESTORE + V4.0 Cards
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
# MODULE 9: SERVICE DESIGN (Chapter 9) - FULL V3.5 RESTORE
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
# MODULE 10: QUEUING THEORY (Chapter 10) - FULL V3.5 RESTORE + V4.0 Elements
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
        
        with st.expander("🟡 Problem: Calculate Lq"):
            st.write("λ = 12/hr, μ = 18/hr. Calculate Lq for M/M/1.")
            user_ans = st.number_input("Your Answer:", key="q_p1", format="%.2f")
            if st.button("Check Answer", key="q_p1_btn"):
                correct = (12**2) / (18 * (18 - 12))
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Lq = 12²/(18×6) = {correct:.2f}")
                else:
                    st.error(f"❌ Incorrect. Lq = 12²/(18×6) = {correct:.2f}")

# ============================================================
# MODULE 11: DISTRIBUTIONS (Chapter 10) - FULL V3.5 RESTORE
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
# MODULE 12: LITTLE'S LAW (Chapter 11) - FULL V3.5 RESTORE + V4.0 Cards
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
        
        display_formula_card("Little's Law", r"I = R \times T")
        st.write("Where I = Inventory (WIP), R = Throughput Rate, T = Flow Time")
        
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
# MODULE 13: SIX SIGMA / DPMO (Chapter 12) - FULL V3.5 RESTORE + V4.0
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
        
        display_formula_card("DPMO", r"DPMO = \frac{\text{Total Defects}}{\text{Total Opportunities}} \times 1,000,000")
        
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
# MODULE 14: FMEA (Chapter 12) - FULL V3.5 RESTORE + V4.0
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
        
        display_formula_card("Risk Priority Number", r"RPN = \text{Severity} \times \text{Occurrence} \times \text{Detection}")
        
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
# MODULE 15: SQC - CONTROL CHARTS (Chapter 13) - FULL V3.5 RESTORE + V4.0
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
            display_formula_card("Center Line", r"\bar{p} = \frac{\text{Total Defectives}}{\text{Total Inspected}}")
            display_formula_card("Standard Error", r"S_p = \sqrt{\frac{\bar{p}(1-\bar{p})}{n}}")
            display_formula_card("Upper Control Limit", r"UCL = \bar{p} + 3S_p")
            display_formula_card("Lower Control Limit", r"LCL = \bar{p} - 3S_p")
        
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
        
        if n > 0:
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
# MODULE 16: PROCESS CAPABILITY (Chapter 13) - FULL V3.5 RESTORE + V4.0
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
            display_formula_card("Cp", r"C_p = \frac{USL - LSL}{6\sigma}")
        
        with col2:
            st.markdown("#### Cpk (Actual)")
            display_formula_card("Cpk", r"C_{pk} = \min\left(\frac{USL - \bar{X}}{3\sigma}, \frac{\bar{X} - LSL}{3\sigma}\right)")
        
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
# MODULE 17: ACCEPTANCE SAMPLING (Chapter 13) - FULL V3.5 RESTORE
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
            p_accept = sum([poisson_pmf(k, np_val) for k in range(int(c) + 1)])
            
            st.metric("P(Accept Lot)", f"{p_accept:.3f}")
            
            st.info(f"""
            📊 **Sampling Plan (n={n}, c={c})**
            - At p = {p:.1%} defective, probability of acceptance = {p_accept:.1%}
            """)

# ============================================================
# MODULE 18: PARETO ANALYSIS (Chapter 13) - FULL V3.5 RESTORE
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
# MODULE 19: FISHBONE DIAGRAM (Chapter 13) - FULL V3.5 RESTORE
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
# MODULE 20: LEAN SUPPLY CHAINS (Chapter 14) - FULL V3.5 RESTORE + V4.0 Cards
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
            
        display_textbook_content(
            "Value Stream Mapping",
            """VSM is a great visual way to analyze an existing system and to find areas where waste 
            can be eliminated. Value stream maps are simple to draw, and it is possible to construct 
            the maps totally with paper and pencil."""
        )
    
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
            
            reduction_pct = (1 - future_lt/current_lt) * 100 if current_lt > 0 else 0
            
            st.metric("Future Lead Time", f"{future_lt:.1f} days")
            st.metric("Lead Time Reduction", f"{reduction_pct:.0f}%")
            
            display_citation(
                "Note that the lead time in the new system is only five days, compared to the "
                "34-day lead time with the old system.",
                "Jacobs & Chase (2024)"
            )
    
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
# MODULE 21: CENTROID METHOD (Chapter 15) - FULL V3.5 RESTORE + V4.0 Cards
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
        
        display_formula_card("X Coordinate", r"C_x = \frac{\sum(d_{ix} \times V_i)}{\sum V_i}")
        display_formula_card("Y Coordinate", r"C_y = \frac{\sum(d_{iy} \times V_i)}{\sum V_i}")
    
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
        if total_v > 0:
            cx = sum(loc["x"] * loc["v"] for loc in locations) / total_v
            cy = sum(loc["y"] * loc["v"] for loc in locations) / total_v
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimal X (Cx)", f"{cx:.1f}")
            with col2:
                st.metric("Optimal Y (Cy)", f"{cy:.1f}")

# ============================================================
# MODULE 22: FACTOR RATING (Chapter 15) - FULL V3.5 RESTORE + V4.0 Cards
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
        
        display_formula_card("Weighted Score", r"\text{Score} = \sum_{i=1}^{n} (w_i \times s_i)")
    
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
# MODULE 23: TRANSPORTATION METHOD (Chapter 15) - FULL V3.5 RESTORE + V4.0 Cards
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
        
        display_formula_card("Objective Function", r"\text{Minimize } Z = \sum_{i}\sum_{j} c_{ij} \cdot x_{ij}")
        
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
# MODULE 24: GLOBAL SOURCING (Chapter 16) - FULL V3.5 RESTORE
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
# MODULE 25: FORECASTING (Chapter 18) - FULL V3.5 RESTORE + V4.0 Cards
# ============================================================
def module_forecast():
    display_header("📈", "Chapter 18", "Enhanced Forecasting", "WMA, Holt's, Seasonal & Tracking Signal")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Weighted MA", "📈 Holt's Method", "🌊 Seasonal Index", "📡 Tracking Signal"])
    
    with tab1:
        st.markdown("### Weighted Moving Average")
        display_formula_card("Weighted Moving Average", r"F_t = \frac{\sum_{i=1}^{n} w_i \cdot A_{t-i}}{\sum_{i=1}^{n} w_i}")
        
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
        display_formula_card("Tracking Signal", r"TS = \frac{RSFE}{MAD}")
        
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
# MODULE 26: REGRESSION (Chapter 18) - FULL V3.5 RESTORE + V4.0 Cards
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
        
        display_formula_card("Regression Line", r"Y = a + bx")
        display_formula_card("Slope", r"b = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
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
# MODULE 27: AGGREGATE PLANNING (Chapter 19) - FULL V3.5 RESTORE + V4.0
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
# MODULE 28: EOQ (Chapter 20) - FULL V3.5 RESTORE + V4.0 Elements
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
        
        display_formula_card("Total Cost", r"TC = \frac{D}{Q} \cdot S + \frac{Q}{2} \cdot H")
        display_formula_card("Optimal Order Quantity", r"Q^* = \sqrt{\frac{2DS}{H}}")
        display_formula_card("Minimum Total Cost", r"TC_{min} = \sqrt{2DSH}")
        
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
        display_formula_card("Optimal Production Lot Size", r"Q^*_p = \sqrt{\frac{2DS}{H(1-d/p)}}")
        
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
        st.markdown("### 📝 Enhanced Practice Problems")
        
        with st.expander("🟢 Problem 1: Calculate Q* (Easy)"):
            display_practice_problem(1, "Easy", "D = 5,000, S = $100, H = $4. Calculate optimal Q*.")
            user_ans = st.number_input("Your Answer:", key="eoq_p1")
            
            if st.button("Check Answer", key="eoq_p1_btn"):
                correct = math.sqrt(2 * 5000 * 100 / 4)
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Q* = {correct:.0f}")
                else:
                    display_solution(f"""
                    Q* = √(2DS / H)<br>
                    Q* = √(2 × 5000 × 100 / 4)<br>
                    Q* = <strong>{correct:.0f} units</strong>
                    """)

# ============================================================
# MODULE 29: SAFETY STOCK (Chapter 20) - FULL V3.5 RESTORE + V4.0
# ============================================================
def module_safetystock():
    display_header("🛡️", "Chapter 20", "Safety Stock & Reorder Point", 
                   "Protecting against demand and lead time variability")
    
    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Calculator"])
    
    with tab1:
        st.markdown("### Safety Stock Theory")
        
        display_formula_card("Safety Stock", r"SS = z \cdot \sigma_d \cdot \sqrt{LT}")
        display_formula_card("Reorder Point", r"ROP = \bar{d} \times LT + SS")
        
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
# MODULE 30: NEWSVENDOR (Chapter 20) - FULL V3.5 RESTORE + V4.0
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
        
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Cost of Understocking", r"C_u = \text{Price} - \text{Cost}")
            display_formula_card("Cost of Overstocking", r"C_o = \text{Cost} - \text{Salvage}")
        with col2:
            display_formula_card("Critical Ratio", r"P \leq \frac{C_u}{C_u + C_o}")
            display_formula_card("Optimal Quantity", r"Q^* = \mu + z\sigma")
    
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
# MODULE 31: MRP (Chapter 21) - FULL V3.5 RESTORE + V4.0
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
        
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Net Requirements", r"\text{Net} = \text{Gross} - (\text{OH} + \text{SR})")
        with col2:
            display_formula_card("Release Period", r"\text{Release} = \text{Receipt Period} - \text{LT}")
        
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
# MODULE 32: MRP LOT SIZING (Chapter 21) - FULL V3.5 RESTORE
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
# MODULE 33: JOB SCHEDULING (Chapter 22) - FULL V3.5 RESTORE + V4.0
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
        
        display_formula_card("Critical Ratio", r"CR = \frac{\text{Due Date} - \text{Today}}{\text{Processing Time}}")
        
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
        st.markdown("### 📝 Enhanced Practice Problems")
        
        with st.expander("🟢 Problem 1: Flow Time Calculation (Easy)"):
            display_practice_problem(1, "Easy", "Which rule mathematically minimizes the average flow time for a set of jobs?")
            
            if st.button("Show Answer", key="sch_p1"):
                display_solution("**SPT (Shortest Processing Time)** minimizes average flow time by ensuring that the shortest jobs get out of the system quickly, preventing them from waiting behind long jobs.")

# ============================================================
# MODULE 34: POKA-YOKE (Chapter 9) - FULL V3.5 RESTORE
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
# MODULE 35: SQC PRACTICE (Chapter 13) - FULL V3.5 RESTORE + V4.0
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
            display_practice_problem(i+1, "Medium", q)
            if st.button(f"Show Answer", key=f"sqc_prac_{i}"):
                display_solution(a)

# ============================================================
# MODULE 36: PRACTICE PROBLEMS (General) - FULL V3.5 RESTORE + V4.0
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
            display_practice_problem(i+1, "Medium", f"**{title}**: {question}")
            if st.button(f"Show Solution", key=f"prac_{i}"):
                display_solution(answer)

# ============================================================
# SIDEBAR NAVIGATION & MAIN APP RUNNER (V4.0 Structure)
# ============================================================
def main():
    # Theme toggle in sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 2rem;">📊</div>
        <div style="font-weight: 700; font-size: 1.2rem;">OSCM Simulator</div>
        <div style="font-size: 0.8rem; opacity: 0.8;">v4.0 Enhanced Edition</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Theme toggle button
    theme_label = "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"
    if st.sidebar.button(theme_label, key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Statistics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Modules", "40")
    with col2:
        st.metric("Formulas", "75+")
    
    st.sidebar.markdown("""
    <div style="font-size: 0.75rem; padding: 0.5rem; border-radius: 6px; margin: 0.5rem 0;">
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