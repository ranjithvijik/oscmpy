# ===========================================================# IMPORTS & COMPATIBILITY PATCHES
# ============================================================
import math
import sys
import logging
import warnings
from collections import OrderedDict
from functools import lru_cache

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import scipy
import plotly.graph_objects as go
import plotly.express as px

# ── Optional imports (graceful degradation) ──────────────────
try:
    import plotly.figure_factory as ff
    _HAS_FF = True
except ImportError:
    _HAS_FF = False

try:
    from scipy.optimize import linprog
    _HAS_LINPROG = True
except ImportError:
    _HAS_LINPROG = False

try:
    from scipy.special import factorial
    _HAS_FACTORIAL = True
except ImportError:
    from math import factorial as _factorial
    _HAS_FACTORIAL = False

# ── Pandas Styler patch (.applymap removed in pandas ≥ 2.2) ──
try:
    from pandas.io.formats.style import Styler as _Styler
    if not hasattr(_Styler, "applymap"):
        _Styler.applymap = _Styler.map
except (ImportError, AttributeError):
    pass

# ── NumPy legacy scalar alias patch (removed in NumPy ≥ 2.0) ─
for _alias, _builtin in {
    "bool": bool, "int": int, "float": float,
    "complex": complex, "object": object, "str": str,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

# ── Runtime version diagnostics ──────────────────────────────
_VERSIONS = {
    "python":    sys.version.split()[0],
    "streamlit": st.__version__,
    "pandas":    pd.__version__,
    "numpy":     np.__version__,
    "scipy":     scipy.__version__,   
    "plotly":    px.__version__,
}
logging.info("OSCM runtime: %s", _VERSIONS)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="OSCM Simulator – Enhanced Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com",
        "Report a bug": "https://github.com",
        "About": (
            "📊 **OSCM Interactive Simulator**\n\n"
            "Based on Jacobs & Chase — *Operations and Supply Chain Management*, "
            "17th ed. (McGraw-Hill, 2024).\n\n"
            f"Python {_VERSIONS['python']} · "
            f"Streamlit {_VERSIONS['streamlit']} · "
            f"Pandas {_VERSIONS['pandas']}"
        ),
    },
)

# ============================================================
# SESSION STATE INITIALIZER
# ============================================================
def init_session_state():
    """
    Initialize all session-state keys exactly once per session.
    Grouped by concern so new keys are easy to find and add.
    """
    defaults = {
        # ── Navigation ────────────────────────────────────────
        "selected_module":  None,       # None = show welcome screen
        "last_module":      None,
        "recent_modules":   [],         # list[str], max 5
        "modules_visited":  set(),      # set[str]
        "bookmarks":        set(),      # set[str]

        # ── Theme ─────────────────────────────────────────────
        "dark_mode":        False,

        # ── Gamification ──────────────────────────────────────
        "problems_solved":  0,
        "correct_streak":   0,
        "best_streak":      0,

        # ── Quiz state (per-module, cleared on navigation) ────
        "sqc_quiz_score":   0,
        "sqc_quiz_total":   0,
        "sqc_quiz_streak":  0,

        # ── Diagnostics ───────────────────────────────────────
        "app_load_count":   0,
        "runtime_versions": _VERSIONS,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Increment load counter each hard rerun
    st.session_state.app_load_count += 1


init_session_state()


# ============================================================
# THEME MANAGEMENT
# ============================================================
def toggle_theme():
    """Flip dark/light mode and clear any cached palette."""
    st.session_state.dark_mode = not st.session_state.dark_mode


def render_theme_toggle():
    """Render the sidebar theme toggle button."""
    label = "☀️ Switch to Light Mode" if st.session_state.dark_mode else "🌙 Switch to Dark Mode"
    if st.sidebar.button(label, key="theme_toggle", use_container_width=True):
        toggle_theme()
        st.rerun()


def is_dark() -> bool:
    """Convenience helper — use in modules instead of reading session state directly."""
    return st.session_state.get("dark_mode", False)


# ============================================================
# THEME COLOR PALETTES
# ============================================================
@lru_cache(maxsize=2)   # cache light + dark variants separately
def _get_palette_cached(dark: bool) -> dict:
    """
    Cached palette builder.  The cache is keyed on dark=True/False so
    theme toggles immediately invalidate the correct entry.

    All light-mode contrast ratios verified against WCAG AA (≥4.5:1 for
    normal text, ≥3:1 for large/UI text) on their respective backgrounds.
    """
    d = dark
    return {
        # ── Backgrounds ──────────────────────────────────────
        "bg_app":           "#0f172a" if d else "#f8fafc",
        "bg_card":          "#1e293b" if d else "#ffffff",
        "bg_secondary":     "#1e293b" if d else "#f1f5f9",
        "bg_input":         "#334155" if d else "#ffffff",
        "bg_code":          "#0f172a" if d else "#f1f5f9",

        # ── Text ─────────────────────────────────────────────
        # Light ratios on white: primary #0f172a≈19:1, secondary #374151≈9:1,
        # muted #4b5563≈7:1 — all WCAG AAA.
        "text_primary":     "#e2e8f0" if d else "#0f172a",
        "text_secondary":   "#94a3b8" if d else "#374151",
        "text_muted":       "#64748b" if d else "#4b5563",
        "text_inverse":     "#0f172a" if d else "#ffffff",   # text on accent bg

        # ── Borders ──────────────────────────────────────────
        "border":           "#334155" if d else "#e2e8f0",
        "border_strong":    "#475569" if d else "#cbd5e1",

        # ── Accent (Indigo) ──────────────────────────────────
        # Dark #818cf8 on #0f172a ≈ 7:1; Light #4f46e5 on white ≈ 5.9:1
        "accent":           "#818cf8" if d else "#4f46e5",
        "accent_hover":     "#6366f1" if d else "#3730a3",
        "accent_soft":      "rgba(129,140,248,0.15)" if d else "rgba(99,102,241,0.08)",

        # ── Semantic: Success (Emerald) ──────────────────────
        # Light: #15803d on #f0fdf4 ≈ 7.2:1
        "success_bg":       "#022c22" if d else "#f0fdf4",
        "success_text":     "#86efac" if d else "#15803d",
        "success_border":   "#16a34a" if d else "#22c55e",

        # ── Semantic: Warning (Amber) ─────────────────────────
        # Light: #92400e on #fffbeb ≈ 7.5:1
        "warning_bg":       "#2d1a00" if d else "#fffbeb",
        "warning_text":     "#fde68a" if d else "#92400e",
        "warning_border":   "#d97706" if d else "#d97706",

        # ── Semantic: Danger (Red) ────────────────────────────
        # Light: #991b1b on #fef2f2 ≈ 7.1:1
        "danger_bg":        "#450a0a" if d else "#fef2f2",
        "danger_text":      "#fca5a5" if d else "#991b1b",
        "danger_border":    "#dc2626" if d else "#ef4444",

        # ── Semantic: Info (Blue) ─────────────────────────────
        # Light: #1d4ed8 on #eff6ff ≈ 7.3:1
        "info_bg":          "#0c1a3d" if d else "#eff6ff",
        "info_text":        "#93c5fd" if d else "#1d4ed8",
        "info_border":      "#3b82f6" if d else "#3b82f6",

        # ── Semantic: Tip (Teal) — new ────────────────────────
        "tip_bg":           "#042f2e" if d else "#f0fdfa",
        "tip_text":         "#5eead4" if d else "#0f766e",
        "tip_border":       "#0d9488" if d else "#14b8a6",

        # ── Citation box ─────────────────────────────────────
        # Light: #713f12 on #fefce8 ≈ 8.9:1
        "citation_bg":      "#1c1000" if d else "#fefce8",
        "citation_text":    "#fef08a" if d else "#713f12",
        "citation_border":  "#ca8a04" if d else "#92400e",

        # ── Equation box ─────────────────────────────────────
        "equation_bg":      "#071e2e" if d else "#f0f9ff",
        "equation_border":  "#0ea5e9" if d else "#7dd3fc",
        "equation_text":    "#e0f2fe" if d else "#0c4a6e",

        # ── Key Insight box ──────────────────────────────────
        # Light: #065f46 on gradient ending #d1fae5 ≈ 8.2:1
        "insight_bg":       "linear-gradient(135deg,#022c22 0%,#052e16 100%)" if d
                            else "linear-gradient(135deg,#ecfdf5 0%,#d1fae5 100%)",
        "insight_border":   "#16a34a" if d else "#16a34a",
        "insight_title":    "#4ade80" if d else "#065f46",

        # ── Textbook content box ─────────────────────────────
        # Light: #6d28d9 on #faf5ff ≈ 6.7:1
        "textbook_bg":      "#160d2a" if d else "#faf5ff",
        "textbook_border":  "#a78bfa" if d else "#8b5cf6",
        "textbook_h4":      "#c4b5fd" if d else "#6d28d9",

        # ── Solution / hint boxes ────────────────────────────
        # solution: #14532d on #f0fdf4 ≈ 9.1:1
        "solution_bg":      "#022c22" if d else "#f0fdf4",
        "solution_border":  "#16a34a" if d else "#22c55e",
        "solution_text":    "#bbf7d0" if d else "#14532d",

        # hint: #78350f on #fffbeb ≈ 7.5:1
        "hint_bg":          "#2d1a00" if d else "#fffbeb",
        "hint_border":      "#d97706" if d else "#d97706",
        "hint_text":        "#fde68a" if d else "#78350f",

        # ── Practice problem box ─────────────────────────────
        "practice_bg":      "#0c1a3d" if d else "#fafbff",
        "practice_border":  "#4f46e5" if d else "#4f46e5",

        # ── Code block ───────────────────────────────────────
        "code_bg":          "#0d1117" if d else "#f6f8fa",
        "code_text":        "#e6edf3" if d else "#24292f",
        "code_border":      "#30363d" if d else "#d0d7de",

        # ── Metric cards ─────────────────────────────────────
        "metric_grad_hi":   "linear-gradient(135deg,#4338ca 0%,#7c3aed 100%)",
        "metric_grad_lo":   ("linear-gradient(135deg,#1e293b 0%,#334155 100%)" if d
                             else "linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%)"),
        "metric_hi_text":   "#ffffff",
        "metric_lo_text":   "#e2e8f0" if d else "#0f172a",

        # Metric card semantic variants
        "metric_success_bg":    "#022c22" if d else "#f0fdf4",
        "metric_success_text":  "#4ade80" if d else "#15803d",
        "metric_danger_bg":     "#450a0a" if d else "#fef2f2",
        "metric_danger_text":   "#f87171" if d else "#991b1b",
        "metric_warning_bg":    "#2d1a00" if d else "#fffbeb",
        "metric_warning_text":  "#fbbf24" if d else "#92400e",

        # ── Chapter / progress UI ────────────────────────────
        "chapter_badge_bg":     "#312e81" if d else "#eef2ff",
        "chapter_badge_text":   "#a5b4fc" if d else "#3730a3",

        # ── Sidebar ───────────────────────────────────────────
        "sidebar_bg":       "#1e293b" if d else "#f8fafc",

        # ── Table ─────────────────────────────────────────────
        "table_head_bg":    "#334155" if d else "#f1f5f9",
        "table_row_alt":    "#1a2744" if d else "#f8fafc",

        # ── Scrollbar ─────────────────────────────────────────
        "scroll_track":     "#1e293b" if d else "#f1f5f9",
        "scroll_thumb":     "#475569" if d else "#cbd5e1",
    }


def _get_palette() -> dict:
    """Public accessor — always returns the palette for the current theme."""
    return _get_palette_cached(is_dark())

# ============================================================
# DYNAMIC CSS
# ============================================================
@lru_cache(maxsize=2)
def _get_theme_css_cached(dark: bool) -> str:
    """
    Build the full theme CSS string once per theme mode and cache it.
    Avoids regenerating hundreds of lines of CSS on every widget interaction.
    """
    p = _get_palette_cached(dark)
    return f"""
    <style>
    /* ── Google Font (Inter) ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ── CSS Custom Properties ───────────────────────────── */
    :root {{
        --bg-app:           {p['bg_app']};
        --bg-card:          {p['bg_card']};
        --bg-secondary:     {p['bg_secondary']};
        --text-primary:     {p['text_primary']};
        --text-secondary:   {p['text_secondary']};
        --text-muted:       {p['text_muted']};
        --border:           {p['border']};
        --border-strong:    {p['border_strong']};
        --accent:           {p['accent']};
        --accent-hover:     {p['accent_hover']};
        --accent-soft:      {p['accent_soft']};
        --radius-sm:        6px;
        --radius-md:        10px;
        --radius-lg:        14px;
        --radius-xl:        20px;
        --shadow-sm:        0 1px 4px rgba(0,0,0,0.08);
        --shadow-md:        0 4px 16px rgba(0,0,0,0.12);
        --shadow-lg:        0 8px 32px rgba(0,0,0,0.18);
        --shadow-accent:    0 4px 16px rgba(99,102,241,0.25);
        --transition:       all 0.2s ease;
        --font-sans:        'Inter', system-ui, -apple-system, sans-serif;
        --font-mono:        'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}

    /* ── Global Reset ────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; }}

    .stApp {{
        background-color: {p['bg_app']};
        color: {p['text_primary']};
        font-family: var(--font-sans);
        transition: background-color 0.3s ease, color 0.3s ease;
    }}

    /* ── Global Text Propagation ─────────────────────────── */
    .stApp p, .stApp li, .stApp span,
    .stMarkdown p, .stMarkdown li,
    .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4,
    .stMarkdown h5, .stMarkdown h6 {{
        color: {p['text_primary']};
    }}
    .stMarkdown strong, .stMarkdown b {{ color: {p['text_primary']}; font-weight: 700; }}
    .stMarkdown em, .stMarkdown i     {{ color: {p['text_secondary']}; }}
    .stMarkdown a                     {{ color: {p['accent']}; text-decoration: underline;
                                         text-underline-offset: 2px; }}
    .stMarkdown a:hover               {{ color: {p['accent_hover']}; }}
    .stMarkdown hr                    {{ border-color: {p['border']}; margin: 1.2rem 0; }}
    .stMarkdown blockquote            {{ border-left: 3px solid {p['accent']};
                                         padding-left: 1rem; color: {p['text_secondary']}; }}

    /* Caption / muted text */
    .stCaption, small,
    div[data-testid="stCaptionContainer"] p {{
        color: {p['text_muted']} !important;
        font-size: 0.82rem;
    }}

    /* Widget labels — all input types */
    label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    .stRadio label, .stCheckbox label,
    .stSelectbox label, .stSlider label,
    .stNumberInput label, .stTextInput label,
    .stTextArea label, .stMultiSelect label,
    .stDateInput label, .stTimeInput label {{
        color: {p['text_primary']} !important;
        font-weight: 500;
    }}
    .stRadio div[role="radiogroup"] label,
    .stCheckbox div[data-testid="stCheckbox"] label {{
        color: {p['text_primary']} !important;
    }}

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar              {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track        {{ background: {p['scroll_track']}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb        {{ background: {p['scroll_thumb']}; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover  {{ background: {p['accent']}; }}

    /* ── Module Header ───────────────────────────────────── */
    .main-header {{
        background: {p['metric_grad_hi']};
        padding: 1.5rem 1.8rem;
        border-radius: var(--radius-lg);
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-accent);
    }}
    .main-header::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at bottom left,
            rgba(0,0,0,0.12) 0%, transparent 65%);
        pointer-events: none;
    }}
    .main-header::after {{
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at top right,
            rgba(255,255,255,0.12) 0%, transparent 60%);
        pointer-events: none;
    }}
    .main-header h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 800;
        color: white !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
        position: relative; z-index: 1;
    }}
    .main-header p {{
        margin: 0.4rem 0 0;
        opacity: 0.92;
        color: white !important;
        font-size: 0.95rem;
        line-height: 1.5;
        position: relative; z-index: 1;
    }}
    .chapter-badge {{
        background: rgba(255,255,255,0.22);
        color: white;
        padding: 0.22rem 0.7rem;
        border-radius: var(--radius-xl);
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.5rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,0.3);
        display: inline-block;
        vertical-align: middle;
    }}

    /* ── Metric Cards ────────────────────────────────────── */
    .metric-card {{
        background: {p['metric_grad_lo']};
        border: 1px solid {p['border']};
        border-radius: var(--radius-lg);
        padding: 1.2rem;
        text-align: center;
        margin: 0.4rem 0;
        color: {p['metric_lo_text']};
        transition: var(--transition);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: {p['border']};
        transition: background 0.2s ease;
    }}
    .metric-card:hover {{
        box-shadow: var(--shadow-md);
        border-color: {p['accent']};
        transform: translateY(-1px);
    }}
    .metric-card:hover::before {{ background: {p['accent']}; }}
    .metric-card.highlight {{
        background: {p['metric_grad_hi']};
        color: {p['metric_hi_text']};
        border-color: {p['accent']};
        box-shadow: var(--shadow-accent);
    }}
    .metric-card.highlight::before {{ background: rgba(255,255,255,0.4); }}
    .metric-card.success {{
        background: {p['success_bg']};
        border-color: {p['success_border']};
        color: {p['success_text']};
    }}
    .metric-card.success::before  {{ background: {p['success_border']}; }}
    .metric-card.danger {{
        background: {p['danger_bg']};
        border-color: {p['danger_border']};
        color: {p['danger_text']};
    }}
    .metric-card.danger::before   {{ background: {p['danger_border']}; }}
    .metric-card.warning {{
        background: {p['warning_bg']};
        border-color: {p['warning_border']};
        color: {p['warning_text']};
    }}
    .metric-card.warning::before  {{ background: {p['warning_border']}; }}
    .metric-value {{
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    .metric-label {{
        font-size: 0.82rem;
        color: {p['text_secondary']};
        margin-top: 0.35rem;
        font-weight: 500;
    }}
    .metric-card.highlight .metric-label {{ color: rgba(255,255,255,0.8); }}
    .metric-delta             {{ font-size: 0.78rem; margin-top: 0.2rem; font-weight: 600; }}
    .metric-delta.positive    {{ color: {p['success_text']}; }}
    .metric-delta.negative    {{ color: {p['danger_text']}; }}

    /* ── Theory / Textbook Box ───────────────────────────── */
    .textbook-content {{
        background: {p['textbook_bg']};
        border-left: 4px solid {p['textbook_border']};
        padding: 1.2rem 1.4rem;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        margin: 1rem 0;
        color: {p['text_primary']};
        box-shadow: var(--shadow-sm);
    }}
    .textbook-content h4 {{
        color: {p['textbook_h4']};
        margin: 0 0 0.8rem;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .textbook-content p, .textbook-content li,
    .textbook-content span, .textbook-content div {{
        color: {p['text_primary']};
        line-height: 1.65;
    }}

    /* ── Citation Box ────────────────────────────────────── */
    .citation-box {{
        background: {p['citation_bg']};
        border-left: 4px solid {p['citation_border']};
        padding: 1rem 1.4rem;
        border-radius: 0 var(--radius-md) var(--radius-md) 0;
        margin: 1rem 0;
        font-style: italic;
        color: {p['citation_text']};
        font-size: 0.92rem;
        line-height: 1.7;
        position: relative;
    }}
    .citation-box::before {{
        content: "\201C";
        position: absolute;
        top: 0.2rem; left: 0.9rem;
        font-size: 2.5rem;
        color: {p['citation_border']};
        opacity: 0.35;
        font-family: Georgia, serif;
        line-height: 1;
    }}
    .citation-source {{
        display: block;
        margin-top: 0.6rem;
        font-style: normal;
        font-weight: 700;
        font-size: 0.82rem;
        color: {p['citation_border']};
        letter-spacing: 0.02em;
    }}

    /* ── Equation Box ────────────────────────────────────── */
    .equation-box {{
        background: {p['equation_bg']};
        border: 1px solid {p['equation_border']};
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem 0.6rem;
        margin: 0.8rem 0;
        text-align: center;
        color: {p['equation_text']};
    }}
    .equation-label {{
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {p['equation_text']};
        margin-bottom: 0.4rem;
        opacity: 0.85;
    }}

    /* ── Formula Card ────────────────────────────────────── */
    .formula-card {{
        background: {p['bg_secondary']};
        border: 2px solid {p['accent']};
        border-radius: var(--radius-md);
        padding: 0.9rem 1rem 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }}
    .formula-card::after {{
        content: "";
        position: absolute;
        inset: 0;
        background: {p['accent_soft']};
        opacity: 0;
        transition: opacity 0.2s ease;
        pointer-events: none;
    }}
    .formula-card:hover::after  {{ opacity: 1; }}
    .formula-card:hover         {{ border-color: {p['accent_hover']};
                                    box-shadow: 0 0 0 3px {p['accent_soft']}; }}
    .formula-title {{
        color: {p['accent_hover']};
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        position: relative; z-index: 1;
    }}

    /* ── Key Insight ─────────────────────────────────────── */
    .key-insight {{
        background: {p['insight_bg']};
        border: 1px solid {p['insight_border']};
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }}
    .key-insight-title {{
        font-weight: 700;
        color: {p['insight_title']};
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
    }}
    .key-insight-text {{
        color: {p['success_text']};
        font-size: 0.9rem;
        line-height: 1.65;
    }}

    /* ── Practice Problem ────────────────────────────────── */
    .practice-problem {{
        background: {p['practice_bg']};
        border: 2px solid {p['practice_border']};
        border-radius: var(--radius-lg);
        padding: 1.4rem 1.6rem;
        margin: 0.8rem 0;
        color: {p['text_primary']};
    }}
    .practice-problem h4 {{
        color: {p['accent_hover']};
        margin: 0 0 0.8rem;
        font-size: 1rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }}
    .practice-problem p {{
        color: {p['text_primary']};
        line-height: 1.65;
        margin: 0;
    }}

    /* ── Solution Box ────────────────────────────────────── */
    .solution-box {{
        background: {p['solution_bg']};
        border: 1px solid {p['solution_border']};
        border-left: 4px solid {p['solution_border']};
        border-radius: var(--radius-md);
        padding: 1.1rem 1.3rem;
        margin-top: 0.8rem;
        color: {p['solution_text']};
        line-height: 1.7;
        font-size: 0.92rem;
    }}

    /* ── Hint Box ────────────────────────────────────────── */
    .hint-box {{
        background: {p['hint_bg']};
        border: 1px solid {p['hint_border']};
        border-left: 3px solid {p['hint_border']};
        border-radius: var(--radius-md);
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        color: {p['hint_text']};
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    /* ── Callout Boxes ───────────────────────────────────── */
    .callout {{
        border-radius: var(--radius-md);
        padding: 0.9rem 1.2rem;
        margin: 0.75rem 0;
        line-height: 1.6;
        font-size: 0.9rem;
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
    }}
    .callout-icon    {{ font-size: 1.1rem; flex-shrink: 0; margin-top: 0.05rem; }}
    .callout-content {{ flex: 1; min-width: 0; }}
    .callout-title   {{
        font-weight: 700;
        margin-bottom: 0.25rem;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .callout.info    {{ background:{p['info_bg']};    border:1px solid {p['info_border']};
                        color:{p['info_text']};    border-left:4px solid {p['info_border']}; }}
    .callout.success {{ background:{p['success_bg']}; border:1px solid {p['success_border']};
                        color:{p['success_text']}; border-left:4px solid {p['success_border']}; }}
    .callout.warning {{ background:{p['warning_bg']}; border:1px solid {p['warning_border']};
                        color:{p['warning_text']}; border-left:4px solid {p['warning_border']}; }}
    .callout.danger  {{ background:{p['danger_bg']};  border:1px solid {p['danger_border']};
                        color:{p['danger_text']};  border-left:4px solid {p['danger_border']}; }}
    .callout.tip     {{ background:{p['tip_bg']};     border:1px solid {p['tip_border']};
                        color:{p['tip_text']};     border-left:4px solid {p['tip_border']}; }}

    /* ── Concept Cards ───────────────────────────────────── */
    .concept-card {{
        background: {p['bg_card']};
        border: 1px solid {p['border']};
        border-radius: var(--radius-lg);
        padding: 1.1rem;
        margin: 0.4rem 0;
        transition: var(--transition);
        height: 100%;
        display: flex;
        flex-direction: column;
    }}
    .concept-card:hover {{
        border-color: {p['accent']};
        box-shadow: 0 4px 16px {p['accent_soft']};
        transform: translateY(-2px);
    }}
    .concept-icon  {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
    .concept-title {{
        font-weight: 700;
        color: {p['text_primary']};
        margin-bottom: 0.3rem;
        font-size: 0.95rem;
    }}
    .concept-desc  {{
        font-size: 0.84rem;
        color: {p['text_secondary']};
        line-height: 1.55;
        flex: 1;
    }}

    /* ── Solution Steps ──────────────────────────────────── */
    .solution-step {{
        background: {p['bg_secondary']};
        border-left: 3px solid {p['accent']};
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        line-height: 1.65;
        color: {p['text_primary']};
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
    }}
    .step-number {{
        background: {p['accent']};
        color: white;
        min-width: 22px; height: 22px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center; justify-content: center;
        font-size: 0.72rem; font-weight: 800;
        flex-shrink: 0;
        margin-top: 0.1rem;
    }}

    /* ── Process Flow ────────────────────────────────────── */
    .process-flow {{
        display: flex;
        align-items: stretch;
        margin: 1rem 0;
        flex-wrap: wrap;
        gap: 0;
    }}
    .process-step {{
        flex: 1; min-width: 110px;
        background: {p['bg_secondary']};
        border: 1px solid {p['border']};
        padding: 0.9rem 0.6rem;
        text-align: center;
        position: relative;
        font-size: 0.82rem;
        transition: var(--transition);
    }}
    .process-step:hover {{ background: {p['accent_soft']}; }}
    .process-step:first-child {{ border-radius: var(--radius-sm) 0 0 var(--radius-sm); }}
    .process-step:last-child  {{ border-radius: 0 var(--radius-sm) var(--radius-sm) 0; }}
    .process-step:not(:last-child)::after {{
        content: "→";
        position: absolute;
        right: -0.7rem; top: 50%;
        transform: translateY(-50%);
        color: {p['accent']};
        font-size: 1rem; font-weight: 800;
        z-index: 2;
        background: {p['bg_app']};
        padding: 0 3px;
        border-radius: 2px;
    }}
    .process-step-num  {{ font-weight: 800; color: {p['accent']}; font-size: 1.05rem; }}
    .process-step-text {{ color: {p['text_primary']}; margin-top: 0.2rem; line-height: 1.3; }}

    /* ── Badges ──────────────────────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 0.18rem 0.6rem;
        border-radius: var(--radius-xl);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        line-height: 1.4;
        vertical-align: middle;
    }}
    .badge-accent  {{ background:{p['accent_soft']};  color:{p['accent_hover']};
                      border: 1px solid {p['accent']}; }}
    .badge-success {{ background:{p['success_bg']};   color:{p['success_text']};
                      border: 1px solid {p['success_border']}; }}
    .badge-warning {{ background:{p['warning_bg']};   color:{p['warning_text']};
                      border: 1px solid {p['warning_border']}; }}
    .badge-danger  {{ background:{p['danger_bg']};    color:{p['danger_text']};
                      border: 1px solid {p['danger_border']}; }}
    .badge-info    {{ background:{p['info_bg']};      color:{p['info_text']};
                      border: 1px solid {p['info_border']}; }}
    .badge-new     {{ background: linear-gradient(135deg,#16a34a,#15803d);
                      color: white; border: none; }}

    /* ── Chapter / Progress Summary Box ─────────────────── */
    .chapter-summary {{
        background: {p['bg_secondary']};
        border: 1px solid {p['border']};
        border-radius: var(--radius-lg);
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }}
    .chapter-summary-title {{
        font-weight: 800;
        font-size: 1rem;
        color: {p['accent']};
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}
    .chapter-summary ul {{ margin: 0; padding-left: 1.4rem; }}
    .chapter-summary li {{
        color: {p['text_primary']};
        font-size: 0.9rem;
        line-height: 1.7;
        margin-bottom: 0.15rem;
    }}
    .chapter-summary li::marker {{ color: {p['accent']}; }}

    /* ── Comparison Table (HTML) ─────────────────────────── */
    .comparison-card {{
        background: {p['bg_card']};
        border: 1px solid {p['border']};
        border-radius: var(--radius-lg);
        overflow: hidden;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-sm);
    }}
    .comparison-header {{
        background: {p['metric_grad_hi']};
        color: white;
        padding: 0.7rem 1rem;
        font-weight: 700;
        font-size: 0.9rem;
        text-align: center;
    }}
    .comparison-body  {{ padding: 0.9rem 1rem; }}
    .comparison-row   {{
        display: flex; justify-content: space-between;
        padding: 0.35rem 0; border-bottom: 1px solid {p['border']};
        font-size: 0.88rem;
    }}
    .comparison-row:last-child {{ border-bottom: none; }}
    .comparison-key   {{ color: {p['text_secondary']}; font-weight: 600; }}
    .comparison-val   {{ color: {p['text_primary']}; text-align: right; font-weight: 500; }}

    /* ── Progress Bar ────────────────────────────────────── */
    .progress-wrap {{
        background: {p['border_strong']};
        border-radius: 99px;
        height: 10px;
        overflow: hidden;
        margin: 0.4rem 0;
    }}
    .progress-fill             {{ height: 100%; border-radius: 99px; transition: width 0.6s ease; }}
    .progress-fill.accent      {{ background: {p['metric_grad_hi']}; }}
    .progress-fill.success     {{ background: linear-gradient(90deg,#16a34a,#4ade80); }}
    .progress-fill.danger      {{ background: linear-gradient(90deg,#dc2626,#f87171); }}
    .progress-fill.warning     {{ background: linear-gradient(90deg,#d97706,#fbbf24); }}

    /* ── Styled HTML Table ───────────────────────────────── */
    .styled-table {{
        width: 100%; border-collapse: collapse; margin: 0.8rem 0;
        font-size: 0.88rem; border-radius: var(--radius-md);
        overflow: hidden; box-shadow: var(--shadow-sm);
    }}
    .styled-table th {{
        background: {p['table_head_bg']}; color: {p['text_primary']};
        padding: 0.75rem 1rem; text-align: left; font-weight: 700;
        font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .styled-table td {{
        padding: 0.65rem 1rem; border-bottom: 1px solid {p['border']};
        color: {p['text_primary']}; vertical-align: top;
    }}
    .styled-table tr:nth-child(even) td {{ background: {p['table_row_alt']}; }}
    .styled-table tr:hover td {{
        background: {p['accent_soft']};
        transition: background 0.15s ease;
    }}

    /* ── Code Blocks ─────────────────────────────────────── */
    code {{
        background: {p['code_bg']}; color: {p['accent_hover']};
        padding: 0.15em 0.45em; border-radius: 4px;
        font-size: 0.87em; font-family: var(--font-mono);
        border: 1px solid {p['code_border']};
    }}
    pre {{
        background: {p['code_bg']}; border-radius: var(--radius-md);
        padding: 1rem; overflow-x: auto;
        border: 1px solid {p['code_border']};
    }}
    pre code {{
        color: {p['code_text']}; background: transparent;
        border: none; padding: 0; font-size: 0.9em;
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background-color: {p['sidebar_bg']};
        border-right: 1px solid {p['border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: {p['text_primary']} !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {p['border']};
        margin: 0.6rem 0;
    }}
    /* Sidebar nav buttons */
    section[data-testid="stSidebar"] .stButton button {{
        background: transparent;
        color: {p['text_primary']};
        border: 1px solid transparent;
        text-align: left;
        font-size: 0.85rem;
        transition: var(--transition);
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background: {p['accent_soft']};
        border-color: {p['accent']};
        color: {p['accent']};
    }}

    /* ── Streamlit Input Widgets ─────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {{
        background: {p['bg_input']};
        color: {p['text_primary']};
        border-color: {p['border']};
        border-radius: var(--radius-sm);
        transition: border-color 0.2s, box-shadow 0.2s;
        font-family: var(--font-sans);
    }}
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: {p['accent']};
        box-shadow: 0 0 0 3px {p['accent_soft']};
        outline: none;
    }}
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder,
    .stTextArea textarea::placeholder {{
        color: {p['text_muted']};
    }}
    .stSelectbox > div > div,
    .stMultiSelect > div > div {{
        background: {p['bg_input']};
        color: {p['text_primary']};
        border-color: {p['border']};
        border-radius: var(--radius-sm);
    }}

    /* ── Expanders ───────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {p['bg_secondary']};
        color: {p['text_primary']};
        border-radius: var(--radius-sm);
        border: 1px solid {p['border']};
        transition: var(--transition);
    }}
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span {{
        color: {p['text_primary']} !important;
        font-weight: 600;
    }}
    .streamlit-expanderHeader svg {{
        fill: {p['text_secondary']} !important;
    }}
    .streamlit-expanderHeader:hover {{
        border-color: {p['accent']};
        background: {p['accent_soft']};
    }}
    .streamlit-expanderHeader:hover p,
    .streamlit-expanderHeader:hover span {{
        color: {p['accent']} !important;
    }}

    /* ── Tabs ────────────────────────────────────────────── */
    button[data-baseweb="tab"] {{
        background: transparent;
        color: {p['text_secondary']};
        border-bottom: 2px solid transparent;
        font-weight: 600;
        font-size: 0.9rem;
        transition: var(--transition);
        padding-bottom: 0.5rem;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {p['accent']};
        background: {p['accent_soft']};
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {p['accent']};
        border-bottom-color: {p['accent']};
    }}
    div[data-testid="stTabs"] > div > div[role="tablist"] {{
        border-bottom: 1px solid {p['border']};
        gap: 0.25rem;
    }}

    /* ── Native st.metric ────────────────────────────────── */
    div[data-testid="stMetricValue"] {{
        font-size: 1.6rem;
        color: {p['text_primary']};
        font-weight: 800;
        letter-spacing: -0.02em;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {p['text_secondary']} !important;
        font-size: 0.85rem;
    }}
    div[data-testid="stMetricDelta"] {{ font-size: 0.82rem; font-weight: 600; }}

    /* ── Slider ──────────────────────────────────────────── */
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {{
        color: {p['accent']};
        font-weight: 700;
    }}
    div[data-testid="stSlider"] [role="slider"] {{
        background: {p['accent']} !important;
    }}

    /* ── DataFrame ───────────────────────────────────────── */
    .stDataFrame {{
        background: {p['bg_card']};
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid {p['border']};
    }}

    /* ── Plotly Container ────────────────────────────────── */
    .js-plotly-plot {{
        border-radius: var(--radius-md);
        overflow: hidden;
    }}

    /* ── Alert / Info boxes (Streamlit native) ───────────── */
    div[data-testid="stAlert"] {{
        border-radius: var(--radius-md);
        border: 1px solid {p['border']};
    }}

    /* ── Spinner ─────────────────────────────────────────── */
    div[data-testid="stSpinner"] p {{ color: {p['text_secondary']}; }}

    /* ── Tooltip ─────────────────────────────────────────── */
    div[data-testid="stTooltipContent"] {{
        background: {p['bg_card']};
        color: {p['text_primary']};
        border: 1px solid {p['border']};
        border-radius: var(--radius-sm);
        font-size: 0.82rem;
        box-shadow: var(--shadow-md);
    }}

    /* ── Print styles ────────────────────────────────────── */
    @media print {{
        section[data-testid="stSidebar"],
        .stButton, button {{ display: none !important; }}
        .stApp {{ background: white !important; color: black !important; }}
        .main-header {{ background: #4338ca !important; -webkit-print-color-adjust: exact; }}
    }}
    </style>
    """


def get_theme_css() -> str:
    """Public accessor — always returns CSS for the current theme mode."""
    return _get_theme_css_cached(is_dark())


st.markdown(get_theme_css(), unsafe_allow_html=True)


# ============================================================
# PLOTLY THEME FACTORY
# ============================================================
def get_plotly_layout(title: str = "", height: int = 400,
                      show_legend: bool = True,
                      xaxis_title: str = "", yaxis_title: str = "") -> dict:
    """
    Returns a consistent Plotly layout dict tuned to the current theme.
    Usage: fig.update_layout(**get_plotly_layout("My Chart", height=420))
    """
    p = _get_palette()
    axis_common = dict(
        gridcolor=p["border"],
        zerolinecolor=p["border_strong"],
        zerolinewidth=1,
        linecolor=p["border"],
        linewidth=1,
        tickfont=dict(color=p["text_secondary"], size=11,
                      family="Inter, system-ui, sans-serif"),
        title_font=dict(color=p["text_primary"], size=12,
                        family="Inter, system-ui, sans-serif"),
        showgrid=True,
        showline=True,
        mirror=False,
    )
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=p["text_primary"],
                      family="Inter, system-ui, sans-serif"),
            x=0.01, xanchor="left",
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=p["bg_secondary"],
        font=dict(family="Inter, system-ui, sans-serif",
                  color=p["text_primary"], size=12),
        showlegend=show_legend,
        legend=dict(
            bgcolor=p["bg_card"],
            bordercolor=p["border"],
            borderwidth=1,
            font=dict(color=p["text_primary"], size=11,
                      family="Inter, system-ui, sans-serif"),
            orientation="v",
        ),
        xaxis=dict(**axis_common, title=xaxis_title),
        yaxis=dict(**axis_common, title=yaxis_title),
        annotationdefaults=dict(
            font=dict(color=p["text_primary"], size=11,
                      family="Inter, system-ui, sans-serif"),
            arrowcolor=p["text_secondary"],
            bgcolor=p["bg_card"],
            bordercolor=p["border"],
        ),
        margin=dict(l=55, r=25, t=55, b=55),
        hoverlabel=dict(
            bgcolor=p["bg_card"],
            bordercolor=p["border"],
            font_color=p["text_primary"],
            font_size=12,
            font_family="Inter, system-ui, sans-serif",
            namelength=-1,      # show full trace name in hover
        ),
        colorway=get_plotly_colors(),
        dragmode="pan",         # more intuitive default than "zoom"
        hovermode="x unified",  # unified tooltip for time-series / multi-trace
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=p["text_secondary"],
            activecolor=p["accent"],
        ),
    )


def get_plotly_colors() -> list:
    """Return a consistent 8-color palette for Plotly traces."""
    return [
        "#6366f1",  # Indigo   — primary accent
        "#10b981",  # Emerald  — success
        "#f59e0b",  # Amber    — warning
        "#ef4444",  # Red      — danger
        "#3b82f6",  # Blue     — info
        "#8b5cf6",  # Violet
        "#ec4899",  # Pink
        "#14b8a6",  # Teal
    ]


def get_plotly_layout_2axis(title: str = "", height: int = 420,
                             y1_title: str = "", y2_title: str = "") -> dict:
    """
    Layout dict for dual-axis charts (y + y2).
    Usage:
        layout = get_plotly_layout_2axis("Revenue vs Units", y1_title="$", y2_title="Units")
        fig.update_layout(**layout)
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", ...))
    """
    base = get_plotly_layout(title, height, show_legend=True)
    p    = _get_palette()
    base["yaxis"]["title"] = y1_title
    base["yaxis2"] = dict(
        title=y2_title,
        overlaying="y",
        side="right",
        gridcolor="rgba(0,0,0,0)",   # suppress right-axis grid lines
        zerolinecolor=p["border_strong"],
        tickfont=dict(color=p["text_secondary"], size=11),
        title_font=dict(color=p["text_primary"], size=12),
        showgrid=False,
    )
    return base
# ============================================================
# DISPLAY HELPER FUNCTIONS
# ============================================================

def display_header(icon: str, chapter: str, title: str, subtitle: str,
                   show_divider: bool = False):
    """
    Module-level hero header.
    ENHANCED: optional divider rule beneath the header.
    """
    st.markdown(f"""
    <div class="main-header">
        <h1>{icon} {title}</h1>
        <p><span class="chapter-badge">{chapter}</span>{subtitle}</p>
    </div>
    {"<hr style='border:none;border-top:1px solid var(--border);margin:0 0 1rem;'>" if show_divider else ""}
    """, unsafe_allow_html=True)


def display_citation(quote: str, source: str, page: str = ""):
    """
    Styled blockquote citation.
    ENHANCED: optional page reference appended to source.
    """
    page_html = f", p. {page}" if page else ""
    st.markdown(f"""
    <div class="citation-box">
        &#8220;{quote}&#8221;
        <span class="citation-source">— {source}{page_html}</span>
    </div>
    """, unsafe_allow_html=True)


def display_key_insight(title: str, content: str, icon: str = "💡"):
    """
    Green insight / takeaway box.
    ENHANCED: customisable icon; icon defaults to 💡.
    """
    st.markdown(f"""
    <div class="key-insight">
        <div class="key-insight-title">{icon} {title}</div>
        <div class="key-insight-text">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def display_textbook_content(title: str, content: str, icon: str = "📖"):
    """
    Purple-accented textbook excerpt box.
    ENHANCED: customisable icon.
    """
    st.markdown(f"""
    <div class="textbook-content">
        <h4>{icon} {title}</h4>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)


def display_formula_card(title: str, formula_latex: str,
                          description: str = "", numbered: bool = False,
                          number: int = 0):
    """
    Accent-bordered formula card with LaTeX.
    ENHANCED:
    - Optional description line beneath the LaTeX.
    - Optional equation number (e.g. "(1)") shown top-right.
    FIX: removed unused _get_palette() call — palette not needed in this function.
    """
    num_html = (
        f'<div style="position:absolute;top:0.5rem;right:0.75rem;'
        f'font-size:0.75rem;color:var(--text-muted);font-weight:600;">({number})</div>'
        if numbered else ""
    )
    st.markdown(f"""
    <div class="formula-card" style="position:relative;">
        {num_html}
        <div class="formula-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.latex(formula_latex)
    if description:
        p = _get_palette()
        st.markdown(
            f"<p style='font-size:0.82rem;color:{p['text_muted']};"
            f"text-align:center;margin-top:-0.4rem;line-height:1.5;'>"
            f"{description}</p>",
            unsafe_allow_html=True,
        )


def display_equation(label: str, latex_eq: str, description: str = "",
                      number: str = ""):
    """
    Full-width equation box.
    ENHANCED: optional equation number displayed top-right of box.
    FIX: explicit palette color for description text (avoids browser default black
    on dark backgrounds).
    """
    p = _get_palette()
    num_html = (
        f'<span style="position:absolute;top:0.5rem;right:0.75rem;'
        f'font-size:0.75rem;color:{p["text_muted"]};font-weight:600;">({number})</span>'
        if number else ""
    )
    st.markdown(f"""
    <div class="equation-box" style="position:relative;">
        {num_html}
        <div class="equation-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    st.latex(latex_eq)
    if description:
        st.markdown(
            f"<p style='font-size:0.85rem;color:{p['text_secondary']};"
            f"margin-top:0.4rem;line-height:1.5;text-align:center;'>"
            f"{description}</p>",
            unsafe_allow_html=True,
        )


def display_metric_card(value, label: str, card_type: str = "normal",
                         delta=None, delta_label: str = "",
                         icon: str = "", tooltip: str = ""):
    """
    Metric card.
    card_type: 'normal' | 'highlight' | 'success' | 'danger' | 'warning'
    ENHANCED:
    - Optional icon shown above value.
    - Optional tooltip title attribute on the card div.
    - delta sign detection handles string '+'/'-' and numeric values.
    FIX: card_type=True backward-compat guard kept.
    FIX: delta=0 correctly renders as positive (no change = not negative).
    """
    if card_type is True:
        card_type = "highlight"
    if card_type not in ("normal", "highlight", "success", "danger", "warning"):
        card_type = "normal"

    icon_html = (
        f'<div style="font-size:1.5rem;margin-bottom:0.3rem;">{icon}</div>'
        if icon else ""
    )
    delta_html = ""
    if delta is not None:
        if isinstance(delta, (int, float)):
            sign  = "positive" if delta >= 0 else "negative"
            arrow = "▲" if delta >= 0 else "▼"
            d_str = f"{delta:+,.4g}"
        else:
            d_str = str(delta)
            sign  = "positive" if d_str.startswith("+") else "negative"
            arrow = "▲" if sign == "positive" else "▼"
        delta_html = (
            f'<div class="metric-delta {sign}">'
            f'{arrow} {d_str} {delta_label}</div>'
        )
    title_attr = f'title="{tooltip}"' if tooltip else ""
    st.markdown(f"""
    <div class="metric-card {card_type}" {title_attr}>
        {icon_html}
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def display_metric_row(metrics: list):
    """
    Render a row of metric cards in equal-width columns.
    metrics: list of dicts with keys matching display_metric_card parameters:
             value, label, card_type (opt), delta (opt), delta_label (opt),
             icon (opt), tooltip (opt)
    ENHANCED: NEW helper — avoids repetitive st.columns + display_metric_card calls.

    Example:
        display_metric_row([
            {"value": "707", "label": "EOQ (units)", "card_type": "highlight"},
            {"value": "$1,414", "label": "Min Total Cost", "card_type": "success"},
        ])
    """
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            display_metric_card(
                value       = m.get("value", "—"),
                label       = m.get("label", ""),
                card_type   = m.get("card_type", "normal"),
                delta       = m.get("delta"),
                delta_label = m.get("delta_label", ""),
                icon        = m.get("icon", ""),
                tooltip     = m.get("tooltip", ""),
            )


def display_concept_card(icon: str, title: str, description: str,
                          badge: str = "", badge_type: str = "accent"):
    """
    Icon + title + description card.
    ENHANCED: optional badge displayed in the top-right corner of the card.
    """
    badge_html = (
        f'<span class="badge badge-{badge_type}" '
        f'style="float:right;margin-top:-0.1rem;">{badge}</span>'
        if badge else ""
    )
    st.markdown(f"""
    <div class="concept-card">
        {badge_html}
        <div class="concept-icon">{icon}</div>
        <div class="concept-title">{title}</div>
        <div class="concept-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def display_solution_step(step_num, content: str, step_label: str = "Step"):
    """
    Numbered solution step with accent bar.
    ENHANCED: customisable step label (e.g. "Phase", "Stage") and supports
    non-integer step_num (e.g. "1a", "2b").
    """
    st.markdown(f"""
    <div class="solution-step">
        <span class="step-number" title="{step_label} {step_num}">{step_num}</span>
        <div style="flex:1;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def display_solution_steps(steps: list, step_label: str = "Step"):
    """
    Render multiple solution steps from a list.
    ENHANCED: NEW helper — avoids a for-loop in every module.
    steps: list of str (content) — numbered automatically, OR
           list of (num, content) tuples for custom numbering.
    """
    for i, step in enumerate(steps, start=1):
        if isinstance(step, (list, tuple)) and len(step) == 2:
            num, content = step
        else:
            num, content = i, step
        display_solution_step(num, content, step_label)


def display_practice_problem(problem_num, difficulty: str, problem_text: str,
                              topic: str = ""):
    """
    Styled practice problem header block.
    ENHANCED: optional topic tag displayed as a muted sub-label.
    FIX: defaults ⚪/accent for unrecognised difficulty levels.
    """
    icons     = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
    badge_cls = {"Easy": "success", "Medium": "warning", "Hard": "danger"}
    icon = icons.get(difficulty, "⚪")
    bcls = badge_cls.get(difficulty, "accent")
    topic_html = (
        f'<div style="font-size:0.78rem;color:var(--text-muted);"'
        f'margin-bottom:0.4rem;>{topic}</div>'
        if topic else ""
    )
    st.markdown(f"""
    <div class="practice-problem">
        <h4>Problem {problem_num}&nbsp;
            <span class="badge badge-{bcls}">{icon} {difficulty}</span>
        </h4>
        {topic_html}
        <p>{problem_text}</p>
    </div>
    """, unsafe_allow_html=True)


def display_hint(hint_text: str, collapsible: bool = False,
                 label: str = "Hint"):
    """
    Hint box.
    ENHANCED: optional collapsible mode using st.expander.
    """
    if collapsible:
        with st.expander(f"💡 Show {label}"):
            st.markdown(
                f'<div class="hint-box">💡 <strong>{label}:</strong>'
                f' {hint_text}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(f"""
        <div class="hint-box">
            💡 <strong>{label}:</strong> {hint_text}
        </div>
        """, unsafe_allow_html=True)


def display_solution(solution_html: str, collapsible: bool = False,
                     label: str = "Solution"):
    """
    Green solution reveal box.
    ENHANCED: optional collapsible mode — hides full solution until expanded.
    """
    if collapsible:
        with st.expander(f"✅ Show {label}"):
            st.markdown(f"""
            <div class="solution-box">
                ✅ <strong>{label}:</strong><br><br>{solution_html}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="solution-box">
            ✅ <strong>{label}:</strong><br><br>{solution_html}
        </div>
        """, unsafe_allow_html=True)


def display_callout(content: str, callout_type: str = "info",
                    title: str = "", icon: str = ""):
    """
    Flexible callout box.
    callout_type: 'info' | 'success' | 'warning' | 'danger' | 'tip'
    ENHANCED: falls back to 'info' style for unrecognised types instead of
    rendering an unstyled div.
    FIX: icon/title overrides now evaluated before fallback to avoid empty
    strings shadowing the defaults.
    """
    default_icons  = {
        "info":    "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "danger":  "🚨",
        "tip":     "📌",
    }
    default_titles = {
        "info":    "Note",
        "success": "Key Takeaway",
        "warning": "Caution",
        "danger":  "Critical",
        "tip":     "Tip",
    }
    if callout_type not in default_icons:
        callout_type = "info"

    resolved_icon  = icon  if icon  else default_icons[callout_type]
    resolved_title = title if title else default_titles[callout_type]

    st.markdown(f"""
    <div class="callout {callout_type}">
        <div class="callout-icon">{resolved_icon}</div>
        <div class="callout-content">
            <div class="callout-title">{resolved_title}</div>
            <div>{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_alert(content: str, alert_type: str = "info"):
    """Backward-compat alias → delegates to display_callout."""
    display_callout(content, callout_type=alert_type)


def display_theory(content: str):
    """Plain theory box (legacy helper)."""
    st.markdown(
        f'<div class="textbook-content"><div>{content}</div></div>',
        unsafe_allow_html=True,
    )


def display_process_flow(steps: list, orientation: str = "horizontal"):
    """
    Render a process-flow strip.
    steps: list of (number_or_icon, label) tuples.
    ENHANCED: 'vertical' orientation option for narrow layouts.
    """
    if orientation == "vertical":
        for n, lbl in steps:
            st.markdown(f"""
            <div class="solution-step" style="margin:0.3rem 0;">
                <span class="step-number">{n}</span>
                <div style="flex:1;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        steps_html = "".join(
            f'<div class="process-step">'
            f'<div class="process-step-num">{n}</div>'
            f'<div class="process-step-text">{lbl}</div>'
            f'</div>'
            for n, lbl in steps
        )
        st.markdown(
            f'<div class="process-flow">{steps_html}</div>',
            unsafe_allow_html=True,
        )


def display_badge(text: str, badge_type: str = "accent") -> str:
    """
    Inline badge. Returns HTML string for embedding.
    ENHANCED: validates badge_type; falls back to 'accent'.
    """
    valid = {"accent", "success", "warning", "danger", "info", "new"}
    if badge_type not in valid:
        badge_type = "accent"
    return f'<span class="badge badge-{badge_type}">{text}</span>'


def display_chapter_summary(points: list, title: str = "Chapter Key Points",
                             icon: str = "📋"):
    """
    Bulleted summary box at end of theory sections.
    ENHANCED: customisable icon; empty list guard.
    """
    if not points:
        return
    items_html = "".join(f"<li>{pt}</li>" for pt in points)
    st.markdown(f"""
    <div class="chapter-summary">
        <div class="chapter-summary-title">{icon} {title}</div>
        <ul>{items_html}</ul>
    </div>
    """, unsafe_allow_html=True)


def display_progress_bar(value: float, max_value: float = 100,
                          label: str = "", bar_type: str = "accent",
                          show_pct: bool = True):
    """
    Custom HTML progress bar.
    bar_type: 'accent' | 'success' | 'danger' | 'warning'
    ENHANCED: show_pct flag — set False to show only the label without %.
    FIX: palette used for label color (was missing — browser default on dark bg).
    FIX: guards against max_value=0 division by zero.
    """
    if max_value <= 0:
        return
    valid_types = {"accent", "success", "danger", "warning"}
    if bar_type not in valid_types:
        bar_type = "accent"

    p   = _get_palette()
    pct = min(100.0, max(0.0, value / max_value * 100))

    if label and show_pct:
        label_html = (
            f"<div style='font-size:0.8rem;color:{p['text_secondary']};"
            f"margin-bottom:0.2rem;display:flex;justify-content:space-between;'>"
            f"<span>{label}</span><span>{pct:.0f}%</span></div>"
        )
    elif label:
        label_html = (
            f"<div style='font-size:0.8rem;color:{p['text_secondary']};"
            f"margin-bottom:0.2rem;'>{label}</div>"
        )
    elif show_pct:
        label_html = (
            f"<div style='font-size:0.8rem;color:{p['text_secondary']};"
            f"margin-bottom:0.2rem;text-align:right;'>{pct:.0f}%</div>"
        )
    else:
        label_html = ""

    st.markdown(f"""
    {label_html}
    <div class="progress-wrap">
        <div class="progress-fill {bar_type}" style="width:{pct:.1f}%;"></div>
    </div>
    """, unsafe_allow_html=True)


def display_comparison_table(html_rows: list,
                              headers: tuple = ("Feature", "Option A", "Option B"),
                              highlight_col: int = None):
    """
    Render a styled HTML comparison table.
    html_rows: list of row tuples — any number of columns.
    ENHANCED:
    - highlight_col: 1-based column index to bold (e.g. 2 = Option A wins).
    - Dynamic column count: headers drives number of <th> elements.
    FIX: inner f-string f'<td>{v}</td>' caused SyntaxWarning in Python 3.12+;
         replaced with join over a list comprehension.
    """
    def _cell(v, col_idx):
        style = " style='font-weight:700;'" if highlight_col == col_idx else ""
        return f"<td{style}>{v}</td>"

    th_html   = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = "".join(
        "<tr>" + "".join(
            _cell(cell, j + 1) for j, cell in enumerate(row)
        ) + "</tr>"
        for row in html_rows
    )
    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>{th_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


def display_two_column_content(left_content_fn, right_content_fn,
                                left_width: int = 1, right_width: int = 1,
                                gap: str = "medium"):
    """
    Render two callables side-by-side using st.columns.
    ENHANCED: NEW helper — avoids boilerplate column setup in every module.
    left_content_fn / right_content_fn: zero-argument callables (lambdas or funcs).

    Example:
        display_two_column_content(
            lambda: display_formula_card("EOQ", r"Q^* = \\sqrt{2DS/H}"),
            lambda: display_key_insight("Rule", "Holding = Ordering at EOQ"),
        )
    """
    col_l, col_r = st.columns([left_width, right_width], gap=gap)
    with col_l:
        left_content_fn()
    with col_r:
        right_content_fn()


def display_info_grid(items: list, cols: int = 3):
    """
    Render a responsive grid of concept cards.
    ENHANCED: NEW helper — wraps display_concept_card in a column grid.
    items: list of (icon, title, description) or
           (icon, title, description, badge, badge_type) tuples.

    Example:
        display_info_grid([
            ("🔬", "Simulator", "Interactive tools"),
            ("📊", "Charts",    "Plotly visuals"),
        ], cols=2)
    """
    if not items:
        return
    columns = st.columns(cols)
    for i, item in enumerate(items):
        with columns[i % cols]:
            if len(item) == 3:
                display_concept_card(*item)
            elif len(item) == 5:
                display_concept_card(*item)
            else:
                display_concept_card(item[0], item[1],
                                     item[2] if len(item) > 2 else "")


def display_spacer(rem: float = 1.0):
    """
    Insert vertical whitespace.
    ENHANCED: NEW micro-helper — cleaner than repeated st.markdown('<br>').
    """
    st.markdown(
        f"<div style='height:{rem}rem;'></div>",
        unsafe_allow_html=True,
    )


def display_divider(label: str = "", color: str = ""):
    """
    Themed horizontal rule with optional centered label.
    ENHANCED: NEW helper — replaces st.divider() which ignores theme tokens.
    """
    p = _get_palette()
    c = color or p["border"]
    if label:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.75rem;margin:1rem 0;">
            <div style="flex:1;height:1px;background:{c};"></div>
            <span style="font-size:0.78rem;color:{p['text_muted']};
                         font-weight:600;text-transform:uppercase;
                         letter-spacing:0.06em;">{label}</span>
            <div style="flex:1;height:1px;background:{c};"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<hr style='border:none;border-top:1px solid {c};margin:1rem 0;'>",
            unsafe_allow_html=True,
        )

# ============================================================
# MATH / STATISTICS HELPERS
# ============================================================

# ── Answer checking ───────────────────────────────────────

def check_answer(user_answer, correct_answer, tolerance: float = 0.05) -> bool:
    """
    Returns True if |user − correct| ≤ threshold.
    threshold = max(|correct| × tolerance, 0.01) for non-zero correct values.
    threshold = max(tolerance, 0.01) when correct = 0 (avoids 0 × tol = 0).
    FIX: gracefully handles None, empty string, and non-numeric inputs.
    """
    try:
        u, c  = float(user_answer), float(correct_answer)
        threshold = max(abs(c) * tolerance, 0.01) if c != 0 else max(tolerance, 0.01)
        return abs(u - c) <= threshold
    except (TypeError, ValueError):
        return False


def check_answer_with_feedback(user_answer, correct_answer,
                                tolerance: float = 0.05,
                                unit: str = "",
                                fmt: str = ",.4g") -> tuple:
    """
    Like check_answer but returns (is_correct: bool, message: str).
    ENHANCED: fmt param controls number display format (default ',.4g').
    FIX: pct_off uses max(|c|, 1e-9) to guard against zero denominator.
    FIX: delta shown as absolute and percentage for clearer feedback.
    """
    try:
        u, c      = float(user_answer), float(correct_answer)
        threshold = max(abs(c) * tolerance, 0.01) if c != 0 else max(tolerance, 0.01)
        is_correct = abs(u - c) <= threshold
        unit_str   = f" {unit}" if unit else ""
        if is_correct:
            return True, f"✅ Correct! Answer = {c:{fmt}}{unit_str}"
        pct_off = abs(u - c) / max(abs(c), 1e-9) * 100
        abs_off = abs(u - c)
        return False, (
            f"❌ Not quite. Your answer: {u:{fmt}}{unit_str} | "
            f"Correct: {c:{fmt}}{unit_str} "
            f"(off by {abs_off:{fmt}}{unit_str} = {pct_off:.1f}%)"
        )
    except (TypeError, ValueError):
        return False, "❌ Please enter a valid number."


def check_multiple_answers(user_answers: dict, correct_answers: dict,
                            tolerance: float = 0.05) -> dict:
    """
    Check several named answers at once.
    ENHANCED: NEW helper for multi-part problems.
    Returns dict of {key: (is_correct, feedback_str)}.

    Example:
        results = check_multiple_answers(
            {"Cp": 1.17, "Cpk": 1.00},
            {"Cp": 1.17, "Cpk": 1.00},
        )
    """
    return {
        k: check_answer_with_feedback(user_answers.get(k), v, tolerance)
        for k, v in correct_answers.items()
    }


# ── Probability distributions ─────────────────────────────

def normal_cdf(z: float) -> float:
    """Standard normal CDF: P(Z ≤ z)."""
    return float(stats.norm.cdf(z))


def normal_ppf(p: float) -> float:
    """
    Inverse standard normal CDF: z such that P(Z ≤ z) = p.
    FIX: clamps p to (1e-9, 1-1e-9) to prevent ±inf from edge inputs.
    """
    p = max(1e-9, min(1 - 1e-9, p))
    return float(stats.norm.ppf(p))


def normal_pdf(z: float) -> float:
    """
    Standard normal PDF: φ(z).
    ENHANCED: NEW — used in newsvendor expected-shortage calculations.
    """
    return float(stats.norm.pdf(z))


def normal_between(a: float, b: float) -> float:
    """P(a ≤ Z ≤ b) for the standard normal."""
    return float(stats.norm.cdf(b) - stats.norm.cdf(a))


def normal_tail(z: float, two_tail: bool = False) -> float:
    """
    P(Z > z) for one-tail, or P(|Z| > |z|) for two-tail.
    ENHANCED: NEW — convenient for control chart false-alarm probability.
    """
    one = 1.0 - normal_cdf(abs(z))
    return 2 * one if two_tail else one


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for Poisson(λ). Guards against λ ≤ 0."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(stats.poisson.pmf(k, lam))


def poisson_cdf(k: int, lam: float) -> float:
    """P(X ≤ k) for Poisson(λ). Guards against λ ≤ 0."""
    if lam <= 0:
        return 1.0
    return float(stats.poisson.cdf(k, lam))


def poisson_table(lam: float, k_max: int = 15) -> pd.DataFrame:
    """
    Poisson probability table for display.
    ENHANCED: NEW — returns DataFrame with P(X=k) and P(X≤k) columns.
    """
    k_vals = list(range(k_max + 1))
    return pd.DataFrame({
        "k":         k_vals,
        "P(X = k)":  [round(poisson_pmf(k, lam), 6) for k in k_vals],
        "P(X ≤ k)":  [round(poisson_cdf(k, lam), 6) for k in k_vals],
    })


def binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for Binomial(n, p)."""
    return float(stats.binom.pmf(k, n, p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for Binomial(n, p)."""
    return float(stats.binom.cdf(k, n, p))


def binom_table(n: int, p: float) -> pd.DataFrame:
    """
    Binomial probability table for display.
    ENHANCED: NEW — returns DataFrame with P(X=k) and P(X≤k) columns.
    """
    k_vals = list(range(n + 1))
    return pd.DataFrame({
        "k":         k_vals,
        "P(X = k)":  [round(binom_pmf(k, n, p), 6) for k in k_vals],
        "P(X ≤ k)":  [round(binom_cdf(k, n, p), 6) for k in k_vals],
    })


# ── Confidence intervals ──────────────────────────────────

def confidence_interval_mean(x_bar: float, sigma: float,
                              n: int, conf: float = 0.95) -> tuple:
    """
    (lower, upper) CI for population mean with known σ (Z-based).
    FIX: guards against n ≤ 0.
    """
    if n <= 0 or sigma < 0:
        return (None, None)
    z      = normal_ppf(1 - (1 - conf) / 2)
    margin = z * sigma / math.sqrt(n)
    return x_bar - margin, x_bar + margin


def confidence_interval_mean_t(x_bar: float, s: float,
                                n: int, conf: float = 0.95) -> tuple:
    """
    (lower, upper) CI for population mean with unknown σ (t-based).
    FIX: guards against n ≤ 1 (df = 0 would raise in scipy).
    """
    if n <= 1 or s < 0:
        return (None, None)
    t_crit = float(stats.t.ppf(1 - (1 - conf) / 2, df=n - 1))
    margin = t_crit * s / math.sqrt(n)
    return x_bar - margin, x_bar + margin


def confidence_interval_proportion(p_hat: float, n: int,
                                    conf: float = 0.95) -> tuple:
    """
    Wilson score CI for a proportion.
    ENHANCED: NEW — more accurate than normal approximation for small n or
    extreme p̂. Used in p-chart analysis.
    Returns (lower, upper).
    """
    if n <= 0:
        return (None, None)
    z    = normal_ppf(1 - (1 - conf) / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


# ── Process capability (SQC) ──────────────────────────────

def process_capability(mean: float, std: float,
                        lsl: float, usl: float) -> dict:
    """
    Returns dict with Cp, Cpk, Cpl, Cpu, and estimated ppm defect rate.
    ENHANCED: added 'ppm_total' and 'yield_pct' to the return dict.
    FIX: returns all-None dict (not crash) when std ≤ 0.
    """
    if std <= 0:
        return {"Cp": None, "Cpk": None, "Cpu": None,
                "Cpl": None, "ppm_total": None, "yield_pct": None}
    cp   = (usl - lsl) / (6 * std)
    cpu  = (usl - mean) / (3 * std)
    cpl  = (mean - lsl) / (3 * std)
    cpk  = min(cpu, cpl)
    # Estimated defect rate (assumes normality, no 1.5σ shift)
    p_defect  = normal_cdf((lsl - mean) / std) + (1 - normal_cdf((usl - mean) / std))
    ppm_total = round(p_defect * 1_000_000, 2)
    yield_pct = round((1 - p_defect) * 100, 5)
    return {
        "Cp":        round(cp,  4),
        "Cpk":       round(cpk, 4),
        "Cpu":       round(cpu, 4),
        "Cpl":       round(cpl, 4),
        "ppm_total": ppm_total,
        "yield_pct": yield_pct,
    }


def sigma_level(cpk: float) -> float:
    """
    Convert Cpk to approximate sigma level (σ = 3 × Cpk).
    FIX: handles None and negative cpk gracefully.
    """
    if cpk is None:
        return 0.0
    return max(0.0, 3.0 * cpk)


def dpmo_to_sigma(dpmo: float) -> float:
    """
    Approximate sigma level from DPMO (includes 1.5σ shift convention).
    ENHANCED: NEW — inverse of the standard DPMO table used in Six Sigma.
    Returns 0.0 for invalid DPMO values.
    """
    if dpmo <= 0 or dpmo >= 1_000_000:
        return 0.0
    try:
        return 0.8406 + math.sqrt(max(0.0, 29.37 - 2.221 * math.log(dpmo)))
    except (ValueError, ZeroDivisionError):
        return 0.0


def sigma_to_dpmo(sigma: float) -> float:
    """
    Approximate DPMO from sigma level (includes 1.5σ shift convention).
    ENHANCED: NEW — inverse lookup for display tables and calculators.
    """
    if sigma <= 0:
        return 1_000_000.0
    z = sigma - 1.5          # account for standard 1.5σ shift
    return max(0.0, (1 - normal_cdf(z)) * 1_000_000)


def control_chart_limits(center: float, std_dev: float,
                          sigma_multiplier: float = 3.0) -> tuple:
    """
    Generic UCL / LCL for any control chart.
    ENHANCED: NEW — consolidates the UCL/LCL formula used across all chart types.
    Returns (UCL, CL, LCL). LCL is clamped to 0 for attribute charts if needed
    by the caller.
    """
    ucl = center + sigma_multiplier * std_dev
    lcl = center - sigma_multiplier * std_dev
    return ucl, center, lcl


# ── PERT / Project helpers ────────────────────────────────

def pert_te(a: float, m: float, b: float) -> float:
    """PERT expected time: (a + 4m + b) / 6."""
    return (a + 4 * m + b) / 6


def pert_variance(a: float, b: float) -> float:
    """PERT activity variance: ((b − a) / 6)²."""
    return ((b - a) / 6) ** 2


def pert_sigma(a: float, b: float) -> float:
    """PERT activity standard deviation: (b − a) / 6."""
    return (b - a) / 6


def pert_path_stats(activities: list) -> dict:
    """
    Compute expected duration and std dev for a complete PERT path.
    ENHANCED: NEW — sums te and variances across a list of (a, m, b) tuples.
    Returns {"te": float, "variance": float, "sigma": float}.

    Example:
        pert_path_stats([(2,4,6), (1,3,5), (3,5,9)])
    """
    total_te  = sum(pert_te(a, m, b) for a, m, b in activities)
    total_var = sum(pert_variance(a, b) for a, m, b in activities)
    return {
        "te":       round(total_te,  4),
        "variance": round(total_var, 4),
        "sigma":    round(math.sqrt(total_var), 4),
    }


def pert_prob_complete(path_te: float, path_sigma: float,
                        target: float) -> float:
    """
    Probability that a PERT path finishes by 'target' time.
    ENHANCED: NEW — P(T ≤ target) = Φ((target − te) / σ_path).
    """
    if path_sigma <= 0:
        return 1.0 if target >= path_te else 0.0
    z = (target - path_te) / path_sigma
    return normal_cdf(z)


def crash_cost_per_day(normal_time: float, crash_time: float,
                        normal_cost: float, crash_cost: float):
    """
    Cost per time unit to crash an activity.
    Returns None if crashing is not possible (crash_time ≥ normal_time).
    FIX: also returns None if crash_cost < normal_cost (invalid input).
    """
    days = normal_time - crash_time
    if days <= 0 or crash_cost < normal_cost:
        return None
    return (crash_cost - normal_cost) / days


# ── Forecasting helpers ───────────────────────────────────

def moving_average(data: list, n: int) -> list:
    """
    Simple n-period moving average.
    Returns list of length len(data); first n-1 values are None.
    FIX: guards against n ≤ 0 and n > len(data).
    """
    if n <= 0 or n > len(data):
        return [None] * len(data)
    result = [None] * (n - 1)
    for i in range(n - 1, len(data)):
        result.append(sum(data[i - n + 1: i + 1]) / n)
    return result


def weighted_moving_average(data: list, weights: list) -> list:
    """
    Weighted moving average.
    ENHANCED: NEW — weights are normalised internally so they need not sum to 1.
    Returns list same length as data; first len(weights)-1 values are None.

    Example:
        wma = weighted_moving_average([10,12,14,13,15], [0.2, 0.3, 0.5])
    """
    n = len(weights)
    if n == 0 or n > len(data):
        return [None] * len(data)
    w_sum = sum(weights)
    w_norm = [w / w_sum for w in weights]
    result = [None] * (n - 1)
    for i in range(n - 1, len(data)):
        val = sum(w_norm[j] * data[i - n + 1 + j] for j in range(n))
        result.append(round(val, 6))
    return result


def exponential_smoothing(data: list, alpha: float,
                           initial: float = None) -> list:
    """
    Simple exponential smoothing (one-step-ahead forecasts).
    Returns list same length as data; forecast[0] = initial or data[0].
    FIX: clamps alpha to (0, 1) to prevent degenerate behaviour.
    ENHANCED: uses data[0] as initial when not provided, consistent with
    common textbook convention.
    """
    if not data:
        return []
    alpha = max(1e-6, min(1 - 1e-6, alpha))
    f0    = initial if initial is not None else data[0]
    forecasts = [f0]
    for i in range(1, len(data)):
        forecasts.append(alpha * data[i - 1] + (1 - alpha) * forecasts[-1])
    return forecasts


def double_exponential_smoothing(data: list, alpha: float,
                                  beta: float) -> tuple:
    """
    Holt's double exponential smoothing (trend-adjusted).
    ENHANCED: NEW — handles trended data where simple ES under/overshoots.
    Returns (level_series, trend_series, forecast_series), all same length as data.
    alpha: level smoothing; beta: trend smoothing.
    """
    if len(data) < 2:
        return [None] * len(data), [None] * len(data), [None] * len(data)
    alpha = max(1e-6, min(1 - 1e-6, alpha))
    beta  = max(1e-6, min(1 - 1e-6, beta))
    L  = [data[0]]
    T  = [data[1] - data[0]]
    F  = [data[0]]
    for i in range(1, len(data)):
        l_new = alpha * data[i] + (1 - alpha) * (L[-1] + T[-1])
        t_new = beta  * (l_new - L[-1]) + (1 - beta) * T[-1]
        f_new = L[-1] + T[-1]
        L.append(round(l_new, 6))
        T.append(round(t_new, 6))
        F.append(round(f_new, 6))
    return L, T, F


def forecast_error_metrics(actual: list, forecast: list) -> dict:
    """
    Returns MAD, MSE, RMSE, MAPE, and Bias for paired actual/forecast lists.
    ENHANCED: added RMSE (√MSE) — frequently required in OSCM textbook problems.
    FIX: skips pairs where actual = 0 (MAPE would be undefined).
    FIX: skips pairs where forecast is None.
    """
    pairs = [(a, f) for a, f in zip(actual, forecast)
             if f is not None and a != 0]
    if not pairs:
        return {"MAD": None, "MSE": None, "RMSE": None,
                "MAPE": None, "Bias": None, "n": 0}
    n      = len(pairs)
    errors = [a - f for a, f in pairs]
    mad    = sum(abs(e) for e in errors) / n
    mse    = sum(e ** 2 for e in errors) / n
    rmse   = math.sqrt(mse)
    mape   = sum(abs((a - f) / a) for a, f in pairs) / n * 100
    bias   = sum(errors) / n
    return {
        "MAD":  round(mad,  4),
        "MSE":  round(mse,  4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "Bias": round(bias, 4),
        "n":    n,
    }


def linear_trend_forecast(data: list) -> dict:
    """
    Fit a simple linear trend (OLS) and return slope, intercept, and
    one-step-ahead forecast.
    ENHANCED: NEW — uses scipy.stats.linregress; avoids numpy dependency
    for a single regression call.
    Returns {"slope", "intercept", "r_squared", "next_forecast"}.
    """
    if len(data) < 2:
        return {"slope": None, "intercept": None,
                "r_squared": None, "next_forecast": None}
    x = list(range(1, len(data) + 1))
    result = stats.linregress(x, data)
    return {
        "slope":          round(result.slope,     4),
        "intercept":      round(result.intercept, 4),
        "r_squared":      round(result.rvalue**2, 4),
        "next_forecast":  round(result.intercept + result.slope * (len(data) + 1), 4),
    }


# ── Inventory helpers ─────────────────────────────────────

def eoq(demand: float, order_cost: float, holding_cost: float):
    """
    Economic Order Quantity: Q* = √(2DS / H).
    FIX: returns None (not crash) for non-positive inputs.
    """
    if demand <= 0 or order_cost <= 0 or holding_cost <= 0:
        return None
    return math.sqrt(2 * demand * order_cost / holding_cost)


def eoq_full(demand: float, order_cost: float,
             holding_cost: float, unit_cost: float = 0.0) -> dict:
    """
    EOQ with full cost breakdown.
    ENHANCED: NEW — returns Q*, annual order cost, holding cost, purchase
    cost, and total cost in a single dict.
    unit_cost: purchase price per unit (0 = ignore purchase cost).
    """
    q = eoq(demand, order_cost, holding_cost)
    if q is None:
        return {}
    orders_per_yr  = demand / q
    annual_order   = orders_per_yr * order_cost
    annual_holding = (q / 2) * holding_cost
    annual_purchase = demand * unit_cost
    return {
        "Q_star":          round(q, 2),
        "orders_per_year": round(orders_per_yr, 2),
        "annual_order":    round(annual_order, 2),
        "annual_holding":  round(annual_holding, 2),
        "annual_purchase": round(annual_purchase, 2),
        "total_cost":      round(annual_order + annual_holding + annual_purchase, 2),
        "cycle_time_days": round(q / demand * 365, 1),
    }


def reorder_point(avg_daily_demand: float, lead_time_days: float,
                  safety_stock: float = 0.0) -> float:
    """Reorder point: ROP = d̄ × LT + SS."""
    return avg_daily_demand * lead_time_days + safety_stock


def safety_stock_units(z: float, sigma_demand: float,
                        lead_time_days: float) -> float:
    """
    Safety stock: SS = z × σ_d × √LT.
    FIX: guards against negative lead_time_days.
    """
    if lead_time_days < 0:
        return 0.0
    return z * sigma_demand * math.sqrt(lead_time_days)


def safety_stock_variable_lt(z: float, avg_demand: float,
                               sigma_demand: float,
                               avg_lt: float, sigma_lt: float) -> float:
    """
    Safety stock when both demand and lead time are variable.
    ENHANCED: NEW — SS = z × √(avg_LT × σ_d² + avg_d² × σ_LT²).
    Used in Chapters 20–21 variable lead-time problems.
    """
    if avg_lt < 0 or avg_demand < 0:
        return 0.0
    return z * math.sqrt(avg_lt * sigma_demand**2 + avg_demand**2 * sigma_lt**2)


def total_inventory_cost(demand: float, order_qty: float,
                          order_cost: float, holding_cost: float) -> float:
    """
    Annual total inventory cost: TC = (D/Q)S + (Q/2)H.
    FIX: returns inf for order_qty ≤ 0 rather than raising ZeroDivisionError.
    """
    if order_qty <= 0:
        return float("inf")
    return (demand / order_qty) * order_cost + (order_qty / 2) * holding_cost


def newsvendor(price: float, cost: float, salvage: float,
               mean_demand: float, std_demand: float) -> dict:
    """
    Newsvendor (critical ratio) model.
    ENHANCED: NEW — returns Q*, critical ratio, expected profit,
    expected sales, and expected leftover inventory.
    Cu = price − cost (underage), Co = cost − salvage (overage).
    """
    cu = price - cost
    co = cost - salvage
    if cu + co <= 0:
        return {}
    cr  = cu / (cu + co)
    z   = normal_ppf(cr)
    q   = mean_demand + z * std_demand

    # Expected sales = E[min(D,Q)] = μΦ(z) − σφ(z) + Q(1−Φ(z))  [textbook form]
    phi_z  = normal_cdf(z)
    pdf_z  = normal_pdf(z)
    e_sales   = mean_demand * phi_z - std_demand * pdf_z + q * (1 - phi_z)
    e_leftover = max(0.0, q - e_sales)
    e_profit   = (price - cost) * e_sales - (cost - salvage) * e_leftover - cost * 0

    return {
        "Cu":             round(cu, 4),
        "Co":             round(co, 4),
        "critical_ratio": round(cr, 4),
        "z":              round(z,  4),
        "Q_star":         round(q,  2),
        "expected_sales":    round(e_sales,    2),
        "expected_leftover": round(e_leftover, 2),
        "expected_profit":   round(e_profit,   2),
    }


# ── Decision / financial helpers ──────────────────────────

def emv(probability: float, impact: float) -> float:
    """Expected Monetary Value: EMV = P × Impact."""
    return probability * impact


def emv_table(outcomes: list) -> dict:
    """
    EMV for a decision node with multiple outcomes.
    ENHANCED: NEW — outcomes is list of (probability, impact) tuples.
    Returns {"emv": float, "outcomes": list of floats}.

    Example:
        emv_table([(0.4, 200_000), (0.4, 25_000), (0.2, -40_000)])
    """
    vals = [emv(p, i) for p, i in outcomes]
    return {"emv": round(sum(vals), 2), "outcomes": [round(v, 2) for v in vals]}


def npv(cash_flows: list, rate: float) -> float:
    """
    Net Present Value: NPV = Σ CF_t / (1+r)^t.
    Index 0 = period 0 (usually negative initial investment).
    FIX: guards against rate = -1 (division by zero).
    """
    if rate <= -1:
        return float("nan")
    return sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))


def irr(cash_flows: list, guess: float = 0.1) -> float:
    """
    Internal Rate of Return via bisection search.
    ENHANCED: NEW — useful for capital budgeting modules.
    Returns None if no IRR found in (−99%, 1000%) range.
    """
    lo, hi = -0.9999, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if abs(hi - lo) < 1e-8:
            return round(mid, 6)
        if npv(cash_flows, mid) > 0:
            lo = mid
        else:
            hi = mid
    return None


def break_even_units(fixed_cost: float, price: float,
                      variable_cost: float):
    """
    Break-even point in units: BEP = FC / (P − VC).
    Returns None if contribution margin ≤ 0.
    """
    cm = price - variable_cost
    return fixed_cost / cm if cm > 0 else None


def break_even_revenue(fixed_cost: float, price: float,
                        variable_cost: float):
    """
    Break-even revenue: BEP_$ = BEP_units × P.
    ENHANCED: NEW — frequently asked separately from unit BEP.
    """
    units = break_even_units(fixed_cost, price, variable_cost)
    return units * price if units is not None else None


def target_profit_units(fixed_cost: float, target_profit: float,
                         price: float, variable_cost: float):
    """Units required to achieve a target profit: (FC + π) / CM."""
    cm = price - variable_cost
    return (fixed_cost + target_profit) / cm if cm > 0 else None


def indifference_point(fc1: float, vc1: float,
                        fc2: float, vc2: float):
    """
    Volume at which total cost of two alternatives is equal.
    Returns None if variable costs are parallel (no intersection).
    """
    dv = vc1 - vc2
    return (fc2 - fc1) / dv if abs(dv) > 1e-9 else None


def learning_curve_time(t1: float, n: int, rate: float) -> float:
    """
    Learning curve: cumulative average time for unit n.
    ENHANCED: NEW — Y_n = T1 × n^b where b = log(rate)/log(2).
    rate: e.g. 0.80 for 80% learning curve.
    FIX: guards against n ≤ 0 and rate outside (0, 1).
    """
    if n <= 0 or not (0 < rate < 1) or t1 <= 0:
        return float("nan")
    b = math.log(rate) / math.log(2)
    return t1 * (n ** b)


def learning_curve_table(t1: float, rate: float,
                           units: list = None) -> pd.DataFrame:
    """
    Full learning curve table for a sequence of units.
    ENHANCED: NEW — returns DataFrame with cumulative avg, total time,
    and marginal (individual unit) time.

    Example:
        learning_curve_table(100, 0.80, [1, 2, 4, 8, 16])
    """
    if units is None:
        units = [1, 2, 4, 8, 16, 32]
    rows = []
    prev_total = 0.0
    for n in units:
        cum_avg   = learning_curve_time(t1, n, rate)
        total     = cum_avg * n
        marginal  = total - prev_total
        rows.append({
            "Unit (N)":      n,
            "Cum Avg Time":  round(cum_avg, 3),
            "Total Time":    round(total, 3),
            "Marginal Time": round(marginal, 3),
        })
        prev_total = total
    return pd.DataFrame(rows)


# ── Formatters ────────────────────────────────────────────

def format_currency(value: float, decimals: int = 0,
                    symbol: str = "$") -> str:
    """
    Format a number as a currency string.
    ENHANCED: customisable symbol (e.g. "€", "£").
    FIX: removed nested f-string brace escaping issue in original.
    """
    return f"{symbol}{value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator."""
    return f"{value:,.{decimals}f}"


def format_pct(value: float, decimals: int = 1,
               already_pct: bool = False) -> str:
    """
    Format as percentage string.
    ENHANCED: already_pct=True skips ×100 (for values already in percent).
    """
    v = value if already_pct else value * 100
    return f"{v:.{decimals}f}%"


def format_delta(value: float, decimals: int = 2,
                 unit: str = "") -> str:
    """Format a signed delta value with ▲/▼ arrow prefix."""
    arrow  = "▲" if value >= 0 else "▼"
    unit_s = f" {unit}" if unit else ""
    return f"{arrow} {abs(value):,.{decimals}f}{unit_s}"


def format_sigma(cpk: float) -> str:
    """
    Format Cpk as a sigma level string.
    ENHANCED: NEW — convenience wrapper for metric cards.
    Example: format_sigma(1.33) → "4.0σ"
    """
    return f"{sigma_level(cpk):.2f}σ"


def format_dpmo(dpmo: float) -> str:
    """
    Format DPMO with thousands separator and sigma level.
    ENHANCED: NEW — e.g. "6,210 DPMO (≈4.0σ)".
    """
    sl = dpmo_to_sigma(dpmo)
    return f"{dpmo:,.1f} DPMO (≈{sl:.1f}σ)"


# ── Scheduling helpers ────────────────────────────────────

def critical_ratio(due_date: float, today: float,
                   remaining_time: float) -> float:
    """
    Critical Ratio: CR = (Due Date − Today) / Remaining Processing Time.
    ENHANCED: NEW — CR < 1 = behind, CR = 1 = on track, CR > 1 = ahead.
    Returns inf if remaining_time = 0 (job is finished).
    """
    if remaining_time <= 0:
        return float("inf")
    return (due_date - today) / remaining_time


def spt_sequence(jobs: dict) -> list:
    """
    Shortest Processing Time sequencing.
    ENHANCED: NEW — sorts jobs by processing time ascending.
    jobs: {job_name: {"pt": float, "dd": float}} dict.
    Returns sorted list of (job_name, pt, dd, completion_time, tardiness).
    """
    sorted_jobs = sorted(jobs.items(), key=lambda x: x[1]["pt"])
    result, t = [], 0
    for name, j in sorted_jobs:
        t += j["pt"]
        tardiness = max(0, t - j["dd"])
        result.append({
            "job":         name,
            "pt":          j["pt"],
            "dd":          j["dd"],
            "completion":  t,
            "flow_time":   t,
            "tardiness":   tardiness,
        })
    return result


def schedule_metrics(schedule: list) -> dict:
    """
    Aggregate metrics for a job sequence produced by spt_sequence (or EDD, etc.).
    ENHANCED: NEW — returns avg flow time, avg tardiness, makespan, jobs late.
    """
    if not schedule:
        return {}
    n          = len(schedule)
    avg_flow   = sum(j["flow_time"] for j in schedule) / n
    avg_tard   = sum(j["tardiness"] for j in schedule) / n
    makespan   = schedule[-1]["completion"]
    jobs_late  = sum(1 for j in schedule if j["tardiness"] > 0)
    return {
        "avg_flow_time": round(avg_flow, 3),
        "avg_tardiness": round(avg_tard, 3),
        "makespan":      round(makespan, 3),
        "jobs_late":     jobs_late,
        "n_jobs":        n,
    }


# ============================================================
# SIDEBAR RENDERER
# ============================================================
def render_sidebar(modules: dict):
    """
    Render the full sidebar navigation + theme toggle.
    modules: OrderedDict of { display_label: function_ref }
    Returns the selected module function.
    ENHANCED: bookmarks indicator, recent modules, version footer.
    FIX: removed unused pct variable.
    FIX: modules_visited is a set and cannot index — used len() correctly.
    """
    render_theme_toggle()
    st.sidebar.markdown("---")

    # ── Search / filter ───────────────────────────────────
    search = st.sidebar.text_input("🔍 Filter modules", "",
                                    placeholder="Type to search…",
                                    key="sidebar_search",
                                    label_visibility="collapsed")
    labels = list(modules.keys())
    if search.strip():
        labels = [l for l in labels if search.strip().lower() in l.lower()]
        if not labels:
            st.sidebar.caption("No modules match your search.")

    st.sidebar.markdown("### 📚 Modules")
    choice = st.sidebar.radio(
        "Select module:",
        labels if labels else list(modules.keys()),
        label_visibility="collapsed",
        key="module_radio",
    )

    # ── Track visits ─────────────────────────────────────
    st.session_state.modules_visited.add(choice)
    st.session_state.last_module = choice
    recent = st.session_state.get("recent_modules", [])
    if not recent or recent[0] != choice:
        recent = [choice] + [r for r in recent if r != choice]
        st.session_state.recent_modules = recent[:5]

    st.sidebar.markdown("---")

    # ── Progress indicator ────────────────────────────────
    visited = len(st.session_state.modules_visited)
    total   = len(modules)
    st.sidebar.markdown(f"**Progress:** {visited} / {total} modules explored")
    display_progress_bar(visited, total, bar_type="accent", show_pct=False)

    # ── Gamification stats ────────────────────────────────
    solved = st.session_state.get("problems_solved", 0)
    streak = st.session_state.get("correct_streak", 0)
    best   = st.session_state.get("best_streak", 0)
    if solved > 0:
        st.sidebar.markdown(
            f"🎯 **Problems solved:** {solved}  \n"
            f"🔥 **Streak:** {streak}  |  🏆 **Best:** {best}"
        )

    st.sidebar.markdown("---")

    # ── Version footer ────────────────────────────────────
    v = st.session_state.get("runtime_versions", {})
    st.sidebar.caption(
        f"📖 *Operations & Supply Chain Management*  \n"
        f"Jacobs & Chase (2024, 17th ed.)  \n"
        f"Streamlit {v.get('streamlit','—')} · Pandas {v.get('pandas','—')}"
    )

    return modules[choice]


# ============================================================
# PRE-COMPUTED Z-TABLE  (z from −3.49 to 3.49, step 0.01)
# ============================================================
Z_TABLE: dict = {
    round(z / 100, 2): round(float(stats.norm.cdf(z / 100)), 4)
    for z in range(-349, 350)
}


def z_lookup(z: float) -> float:
    """
    Look up P(Z ≤ z) from the pre-computed Z_TABLE.
    Rounds z to 2 decimal places; falls back to scipy for out-of-range.
    FIX: type annotation dict[float, float] replaced with plain dict for
    Python 3.8 compatibility (subscript syntax only available in 3.9+).
    """
    key = round(z, 2)
    return Z_TABLE.get(key, normal_cdf(z))


# ============================================================
# PRE-COMPUTED STANDARD NORMAL REFERENCE TABLE
# ============================================================
def build_z_reference_table() -> pd.DataFrame:
    """
    Returns a DataFrame with key Z-score ↔ probability mappings.
    ENHANCED: added 'σ Context' column mapping to common OSCM usage.
    FIX: dict key lookup replaced with a tuple list to avoid float
    key mismatch (e.g. -1.6500000001 not found in dict).
    FIX: Python 3.8 compat — removed dict[float, float] type hint.
    """
    z_annotations = [
        (-3.00, "Extremely rare — process alert",              "3σ lower limit"),
        (-2.58, "99% CI (two-tail lower)",                     ""),
        (-2.33, "99% conf (one-tail lower)",                   ""),
        (-2.00, "Lower 2.3%",                                  ""),
        (-1.96, "95% CI (two-tail lower)",                     ""),
        (-1.65, "95% conf (one-tail) / Newsvendor z",          "Safety stock (95% SL)"),
        (-1.28, "90% conf (one-tail)",                         "Safety stock (90% SL)"),
        (-1.00, "Lower 16%",                                   ""),
        (-0.50, "Lower 31%",                                   ""),
        ( 0.00, "50th percentile (median)",                    ""),
        ( 0.50, "Upper 31%",                                   ""),
        ( 1.00, "Upper 16%",                                   ""),
        ( 1.28, "90% conf (one-tail) / 80% CI",               ""),
        ( 1.65, "95% conf (one-tail) / 90% CI",               "Safety stock (95% SL)"),
        ( 1.96, "95% CI (two-tail)",                           ""),
        ( 2.00, "Upper 2.3%",                                  ""),
        ( 2.33, "99% conf (one-tail)",                         ""),
        ( 2.58, "99% CI (two-tail)",                           ""),
        ( 3.00, "Upper 0.13%",                                 "3σ UCL/LCL — control charts"),
    ]
    rows = []
    for z, note, context in z_annotations:
        p_le  = normal_cdf(z)
        p_gt  = 1.0 - p_le
        p_two = 2.0 * min(p_le, p_gt)
        rows.append({
            "Z-Score":        f"{z:+.2f}",
            "P(Z ≤ z)":       f"{p_le:.4f}",
            "P(Z > z)":       f"{p_gt:.4f}",
            "Two-Tail Area":  f"{p_two:.4f}",
            "Common Use":     note,
            "OSCM Context":   context,
        })
    return pd.DataFrame(rows)


# ── Control chart helper tables ───────────────────────────

_XBAR_R_CONSTANTS = {
#  n:  (A2,    D3,    D4,    d2)
    2:  (1.880, 0.000, 3.267, 1.128),
    3:  (1.023, 0.000, 2.574, 1.693),
    4:  (0.729, 0.000, 2.282, 2.059),
    5:  (0.577, 0.000, 2.114, 2.326),
    6:  (0.483, 0.000, 2.004, 2.534),
    7:  (0.419, 0.076, 1.924, 2.704),
    8:  (0.373, 0.136, 1.864, 2.847),
    9:  (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}


def get_xbar_r_constants(n: int) -> dict:
    """
    Return the standard x̄-R control chart constants for subgroup size n.
    ENHANCED: NEW — eliminates magic numbers scattered across modules.
    Returns {"A2", "D3", "D4", "d2"} or None if n out of range.
    """
    if n not in _XBAR_R_CONSTANTS:
        return None
    a2, d3, d4, d2 = _XBAR_R_CONSTANTS[n]
    return {"A2": a2, "D3": d3, "D4": d4, "d2": d2}


def build_control_chart_constants_table() -> pd.DataFrame:
    """
    Returns a DataFrame of all x̄-R constants for display in Formula Reference tabs.
    ENHANCED: NEW.
    """
    rows = []
    for n, (a2, d3, d4, d2) in _XBAR_R_CONSTANTS.items():
        rows.append({"n": n, "A₂": a2, "D₃": d3, "D₄": d4, "d₂": d2})
    return pd.DataFrame(rows)

# ============================================================
# MODULE 1: SUPPLY CHAIN RISK (Chapter 1) - ENHANCED V5.0
# ============================================================
def module_risk():
    display_header("🛡️", "Chapter 1", "Supply Chain Risk Assessment",
                   "Probability-Impact analysis, EMV prioritization, and mitigation strategies")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "🔬 Risk Matrix", "📊 EMV Prioritizer", "🎓 Practice"])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Operations & Supply Chain Management")
        st.write(
            "**Operations and Supply Chain Management (OSCM)** is the design, operation, "
            "and improvement of systems that create and deliver a firm's products and services. "
            "Every business function — finance, marketing, accounting — depends on these systems, "
            "making supply chain risk one of the most consequential areas of management."
        )

        display_citation(
            "OSCM is defined as the design, operation, and improvement of the systems that create "
            "and deliver the firm's primary products and services. Like accounting and finance, "
            "OSCM is a functional field of business with clear line management responsibilities.",
            "Jacobs & Chase (2024, p. 7)"
        )

        st.markdown("#### The Three Elements of OSCM Integration")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("📊", "Strategy",
                                  "Competitive positioning and service vision; "
                                  "defines what trade-offs the firm will and will not make")
        with col2:
            display_concept_card("⚙️", "Processes",
                                  "Operations that create and deliver value; "
                                  "the physical and information flows that run the business")
        with col3:
            display_concept_card("👥", "People",
                                  "Workforce skills and organizational culture; "
                                  "the human capability to execute strategy through processes")

        st.markdown("### Supply Chain Risk Framework")
        st.write(
            "**Supply chain risk** is the likelihood of a disruption that would impair "
            "a company's ability to continuously supply products or services. "
            "Effective risk management requires identifying, quantifying, and mitigating "
            "risks before they materialize."
        )

        display_citation(
            "Supply chain risk management involves the identification of potential sources of risk "
            "and implementation of appropriate strategies through a coordinated approach among "
            "supply chain members to reduce supply chain vulnerability.",
            "Jacobs & Chase (2024, p. 12)"
        )

        display_formula_card("Risk Score",
            r"\text{Risk Score} = \text{Probability (1--5)} \times \text{Impact (1--5)}")
        display_formula_card("Expected Monetary Value (EMV)",
            r"EMV = \text{Probability} \times \text{Financial Impact (\$)}")

        st.markdown("#### Risk Management Three-Step Process")
        steps_df = pd.DataFrame({
            "Step":       [1, 2, 3],
            "Phase":      ["Identification", "Assessment", "Mitigation"],
            "Question":   ["What could go wrong?",
                            "How likely and how bad?",
                            "What can we do about it?"],
            "Tools":      ["SIPOC, brainstorming, historical data",
                            "Risk matrix (P × I), EMV calculation",
                            "Redundancy, insurance, contracts, diversification"],
            "Output":     ["Risk register",
                            "Prioritized risk list (by score or EMV)",
                            "Contingency plans and monitoring protocols"]
        })
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

        st.markdown("#### Risk Categories & Examples")
        cat_df = pd.DataFrame({
            "Category":    ["Supply-Side", "Demand-Side", "Operational",
                             "External", "Financial", "Technology"],
            "Examples":    ["Supplier bankruptcy, single-source dependency, quality failures",
                             "Demand spikes/drops, forecast error, customer concentration",
                             "Machine breakdown, labor strikes, process failures",
                             "Natural disasters, geopolitical instability, pandemics",
                             "Currency fluctuation, credit risk, commodity price volatility",
                             "Cybersecurity breach, IT outage, ERP failure"],
            "Mitigation":  ["Dual/multi-source, safety stock, supplier audits",
                             "Flexible capacity, postponement strategy",
                             "Preventive maintenance, cross-training, redundancy",
                             "Geographic diversification, business continuity planning",
                             "Hedging, diversified revenue, credit insurance",
                             "Backups, incident response plan, cybersecurity protocols"]
        })
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

        display_textbook_content(
            "The Triple Bottom Line",
            """Modern OSCM extends beyond profit to the Triple Bottom Line (TBL):
            • People — social responsibility: fair labor, community impact, employee welfare
            • Planet — environmental sustainability: emissions, waste, resource use
            • Profit — economic performance: long-term financial viability
            Companies that balance all three tend to have more resilient supply chains
            and outperform peers over the long run (Jacobs & Chase, 2024, p. 22)."""
        )

        display_textbook_content(
            "Process Analysis — A Basic Skill",
            """Process analysis is a foundational OSCM skill. Drawing a simple flowchart
            of material or information flow reveals that 90% or more of the time required
            to serve a customer is often spent waiting, not being processed.
            Eliminating wait time alone can dramatically improve performance
            without any capital investment."""
        )

        display_key_insight(
            "Efficiency vs. Effectiveness",
            "**Efficiency** = doing things right (minimizing cost per unit of output).  \n"
            "**Effectiveness** = doing the right things (producing the output customers value).  \n"
            "A company can be perfectly efficient at producing the wrong product — "
            "optimizing for efficiency at the expense of effectiveness is one of the most "
            "common strategic failures in operations management."
        )

        display_key_insight(
            "Straddling — Why Trying to Be Everything Fails",
            "Straddling occurs when a firm tries to copy a competitor's strategy while "
            "maintaining its existing position — e.g., a full-service airline launching "
            "a 'budget' subsidiary. Result: conflicting operational requirements, diluted "
            "brand, neither strategy executed well. The classic example is Continental's "
            "'Continental Lite' attempt to replicate Southwest Airlines."
        )

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Interactive Risk Assessment Matrix")
        st.write(
            "Score each risk from 1 (Low) to 5 (High) on both dimensions. "
            "Risk Score = Probability × Impact. Scores ≥ 15 require immediate action."
        )

        col1, col2 = st.columns([1.1, 0.9])

        with col1:
            st.markdown("#### Risk Event Scoring")
            risks = []
            risk_names = [
                "Supplier Failure (Financial)",
                "Natural Disaster / Weather",
                "Quality Issue / Product Recall",
                "Logistics / Customs Delay",
                "Demand Volatility",
                "Cybersecurity Breach",
                "Geopolitical Disruption",
                "Labor Strike / Workforce"
            ]
            risk_icons = ["💰", "🌪️", "⚠️", "🚚", "📈", "💻", "🌍", "👷"]

            for i, (name, icon) in enumerate(zip(risk_names, risk_icons)):
                with st.expander(f"{icon} {name}", expanded=(i < 3)):
                    c1, c2 = st.columns(2)
                    with c1:
                        prob = st.slider(
                            "Probability", 1, 5, [3, 4, 3, 4, 3, 2, 2, 2][i],
                            key=f"risk_p_{i}",
                            help="1=Rare  2=Unlikely  3=Possible  4=Likely  5=Almost Certain"
                        )
                    with c2:
                        impact = st.slider(
                            "Impact", 1, 5, [4, 5, 4, 3, 3, 5, 4, 3][i],
                            key=f"risk_i_{i}",
                            help="1=Negligible  2=Minor  3=Moderate  4=Major  5=Catastrophic"
                        )
                    score = prob * impact
                    tier  = "🔴 HIGH"   if score >= 15 else ("🟡 MEDIUM" if score >= 8 else "🟢 LOW")
                    st.write(f"Risk Score: **{score}** — {tier}")
                    risks.append({
                        "name":   name,
                        "prob":   prob,
                        "impact": impact,
                        "score":  score
                    })

        with col2:
            st.markdown("#### Risk Analysis Results")
            df_risk = pd.DataFrame(risks)
            df_risk.columns = ["Risk Event", "Probability", "Impact", "Risk Score"]
            df_risk = df_risk.sort_values("Risk Score", ascending=False).reset_index(drop=True)
            df_risk.index = df_risk.index + 1

            # Color-coded tier column
            df_risk["Priority"] = df_risk["Risk Score"].apply(
                lambda s: "🔴 HIGH" if s >= 15 else ("🟡 MEDIUM" if s >= 8 else "🟢 LOW")
            )
            st.dataframe(df_risk, use_container_width=True)

            total_score    = sum(r["score"] for r in risks)
            max_score      = len(risks) * 25
            risk_pct       = (total_score / max_score) * 100
            high_count     = sum(1 for r in risks if r["score"] >= 15)
            critical_score = max(r["score"] for r in risks)

            col_a, col_b = st.columns(2)
            col_a.metric("Total Risk Score",     f"{total_score} / {max_score}")
            col_b.metric("Exposure Level",       f"{risk_pct:.0f}%")
            col_a.metric("High-Priority Risks",  f"{high_count}")
            col_b.metric("Highest Single Score", f"{critical_score}")

            # Progress bar
            st.markdown("#### Overall Exposure")
            bar_color = "danger" if risk_pct > 60 else ("success" if risk_pct < 40 else "normal")
            st.progress(risk_pct / 100,
                         text=f"Portfolio Exposure: {risk_pct:.0f}%  "
                              f"({'🔴 HIGH' if risk_pct > 60 else '🟡 MODERATE' if risk_pct > 40 else '🟢 LOW'})")

        # ── Priority Alerts ──
        st.markdown("---")
        high_risks = [r for r in risks if r["score"] >= 15]
        med_risks  = [r for r in risks if 8 <= r["score"] < 15]
        low_risks  = [r for r in risks if r["score"] < 8]

        if high_risks:
            st.error(
                f"🔴 **IMMEDIATE ACTION ({len(high_risks)}):** "
                f"{', '.join(r['name'] for r in sorted(high_risks, key=lambda x: -x['score']))}"
                f"\n*Develop and activate contingency plans now.*"
            )
        if med_risks:
            st.warning(
                f"🟡 **MONITOR CLOSELY ({len(med_risks)}):** "
                f"{', '.join(r['name'] for r in sorted(med_risks, key=lambda x: -x['score']))}"
                f"\n*Review quarterly; maintain contingency plans.*"
            )
        if low_risks:
            st.success(
                f"🟢 **PERIODIC REVIEW ({len(low_risks)}):** "
                f"{', '.join(r['name'] for r in low_risks)}"
                f"\n*Annual review sufficient.*"
            )

        # ── Plotly Bubble Chart ──
        st.markdown("#### Risk Heat Map")
        fig_risk = go.Figure()

        tier_colors = {
            "🔴 HIGH":   "#e74c3c",
            "🟡 MEDIUM": "#f39c12",
            "🟢 LOW":    "#2ecc71"
        }

        for _, row in df_risk.iterrows():
            color = tier_colors[row["Priority"]]
            fig_risk.add_trace(go.Scatter(
                x=[row["Probability"]],
                y=[row["Impact"]],
                mode="markers+text",
                marker=dict(size=row["Risk Score"] * 4,
                             color=color, opacity=0.7,
                             line=dict(color="white", width=1.5)),
                text=[row["Risk Event"].split(" (")[0][:20]],
                textposition="top center",
                textfont=dict(size=9),
                name=row["Risk Event"],
                hovertemplate=(
                    f"<b>{row['Risk Event']}</b><br>"
                    f"P={row['Probability']}  I={row['Impact']}<br>"
                    f"Score={row['Risk Score']}<extra></extra>"
                )
            ))

        # Threshold lines
        for threshold, color, label in [(3, "#f39c12", "Medium threshold"),
                                          (3, "#e74c3c",  "High threshold")]:
            pass  # drawn via shapes

        fig_risk.add_shape(type="rect", x0=0.5, y0=0.5, x1=3, y1=3,
                            fillcolor="rgba(46,204,113,0.08)", line_width=0)
        fig_risk.add_shape(type="rect", x0=3, y0=0.5, x1=5.5, y1=3,
                            fillcolor="rgba(243,156,18,0.08)", line_width=0)
        fig_risk.add_shape(type="rect", x0=0.5, y0=3, x1=3, y1=5.5,
                            fillcolor="rgba(243,156,18,0.08)", line_width=0)
        fig_risk.add_shape(type="rect", x0=3, y0=3, x1=5.5, y1=5.5,
                            fillcolor="rgba(231,76,60,0.08)", line_width=0)

        fig_risk.update_layout(
            title="Risk Heat Map  (bubble size ∝ Risk Score)",
            xaxis=dict(title="Probability", range=[0.5, 5.5],
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=["1 Rare", "2 Unlikely", "3 Possible",
                                   "4 Likely", "5 Certain"]),
            yaxis=dict(title="Impact", range=[0.5, 5.5],
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=["1 Negligible", "2 Minor", "3 Moderate",
                                   "4 Major", "5 Catastrophic"]),
            showlegend=False,
            template="plotly_white",
            height=480
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### EMV-Based Risk Prioritizer")
        st.write(
            "Extend the qualitative 1–5 matrix to dollar-denominated Expected Monetary Values. "
            "This allows direct comparison of risks against mitigation investment costs."
        )

        num_emv_risks = st.number_input("Number of Risks to Evaluate", 2, 8, 3,
                                         key="emv_num")
        emv_risks = []

        cols_hdr = st.columns([1.8, 1, 1.2, 1, 1.2])
        for h, c in zip(["Risk Event", "Probability", "Financial Impact ($)",
                          "Mitigation Cost ($)", "Net Value of Mitigation ($)"], cols_hdr):
            c.markdown(f"**{h}**")

        for i in range(int(num_emv_risks)):
            cols = st.columns([1.8, 1, 1.2, 1, 1.2])
            with cols[0]:
                rname = st.text_input("", value=f"Risk {chr(65+i)}",
                                       key=f"emv_name_{i}", label_visibility="collapsed")
            with cols[1]:
                prob_f = st.number_input("", value=[0.15, 0.30, 0.05][i % 3],
                                          min_value=0.0, max_value=1.0, step=0.01,
                                          format="%.2f", key=f"emv_p_{i}",
                                          label_visibility="collapsed")
            with cols[2]:
                fin_impact = st.number_input("", value=[2000000, 500000, 10000000][i % 3],
                                              min_value=0, step=50000,
                                              key=f"emv_imp_{i}",
                                              label_visibility="collapsed")
            with cols[3]:
                mit_cost = st.number_input("", value=[200000, 50000, 300000][i % 3],
                                            min_value=0, step=10000,
                                            key=f"emv_mit_{i}",
                                            label_visibility="collapsed")

            emv_val    = prob_f * fin_impact
            net_value  = emv_val - mit_cost  # positive = worth mitigating

            with cols[4]:
                color = "✅" if net_value > 0 else "❌"
                st.write(f"{color} ${net_value:,.0f}")

            emv_risks.append({
                "Risk":            rname,
                "Probability":     prob_f,
                "Financial Impact": fin_impact,
                "EMV":             emv_val,
                "Mitigation Cost": mit_cost,
                "Net Value":       net_value,
                "Mitigate?":       "✅ Yes" if net_value > 0 else "❌ No"
            })

        if emv_risks:
            df_emv = pd.DataFrame(emv_risks).sort_values("EMV", ascending=False).reset_index(drop=True)
            df_emv.index = df_emv.index + 1

            st.markdown("#### EMV Priority Table")
            display_df = df_emv.copy()
            display_df["Financial Impact"] = display_df["Financial Impact"].apply(lambda x: f"${x:,.0f}")
            display_df["EMV"]              = display_df["EMV"].apply(lambda x: f"${x:,.0f}")
            display_df["Mitigation Cost"]  = display_df["Mitigation Cost"].apply(lambda x: f"${x:,.0f}")
            display_df["Net Value"]        = display_df["Net Value"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True)

            total_emv = sum(r["EMV"] for r in emv_risks)
            top_risk  = max(emv_risks, key=lambda x: x["EMV"])
            st.metric("Total Risk Exposure (EMV)", f"${total_emv:,.0f}")
            st.info(
                f"💡 Highest priority: **{top_risk['Risk']}** "
                f"(EMV = ${top_risk['EMV']:,.0f}). "
                f"{'Mitigation is **cost-effective** — net value is positive.' if top_risk['Net Value'] > 0 else 'Mitigation cost exceeds EMV — explore lower-cost alternatives.'}"
            )

            # ── Plotly EMV vs Mitigation Cost Bar ──
            fig_emv_chart = go.Figure()
            names = [r["Risk"] for r in emv_risks]
            emv_vals = [r["EMV"] for r in emv_risks]
            mit_vals = [r["Mitigation Cost"] for r in emv_risks]

            fig_emv_chart.add_trace(go.Bar(
                name="EMV (Risk Exposure)", x=names, y=emv_vals,
                marker_color="#e74c3c",
                text=[f"${v:,.0f}" for v in emv_vals], textposition="outside"))
            fig_emv_chart.add_trace(go.Bar(
                name="Mitigation Cost", x=names, y=mit_vals,
                marker_color="#3498db",
                text=[f"${v:,.0f}" for v in mit_vals], textposition="outside"))

            fig_emv_chart.update_layout(
                barmode="group",
                title="EMV vs. Mitigation Cost — Risks Worth Mitigating When EMV > Mitigation Cost",
                xaxis_title="Risk Event",
                yaxis_title="$ Amount",
                template="plotly_white",
                height=380,
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig_emv_chart, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Triple Bottom Line (Easy)"):
            display_practice_problem(1, "Easy",
                "What are the three components of the Triple Bottom Line (TBL), "
                "and why is each important for sustainable supply chain operations?")

            show_h1 = st.checkbox("Show Hint", key="risk_h1")
            if show_h1:
                display_hint("Think about the three P's: People, Profit, and Planet.")

            if st.button("Show Complete Solution", key="risk_q1_v2"):
                display_solution("""
                <strong>The Triple Bottom Line (3BL)</strong> evaluates a firm on three dimensions:<br><br>
                <strong>1. Social (People)</strong><br>
                • Impact on employees, communities, and broader society<br>
                • Fair wages, safe working conditions, community investment<br>
                • Ethical sourcing — no forced or child labor in the supply chain<br><br>
                <strong>2. Economic (Profit)</strong><br>
                • Long-term financial viability, not just quarterly earnings<br>
                • Value creation for shareholders AND other stakeholders<br>
                • Resilient revenue streams that withstand disruptions<br><br>
                <strong>3. Environmental (Planet)</strong><br>
                • Carbon footprint, emissions, water use, waste generation<br>
                • Sustainable sourcing of materials and energy<br>
                • Circular economy principles (reduce, reuse, recycle)<br><br>
                <em>Key Insight:</em> Companies balancing all three tend to have more resilient
                supply chains. ESG-rated firms demonstrate lower volatility and better
                long-term risk-adjusted returns.
                """)

        with st.expander("🟡 Problem 2: Efficiency vs. Effectiveness (Medium)"):
            display_practice_problem(2, "Medium",
                "A manufacturing company reduced its production costs by 15%, "
                "but customer complaints increased by 25%.  \n"
                "Analyze this using the concepts of efficiency and effectiveness. "
                "What went wrong, and what should management do?")

            show_h2 = st.checkbox("Show Hint", key="risk_h2")
            if show_h2:
                display_hint(
                    "Efficiency = doing things right (low cost per unit). "
                    "Effectiveness = doing the right things (delivering customer value). "
                    "A company can be both, neither, or one without the other."
                )

            if st.button("Show Complete Solution", key="risk_q2_v2"):
                display_solution("""
                <strong>Efficiency Improved:</strong><br>
                Costs fell 15% — the process is now more efficient (less input per output).<br><br>
                <strong>Effectiveness Declined:</strong><br>
                Complaints rose 25% — the output no longer meets customer requirements.<br>
                The cost cuts likely compromised quality, delivery speed, or service levels.<br><br>
                <strong>What Went Wrong:</strong><br>
                The company fell into the classic "efficiency trap" — optimizing one metric
                (cost) while damaging another (customer value). Common causes:<br>
                • Reduced material quality to save money<br>
                • Cut inspection/QC steps<br>
                • Reduced customer service staffing<br>
                • Longer lead times due to fewer shipments<br><br>
                <strong>Recommendation:</strong><br>
                1. Use customer feedback to identify which specific cuts caused problems<br>
                2. Restore those specific items — not all cuts were harmful<br>
                3. Target cost reductions that do NOT touch customer-facing quality attributes<br>
                4. Balance KPIs: track both internal efficiency AND customer satisfaction
                """)

        with st.expander("🔴 Problem 3: EMV Risk Prioritization (Hard)"):
            display_practice_problem(3, "Hard",
                "A company identifies three supply chain risks:  \n"
                "- **Risk A:** P = 0.15, Financial Impact = $2,000,000  \n"
                "- **Risk B:** P = 0.30, Financial Impact = $500,000  \n"
                "- **Risk C:** P = 0.05, Financial Impact = $10,000,000  \n\n"
                "a) Calculate EMV for each risk.  \n"
                "b) Which risk has the highest EMV?  \n"
                "c) Total risk exposure?  \n"
                "d) Risk B's mitigation costs $100,000. Is it worth it?")

            col1, col2, col3 = st.columns(3)
            with col1:
                u_a = st.number_input("EMV Risk A ($):", key="risk_emv_a_v2")
            with col2:
                u_b = st.number_input("EMV Risk B ($):", key="risk_emv_b_v2")
            with col3:
                u_c = st.number_input("EMV Risk C ($):", key="risk_emv_c_v2")

            u_total    = st.number_input("Total Exposure ($):", key="risk_total_v2")
            u_priority = st.selectbox("Highest Priority Risk:",
                                       ["Select...", "Risk A", "Risk B", "Risk C"],
                                       key="risk_pri_v2")
            u_worth    = st.selectbox("Mitigate Risk B for $100K?",
                                       ["Select...", "Yes", "No"],
                                       key="risk_worth_v2")

            if st.button("Check All Answers", key="risk_q3_btn_v2"):
                ca = 0.15*2000000; cb = 0.30*500000; cc = 0.05*10000000
                total_c = ca+cb+cc
                priority_c = "Risk C"  # highest EMV
                worth_c = "Yes" if cb > 100000 else "No"

                results = []
                for u, c, lbl in [(u_a, ca, "EMV A"), (u_b, cb, "EMV B"), (u_c, cc, "EMV C")]:
                    if check_answer(u, c):   results.append(f"✅ {lbl}: ${c:,.0f} correct")
                    else:                     results.append(f"❌ {lbl}: Should be ${c:,.0f}")
                if check_answer(u_total, total_c):
                    results.append(f"✅ Total: ${total_c:,.0f} correct")
                else:
                    results.append(f"❌ Total: Should be ${total_c:,.0f}")
                if u_priority == priority_c:
                    results.append(f"✅ Priority: {priority_c} correct")
                elif u_priority != "Select...":
                    results.append(f"❌ Priority: Should be {priority_c} (EMV=${cc:,.0f})")
                if u_worth == worth_c:
                    results.append(f"✅ Mitigation decision correct")
                elif u_worth != "Select...":
                    results.append(f"❌ Mitigation: {'Yes' if worth_c=='Yes' else 'No'} — "
                                    f"EMV=${ cb:,.0f} {'>' if worth_c=='Yes' else '<'} $100,000")
                for r in results:
                    st.write(r)

            if st.button("Show Complete Solution", key="risk_q3_sol_v2"):
                ca=0.15*2000000; cb=0.30*500000; cc=0.05*10000000
                display_solution(f"""
                <strong>a) EMV Calculations (EMV = P × Impact)</strong><br>
                Risk A: 0.15 × $2,000,000 = <strong>${ca:,.0f}</strong><br>
                Risk B: 0.30 × $500,000 &nbsp;= <strong>${cb:,.0f}</strong><br>
                Risk C: 0.05 × $10,000,000 = <strong>${cc:,.0f}</strong><br><br>
                <strong>b) Priority Ranking by EMV</strong><br>
                1. Risk C — ${cc:,.0f} ← Highest (catastrophic consequence despite low P)<br>
                2. Risk A — ${ca:,.0f}<br>
                3. Risk B — ${cb:,.0f}<br><br>
                <strong>c) Total Risk Exposure</strong><br>
                ${ca:,.0f} + ${cb:,.0f} + ${cc:,.0f} = <strong>${ca+cb+cc:,.0f}</strong><br><br>
                <strong>d) Mitigate Risk B for $100,000?</strong><br>
                EMV(B) = ${cb:,.0f} {'> $100,000 → ✅ YES, mitigation is cost-effective' if cb > 100000 else '< $100,000 → ❌ NO, mitigation costs more than expected loss'}<br><br>
                <em>Key insight: Risk C has the highest EMV despite having the lowest probability (5%),
                because its impact is catastrophic ($10M). Never ignore low-probability,
                high-impact risks.</em>
                """)

        with st.expander("🟡 Problem 4: Risk Score Matrix (Medium)"):
            display_practice_problem(4, "Medium",
                "Score each risk on the 1–5 matrix and classify as High/Medium/Low:  \n\n"
                "| Risk | Probability | Impact |  \n"
                "|------|-------------|--------|  \n"
                "| Supplier bankruptcy | 2 | 5 |  \n"
                "| Minor delivery delay | 4 | 2 |  \n"
                "| Major IT outage | 2 | 4 |  \n"
                "| Demand spike | 3 | 3 |  \n\n"
                "a) Calculate all four risk scores.  \n"
                "b) Which risk(s) are HIGH priority (score ≥ 15)?  \n"
                "c) Which is more dangerous per mitigation dollar: supplier bankruptcy or IT outage?")

            col1, col2, col3, col4 = st.columns(4)
            with col1: u_s1 = st.number_input("Supplier score:", key="risk_p4_s1")
            with col2: u_s2 = st.number_input("Delay score:", key="risk_p4_s2")
            with col3: u_s3 = st.number_input("IT outage score:", key="risk_p4_s3")
            with col4: u_s4 = st.number_input("Demand spike score:", key="risk_p4_s4")

            if st.button("Check Answers", key="risk_p4_btn"):
                s1=2*5; s2=4*2; s3=2*4; s4=3*3
                for u, c, lbl in [(u_s1,s1,"Supplier"),(u_s2,s2,"Delay"),
                                   (u_s3,s3,"IT outage"),(u_s4,s4,"Demand spike")]:
                    if check_answer(u, c):  st.write(f"✅ {lbl}: {c} correct")
                    else:                   st.write(f"❌ {lbl}: Should be {c}")

            if st.button("Show Solution", key="risk_p4_sol"):
                display_solution("""
                <strong>a) Risk Scores</strong><br>
                Supplier bankruptcy: 2 × 5 = <strong>10</strong> 🟡 MEDIUM<br>
                Minor delivery delay: 4 × 2 = <strong>8</strong> 🟡 MEDIUM (borderline)<br>
                Major IT outage: 2 × 4 = <strong>8</strong> 🟡 MEDIUM<br>
                Demand spike: 3 × 3 = <strong>9</strong> 🟡 MEDIUM<br><br>
                <strong>b) HIGH priority (score ≥ 15):</strong> None in this set.<br>
                Closest: Supplier bankruptcy (10) and Demand spike (9).<br><br>
                <strong>c) Supplier bankruptcy vs. IT outage (both score 8–10)</strong><br>
                Supplier bankruptcy: Score=10 — higher because Impact=5 (catastrophic)<br>
                IT outage: Score=8 — impact is major but slightly lower<br>
                Despite similar scores, <strong>supplier bankruptcy</strong> is more dangerous
                per mitigation dollar because Severity=5 means it could be existential.
                Always pay special attention to any risk with Impact=5 regardless of score.
                """)

        with st.expander("🟡 Problem 5: Straddling & Competitive Strategy (Medium)"):
            display_practice_problem(5, "Medium",
                "What is 'straddling' in competitive strategy?  \n"
                "a) Define straddling and explain why it typically fails operationally.  \n"
                "b) Describe the Continental Airlines / Continental Lite case.  \n"
                "c) What does this imply for supply chain design?")

            if st.button("Show Complete Solution", key="risk_q5_v2"):
                display_solution("""
                <strong>a) Straddling Defined</strong><br>
                Straddling occurs when a company tries to copy a successful competitor's
                position while <em>also</em> maintaining its existing position — attempting
                to be "all things to all customers."<br><br>
                <strong>Why It Fails:</strong><br>
                • Conflicting operational requirements (e.g., speed vs. low cost)<br>
                • Diluted brand: customers don't know what you stand for<br>
                • "Stuck in the middle" — neither the lowest cost nor the most differentiated<br>
                • Resources spread thin; no activity system is optimized<br><br>
                <strong>b) Continental Airlines Case</strong><br>
                Continental launched "Continental Lite" to replicate Southwest Airlines' model.<br>
                Results: customer confusion, scheduling conflicts between full-service and
                low-cost operations, inconsistent service quality, financial losses.
                Continental eventually abandoned the strategy entirely.<br><br>
                <strong>c) Supply Chain Implications</strong><br>
                Supply chains must be <em>aligned</em> with competitive strategy.
                A low-cost supply chain (lean, efficient) conflicts with a
                differentiation supply chain (flexible, responsive, customized).
                Trying to run both simultaneously creates cost overruns and service failures.
                Make a clear strategic choice and design your supply chain to support it.
                """)

        with st.expander("🔴 Problem 6: What-If Sensitivity on Profitability (Hard)"):
            display_practice_problem(6, "Hard",
                "A product has: Revenue = $500,000 | Development Cost = $200,000 | "
                "Variable Cost = $10/unit | Selling Price = $50/unit | "
                "Expected Volume = 10,000 units  \n\n"
                "Calculate the base profit, then analyze:  \n"
                "a) Development time increases 25% (assume $50,000 additional cost)  \n"
                "b) Sales volume decreases 25%  \n"
                "c) Variable cost increases $1/unit  \n"
                "d) Which factor has the largest impact on profit?")

            if st.button("Show Complete Solution", key="risk_q6_sol"):
                base_rev = 50*10000; base_cost = 200000 + 10*10000
                base_profit = base_rev - base_cost

                # a) Dev time +25% = +$50k cost
                profit_a = base_rev - (base_cost + 50000)
                # b) Volume -25% = 7500 units
                profit_b = 50*7500 - (200000 + 10*7500)
                # c) VC +$1
                profit_c = base_rev - (200000 + 11*10000)

                display_solution(f"""
                <strong>Base Case</strong><br>
                Revenue = $50 × 10,000 = $500,000<br>
                Total Cost = $200,000 + ($10 × 10,000) = $300,000<br>
                Base Profit = $500,000 − $300,000 = <strong>${base_profit:,.0f}</strong><br><br>
                <strong>a) Dev Time +25% (+$50,000 cost)</strong><br>
                Profit = ${base_profit:,.0f} − $50,000 = <strong>${profit_a:,.0f}</strong>
                (drop of ${base_profit-profit_a:,.0f}  = {(base_profit-profit_a)/base_profit:.0%})<br><br>
                <strong>b) Volume −25% (7,500 units)</strong><br>
                Revenue = $50 × 7,500 = $375,000<br>
                Cost = $200,000 + $10 × 7,500 = $275,000<br>
                Profit = <strong>${profit_b:,.0f}</strong>
                (drop of ${base_profit-profit_b:,.0f}  = {(base_profit-profit_b)/base_profit:.0%})<br><br>
                <strong>c) Variable Cost +$1/unit</strong><br>
                Cost = $200,000 + $11 × 10,000 = $310,000<br>
                Profit = <strong>${profit_c:,.0f}</strong>
                (drop of ${base_profit-profit_c:,.0f}  = {(base_profit-profit_c)/base_profit:.0%})<br><br>
                <strong>d) Most Impactful Factor</strong><br>
                Volume decline (−25%) causes the largest profit drop ({(base_profit-profit_b)/base_profit:.0%}),
                followed by development cost overrun ({(base_profit-profit_a)/base_profit:.0%}).
                Variable cost increase has the smallest impact at this volume.
                """)


# ============================================================
# MODULE 2: PERT NETWORK (Chapter 4) - ENHANCED V5.0
# ============================================================
def module_pert():
    display_header("🔗", "Chapter 4", "PERT Network & Completion Probability",
                   "Probabilistic time estimates, critical path variance, and Z-score analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "🔬 Activity Estimator", "📊 Probability Calculator", "🎓 Practice"])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### PERT Network Analysis")
        st.write(
            "**PERT (Program Evaluation and Review Technique)** augments CPM with "
            "probabilistic time estimates, acknowledging that activity durations are "
            "uncertain. Rather than a single estimate, each activity gets three: "
            "optimistic, most likely, and pessimistic. This yields both expected durations "
            "and a measure of uncertainty (variance) for each path."
        )

        display_citation(
            "A conservative approach dictates using the critical path with the largest total "
            "variance to focus management's attention on the activities most likely to exhibit "
            "broad variations in completion time.",
            "Jacobs & Chase (2024, p. 99)"
        )

        st.markdown("#### CPM vs. PERT at a Glance")
        cpm_pert_df = pd.DataFrame({
            "Feature":         ["Time estimates", "Uncertainty", "Output", "Primary use",
                                  "Statistical tool", "Best for"],
            "CPM":             ["Single deterministic estimate", "Ignored",
                                  "Critical path + float/slack", "Resource management",
                                  "None", "Well-understood, repetitive projects"],
            "PERT":            ["Three estimates (a, m, b)", "Explicitly modeled via σ²",
                                  "Critical path + completion probability",
                                  "Deadline risk management",
                                  "Beta distribution → Normal approximation",
                                  "R&D, new product development, uncertain projects"]
        })
        st.dataframe(cpm_pert_df, use_container_width=True, hide_index=True)

        st.markdown("### Core Formulas")
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("PERT Expected Time",
                r"T_E = \frac{a + 4m + b}{6}")
            st.write(
                "**a** = optimistic (best case)  \n"
                "**m** = most likely (mode)  \n"
                "**b** = pessimistic (worst case)"
            )
            display_formula_card("Activity Standard Deviation",
                r"\sigma = \frac{b - a}{6}")
        with col2:
            display_formula_card("Activity Variance",
                r"\sigma^2 = \left(\frac{b - a}{6}\right)^2")
            display_formula_card("Path Variance (sum of activities)",
                r"\sigma^2_{path} = \sum_{i \in CP} \sigma^2_i")
            display_formula_card("Z-Score for Deadline",
                r"Z = \frac{D - T_{E,path}}{\sqrt{\sigma^2_{path}}}")

        st.markdown("### Activity Float (Slack)")
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Total Float (Slack)",
                r"TF = LS - ES = LF - EF")
        with col2:
            display_formula_card("Free Float",
                r"FF = ES_{successor} - EF_{activity}")

        float_df = pd.DataFrame({
            "Term":        ["Early Start (ES)", "Early Finish (EF)",
                             "Late Start (LS)", "Late Finish (LF)", "Total Float (TF)"],
            "Definition":  ["Earliest an activity can start",
                             "ES + Duration",
                             "Latest an activity can start without delaying project",
                             "LS + Duration (= project deadline for critical activities)",
                             "LS − ES (= 0 for critical path activities)"],
            "Calculation": ["Forward pass from start",
                             "ES + T_E",
                             "Backward pass from end",
                             "LF of predecessor chain",
                             "LS − ES  or  LF − EF"]
        })
        st.dataframe(float_df, use_container_width=True, hide_index=True)

        display_textbook_content(
            "The Beta Distribution in PERT",
            """PERT uses the beta distribution because it is bounded (has a defined minimum
            and maximum), can be asymmetric (capturing real-world skew), and is flexible
            enough to model many shapes. The formula T_E = (a + 4m + b)/6 approximates
            the mean of a beta distribution, where the weight of 4 on m reflects that
            the most-likely value carries the most information. The variance formula
            σ² = ((b−a)/6)² is a conservative approximation of the beta variance."""
        )

        display_key_insight(
            "Critical Path Selection for Probability",
            "When two paths have the same expected duration, ALWAYS use the path with the "
            "**larger total variance** for probability calculations. More variance = more "
            "uncertainty = lower probability of on-time completion. This conservative approach "
            "protects against underestimating schedule risk."
        )

        st.markdown("### Standard Normal Distribution Table")
        z_data = []
        for z in [-2.5, -2.0, -1.65, -1.28, -1.0, -0.5, 0.0,
                   0.5, 1.0, 1.28, 1.65, 1.96, 2.0, 2.33, 2.5, 3.0]:
            z_data.append({
                "Z":          z,
                "P(Z ≤ z)":   f"{normal_cdf(z):.4f}",
                "P(Z > z)":   f"{1 - normal_cdf(z):.4f}",
                "Typical Use": {
                    -1.65: "10% chance deadline missed",
                    -1.28: "10% ahead of schedule",
                     0.0:  "50% chance on time",
                     1.28: "90% confidence",
                     1.65: "95% confidence",
                     1.96: "97.5% confidence",
                     2.33: "99% confidence"
                }.get(z, "")
            })
        st.dataframe(pd.DataFrame(z_data), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Single Activity Time Estimator")

        col1, col2 = st.columns(2)
        with col1:
            a = st.slider("Optimistic Time (a)", 1, 20, 4, help="Best-case — rare, everything goes right")
            m = st.slider("Most Likely Time (m)", 1, 30, 8, help="Most probable — normal conditions")
            b = st.slider("Pessimistic Time (b)", 1, 50, 16, help="Worst-case — rare, many problems")
            time_unit = st.text_input("Time unit", value="days")

            if not (a <= m <= b):
                st.warning("⚠️ Estimates must satisfy a ≤ m ≤ b for PERT to be valid.")

        with col2:
            if a <= m <= b:
                te       = (a + 4*m + b) / 6
                variance = ((b - a) / 6) ** 2
                std_dev  = math.sqrt(variance)
                cv       = std_dev / te if te > 0 else 0
                range_w  = b - a

                st.markdown("#### Results")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(f"Expected Time Tₑ ({time_unit})", f"{te:.2f}")
                col_b.metric("Variance σ²",                      f"{variance:.3f}")
                col_c.metric(f"Std Dev σ ({time_unit})",         f"{std_dev:.3f}")

                st.markdown("#### Step-by-Step")
                st.latex(
                    rf"T_E = \frac{{{a} + 4({m}) + {b}}}{{6}} "
                    rf"= \frac{{{a + 4*m + b}}}{{6}} = {te:.2f} \text{{ {time_unit}}}"
                )
                st.latex(
                    rf"\sigma^2 = \left(\frac{{{b} - {a}}}{{6}}\right)^2 "
                    rf"= \left(\frac{{{b-a}}}{{6}}\right)^2 = {variance:.3f}"
                )
                st.latex(rf"\sigma = \sqrt{{{variance:.3f}}} = {std_dev:.3f}")

                # Interpretation
                if cv < 0.15:
                    uncert = "🟢 Low uncertainty (CV < 15%)"
                elif cv < 0.30:
                    uncert = "🟡 Moderate uncertainty"
                else:
                    uncert = "🔴 High uncertainty (CV > 30%)"

                st.info(
                    f"**Range:** {a} – {b} {time_unit} (width = {range_w} {time_unit})  \n"
                    f"**CV = σ/Tₑ = {cv:.1%}** — {uncert}  \n"
                    f"**68% likely:** {te-std_dev:.2f} – {te+std_dev:.2f} {time_unit}  \n"
                    f"**95% likely:** {te-2*std_dev:.2f} – {te+2*std_dev:.2f} {time_unit}"
                )

                # ── Plotly Triangular Approximation ──
                st.markdown("#### Estimate Distribution (Beta Approximation)")
                t_pts  = [a + (b-a)*i/200 for i in range(201)]
                # Triangular distribution as visual approximation
                tri_y  = []
                for t in t_pts:
                    if t <= m:
                        tri_y.append(2*(t-a)/((b-a)*(m-a)) if (m-a) > 0 else 0)
                    else:
                        tri_y.append(2*(b-t)/((b-a)*(b-m)) if (b-m) > 0 else 0)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Scatter(
                    x=t_pts, y=tri_y, fill="tozeroy",
                    fillcolor="rgba(52,152,219,0.2)",
                    line=dict(color="#3498db", width=2), name="Probability"))
                fig_dist.add_vline(x=te, line_dash="dash", line_color="#e74c3c",
                                    annotation_text=f"Tₑ = {te:.2f}",
                                    annotation_position="top right")
                fig_dist.add_vline(x=m, line_dash="dot", line_color="#2ecc71",
                                    annotation_text=f"m = {m}",
                                    annotation_position="top left")
                fig_dist.update_layout(
                    title=f"PERT Distribution  |  a={a}  m={m}  b={b}  {time_unit}",
                    xaxis_title=f"Duration ({time_unit})",
                    yaxis_title="Relative Probability",
                    template="plotly_white", height=320
                )
                st.plotly_chart(fig_dist, use_container_width=True)

        # ── Multi-Activity Path Builder ──
        st.markdown("---")
        st.markdown("### Critical Path Variance Builder")
        st.write(
            "Enter PERT estimates for each activity on a path to compute "
            "total expected duration and total variance."
        )

        num_cp = st.number_input("Number of CP Activities", 1, 10, 3, key="pert_cp_num_v2")

        cols_hdr = st.columns([0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 0.8])
        for h, c in zip(["Activity", "a", "m", "b", "Tₑ", "σ²", "σ"], cols_hdr):
            c.markdown(f"**{h}**")

        total_te  = 0.0
        total_var = 0.0
        path_acts = []

        for i in range(int(num_cp)):
            cols = st.columns([0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 0.8])
            with cols[0]: st.write(f"**{chr(65+i)}**")
            with cols[1]: a_i = st.number_input(f"a{i}", value=2+i, min_value=1, key=f"pert_a_{i}", label_visibility="collapsed")
            with cols[2]: m_i = st.number_input(f"m{i}", value=4+i, min_value=1, key=f"pert_m_{i}", label_visibility="collapsed")
            with cols[3]: b_i = st.number_input(f"b{i}", value=8+i*2, min_value=1, key=f"pert_b_{i}", label_visibility="collapsed")

            te_i  = (a_i + 4*m_i + b_i) / 6
            var_i = ((b_i - a_i) / 6) ** 2
            sd_i  = math.sqrt(var_i)
            total_te  += te_i
            total_var += var_i

            with cols[4]: st.write(f"{te_i:.2f}")
            with cols[5]: st.write(f"{var_i:.3f}")
            with cols[6]: st.write(f"{sd_i:.3f}")
            path_acts.append({
                "Activity": chr(65+i), "a": a_i, "m": m_i, "b": b_i,
                "Tₑ": round(te_i, 2), "σ²": round(var_i, 3),
                "σ":  round(sd_i, 3)
            })

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Path Duration",    f"{total_te:.2f}")
        col2.metric("Total Path Variance",    f"{total_var:.3f}")
        col3.metric("Path Std Dev",           f"{math.sqrt(total_var):.3f}")
        col4.metric("Activities on Path",     len(path_acts))

        # ── Variance contribution chart ──
        if path_acts and total_var > 0:
            st.markdown("#### Variance Contribution by Activity")
            labels_v = [a["Activity"] for a in path_acts]
            vars_v   = [a["σ²"] for a in path_acts]
            pcts_v   = [v / total_var * 100 for v in vars_v]

            fig_var = go.Figure(go.Bar(
                x=labels_v, y=vars_v,
                marker_color=["#e74c3c" if p == max(pcts_v) else "#3498db" for p in pcts_v],
                text=[f"σ²={v:.3f}<br>({p:.0f}%)" for v, p in zip(vars_v, pcts_v)],
                textposition="outside"
            ))
            fig_var.update_layout(
                title="Variance Contribution — Red = Most Uncertain Activity",
                xaxis_title="Activity", yaxis_title="Variance (σ²)",
                template="plotly_white", height=320
            )
            st.plotly_chart(fig_var, use_container_width=True)

            top_var_act = path_acts[pcts_v.index(max(pcts_v))]
            st.warning(
                f"⚠️ Activity **{top_var_act['Activity']}** contributes "
                f"**{max(pcts_v):.0f}%** of total path variance (σ²={top_var_act['σ²']:.3f}). "
                f"Focus risk management efforts here first."
            )

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Project Completion Probability Calculator")

        col1, col2 = st.columns(2)
        with col1:
            te_project  = st.number_input("Expected Project Duration Tₑ", value=38.0,
                                           step=0.5, key="pert_prob_te",
                                           help="Sum of Tₑ for all critical path activities")
            d_target    = st.number_input("Desired Completion Date D", value=40.0,
                                           step=0.5, key="pert_prob_d",
                                           help="Your target completion (same units as Tₑ)")
            sum_var     = st.number_input("Sum of CP Variances (Σσ²)", value=8.33,
                                           step=0.1, key="pert_prob_var",
                                           help="Sum of σ² for all critical path activities")
            t_unit_p    = st.text_input("Time unit", value="weeks", key="pert_unit_p")

        with col2:
            if sum_var > 0:
                z_score  = (d_target - te_project) / math.sqrt(sum_var)
                prob     = normal_cdf(z_score)
                sigma_p  = math.sqrt(sum_var)

                st.metric("Z-Score",               f"{z_score:.3f}")
                st.metric("P(Complete ≤ D)",        f"{prob:.4f}  ({prob:.1%})")
                st.metric("Path Std Dev",            f"{sigma_p:.3f} {t_unit_p}")

                st.latex(
                    rf"Z = \frac{{{d_target} - {te_project}}}{{\sqrt{{{sum_var}}}}} "
                    rf"= \frac{{{d_target - te_project:.2f}}}{{{sigma_p:.3f}}} "
                    rf"= {z_score:.3f}"
                )

                if prob < 0.3:
                    st.error(
                        f"🔴 Only **{prob:.1%}** probability of completing by {d_target} {t_unit_p}. "
                        f"Strongly consider crashing or scope reduction."
                    )
                elif prob < 0.5:
                    st.warning(
                        f"🟡 **{prob:.1%}** chance of meeting the deadline. "
                        f"Consider crashing critical activities."
                    )
                else:
                    st.success(
                        f"🟢 **{prob:.1%}** chance of completing by {d_target} {t_unit_p}."
                    )

        # ── Deadline Probability Curve ──
        if sum_var > 0:
            st.markdown("---")
            st.markdown("### Probability vs. Target Date Curve")
            d_min  = te_project - 3 * sigma_p
            d_max  = te_project + 3 * sigma_p
            d_pts  = [d_min + (d_max - d_min) * i / 200 for i in range(201)]
            p_pts  = [normal_cdf((d - te_project) / sigma_p) for d in d_pts]

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=d_pts, y=[p * 100 for p in p_pts],
                mode="lines", line=dict(color="#3498db", width=2.5),
                name="P(Complete ≤ D)"))

            # Confidence markers
            for conf, color, label in [(0.5, "gray", "50%"),
                                        (0.9, "#f39c12", "90%"),
                                        (0.95, "#e74c3c", "95%")]:
                d_conf = te_project + stats.norm.ppf(conf) * sigma_p
                fig_prob.add_vline(x=d_conf, line_dash="dot", line_color=color,
                                    annotation_text=f"{label}: {d_conf:.1f} {t_unit_p}",
                                    annotation_position="bottom right" if conf < 0.9 else "top left")

            fig_prob.add_vline(x=d_target, line_dash="dash", line_color="#2ecc71",
                                annotation_text=f"Your target: D={d_target} ({prob:.1%})",
                                annotation_position="top right")

            fig_prob.update_layout(
                title=f"Completion Probability Curve  |  Tₑ={te_project}  σ={sigma_p:.2f} {t_unit_p}",
                xaxis_title=f"Target Completion Date ({t_unit_p})",
                yaxis_title="P(Complete on time) %",
                template="plotly_white", height=380
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Required date table
            st.markdown("#### Required Date for Confidence Level")
            conf_rows = []
            for conf in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
                d_req = te_project + stats.norm.ppf(conf) * sigma_p
                conf_rows.append({
                    "Confidence Level": f"{conf:.0%}",
                    f"Required Date ({t_unit_p})": f"{d_req:.2f}",
                    f"Buffer vs. Tₑ ({t_unit_p})":  f"{d_req - te_project:+.2f}",
                    "Your Target?": "← current" if abs(d_req - d_target) < (sigma_p * 0.1) else ""
                })
            st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Expected Time & Variance (Easy)"):
            display_practice_problem(1, "Easy",
                "Activity data: **a = 5 days, m = 8 days, b = 17 days**  \n"
                "a) Calculate Tₑ  \n"
                "b) Calculate σ²  \n"
                "c) Calculate σ")

            show_h1 = st.checkbox("Show Hint", key="pert_h1")
            if show_h1:
                display_hint("Tₑ = (a + 4m + b) / 6.  σ² = ((b − a) / 6)².  σ = √σ²")

            col1, col2, col3 = st.columns(3)
            with col1: u_te  = st.number_input("Tₑ:", format="%.2f", key="pert_p1_te")
            with col2: u_var = st.number_input("σ²:", format="%.2f", key="pert_p1_var")
            with col3: u_sd  = st.number_input("σ:", format="%.2f", key="pert_p1_sd")

            if st.button("Check Answers", key="pert_p1_btn_v2"):
                te_c  = (5 + 4*8 + 17) / 6
                var_c = ((17-5)/6)**2
                sd_c  = math.sqrt(var_c)
                for u, c, lbl in [(u_te, te_c, "Tₑ"), (u_var, var_c, "σ²"), (u_sd, sd_c, "σ")]:
                    if check_answer(u, c, 0.01):  st.write(f"✅ {lbl} = {c:.2f} correct")
                    else:                          st.write(f"❌ {lbl}: Should be {c:.2f}")

            if st.button("Show Solution", key="pert_p1_sol"):
                te_c = (5+4*8+17)/6; var_c = ((17-5)/6)**2
                display_solution(f"""
                <strong>a) Tₑ</strong><br>
                Tₑ = (5 + 4×8 + 17) / 6 = (5 + 32 + 17) / 6 = 54 / 6 = <strong>{te_c:.2f} days</strong><br><br>
                <strong>b) σ²</strong><br>
                σ² = ((17 − 5) / 6)² = (12/6)² = 2² = <strong>{var_c:.2f}</strong><br><br>
                <strong>c) σ</strong><br>
                σ = √{var_c:.2f} = <strong>{math.sqrt(var_c):.2f} days</strong>
                """)

        with st.expander("🟢 Problem 2: Variance Calculation (Easy)"):
            display_practice_problem(2, "Easy",
                "Activity: a = 3 days, b = 15 days  \n"
                "a) Calculate σ²  \n"
                "b) If another activity has a = 1, b = 7, which has more uncertainty?")

            col1, col2 = st.columns(2)
            with col1: u_var1 = st.number_input("σ² (a=3, b=15):", format="%.2f", key="pert_p2_v1")
            with col2: u_var2 = st.number_input("σ² (a=1, b=7):",  format="%.2f", key="pert_p2_v2")
            u_more = st.selectbox("More uncertain:", ["Select...", "First (a=3,b=15)",
                                                       "Second (a=1,b=7)", "Equal"],
                                   key="pert_p2_more")

            if st.button("Check Answers", key="pert_p2_btn_v2"):
                var1_c = ((15-3)/6)**2; var2_c = ((7-1)/6)**2
                more_c = "First (a=3,b=15)" if var1_c > var2_c else "Second (a=1,b=7)"
                for u, c, lbl in [(u_var1, var1_c, "σ²(1)"), (u_var2, var2_c, "σ²(2)")]:
                    if check_answer(u, c, 0.01):  st.write(f"✅ {lbl} = {c:.2f} correct")
                    else:                          st.write(f"❌ {lbl}: Should be {c:.2f}")
                if u_more == more_c:  st.write(f"✅ More uncertain: {more_c} correct")
                elif u_more != "Select...":
                    st.write(f"❌ More uncertain: Should be {more_c} — larger range = larger variance")

            if st.button("Show Solution", key="pert_p2_sol"):
                v1=((15-3)/6)**2; v2=((7-1)/6)**2
                display_solution(f"""
                <strong>a) σ² = ((b − a) / 6)²</strong><br>
                First: ((15 − 3) / 6)² = (12/6)² = 2² = <strong>{v1:.2f}</strong><br>
                Second: ((7 − 1) / 6)² = (6/6)² = 1² = <strong>{v2:.2f}</strong><br><br>
                <strong>b) First activity (a=3, b=15) has more uncertainty</strong><br>
                σ²=4.00 > σ²=1.00. The wider the range (b−a), the greater the variance.
                """)

        with st.expander("🟡 Problem 3: Full Path Probability (Medium)"):
            display_practice_problem(3, "Medium",
                "Critical path activities:  \n\n"
                "| Activity | a | m | b |  \n"
                "|----------|---|---|---|  \n"
                "| A | 2 | 4 | 6 |  \n"
                "| B | 3 | 5 | 13 |  \n"
                "| C | 4 | 6 | 8 |  \n\n"
                "a) Expected project duration  \n"
                "b) Total path variance  \n"
                "c) P(complete ≤ 17 days)")

            col1, col2, col3 = st.columns(3)
            with col1: u_dur = st.number_input("Expected Duration:", format="%.2f", key="pert_p3_dur_v2")
            with col2: u_var = st.number_input("Total Variance:",    format="%.2f", key="pert_p3_var_v2")
            with col3: u_prb = st.number_input("P(≤17 days) %:",     format="%.1f", key="pert_p3_prb_v2")

            if st.button("Check Answers", key="pert_p3_btn_v2"):
                te_a=(2+16+6)/6; te_b=(3+20+13)/6; te_c=(4+24+8)/6
                tot_te=te_a+te_b+te_c
                va=((6-2)/6)**2; vb=((13-3)/6)**2; vc=((8-4)/6)**2
                tot_var=va+vb+vc
                z=(17-tot_te)/math.sqrt(tot_var); p=normal_cdf(z)*100
                for u,c,lbl in [(u_dur,tot_te,"Duration"),(u_var,tot_var,"Variance"),
                                 (u_prb,p,"Probability")]:
                    tol = 0.02 if lbl == "Probability" else 0.01
                    if check_answer(u, c, tol):
                        st.write(f"✅ {lbl}: {c:.2f} correct")
                    else:
                        st.write(f"❌ {lbl}: Should be {c:.2f}")

            if st.button("Show Complete Solution", key="pert_p3_sol_v2"):
                te_a=(2+16+6)/6; te_b=(3+20+13)/6; te_c=(4+24+8)/6
                tot=te_a+te_b+te_c
                va=((6-2)/6)**2; vb=((13-3)/6)**2; vc=((8-4)/6)**2
                tv=va+vb+vc; z=(17-tot)/math.sqrt(tv); p=normal_cdf(z)*100
                display_solution(f"""
                <strong>Step 1: Expected Times</strong><br>
                Tₑ(A) = (2+4×4+6)/6 = {te_a:.2f}  |  Tₑ(B) = (3+4×5+13)/6 = {te_b:.2f}  |  Tₑ(C) = (4+4×6+8)/6 = {te_c:.2f}<br>
                <strong>Total Duration = {tot:.2f} days</strong><br><br>
                <strong>Step 2: Variances</strong><br>
                σ²(A) = ((6−2)/6)² = {va:.3f}  |  σ²(B) = ((13−3)/6)² = {vb:.3f}  |  σ²(C) = ((8−4)/6)² = {vc:.3f}<br>
                <strong>Total Variance = {tv:.3f}</strong><br><br>
                <strong>Step 3: Z-Score</strong><br>
                Z = (17 − {tot:.2f}) / √{tv:.3f} = {17-tot:.2f} / {math.sqrt(tv):.3f} = <strong>{z:.3f}</strong><br><br>
                <strong>Step 4: Probability</strong><br>
                P(Z ≤ {z:.3f}) = <strong>{p:.1f}%</strong>
                """)

        with st.expander("🟡 Problem 4: Finding Required Date for a Confidence Level (Medium)"):
            display_practice_problem(4, "Medium",
                "A project has Tₑ = 25 weeks and Σσ² = 9.0 on the critical path.  \n"
                "a) What is the project standard deviation?  \n"
                "b) What deadline provides a 90% chance of on-time completion?  \n"
                "c) What is P(complete ≤ 24 weeks)?")

            col1, col2, col3 = st.columns(3)
            with col1: u_sd4 = st.number_input("σ (weeks):", format="%.2f", key="pert_p4_sd")
            with col2: u_d90 = st.number_input("D for 90% confidence:", format="%.2f", key="pert_p4_d90")
            with col3: u_p24 = st.number_input("P(≤24 weeks) %:", format="%.1f", key="pert_p4_p24")

            if st.button("Check Answers", key="pert_p4_btn"):
                sd_c  = math.sqrt(9); d90_c = 25 + 1.28*sd_c
                p24_c = normal_cdf((24-25)/sd_c)*100
                for u, c, lbl in [(u_sd4, sd_c, "σ"), (u_d90, d90_c, "D (90%)"),
                                   (u_p24, p24_c, "P(≤24)")]:
                    if check_answer(u, c, 0.05):
                        st.write(f"✅ {lbl}: {c:.2f} correct")
                    else:
                        st.write(f"❌ {lbl}: Should be {c:.2f}")

            if st.button("Show Solution", key="pert_p4_sol"):
                sd_c=math.sqrt(9); d90_c=25+1.28*sd_c; p24_c=normal_cdf((24-25)/sd_c)*100
                display_solution(f"""
                <strong>a) σ = √Σσ²</strong><br>
                σ = √9.0 = <strong>{sd_c:.2f} weeks</strong><br><br>
                <strong>b) D for 90% confidence — use Z = 1.28</strong><br>
                D = Tₑ + Z × σ = 25 + 1.28 × {sd_c:.2f} = <strong>{d90_c:.2f} weeks</strong><br><br>
                <strong>c) P(complete ≤ 24 weeks)</strong><br>
                Z = (24 − 25) / {sd_c:.2f} = {(24-25)/sd_c:.3f}<br>
                P = <strong>{p24_c:.1f}%</strong>  — only a {p24_c:.0f}% chance of finishing in 24 weeks.
                """)

        with st.expander("🔴 Problem 5: Multiple Critical Paths (Hard)"):
            display_practice_problem(5, "Hard",
                "After CPM analysis, two paths of equal duration (20 days) emerge:  \n"
                "- **Path 1 (A–B–C):** Σσ² = 4.0  \n"
                "- **Path 2 (D–E–F):** Σσ² = 9.0  \n\n"
                "a) Which path should be used for probability calculations, and why?  \n"
                "b) P(complete ≤ 18 days)?  \n"
                "c) What deadline gives 80% confidence?  \n"
                "d) If management requires 95% confidence, what buffer must they add to Tₑ?")

            col1, col2, col3 = st.columns(3)
            with col1: u_p5_p18 = st.number_input("P(≤18 days) %:", format="%.1f", key="pert_p5_p18")
            with col2: u_p5_d80 = st.number_input("D for 80%:", format="%.2f", key="pert_p5_d80")
            with col3: u_p5_buf = st.number_input("Buffer for 95% (days):", format="%.2f", key="pert_p5_buf")

            if st.button("Check Answers", key="pert_p5_btn"):
                sd_p2 = math.sqrt(9); z18 = (18-20)/sd_p2
                p18_c = normal_cdf(z18)*100
                d80_c = 20 + stats.norm.ppf(0.80)*sd_p2
                buf95 = stats.norm.ppf(0.95)*sd_p2
                for u, c, lbl in [(u_p5_p18, p18_c, "P(≤18)"),
                                   (u_p5_d80, d80_c, "D(80%)"),
                                   (u_p5_buf, buf95, "Buffer(95%)")]:
                    if check_answer(u, c, 0.05):
                        st.write(f"✅ {lbl}: {c:.2f} correct")
                    else:
                        st.write(f"❌ {lbl}: Should be {c:.2f}")

            if st.button("Show Complete Solution", key="pert_p5_sol"):
                sd2=math.sqrt(9); z18=(18-20)/sd2
                p18=normal_cdf(z18)*100; d80=20+stats.norm.ppf(0.80)*sd2
                buf95=stats.norm.ppf(0.95)*sd2
                display_solution(f"""
                <strong>a) Use Path 2 (Σσ² = 9.0)</strong><br>
                When paths have equal duration, use the <em>larger variance</em> path —
                this is the conservative approach and gives the lower probability.<br><br>
                <strong>b) P(complete ≤ 18 days)</strong><br>
                σ = √9.0 = {sd2:.2f} days<br>
                Z = (18 − 20) / {sd2:.2f} = {z18:.3f}<br>
                P = <strong>{p18:.1f}%</strong><br><br>
                <strong>c) D for 80% confidence</strong><br>
                D = 20 + Z₀.₈₀ × {sd2:.2f} = 20 + 0.842 × {sd2:.2f} = <strong>{d80:.2f} days</strong><br><br>
                <strong>d) Buffer for 95% confidence</strong><br>
                Buffer = Z₀.₉₅ × σ = 1.645 × {sd2:.2f} = <strong>{buf95:.2f} days</strong><br>
                Required date = 20 + {buf95:.2f} = {20+buf95:.2f} days
                """)

        with st.expander("🔴 Problem 6: Full PERT Network (Hard)"):
            display_practice_problem(6, "Hard",
                "A project has five activities with two paths:  \n\n"
                "**Path 1 (A–B–C):**  \n"
                "| A: a=2, m=4, b=6 | B: a=1, m=3, b=5 | C: a=3, m=5, b=7 |  \n\n"
                "**Path 2 (D–E):**  \n"
                "| D: a=4, m=8, b=18 | E: a=1, m=2, b=3 |  \n\n"
                "a) Which path is critical (longest Tₑ)?  \n"
                "b) P(complete ≤ 16 weeks)?  \n"
                "c) What date gives 95% confidence?")

            if st.button("Show Complete Solution", key="pert_p6_sol"):
                teA=(2+16+6)/6; teB=(1+12+5)/6; teC=(3+20+7)/6
                path1_te = teA+teB+teC
                teD=(4+32+18)/6; teE=(1+8+3)/6
                path2_te = teD+teE

                vA=((6-2)/6)**2; vB=((5-1)/6)**2; vC=((7-3)/6)**2; path1_var=vA+vB+vC
                vD=((18-4)/6)**2; vE=((3-1)/6)**2; path2_var=vD+vE

                cp = 1 if path1_te >= path2_te else 2
                cp_te  = max(path1_te, path2_te)
                cp_var = path1_var if cp == 1 else path2_var
                # If equal, use larger variance
                if abs(path1_te - path2_te) < 0.01:
                    cp_var = max(path1_var, path2_var)

                z16   = (16 - cp_te) / math.sqrt(cp_var)
                p16   = normal_cdf(z16) * 100
                d95   = cp_te + stats.norm.ppf(0.95) * math.sqrt(cp_var)

                display_solution(f"""
                <strong>a) Path Expected Durations</strong><br>
                Path 1 (A+B+C): Tₑ(A)={teA:.2f} + Tₑ(B)={teB:.2f} + Tₑ(C)={teC:.2f} = <strong>{path1_te:.2f} weeks</strong><br>
                Path 2 (D+E):   Tₑ(D)={teD:.2f} + Tₑ(E)={teE:.2f} = <strong>{path2_te:.2f} weeks</strong><br>
                <strong>Critical Path: Path {cp} ({cp_te:.2f} weeks)</strong><br><br>
                <strong>Path Variances</strong><br>
                Path 1: σ²(A)={vA:.3f} + σ²(B)={vB:.3f} + σ²(C)={vC:.3f} = <strong>{path1_var:.3f}</strong><br>
                Path 2: σ²(D)={vD:.3f} + σ²(E)={vE:.3f} = <strong>{path2_var:.3f}</strong><br>
                Using Path {cp} variance = <strong>{cp_var:.3f}</strong><br><br>
                <strong>b) P(complete ≤ 16)</strong><br>
                Z = (16 − {cp_te:.2f}) / √{cp_var:.3f} = {z16:.3f}<br>
                P = <strong>{p16:.1f}%</strong><br><br>
                <strong>c) 95% Confidence Date</strong><br>
                D = {cp_te:.2f} + 1.645 × {math.sqrt(cp_var):.3f} = <strong>{d95:.2f} weeks</strong>
                """)

# ============================================================
# MODULE 3: PROJECT CRASHING (Chapter 4) - ENHANCED V5.0
# ============================================================
def module_crashing():
    display_header("⚡", "Chapter 4", "Project Crashing",
                   "Time-cost trade-off: compress schedules by adding resources strategically")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "🔬 Simulator", "📈 Cost Curves", "🎓 Practice"])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Project Crashing Theory")
        st.write(
            "**Project crashing** is the deliberate acceleration of one or more activities "
            "by adding resources — overtime labor, additional equipment, or expedited materials. "
            "It is a time-cost trade-off: every day saved has a price, and the goal is to "
            "minimize total project cost while meeting the target deadline."
        )

        display_citation(
            "Crashing involves shortening the overall project duration by reducing the time "
            "of one or more of the critical path activities, usually by adding resources. "
            "The decision to crash should consider both the direct costs of acceleration "
            "and the indirect savings from finishing sooner.",
            "Jacobs & Chase (2024, p. 101)"
        )

        display_formula_card("Crash Cost per Time Unit",
            r"\text{Crash Cost/Day} = \frac{\text{Crash Cost} - \text{Normal Cost}}"
            r"{\text{Normal Time} - \text{Crash Time}}")

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Total Direct Cost",
                r"TC_{direct}(d) = \text{Normal Cost} + \sum(\text{Cost/Day} \times \text{Days Crashed})")
        with col2:
            display_formula_card("Total Project Cost",
                r"TC_{total}(d) = TC_{direct}(d) + TC_{indirect}(d)")

        st.markdown("### The Two Cost Components")
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("📈", "Direct Costs",
                                  "Rise as you crash — overtime, extra crew, premium materials. "
                                  "Each day crashed adds the crash cost/day for that activity.")
        with col2:
            display_concept_card("📉", "Indirect Costs",
                                  "Fall as duration decreases — overhead, project management fees, "
                                  "daily penalties, and opportunity costs of delayed revenue.")

        display_textbook_content(
            "Optimal Crash Duration",
            """The optimal project duration minimizes total project cost (direct + indirect).
            This occurs at the point where the marginal cost of crashing one more day
            equals the marginal indirect cost savings from that day.
            Beyond that point, crashing becomes counterproductive — you spend more on
            direct costs than you save in indirect costs."""
        )

        st.markdown("### Step-by-Step Crashing Procedure")
        steps_df = pd.DataFrame({
            "Step": [1, 2, 3, 4, 5, 6],
            "Action": [
                "Identify the current critical path(s)",
                "Calculate crash cost/day for ALL critical activities",
                "Select the cheapest activity to crash on the critical path",
                "Crash it by 1 day (or until it hits its limit or a new CP emerges)",
                "Check if a new critical path has formed (multiple CPs may now exist)",
                "Repeat until target duration is reached or all activities are at crash limit"
            ],
            "Key Rule": [
                "Only crashing CP activities reduces project duration",
                "Sort ascending — cheapest first",
                "Minimum cost/day among all critical activities",
                "Cannot crash below crash time limit",
                "When two CPs exist, must crash BOTH simultaneously",
                "Stop when crash cost/day > indirect savings/day"
            ]
        })
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

        display_key_insight(
            "Multiple Critical Paths",
            "When crashing creates two or more critical paths of equal length, you must "
            "crash one activity on EACH path simultaneously. The cost to shorten the project "
            "by 1 day then equals the sum of cheapest crash cost/day from each path. "
            "This is where crashing can become uneconomical quickly."
        )

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Crash Cost Calculator & Priority Ranker")

        col1, col2 = st.columns([2, 1])
        with col1:
            num_activities = st.number_input("Number of Activities", 2, 10, 4,
                                              help="Enter data for all activities on the critical path")
        with col2:
            indirect_per_day = st.number_input("Indirect Cost per Day ($)",
                                                value=500, min_value=0, step=50,
                                                help="Overhead, penalties, or opportunity cost per day")

        activities = []
        st.markdown("#### Activity Data Entry")
        cols_hdr = st.columns([0.8, 1.2, 1.2, 1.2, 1.2, 1, 1.2])
        for h, c in zip(["Activity", "Normal Time", "Crash Time",
                          "Normal Cost ($)", "Crash Cost ($)",
                          "Max Crash", "Cost/Day"], cols_hdr):
            c.markdown(f"**{h}**")

        for i in range(int(num_activities)):
            cols = st.columns([0.8, 1.2, 1.2, 1.2, 1.2, 1, 1.2])
            with cols[0]:
                st.write(f"**{chr(65+i)}**")
            with cols[1]:
                nt = st.number_input(f"NT_{i}", value=5+i, min_value=1,
                                      key=f"crash_nt_{i}", label_visibility="collapsed")
            with cols[2]:
                ct = st.number_input(f"CT_{i}", value=max(1, 3+i//2), min_value=1,
                                      key=f"crash_ct_{i}", label_visibility="collapsed")
            with cols[3]:
                nc = st.number_input(f"NC_{i}", value=1000+i*200,
                                      key=f"crash_nc_{i}", label_visibility="collapsed")
            with cols[4]:
                cc = st.number_input(f"CC_{i}", value=1800+i*400,
                                      key=f"crash_cc_{i}", label_visibility="collapsed")

            max_crash = max(0, nt - ct)
            cpd = (cc - nc) / max_crash if max_crash > 0 else None
            profitable = (cpd is not None and cpd < indirect_per_day)

            with cols[5]:
                st.write(f"{max_crash} days")
            with cols[6]:
                if cpd is not None:
                    flag = "✅" if profitable else "❌"
                    st.write(f"{flag} ${cpd:,.0f}")
                else:
                    st.write("—")

            activities.append({
                "Activity":       chr(65+i),
                "Normal Time":    nt,
                "Crash Time":     ct,
                "Normal Cost":    nc,
                "Crash Cost":     cc,
                "Max Crash Days": max_crash,
                "Cost/Day":       cpd,
                "Profitable?":    "✅ Yes" if profitable else ("❌ No" if cpd else "—")
            })

        # ── Summary Metrics ──
        st.markdown("---")
        total_normal_time  = sum(a["Normal Time"]  for a in activities)
        total_crash_time   = sum(a["Crash Time"]   for a in activities)
        total_normal_cost  = sum(a["Normal Cost"]  for a in activities)
        total_crash_cost   = sum(a["Crash Cost"]   for a in activities)
        max_crashable_days = sum(a["Max Crash Days"] for a in activities)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Normal Duration",    f"{total_normal_time} days")
        col2.metric("Min Crash Duration", f"{total_crash_time} days")
        col3.metric("Normal Cost",        f"${total_normal_cost:,}")
        col4.metric("Full Crash Cost",    f"${total_crash_cost:,}")
        col5.metric("Total Crashable",    f"{max_crashable_days} days")

        # ── Priority Table ──
        crashable = [a for a in activities if a["Cost/Day"] is not None and a["Max Crash Days"] > 0]
        if crashable:
            crashable_sorted = sorted(crashable, key=lambda x: x["Cost/Day"])
            st.markdown("#### Crashing Priority Order (cheapest first)")
            priority_df = pd.DataFrame([{
                "Priority": i+1,
                "Activity": a["Activity"],
                "Cost/Day":  f"${a['Cost/Day']:,.0f}",
                "Max Days":  a["Max Crash Days"],
                "Max Savings": f"${a['Max Crash Days'] * indirect_per_day:,}",
                "Max Crash Cost": f"${a['Max Crash Days'] * a['Cost/Day']:,.0f}",
                "Net if Fully Crashed": f"${a['Max Crash Days'] * (indirect_per_day - a['Cost/Day']):+,.0f}",
                "Worth Crashing?": a["Profitable?"]
            } for i, a in enumerate(crashable_sorted)])
            st.dataframe(priority_df, use_container_width=True, hide_index=True)

            # ── Optimal Crash Recommendation ──
            profitable_activities = [a for a in crashable_sorted
                                      if a["Cost/Day"] is not None and a["Cost/Day"] < indirect_per_day]
            if profitable_activities:
                total_profit_days = sum(a["Max Crash Days"] for a in profitable_activities)
                total_crash_spend = sum(a["Max Crash Days"] * a["Cost/Day"]
                                        for a in profitable_activities)
                total_indirect_save = total_profit_days * indirect_per_day
                net_benefit = total_indirect_save - total_crash_spend

                st.success(
                    f"💡 **Optimal Strategy:** Crash {', '.join(a['Activity'] for a in profitable_activities)} "
                    f"fully ({total_profit_days} days total).  \n"
                    f"Crash cost: **${total_crash_spend:,.0f}** | "
                    f"Indirect savings: **${total_indirect_save:,.0f}** | "
                    f"Net benefit: **${net_benefit:,.0f}**"
                )
            else:
                st.error("❌ No activities are worth crashing — crash cost/day exceeds indirect savings/day for all activities.")

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Total Cost Curves")
        st.write(
            "The optimal project duration is where total cost (direct + indirect) is minimized. "
            "This chart shows how costs change at each possible project duration."
        )

        if not activities:
            st.info("Enter activity data in the Simulator tab first.")
        else:
            # Build cumulative crash schedule
            # Assume all activities are on critical path; crash cheapest first
            crashable_s = sorted(
                [a for a in activities if a["Cost/Day"] is not None and a["Max Crash Days"] > 0],
                key=lambda x: x["Cost/Day"]
            )
            base_duration    = total_normal_time
            base_direct_cost = total_normal_cost

            durations     = [base_duration]
            direct_costs  = [base_direct_cost]
            indirect_costs= [base_duration * indirect_per_day]
            total_costs   = [base_direct_cost + base_duration * indirect_per_day]
            labels        = ["Normal"]

            current_duration    = base_duration
            current_direct_cost = base_direct_cost

            for a in crashable_s:
                for day in range(1, int(a["Max Crash Days"]) + 1):
                    current_duration    -= 1
                    current_direct_cost += a["Cost/Day"]
                    indirect = current_duration * indirect_per_day
                    total    = current_direct_cost + indirect

                    durations.append(current_duration)
                    direct_costs.append(current_direct_cost)
                    indirect_costs.append(indirect)
                    total_costs.append(total)
                    labels.append(f"Crash {a['Activity']} day {day}")

            opt_idx      = total_costs.index(min(total_costs))
            opt_duration = durations[opt_idx]
            opt_cost     = total_costs[opt_idx]

            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=durations, y=direct_costs, mode="lines+markers",
                name="Direct Costs", line=dict(color="#e74c3c", width=2)))
            fig_cost.add_trace(go.Scatter(
                x=durations, y=indirect_costs, mode="lines+markers",
                name="Indirect Costs", line=dict(color="#3498db", width=2, dash="dash")))
            fig_cost.add_trace(go.Scatter(
                x=durations, y=total_costs, mode="lines+markers",
                name="Total Cost", line=dict(color="#2ecc71", width=3)))
            fig_cost.add_vline(
                x=opt_duration, line_dash="dot", line_color="purple",
                annotation_text=f"Optimal: {opt_duration} days  (${opt_cost:,.0f})",
                annotation_position="top left")

            fig_cost.update_layout(
                title="Project Cost Curves — Optimal Crashing Point",
                xaxis_title="Project Duration (days)",
                yaxis_title="Cost ($)",
                template="plotly_white",
                height=420,
                legend=dict(orientation="h", y=1.02),
                xaxis=dict(autorange="reversed")  # Shorter duration = more crashing
            )
            st.plotly_chart(fig_cost, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Normal Duration",  f"{base_duration} days")
            col2.metric("Optimal Duration", f"{opt_duration} days  (save {base_duration-opt_duration} days)")
            col3.metric("Min Total Cost",   f"${opt_cost:,.0f}  (save ${total_costs[0]-opt_cost:,.0f})")

            # ── Crash Schedule Table ──
            st.markdown("#### Crash Schedule Detail")
            sched_df = pd.DataFrame({
                "Action":          labels,
                "Duration (days)": durations,
                "Direct Cost ($)": [f"${c:,.0f}" for c in direct_costs],
                "Indirect Cost ($)":[f"${c:,.0f}" for c in indirect_costs],
                "Total Cost ($)":  [f"${c:,.0f}" for c in total_costs],
                "Optimal?":        ["🌟 OPTIMAL" if d == opt_duration else "" for d in durations]
            })
            st.dataframe(sched_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Crash Cost per Day (Easy)"):
            display_practice_problem(1, "Easy",
                "Activity X:  \n"
                "- Normal Time = 10 days, Normal Cost = $5,000  \n"
                "- Crash Time = 6 days, Crash Cost = $9,000  \n\n"
                "Calculate the crash cost per day.")

            show_h1 = st.checkbox("Show Hint", key="crash_h1")
            if show_h1:
                display_hint("Cost/Day = (Crash Cost − Normal Cost) ÷ (Normal Time − Crash Time)")

            u_cpd = st.number_input("Crash Cost/Day ($):", key="crash_p1_v2")
            if st.button("Check Answer", key="crash_p1_btn"):
                correct = (9000 - 5000) / (10 - 6)
                if check_answer(u_cpd, correct):
                    st.success(f"✅ Correct! ${correct:,.0f}/day")
                else:
                    display_solution(
                        f"Cost/Day = ($9,000 − $5,000) ÷ (10 − 6) = $4,000 ÷ 4 = "
                        f"<strong>${correct:,.0f}/day</strong>"
                    )

        with st.expander("🟡 Problem 2: Which Activity to Crash First? (Medium)"):
            display_practice_problem(2, "Medium",
                "A project's critical path is A → B → C (total: 16 days).  \n"
                "Indirect costs = $600/day.  \n\n"
                "| Activity | Normal | Crash | Crash Cost/Day |  \n"
                "|----------|--------|-------|----------------|  \n"
                "| A | 6 | 4 | $400 |  \n"
                "| B | 5 | 3 | $700 |  \n"
                "| C | 5 | 4 | $900 |  \n\n"
                "a) Which activity should be crashed first?  \n"
                "b) How many total days should be crashed, and why?  \n"
                "c) What is the net benefit of the optimal crashing plan?")

            show_h2 = st.checkbox("Show Hint", key="crash_h2")
            if show_h2:
                display_hint(
                    "Crash cheapest first. Stop when crash cost/day ≥ indirect savings/day ($600)."
                )

            col1, col2 = st.columns(2)
            with col1:
                u_first = st.selectbox("First activity to crash:", ["Select...", "A", "B", "C"],
                                        key="crash_p2_first")
                u_days  = st.number_input("Total days to crash:", min_value=0, key="crash_p2_days")
            with col2:
                u_net   = st.number_input("Net benefit ($):", key="crash_p2_net")

            if st.button("Check Answers", key="crash_p2_btn"):
                # A=$400 profitable (400<600), B=$700 profitable (700>600 no), C=$900 no
                # Crash A by 2 days, B by 0, C by 0 — only A is worth crashing
                net = 2 * (600 - 400)  # 2 days × net savings
                results = []
                if u_first == "A":     results.append("✅ Activity A — correct (lowest cost/day = $400)")
                else:                  results.append("❌ First crash: Should be Activity A ($400/day is cheapest)")
                if check_answer(u_days, 2):   results.append("✅ 2 days — correct")
                else:                          results.append("❌ Days: Should be 2 (A can be crashed 2 days; B and C cost more than $600/day)")
                if check_answer(u_net, net):   results.append(f"✅ Net benefit ${net:,} — correct")
                else:                          results.append(f"❌ Net benefit: Should be ${net:,}")
                for r in results: st.write(r)

            if st.button("Show Solution", key="crash_p2_sol"):
                display_solution(f"""
                <strong>Step 1: Compare crash cost/day vs indirect savings ($600/day)</strong><br>
                • Activity A: $400/day → $400 < $600 ✅ <em>Profitable</em><br>
                • Activity B: $700/day → $700 > $600 ❌ <em>Not profitable</em><br>
                • Activity C: $900/day → $900 > $600 ❌ <em>Not profitable</em><br><br>
                <strong>Step 2: Crash Activity A (cheapest) by 2 days</strong><br>
                (A can be crashed from 6 → 4 days, max 2 days)<br><br>
                <strong>Step 3: Net Benefit</strong><br>
                Indirect savings: 2 × $600 = $1,200<br>
                Crash cost:       2 × $400 = $800<br>
                Net benefit = $1,200 − $800 = <strong>$400</strong><br><br>
                Stop here — B and C are not worth crashing.
                """)

        with st.expander("🔴 Problem 3: Bonus/Penalty Decision (Hard)"):
            display_practice_problem(3, "Hard",
                "A project has critical path A → B → C (15 days total).  \n"
                "Client offers a **$2,000 bonus per day** finished early.  \n\n"
                "| Activity | Normal | Crash | Crash Cost/Day | Max Crash |  \n"
                "|----------|--------|-------|----------------|-----------|  \n"
                "| A | 5 | 3 | $800 | 2 days |  \n"
                "| B | 6 | 4 | $1,500 | 2 days |  \n"
                "| C | 4 | 3 | $2,500 | 1 day |  \n\n"
                "a) How many total days should you crash?  \n"
                "b) What is the final project duration?  \n"
                "c) What is the total net benefit?")

            col1, col2, col3 = st.columns(3)
            with col1:
                u_days_h = st.number_input("Days to crash:", min_value=0, key="crash_p3_d")
            with col2:
                u_dur_h  = st.number_input("Final duration (days):", min_value=0, key="crash_p3_dur")
            with col3:
                u_net_h  = st.number_input("Net benefit ($):", key="crash_p3_net")

            if st.button("Check Answers", key="crash_p3_btn"):
                # A $800 < $2000 → crash 2 days; B $1500 < $2000 → crash 2 days; C $2500 > $2000 → no
                days = 4; dur = 15-4; net = (2*2000-2*800) + (2*2000-2*1500)
                results = []
                if check_answer(u_days_h, days):  results.append(f"✅ {days} days — correct")
                else:                              results.append(f"❌ Days: Should be {days}")
                if check_answer(u_dur_h, dur):    results.append(f"✅ {dur} days — correct")
                else:                              results.append(f"❌ Duration: Should be {dur} days")
                if check_answer(u_net_h, net):    results.append(f"✅ Net benefit ${net:,} — correct")
                else:                              results.append(f"❌ Net benefit: Should be ${net:,}")
                for r in results: st.write(r)

            if st.button("Show Complete Solution", key="crash_p3_sol"):
                display_solution("""
                <strong>Step 1: Is each activity worth crashing? (Bonus = $2,000/day)</strong><br>
                • Activity A: $800/day < $2,000 ✅ → Crash 2 days<br>
                • Activity B: $1,500/day < $2,000 ✅ → Crash 2 days<br>
                • Activity C: $2,500/day > $2,000 ❌ → Do NOT crash<br><br>
                <strong>Step 2: Calculate net benefit per activity</strong><br>
                A: 2 days × ($2,000 − $800) = 2 × $1,200 = <strong>$2,400</strong><br>
                B: 2 days × ($2,000 − $1,500) = 2 × $500 = <strong>$1,000</strong><br><br>
                <strong>Step 3: Final Summary</strong><br>
                • Total days crashed: 4 (A: 2, B: 2)<br>
                • Final duration: 15 − 4 = <strong>11 days</strong><br>
                • Total crash cost: (2×$800) + (2×$1,500) = $1,600 + $3,000 = $4,600<br>
                • Total bonus: 4 × $2,000 = $8,000<br>
                • <strong>Net benefit: $8,000 − $4,600 = $3,400</strong>
                """)

        with st.expander("🔴 Problem 4: Multiple Critical Paths (Hard)"):
            display_practice_problem(4, "Hard",
                "After crashing Activity A by 1 day, two equal critical paths emerge (14 days each):  \n"
                "- **Path 1:** A–B–C  \n"
                "- **Path 2:** D–E–F  \n\n"
                "Indirect cost = $500/day.  \n\n"
                "| Activity | Path | Crash Cost/Day | Max Crash Left |  \n"
                "|----------|------|----------------|----------------|  \n"
                "| B | 1 | $300 | 2 |  \n"
                "| C | 1 | $600 | 1 |  \n"
                "| E | 2 | $400 | 2 |  \n"
                "| F | 2 | $700 | 1 |  \n\n"
                "To shorten the project by 1 day, what is the minimum cost combination,  \n"
                "and is it worth it?")

            if st.button("Show Complete Solution", key="crash_p4_sol"):
                display_solution("""
                <strong>Rule:</strong> When two CPs exist, you must shorten BOTH by 1 day simultaneously.<br><br>
                <strong>Options to shorten both paths by 1 day:</strong><br>
                • B + E: $300 + $400 = <strong>$700</strong> ← Cheapest<br>
                • B + F: $300 + $700 = $1,000<br>
                • C + E: $600 + $400 = $1,000<br>
                • C + F: $600 + $700 = $1,300<br><br>
                <strong>Best combination: Crash B + E</strong><br>
                Cost = $700 | Indirect savings = $500<br>
                Net = $500 − $700 = <strong>−$200 (loss!)</strong><br><br>
                <strong>Decision: Do NOT crash further.</strong><br>
                Even the cheapest combination ($700) exceeds the indirect savings ($500/day).
                The optimal project duration has already been reached.
                """)


# ============================================================
# MODULE 4: BREAK-EVEN ANALYSIS (Chapter 5) - ENHANCED V5.0
# ============================================================
def module_breakeven():
    display_header("📈", "Chapter 5", "Break-Even Analysis",
                   "Cost-Volume-Profit (CVP) analysis for capacity and sourcing decisions")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📚 Theory", "🔬 Calculator", "📊 Sensitivity", "⚖️ Comparison", "🎓 Practice"])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Cost-Volume-Profit Analysis")
        st.write(
            "**Break-even analysis** finds the volume at which total revenue exactly equals "
            "total cost — the point of zero profit. Below the BEP the firm loses money; "
            "above it the firm is profitable. It is one of the most widely used tools for "
            "evaluating capacity alternatives, pricing decisions, and make-vs-buy choices."
        )

        display_citation(
            "Break-even analysis is a standard approach to choosing among capacity alternatives. "
            "The object of break-even analysis is to find the point in dollars and units at which "
            "cost equals revenue. It requires estimation of fixed costs, variable costs, and revenue.",
            "Jacobs & Chase (2024, p. 155)"
        )

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("BEP (Units)",
                r"BEP_{units} = \frac{F}{P - V}")
            display_formula_card("BEP (Revenue)",
                r"BEP_{\$} = \frac{F}{1 - \dfrac{V}{P}}")
            display_formula_card("Target Profit Volume",
                r"Q_{target} = \frac{F + \pi}{P - V}")
        with col2:
            display_formula_card("Contribution Margin (CM)",
                r"CM = P - V")
            display_formula_card("CM Ratio",
                r"CM\% = \frac{P - V}{P}")
            display_formula_card("Indifference Point (2 Alternatives)",
                r"Q^* = \frac{F_2 - F_1}{V_1 - V_2}")

        st.markdown("#### Variable Definitions")
        var_df = pd.DataFrame({
            "Symbol": ["F", "P", "V", "Q", "π", "CM", "BEP"],
            "Meaning": ["Total Fixed Costs", "Selling Price per unit",
                        "Variable Cost per unit", "Volume (units)",
                        "Target Profit", "Contribution Margin = P − V",
                        "Break-Even Point"],
            "Example": ["$50,000 rent + salaries", "$100 per unit",
                        "$60 materials + labor", "1,250 units",
                        "$20,000 annual target", "$40 per unit",
                        "1,250 units or $125,000"]
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

        display_textbook_content(
            "Indifference Point — Choosing Between Alternatives",
            """When comparing two process alternatives with different fixed/variable cost
            structures (e.g., manual vs. automated), the indifference point is the volume
            at which both alternatives cost the same. Below this volume, choose the option
            with lower fixed costs. Above it, choose the option with lower variable costs."""
        )

        display_key_insight(
            "CM Sensitivity Rule",
            "BEP is most sensitive to changes in the contribution margin (P − V). "
            "A 10% increase in price has a much larger BEP reduction than a 10% "
            "reduction in fixed costs, because price affects both CM and revenue simultaneously. "
            "This is why pricing decisions are strategically critical."
        )

        st.markdown("#### Assumptions & Limitations")
        st.write("""
        - Revenue and costs are **linear** within the relevant range
        - Fixed costs remain **constant** (no step-fixed costs)
        - All units produced are **sold** (no inventory buildup)
        - **Single product** (or constant product mix)
        - Price and variable cost per unit are **constant** regardless of volume
        """)

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Break-Even Calculator")

        col1, col2 = st.columns(2)
        with col1:
            fixed_cost    = st.slider("Fixed Costs ($)", 5000, 300000, 50000, 5000)
            price         = st.slider("Price per Unit ($)", 10, 500, 100, 5)
            variable_cost = st.slider("Variable Cost per Unit ($)", 5, 490, 60, 5)

        with col2:
            if price > variable_cost:
                bep_units   = fixed_cost / (price - variable_cost)
                bep_revenue = bep_units * price
                cm          = price - variable_cost
                cm_ratio    = cm / price

                st.metric("Break-Even Units",    f"{bep_units:,.0f}")
                st.metric("Break-Even Revenue",  f"${bep_revenue:,.0f}")
                st.metric("Contribution Margin", f"${cm:.2f}/unit")
                st.metric("CM Ratio",            f"{cm_ratio:.1%}")

                st.latex(
                    rf"BEP = \frac{{{fixed_cost:,}}}{{{price} - {variable_cost}}} "
                    rf"= \frac{{{fixed_cost:,}}}{{{cm}}} = {bep_units:,.0f} \text{{ units}}"
                )
            else:
                st.error("⚠️ Price must exceed Variable Cost — contribution margin cannot be ≤ 0.")

        # ── Target Profit ──
        st.markdown("---")
        st.markdown("### Target Profit Volume")
        target_profit = st.number_input("Target Profit ($)", value=20000, step=5000,
                                         min_value=0, key="be_tp")
        if price > variable_cost:
            target_units   = (fixed_cost + target_profit) / (price - variable_cost)
            target_revenue = target_units * price

            col1, col2, col3 = st.columns(3)
            col1.metric("Units Required",    f"{target_units:,.0f}")
            col2.metric("Revenue Required",  f"${target_revenue:,.0f}")
            col3.metric("Above BEP by",      f"{target_units - bep_units:,.0f} units")

            st.latex(
                rf"Q = \frac{{{fixed_cost:,} + {target_profit:,}}}{{{cm}}} = {target_units:,.0f}"
            )

        # ── Plotly Break-Even Chart ──
        if price > variable_cost:
            st.markdown("---")
            st.markdown("### Break-Even Chart")
            max_q   = int(bep_units * 2.5)
            q_range = list(range(0, max_q + 1, max(1, max_q // 100)))

            total_cost  = [fixed_cost + variable_cost * q for q in q_range]
            total_rev   = [price * q for q in q_range]
            fixed_line  = [fixed_cost for _ in q_range]

            fig_be = go.Figure()
            fig_be.add_trace(go.Scatter(
                x=q_range, y=fixed_line, mode="lines", name="Fixed Costs",
                line=dict(color="#95a5a6", width=1.5, dash="dot")))
            fig_be.add_trace(go.Scatter(
                x=q_range, y=total_cost, mode="lines", name="Total Cost",
                line=dict(color="#e74c3c", width=2.5)))
            fig_be.add_trace(go.Scatter(
                x=q_range, y=total_rev, mode="lines", name="Total Revenue",
                line=dict(color="#2ecc71", width=2.5)))

            fig_be.add_vline(x=bep_units, line_dash="dash", line_color="#3498db",
                              annotation_text=f"BEP = {bep_units:,.0f} units",
                              annotation_position="top right")
            fig_be.add_annotation(
                x=bep_units, y=bep_revenue,
                text=f"BEP = ${bep_revenue:,.0f}",
                showarrow=True, arrowhead=2, bgcolor="white"
            )

            # Shade profit region
            profit_q = [q for q in q_range if q >= bep_units]
            profit_rev = [price * q for q in profit_q]
            profit_cost = [fixed_cost + variable_cost * q for q in profit_q]
            if profit_q:
                fig_be.add_trace(go.Scatter(
                    x=profit_q + profit_q[::-1],
                    y=profit_rev + profit_cost[::-1],
                    fill="toself", fillcolor="rgba(46,204,113,0.1)",
                    line=dict(width=0), name="Profit Zone", showlegend=True))

            fig_be.update_layout(
                title=f"Break-Even Chart  |  F=${fixed_cost:,}  P=${price}  V=${variable_cost}",
                xaxis_title="Volume (units)", yaxis_title="$ Amount",
                template="plotly_white", height=420,
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig_be, use_container_width=True)

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Sensitivity Analysis")
        st.write(
            "Sensitivity analysis tests how the BEP changes when key assumptions change. "
            "It answers the critical question: *which variable has the most impact on our break-even?*"
        )

        display_citation(
            "Some companies call this 'what if' analysis. Answering 'what if' questions can be "
            "useful for understanding how sensitive an analysis is to cost and profit assumptions.",
            "Jacobs & Chase (2024, p. 60)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            base_fc    = st.number_input("Base Fixed Cost ($)", value=50000, key="sens_fc")
        with col2:
            base_price = st.number_input("Base Price ($)", value=100, key="sens_p")
        with col3:
            base_vc    = st.number_input("Base Variable Cost ($)", value=60, key="sens_vc")

        if base_price > base_vc:
            base_bep = base_fc / (base_price - base_vc)
            base_cm  = base_price - base_vc

            scenarios = [
                ("📍 Base Case",          base_fc,        base_price,     base_vc),
                ("📈 +25% Fixed Costs",    base_fc*1.25,   base_price,     base_vc),
                ("📉 −25% Fixed Costs",    base_fc*0.75,   base_price,     base_vc),
                ("💰 +$10 Price",          base_fc,        base_price+10,  base_vc),
                ("💸 −$10 Price",          base_fc,        base_price-10,  base_vc),
                ("⬆️ +$5 Variable Cost",   base_fc,        base_price,     base_vc+5),
                ("⬇️ −$5 Variable Cost",   base_fc,        base_price,     base_vc-5),
            ]

            results = []
            for name, fc, p, vc in scenarios:
                if p > vc:
                    bep     = fc / (p - vc)
                    change  = ((bep - base_bep) / base_bep) * 100
                    results.append({
                        "Scenario":    name,
                        "Fixed Cost":  f"${fc:,.0f}",
                        "Price":       f"${p:.0f}",
                        "Var Cost":    f"${vc:.0f}",
                        "CM":          f"${p-vc:.0f}",
                        "BEP (units)": f"{bep:,.0f}",
                        "Δ vs Base":   f"{change:+.1f}%"
                    })

            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            # Tornado chart
            st.markdown("#### Sensitivity Tornado Chart")
            tornado_data = [(s["Scenario"], float(s["BEP (units)"].replace(",", "")))
                             for s in results]
            tornado_sorted = sorted(tornado_data, key=lambda x: abs(x[1] - base_bep))

            labels_t = [t[0] for t in tornado_sorted]
            beps_t   = [t[1] for t in tornado_sorted]
            diffs_t  = [b - base_bep for b in beps_t]
            colors_t = ["#e74c3c" if d > 0 else "#2ecc71" for d in diffs_t]

            fig_tornado = go.Figure(go.Bar(
                x=diffs_t, y=labels_t, orientation="h",
                marker_color=colors_t,
                text=[f"{d:+.0f} units" for d in diffs_t],
                textposition="outside"
            ))
            fig_tornado.add_vline(x=0, line_color="black", line_width=1)
            fig_tornado.update_layout(
                title=f"BEP Change vs. Base Case ({base_bep:,.0f} units)",
                xaxis_title="Change in BEP (units)",
                template="plotly_white", height=380,
                margin=dict(l=180)
            )
            st.plotly_chart(fig_tornado, use_container_width=True)

            display_key_insight(
                "Most Sensitive Variable",
                f"The longest bars show which assumptions matter most. "
                f"At base CM = ${base_cm}/unit, a $10 price drop raises BEP by "
                f"{(base_fc/(base_price-10-base_vc) - base_bep):,.0f} units — "
                f"more impactful than a 25% increase in fixed costs."
            )

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Scenario Comparison & Indifference Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Option A (e.g., Manual / Low-Tech)")
            fc_a  = st.number_input("Fixed Costs A ($)", value=20000, key="be_fc_a")
            p_a   = st.number_input("Price A ($)",       value=100,   key="be_p_a")
            vc_a  = st.number_input("Variable Cost A ($)", value=15,  key="be_vc_a")

        with col2:
            st.markdown("#### Option B (e.g., Automated / High-Tech)")
            fc_b  = st.number_input("Fixed Costs B ($)", value=80000, key="be_fc_b")
            p_b   = st.number_input("Price B ($)",       value=100,   key="be_p_b")
            vc_b  = st.number_input("Variable Cost B ($)", value=5,   key="be_vc_b")

        if p_a > vc_a and p_b > vc_b:
            bep_a = fc_a / (p_a - vc_a)
            bep_b = fc_b / (p_b - vc_b)

            col1, col2 = st.columns(2)
            col1.metric("BEP — Option A", f"{bep_a:,.0f} units")
            col2.metric("BEP — Option B", f"{bep_b:,.0f} units")

            if vc_a != vc_b:
                indiff = (fc_b - fc_a) / (vc_a - vc_b)
                if indiff > 0:
                    st.markdown("---")
                    better_low  = "A" if fc_a < fc_b else "B"
                    better_high = "B" if vc_b < vc_a else "A"

                    col1, col2 = st.columns(2)
                    col1.metric("Indifference Point", f"{indiff:,.0f} units")
                    col2.metric("Same Total Cost at", f"${(fc_a + vc_a*indiff):,.0f}")

                    st.latex(
                        rf"Q^* = \frac{{{fc_b:,} - {fc_a:,}}}{{{vc_a} - {vc_b}}} "
                        rf"= \frac{{{fc_b-fc_a:,}}}{{{vc_a-vc_b}}} = {indiff:,.0f}"
                    )
                    st.info(
                        f"📊 **Decision Rule:**  \n"
                        f"- **Below {indiff:,.0f} units** → Choose **Option {better_low}** (lower fixed costs)  \n"
                        f"- **Above {indiff:,.0f} units** → Choose **Option {better_high}** (lower variable costs)"
                    )

                    # ── Plotly Comparison Chart ──
                    q_max    = int(indiff * 2.2)
                    q_range  = list(range(0, q_max+1, max(1, q_max//150)))
                    tc_a_pts = [fc_a + vc_a * q for q in q_range]
                    tc_b_pts = [fc_b + vc_b * q for q in q_range]

                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(
                        x=q_range, y=tc_a_pts, mode="lines", name="Option A Total Cost",
                        line=dict(color="#3498db", width=2.5)))
                    fig_comp.add_trace(go.Scatter(
                        x=q_range, y=tc_b_pts, mode="lines", name="Option B Total Cost",
                        line=dict(color="#e74c3c", width=2.5)))
                    fig_comp.add_vline(
                        x=indiff, line_dash="dash", line_color="purple",
                        annotation_text=f"Q* = {indiff:,.0f} units",
                        annotation_position="top left")

                    # Shade preferred regions
                    left_q = [q for q in q_range if q <= indiff]
                    right_q = [q for q in q_range if q >= indiff]
                    preferred_low  = tc_a_pts if fc_a < fc_b else tc_b_pts
                    preferred_high = tc_b_pts if vc_b < vc_a else tc_a_pts
                    other_low  = tc_b_pts if fc_a < fc_b else tc_a_pts

                    fig_comp.update_layout(
                        title="Total Cost Comparison — Option A vs. Option B",
                        xaxis_title="Volume (units)", yaxis_title="Total Cost ($)",
                        template="plotly_white", height=400,
                        legend=dict(orientation="h", y=1.02)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # ── Volume Analysis Table ──
                    st.markdown("#### Cost Comparison at Key Volumes")
                    check_vols = [int(indiff * m) for m in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
                    vol_rows = []
                    for q in check_vols:
                        ca = fc_a + vc_a * q
                        cb = fc_b + vc_b * q
                        vol_rows.append({
                            "Volume": f"{q:,} units",
                            "Cost A": f"${ca:,.0f}",
                            "Cost B": f"${cb:,.0f}",
                            "Difference": f"${abs(ca-cb):,.0f}",
                            "Better Option": f"A (saves ${cb-ca:,.0f})" if ca < cb else
                                             f"B (saves ${ca-cb:,.0f})" if cb < ca else "Indifferent"
                        })
                    st.dataframe(pd.DataFrame(vol_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Basic BEP (Easy)"):
            display_practice_problem(1, "Easy",
                "F = $40,000 | P = $120/unit | V = $80/unit  \n"
                "Calculate the break-even point in units.")

            show_h1 = st.checkbox("Show Hint", key="be_h1")
            if show_h1:
                display_hint("BEP = F ÷ (P − V). First find the contribution margin.")

            u_bep = st.number_input("BEP (units):", key="be_p1_v2")
            if st.button("Check Answer", key="be_p1_btn"):
                correct = 40000 / (120 - 80)
                if check_answer(u_bep, correct):
                    st.success(f"✅ BEP = {correct:,.0f} units")
                else:
                    display_solution(
                        f"BEP = $40,000 ÷ ($120 − $80) = $40,000 ÷ $40 = <strong>{correct:,.0f} units</strong>"
                    )

        with st.expander("🟡 Problem 2: Target Profit (Medium)"):
            display_practice_problem(2, "Medium",
                "F = $60,000 | P = $50/unit | V = $30/unit | Target Profit = $20,000  \n"
                "a) How many units must be sold?  \n"
                "b) What revenue does that represent?")

            col1, col2 = st.columns(2)
            with col1:
                u_units = st.number_input("Units required:", key="be_p2_u")
            with col2:
                u_rev   = st.number_input("Revenue required ($):", key="be_p2_r")

            if st.button("Check Answers", key="be_p2_btn"):
                q_c = (60000 + 20000) / (50 - 30); rev_c = q_c * 50
                for u, c, lbl in [(u_units, q_c, "Units"), (u_rev, rev_c, "Revenue")]:
                    if check_answer(u, c):
                        st.write(f"✅ {lbl}: {c:,.0f} correct")
                    else:
                        st.write(f"❌ {lbl}: Should be {c:,.0f}")

            if st.button("Show Solution", key="be_p2_sol"):
                q_c = (60000+20000)/(50-30)
                display_solution(f"""
                <strong>a) Q = (F + π) ÷ (P − V)</strong><br>
                Q = ($60,000 + $20,000) ÷ ($50 − $30) = $80,000 ÷ $20 = <strong>{q_c:,.0f} units</strong><br><br>
                <strong>b) Revenue = Q × P</strong><br>
                Revenue = {q_c:,.0f} × $50 = <strong>${q_c*50:,.0f}</strong>
                """)

        with st.expander("🔴 Problem 3: Indifference Point — Manual vs. Automated (Hard)"):
            display_practice_problem(3, "Hard",
                "| Option | Fixed Costs | Variable Cost/Unit |  \n"
                "|--------|-------------|-------------------|  \n"
                "| Manual | $20,000 | $15 |  \n"
                "| Automated | $80,000 | $5 |  \n\n"
                "a) Find the indifference point.  \n"
                "b) Which is better at 5,000 units?  \n"
                "c) Which is better at 8,000 units?  \n"
                "d) By how much does the better option save at each volume?")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                u_ip  = st.number_input("Indifference point (units):", key="be_p3_ip")
            with col2:
                u_b5  = st.selectbox("Better at 5,000:", ["Select...", "Manual", "Automated"],
                                      key="be_p3_5k")
            with col3:
                u_b8  = st.selectbox("Better at 8,000:", ["Select...", "Manual", "Automated"],
                                      key="be_p3_8k")
            with col4:
                pass  # spacer

            if st.button("Check Answers", key="be_p3_btn_v2"):
                ip = (80000-20000)/(15-5)
                tc_m5 = 20000+15*5000; tc_a5 = 80000+5*5000
                tc_m8 = 20000+15*8000; tc_a8 = 80000+5*8000
                results = []
                if check_answer(u_ip, ip):    results.append(f"✅ Indifference point: {ip:,.0f} units")
                else:                          results.append(f"❌ Indifference: Should be {ip:,.0f}")
                if u_b5 == "Manual":           results.append(f"✅ Manual better at 5,000 (${tc_m5:,} vs ${tc_a5:,})")
                elif u_b5 != "Select...":      results.append(f"❌ At 5,000: Manual (${tc_m5:,}) < Automated (${tc_a5:,})")
                if u_b8 == "Automated":        results.append(f"✅ Automated better at 8,000 (${tc_a8:,} vs ${tc_m8:,})")
                elif u_b8 != "Select...":      results.append(f"❌ At 8,000: Automated (${tc_a8:,}) < Manual (${tc_m8:,})")
                for r in results: st.write(r)

            if st.button("Show Complete Solution", key="be_p3_sol"):
                ip=(80000-20000)/(15-5)
                tc_m5=20000+15*5000; tc_a5=80000+5*5000
                tc_m8=20000+15*8000; tc_a8=80000+5*8000
                display_solution(f"""
                <strong>a) Indifference Point</strong><br>
                Q* = (F_B − F_A) / (V_A − V_B) = ($80,000 − $20,000) / ($15 − $5)
                = $60,000 / $10 = <strong>{ip:,.0f} units</strong><br><br>
                <strong>b) At 5,000 units</strong> (below indifference — Manual wins):<br>
                Manual: $20,000 + $15×5,000 = ${tc_m5:,}<br>
                Automated: $80,000 + $5×5,000 = ${tc_a5:,}<br>
                <strong>Manual saves ${tc_a5-tc_m5:,}</strong><br><br>
                <strong>c) At 8,000 units</strong> (above indifference — Automated wins):<br>
                Manual: $20,000 + $15×8,000 = ${tc_m8:,}<br>
                Automated: $80,000 + $5×8,000 = ${tc_a8:,}<br>
                <strong>Automated saves ${tc_m8-tc_a8:,}</strong>
                """)

        with st.expander("🔴 Problem 4: Make vs. Buy (Hard)"):
            display_practice_problem(4, "Hard",
                "**Make:** F = $100,000/year, V = $15/unit  \n"
                "**Buy:** $25/unit (no fixed costs)  \n\n"
                "a) Indifference point?  \n"
                "b) At demand = 8,000 units, which is better and by how much?  \n"
                "c) At what demand does Make save $50,000 over Buy?")

            col1, col2, col3 = st.columns(3)
            with col1:
                u_ip4  = st.number_input("Indifference (units):", key="be_p4_ip")
            with col2:
                u_b4   = st.selectbox("Better at 8,000:", ["Select...", "Make", "Buy"],
                                       key="be_p4_b")
            with col3:
                u_sav4 = st.number_input("Savings ($):", key="be_p4_sav")

            u_d50 = st.number_input("Volume where Make saves $50,000:", key="be_p4_d50")

            if st.button("Check Answers", key="be_p4_btn_v2"):
                ip = 100000/(25-15)
                mk8 = 100000+15*8000; by8 = 25*8000
                d50 = (100000+50000)/(25-15)   # Make saves 50k when (25-15)Q - 100000 = 50000
                # Actually: Buy - Make = 50000 → 25Q - (100000+15Q) = 50000 → 10Q = 150000 → Q = 15000
                d50_correct = (100000 + 50000) / (25 - 15)  # = 15000

                results = []
                if check_answer(u_ip4, ip):            results.append(f"✅ Indifference: {ip:,.0f} units")
                else:                                   results.append(f"❌ Indifference: Should be {ip:,.0f}")
                best8 = "Buy" if by8 < mk8 else "Make"
                if u_b4 == best8:                      results.append(f"✅ {best8} better at 8,000")
                elif u_b4 != "Select...":               results.append(f"❌ At 8,000: {best8} is better")
                if check_answer(u_sav4, abs(mk8-by8)): results.append(f"✅ Savings ${abs(mk8-by8):,} correct")
                else:                                   results.append(f"❌ Savings: Should be ${abs(mk8-by8):,}")
                if check_answer(u_d50, d50_correct):   results.append(f"✅ Make saves $50K at {d50_correct:,.0f} units")
                else:                                   results.append(f"❌ $50K savings volume: Should be {d50_correct:,.0f}")
                for r in results: st.write(r)

            if st.button("Show Complete Solution", key="be_p4_sol_v2"):
                ip=100000/(25-15); mk8=100000+15*8000; by8=25*8000
                d50=(100000+50000)/(25-15)
                display_solution(f"""
                <strong>a) Indifference: Set Make = Buy</strong><br>
                $100,000 + $15Q = $25Q → $100,000 = $10Q → <strong>Q* = {ip:,.0f} units</strong><br><br>
                <strong>b) At 8,000 units</strong> (8,000 < 10,000 → Buy wins):<br>
                Make = $100,000 + $15×8,000 = ${mk8:,}<br>
                Buy  = $25 × 8,000 = ${by8:,}<br>
                <strong>Buy saves ${mk8-by8:,}</strong><br><br>
                <strong>c) Volume where Make saves $50,000 over Buy</strong><br>
                Buy − Make = $50,000<br>
                $25Q − ($100,000 + $15Q) = $50,000<br>
                $10Q = $150,000 → <strong>Q = {d50:,.0f} units</strong>
                """)


# ============================================================
# MODULE 5: DECISION TREES (Chapter 5) - ENHANCED V5.0
# ============================================================
def module_decision():
    display_header("🌳", "Chapter 5", "Decision Trees & EMV",
                   "Structured decision-making under uncertainty using Expected Monetary Value")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "🔬 EMV Calculator", "📊 EVPI & Sensitivity", "🎓 Practice"])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Decision Tree Analysis")
        st.write(
            "A **decision tree** is a visual, quantitative model for evaluating sequential choices "
            "under uncertainty. It maps out decisions, chance outcomes, and their payoffs, "
            "then uses backward induction (the 'rollback' method) to find the optimal strategy."
        )

        display_citation(
            "A decision tree is a schematic model of the alternatives available to the decision "
            "maker along with their possible consequences. Decision trees are particularly "
            "useful for analyzing capacity planning and investment decisions that have "
            "sequential components.",
            "Jacobs & Chase (2024, p. 148)"
        )

        st.markdown("#### Decision Tree Components")
        comp_df = pd.DataFrame({
            "Symbol":   ["◻️ Square", "⭕ Circle", "→ Branch", "$ Terminal"],
            "Name":     ["Decision Node", "Chance Node",
                          "Branch (alternative or outcome)", "Terminal Node / Payoff"],
            "Meaning":  ["Decision maker CHOOSES one path",
                          "Nature/chance determines outcome — probabilities sum to 1",
                          "Alternative being considered or outcome occurring",
                          "Final payoff (profit, NPV, cost) at end of path"],
            "Solved By":["Choose branch with highest EMV",
                          "Calculate weighted average (EMV)",
                          "Label with probability or description",
                          "Stated given — not calculated"]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        display_formula_card("Expected Monetary Value",
            r"EMV = \sum_{i=1}^{n} P_i \times V_i")
        display_formula_card("Expected Value of Perfect Information",
            r"EVPI = EV_{\text{with PI}} - EV_{\text{without PI}}")
        display_formula_card("EV with Perfect Information",
            r"EV_{\text{PI}} = \sum_{j} P_j \times \max_i(V_{ij})")

        display_textbook_content(
            "The Rollback (Backward Induction) Method",
            """Decision trees are solved from RIGHT to LEFT:
            1. Start at the terminal nodes (rightmost payoffs)
            2. At each CHANCE node: compute EMV = Σ(P × payoff)
            3. At each DECISION node: select the branch with highest EMV
               (write the chosen EMV at the node; prune inferior branches)
            4. Work leftward until reaching the root (first decision)
            The surviving path is the optimal strategy."""
        )

        display_key_insight(
            "EVPI Upper Bound",
            "The EVPI tells you the maximum you should pay for a perfect market research study. "
            "If a study costs less than EVPI, it may be worth commissioning. "
            "In practice, market research is never perfect — Expected Value of "
            "Sample Information (EVSI) < EVPI always."
        )

        st.markdown("#### When to Use Decision Trees vs. Other Tools")
        when_df = pd.DataFrame({
            "Situation":      ["One-time decision, two demand states",
                                "Sequential decisions (build small, then expand?)",
                                "Multiple alternatives with known probabilities",
                                "Continuous demand distribution",
                                "Repetitive decisions over many cycles"],
            "Best Tool":      ["EMV formula", "Decision Tree",
                                "Decision Tree or Payoff Table",
                                "Simulation / Newsvendor model",
                                "Statistical Process Control"],
            "Chapter":        ["5", "5", "5", "Advanced", "13"]
        })
        st.dataframe(when_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### EMV Calculator — Up to 3 Alternatives")

        num_alts = st.radio("Number of Alternatives:", [2, 3], horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            prob_high = st.slider("P(High Demand) %", 0, 100, 60) / 100
            prob_low  = 1 - prob_high
            st.write(f"P(Low Demand) = {prob_low:.0%}")

        with col2:
            demand_labels = st.text_input(
                "Scenario labels (comma-separated)",
                value="High Demand, Low Demand"
            ).split(",")
            demand_labels = [d.strip() for d in demand_labels]
            if len(demand_labels) < 2:
                demand_labels = ["High Demand", "Low Demand"]

        st.markdown("---")
        alt_data = []
        alt_cols = st.columns(num_alts)

        default_alts = [
            {"name": "Large Facility",  "high": 200000, "low": -50000},
            {"name": "Small Facility",  "high": 90000,  "low": 25000},
            {"name": "No Investment",   "high": 0,      "low": 0},
        ]

        for i, col in enumerate(alt_cols):
            with col:
                st.markdown(f"#### Alternative {i+1}")
                name = st.text_input("Name", value=default_alts[i]["name"],
                                      key=f"dt_name_{i}")
                v_high = st.number_input(f"Payoff: {demand_labels[0]} ($)",
                                          value=default_alts[i]["high"],
                                          key=f"dt_h_{i}", step=5000)
                v_low  = st.number_input(f"Payoff: {demand_labels[1]} ($)",
                                          value=default_alts[i]["low"],
                                          key=f"dt_l_{i}", step=5000)

                emv = prob_high * v_high + prob_low * v_low
                st.metric("EMV", f"${emv:,.0f}")
                st.latex(
                    rf"EMV = {prob_high:.2f}({v_high:,}) + {prob_low:.2f}({v_low:,})"
                )
                alt_data.append({"name": name, "v_high": v_high,
                                  "v_low": v_low, "emv": emv})

        # ── Recommendation ──
        best      = max(alt_data, key=lambda x: x["emv"])
        worst     = min(alt_data, key=lambda x: x["emv"])
        margin    = best["emv"] - sorted(alt_data, key=lambda x: x["emv"])[-2]["emv"]

        st.markdown("---")
        st.success(
            f"✅ **Optimal Decision: {best['name']}**  \n"
            f"EMV = **${best['emv']:,.0f}**  |  "
            f"Margin over next-best = **${margin:,.0f}**"
        )

        # ── EMV Comparison Chart ──
        fig_emv = go.Figure(go.Bar(
            x=[a["name"] for a in alt_data],
            y=[a["emv"] for a in alt_data],
            marker_color=["#2ecc71" if a["name"] == best["name"] else "#3498db"
                           for a in alt_data],
            text=[f"${a['emv']:,.0f}" for a in alt_data],
            textposition="outside"
        ))
        fig_emv.add_hline(y=0, line_color="black", line_width=1)
        fig_emv.update_layout(
            title="Expected Monetary Value Comparison",
            xaxis_title="Alternative", yaxis_title="EMV ($)",
            template="plotly_white", height=360
        )
        st.plotly_chart(fig_emv, use_container_width=True)

        # ── Payoff Summary Table ──
        st.markdown("#### Full Payoff Table")
        payoff_df = pd.DataFrame({
            "Alternative":               [a["name"] for a in alt_data],
            f"Payoff: {demand_labels[0]} (P={prob_high:.0%})":
                                          [f"${a['v_high']:,.0f}" for a in alt_data],
            f"Payoff: {demand_labels[1]} (P={prob_low:.0%})":
                                          [f"${a['v_low']:,.0f}"  for a in alt_data],
            "EMV":                        [f"${a['emv']:,.0f}"    for a in alt_data],
            "Decision":                   ["← BEST" if a["name"] == best["name"]
                                            else ("← WORST" if a["name"] == worst["name"] else "")
                                            for a in alt_data]
        })
        st.dataframe(payoff_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### EVPI & Probability Sensitivity")

        if not alt_data:
            st.info("Set up alternatives in the EMV Calculator tab first.")
        else:
            # ── EVPI ──
            ev_with_pi    = (prob_high * max(a["v_high"] for a in alt_data) +
                              prob_low  * max(a["v_low"]  for a in alt_data))
            ev_without_pi = max(a["emv"] for a in alt_data)
            evpi          = ev_with_pi - ev_without_pi

            st.markdown("#### Expected Value of Perfect Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("EV with Perfect Info",    f"${ev_with_pi:,.0f}")
            col2.metric("EV without Perfect Info", f"${ev_without_pi:,.0f}")
            col3.metric("EVPI",                    f"${evpi:,.0f}",
                         help="Maximum you should pay for a perfect market forecast")

            st.latex(
                rf"EV_{{PI}} = {prob_high:.2f} \times {max(a['v_high'] for a in alt_data):,} "
                rf"+ {prob_low:.2f} \times {max(a['v_low'] for a in alt_data):,} = {ev_with_pi:,.0f}"
            )
            st.latex(rf"EVPI = {ev_with_pi:,.0f} - {ev_without_pi:,.0f} = {evpi:,.0f}")

            st.info(
                f"💡 Commission market research only if its cost is **less than ${evpi:,.0f}**. "
                f"Research costing more than EVPI can never be worthwhile, even if perfect."
            )

            # ── Probability Sensitivity Chart ──
            st.markdown("---")
            st.markdown("#### Sensitivity: Optimal Decision vs. P(High Demand)")
            prob_range = [p/100 for p in range(0, 101, 2)]
            emv_series = {a["name"]: [] for a in alt_data}
            for p in prob_range:
                for a in alt_data:
                    emv_series[a["name"]].append(p * a["v_high"] + (1-p) * a["v_low"])

            fig_sens = go.Figure()
            colors_s = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            for i, a in enumerate(alt_data):
                fig_sens.add_trace(go.Scatter(
                    x=prob_range, y=emv_series[a["name"]],
                    mode="lines", name=a["name"],
                    line=dict(color=colors_s[i % len(colors_s)], width=2)
                ))
            fig_sens.add_vline(x=prob_high, line_dash="dot", line_color="gray",
                                annotation_text=f"Current P={prob_high:.0%}",
                                annotation_position="top right")
            fig_sens.update_layout(
                title="EMV Sensitivity to P(High Demand) — Crossover Points Show When to Switch Decisions",
                xaxis_title="P(High Demand)", yaxis_title="EMV ($)",
                template="plotly_white", height=400,
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig_sens, use_container_width=True)

            # Find crossover points
            st.markdown("#### Decision Crossover Points")
            st.write(
                "The probability value(s) at which the optimal decision changes "
                "(where two EMV lines intersect):"
            )
            for i in range(len(alt_data)):
                for j in range(i+1, len(alt_data)):
                    a1, a2 = alt_data[i], alt_data[j]
                    diff_high = a1["v_high"] - a2["v_high"]
                    diff_low  = a1["v_low"]  - a2["v_low"]
                    if diff_high != diff_low:  # non-parallel lines
                        # p*diff_high + (1-p)*diff_low = 0 → p*(diff_high-diff_low) = -diff_low
                        p_cross = -diff_low / (diff_high - diff_low)
                        if 0 < p_cross < 1:
                            emv_cross = p_cross * a1["v_high"] + (1-p_cross) * a1["v_low"]
                            st.write(
                                f"**{a1['name']} ↔ {a2['name']}**: Switch at "
                                f"P(High) = **{p_cross:.1%}**  "
                                f"(EMV = ${emv_cross:,.0f})"
                            )

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Basic EMV (Easy)"):
            display_practice_problem(1, "Easy",
                "Calculate the EMV for Option A:  \n"
                "- 40% chance of **$100,000**  \n"
                "- 60% chance of **$20,000**")

            show_h1 = st.checkbox("Show Hint", key="dt_h1")
            if show_h1:
                display_hint("EMV = Σ(probability × payoff). Sum all branches.")

            u_emv = st.number_input("EMV ($):", key="dt_p1_v2")
            if st.button("Check Answer", key="dt_p1_btn"):
                correct = 0.4*100000 + 0.6*20000
                if check_answer(u_emv, correct):
                    st.success(f"✅ EMV = ${correct:,.0f}")
                else:
                    display_solution(
                        f"EMV = 0.40×$100,000 + 0.60×$20,000 = $40,000 + $12,000 = "
                        f"<strong>${correct:,.0f}</strong>"
                    )

        with st.expander("🟡 Problem 2: Choose Best Alternative (Medium)"):
            display_practice_problem(2, "Medium",
                "P(High) = 40%, P(Low) = 60%  \n\n"
                "| Alternative | High Demand | Low Demand |  \n"
                "|-------------|-------------|------------|  \n"
                "| Option A | $100,000 | $20,000 |  \n"
                "| Option B | $80,000 | $30,000 |  \n"
                "| Option C | $50,000 | $45,000 |  \n\n"
                "a) Calculate EMV for all three.  \n"
                "b) Which option should be chosen?  \n"
                "c) At what P(High) does Option B become better than Option A?")

            col1, col2, col3 = st.columns(3)
            with col1:
                u_a = st.number_input("EMV(A) ($):", key="dt_p2_a")
            with col2:
                u_b = st.number_input("EMV(B) ($):", key="dt_p2_b")
            with col3:
                u_c = st.number_input("EMV(C) ($):", key="dt_p2_c")
            u_best   = st.selectbox("Best option:", ["Select...", "A", "B", "C"], key="dt_p2_best")
            u_cross  = st.number_input("Crossover P(High) %:", format="%.1f", key="dt_p2_cross")

            if st.button("Check Answers", key="dt_p2_btn"):
                emv_a = 0.4*100000+0.6*20000
                emv_b = 0.4*80000 +0.6*30000
                emv_c = 0.4*50000 +0.6*45000
                best  = max([("A", emv_a), ("B", emv_b), ("C", emv_c)], key=lambda x: x[1])
                # Crossover A=B: p*100k+(1-p)*20k = p*80k+(1-p)*30k
                # p*80k+20k = p*60k+30k → p*20k = 10k → p = 0.5
                cross_ab = (30000-20000)/((100000-20000)-(80000-30000)) * 100  # = 10k/20k = 50%
                for u, c, lbl in [(u_a, emv_a, "EMV(A)"), (u_b, emv_b, "EMV(B)"),
                                   (u_c, emv_c, "EMV(C)")]:
                    if check_answer(u, c):
                        st.write(f"✅ {lbl} = ${c:,.0f} correct")
                    else:
                        st.write(f"❌ {lbl}: Should be ${c:,.0f}")
                if u_best == best[0]:
                    st.write(f"✅ Best option: {best[0]}")
                elif u_best != "Select...":
                    st.write(f"❌ Best: Should be {best[0]} (${best[1]:,.0f})")
                if check_answer(u_cross, cross_ab, 0.5):
                    st.write(f"✅ Crossover: {cross_ab:.1f}% correct")
                else:
                    st.write(f"❌ Crossover: Should be {cross_ab:.1f}%")

            if st.button("Show Solution", key="dt_p2_sol"):
                emv_a=0.4*100000+0.6*20000
                emv_b=0.4*80000+0.6*30000
                emv_c=0.4*50000+0.6*45000
                display_solution(f"""
                <strong>EMV Calculations</strong><br>
                EMV(A) = 0.4×$100,000 + 0.6×$20,000 = $40,000 + $12,000 = <strong>${emv_a:,.0f}</strong><br>
                EMV(B) = 0.4×$80,000 + 0.6×$30,000 = $32,000 + $18,000 = <strong>${emv_b:,.0f}</strong><br>
                EMV(C) = 0.4×$50,000 + 0.6×$45,000 = $20,000 + $27,000 = <strong>${emv_c:,.0f}</strong><br><br>
                <strong>Best: Option A</strong> at ${emv_a:,.0f}<br><br>
                <strong>A vs. B crossover:</strong><br>
                p×$100K + (1−p)×$20K = p×$80K + (1−p)×$30K<br>
                20p + 20K = 50p + 30K → wrong setup — let me solve properly:<br>
                p(100K−80K) = (1−p)(30K−20K)<br>
                20p = 10(1−p) → 20p = 10 − 10p → 30p = 10 → p = 1/3 ≈ <strong>33.3%</strong><br>
                At P(High) &gt; 33.3%, Option A is better. Below that, Option B is preferred.
                """)

        with st.expander("🔴 Problem 3: EVPI Full Calculation (Hard)"):
            display_practice_problem(3, "Hard",
                "Using three alternatives from Problem 2 (P(High)=40%, P(Low)=60%):  \n\n"
                "| Alternative | High | Low |  \n"
                "|-------------|------|-----|  \n"
                "| A | $100,000 | $20,000 |  \n"
                "| B | $80,000 | $30,000 |  \n"
                "| C | $50,000 | $45,000 |  \n\n"
                "a) Calculate EV with Perfect Information  \n"
                "b) Calculate EVPI  \n"
                "c) Interpret: should the firm pay $5,000 for a market forecast?")

            col1, col2 = st.columns(2)
            with col1:
                u_ev_pi  = st.number_input("EV with PI ($):", key="dt_p3_evpi")
            with col2:
                u_evpi3  = st.number_input("EVPI ($):", key="dt_p3_evpi_v")
            u_worth = st.selectbox("Pay $5,000 for forecast?",
                                    ["Select...", "Yes", "No"], key="dt_p3_worth")

            if st.button("Check Answers", key="dt_p3_btn_v2"):
                ev_pi_c  = 0.4*100000 + 0.6*45000  # best in each state: A in high, C in low
                ev_wo_pi = max(0.4*100000+0.6*20000,
                                0.4*80000+0.6*30000,
                                0.4*50000+0.6*45000)
                evpi_c   = ev_pi_c - ev_wo_pi
                worth_c  = "Yes" if evpi_c > 5000 else "No"
                for u,c,lbl in [(u_ev_pi, ev_pi_c, "EV with PI"),
                                 (u_evpi3, evpi_c, "EVPI")]:
                    if check_answer(u, c):
                        st.write(f"✅ {lbl}: ${c:,.0f} correct")
                    else:
                        st.write(f"❌ {lbl}: Should be ${c:,.0f}")
                if u_worth == worth_c:
                    st.write(f"✅ {'Pay $5K — EVPI > $5,000' if worth_c == 'Yes' else 'Do not pay — EVPI < $5,000'}")
                elif u_worth != "Select...":
                    st.write(f"❌ {'Pay' if worth_c == 'Yes' else 'Do not pay'}: EVPI = ${evpi_c:,.0f}")

            if st.button("Show Complete Solution", key="dt_p3_sol_v2"):
                ev_pi_c  = 0.4*100000 + 0.6*45000
                ev_wo_pi = max(0.4*100000+0.6*20000, 0.4*80000+0.6*30000, 0.4*50000+0.6*45000)
                evpi_c   = ev_pi_c - ev_wo_pi
                display_solution(f"""
                <strong>Step 1: What would we choose under perfect information?</strong><br>
                • If High Demand → Choose A: best payoff = $100,000<br>
                • If Low Demand  → Choose C: best payoff = $45,000<br><br>
                <strong>Step 2: EV with Perfect Information</strong><br>
                EV_PI = 0.40×$100,000 + 0.60×$45,000 = $40,000 + $27,000 = <strong>${ev_pi_c:,.0f}</strong><br><br>
                <strong>Step 3: EV without Perfect Information (best EMV)</strong><br>
                Best EMV = max(${0.4*100000+0.6*20000:,.0f}, ${0.4*80000+0.6*30000:,.0f}, ${0.4*50000+0.6*45000:,.0f}) = <strong>${ev_wo_pi:,.0f}</strong><br><br>
                <strong>Step 4: EVPI</strong><br>
                EVPI = ${ev_pi_c:,.0f} − ${ev_wo_pi:,.0f} = <strong>${evpi_c:,.0f}</strong><br><br>
                <strong>Step 5: Pay $5,000?</strong><br>
                {'✅ Yes — EVPI = $' + f'{evpi_c:,.0f}' + ' > $5,000. The information is worth more than its cost.'
                 if evpi_c > 5000 else
                 '❌ No — EVPI = $' + f'{evpi_c:,.0f}' + ' < $5,000. Do not pay more than EVPI for any forecast.'}
                """)

        with st.expander("🔴 Problem 4: Sequential Decision Tree (Hard)"):
            display_practice_problem(4, "Hard",
                "A company can build a **small** or **large** plant.  \n"
                "If they build small and demand is high, they can **expand** later.  \n\n"
                "**Large Plant:**  \n"
                "- High demand (60%): $250,000  \n"
                "- Low demand (40%): −$100,000  \n\n"
                "**Small Plant:**  \n"
                "- High demand (60%): expand for $50,000 extra cost → net payoff = $180,000  \n"
                "- High demand, no expand: $130,000  \n"
                "- Low demand (40%): $40,000  \n\n"
                "a) What is the EMV of the Large Plant?  \n"
                "b) If high demand occurs for Small, should they expand?  \n"
                "c) What is the optimal strategy and its EMV?")

            if st.button("Show Complete Solution", key="dt_p4_sol"):
                emv_large = 0.6*250000 + 0.4*(-100000)
                # Small: if high demand → expand (180k) vs no expand (130k) → expand wins
                best_small_high = 180000  # expand
                emv_small = 0.6*best_small_high + 0.4*40000
                display_solution(f"""
                <strong>a) EMV — Large Plant</strong><br>
                EMV = 0.60×$250,000 + 0.40×(−$100,000)<br>
                EMV = $150,000 − $40,000 = <strong>${emv_large:,.0f}</strong><br><br>
                <strong>b) Small Plant — If High Demand, Expand?</strong><br>
                Expand:    net payoff = $180,000<br>
                No Expand: payoff     = $130,000<br>
                <strong>✅ Yes — Expand ($180,000 > $130,000)</strong><br><br>
                <strong>c) EMV — Small Plant (using expand if high)</strong><br>
                EMV = 0.60×$180,000 + 0.40×$40,000<br>
                EMV = $108,000 + $16,000 = <strong>${emv_small:,.0f}</strong><br><br>
                <strong>Optimal Strategy: {'Large Plant' if emv_large > emv_small else 'Small Plant (expand if high)'}</strong><br>
                EMV = ${max(emv_large, emv_small):,.0f}
                (margin = ${abs(emv_large-emv_small):,.0f} over the other option)
                """)

# ============================================================
# MODULE 6: LEARNING CURVES (Chapter 6) - ENHANCED V5.0
# ============================================================
def module_learning():
    display_header("📉", "Chapter 6", "Learning Curves",
                   "Experience-based cost reduction through repetition and organizational learning")

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Simulator", "📊 Curve Chart", "🎓 Practice"])

    with tab1:
        st.markdown("### Learning Curve Theory")
        st.write(
            "The **learning curve** (experience curve) captures the systematic reduction in "
            "per-unit time or cost as cumulative production doubles. Workers become faster "
            "through repetition, managers improve scheduling, and tooling is refined — "
            "compounding to produce predictable, quantifiable improvement."
        )

        display_citation(
            "The learning curve theory is based on three assumptions: (1) the amount of time "
            "required to complete a given task will be less each time the task is undertaken, "
            "(2) the unit time will decrease at a decreasing rate, and (3) the reduction in "
            "time will follow a specific and predictable pattern.",
            "Jacobs & Chase (2024, p. 168)"
        )

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Unit Time (Power Model)", r"Y_x = K \cdot x^n")
            display_formula_card("Learning Exponent",        r"n = \frac{\log(b)}{\log(2)}")
            display_formula_card("Cumulative Average Time",
                                 r"\bar{Y}_x \approx \frac{K \cdot x^{n+1}}{(n+1) \cdot x}")
        with col2:
            display_formula_card("Cumulative Total Time",
                                 r"T_x = K \cdot \frac{x^{n+1}}{n+1} \quad (n \neq -1)")
            display_formula_card("Unit-to-Unit Ratio",
                                 r"\frac{Y_{2x}}{Y_x} = b \quad \text{(doubling rule)}")

        st.markdown("#### Interpreting the Learning Rate (b)")
        lr_table = pd.DataFrame({
            "Learning Rate": ["60%", "70%", "75%", "80%", "85%", "90%", "95%"],
            "n (exponent)":  [f"{math.log(r)/math.log(2):.4f}" for r in [0.60,0.70,0.75,0.80,0.85,0.90,0.95]],
            "Industry":      ["Aerospace (complex)", "Electronics assembly",
                              "Shipbuilding", "Aircraft manufacturing",
                              "Automotive assembly", "Machine operations",
                              "Simple repetitive tasks"],
            "Unit 8 / Unit 1": [f"{r**3*100:.1f}%" for r in [0.60,0.70,0.75,0.80,0.85,0.90,0.95]]
        })
        st.dataframe(lr_table, use_container_width=True)

        display_key_insight(
            "The Doubling Rule",
            "An 80% learning curve means every time cumulative production **doubles**, "
            "unit time drops to 80% of the previous level. "
            "Unit 1 = 100 hrs → Unit 2 = 80 hrs → Unit 4 = 64 hrs → Unit 8 = 51.2 hrs → Unit 16 = 40.96 hrs."
        )

        st.markdown("#### Unit vs. Cumulative Average Models")
        model_df = pd.DataFrame({
            "Model":       ["Unit Time Model", "Cumulative Average Model"],
            "Formula":     ["Yₓ = K·xⁿ",       "Ȳₓ = K·xⁿ (n based on cum avg, not unit)"],
            "What It Tracks": ["Time for the x-th unit",
                               "Average time across all units 1 through x"],
            "Use Case":    ["Estimating a specific unit's time",
                             "Cost estimating and bidding on contracts"]
        })
        st.dataframe(model_df, use_container_width=True)

    with tab2:
        st.markdown("### Learning Curve Calculator")

        col1, col2 = st.columns(2)
        with col1:
            k             = st.number_input("First Unit Time (K, hours)", value=100.0, min_value=0.1)
            learning_rate = st.slider("Learning Rate (%)", 50, 99, 80)
            b             = learning_rate / 100
            n             = math.log(b) / math.log(2)

            st.metric("Learning Exponent (n)", f"{n:.4f}")
            st.latex(rf"n = \frac{{\log({b})}}{{\log(2)}} = {n:.4f}")

        with col2:
            st.markdown("#### Doubling Table")
            doublings = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            times_d   = [k * (u ** n) for u in doublings]
            pct_d     = [t/k*100 for t in times_d]
            df_double = pd.DataFrame({
                "Unit (doubling)": doublings,
                "Time (hrs)":      [f"{t:.2f}" for t in times_d],
                "% of Unit 1":     [f"{p:.1f}%" for p in pct_d],
                "Savings vs prev": ["—"] + [f"{(1-times_d[i]/times_d[i-1])*100:.1f}%" for i in range(1, len(times_d))]
            })
            st.dataframe(df_double, use_container_width=True)

        st.markdown("---")
        st.markdown("### Specific Unit Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            target_unit = st.number_input("Calculate for Unit #", value=10, min_value=1)
        with col2:
            bid_units_from = st.number_input("Contract From Unit", value=11, min_value=1)
        with col3:
            bid_units_to   = st.number_input("Contract To Unit",   value=20, min_value=2)

        # Unit time
        unit_time = k * (target_unit ** n)

        # True cumulative: numerical sum (more accurate than integral for small x)
        cum_time_to_x   = sum(k * (u ** n) for u in range(1, int(target_unit) + 1))
        cum_avg_to_x    = cum_time_to_x / target_unit

        # Contract block cost
        block_time = sum(k * (u ** n) for u in range(int(bid_units_from), int(bid_units_to) + 1))
        block_avg  = block_time / (bid_units_to - bid_units_from + 1)

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Time for Unit {target_unit}",  f"{unit_time:.2f} hrs")
        col2.metric(f"Cum. Total (1–{target_unit})", f"{cum_time_to_x:.2f} hrs")
        col3.metric(f"Cum. Average (1–{target_unit})",f"{cum_avg_to_x:.2f} hrs")

        col1, col2 = st.columns(2)
        col1.metric(f"Contract Block Total ({bid_units_from}–{bid_units_to})",
                    f"{block_time:.2f} hrs")
        col2.metric("Contract Block Avg/Unit", f"{block_avg:.2f} hrs")

        st.latex(rf"Y_{{{target_unit}}} = {k} \times {target_unit}^{{{n:.4f}}} = {unit_time:.2f} \text{{ hrs}}")

        # ─── Full Data Table ───
        with st.expander("📋 Full Unit-by-Unit Table"):
            max_tbl = st.number_input("Show units 1 through:", value=20, min_value=2, max_value=200)
            tbl_units = list(range(1, int(max_tbl)+1))
            tbl_times = [k*(u**n) for u in tbl_units]
            tbl_cum   = [sum(k*(j**n) for j in range(1, i+1)) for i in tbl_units]
            tbl_avg   = [c/u for c, u in zip(tbl_cum, tbl_units)]
            tbl_pct   = [t/k*100 for t in tbl_times]
            tbl_df = pd.DataFrame({
                "Unit":          tbl_units,
                "Unit Time":     [f"{t:.2f}" for t in tbl_times],
                "Cum. Total":    [f"{c:.2f}" for c in tbl_cum],
                "Cum. Average":  [f"{a:.2f}" for a in tbl_avg],
                "% of Unit 1":   [f"{p:.1f}%" for p in tbl_pct]
            })
            st.dataframe(tbl_df, use_container_width=True)

    with tab3:
        st.markdown("### Learning Curve Chart")

        col1, col2 = st.columns([1, 2])
        with col1:
            k_c  = st.number_input("K (first unit hrs)", value=100.0, key="lc_k_c")
            b_c  = st.slider("Learning Rate (%)", 50, 99, 80, key="lc_b_c") / 100
            n_c  = math.log(b_c) / math.log(2)
            max_u = st.number_input("Max Unit to Plot", value=50, min_value=5, max_value=500,
                                    step=5)
            show_rates = st.multiselect("Compare Rates (%)",
                                        [60, 70, 75, 80, 85, 90, 95], default=[70, 80, 90])

        x_plot = list(range(1, int(max_u)+1))

        fig = go.Figure()
        colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#9b59b6","#1abc9c"]
        for i, rate in enumerate(show_rates):
            ni    = math.log(rate/100) / math.log(2)
            times = [k_c * (u**ni) for u in x_plot]
            fig.add_trace(go.Scatter(x=x_plot, y=times, mode="lines",
                                     name=f"{rate}% curve",
                                     line=dict(color=colors[i%len(colors)], width=2)))

        # Mark doubling points
        for dbl in [1, 2, 4, 8, 16, 32]:
            if dbl <= max_u:
                y_dbl = k_c * (dbl ** n_c)
                fig.add_trace(go.Scatter(x=[dbl], y=[y_dbl], mode="markers",
                                         marker=dict(size=10, color="#2c3e50", symbol="circle"),
                                         showlegend=(dbl == 1),
                                         name="Doubling Points"))

        fig.update_layout(title="Learning Curves — Unit Time by Cumulative Output",
                          xaxis_title="Cumulative Unit Number",
                          yaxis_title="Hours per Unit",
                          template="plotly_white", height=450,
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Log-log version
        with st.expander("📊 Log-Log Chart (linearized view)"):
            fig_log = go.Figure()
            import math
            for i, rate in enumerate(show_rates):
                ni     = math.log(rate/100) / math.log(2)
                log_x  = [math.log10(u) for u in x_plot]
                log_y  = [math.log10(k_c*(u**ni)) for u in x_plot]
                fig_log.add_trace(go.Scatter(x=log_x, y=log_y, mode="lines",
                                              name=f"{rate}%",
                                              line=dict(color=colors[i%len(colors)], width=2)))
            fig_log.update_layout(title="Learning Curve — Log-Log Scale (straight line = power model)",
                                  xaxis_title="log₁₀(Unit Number)",
                                  yaxis_title="log₁₀(Hours)",
                                  template="plotly_white", height=380)
            st.plotly_chart(fig_log, use_container_width=True)
            st.info("💡 On a log-log scale, learning curves become straight lines. "
                    "The slope equals n (the learning exponent).")

        with col2:
            st.metric("n at selected rate", f"{n_c:.4f}")
            st.metric(f"Unit {int(max_u)} time", f"{k_c*(max_u**n_c):.2f} hrs")
            st.metric("Total reduction",
                      f"{(1 - k_c*(max_u**n_c)/k_c)*100:.1f}% from unit 1 to {int(max_u)}")

    with tab4:
        st.markdown("### 📝 Learning Curve Practice Problems")

        with st.expander("🟢 P1: Basic Unit Time (Easy)"):
            display_practice_problem(1, "Easy",
                "First unit = 100 hrs, 80% learning curve. How long will unit 8 take?")
            show_hint = st.checkbox("Show Hint", key="lc_h1")
            if show_hint:
                display_hint("Unit 8 is the 3rd doubling from unit 1: 1→2→4→8. Each doubling × 0.80.")
            user_ans = st.number_input("Your Answer (hrs):", key="lc_p1", value=0.0)
            if st.button("Check Answer", key="lc_p1_btn"):
                n1 = math.log(0.8)/math.log(2)
                correct = 100 * (8**n1)
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Y₈ = {correct:.1f} hrs")
                else:
                    display_solution(
                        f"**Formula:** n = log(0.80)/log(2) = {n1:.4f}\n\n"
                        f"Y₈ = 100 × 8^{n1:.4f} = **{correct:.1f} hrs**\n\n"
                        "**Doubling check:** 100 → 80 → 64 → **51.2 hrs** ✓"
                    )

        with st.expander("🟢 P2: Find Learning Rate (Easy)"):
            display_practice_problem(2, "Easy",
                "Unit 1 = 200 hrs. Unit 4 = 128 hrs. What is the learning rate?")
            if st.button("Show Answer", key="lc_p2"):
                # Y4 = K*4^n → 128 = 200*4^n → 4^n = 0.64
                # n = log(0.64)/log(4); b = 2^n
                n2 = math.log(128/200)/math.log(4)
                b2 = 2**n2
                display_solution(
                    f"4^n = 128/200 = 0.640\n\n"
                    f"n = log(0.640)/log(4) = {n2:.4f}\n\n"
                    f"b = 2^n = 2^{n2:.4f} = **{b2*100:.1f}%** learning curve"
                )

        with st.expander("🟡 P3: Cumulative Time & Bidding (Medium)"):
            display_practice_problem(3, "Medium",
                "Unit 1 = 300 hrs, 85% learning curve. A contract requires units 9–12. "
                "Estimate total labor hours for the contract.")
            if st.button("Show Answer", key="lc_p3"):
                n3    = math.log(0.85)/math.log(2)
                block = sum(300*(u**n3) for u in range(9, 13))
                display_solution(
                    f"n = log(0.85)/log(2) = {n3:.4f}\n\n"
                    "Unit times:\n"
                    + "\n".join([f"  Unit {u}: 300×{u}^{n3:.4f} = {300*(u**n3):.1f} hrs"
                                 for u in range(9, 13)])
                    + f"\n\n**Contract total = {block:.1f} hrs**"
                )

        with st.expander("🟡 P4: Cost Estimation (Medium)"):
            display_practice_problem(4, "Medium",
                "Unit 1 costs $50,000. 75% learning curve applies. Labor = $100/hr. "
                "Find unit 16 cost and cumulative cost through unit 16.")
            if st.button("Show Answer", key="lc_p4"):
                n4    = math.log(0.75)/math.log(2)
                k4    = 50000/100  # hours for unit 1
                u16   = 100*k4*(16**n4)   # $ cost
                cum16 = sum(100*k4*(u**n4) for u in range(1, 17))
                display_solution(
                    f"K = $50,000 / $100/hr = {k4:.0f} hrs\n\n"
                    f"n = log(0.75)/log(2) = {n4:.4f}\n\n"
                    f"Unit 16 hrs = {k4:.0f}×16^{n4:.4f} = {k4*(16**n4):.1f} hrs\n\n"
                    f"Unit 16 cost = ${u16:,.0f}\n\n"
                    f"**Cumulative cost (units 1–16) = ${cum16:,.0f}**"
                )

        with st.expander("🔴 P5: Learning Rate from Two Data Points (Hard)"):
            display_practice_problem(5, "Hard",
                "Unit 3 took 180 hrs. Unit 12 took 108 hrs. "
                "Find the learning rate and estimate unit 48 time.")
            if st.button("Show Answer", key="lc_p5"):
                # Y3 = K*3^n = 180; Y12 = K*12^n = 108
                # Y12/Y3 = (12/3)^n = 108/180 → 4^n = 0.6
                n5  = math.log(108/180) / math.log(12/3)
                b5  = 2**n5
                K5  = 180 / (3**n5)
                y48 = K5 * (48**n5)
                display_solution(
                    f"Y₁₂/Y₃ = (12/3)^n → 4^n = 108/180 = 0.600\n\n"
                    f"n = log(0.600)/log(4) = {n5:.4f}\n\n"
                    f"Learning rate b = 2^{n5:.4f} = **{b5*100:.1f}%**\n\n"
                    f"K = 180/3^{n5:.4f} = {K5:.1f} hrs\n\n"
                    f"Unit 48: Y₄₈ = {K5:.1f}×48^{n5:.4f} = **{y48:.1f} hrs**"
                )


# ============================================================
# MODULE 7: DECOUPLING POINT (Chapter 7) - ENHANCED V5.0
# ============================================================
def module_decoupling():
    display_header("🔀", "Chapter 7", "Customer Order Decoupling Point",
                   "Positioning inventory to decouple forecast-driven from order-driven operations")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Configuration Calculator", "📊 Strategy Selector"])

    with tab1:
        st.markdown("### Customer Order Decoupling Point (CODP)")
        st.write(
            "The **CODP** is the inventory point that separates the forecast-driven upstream "
            "supply chain from the customer-order-driven downstream. Positioning the CODP "
            "trades off lead-time responsiveness against inventory investment and forecast risk."
        )

        st.latex(r"\text{Configurations} = \prod_{i=1}^{n} N_i = N_1 \times N_2 \times \cdots \times N_n")

        st.markdown("#### The Four Manufacturing Strategies")
        strat_df = pd.DataFrame({
            "Strategy":         ["Make-to-Stock (MTS)", "Assemble-to-Order (ATO)",
                                 "Make-to-Order (MTO)", "Engineer-to-Order (ETO)"],
            "CODP Position":    ["Finished goods inventory", "Sub-assembly / module stock",
                                 "Raw material / purchased parts", "No pre-positioned inventory"],
            "Lead Time":        ["Immediate (from shelf)", "Hours to days",
                                 "Weeks to months", "Months to years"],
            "Customization":    ["None / standard", "Moderate (config from modules)",
                                 "High (any spec)", "Full custom design"],
            "Forecast Risk":    ["Very high", "Moderate", "Low", "Near zero"],
            "Inventory Level":  ["Very high FG", "Moderate WIP", "Low RM only", "Minimal"],
            "Example":          ["Supermarket goods", "Dell PCs, cars",
                                 "Boeing 737", "Offshore oil platform"]
        })
        st.dataframe(strat_df, use_container_width=True)

        st.markdown("#### Supply Chain Positioning Map")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_concept_card("📦", "MTS",
                                 "All production before order. "
                                 "Lowest lead time, highest inventory. "
                                 "Works for stable, predictable demand.")
        with col2:
            display_concept_card("🔧", "ATO",
                                 "Standard modules pre-built; "
                                 "configured on order. "
                                 "Balance of speed & variety (mass customization).")
        with col3:
            display_concept_card("📐", "MTO",
                                 "Raw material stocked; "
                                 "production starts on receipt of order. "
                                 "Customer specifies product.")
        with col4:
            display_concept_card("✏️", "ETO",
                                 "Design starts from customer specs. "
                                 "Longest lead time; infinite variety. "
                                 "Aerospace, defense, custom machinery.")

        display_key_insight(
            "Mass Customization via ATO",
            "A product with 5 option categories each having 4 choices yields "
            "4⁵ = 1,024 unique configurations from only 20 components. "
            "ATO achieves MTS speed with near-ETO variety."
        )

    with tab2:
        st.markdown("### Configuration & Variety Calculator")
        st.write("Quantify the product variety achievable from modular design.")

        num_opts = st.number_input("Number of Option Categories", 1, 10, 5)

        default_labels  = ["Processor", "Memory", "Storage", "Display", "Color",
                           "OS", "Keyboard", "GPU", "Battery", "Warranty"]
        default_choices = [3, 4, 4, 2, 5, 2, 3, 3, 2, 2]

        option_data = []
        col1, col2 = st.columns(2)
        for i in range(int(num_opts)):
            with (col1 if i % 2 == 0 else col2):
                cc = st.columns(2)
                with cc[0]:
                    label = st.text_input(f"Category {i+1}",
                                          value=default_labels[i] if i < len(default_labels) else f"Option {i+1}",
                                          key=f"dc_label_{i}")
                with cc[1]:
                    choices = st.number_input(f"# Choices",
                                              value=default_choices[i] if i < len(default_choices) else 2,
                                              min_value=1, key=f"dc_ch_{i}")
                option_data.append({"Category": label, "Choices": choices})

        total_components  = sum(d["Choices"] for d in option_data)
        total_configs     = 1
        for d in option_data:
            total_configs *= d["Choices"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Component SKUs",  f"{total_components:,}")
        col2.metric("Total Configurations",  f"{total_configs:,}")
        col3.metric("Config / Component",    f"{total_configs/total_components:.0f}×")

        st.success(
            f"💡 Only **{total_components}** component SKUs create **{total_configs:,}** "
            f"unique product configurations — a {total_configs/total_components:.0f}× "
            "variety multiplier from modular design!"
        )

        # Variety table
        opt_df = pd.DataFrame(option_data)
        opt_df["Running Product"] = opt_df["Choices"].cumprod()
        opt_df["Marginal Configs Added"] = opt_df["Running Product"] - opt_df["Running Product"].shift(1).fillna(1)
        st.dataframe(opt_df, use_container_width=True)

    with tab3:
        st.markdown("### Strategy Selection Guide")
        st.write("Rate your product/market on key dimensions to identify the optimal CODP strategy.")

        col1, col2 = st.columns(2)
        with col1:
            demand_pred   = st.slider("Demand Predictability (1=Unpredictable, 5=Stable)", 1, 5, 3)
            customization = st.slider("Required Customization (1=None, 5=Full custom)",     1, 5, 2)
            lead_sens     = st.slider("Customer Lead-Time Sensitivity (1=Not, 5=Critical)", 1, 5, 4)
            vol           = st.slider("Production Volume (1=Low/unique, 5=High/commodity)", 1, 5, 3)

        with col2:
            mts_score = demand_pred * 2 + (6-customization)*1.5 + lead_sens*2   + vol*1.5
            ato_score = demand_pred * 1 + (6-customization)*2   + lead_sens*1.5 + vol*1
            mto_score = (6-demand_pred)*1.5 + customization*2   + (6-lead_sens)*1 + (6-vol)*1
            eto_score = (6-demand_pred)*1   + customization*2.5 + (6-lead_sens)*0.5 + (6-vol)*1.5

            scores = {"MTS": mts_score, "ATO": ato_score, "MTO": mto_score, "ETO": eto_score}
            best   = max(scores, key=scores.get)

            score_df = pd.DataFrame({
                "Strategy": list(scores.keys()),
                "Score":    [f"{v:.1f}" for v in scores.values()],
                "Match":    ["✅ Best Fit" if k == best else "" for k in scores.keys()]
            })
            st.dataframe(score_df, use_container_width=True)
            st.success(f"📍 **Recommended Strategy: {best}**")

            fig = go.Figure(go.Bar(
                x=list(scores.keys()), y=list(scores.values()),
                marker_color=["#2ecc71" if k == best else "#3498db" for k in scores.keys()]
            ))
            fig.update_layout(title="Strategy Fit Scores", yaxis_title="Score",
                              template="plotly_white", height=300)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODULE 8: LINE BALANCING (Chapter 8) - ENHANCED V5.0
# ============================================================
def module_linebalance():
    display_header("⚖️", "Chapter 8", "Assembly Line Balancing",
                   "Assigning tasks to workstations to minimize idle time and maximize efficiency")

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Simulator", "📊 Station Chart", "🎓 Practice"])

    with tab1:
        st.markdown("### Line Balancing Fundamentals")
        st.write(
            "**Assembly line balancing** assigns a set of tasks to workstations such that "
            "each station's total time does not exceed the cycle time, precedence constraints "
            "are respected, and the number of stations (or idle time) is minimized."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            display_formula_card("Cycle Time",
                                 r"C = \frac{\text{Production time/day}}{\text{Required output/day}}")
        with col2:
            display_formula_card("Min Workstations",
                                 r"N_{min} = \left\lceil \frac{\sum t_i}{C} \right\rceil")
        with col3:
            display_formula_card("Line Efficiency",
                                 r"\eta = \frac{\sum t_i}{N_{actual} \times C} \times 100\%")

        display_formula_card("Balance Delay",
                             r"BD = 100\% - \eta = \frac{N_{actual} \cdot C - \sum t_i}{N_{actual} \cdot C} \times 100\%")

        st.markdown("#### Heuristic Assignment Rules")
        heuristic_df = pd.DataFrame({
            "Rule":        ["Longest Task Time", "Most Following Tasks",
                            "Ranked Positional Weight", "Shortest Task Time"],
            "Priority":    ["Assign longest eligible task first",
                            "Assign task with most successors first",
                            "Sum of task + all following task times",
                            "Assign shortest eligible task first (fill gaps)"],
            "Best For":    ["Reduces bottleneck risk", "Clears precedence bottlenecks",
                            "Best overall heuristic", "Minimizing balance delay at end"]
        })
        st.dataframe(heuristic_df, use_container_width=True)

        display_key_insight(
            "Theoretical Minimum vs. Practical",
            "N_min is a lower bound — it may not be achievable due to precedence constraints. "
            "The real goal is to get as close to N_min as possible while keeping each "
            "workstation at or below cycle time C."
        )

    with tab2:
        st.markdown("### Line Balancing Calculator")

        col1, col2 = st.columns(2)
        with col1:
            prod_time    = st.number_input("Production Time per Day (sec)", value=28800)
            output_req   = st.number_input("Required Output per Day (units)", value=480)
            num_stations = st.number_input("Actual Number of Workstations", value=5, min_value=1)
            n_tasks      = st.number_input("Number of Tasks", value=8, min_value=1, max_value=20)

        with col2:
            if output_req > 0:
                cycle_time = prod_time / output_req
                st.metric("Cycle Time (C)", f"{cycle_time:.1f} sec")
                st.metric("Max Output Possible", f"{prod_time/cycle_time:.0f} units/day",
                          help="If C is exactly achieved at every station")

        # ─── Task Entry ───
        st.markdown("#### Task Times")
        default_times = [12, 15, 8, 20, 18, 10, 14, 16, 9, 11, 13, 7, 19, 17, 6, 21, 8, 15, 12, 10]
        task_times = []
        task_names = []
        t_cols = st.columns(min(int(n_tasks), 5))
        for i in range(int(n_tasks)):
            with t_cols[i % 5]:
                t = st.number_input(f"Task {chr(65+i)}",
                                    value=default_times[i] if i < len(default_times) else 10,
                                    key=f"lb_t_{i}", min_value=0)
                task_times.append(t)
                task_names.append(chr(65+i))

        sum_task = sum(task_times)

        if output_req > 0 and cycle_time > 0:
            n_min      = math.ceil(sum_task / cycle_time)
            efficiency = (sum_task / (num_stations * cycle_time)) * 100
            bd         = 100 - efficiency
            total_idle = num_stations * cycle_time - sum_task
            idle_per_s = total_idle / num_stations

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sum of Task Times",     f"{sum_task} sec")
            col2.metric("Min Workstations (N_min)", f"{n_min}")
            col3.metric("Line Efficiency",       f"{efficiency:.1f}%")
            col4.metric("Balance Delay",         f"{bd:.1f}%")
            st.metric("Total Idle Time/Cycle",   f"{total_idle:.1f} sec")
            st.metric("Avg Idle per Station",    f"{idle_per_s:.1f} sec")

            if n_min == int(num_stations):
                st.success("✅ Achieved theoretical minimum workstations!")
            elif int(num_stations) <= n_min + 1:
                st.info(f"💡 One above minimum — consider if task precedence prevents N_min={n_min}")
            else:
                st.warning(f"⚠️ {int(num_stations)-n_min} stations above minimum — rebalance to improve.")

            # ─── Manual Assignment Table ───
            st.markdown("#### Assign Tasks to Stations")
            st.write("Enter which tasks go to each station (comma-separated letters):")
            assignments = {}
            assign_cols = st.columns(min(int(num_stations), 5))
            for s_idx in range(int(num_stations)):
                with assign_cols[s_idx % 5]:
                    default_assign = ""
                    if s_idx == 0: default_assign = "A,B"
                    elif s_idx == 1: default_assign = "C,D"
                    elif s_idx == 2: default_assign = "E,F"
                    elif s_idx == 3: default_assign = "G,H"
                    assignment = st.text_input(f"Station {s_idx+1}",
                                               value=default_assign if s_idx < 4 else "",
                                               key=f"lb_assign_{s_idx}")
                    assignments[f"S{s_idx+1}"] = assignment

            assign_results = []
            for s_name, task_str in assignments.items():
                tasks_in_s = [t.strip().upper() for t in task_str.split(",") if t.strip()]
                s_time = sum(task_times[task_names.index(t)] for t in tasks_in_s if t in task_names)
                idle   = cycle_time - s_time
                assign_results.append({
                    "Station": s_name,
                    "Tasks": ", ".join(tasks_in_s),
                    "Station Time (sec)": round(s_time, 1),
                    "Idle Time (sec)":    round(idle, 1),
                    "Utilization":        f"{s_time/cycle_time*100:.1f}%",
                    "Over CT?":           "❌ OVER!" if s_time > cycle_time else "✅ OK"
                })
            assign_df = pd.DataFrame(assign_results)
            st.dataframe(assign_df, use_container_width=True)

            # Recalculate efficiency from actual assignment
            actual_sum   = sum(r["Station Time (sec)"] for r in assign_results)
            actual_eff   = actual_sum / (num_stations * cycle_time) * 100
            assigned_all = sum(len(a["Tasks"].split(",")) for a in assign_results
                               if a["Tasks"].strip())
            st.metric("Actual Efficiency (from assignment)", f"{actual_eff:.1f}%")

    with tab3:
        st.markdown("### Workstation Utilization Chart")

        if 'assign_results' in dir() and assign_results:
            station_names = [r["Station"] for r in assign_results]
            station_times = [r["Station Time (sec)"] for r in assign_results]
            idles         = [r["Idle Time (sec)"] for r in assign_results]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Work Time", x=station_names, y=station_times,
                                 marker_color="#3498db"))
            fig.add_trace(go.Bar(name="Idle Time", x=station_names, y=idles,
                                 marker_color="#ecf0f1"))
            fig.add_hline(y=cycle_time, line_dash="dash", line_color="red",
                          annotation_text=f"Cycle Time = {cycle_time:.1f}s", annotation_position="right")
            fig.update_layout(barmode="stack", title="Station Loading vs. Cycle Time",
                              xaxis_title="Workstation", yaxis_title="Time (sec)",
                              template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Efficiency gauge
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=efficiency,
                delta={"reference": 100},
                title={"text": "Line Efficiency (%)"},
                gauge={"axis":    {"range": [0, 100]},
                       "bar":     {"color": "#2ecc71" if efficiency >= 90 else
                                             "#f39c12" if efficiency >= 75 else "#e74c3c"},
                       "steps":   [{"range": [0, 75],   "color": "rgba(231,76,60,0.2)"},
                                   {"range": [75, 90],  "color": "rgba(243,156,18,0.2)"},
                                   {"range": [90, 100], "color": "rgba(46,204,113,0.2)"}],
                       "threshold": {"line": {"color": "darkblue", "width": 3},
                                     "thickness": 0.75, "value": 95}}
            ))
            fig2.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Complete the Simulator tab first to generate charts.")

    with tab4:
        st.markdown("### 📝 Line Balancing Practice")

        with st.expander("🟢 P1: Cycle Time & N_min (Easy)"):
            display_practice_problem(1, "Easy",
                "Production time = 480 min/day, Output required = 120 units/day, "
                "Σtᵢ = 45 min. Find C, N_min.")
            if st.button("Show Answer", key="lb_p1"):
                C_p1   = 480/120
                Nmin_p1 = math.ceil(45/C_p1)
                display_solution(
                    f"C = 480/120 = **{C_p1} min/unit**\n\n"
                    f"N_min = ⌈45/{C_p1}⌉ = ⌈{45/C_p1:.2f}⌉ = **{Nmin_p1} workstations**"
                )

        with st.expander("🟡 P2: Efficiency & Balance Delay (Medium)"):
            display_practice_problem(2, "Medium",
                "C = 60 sec, 5 workstations used. Station times: 55, 58, 50, 60, 42 sec. "
                "Calculate efficiency and balance delay.")
            if st.button("Show Answer", key="lb_p2"):
                times_p2 = [55, 58, 50, 60, 42]
                sum_p2   = sum(times_p2)
                eff_p2   = sum_p2/(5*60)*100
                bd_p2    = 100 - eff_p2
                idle_p2  = [60-t for t in times_p2]
                display_solution(
                    f"Σtᵢ = {sum_p2} sec\n\n"
                    f"Efficiency = {sum_p2}/(5×60) = **{eff_p2:.1f}%**\n\n"
                    f"Balance Delay = **{bd_p2:.1f}%**\n\n"
                    f"Idle times: {idle_p2} → Total idle = {sum(idle_p2)} sec/cycle\n\n"
                    f"Bottleneck: Station 4 (60 sec = full CT, zero slack)"
                )

        with st.expander("🔴 P3: Full Balance Problem (Hard)"):
            display_practice_problem(3, "Hard",
                "8 tasks with times A=10, B=11, C=5, D=4, E=12, F=3, G=7, H=11 sec. "
                "C = 15 sec. Assign tasks using longest-time heuristic. "
                "Calculate efficiency.")
            if st.button("Show Answer", key="lb_p3"):
                tasks_p3 = {"A":10,"B":11,"C":5,"D":4,"E":12,"F":3,"G":7,"H":11}
                sorted_t  = sorted(tasks_p3.items(), key=lambda x: -x[1])
                sum_p3    = sum(tasks_p3.values())
                Nmin_p3   = math.ceil(sum_p3/15)
                display_solution(
                    f"Tasks ranked: {sorted_t}\n\n"
                    f"Σtᵢ = {sum_p3} sec, N_min = ⌈{sum_p3}/15⌉ = {Nmin_p3}\n\n"
                    "Assignment (longest first, fill to CT=15):\n"
                    "  Station 1: E(12) + C(5)? → 12+5=17 > 15; E(12) + F(3) = 15 ✅\n"
                    "  Station 2: B(11) + D(4) = 15 ✅\n"
                    "  Station 3: H(11) + C(5)? = 16 > 15; H(11) alone = 11\n"
                    "  Station 4: G(7) + C(5) = 12; + A(10)? No; G+C = 12\n"
                    "  Station 5: A(10) remaining\n\n"
                    f"(Actual assignments depend on precedence — this illustrates the heuristic)\n\n"
                    f"**Min N = {Nmin_p3}, Target eff. = {sum_p3/(Nmin_p3*15)*100:.1f}%**"
                )


# ============================================================
# MODULE 9: SERVICE DESIGN (Chapter 9) - ENHANCED V5.0
# ============================================================
def module_service():
    display_header("🎯", "Chapter 9", "Service Process Design",
                   "Designing responsive, efficient customer-centric service delivery systems")

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔺 Service Triangle",
                                       "📋 Blueprinting", "📊 Contact Matrix"])

    with tab1:
        st.markdown("### Service Design Principles")
        st.write(
            "**Service design** differs fundamentally from product design: the process and the "
            "product are inseparable. Services are simultaneously produced and consumed, cannot "
            "be inventoried, and require customer participation in delivery."
        )

        display_citation(
            "The process and the product must be developed simultaneously; indeed, in services, "
            "the process is the product.",
            "Jacobs & Chase (2024, p. 229)"
        )

        st.markdown("#### Key Service Characteristics (IHIP)")
        ihip = pd.DataFrame({
            "Characteristic": ["Intangibility", "Heterogeneity", "Inseparability", "Perishability"],
            "Meaning":        ["Cannot be seen/touched before purchase",
                               "Output varies by provider, customer, time",
                               "Production and consumption simultaneous",
                               "Unused capacity cannot be stored"],
            "Design Implication": ["Use tangible cues; standardize evidence",
                                   "Train staff; use scripts/checklists",
                                   "Co-produce with customers; manage front office",
                                   "Yield management; demand smoothing"]
        })
        st.dataframe(ihip, use_container_width=True)

        st.markdown("#### Service-System Design Matrix")
        design_matrix = pd.DataFrame({
            "System Type":     ["Mail contact", "On-site technology", "Phone contact",
                                "Face-to-face (tight specs)", "Face-to-face (loose specs)",
                                "Face-to-face (total customization)"],
            "Customer Contact":["Very low", "Low", "Medium", "Medium-High", "High", "Very High"],
            "Efficiency":      ["Very High", "High", "Medium", "Medium", "Low", "Very Low"],
            "Sales Opportunity":["Very Low", "Low", "Medium", "Medium", "High", "Very High"],
            "Worker Skills":   ["Clerical", "Technical", "Procedural",
                                "Service scripted", "Service diagnostic", "Professional judgment"]
        })
        st.dataframe(design_matrix, use_container_width=True)

        display_key_insight(
            "Contact–Efficiency Tradeoff",
            "Higher customer contact increases the sales opportunity and customization "
            "potential, but reduces efficiency because customer involvement introduces "
            "variability into the production process."
        )

    with tab2:
        st.markdown("### The Service Triangle")
        st.write(
            "The **Service Triangle** frames the three forces that must align to deliver "
            "consistent, high-quality service: Strategy, Systems, and People — "
            "all centered on the Customer."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🎯", "Service Strategy",
                                 "Defines what the organization stands for and promises customers. "
                                 "Must be clear, communicated, and consistently upheld.")
        with col2:
            display_concept_card("👤", "Customer (Center)",
                                 "The reason for the system. All triangle elements should "
                                 "orient toward delivering the customer value proposition.")
        with col3:
            display_concept_card("⚙️", "Systems",
                                 "Technology, procedures, and physical layout that allow "
                                 "people to deliver the strategy. Must not hinder service.")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("👥", "People",
                                 "Frontline employees are the service. Their attitude, skills, "
                                 "and empowerment determine the customer experience.")
        with col2:
            display_concept_card("🔁", "Strategy ↔ People",
                                 "Internal marketing: employees must understand and believe "
                                 "in the service strategy to deliver it authentically.")
        with col3:
            display_concept_card("🔧", "Systems ↔ People",
                                 "Systems should empower, not constrain, people. Poor systems "
                                 "create workarounds that erode service quality.")

        st.markdown("#### Service Triangle Self-Assessment")
        st.write("Rate alignment on each dimension (1=Poor, 5=Excellent):")
        col1, col2 = st.columns(2)
        with col1:
            strat_score  = st.slider("Strategy clarity & communication",    1, 5, 4)
            system_score = st.slider("Systems support service delivery",     1, 5, 3)
            people_score = st.slider("People trained & empowered",           1, 5, 4)
        with col2:
            avg_score = (strat_score + system_score + people_score) / 3
            st.metric("Triangle Balance Score", f"{avg_score:.1f}/5.0")
            weakest = min(["Strategy", "Systems", "People"],
                          key=lambda x: {"Strategy": strat_score,
                                         "Systems":  system_score,
                                         "People":   people_score}[x])
            st.warning(f"⚠️ Weakest link: **{weakest}** — focus improvement here first.")

            fig = go.Figure(go.Scatterpolar(
                r=[strat_score, system_score, people_score, strat_score],
                theta=["Strategy", "Systems", "People", "Strategy"],
                fill="toself", line_color="#3498db", fillcolor="rgba(52,152,219,0.3)"
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5])),
                              title="Service Triangle Assessment",
                              template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Service Blueprinting")
        st.write(
            "A **service blueprint** maps the complete service delivery process — "
            "customer actions, employee interactions, backstage activities, and support "
            "processes — separated by lines of visibility and interaction."
        )

        st.markdown("#### Blueprint Anatomy")
        anatomy = pd.DataFrame({
            "Layer":      ["Customer Actions",
                           "Line of Interaction",
                           "Onstage (Visible) Contact",
                           "Line of Visibility",
                           "Backstage (Invisible) Contact",
                           "Line of Internal Interaction",
                           "Support Processes"],
            "Who/What":   ["What customers do to receive the service",
                           "─── Direct customer-employee contact ───",
                           "Employee actions customer can observe",
                           "─── Visibility boundary ───",
                           "Employee actions customer cannot see",
                           "─── Internal operational boundary ───",
                           "IT systems, back-office, suppliers"]
        })
        st.dataframe(anatomy, use_container_width=True)

        st.markdown("#### Blueprint Builder — Choose Service Type")
        service_type = st.selectbox("Service Template",
                                    ["Restaurant", "Hotel Check-In", "Bank Branch",
                                     "Emergency Room", "Custom (blank)"])

        blueprints = {
            "Restaurant": {
                "steps":  ["Arrive & Seated", "Order Taken", "Wait for Food",
                           "Food Served", "Eat", "Bill Requested", "Pay & Leave"],
                "customer": ["Walk in; wait to be seated", "Review menu; place order",
                             "Wait; perhaps order drinks", "Receive dishes",
                             "Consume meal", "Request check", "Pay; leave tip; exit"],
                "onstage":  ["Greet; escort to table", "Take order; suggest items",
                             "Refill drinks; check satisfaction", "Serve plates; confirm order",
                             "Periodic table checks", "Present itemized check",
                             "Process payment; thank"],
                "backstage":["Reserve table in system", "Transmit order to kitchen",
                             "Kitchen prepares food", "Plate and garnish",
                             "Monitor table progress", "Print check from POS",
                             "Close ticket; turn table"],
                "support":  ["Reservations system", "POS; kitchen display",
                             "Inventory; purchasing", "Supply chain",
                             "Scheduling", "Accounting; POS", "CRM; loyalty"]
            },
            "Hotel Check-In": {
                "steps":    ["Arrive", "Queue", "Check In", "Get Key", "Go to Room", "Use Amenities", "Check Out"],
                "customer": ["Drive/walk to front desk", "Wait in line", "Provide ID/booking",
                             "Receive key card", "Navigate to room", "Use pool/restaurant",
                             "Return key; depart"],
                "onstage":  ["Valet/doorman greet", "Agent acknowledges wait",
                             "Verify booking; upsell", "Explain amenities",
                             "Bellhop assistance", "Concierge recommendations",
                             "Process checkout; handle issues"],
                "backstage":["Parking management", "Queue management screen",
                             "PMS lookup; room assignment", "Key programming",
                             "Housekeeping status", "F&B prep; maintenance",
                             "Billing reconciliation"],
                "support":  ["Valet system", "Queue display", "PMS/POS",
                             "Key card system", "Housekeeping app",
                             "HVAC/maintenance", "Accounting/ERP"]
            }
        }

        if service_type in blueprints:
            bp = blueprints[service_type]
            bp_df = pd.DataFrame({
                "Process Step":       bp["steps"],
                "Customer Action":    bp["customer"],
                "Onstage (Visible)":  bp["onstage"],
                "Backstage (Hidden)": bp["backstage"],
                "Support Process":    bp["support"]
            })
            st.dataframe(bp_df, use_container_width=True)
        else:
            st.markdown("#### Custom Blueprint")
            n_steps = st.number_input("Number of Process Steps", 3, 10, 5)
            rows = []
            for i in range(int(n_steps)):
                rc = st.columns(5)
                step_name   = rc[0].text_input(f"Step {i+1}", value=f"Step {i+1}", key=f"bp_s_{i}")
                cust_action = rc[1].text_input(f"Customer",   value="",            key=f"bp_c_{i}")
                onstage     = rc[2].text_input(f"Onstage",    value="",            key=f"bp_on_{i}")
                backstage   = rc[3].text_input(f"Backstage",  value="",            key=f"bp_bs_{i}")
                support     = rc[4].text_input(f"Support",    value="",            key=f"bp_sp_{i}")
                rows.append({"Step": step_name, "Customer": cust_action,
                             "Onstage": onstage, "Backstage": backstage, "Support": support})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("#### Fail Point Analysis")
        st.write("Mark steps where failure most likely occurs:")
        fail_points = st.multiselect("High-Risk Steps (Fail Points)",
                                     bp["steps"] if service_type in blueprints else [f"Step {i+1}" for i in range(5)])
        if fail_points:
            st.error(f"⚠️ Fail points identified: {', '.join(fail_points)}. "
                     "Apply poka-yoke (error-proofing) or add recovery scripts at these steps.")

    with tab4:
        st.markdown("### Service-System Design Matrix Calculator")
        st.write("Score a service encounter on contact level and efficiency tradeoffs.")

        col1, col2 = st.columns(2)
        with col1:
            contact_pct  = st.slider("Customer Contact Time / Service Time (%)", 0, 100, 40,
                                     help="% of service time spent in direct customer contact")
            customization = st.slider("Degree of Customization (%)", 0, 100, 50)
            labor_pct     = st.slider("Labor Intensity (%)", 0, 100, 60,
                                      help="Labor cost as % of total service cost")

        with col2:
            # Service process matrix quadrant
            if contact_pct + customization < 100:
                quadrant = "Service Factory"
                example  = "Airlines, trucking, hotels, resorts"
                strategy = "Volume; efficiency; standardization"
            elif contact_pct + customization < 150:
                quadrant = "Service Shop"
                example  = "Hospitals, auto repair, other repair"
                strategy = "Customized service with standard procedures"
            elif labor_pct < 50:
                quadrant = "Mass Service"
                example  = "Retailing, wholesaling, schools"
                strategy = "Manage people; reduce labor cost"
            else:
                quadrant = "Professional Service"
                example  = "Doctors, lawyers, accountants, architects"
                strategy = "Fight cost increases; maintain quality"

            st.metric("Service Process Type", quadrant)
            st.info(f"**Examples:** {example}\n\n**Strategy:** {strategy}")
            st.metric("Contact × Customization Index",
                      f"{(contact_pct * customization)/100:.0f}",
                      help="Higher = more complex service management challenge")


# ============================================================
# MODULE 10: QUEUING THEORY (Chapter 10) - ENHANCED V5.0
# ============================================================
def module_queuing():
    display_header("👥", "Chapter 10", "Queuing Theory — Waiting Line Models",
                   "Analyzing the tradeoff between service capacity and customer waiting")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📚 Theory", "🔬 M/M/1", "👥 M/M/s", "💰 Cost Analysis", "🎓 Practice"])

    with tab1:
        st.markdown("### Waiting Line Theory")
        st.write(
            "**Queuing theory** provides mathematical models to analyze waiting lines, "
            "helping managers balance service capacity costs against customer waiting costs. "
            "Real queuing systems consist of an arrival process, a service mechanism, "
            "and a queue discipline."
        )

        display_citation(
            "Queuing theory enables us to analyze the relationship between demand on a service "
            "system and the delays suffered by users of that system.",
            "Jacobs & Chase (2024, p. 286)"
        )

        st.markdown("#### Kendall's Notation: A/B/s/K/N/D")
        kendall_df = pd.DataFrame({
            "Symbol":  ["A", "B", "s", "K", "N", "D"],
            "Meaning": ["Arrival distribution", "Service time distribution",
                        "Number of servers", "System capacity (max in system)",
                        "Population size", "Queue discipline"],
            "Common":  ["M = Poisson/Memoryless", "M = Exponential", "1, 2, 3…",
                        "∞ (default)", "∞ (default)", "FIFO (default)"]
        })
        st.dataframe(kendall_df, use_container_width=True)

        st.markdown("#### M/M/1 Formulas")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_formula_card("Utilization",       r"\rho = \frac{\lambda}{\mu}")
            display_formula_card("Avg in System",     r"L_s = \frac{\lambda}{\mu - \lambda}")
        with col2:
            display_formula_card("Avg in Queue",      r"L_q = \frac{\lambda^2}{\mu(\mu-\lambda)}")
            display_formula_card("Avg Time in System",r"W_s = \frac{1}{\mu - \lambda}")
        with col3:
            display_formula_card("Avg Wait in Queue", r"W_q = \frac{\lambda}{\mu(\mu-\lambda)}")
            display_formula_card("Prob. System Empty",r"P_0 = 1 - \rho")

        display_formula_card("Little's Law (connects all metrics)",
                             r"L = \lambda \cdot W \quad \Rightarrow \quad L_s = \lambda W_s \;\;\; L_q = \lambda W_q")

        display_key_insight(
            "Nonlinear Congestion Effect",
            "Lq grows as ρ²/(1−ρ). At ρ = 0.5, Lq ≈ 0.5. At ρ = 0.9, Lq ≈ 8.1. "
            "At ρ = 0.95, Lq ≈ 18. The queue explodes near full utilization — "
            "100% utilization is never sustainable."
        )

    with tab2:
        st.markdown("### M/M/1 Single Server Queue")

        col1, col2 = st.columns(2)
        with col1:
            lam = st.slider("Arrival Rate (λ) per hour", 1, 50, 10)
            mu  = st.slider("Service Rate (μ) per hour", 1, 60, 15)
            hrs = st.number_input("Operating Hours/Day", value=8.0, key="mm1_hrs")

        with col2:
            if lam < mu:
                rho = lam / mu
                Ls  = lam / (mu - lam)
                Lq  = lam**2 / (mu * (mu - lam))
                Ws  = 1 / (mu - lam)
                Wq  = lam / (mu * (mu - lam))
                P0  = 1 - rho

                # Prob of n or more in system
                Pn_gt_5 = rho**6  # P(n ≥ 5) for M/M/1

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Utilization (ρ)",         f"{rho:.1%}")
                    st.metric("Avg in System (Ls)",       f"{Ls:.3f}")
                    st.metric("Avg in Queue (Lq)",        f"{Lq:.3f}")
                    st.metric("P(System Empty)",          f"{P0:.1%}")
                with col_b:
                    st.metric("Avg Time in System (Ws)",  f"{Ws*60:.2f} min")
                    st.metric("Avg Wait in Queue (Wq)",   f"{Wq*60:.2f} min")
                    st.metric("Daily Customers Served",   f"{int(lam*hrs):,}")
                    st.metric("P(≥ 5 in system)",         f"{Pn_gt_5:.1%}")

                # Utilization gauge
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=rho*100,
                    title={"text": "Server Utilization (%)"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar":  {"color": "#e74c3c" if rho > 0.85 else
                                              "#f39c12" if rho > 0.70 else "#2ecc71"},
                           "steps": [{"range":[0,70],  "color":"rgba(46,204,113,0.2)"},
                                     {"range":[70,85], "color":"rgba(243,156,18,0.2)"},
                                     {"range":[85,100],"color":"rgba(231,76,60,0.2)"}]}
                ))
                fig_g.update_layout(height=280, template="plotly_white")
                st.plotly_chart(fig_g, use_container_width=True)

                if rho >= 0.9:
                    st.error(f"⚠️ ρ = {rho:.1%} — dangerously high. Queue will explode; add capacity.")
                elif rho >= 0.75:
                    st.warning(f"⚠️ ρ = {rho:.1%} — heavy utilization; monitor closely.")
                else:
                    st.success(f"✅ ρ = {rho:.1%} — manageable utilization.")
            else:
                st.error("⚠️ Unstable system! λ must be < μ for steady state.")

        # ─── Sensitivity to ρ ───
        st.markdown("#### Utilization Sensitivity Table")
        util_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        sens_df = pd.DataFrame({
            "ρ":    [f"{r:.0%}" for r in util_range],
            "Lq":   [f"{r**2/(1-r):.3f}" for r in util_range],
            "Wq (×1/μ)": [f"{r/(1-r):.3f}" for r in util_range],
            "Ls":   [f"{r/(1-r):.3f}" for r in util_range],
        })
        st.dataframe(sens_df, use_container_width=True)

    with tab3:
        st.markdown("### M/M/s Multi-Server Queue")

        col1, col2 = st.columns(2)
        with col1:
            lam_s = st.number_input("Arrival Rate (λ)", value=15.0, key="mms_lam")
            mu_s  = st.number_input("Service Rate per Server (μ)", value=6.0, key="mms_mu")
            s     = st.number_input("Number of Servers (s)", value=3, min_value=1, key="mms_s")

        with col2:
            if lam_s < s * mu_s:
                rho_s    = lam_s / (s * mu_s)
                r        = lam_s / mu_s  # traffic intensity

                # P0 for M/M/s
                sum_term = sum(r**n / math.factorial(n) for n in range(int(s)))
                last_term = r**s / (math.factorial(int(s)) * (1 - rho_s))
                P0_s     = 1 / (sum_term + last_term)

                Lq_s     = (P0_s * r**s * rho_s) / (math.factorial(int(s)) * (1 - rho_s)**2)
                Wq_s     = Lq_s / lam_s
                Ws_s     = Wq_s + 1/mu_s
                Ls_s     = lam_s * Ws_s
                P_wait   = (P0_s * r**s) / (math.factorial(int(s)) * (1 - rho_s))

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Server Utilization (ρ)",   f"{rho_s:.1%}")
                    st.metric("Avg in Queue (Lq)",        f"{Lq_s:.4f}")
                    st.metric("Avg Wait in Queue (Wq)",   f"{Wq_s*60:.3f} min")
                with col_b:
                    st.metric("Avg in System (Ls)",       f"{Ls_s:.4f}")
                    st.metric("Avg Time in System (Ws)",  f"{Ws_s*60:.3f} min")
                    st.metric("P(Must Wait)",             f"{P_wait:.1%}")
                    st.metric("P₀ (All servers idle)",   f"{P0_s:.1%}")

            else:
                st.error("⚠️ Unstable system: λ must be < s×μ")

        # ─── Server Comparison ───
        st.markdown("#### Server Count Comparison")
        if lam_s < 20 * mu_s:
            comp_rows = []
            for s_test in range(max(1, math.ceil(lam_s/mu_s)), min(10, math.ceil(lam_s/mu_s))+5):
                if lam_s < s_test * mu_s:
                    rho_t    = lam_s / (s_test * mu_s)
                    r_t      = lam_s / mu_s
                    sum_t    = sum(r_t**n/math.factorial(n) for n in range(s_test))
                    last_t   = r_t**s_test/(math.factorial(s_test)*(1-rho_t))
                    P0_t     = 1/(sum_t+last_t)
                    Lq_t     = (P0_t*r_t**s_test*rho_t)/(math.factorial(s_test)*(1-rho_t)**2)
                    Wq_t     = Lq_t/lam_s*60
                    comp_rows.append({
                        "Servers (s)": s_test,
                        "ρ":          f"{rho_t:.1%}",
                        "Lq":         f"{Lq_t:.4f}",
                        "Wq (min)":   f"{Wq_t:.3f}",
                        "Feasible?":  "✅" if rho_t < 0.85 else "⚠️ High"
                    })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    with tab4:
        st.markdown("### Queue Cost Analysis")
        st.write("Minimize total system cost = waiting cost + service cost.")

        display_formula_card("Total Cost",
                             r"TC = L_s \times C_w + S \times C_s")

        col1, col2 = st.columns(2)
        with col1:
            lam_c = st.number_input("Arrival Rate (λ)",          value=3.0, key="qc_lam")
            Cw    = st.number_input("Waiting Cost ($/hr/customer)",value=25.0,key="qc_cw")
            Cs    = st.number_input("Service Cost ($/hr/server)",  value=16.0,key="qc_cs")

        with col2:
            cost_scenarios = []
            for servers, mu_v in [(1,4),(1,7),(2,4),(3,4),(2,6)]:
                if lam_c < servers * mu_v:
                    r_v   = lam_c / mu_v
                    rho_v = lam_c / (servers * mu_v)
                    if servers == 1:
                        Ls_v = lam_c / (mu_v - lam_c)
                    else:
                        sum_v  = sum(r_v**n/math.factorial(n) for n in range(servers))
                        last_v = r_v**servers/(math.factorial(servers)*(1-rho_v))
                        P0_v   = 1/(sum_v+last_v)
                        Lq_v   = (P0_v*r_v**servers*rho_v)/(math.factorial(servers)*(1-rho_v)**2)
                        Ls_v   = lam_c/(mu_v-lam_c/servers) if rho_v < 1 else float("inf")
                        Ls_v   = Lq_v + lam_c/mu_v

                    wait_cost  = Ls_v * Cw
                    labor_cost = servers * Cs
                    total_cost = wait_cost + labor_cost
                    cost_scenarios.append({
                        "Scenario":  f"s={servers}, μ={mu_v}",
                        "Servers":   servers,
                        "μ":         mu_v,
                        "ρ":         f"{rho_v:.1%}",
                        "Ls":        f"{Ls_v:.3f}",
                        "Wait Cost": f"${wait_cost:.2f}",
                        "Labor Cost":f"${labor_cost:.2f}",
                        "Total Cost":f"${total_cost:.2f}",
                        "TC_num":    total_cost
                    })

            cost_df = pd.DataFrame(cost_scenarios)
            min_tc  = cost_df["TC_num"].min()
            cost_df["Optimal?"] = cost_df["TC_num"].apply(
                lambda x: "✅ Best" if x == min_tc else "")
            st.dataframe(cost_df.drop("TC_num", axis=1), use_container_width=True)

            best_row = cost_df.loc[cost_df["TC_num"].idxmin()]
            st.success(f"📊 Minimum cost: {best_row['Scenario']} at {best_row['Total Cost']}/hr")

        # Cost chart
        if 'cost_scenarios' in dir() and cost_scenarios:
            fig_c = go.Figure(data=[
                go.Bar(name="Wait Cost",  x=[s["Scenario"] for s in cost_scenarios],
                       y=[float(s["Wait Cost"].replace("$","")) for s in cost_scenarios],
                       marker_color="#e74c3c"),
                go.Bar(name="Labor Cost", x=[s["Scenario"] for s in cost_scenarios],
                       y=[float(s["Labor Cost"].replace("$","")) for s in cost_scenarios],
                       marker_color="#3498db"),
            ])
            fig_c.update_layout(barmode="stack", title="Cost Breakdown by Scenario",
                                yaxis_title="$/hour", template="plotly_white", height=360)
            st.plotly_chart(fig_c, use_container_width=True)

    with tab5:
        st.markdown("### 📝 Queuing Practice Problems")

        with st.expander("🟢 P1: M/M/1 Basic (Easy)"):
            display_practice_problem(1, "Easy",
                "λ = 12/hr, μ = 18/hr (M/M/1). Find ρ, Lq, Wq.")
            user_ans = st.number_input("Your Lq answer:", key="q_p1", format="%.4f", value=0.0)
            if st.button("Check Answer", key="q_p1_btn"):
                rho_p1 = 12/18
                Lq_p1  = 12**2/(18*(18-12))
                Wq_p1  = 12/(18*(18-12))
                if check_answer(user_ans, Lq_p1, tolerance=0.01):
                    st.success(f"✅ Correct! Lq = {Lq_p1:.4f}")
                else:
                    display_solution(
                        f"ρ = 12/18 = **{rho_p1:.4f}**\n\n"
                        f"Lq = 12²/(18×6) = 144/108 = **{Lq_p1:.4f}**\n\n"
                        f"Wq = 12/(18×6) = **{Wq_p1:.4f} hr** = {Wq_p1*60:.2f} min"
                    )

        with st.expander("🟡 P2: All M/M/1 Metrics (Medium)"):
            display_practice_problem(2, "Medium",
                "λ = 8 customers/hr, μ = 12 customers/hr (M/M/1). "
                "Find ρ, Ls, Lq, Ws, Wq, P₀, P(n ≥ 3).")
            if st.button("Show Answer", key="q_p2"):
                lam2, mu2 = 8, 12
                rho2 = lam2/mu2
                Ls2  = lam2/(mu2-lam2)
                Lq2  = lam2**2/(mu2*(mu2-lam2))
                Ws2  = 1/(mu2-lam2)
                Wq2  = lam2/(mu2*(mu2-lam2))
                P0_2 = 1-rho2
                Pn3  = rho2**3
                display_solution(
                    f"ρ = {lam2}/{mu2} = **{rho2:.4f}**\n\n"
                    f"Ls = {lam2}/{mu2-lam2} = **{Ls2:.4f}**\n\n"
                    f"Lq = {lam2}²/({mu2}×{mu2-lam2}) = **{Lq2:.4f}**\n\n"
                    f"Ws = 1/{mu2-lam2} = **{Ws2:.4f} hr** = {Ws2*60:.2f} min\n\n"
                    f"Wq = {lam2}/({mu2}×{mu2-lam2}) = **{Wq2:.4f} hr** = {Wq2*60:.2f} min\n\n"
                    f"P₀ = 1 − {rho2:.4f} = **{P0_2:.4f}**\n\n"
                    f"P(n≥3) = ρ³ = {rho2:.4f}³ = **{Pn3:.4f}**"
                )

        with st.expander("🔴 P3: Optimal Server Count (Hard)"):
            display_practice_problem(3, "Hard",
                "λ = 10/hr, μ = 4/hr/server. Waiting cost = $30/hr/customer, "
                "Server cost = $12/hr. How many servers minimize total cost?")
            if st.button("Show Answer", key="q_p3"):
                lam3, mu3, Cw3, Cs3 = 10, 4, 30, 12
                results3 = []
                for s3 in range(3, 7):
                    if lam3 < s3*mu3:
                        rho3  = lam3/(s3*mu3)
                        r3    = lam3/mu3
                        sum3  = sum(r3**n/math.factorial(n) for n in range(s3))
                        last3 = r3**s3/(math.factorial(s3)*(1-rho3))
                        P0_3  = 1/(sum3+last3)
                        Lq3   = P0_3*r3**s3*rho3/(math.factorial(s3)*(1-rho3)**2)
                        Ls3   = Lq3 + lam3/mu3
                        TC3   = Ls3*Cw3 + s3*Cs3
                        results3.append(f"  s={s3}: Ls={Ls3:.3f}, TC=${TC3:.2f}")
                display_solution(
                    f"Need s > λ/μ = {lam3}/{mu3} = {lam3/mu3:.1f} → min s = {math.ceil(lam3/mu3)+1}\n\n"
                    + "\n".join(results3)
                    + "\n\n**Select s that minimizes TC.**"
                )

# ============================================================
# MODULE 11: DISTRIBUTIONS (Chapter 10) - ENHANCED V5.0
# ============================================================
def module_distributions():
    display_header("📐", "Chapter 10", "Exponential & Poisson Distributions",
                   "Probability models underlying queuing, arrival, and service analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "📊 Exponential", "🎲 Poisson", "📈 Visual Comparison"])

    with tab1:
        st.markdown("### Probability Distributions for Operations")
        st.write(
            "Two distributions underpin virtually all queuing models: "
            "the **Poisson distribution** models the random count of arrivals in a time window, "
            "while the **Exponential distribution** models the continuous random time between events. "
            "They are mathematically linked — if arrivals are Poisson, inter-arrival times are Exponential."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Exponential Distribution")
            display_formula_card("PDF",
                                 r"f(t) = \lambda e^{-\lambda t}, \quad t \geq 0")
            display_formula_card("CDF (Survival)",
                                 r"P(T \leq t) = 1 - e^{-\lambda t}")
            display_formula_card("Mean & Std Dev",
                                 r"E[T] = \frac{1}{\lambda}, \quad \sigma_T = \frac{1}{\lambda}")
        with col2:
            st.markdown("#### Poisson Distribution")
            display_formula_card("PMF",
                                 r"P(N=n) = \frac{(\lambda T)^n \, e^{-\lambda T}}{n!}")
            display_formula_card("Mean & Variance",
                                 r"E[N] = \text{Var}[N] = \lambda T")
            display_formula_card("Cumulative",
                                 r"P(N \leq k) = \sum_{n=0}^{k} \frac{(\lambda T)^n e^{-\lambda T}}{n!}")

        st.markdown("#### Relationship Between Distributions")
        rel_df = pd.DataFrame({
            "Property": ["Arrival count in period T",
                         "Time between consecutive arrivals",
                         "Mean inter-arrival time",
                         "Coefficient of Variation (CV)",
                         "Memoryless property"],
            "Distribution": ["Poisson(λT)", "Exponential(λ)",
                              "1/λ", "CV = 1 (always)",
                              "Exponential only"],
            "Implication": ["'M' in Kendall A/B/s notation",
                             "'M' in service time B notation",
                             "λ=10/hr → mean gap = 6 min",
                             "High variability relative to mean",
                             "Past service time gives no info about remaining time"]
        })
        st.dataframe(rel_df, use_container_width=True)

        display_key_insight(
            "Memoryless Property",
            "P(T > s+t | T > s) = P(T > t). If a server has been busy for 5 min, "
            "the remaining service time has exactly the same distribution as if it just started. "
            "This is why M/M/s models are mathematically tractable — history is irrelevant."
        )

        st.markdown("#### When to Use Each Distribution")
        usage_df = pd.DataFrame({
            "Scenario": ["Customer arrivals at a store",
                         "Machine breakdowns per week",
                         "Service time at a counter",
                         "Time between breakdowns",
                         "Website hits per minute",
                         "Time between website hits"],
            "Distribution": ["Poisson", "Poisson", "Exponential", "Exponential",
                              "Poisson", "Exponential"],
            "Parameter": ["λ = avg arrivals/period", "λ = avg breakdowns/week",
                          "μ = 1/mean service time", "λ = 1/mean time between",
                          "λ = avg hits/min", "λ = avg hits/min"]
        })
        st.dataframe(usage_df, use_container_width=True)

    with tab2:
        st.markdown("### Exponential Distribution Calculator")

        col1, col2 = st.columns(2)
        with col1:
            exp_lambda = st.number_input("Rate λ (events per time unit)", value=2.0,
                                         min_value=0.01, step=0.1,
                                         help="e.g., λ=2 means 2 customers/hour served")
            exp_t      = st.number_input("Time threshold t", value=1.0,
                                         min_value=0.0, step=0.1)
            time_unit  = st.text_input("Time unit label", value="hours")
            show_table = st.checkbox("Show extended probability table", value=True)

        with col2:
            p_within = 1 - math.exp(-exp_lambda * exp_t)
            p_beyond = math.exp(-exp_lambda * exp_t)
            mean_t   = 1 / exp_lambda
            median_t = math.log(2) / exp_lambda
            std_t    = 1 / exp_lambda

            st.metric(f"P(T ≤ {exp_t} {time_unit})",  f"{p_within:.4f}  ({p_within:.1%})")
            st.metric(f"P(T > {exp_t} {time_unit})",   f"{p_beyond:.4f}  ({p_beyond:.1%})")
            st.metric(f"Mean (1/λ)",                    f"{mean_t:.4f} {time_unit}")
            st.metric(f"Median (ln2/λ)",                f"{median_t:.4f} {time_unit}")
            st.metric("Std Dev (= Mean)",               f"{std_t:.4f} {time_unit}")
            st.metric("Coefficient of Variation",       "1.000  (always)")

            st.latex(rf"P(T \leq {exp_t}) = 1 - e^{{-{exp_lambda} \times {exp_t}}} = {p_within:.4f}")

        # ─── Extended Table ───
        if show_table:
            t_max   = max(5, int(mean_t * 4))
            t_step  = mean_t / 4
            t_range = [round(t_step * i, 3) for i in range(1, 21)]
            tbl = pd.DataFrame({
                f"t ({time_unit})":       [f"{t:.3f}" for t in t_range],
                "P(T ≤ t)":              [f"{1-math.exp(-exp_lambda*t):.4f}" for t in t_range],
                "P(T > t)":              [f"{math.exp(-exp_lambda*t):.4f}"   for t in t_range],
                "Multiples of mean":     [f"{t*exp_lambda:.2f}×" for t in t_range],
                "% Completed by t":      [f"{(1-math.exp(-exp_lambda*t))*100:.1f}%" for t in t_range]
            })
            st.dataframe(tbl, use_container_width=True)

        # ─── Quantile Calculator ───
        st.markdown("#### Quantile (Inverse) Calculator")
        st.write("Find the time *t* such that P(T ≤ t) = q:")
        col1, col2 = st.columns(2)
        with col1:
            q_val = st.slider("Target Probability q", 0.05, 0.99, 0.90, 0.01)
        with col2:
            t_q = -math.log(1 - q_val) / exp_lambda
            st.metric(f"t such that P(T ≤ t) = {q_val:.0%}",
                      f"{t_q:.4f} {time_unit}")
            st.latex(rf"t = \frac{{-\ln(1-{q_val})}}{{{exp_lambda}}} = {t_q:.4f}")

    with tab3:
        st.markdown("### Poisson Distribution Calculator")

        col1, col2 = st.columns(2)
        with col1:
            poi_lambda = st.number_input("Arrival Rate λ (per unit time)", value=3.0,
                                         min_value=0.1, step=0.5)
            poi_T      = st.number_input("Observation Period T", value=1.0,
                                         min_value=0.1, step=0.5)
            poi_n      = st.number_input("Specific value n", value=3, min_value=0)
            poi_k_le   = st.number_input("Compute P(N ≤ k) for k =", value=4, min_value=0,
                                          key="poi_k")

        with col2:
            lambda_T  = poi_lambda * poi_T
            p_n       = poisson_pmf(poi_n, lambda_T)
            p_le_n    = sum(poisson_pmf(k, lambda_T) for k in range(int(poi_n)+1))
            p_gt_n    = 1 - p_le_n
            p_le_k    = sum(poisson_pmf(k, lambda_T) for k in range(int(poi_k_le)+1))
            p_gt_k    = 1 - p_le_k

            st.metric("λT  (mean arrivals)",         f"{lambda_T:.3f}")
            st.metric(f"P(N = {poi_n})",             f"{p_n:.5f}  ({p_n:.2%})")
            st.metric(f"P(N ≤ {poi_n})",             f"{p_le_n:.5f}  ({p_le_n:.2%})")
            st.metric(f"P(N > {poi_n})",             f"{p_gt_n:.5f}  ({p_gt_n:.2%})")
            st.metric(f"P(N ≤ {poi_k_le})",          f"{p_le_k:.5f}  ({p_le_k:.2%})")
            st.metric("Mean = Variance = λT",        f"{lambda_T:.3f}")
            st.metric("Std Dev = √(λT)",             f"{math.sqrt(lambda_T):.4f}")

            st.latex(rf"P(N={poi_n}) = \frac{{({lambda_T:.2f})^{poi_n} \, e^{{-{lambda_T:.2f}}}}}{{{poi_n}!}} = {p_n:.5f}")

        # ─── Full PMF/CDF Table ───
        st.markdown("#### Full Probability Mass Function")
        n_max    = max(int(lambda_T * 3) + 5, 12)
        poi_rows = []
        cum_p    = 0
        mode_n   = max(range(n_max+1), key=lambda n: poisson_pmf(n, lambda_T))
        for n_i in range(n_max + 1):
            p_i   = poisson_pmf(n_i, lambda_T)
            cum_p += p_i
            poi_rows.append({
                "n":      n_i,
                "P(N=n)": f"{p_i:.5f}",
                "P(N≤n)": f"{cum_p:.5f}",
                "P(N>n)": f"{1-cum_p:.5f}",
                "Mode?":  "← Mode" if n_i == mode_n else ""
            })
        st.dataframe(pd.DataFrame(poi_rows), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mode (most likely n)", mode_n)
        col2.metric("P(zero arrivals)",     f"{poisson_pmf(0, lambda_T):.4f}")
        col3.metric("P(≥1 arrival)",        f"{1-poisson_pmf(0,lambda_T):.4f}")

        # ─── Business Application ───
        st.markdown("#### 📦 Applied Example")
        st.write("A store receives λ = 3 customers/hour. What staffing is needed to handle "
                 "demand 95% of the time in any given hour?")
        needed_n = next((n for n in range(20)
                         if sum(poisson_pmf(k, poi_lambda) for k in range(n+1)) >= 0.95), 15)
        st.success(f"💡 At λ = {poi_lambda:.1f}, staff to handle **{needed_n}** customers "
                   f"covers {sum(poisson_pmf(k,poi_lambda) for k in range(needed_n+1))*100:.1f}% "
                   "of scenarios — exceeds 95% threshold.")

    with tab4:
        st.markdown("### Visual Distribution Comparison")

        col1, col2 = st.columns([1, 2])
        with col1:
            rate_v   = st.number_input("Rate λ", value=2.0, min_value=0.1, step=0.1, key="dv_rate")
            period_v = st.number_input("Period T (for Poisson)", value=1.0, min_value=0.1,
                                        key="dv_T")
            overlay  = st.checkbox("Overlay multiple λ values", value=False)
            lambdas  = [rate_v]
            if overlay:
                extra = st.multiselect("Additional λ values",
                                       [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                                       default=[1.0, 3.0])
                lambdas = sorted(set(lambdas + extra))

        colors_v = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

        # ─── Exponential PDF ───
        st.markdown("#### Exponential PDF — Inter-arrival / Service Time")
        t_max_v = 4 / min(lambdas)
        t_range_v = [i * t_max_v / 300 for i in range(301)]

        fig_exp = go.Figure()
        for i, lam_v in enumerate(lambdas):
            y_exp = [lam_v * math.exp(-lam_v * t) for t in t_range_v]
            fig_exp.add_trace(go.Scatter(
                x=t_range_v, y=y_exp, mode="lines",
                name=f"λ = {lam_v}  (mean={1/lam_v:.2f})",
                line=dict(color=colors_v[i % len(colors_v)], width=2)
            ))
        fig_exp.update_layout(
            title="Exponential Distribution PDF  f(t) = λe^{−λt}",
            xaxis_title="Time t", yaxis_title="Probability Density",
            template="plotly_white", height=380,
            legend=dict(orientation="h", y=1.02)
        )
        st.plotly_chart(fig_exp, use_container_width=True)

        # ─── Exponential CDF ───
        st.markdown("#### Exponential CDF — P(T ≤ t)")
        fig_cdf = go.Figure()
        for i, lam_v in enumerate(lambdas):
            y_cdf = [1 - math.exp(-lam_v * t) for t in t_range_v]
            fig_cdf.add_trace(go.Scatter(
                x=t_range_v, y=y_cdf, mode="lines",
                name=f"λ = {lam_v}",
                line=dict(color=colors_v[i % len(colors_v)], width=2)
            ))
        fig_cdf.add_hline(y=0.632, line_dash="dot", line_color="gray",
                          annotation_text="P=0.632 at t=1/λ")
        fig_cdf.add_hline(y=0.95, line_dash="dot", line_color="orange",
                          annotation_text="95th percentile")
        fig_cdf.update_layout(
            title="Exponential CDF  P(T ≤ t) = 1 − e^{−λt}",
            xaxis_title="Time t", yaxis_title="P(T ≤ t)",
            yaxis_range=[0, 1.05], template="plotly_white", height=360
        )
        st.plotly_chart(fig_cdf, use_container_width=True)

        # ─── Poisson PMF ───
        st.markdown("#### Poisson PMF — Arrivals in Period T")
        lam_T_v = rate_v * period_v
        n_max_v = max(int(lam_T_v * 3) + 5, 12)
        n_range_v = list(range(n_max_v + 1))

        fig_poi = go.Figure()
        for i, lam_v in enumerate(lambdas):
            lt_v = lam_v * period_v
            pmf_v = [poisson_pmf(n, lt_v) for n in n_range_v]
            fig_poi.add_trace(go.Bar(
                x=n_range_v, y=pmf_v,
                name=f"λT = {lt_v:.1f}",
                marker_color=colors_v[i % len(colors_v)],
                opacity=0.7
            ))
        fig_poi.update_layout(
            title=f"Poisson PMF  P(N=n) for λT",
            xaxis_title="Number of Arrivals (n)",
            yaxis_title="Probability P(N=n)",
            template="plotly_white", height=380, barmode="group",
            legend=dict(orientation="h", y=1.02)
        )
        st.plotly_chart(fig_poi, use_container_width=True)

        # ─── CDF Comparison Sidebar ───
        with col2:
            st.markdown("#### Key Quantiles")
            qt_df = pd.DataFrame({
                "λ":      [str(l) for l in lambdas],
                "Mean":   [f"{1/l:.3f}" for l in lambdas],
                "P50 (median)": [f"{math.log(2)/l:.3f}" for l in lambdas],
                "P90":    [f"{-math.log(0.10)/l:.3f}" for l in lambdas],
                "P95":    [f"{-math.log(0.05)/l:.3f}" for l in lambdas],
                "P99":    [f"{-math.log(0.01)/l:.3f}" for l in lambdas],
            })
            st.dataframe(qt_df, use_container_width=True)


# ============================================================
# MODULE 12: LITTLE'S LAW (Chapter 11) - ENHANCED V5.0
# ============================================================
def module_littles():
    display_header("🔄", "Chapter 11", "Little's Law",
                   "The fundamental theorem connecting WIP, throughput rate, and flow time")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "🔬 Calculator", "📊 Flow Time Analyzer", "🎓 Practice"])

    with tab1:
        st.markdown("### Little's Law: I = R × T")
        st.write(
            "**Little's Law** is one of the most powerful and universal results in operations "
            "management. It states that in any stable system, the long-run average number of "
            "items in the system equals the average arrival rate multiplied by the average time "
            "each item spends in the system — regardless of arrival patterns, service distributions, "
            "or routing."
        )

        display_citation(
            "Little's law states a mathematical relationship between throughput rate, flow time, "
            "and the amount of work-in-process inventory. It applies to any stable system, "
            "regardless of the distribution of arrivals or service times.",
            "Jacobs & Chase (2024, p. 312)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            display_formula_card("Little's Law",     r"I = R \times T")
        with col2:
            display_formula_card("Solve for T",      r"T = \frac{I}{R}")
        with col3:
            display_formula_card("Solve for R",      r"R = \frac{I}{T}")

        st.markdown("#### Variable Definitions")
        vars_df = pd.DataFrame({
            "Variable": ["I", "R", "T"],
            "Name":     ["Inventory (WIP)", "Throughput Rate", "Flow Time"],
            "Units":    ["units / patients / calls / etc.",
                         "units per time period",
                         "time per unit"],
            "Also Called": ["Work-In-Process, L (in queuing)",
                             "Arrival rate λ, Throughput",
                             "Cycle time, Lead time, Ws (queuing)"]
        })
        st.dataframe(vars_df, use_container_width=True)

        st.markdown("#### Real-World Applications")
        apps_df = pd.DataFrame({
            "Domain":       ["Manufacturing", "Hospital", "Call Center",
                             "Supermarket checkout", "Airport security", "Software development"],
            "I (Inventory)":["500 units WIP", "80 patients", "5 calls on hold",
                              "3 customers waiting", "120 passengers queued", "40 open tickets"],
            "R (Rate)":     ["100 units/day", "20 patients/day", "60 calls/hour",
                              "2 customers/min", "720 passengers/hour", "8 tickets/day"],
            "T (Flow Time)":["5 days", "4 days", "5 min",
                              "1.5 min", "10 min", "5 days"]
        })
        st.dataframe(apps_df, use_container_width=True)

        display_key_insight(
            "The Lean Connection",
            "To reduce flow time T without cutting throughput R, you *must* reduce WIP I. "
            "This is the mathematical basis for Lean and JIT: smaller batches → less WIP → "
            "shorter lead times. Toyota's success is fundamentally an application of Little's Law."
        )

        st.markdown("#### Applying Little's Law at Multiple Levels")
        st.write(
            "Little's Law holds for **any subsystem**: the entire factory, a single workstation, "
            "a queue alone, or the service portion alone. The law can be applied to the full "
            "system (I = R × T) or to subcomponents:"
        )
        st.latex(r"L_q = \lambda \cdot W_q \quad \text{(queue only)} \qquad L_s = \lambda \cdot W_s \quad \text{(full system)}")

    with tab2:
        st.markdown("### Little's Law Calculator — Solve for Any Variable")

        solve_for = st.radio("Solve for:", ["I (WIP / Inventory)", "R (Throughput Rate)",
                                             "T (Flow Time)"], horizontal=True)

        col1, col2 = st.columns(2)

        with col1:
            if "I" not in solve_for:
                I_in = st.number_input("Inventory I (units)", value=500.0, min_value=0.01)
            if "R" not in solve_for:
                R_in = st.number_input("Throughput Rate R (units/period)", value=100.0, min_value=0.01)
            if "T" not in solve_for:
                T_in = st.number_input("Flow Time T (periods)", value=5.0, min_value=0.01)
            unit_label = st.text_input("Unit label", value="units/day")

        with col2:
            if "I" in solve_for:
                result = R_in * T_in
                st.metric("Inventory I", f"{result:.2f} units")
                st.latex(rf"I = R \times T = {R_in} \times {T_in} = {result:.2f}")
                st.info(f"💡 At R={R_in:.0f} {unit_label} and T={T_in:.1f} periods, "
                        f"**{result:.0f} units** are always in the system.")
            elif "R" in solve_for:
                result = I_in / T_in
                st.metric("Throughput Rate R", f"{result:.2f} {unit_label}")
                st.latex(rf"R = \frac{{I}}{{T}} = \frac{{{I_in}}}{{{T_in}}} = {result:.2f}")
                st.info(f"💡 To process {I_in:.0f} units in {T_in:.1f} periods, "
                        f"need throughput ≥ **{result:.2f}** per period.")
            else:
                result = I_in / R_in
                st.metric("Flow Time T", f"{result:.2f} periods")
                st.latex(rf"T = \frac{{I}}{{R}} = \frac{{{I_in}}}{{{R_in}}} = {result:.2f}")
                st.info(f"💡 With {I_in:.0f} units in system at rate {R_in:.0f} {unit_label}, "
                        f"each unit takes **{result:.2f} periods** end-to-end.")

        # ─── What-If Sensitivity ───
        st.markdown("---")
        st.markdown("#### What-If Sensitivity Table")
        st.write("How does flow time T change as WIP is reduced?")

        try:
            R_base = R_in if "R" not in solve_for else result
            I_base = I_in if "I" not in solve_for else result
            T_base = I_base / R_base

            reductions = [0, 10, 20, 30, 40, 50, 60, 70, 80]
            sensitivity_rows = []
            for pct in reductions:
                I_new   = I_base * (1 - pct/100)
                T_new   = I_new / R_base
                savings = T_base - T_new
                sensitivity_rows.append({
                    "WIP Reduction":  f"{pct}%",
                    "New WIP (I)":    f"{I_new:.0f}",
                    "New Flow Time":  f"{T_new:.2f}",
                    "Time Saved":     f"{savings:.2f}",
                    "% Time Reduction": f"{savings/T_base*100:.0f}%"
                })
            st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True)
        except:
            pass

    with tab3:
        st.markdown("### Process Flow Time Analyzer")
        st.write(
            "Apply Little's Law at each **stage** of a multi-step process to "
            "identify the flow time contribution and WIP bottleneck."
        )

        n_stages = st.number_input("Number of Process Stages", 2, 8, 4)
        throughput_r = st.number_input("System Throughput Rate R (units/day)", value=50.0)

        default_stages = [
            ("Raw Material Storage", 200, "days"),
            ("Work-In-Process",      150, "days"),
            ("Finished Goods",       100, "days"),
            ("Shipping Queue",        50, "days"),
        ]

        stage_data = []
        st.markdown("#### Stage WIP Levels")
        for i in range(int(n_stages)):
            sc = st.columns(3)
            name_d, wip_d, unit_d = default_stages[i] if i < 4 else (f"Stage {i+1}", 50, "days")
            with sc[0]: sname = st.text_input(f"Stage {i+1} Name", value=name_d, key=f"ll_n_{i}")
            with sc[1]: swip  = st.number_input(f"WIP {i+1}", value=wip_d, min_value=0,
                                                  key=f"ll_w_{i}")
            with sc[2]: sunit = st.text_input(f"Unit {i+1}", value=unit_d, key=f"ll_u_{i}")
            stage_data.append({"Stage": sname, "WIP": swip, "Unit": sunit})

        total_wip     = sum(s["WIP"] for s in stage_data)
        total_flow    = total_wip / throughput_r if throughput_r > 0 else 0

        stage_results = []
        for s in stage_data:
            ft = s["WIP"] / throughput_r if throughput_r > 0 else 0
            stage_results.append({
                "Stage":           s["Stage"],
                "WIP (units)":     s["WIP"],
                "Flow Time (days)":round(ft, 2),
                "% of Total FT":   f"{ft/total_flow*100:.1f}%" if total_flow > 0 else "—",
                "Little's Law":    f"T = {s['WIP']}/{throughput_r:.0f} = {ft:.2f} days"
            })

        st.dataframe(pd.DataFrame(stage_results), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total WIP",         f"{total_wip:,.0f} units")
        col2.metric("Total Flow Time",   f"{total_flow:.2f} days")
        col3.metric("Throughput Rate",   f"{throughput_r:.0f} units/day")

        # ─── Stacked bar chart ───
        fig = go.Figure(data=[
            go.Bar(name=s["Stage"], x=["Flow Time Breakdown"],
                   y=[s["WIP"]/throughput_r if throughput_r > 0 else 0])
            for s in stage_data
        ])
        fig.update_layout(barmode="stack",
                          title="Flow Time Contribution by Stage (Little's Law)",
                          yaxis_title="Flow Time (days)",
                          template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

        bottleneck = max(stage_results, key=lambda x: x["Flow Time (days)"])
        st.warning(f"⚠️ **Bottleneck Stage: {bottleneck['Stage']}** — "
                   f"contributes {bottleneck['Flow Time (days)']} days "
                   f"({bottleneck['% of Total FT']} of total flow time). "
                   "Reduce WIP here first.")

        # ─── Lean Target ───
        st.markdown("#### Lean Improvement Target")
        lean_target = st.slider("Target WIP Reduction (%)", 10, 90, 50)
        new_wip   = total_wip * (1 - lean_target/100)
        new_ft    = new_wip / throughput_r if throughput_r > 0 else 0
        st.success(
            f"📉 Reducing WIP by {lean_target}%: "
            f"WIP {total_wip:.0f} → {new_wip:.0f} units | "
            f"Flow Time {total_flow:.2f} → {new_ft:.2f} days "
            f"(**{total_flow-new_ft:.2f} days saved**)"
        )

    with tab4:
        st.markdown("### 📝 Little's Law Practice Problems")

        with st.expander("🟢 P1: Solve for WIP (Easy)"):
            display_practice_problem(1, "Easy",
                "A factory produces 80 units/day. Average flow time is 6 days. "
                "How many units are in the system (WIP)?")
            if st.button("Show Answer", key="ll_p1"):
                display_solution(
                    "I = R × T = 80 × 6 = **480 units**\n\n"
                    "This means at any moment, 480 units are in various stages of production."
                )

        with st.expander("🟢 P2: Solve for Flow Time (Easy)"):
            display_practice_problem(2, "Easy",
                "A hospital has 120 patients at any time. "
                "It admits 30 patients per day. What is the average length of stay?")
            if st.button("Show Answer", key="ll_p2"):
                display_solution("T = I/R = 120/30 = **4 days** average length of stay")

        with st.expander("🟡 P3: Multi-Stage Analysis (Medium)"):
            display_practice_problem(3, "Medium",
                "A plant processes 200 units/week. Stage WIPs: "
                "Raw Materials = 600, WIP = 400, Finished Goods = 200 units. "
                "Find total flow time and identify the highest-contributor stage.")
            if st.button("Show Answer", key="ll_p3"):
                display_solution(
                    "Apply I = R×T at each stage:\n\n"
                    "  T_RM = 600/200 = **3.0 weeks**\n\n"
                    "  T_WIP = 400/200 = **2.0 weeks**\n\n"
                    "  T_FG = 200/200 = **1.0 week**\n\n"
                    "  **Total flow time = 6.0 weeks**\n\n"
                    "Raw Materials stage contributes 50% of flow time — target for lean reduction."
                )

        with st.expander("🟡 P4: Lean Reduction (Medium)"):
            display_practice_problem(4, "Medium",
                "Current: R = 50/day, total WIP = 750. Target: reduce flow time to 10 days. "
                "What WIP reduction is needed? By what %?")
            if st.button("Show Answer", key="ll_p4"):
                T_curr = 750/50
                I_tgt  = 10 * 50
                pct    = (750 - I_tgt)/750*100
                display_solution(
                    f"Current T = 750/50 = {T_curr:.0f} days\n\n"
                    f"Target T = 10 days → I_target = R×T = 50×10 = **{I_tgt} units**\n\n"
                    f"WIP reduction needed = 750 − {I_tgt} = **{750-I_tgt} units** "
                    f"({pct:.0f}% reduction)"
                )

        with st.expander("🔴 P5: Call Center Throughput (Hard)"):
            display_practice_problem(5, "Hard",
                "A call center has 12 calls on hold (queue) + 4 calls being served = 16 total. "
                "Average handle time = 5 min. Average wait time = 15 min. "
                "Find λ (arrival rate), verify with Little's Law for both queue and system.")
            if st.button("Show Answer", key="ll_p5"):
                # System: Ls=16, Ws=20min
                # Queue: Lq=12, Wq=15min
                # Service: 4 calls, service=5min
                lam5 = 16/20  # calls/min from system
                display_solution(
                    "**From system:** Ls=16, Ws = Wq + service = 15+5 = 20 min\n\n"
                    "λ = Ls/Ws = 16/20 = **0.8 calls/min = 48 calls/hour**\n\n"
                    "**Verify queue:** Lq = λ×Wq = 0.8×15 = **12 ✓**\n\n"
                    "**Verify system:** Ls = λ×Ws = 0.8×20 = **16 ✓**\n\n"
                    "**Service portion:** L_service = λ×T_service = 0.8×5 = **4 ✓** (matches 4 being served)"
                )


# ============================================================
# MODULE 13: SIX SIGMA / DPMO (Chapter 12) - ENHANCED V5.0
# ============================================================
def module_dpmo():
    display_header("🎯", "Chapter 12", "Six Sigma — DPMO & DMAIC",
                   "Data-driven quality methodology for near-zero defect performance")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Theory", "🔬 DPMO Calculator", "🔄 DMAIC Framework", "📊 Sigma Benchmark"])

    with tab1:
        st.markdown("### Six Sigma Fundamentals")
        st.write(
            "**Six Sigma** is a disciplined, data-driven methodology for eliminating defects in "
            "any process. It targets fewer than 3.4 defects per million opportunities (DPMO), "
            "combining rigorous statistical tools with structured project management."
        )

        display_citation(
            "Six Sigma is a highly disciplined process that helps us focus on developing and "
            "delivering near-perfect products and services. The name derives from the statistical "
            "measure σ (sigma), with 'Six Sigma' meaning quality at the 6σ level.",
            "Jacobs & Chase (2024, p. 352)"
        )

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("DPMO",
                                 r"DPMO = \frac{\text{Total Defects}}{\text{Total Opportunities}} \times 10^6")
            display_formula_card("Defects per Unit",
                                 r"DPU = \frac{\text{Total Defects}}{\text{Units Inspected}}")
            display_formula_card("Rolled Throughput Yield",
                                 r"RTY = \prod_{i=1}^{n}(1 - DPU_i) = e^{-\sum DPU_i}")
        with col2:
            display_formula_card("Process Yield",
                                 r"\text{Yield} = 1 - \frac{\text{Defects}}{\text{Opportunities}}")
            display_formula_card("Sigma Level (approx.)",
                                 r"\sigma \approx \Phi^{-1}(1 - DPMO/10^6) + 1.5")

        st.markdown("#### Sigma Level Conversion Table")
        sigma_df = pd.DataFrame({
            "Sigma Level": ["1σ", "2σ", "3σ", "4σ", "5σ", "6σ"],
            "DPMO":        ["691,462", "308,538", "66,807", "6,210", "233", "3.4"],
            "Yield":       ["30.9%", "69.1%", "93.3%", "99.38%", "99.977%", "99.9997%"],
            "Industry Example": ["Unacceptable", "Below average", "Average (most companies)",
                                  "Above average", "World class", "Best-in-class / Medical devices"],
            "Defects/Day (1M ops)": ["691,462", "308,538", "66,807", "6,210", "233", "3.4"]
        })
        st.dataframe(sigma_df, use_container_width=True)

        display_key_insight(
            "The 1.5σ Shift",
            "The 3.4 DPMO for Six Sigma accounts for a ±1.5σ long-term process mean drift. "
            "In the short term, 6σ = 2 per billion defects. The 1.5σ shift makes the target "
            "more realistic for real-world processes that drift over time."
        )

        st.markdown("#### Six Sigma Roles")
        roles_df = pd.DataFrame({
            "Role":            ["Champion", "Master Black Belt", "Black Belt",
                                "Green Belt", "Yellow Belt"],
            "Level":           ["Executive", "Expert", "Full-time project leader",
                                "Part-time project leader", "Team member"],
            "Training":        ["1–2 days", "Several weeks", "4 weeks (~160 hrs)",
                                "2 weeks (~80 hrs)", "1–3 days"],
            "Responsibility":  ["Remove barriers; sponsor projects", "Train & mentor BBs",
                                "Lead DMAIC projects full time",
                                "Lead smaller projects; support BBs",
                                "Participate in improvement teams"]
        })
        st.dataframe(roles_df, use_container_width=True)

    with tab2:
        st.markdown("### DPMO & Sigma Level Calculator")

        col1, col2 = st.columns(2)
        with col1:
            units        = st.number_input("Units Inspected",        value=2000, min_value=1)
            defects      = st.number_input("Total Defects Found",    value=33,   min_value=0)
            opportunities = st.number_input("Opportunities per Unit", value=5,   min_value=1,
                                             help="Number of ways a defect can occur per unit")

        with col2:
            total_opp  = units * opportunities
            dpmo       = (defects / total_opp) * 1_000_000 if total_opp > 0 else 0
            dpu        = defects / units if units > 0 else 0
            yield_pct  = (1 - defects/total_opp) * 100 if total_opp > 0 else 0
            rty        = math.exp(-dpu)  # RTY approximation for single process

            if dpmo > 0:
                sigma_val = normal_ppf(1 - dpmo/1_000_000) + 1.5
                sigma_val = max(0, min(sigma_val, 6.5))
            else:
                sigma_val = 6.5

            st.metric("Total Opportunities",  f"{total_opp:,}")
            st.metric("DPMO",                 f"{dpmo:,.1f}")
            st.metric("Sigma Level",          f"{sigma_val:.2f}σ")
            st.metric("DPU",                  f"{dpu:.5f}")
            st.metric("Process Yield",        f"{yield_pct:.4f}%")
            st.metric("RTY (single step)",    f"{rty*100:.4f}%")

            st.latex(rf"DPMO = \frac{{{defects}}}{{{total_opp:,}}} \times 10^6 = {dpmo:,.1f}")
            st.latex(rf"\sigma \approx \Phi^{{-1}}(1 - {dpmo/1e6:.6f}) + 1.5 = {sigma_val:.2f}")

            if sigma_val >= 5.0:
                st.success(f"✅ World-class performance: {sigma_val:.2f}σ")
            elif sigma_val >= 4.0:
                st.info(f"👍 Above average: {sigma_val:.2f}σ — target 5σ+")
            elif sigma_val >= 3.0:
                st.warning(f"⚠️ Industry average: {sigma_val:.2f}σ — significant improvement possible")
            else:
                st.error(f"❌ Below average: {sigma_val:.2f}σ — urgent improvement needed")

        # ─── Multi-Stage RTY ───
        st.markdown("#### Multi-Stage Rolled Throughput Yield (RTY)")
        st.write("RTY = probability a unit passes all stages without any defect.")
        n_stages_dpmo = st.number_input("Number of Process Stages", 2, 8, 4, key="dpmo_ns")
        stage_dpus    = []
        rty_cols      = st.columns(int(n_stages_dpmo))
        default_dpus  = [0.02, 0.015, 0.025, 0.01, 0.03, 0.008, 0.012, 0.005]
        for i in range(int(n_stages_dpmo)):
            with rty_cols[i]:
                dpu_i = st.number_input(f"Stage {i+1} DPU",
                                         value=default_dpus[i] if i < len(default_dpus) else 0.02,
                                         format="%.4f", key=f"dpmo_dpu_{i}")
                stage_dpus.append(dpu_i)

        rty_total    = math.exp(-sum(stage_dpus))
        total_dpu    = sum(stage_dpus)
        rty_simple   = math.prod(1 - d for d in stage_dpus)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total DPU",              f"{total_dpu:.4f}")
        col2.metric("RTY (e^-ΣDPU)",          f"{rty_total*100:.3f}%")
        col3.metric("RTY (Π(1−DPU))",         f"{rty_simple*100:.3f}%")

        rty_rows = []
        running_rty = 1.0
        for i, dpu_s in enumerate(stage_dpus):
            running_rty *= math.exp(-dpu_s)
            rty_rows.append({
                "Stage":        f"Stage {i+1}",
                "DPU":          f"{dpu_s:.4f}",
                "Stage Yield":  f"{math.exp(-dpu_s)*100:.3f}%",
                "Cumulative RTY": f"{running_rty*100:.3f}%",
                "Hidden Factory": f"{(1-math.exp(-dpu_s))*100:.2f}% rework"
            })
        st.dataframe(pd.DataFrame(rty_rows), use_container_width=True)

    with tab3:
        st.markdown("### DMAIC Methodology")
        st.write(
            "**DMAIC** is the structured Six Sigma problem-solving roadmap. "
            "Each phase has defined inputs, tools, and deliverables that gate "
            "progress to the next phase."
        )

        phases_detail = {
            "Define": {
                "icon": "📋", "color": "#3498db",
                "question": "What is the problem?",
                "inputs":   "Customer complaints, business case",
                "tools":    "Project Charter, SIPOC Diagram, Voice of Customer (VOC), CTQ Tree",
                "outputs":  "Signed project charter, SIPOC, CTQ metrics, team roster",
                "gate":     "Is the problem clearly scoped and business case justified?"
            },
            "Measure": {
                "icon": "📏", "color": "#9b59b6",
                "question": "How big is the problem?",
                "inputs":   "Process maps, existing data",
                "tools":    "Process mapping, Data collection plan, MSA/Gage R&R, DPMO, Control charts",
                "outputs":  "Baseline Sigma level, validated measurement system, process map",
                "gate":     "Is the measurement system reliable (GR&R < 30%)?"
            },
            "Analyze": {
                "icon": "🔍", "color": "#e67e22",
                "question": "Why does the problem exist?",
                "inputs":   "Baseline data, process map",
                "tools":    "Fishbone, Pareto, 5 Whys, Regression, Hypothesis testing, FMEA",
                "outputs":  "Validated root causes (vital few Xs), data-supported evidence",
                "gate":     "Have root causes been statistically validated?"
            },
            "Improve": {
                "icon": "⚡", "color": "#e74c3c",
                "question": "How do we fix it?",
                "inputs":   "Root causes, constraints",
                "tools":    "DOE, Poka-yoke, FMEA, Pilot testing, Solution matrix",
                "outputs":  "Verified solution, pilot results, implementation plan",
                "gate":     "Did pilot testing show significant improvement?"
            },
            "Control": {
                "icon": "🎛️", "color": "#2ecc71",
                "question": "How do we sustain the gains?",
                "inputs":   "Improved process, pilot data",
                "tools":    "SPC, Control plans, Standard work, Visual management, Response plan",
                "outputs":  "Control plan, updated SOPs, handoff to process owner, closure",
                "gate":     "Are controls in place to prevent regression?"
            }
        }

        for phase, details in phases_detail.items():
            with st.expander(f"{details['icon']} **{phase}** — _{details['question']}_"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Inputs:** {details['inputs']}")
                    st.write(f"**Key Tools:** {details['tools']}")
                with col2:
                    st.write(f"**Deliverables:** {details['outputs']}")
                    st.info(f"🚪 **Phase Gate:** {details['gate']}")

        # ─── SIPOC Builder ───
        st.markdown("---")
        st.markdown("#### SIPOC Diagram Builder")
        st.write("Map the high-level process for your Define phase:")
        process_name = st.text_input("Process Name", value="Order Fulfillment Process")

        sipoc_cols = st.columns(5)
        headers = ["Suppliers", "Inputs", "Process Steps", "Outputs", "Customers"]
        defaults = [
            "Manufacturer\nWarehouse",
            "Customer order\nInventory",
            "Receive order\nPick items\nPack\nShip",
            "Shipped package\nInvoice\nTracking #",
            "Online buyer\nRetail store"
        ]
        sipoc_data = {}
        for i, (hdr, dflt) in enumerate(zip(headers, defaults)):
            with sipoc_cols[i]:
                st.write(f"**{hdr}**")
                content = st.text_area(hdr, value=dflt, height=130,
                                       key=f"sipoc_{i}", label_visibility="collapsed")
                sipoc_data[hdr] = content

        st.markdown(f"**Process:** _{process_name}_")
        for hdr, content in sipoc_data.items():
            items = [c.strip() for c in content.split("\n") if c.strip()]
            st.write(f"**{hdr}:** {' | '.join(items)}")

    with tab4:
        st.markdown("### Sigma Level Benchmarking")
        st.write("Visualize DPMO and sigma level across industries and compare to your process.")

        # ─── DPMO Gauge ───
        if 'dpmo' in dir() and dpmo >= 0:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sigma_val,
                delta={"reference": 3.0, "increasing": {"color": "green"}},
                title={"text": "Sigma Level"},
                gauge={
                    "axis":  {"range": [0, 7], "tickvals": [1,2,3,4,5,6]},
                    "bar":   {"color": "#2ecc71" if sigma_val >= 5 else
                                        "#f39c12" if sigma_val >= 3 else "#e74c3c"},
                    "steps": [{"range":[0,3],   "color":"rgba(231,76,60,0.2)"},
                               {"range":[3,4.5], "color":"rgba(243,156,18,0.2)"},
                               {"range":[4.5,7], "color":"rgba(46,204,113,0.2)"}],
                    "threshold": {"line": {"color":"darkblue","width":3},
                                  "thickness":0.75, "value":6}
                }
            ))
            fig_gauge.update_layout(height=300, template="plotly_white")
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ─── Industry Benchmark Chart ───
        industry_benchmarks = {
            "Best Hospitals (mortality)":    5.7,
            "Airline baggage handling":      4.8,
            "Average manufacturing":         3.4,
            "Restaurant orders":             3.2,
            "Payroll processing":            3.0,
            "Doctor prescriptions":          2.8,
            "Average service industry":      2.5,
            "IRS tax advice":                2.0,
        }

        fig_bench = go.Figure()
        sorted_b = sorted(industry_benchmarks.items(), key=lambda x: x[1])
        fig_bench.add_trace(go.Bar(
            x=[v for _, v in sorted_b],
            y=[k for k, _ in sorted_b],
            orientation="h",
            marker_color=["#2ecc71" if v >= 5 else "#f39c12" if v >= 3.5 else "#e74c3c"
                          for _, v in sorted_b]
        ))
        if 'sigma_val' in dir():
            fig_bench.add_vline(x=sigma_val, line_dash="dash", line_color="blue",
                                annotation_text=f"Your Process: {sigma_val:.2f}σ")
        fig_bench.add_vline(x=6, line_dash="dot", line_color="green",
                            annotation_text="6σ Target")
        fig_bench.update_layout(title="Industry Sigma Level Benchmarks",
                                xaxis_title="Sigma Level",
                                xaxis_range=[0, 7],
                                template="plotly_white", height=420)
        st.plotly_chart(fig_bench, use_container_width=True)


# ============================================================
# MODULE 14: FMEA (Chapter 12) - ENHANCED V5.0
# ============================================================
def module_fmea():
    display_header("⚠️", "Chapter 12", "FMEA — Failure Mode & Effects Analysis",
                   "Systematic risk identification and prioritization using RPN scoring")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 FMEA Worksheet", "📊 Risk Matrix"])

    with tab1:
        st.markdown("### FMEA Theory")
        st.write(
            "**Failure Mode and Effects Analysis (FMEA)** is a proactive reliability tool "
            "that systematically identifies potential failure modes, their causes, effects, "
            "and current controls — before failures occur. It prioritizes action using the "
            "**Risk Priority Number (RPN)**."
        )

        display_formula_card("Risk Priority Number",
                             r"RPN = S \times O \times D")
        st.write("Where **S** = Severity, **O** = Occurrence, **D** = Detection (1–10 each)")

        st.markdown("#### Rating Scale Definitions")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Severity (S) — Effect on Customer**")
            s_scale = pd.DataFrame({
                "Score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "Meaning": ["No effect", "Very minor", "Minor", "Very low",
                             "Low", "Moderate", "High", "Very high",
                             "Hazardous (warning)", "Hazardous (no warning)"]
            })
            st.dataframe(s_scale, use_container_width=True)
        with col2:
            st.markdown("**Occurrence (O) — Frequency of Cause**")
            o_scale = pd.DataFrame({
                "Score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "Frequency": ["1 in 1,500,000", "1 in 150,000", "1 in 15,000",
                               "1 in 2,000", "1 in 400", "1 in 80",
                               "1 in 20", "1 in 8", "1 in 3", "1 in 2"]
            })
            st.dataframe(o_scale, use_container_width=True)
        with col3:
            st.markdown("**Detection (D) — Ability to Find Before Reaching Customer**")
            d_scale = pd.DataFrame({
                "Score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "Ability": ["Almost certain", "Very high", "High",
                             "Moderately high", "Moderate", "Low",
                             "Very low", "Remote", "Very remote", "Cannot detect"]
            })
            st.dataframe(d_scale, use_container_width=True)

        st.markdown("#### RPN Interpretation")
        rpn_df = pd.DataFrame({
            "RPN Range":  ["1–50", "51–100", "101–200", "201–500", "501–1000"],
            "Priority":   ["Low", "Moderate", "High", "Very High", "Critical"],
            "Action":     ["Monitor; no immediate action",
                           "Plan improvement; assign owner",
                           "Prioritize in next cycle",
                           "Urgent action required",
                           "Stop process; fix immediately"]
        })
        st.dataframe(rpn_df, use_container_width=True)

        display_key_insight(
            "RPN Limitation",
            "RPN can be misleading: S=10, O=1, D=1 → RPN=10; but S=3, O=3, D=3 → RPN=27. "
            "The 10-severity failure is arguably more critical! Always review high-S items "
            "regardless of RPN. Some organizations use a criticality matrix (S×O) separately."
        )

    with tab2:
        st.markdown("### FMEA Worksheet")

        process_name_f = st.text_input("Process / Product Name", value="Assembly Line Process")
        num_modes      = st.number_input("Number of Failure Modes", 2, 12, 5)

        default_modes = [
            ("Welding",       "Weld crack",         8, 4, 6),
            ("Assembly",      "Missing component",  7, 3, 5),
            ("Testing",       "False pass",         9, 2, 8),
            ("Packaging",     "Wrong label",        5, 5, 4),
            ("Shipping",      "Damage in transit",  6, 4, 7),
            ("Raw Material",  "Out-of-spec input",  7, 3, 6),
            ("Machining",     "Dimensional error",  6, 5, 4),
            ("Inspection",    "Missed defect",      8, 3, 7),
            ("Software",      "Logic error",        9, 2, 6),
            ("Calibration",   "Instrument drift",   7, 3, 5),
            ("Maintenance",   "Lubrication missed", 5, 4, 6),
            ("Training",      "SOP not followed",   6, 5, 5),
        ]

        modes = []
        col_headers = st.columns([2, 2.5, 1.5, 1.5, 1, 1, 1, 1.5, 2])
        for h, lbl in zip(col_headers, ["Process Step","Failure Mode","Effect","Cause",
                                         "S","O","D","RPN","Action"]):
            h.write(f"**{lbl}**")

        for i in range(int(num_modes)):
            d = default_modes[i] if i < len(default_modes) else (f"Step {i+1}", f"Failure {i+1}", 5, 4, 6)
            row_cols = st.columns([2, 2.5, 1.5, 1.5, 1, 1, 1, 1.5, 2])
            with row_cols[0]: step    = st.text_input("", value=d[0], key=f"f_step_{i}", label_visibility="collapsed")
            with row_cols[1]: failure = st.text_input("", value=d[1], key=f"f_fail_{i}", label_visibility="collapsed")
            with row_cols[2]: effect  = st.text_input("", value="Customer impact", key=f"f_eff_{i}", label_visibility="collapsed")
            with row_cols[3]: cause   = st.text_input("", value="Root cause", key=f"f_cau_{i}", label_visibility="collapsed")
            with row_cols[4]: s_val   = st.number_input("", 1, 10, d[2], key=f"f_s_{i}", label_visibility="collapsed")
            with row_cols[5]: o_val   = st.number_input("", 1, 10, d[3], key=f"f_o_{i}", label_visibility="collapsed")
            with row_cols[6]: det_val = st.number_input("", 1, 10, d[4], key=f"f_d_{i}", label_visibility="collapsed")
            rpn_val = s_val * o_val * det_val
            with row_cols[7]:
                color = "🔴" if rpn_val >= 200 else "🟡" if rpn_val >= 100 else "🟢"
                st.write(f"{color} **{rpn_val}**")
            with row_cols[8]:
                action_default = ("Redesign" if rpn_val >= 200 else
                                  "Improve detection" if det_val >= 7 else
                                  "Reduce occurrence" if o_val >= 6 else "Monitor")
                action = st.text_input("", value=action_default, key=f"f_act_{i}",
                                       label_visibility="collapsed")
            modes.append({
                "Step": step, "Failure Mode": failure, "Effect": effect, "Cause": cause,
                "S": s_val, "O": o_val, "D": det_val, "RPN": rpn_val,
                "Action": action
            })

        df_fmea = pd.DataFrame(modes).sort_values("RPN", ascending=False).reset_index(drop=True)
        df_fmea["Priority"] = df_fmea["RPN"].apply(
            lambda r: "🔴 Critical" if r >= 200 else "🟡 High" if r >= 100 else "🟢 Low")
        st.markdown("---")
        st.markdown("#### Ranked FMEA Results")
        st.dataframe(df_fmea, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        rpn_vals = [m["RPN"] for m in modes]
        col1.metric("Total RPN",    f"{sum(rpn_vals):,}")
        col2.metric("Highest RPN",  f"{max(rpn_vals)}")
        col3.metric("Average RPN",  f"{sum(rpn_vals)/len(rpn_vals):.0f}")
        col4.metric("Critical Items (≥200)", sum(1 for r in rpn_vals if r >= 200))

        # ─── After-Action RPN ───
        st.markdown("#### After-Action RPN Improvement Simulator")
        st.write("Simulate the effect of improvement actions on highest-priority item:")
        top_item = df_fmea.iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1: s_new = st.slider(f"New S (was {top_item['S']})", 1, 10, max(1, int(top_item['S'])-1))
        with col2: o_new = st.slider(f"New O (was {top_item['O']})", 1, 10, max(1, int(top_item['O'])-2))
        with col3: d_new = st.slider(f"New D (was {top_item['D']})", 1, 10, max(1, int(top_item['D'])-2))
        rpn_new     = s_new * o_new * d_new
        rpn_old     = int(top_item['RPN'])
        rpn_reduce  = (rpn_old - rpn_new) / rpn_old * 100
        st.metric(f"New RPN for '{top_item['Failure Mode']}'",
                  f"{rpn_new}", delta=f"{rpn_new - rpn_old} ({-rpn_reduce:.0f}% reduction)")

    with tab3:
        st.markdown("### Risk Matrix (Severity × Occurrence)")
        st.write(
            "Plot failure modes on a 10×10 S×O matrix. High-severity high-occurrence items "
            "in the upper-right corner demand immediate action regardless of RPN."
        )

        if 'modes' in dir() and modes:
            fig_risk = go.Figure()

            # Background zones
            fig_risk.add_shape(type="rect", x0=7, y0=7, x1=10.5, y1=10.5,
                               fillcolor="rgba(231,76,60,0.15)", line_width=0)
            fig_risk.add_shape(type="rect", x0=4, y0=4, x1=7, y1=7,
                               fillcolor="rgba(243,156,18,0.15)", line_width=0)
            fig_risk.add_shape(type="rect", x0=0.5, y0=0.5, x1=4, y1=4,
                               fillcolor="rgba(46,204,113,0.15)", line_width=0)

            colors_rpn = ["#e74c3c" if m["RPN"] >= 200 else
                          "#f39c12" if m["RPN"] >= 100 else "#2ecc71"
                          for m in modes]
            sizes_rpn  = [max(10, min(m["RPN"]/10 + 5, 40)) for m in modes]

            fig_risk.add_trace(go.Scatter(
                x=[m["O"] for m in modes],
                y=[m["S"] for m in modes],
                mode="markers+text",
                marker=dict(size=sizes_rpn, color=colors_rpn, opacity=0.8,
                            line=dict(width=1, color="white")),
                text=[f"{m['Failure Mode']}<br>RPN={m['RPN']}" for m in modes],
                textposition="top center",
                hovertemplate="<b>%{text}</b><br>O=%{x}, S=%{y}<extra></extra>"
            ))

            fig_risk.update_layout(
                title="FMEA Risk Matrix — Severity vs. Occurrence<br>"
                      "<sub>🔴 Critical (S≥7,O≥7) | 🟡 Monitor (S≥4,O≥4) | 🟢 Acceptable</sub>",
                xaxis=dict(title="Occurrence (O)", range=[0.5, 10.5],
                           tickvals=list(range(1, 11))),
                yaxis=dict(title="Severity (S)", range=[0.5, 10.5],
                           tickvals=list(range(1, 11))),
                template="plotly_white", height=520, showlegend=False
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # ─── Pareto of RPN ───
            st.markdown("#### Pareto of Failure Modes by RPN")
            df_sorted = df_fmea.sort_values("RPN", ascending=False)
            cum_rpn   = df_sorted["RPN"].cumsum() / df_sorted["RPN"].sum() * 100

            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=df_sorted["Failure Mode"], y=df_sorted["RPN"],
                                     name="RPN", marker_color="#3498db"))
            fig_par.add_trace(go.Scatter(x=df_sorted["Failure Mode"], y=cum_rpn.values,
                                          name="Cumulative %", mode="lines+markers",
                                          line=dict(color="#e74c3c"), yaxis="y2"))
            fig_par.add_hline(y=80, line_dash="dash", line_color="orange",
                              annotation_text="80%", yref="y2")
            fig_par.update_layout(
                yaxis=dict(title="RPN"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                            range=[0, 110], ticksuffix="%"),
                xaxis_tickangle=-30,
                title="RPN Pareto Chart", template="plotly_white", height=400
            )
            st.plotly_chart(fig_par, use_container_width=True)


# ============================================================
# MODULE 15: SQC — CONTROL CHARTS (Chapter 13) - ENHANCED V5.1
# ============================================================
def module_sqc():
    display_header(
        "📉", "Chapter 13", "Statistical Quality Control",
        "p-Chart, c-Chart, x̄-Chart and R-Chart for process monitoring"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📚 Theory", "📊 p-Chart", "📊 c-Chart", "📊 x̄ & R Chart", "🎓 Practice"]
    )

    with tab1:
        st.markdown("### Statistical Process Control (SPC)")
        st.write(
            "**Control charts** separate **common-cause variation** from "
            "**special-cause variation**. Points outside control limits, or repeated "
            "patterns within limits, indicate the process may need investigation."
        )

        display_citation(
            "A control chart is a graph used to study how a process changes over time with data plotted in time order.",
            "ASQ (Statistical Process Control Charts)"
        )

        chart_guide = pd.DataFrame({
            "Data Type": [
                "Attribute — proportions",
                "Attribute — defect counts",
                "Variable — sample mean",
                "Variable — sample range",
                "Variable — individual values"
            ],
            "Chart": [
                "p-Chart",
                "c-Chart",
                "x̄-Chart",
                "R-Chart",
                "I-MR Chart"
            ],
            "Use When": [
                "Measuring fraction defective per sample",
                "Counting defects per unit (Poisson count)",
                "Monitoring average of samples (n ≥ 2)",
                "Monitoring within-sample variability",
                "Only one observation per time period"
            ],
            "Distribution": [
                "Binomial",
                "Poisson",
                "Approximately normal",
                "Range-based",
                "Approximately normal"
            ]
        })
        st.dataframe(chart_guide, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### p-Chart Formulas")
            display_formula_card("Center Line", r"\bar{p} = \frac{\sum d_i}{\sum n_i}")
            display_formula_card("Standard Error", r"\sigma_p = \sqrt{\frac{\bar{p}(1-\bar{p})}{n}}")
            display_formula_card("Control Limits", r"UCL/LCL = \bar{p} \pm 3\sigma_p")
        with col2:
            st.markdown("#### c-Chart Formulas")
            display_formula_card("Center Line", r"\bar{c} = \frac{\text{Total Defects}}{\text{Number of Units}}")
            display_formula_card("Control Limits", r"UCL/LCL = \bar{c} \pm 3\sqrt{\bar{c}}")

        display_key_insight(
            "Western Electric Rules",
            "Watch for nonrandom patterns such as 8 points on one side of the center line, "
            "6 points trending upward or downward, or 2 of 3 points near a control limit."
        )

        st.markdown("#### Type I vs. Type II Errors in SPC")
        err_df = pd.DataFrame({
            "Error Type": [
                "Type I (α) — False Alarm",
                "Type II (β) — Missed Signal"
            ],
            "Description": [
                "Signal a special cause when the process is actually in control",
                "Fail to detect a real special cause"
            ],
            "Consequence": [
                "Unnecessary investigation and adjustment",
                "Defects continue undetected"
            ],
            "Typical Effect": [
                "More false alarms",
                "More missed problems"
            ]
        })
        st.dataframe(err_df, use_container_width=True, hide_index=True)

        display_textbook_content(
            "Control Chart Interpretation",
            """Control charts are used to monitor process stability over time. 
            A stable process is predictable, even if it is not necessarily meeting specifications.
            To improve quality, first determine whether the process is in statistical control;
            then, if needed, improve the process mean or reduce variation."""
        )

    with tab2:
        st.markdown("### p-Chart — Fraction Defective")

        col1, col2 = st.columns([1, 2])
        with col1:
            n_p = st.number_input("Sample Size (n)", value=300, min_value=10)
            num_samp_p = st.number_input("Number of Samples", value=15, min_value=5, max_value=30)

        default_def = [8, 12, 10, 9, 11, 14, 7, 13, 10, 8, 15, 9, 11, 10, 12,
                       8, 16, 7, 11, 9, 10, 13, 8, 12, 11, 9, 14, 8, 10, 12]

        st.markdown("#### Enter Defectives per Sample")
        defectives_p = []
        d_cols = st.columns(5)
        for i in range(int(num_samp_p)):
            with d_cols[i % 5]:
                d = st.number_input(
                    f"S{i+1}",
                    value=default_def[i] if i < len(default_def) else 10,
                    min_value=0,
                    key=f"pc_d_{i}"
                )
                defectives_p.append(d)

        if n_p > 0:
            total_def_p = sum(defectives_p)
            total_insp_p = n_p * num_samp_p
            p_bar = total_def_p / total_insp_p
            sp_p = math.sqrt(p_bar * (1 - p_bar) / n_p) if 0 < p_bar < 1 else 0
            ucl_p = p_bar + 3 * sp_p
            lcl_p = max(0, p_bar - 3 * sp_p)

            proportions = [d / n_p for d in defectives_p]
            ooc_p = [i + 1 for i, prop in enumerate(proportions) if prop > ucl_p or prop < lcl_p]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("p̄ (Center Line)", f"{p_bar:.4f}  ({p_bar:.2%})")
            col2.metric("UCL", f"{ucl_p:.4f}  ({ucl_p:.2%})")
            col3.metric("LCL", f"{lcl_p:.4f}  ({lcl_p:.2%})")
            col4.metric("σp", f"{sp_p:.4f}")
            st.metric("Out-of-Control Points", len(ooc_p),
                      delta="Action required" if ooc_p else "In control")

            sample_ids = list(range(1, int(num_samp_p) + 1))
            colors_p = ["#e74c3c" if p > ucl_p or p < lcl_p else "#3498db" for p in proportions]

            fig_p = go.Figure()
            fig_p.add_hline(y=ucl_p, line_dash="dash", line_color="red",
                            annotation_text=f"UCL={ucl_p:.4f}")
            fig_p.add_hline(y=p_bar, line_dash="dot", line_color="green",
                            annotation_text=f"p̄={p_bar:.4f}")
            fig_p.add_hline(y=lcl_p, line_dash="dash", line_color="red",
                            annotation_text=f"LCL={lcl_p:.4f}")
            fig_p.add_trace(go.Scatter(
                x=sample_ids, y=proportions, mode="lines+markers",
                marker=dict(size=10, color=colors_p, line=dict(width=1.5, color="white")),
                line=dict(color="#95a5a6", width=1.5), name="Sample Proportion"
            ))
            for ooc_i in ooc_p:
                fig_p.add_annotation(
                    x=ooc_i, y=proportions[ooc_i - 1],
                    text="⚠️ OOC", showarrow=True, arrowhead=2,
                    font=dict(color="red")
                )
            fig_p.update_layout(
                title=f"p-Chart — Fraction Defective (n={n_p})",
                xaxis_title="Sample Number",
                yaxis_title="Fraction Defective",
                template="plotly_white",
                height=420
            )
            st.plotly_chart(fig_p, use_container_width=True)

            if ooc_p:
                st.error(f"⚠️ **Out-of-Control:** Samples {ooc_p} — investigate special causes.")
            else:
                st.success("✅ All samples within 3σ control limits — process appears stable.")

            with st.expander("📋 Full p-Chart Data Table"):
                p_tbl = pd.DataFrame({
                    "Sample": sample_ids,
                    "Defectives": defectives_p,
                    "Proportion": [f"{p:.4f}" for p in proportions],
                    "Status": ["🔴 OOC" if i + 1 in ooc_p else "✅ OK" for i in range(int(num_samp_p))],
                    "Dist from CL": [f"{abs(p - p_bar) / sp_p:.2f}σ" if sp_p > 0 else "NA" for p in proportions]
                })
                st.dataframe(p_tbl, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### c-Chart — Count of Defects per Unit")

        col1, col2 = st.columns([1, 2])
        with col1:
            num_units_c = st.number_input("Number of Units Sampled", value=20,
                                          min_value=5, max_value=40, key="cc_units")

        default_defects_c = [4, 3, 5, 6, 4, 7, 3, 4, 5, 8, 3, 4, 6, 5, 4,
                             3, 9, 4, 5, 6, 3, 4, 7, 5, 4, 3, 6, 4, 5, 8,
                             4, 5, 3, 7, 4, 6, 5, 4, 8, 3]

        st.markdown("#### Enter Defect Count per Unit")
        defects_c = []
        c_cols = st.columns(5)
        for i in range(int(num_units_c)):
            with c_cols[i % 5]:
                c_val = st.number_input(
                    f"U{i+1}",
                    value=default_defects_c[i] if i < len(default_defects_c) else 4,
                    min_value=0,
                    key=f"cc_c_{i}"
                )
                defects_c.append(c_val)

        c_bar = sum(defects_c) / num_units_c
        ucl_c = c_bar + 3 * math.sqrt(c_bar)
        lcl_c = max(0, c_bar - 3 * math.sqrt(c_bar))
        ooc_c = [i + 1 for i, c in enumerate(defects_c) if c > ucl_c or c < lcl_c]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("c̄ (Center Line)", f"{c_bar:.3f}")
        col2.metric("UCL", f"{ucl_c:.3f}")
        col3.metric("LCL", f"{lcl_c:.3f}")
        col4.metric("√c̄ (Std Dev)", f"{math.sqrt(c_bar):.3f}")
        st.metric("Out-of-Control Points", len(ooc_c),
                  delta="Action required" if ooc_c else "In control")

        unit_ids = list(range(1, int(num_units_c) + 1))
        colors_c = ["#e74c3c" if c > ucl_c or c < lcl_c else "#2ecc71" for c in defects_c]

        fig_c = go.Figure()
        fig_c.add_hline(y=ucl_c, line_dash="dash", line_color="red",
                        annotation_text=f"UCL={ucl_c:.3f}")
        fig_c.add_hline(y=c_bar, line_dash="dot", line_color="blue",
                        annotation_text=f"c̄={c_bar:.3f}")
        fig_c.add_hline(y=lcl_c, line_dash="dash", line_color="red",
                        annotation_text=f"LCL={lcl_c:.3f}")
        fig_c.add_trace(go.Bar(
            x=unit_ids, y=defects_c, marker_color=colors_c,
            name="Defect Count", opacity=0.85
        ))
        fig_c.add_trace(go.Scatter(
            x=unit_ids, y=defects_c, mode="lines",
            line=dict(color="#2c3e50", width=1.5),
            name="Trend"
        ))
        for ooc_i in ooc_c:
            fig_c.add_annotation(
                x=ooc_i, y=defects_c[ooc_i - 1],
                text="⚠️", showarrow=False,
                font=dict(color="red", size=14)
            )
        fig_c.update_layout(
            title="c-Chart — Defects per Unit",
            xaxis_title="Unit Number",
            yaxis_title="Number of Defects",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_c, use_container_width=True)

        if ooc_c:
            st.error(f"⚠️ **Out-of-Control:** Units {ooc_c}")
        else:
            st.success("✅ All units within control limits — defect rate is stable.")

    with tab4:
        st.markdown("### x̄-Chart & R-Chart (Variables)")
        st.write(
            "**x̄ and R charts** are used together to monitor the mean and range of "
            "continuous measurements in subgroup data."
        )

        a2_table = {2:1.880, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
        d3_table = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0.076, 8:0.136, 9:0.184, 10:0.223}
        d4_table = {2:3.267, 3:2.574, 4:2.282, 5:2.114, 6:2.004, 7:1.924, 8:1.864, 9:1.816, 10:1.777}

        col1, col2 = st.columns([1, 2])
        with col1:
            n_xbar = st.selectbox("Subgroup Size (n)", [2,3,4,5,6,7,8,9,10], index=3)
            num_subs = st.number_input("Number of Subgroups", value=15, min_value=5, max_value=25)
            target_mean = st.number_input("Target Mean (optional)", value=0.0)

        A2 = a2_table[n_xbar]
        D3 = d3_table[n_xbar]
        D4 = d4_table[n_xbar]

        import random
        random.seed(42)
        default_xbar_data = [
            [round(10 + random.gauss(0, 0.3), 2) for _ in range(n_xbar)]
            for _ in range(int(num_subs))
        ]
        if int(num_subs) > 7:
            default_xbar_data[7][0] = 11.5

        subgroup_means = []
        subgroup_ranges = []

        st.markdown("#### Enter or Review Subgroup Data")
        for i in range(int(num_subs)):
            sg_cols = st.columns(n_xbar + 2)
            sg_cols[0].write(f"**SG{i+1}**")
            obs = []
            for j in range(n_xbar):
                with sg_cols[j + 1]:
                    val = st.number_input(
                        "",
                        value=default_xbar_data[i][j] if i < len(default_xbar_data) else 10.0,
                        format="%.2f",
                        key=f"xbar_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    obs.append(val)
            sg_mean = sum(obs) / n_xbar
            sg_rng = max(obs) - min(obs)
            subgroup_means.append(sg_mean)
            subgroup_ranges.append(sg_rng)
            sg_cols[n_xbar + 1].write(f"x̄={sg_mean:.2f} R={sg_rng:.2f}")

        x_dbl_bar = sum(subgroup_means) / num_subs
        R_bar = sum(subgroup_ranges) / num_subs
        base_mean = float(target_mean) if target_mean != 0 else x_dbl_bar

        ucl_xbar = base_mean + A2 * R_bar
        lcl_xbar = base_mean - A2 * R_bar
        ucl_r = D4 * R_bar
        lcl_r = D3 * R_bar

        ooc_xbar = [i + 1 for i, x in enumerate(subgroup_means) if x > ucl_xbar or x < lcl_xbar]
        ooc_r = [i + 1 for i, r in enumerate(subgroup_ranges) if r > ucl_r]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### x̄-Chart Parameters")
            st.metric("x̄̄ (Grand Mean)", f"{x_dbl_bar:.4f}")
            st.metric("UCL_x̄", f"{ucl_xbar:.4f}")
            st.metric("LCL_x̄", f"{lcl_xbar:.4f}")
            st.metric("OOC Points", len(ooc_xbar))
        with col2:
            st.markdown("#### R-Chart Parameters")
            st.metric("R̄ (Avg Range)", f"{R_bar:.4f}")
            st.metric("UCL_R", f"{ucl_r:.4f}")
            st.metric("LCL_R", f"{lcl_r:.4f}")
            st.metric("OOC Points", len(ooc_r))

        st.write(f"**Control Chart Constants (n={n_xbar}):** A₂={A2}, D₃={D3}, D₄={D4}")

        sg_ids = list(range(1, int(num_subs) + 1))

        colors_x = ["#e74c3c" if x > ucl_xbar or x < lcl_xbar else "#3498db" for x in subgroup_means]
        fig_xbar = go.Figure()
        fig_xbar.add_hline(y=ucl_xbar, line_dash="dash", line_color="red",
                           annotation_text=f"UCL={ucl_xbar:.3f}")
        fig_xbar.add_hline(y=x_dbl_bar, line_dash="dot", line_color="green",
                           annotation_text=f"x̄̄={x_dbl_bar:.3f}")
        fig_xbar.add_hline(y=lcl_xbar, line_dash="dash", line_color="red",
                           annotation_text=f"LCL={lcl_xbar:.3f}")
        fig_xbar.add_trace(go.Scatter(
            x=sg_ids, y=subgroup_means, mode="lines+markers",
            marker=dict(size=10, color=colors_x, line=dict(width=1.5, color="white")),
            line=dict(color="#95a5a6", width=1.5), name="Subgroup Mean"
        ))
        fig_xbar.update_layout(
            title=f"x̄-Chart (n={n_xbar})",
            xaxis_title="Subgroup",
            yaxis_title="Subgroup Mean",
            template="plotly_white",
            height=360
        )
        st.plotly_chart(fig_xbar, use_container_width=True)

        colors_r = ["#e74c3c" if r > ucl_r else "#9b59b6" for r in subgroup_ranges]
        fig_r = go.Figure()
        fig_r.add_hline(y=ucl_r, line_dash="dash", line_color="red",
                        annotation_text=f"UCL_R={ucl_r:.3f}")
        fig_r.add_hline(y=R_bar, line_dash="dot", line_color="green",
                        annotation_text=f"R̄={R_bar:.3f}")
        if D3 > 0:
            fig_r.add_hline(y=lcl_r, line_dash="dash", line_color="red",
                            annotation_text=f"LCL_R={lcl_r:.3f}")
        fig_r.add_trace(go.Scatter(
            x=sg_ids, y=subgroup_ranges, mode="lines+markers",
            marker=dict(size=10, color=colors_r, line=dict(width=1.5, color="white")),
            line=dict(color="#95a5a6", width=1.5), name="Subgroup Range"
        ))
        fig_r.update_layout(
            title=f"R-Chart (n={n_xbar})",
            xaxis_title="Subgroup",
            yaxis_title="Subgroup Range",
            template="plotly_white",
            height=320
        )
        st.plotly_chart(fig_r, use_container_width=True)

        if ooc_xbar or ooc_r:
            st.error(f"⚠️ x̄ OOC: {ooc_xbar if ooc_xbar else 'None'} | R OOC: {ooc_r if ooc_r else 'None'}")
        else:
            st.success("✅ Both x̄ and R charts are in statistical control.")

    with tab5:
        st.markdown("### 📝 SQC Practice Problems")

        with st.expander("🟢 P1: Build a p-Chart (Easy)"):
            display_practice_problem(
                1, "Easy",
                "A sample of 200 units contains 12 defectives. What is the sample proportion defective?"
            )
            user_ans = st.number_input("Your Answer (proportion):", key="sqc_p1")
            if st.button("Check Answer", key="sqc_p1_btn"):
                correct = 12 / 200
                if check_answer(user_ans, correct, tolerance=0.02):
                    st.success(f"✅ Correct! p = {correct:.4f}")
                else:
                    display_solution(f"The proportion defective is <strong>{correct:.4f}</strong>.")

        with st.expander("🟡 P2: c-Chart Limits (Medium)"):
            display_practice_problem(
                2, "Medium",
                "A process averages 4.0 defects per unit. Calculate UCL and LCL."
            )
            user_ucl = st.number_input("Your UCL:", key="sqc_p2_ucl")
            user_lcl = st.number_input("Your LCL:", key="sqc_p2_lcl")
            if st.button("Check Answer", key="sqc_p2_btn"):
                c_bar = 4.0
                ucl = c_bar + 3 * math.sqrt(c_bar)
                lcl = max(0, c_bar - 3 * math.sqrt(c_bar))
                results = []
                results.append("✅ UCL correct" if check_answer(user_ucl, ucl) else f"❌ UCL should be {ucl:.2f}")
                results.append("✅ LCL correct" if check_answer(user_lcl, lcl) else f"❌ LCL should be {lcl:.2f}")
                for r in results:
                    st.write(r)

        with st.expander("🔴 P3: x̄ & R Interpretation (Hard)"):
            display_practice_problem(
                3, "Hard",
                "If the x̄-chart is in control but the R-chart is out of control, what does that mean?"
            )
            if st.button("Show Solution", key="sqc_p3_btn"):
                display_solution(
                    "The process average may be stable, but the process variability is not. "
                    "You should investigate special causes affecting dispersion first, because an unstable range chart "
                    "means the process spread is changing even if the center is not."
                )

        with st.expander("🔴 P4: Control Chart Selection (Hard)"):
            display_practice_problem(
                4, "Hard",
                "Which chart should be used for defect counts per unit?"
            )
            if st.button("Show Solution", key="sqc_p4_btn"):
                display_solution(
                    "Use a **c-chart** when counting defects per unit, assuming a constant area of opportunity and a Poisson-type count."
                )

# ============================================================
# MODULE 16: PROCESS CAPABILITY (Chapter 13) - ENHANCED V5.0
# ============================================================
def module_capability():
    display_header("🎯", "Chapter 13", "Process Capability — Cp & Cpk",
                   "Measuring process performance relative to specification limits")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Calculator", "📊 Distribution View"])

    with tab1:
        st.markdown("### Process Capability Indices")
        st.write(
            "**Process capability** compares process output to specification limits. "
            "**Cp** measures potential (if perfectly centered), while **Cpk** measures "
            "actual capability accounting for mean shift from center."
        )

        display_citation(
            "Working with our example in Exhibit 13.4, let's assume our process is centered "
            "at 1.251 and σ = 0.00083. Cpk = 1.6, which is the smaller number. This is a "
            "pretty good capability index because few defects will be produced by this process.",
            "Jacobs & Chase (2024, p. 374)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            display_formula_card("Cp — Potential",
                                 r"C_p = \frac{USL - LSL}{6\sigma}")
        with col2:
            display_formula_card("Cpu — Upper",
                                 r"C_{pu} = \frac{USL - \bar{X}}{3\sigma}")
        with col3:
            display_formula_card("Cpl — Lower",
                                 r"C_{pl} = \frac{\bar{X} - LSL}{3\sigma}")

        display_formula_card("Cpk — Actual",
                             r"C_{pk} = \min(C_{pu},\; C_{pl}) = \min\!\left(\frac{USL-\bar{X}}{3\sigma},\;\frac{\bar{X}-LSL}{3\sigma}\right)")

        st.markdown("#### Capability Interpretation")
        df_interp = pd.DataFrame({
            "Cpk Value":    ["< 1.00", "1.00–1.33", "1.33–1.67", "≥ 1.67"],
            "Assessment":   ["Not Capable", "Marginally Capable", "Capable", "Highly Capable / Six Sigma"],
            "Expected PPM": ["2,700+", "64–2,700", "0.6–64", "< 0.6"],
            "Action":       ["Immediate improvement", "Monitor closely", "Acceptable", "Excellent — benchmark"]
        })
        st.dataframe(df_interp, use_container_width=True)

        st.markdown("#### Taguchi Loss Function Insight")
        display_key_insight(
            "Cp vs. Cpk",
            "Cp = Cpk means the process is perfectly centered. A high Cp with low Cpk means "
            "the process has sufficient spread but the mean is off-center — re-centering "
            "alone (no investment) can dramatically reduce defects."
        )

        display_citation(
            "Taguchi argues that being within specification is not a yes/no decision, "
            "but rather a continuous function. The cost of variability increases as a parabolic "
            "function of deviation from the target — not a step function at the specification limits.",
            "Jacobs & Chase (2024, p. 370–371)"
        )

    with tab2:
        st.markdown("### Cpk Calculator")

        col1, col2 = st.columns(2)
        with col1:
            usl   = st.number_input("Upper Spec Limit (USL)", value=1.255, format="%.5f")
            lsl   = st.number_input("Lower Spec Limit (LSL)", value=1.245, format="%.5f")
            mean  = st.number_input("Process Mean (X̄)",       value=1.251, format="%.5f")
            sigma = st.number_input("Process Std Dev (σ)",     value=0.00083, format="%.6f",
                                    min_value=0.000001)

        with col2:
            cp  = (usl - lsl) / (6 * sigma)
            cpu = (usl - mean) / (3 * sigma)
            cpl = (mean - lsl)  / (3 * sigma)
            cpk = min(cpu, cpl)
            sigma_level = cpk * 3

            # Defect probability (both tails)
            z_upper = (usl - mean) / sigma
            z_lower = (mean - lsl)  / sigma
            p_defect = (1 - normal_cdf(z_upper)) + normal_cdf(-z_lower)
            ppm = p_defect * 1_000_000

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Cp  (Potential)", f"{cp:.3f}")
                st.metric("Cpu (Upper)",     f"{cpu:.3f}")
            with col_b:
                st.metric("Cpk (Actual)",    f"{cpk:.3f}",
                          delta="↑ Good" if cpk >= 1.33 else "↓ Needs work")
                st.metric("Cpl (Lower)",     f"{cpl:.3f}")

            st.metric("Sigma Level",   f"{sigma_level:.2f}σ")
            st.metric("Expected PPM",  f"{ppm:,.2f}")
            st.metric("% Centering",   f"{(cpk/cp*100):.1f}%" if cp > 0 else "—",
                      help="100% = perfectly centered")

            st.latex(rf"C_p = \frac{{{usl:.5f}-{lsl:.5f}}}{{6 \times {sigma:.5f}}} = {cp:.3f}")
            st.latex(rf"C_{{pk}} = \min({cpu:.3f},\; {cpl:.3f}) = {cpk:.3f}")

            if cpk >= 1.67:
                st.success("✅ Highly Capable — Excellent process")
            elif cpk >= 1.33:
                st.success("✅ Capable — Process meets standards")
            elif cpk >= 1.0:
                st.warning("⚠️ Marginally Capable — Monitor closely")
            else:
                st.error("❌ Not Capable — Immediate improvement required")

        # Cp vs Cpk comparison table
        st.markdown("#### Reference: Cpk → Expected Defect Rate")
        ref_data = pd.DataFrame({
            "Design Limits": ["±1σ", "±2σ", "±3σ (3-sigma)", "±4σ", "±5σ", "±6σ (Six Sigma)"],
            "Cpk":           [0.33, 0.67, 1.00, 1.33, 1.67, 2.00],
            "Defective PPM":  ["317,311", "45,500", "2,700", "63", "0.57", "0.002"],
            "Fraction Def.":  ["31.7%", "4.55%", "0.27%", "0.006%", "0.000057%", "0.0000002%"]
        })
        st.dataframe(ref_data, use_container_width=True)

    with tab3:
        st.markdown("### Process Distribution vs. Spec Limits")
        st.write("Visualize how process spread compares to specification limits.")

        col1, col2 = st.columns(2)
        with col1:
            v_usl   = st.number_input("USL",   value=1.255, format="%.5f", key="dv_usl")
            v_lsl   = st.number_input("LSL",   value=1.245, format="%.5f", key="dv_lsl")
            v_mean  = st.number_input("Mean",  value=1.251, format="%.5f", key="dv_m")
            v_sigma = st.number_input("σ",     value=0.00083, format="%.6f", key="dv_s",
                                      min_value=0.000001)

        # Build distribution plot
        x_min = v_mean - 4*v_sigma
        x_max = v_mean + 4*v_sigma
        x = [x_min + i*(x_max-x_min)/500 for i in range(501)]
        y = [math.exp(-0.5*((xi-v_mean)/v_sigma)**2)/(v_sigma*math.sqrt(2*math.pi)) for xi in x]

        # Split into regions: below LSL, between, above USL
        x_ok   = [xi for xi in x if v_lsl <= xi <= v_usl]
        y_ok   = [math.exp(-0.5*((xi-v_mean)/v_sigma)**2)/(v_sigma*math.sqrt(2*math.pi)) for xi in x_ok]
        x_lo   = [xi for xi in x if xi < v_lsl]
        y_lo   = [math.exp(-0.5*((xi-v_mean)/v_sigma)**2)/(v_sigma*math.sqrt(2*math.pi)) for xi in x_lo]
        x_hi   = [xi for xi in x if xi > v_usl]
        y_hi   = [math.exp(-0.5*((xi-v_mean)/v_sigma)**2)/(v_sigma*math.sqrt(2*math.pi)) for xi in x_hi]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_lo+[v_lsl], y=y_lo+[0], fill="tozeroy",
                                 mode="lines", line_color="#e74c3c", name="Defect (<LSL)",
                                 fillcolor="rgba(231,76,60,0.3)"))
        fig.add_trace(go.Scatter(x=[v_lsl]+x_ok+[v_usl], y=[0]+y_ok+[0], fill="tozeroy",
                                 mode="lines", line_color="#2ecc71", name="In-Spec",
                                 fillcolor="rgba(46,204,113,0.3)"))
        fig.add_trace(go.Scatter(x=[v_usl]+x_hi, y=[0]+y_hi, fill="tozeroy",
                                 mode="lines", line_color="#e74c3c", name="Defect (>USL)",
                                 fillcolor="rgba(231,76,60,0.3)"))
        fig.add_vline(x=v_lsl, line_dash="dash", line_color="red",
                      annotation_text="LSL", annotation_position="top left")
        fig.add_vline(x=v_usl, line_dash="dash", line_color="red",
                      annotation_text="USL")
        fig.add_vline(x=v_mean, line_dash="dot", line_color="blue",
                      annotation_text=f"X̄={v_mean:.5f}")

        v_cp  = (v_usl - v_lsl) / (6*v_sigma)
        v_cpu = (v_usl - v_mean) / (3*v_sigma)
        v_cpl = (v_mean - v_lsl) / (3*v_sigma)
        v_cpk = min(v_cpu, v_cpl)

        fig.update_layout(title=f"Process Distribution | Cp={v_cp:.2f}, Cpk={v_cpk:.2f}",
                          xaxis_title="Measurement", yaxis_title="Probability Density",
                          template="plotly_white", height=420, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        with col2:
            v_ppm = ((1-normal_cdf((v_usl-v_mean)/v_sigma)) +
                     normal_cdf((v_lsl-v_mean)/v_sigma)) * 1_000_000
            st.metric("Cp",         f"{v_cp:.3f}")
            st.metric("Cpk",        f"{v_cpk:.3f}")
            st.metric("Expected PPM", f"{v_ppm:,.2f}")

            if abs(v_mean - (v_usl+v_lsl)/2) > 0.01*(v_usl-v_lsl):
                st.info("💡 Process mean is off-center. Re-centering could "
                        f"improve Cpk from {v_cpk:.2f} → {v_cp:.2f}.")


# ============================================================
# MODULE 17: ACCEPTANCE SAMPLING (Chapter 13) - ENHANCED V5.0
# ============================================================
def module_sampling():
    display_header("📊", "Chapter 13", "Acceptance Sampling",
                   "Statistical lot acceptance using OC curves and sampling plans")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Sampling Plan", "📈 OC Curve"])

    with tab1:
        st.markdown("### Acceptance Sampling Theory")
        st.write(
            "**Acceptance sampling** evaluates batches of existing products to determine "
            "conformance. It is used when 100% inspection is impractical, costly, or destructive."
        )

        display_citation(
            "A single sampling plan is defined by n and c, where n is the number of units in "
            "the sample and c is the acceptance number — the maximum number of defective items "
            "found before the lot is rejected.",
            "Jacobs & Chase (2024, p. 384)"
        )

        st.markdown("#### Four Key Parameters")
        params = pd.DataFrame({
            "Parameter": ["AQL", "LTPD", "α (Alpha)", "β (Beta)"],
            "Full Name":  ["Acceptable Quality Level", "Lot Tolerance Percent Defective",
                           "Producer's Risk", "Consumer's Risk"],
            "Definition": ["Max defect rate considered acceptable",
                           "Defect rate that should be rejected",
                           "Prob. of rejecting a good lot (Type I)",
                           "Prob. of accepting a bad lot (Type II)"],
            "Typical Value": ["1–5%", "5–15%", "0.05 (5%)", "0.10 (10%)"]
        })
        st.dataframe(params, use_container_width=True)

        st.markdown("#### Textbook Sampling Plan Table (n·AQL at α=0.05, β=0.10)")
        exhibit_data = pd.DataFrame({
            "c": [0, 1, 2, 3, 4, 5, 6, 7],
            "LTPD/AQL": [44.890, 10.946, 6.509, 4.890, 4.057, 3.549, 3.206, 2.957],
            "n·AQL":    [0.052, 0.355, 0.818, 1.366, 1.970, 2.613, 3.286, 3.981]
        })
        st.dataframe(exhibit_data, use_container_width=True)

        st.markdown("**Example (from textbook):** AQL=2%, LTPD=8% → LTPD/AQL=4.0 → use c=4, n=99")

        display_key_insight(
            "Lot Size Effect",
            "The size of the lot (N) has relatively little effect on sampling protection. "
            "Whether inspecting from 200 units or 2,000 units, a sample of n=20 gives "
            "approximately the same probability of acceptance for the same defect rate."
        )

    with tab2:
        st.markdown("### Sampling Plan Calculator")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sampling Plan Parameters")
            n = st.number_input("Sample Size (n)", value=99, min_value=1)
            c = st.number_input("Acceptance Number (c)", value=4, min_value=0)
            p_test = st.number_input("Lot Fraction Defective (p)", value=0.05,
                                     min_value=0.0, max_value=1.0, format="%.3f")
            aql  = st.number_input("AQL",  value=0.02, format="%.3f")
            ltpd = st.number_input("LTPD", value=0.08, format="%.3f")

        with col2:
            np_val   = n * p_test
            p_accept = sum(poisson_pmf(k, np_val) for k in range(int(c) + 1))

            np_aql   = n * aql
            p_aql    = sum(poisson_pmf(k, np_aql) for k in range(int(c) + 1))
            alpha    = 1 - p_aql  # producer's risk

            np_ltpd  = n * ltpd
            beta_val = sum(poisson_pmf(k, np_ltpd) for k in range(int(c) + 1))

            st.metric("P(Accept) at p entered",  f"{p_accept:.4f}")
            st.metric("P(Accept) at AQL",        f"{p_aql:.4f}")
            st.metric("Producer's Risk α",       f"{alpha:.4f}")
            st.metric("Consumer's Risk β (LTPD)", f"{beta_val:.4f}")

            if alpha <= 0.05:
                st.success("✅ Producer's risk ≤ 5% — plan protects supplier")
            else:
                st.warning(f"⚠️ Producer's risk = {alpha:.1%} > 5%")

            if beta_val <= 0.10:
                st.success("✅ Consumer's risk ≤ 10% — plan protects buyer")
            else:
                st.warning(f"⚠️ Consumer's risk = {beta_val:.1%} > 10%")

        # AOQ (Average Outgoing Quality) approximation
        st.markdown("#### Average Outgoing Quality (AOQ)")
        N_lot = st.number_input("Lot Size N", value=1000, min_value=100)
        aoq = p_test * p_accept * (N_lot - n) / N_lot
        st.metric("AOQ at current p", f"{aoq:.4f}",
                  help="Average fraction defective in accepted lots after sampling")
        st.info(f"📌 AOQ = p × P(Accept) × (N-n)/N = {p_test}×{p_accept:.3f}×{(N_lot-n)/N_lot:.3f} = {aoq:.4f}")

    with tab3:
        st.markdown("### Operating Characteristic (OC) Curve")
        st.write("The OC curve shows how well the sampling plan discriminates between good and bad lots.")

        col1, col2 = st.columns([1, 2])
        with col1:
            n_oc = st.number_input("Sample Size n", value=99,  key="oc_n")
            c_oc = st.number_input("Accept Number c", value=4, key="oc_c")
            aql_oc  = st.number_input("AQL",  value=0.02, format="%.3f", key="oc_aql")
            ltpd_oc = st.number_input("LTPD", value=0.08, format="%.3f", key="oc_ltpd")

        p_range   = [i/200 for i in range(1, 61)]
        pa_values = [sum(poisson_pmf(k, n_oc*p) for k in range(int(c_oc)+1)) for p in p_range]

        pa_aql_oc  = sum(poisson_pmf(k, n_oc*aql_oc)  for k in range(int(c_oc)+1))
        pa_ltpd_oc = sum(poisson_pmf(k, n_oc*ltpd_oc) for k in range(int(c_oc)+1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_range, y=pa_values, mode="lines",
                                 line=dict(color="royalblue", width=3), name="P(Accept)"))
        fig.add_trace(go.Scatter(x=[aql_oc], y=[pa_aql_oc], mode="markers",
                                 marker=dict(color="green", size=12, symbol="circle"),
                                 name=f"AQL ({aql_oc:.0%}): P={pa_aql_oc:.3f}"))
        fig.add_trace(go.Scatter(x=[ltpd_oc], y=[pa_ltpd_oc], mode="markers",
                                 marker=dict(color="red", size=12, symbol="circle"),
                                 name=f"LTPD ({ltpd_oc:.0%}): P={pa_ltpd_oc:.3f}"))
        fig.add_hline(y=1-0.05, line_dash="dot", line_color="green",
                      annotation_text="1-α (0.95)", annotation_position="right")
        fig.add_hline(y=0.10,   line_dash="dot", line_color="red",
                      annotation_text="β (0.10)", annotation_position="right")
        fig.add_vline(x=aql_oc,  line_dash="dash", line_color="green")
        fig.add_vline(x=ltpd_oc, line_dash="dash", line_color="red")

        fig.update_layout(title=f"OC Curve — n={n_oc}, c={c_oc}",
                          xaxis_title="Fraction Defective (p)",
                          yaxis_title="Probability of Acceptance",
                          xaxis_tickformat=".1%", yaxis_range=[0, 1.05],
                          template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("α (Producer's Risk)", f"{1-pa_aql_oc:.4f}")
            st.metric("β (Consumer's Risk)", f"{pa_ltpd_oc:.4f}")
            if 1-pa_aql_oc <= 0.05 and pa_ltpd_oc <= 0.10:
                st.success("✅ Sampling plan satisfies both AQL and LTPD requirements")
            else:
                st.warning("⚠️ Adjust n or c to satisfy both risk requirements")


# ============================================================
# MODULE 18: PARETO ANALYSIS (Chapter 13) - ENHANCED V5.0
# ============================================================
def module_pareto():
    display_header("📊", "Chapter 13", "Pareto Analysis",
                   "The 80/20 rule: focus on the vital few causes")

    tab1, tab2 = st.tabs(["📚 Theory", "🔬 Pareto Chart Builder"])

    with tab1:
        st.markdown("### Pareto Principle in Quality Management")
        st.write(
            "The **Pareto Principle** (80/20 rule) states that ~80% of effects come from ~20% "
            "of causes. In quality management, a small number of defect categories typically "
            "account for the majority of quality problems."
        )
        display_key_insight(
            "Vital Few vs. Trivial Many",
            "By ranking defect categories from highest to lowest frequency and drawing a "
            "cumulative percentage line, managers can identify which 20% of causes "
            "drive 80% of defects — and focus resources there first."
        )
        st.markdown("#### Steps to Build a Pareto Chart")
        st.markdown("""
        1. **Collect** defect data by category
        2. **Sort** categories from highest to lowest frequency
        3. **Compute** cumulative percentages
        4. **Draw** bars for frequency + line for cumulative %
        5. **Identify** the categories left of the 80% cumulative line — these are the **Vital Few**
        """)

    with tab2:
        st.markdown("### Pareto Chart Builder")

        num_categories = st.number_input("Number of Categories", 3, 10, 6)

        default_names = ["Wrong assembly", "Surface scratch", "Dimensional error",
                         "Missing part", "Weld defect", "Packaging damage", "Color fault",
                         "Contamination", "Label error", "Other"]
        default_freqs = [120, 85, 65, 40, 30, 18, 12, 8, 5, 3]

        categories = []
        cols = st.columns(2)
        for i in range(int(num_categories)):
            with cols[i % 2]:
                c2 = st.columns(2)
                with c2[0]:
                    name = st.text_input(f"Category {i+1}",
                                        value=default_names[i] if i < len(default_names) else f"Cat {i+1}",
                                        key=f"par_name_{i}")
                with c2[1]:
                    freq = st.number_input(f"Count {i+1}",
                                           value=default_freqs[i] if i < len(default_freqs) else 10,
                                           min_value=0, key=f"par_freq_{i}")
                categories.append({"Category": name, "Frequency": freq})

        df = pd.DataFrame(categories).sort_values("Frequency", ascending=False).reset_index(drop=True)
        total       = df["Frequency"].sum()
        df["Pct"]   = df["Frequency"] / total * 100
        df["Cum %"] = df["Pct"].cumsum()
        df["Rank"]  = range(1, len(df)+1)

        vital_few = df[df["Cum %"] <= 80]["Category"].tolist()
        if not vital_few:
            vital_few = [df.iloc[0]["Category"]]
        # Include the category that crosses 80%
        if len(vital_few) < len(df):
            vital_few.append(df.iloc[len(vital_few)]["Category"])

        # Plotly Pareto Chart
        fig = go.Figure()
        bar_colors = ["#2ecc71" if c in vital_few else "#bdc3c7" for c in df["Category"]]
        fig.add_trace(go.Bar(x=df["Category"], y=df["Frequency"],
                             name="Frequency", marker_color=bar_colors,
                             yaxis="y"))
        fig.add_trace(go.Scatter(x=df["Category"], y=df["Cum %"],
                                 name="Cumulative %", mode="lines+markers",
                                 line=dict(color="#e74c3c", width=2),
                                 marker=dict(size=8), yaxis="y2"))
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                      annotation_text="80% Line", yref="y2", annotation_position="right")

        fig.update_layout(
            title="Pareto Chart — Defect Categories",
            xaxis_title="Defect Category",
            yaxis=dict(title="Frequency", side="left"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                        range=[0, 110], ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            template="plotly_white", height=460, barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        df_display = df[["Rank","Category","Frequency","Pct","Cum %"]].copy()
        df_display["Pct"]   = df_display["Pct"].map("{:.1f}%".format)
        df_display["Cum %"] = df_display["Cum %"].map("{:.1f}%".format)
        df_display["Vital Few"] = df_display["Category"].apply(
            lambda x: "✅ Vital Few" if x in vital_few else "")
        st.dataframe(df_display, use_container_width=True)

        vital_pct = df[df["Category"].isin(vital_few)]["Frequency"].sum() / total * 100
        st.success(
            f"🎯 **Vital Few ({len(vital_few)} categories = "
            f"{len(vital_few)/len(df)*100:.0f}% of categories):** "
            f"{', '.join(vital_few)} — account for **{vital_pct:.1f}%** of all defects."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Defects", f"{total:,}")
        col2.metric("Vital Few Count", len(vital_few))
        col3.metric("% Accounted For", f"{vital_pct:.1f}%")


# ============================================================
# MODULE 19: FISHBONE DIAGRAM (Chapter 13) - ENHANCED V5.0
# ============================================================
def module_fishbone():
    display_header("🐟", "Chapter 13", "Fishbone / Ishikawa Diagram",
                   "Systematic cause-and-effect analysis for root cause identification")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Builder", "📋 Service 4S"])

    with tab1:
        st.markdown("### Cause-and-Effect Diagram")
        st.write(
            "The **Fishbone Diagram** (invented by Kaoru Ishikawa) visually organizes "
            "potential causes of a problem into major categories, making it easier for "
            "teams to systematically explore root causes rather than jumping to solutions."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 6M Framework (Manufacturing)")
            mfg_6m = pd.DataFrame({
                "Category":    ["Man", "Machine", "Method", "Material", "Measurement", "Mother Nature"],
                "Focus Area":  ["Skills, training, fatigue", "Equipment, tooling, calibration",
                                "Process steps, SOPs", "Raw materials, components",
                                "Gauges, test methods", "Temperature, humidity, environment"]
            })
            st.dataframe(mfg_6m, use_container_width=True)
        with col2:
            st.markdown("#### 4S Framework (Service)")
            svc_4s = pd.DataFrame({
                "Category":   ["Surroundings", "Suppliers", "Systems", "Skills"],
                "Focus Area": ["Environment, layout, ergonomics",
                               "Inputs, materials, vendors",
                               "Procedures, IT, policies",
                               "Training, competency, attitude"]
            })
            st.dataframe(svc_4s, use_container_width=True)

        display_key_insight(
            "How to Use",
            "Start with the problem (effect) on the right. Draw the 'spine'. Add bones "
            "for each major category. Brainstorm sub-causes as smaller branches. "
            "Use the 5 Whys technique to drill into each branch."
        )

    with tab2:
        st.markdown("### 6M Fishbone Builder")
        problem = st.text_input("Problem / Effect Statement", value="High Defect Rate in Assembly")

        st.markdown("#### Enter Causes by Category (one per line)")
        categories_6m = {
            "👷 Man":         "Untrained operators\nFatigue from long shifts",
            "⚙️ Machine":    "Worn tooling\nMachinery vibration",
            "📋 Method":     "Outdated SOP\nNo incoming inspection",
            "📦 Material":   "Inconsistent supplier quality\nWrong spec material",
            "📏 Measurement":"Uncalibrated gauges\nSampling too infrequent",
            "🌡️ Environment":"High humidity warping parts\nPoor lighting"
        }

        cause_data = {}
        col1, col2 = st.columns(2)
        cats = list(categories_6m.keys())
        for i, cat in enumerate(cats):
            with (col1 if i % 2 == 0 else col2):
                cause_data[cat] = st.text_area(cat, value=categories_6m[cat],
                                               height=90, key=f"fb_{i}")

        st.markdown(f"---\n### 🐟 Fishbone: _{problem}_")
        all_causes = []
        for cat, text in cause_data.items():
            items = [c.strip() for c in text.split("\n") if c.strip()]
            if items:
                st.markdown(f"**{cat}:** {' · '.join(items)}")
                all_causes.extend(items)

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Categories Used", sum(1 for t in cause_data.values() if t.strip()))
        col2.metric("Total Causes Identified", len(all_causes))

        if all_causes:
            st.info("💡 Next step: Use **5-Why Analysis** on the most likely causes. "
                    "Ask 'Why?' up to 5 times to find the true root cause.")

    with tab3:
        st.markdown("### 4S Framework — Service Fishbone")
        problem_s = st.text_input("Service Problem", value="Long Customer Wait Times",
                                  key="fb_svc_prob")
        svc_cats = {
            "🏢 Surroundings": "Cramped waiting area\nPoor signage",
            "🤝 Suppliers":    "Late vendor deliveries\nIncomplete information from upstream",
            "🖥️ Systems":     "Slow software\nNo triage protocol",
            "🎓 Skills":       "Undertrained staff\nNo cross-training"
        }
        col1, col2 = st.columns(2)
        svc_cause_data = {}
        svc_cats_list = list(svc_cats.keys())
        for i, cat in enumerate(svc_cats_list):
            with (col1 if i % 2 == 0 else col2):
                svc_cause_data[cat] = st.text_area(cat, value=svc_cats[cat],
                                                    height=90, key=f"fbs_{i}")

        st.markdown(f"---\n### 🐟 Service Fishbone: _{problem_s}_")
        for cat, text in svc_cause_data.items():
            items = [c.strip() for c in text.split("\n") if c.strip()]
            if items:
                st.markdown(f"**{cat}:** {' · '.join(items)}")


# ============================================================
# MODULE 20: LEAN SUPPLY CHAINS (Chapter 14) - ENHANCED V5.0
# ============================================================
def module_lean():
    display_header("🔄", "Chapter 14", "Lean Supply Chains",
                   "Eliminating waste, maximizing flow, and building pull systems")

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "♻️ Waste Analyzer",
                                       "🔬 Lead Time Calculator", "🗺️ VSM Simulator"])

    with tab1:
        st.markdown("### Lean Production Principles")
        st.write(
            "**Lean production** is an integrated system designed to achieve high-volume "
            "output using minimal inventory at every stage. Derived from the Toyota Production "
            "System (TPS), it focuses on continuous flow, pull scheduling, and relentless "
            "waste elimination."
        )

        st.markdown("#### The 8 Wastes (TIMWOODD)")
        wastes = [
            ("🚚", "Transportation",  "Unnecessary movement of materials between locations"),
            ("📦", "Inventory",       "Excess stock beyond immediate needs ties up capital"),
            ("🏃", "Motion",          "Unnecessary movement by workers (reaching, walking)"),
            ("⏳", "Waiting",          "Idle time between operations — WIP waiting for resources"),
            ("🔁", "Overproduction",  "Making more or earlier than needed — worst waste"),
            ("🔧", "Overprocessing",  "Doing more work than the customer values"),
            ("❌", "Defects",          "Rework, scrap, and inspection of non-conforming output"),
            ("🧠", "Non-used Talent", "Failing to leverage employee knowledge and creativity"),
        ]
        for icon, name, desc in wastes:
            display_concept_card(icon, name, desc)

        display_textbook_content(
            "Value Stream Mapping",
            "VSM is a visual tool to analyze existing systems and find waste elimination "
            "opportunities. Maps show material and information flows from supplier to customer. "
            "Note that the lead time in the new system is only five days compared to the "
            "34-day lead time in the old system."
        )

        st.markdown("#### Lean vs. Traditional Comparison")
        comparison = pd.DataFrame({
            "Dimension":     ["Lot sizes", "Inventory", "Supplier relations",
                              "Setup times", "Quality", "Workforce"],
            "Traditional":   ["Large batches", "Buffer stock", "Many suppliers; arm's length",
                              "Long, tolerated", "Inspect at end", "Specialized"],
            "Lean/JIT":      ["Small (ideally 1)", "Minimal JIT", "Few; long-term partners",
                              "Minimized (<10 min)", "Zero defects at source", "Multi-skilled"]
        })
        st.dataframe(comparison, use_container_width=True)

    with tab2:
        st.markdown("### Waste Cost Analyzer")
        st.write("Estimate the hidden cost of waste in your process.")

        col1, col2 = st.columns(2)
        with col1:
            annual_sales   = st.number_input("Annual Sales ($)", value=5_000_000, step=100_000)
            defect_rate    = st.slider("Defect / Scrap Rate (%)", 0.0, 20.0, 3.0, 0.1)
            inv_turns      = st.slider("Inventory Turns/Year", 1, 52, 4)
            target_turns   = st.slider("Target Inventory Turns", 1, 52, 12)
            overproduction = st.slider("Overproduction (%)", 0.0, 30.0, 5.0, 0.5)

        with col2:
            cogs          = annual_sales * 0.6
            defect_cost   = cogs * defect_rate / 100
            current_inv   = cogs / inv_turns
            target_inv    = cogs / target_turns
            inv_reduction = current_inv - target_inv
            carrying_rate = 0.25  # 25% carrying cost
            inv_saving    = inv_reduction * carrying_rate
            op_cost       = cogs * overproduction / 100 * 0.05  # 5% of overproduced goods wasted

            total_waste   = defect_cost + inv_saving + op_cost

            st.metric("Defect Cost (annual)",     f"${defect_cost:,.0f}")
            st.metric("Inventory Reduction Possible", f"${inv_reduction:,.0f}")
            st.metric("Carrying Cost Saving",     f"${inv_saving:,.0f}")
            st.metric("Overproduction Cost",      f"${op_cost:,.0f}")
            st.metric("💰 Total Waste Opportunity", f"${total_waste:,.0f}",
                      delta=f"{total_waste/annual_sales*100:.1f}% of revenue")

    with tab3:
        st.markdown("### Lead Time Reduction Calculator")

        col1, col2 = st.columns(2)
        with col1:
            rm_days  = st.number_input("RM Inventory (days)",  value=10)
            wip_days = st.number_input("WIP Inventory (days)", value=14)
            fg_days  = st.number_input("FG Inventory (days)",  value=10)
            proc_days = st.number_input("Processing Time (days)", value=3)
            current_lt = rm_days + wip_days + fg_days + proc_days
            st.metric("Current Lead Time", f"{current_lt} days")

        with col2:
            rm_red  = st.slider("RM Reduction %",  0, 100, 85)
            wip_red = st.slider("WIP Reduction %", 0, 100, 85)
            fg_red  = st.slider("FG Reduction %",  0, 100, 85)

            future_rm   = rm_days  * (1 - rm_red/100)
            future_wip  = wip_days * (1 - wip_red/100)
            future_fg   = fg_days  * (1 - fg_red/100)
            future_lt   = future_rm + future_wip + future_fg + proc_days
            reduction   = (1 - future_lt/current_lt) * 100 if current_lt > 0 else 0

            st.metric("Future Lead Time",    f"{future_lt:.1f} days")
            st.metric("Lead Time Reduction", f"{reduction:.0f}%",
                      delta=f"−{current_lt-future_lt:.1f} days")

            display_citation(
                "Note that the lead time in the new system is only five days, compared to the "
                "34-day lead time with the old system.",
                "Jacobs & Chase (2024, Ch. 14)"
            )

        # Bar chart before/after
        fig = go.Figure(data=[
            go.Bar(name="RM",        x=["Current","Future"],
                   y=[rm_days, future_rm],       marker_color="#3498db"),
            go.Bar(name="WIP",       x=["Current","Future"],
                   y=[wip_days, future_wip],      marker_color="#e67e22"),
            go.Bar(name="FG",        x=["Current","Future"],
                   y=[fg_days, future_fg],        marker_color="#2ecc71"),
            go.Bar(name="Processing",x=["Current","Future"],
                   y=[proc_days, proc_days],      marker_color="#9b59b6"),
        ])
        fig.update_layout(barmode="stack", title="Lead Time Breakdown: Before vs. After Lean",
                          yaxis_title="Days", template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Value Stream Map Simulator")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Current State")
            stages = ["Supplier→RM", "Process 1", "WIP 1→2", "Process 2", "WIP 2→FG", "Ship"]
            times  = [5, 2, 8, 3, 7, 1]
            va_flags = [False, True, False, True, False, False]
        with col2:
            st.markdown("#### Map Legend")
            st.write("🟢 Value-Added | 🔴 Non-Value-Added")

        vsm_df = pd.DataFrame({
            "Stage": stages,
            "Time (days)": times,
            "Value Added?": ["✅ VA" if v else "❌ NVA" for v in va_flags]
        })
        st.dataframe(vsm_df, use_container_width=True)

        total_time = sum(times)
        va_time    = sum(t for t, v in zip(times, va_flags) if v)
        pv_ratio   = va_time / total_time * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Lead Time",    f"{total_time} days")
        col2.metric("Value-Added Time",   f"{va_time} days")
        col3.metric("Process Velocity",   f"{pv_ratio:.1f}%",
                    help="VA Time / Total Lead Time")

        st.warning(f"⚠️ Only **{pv_ratio:.0f}%** of time is value-added. "
                   f"**{100-pv_ratio:.0f}%** is waste — target for elimination.")


# ============================================================
# MODULE 21: CENTROID METHOD (Chapter 15) - ENHANCED V5.0
# ============================================================
def module_centroid():
    display_header("📍", "Chapter 15", "Centroid Method",
                   "Weighted center of gravity for optimal facility location")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Calculator + Map", "📊 Sensitivity"])

    with tab1:
        st.markdown("### Weighted Center of Gravity")
        st.write(
            "The **Centroid Method** finds the optimal single-facility location that "
            "minimizes total weighted distance. It weights each location by shipment "
            "volume, placing the facility closer to high-volume destinations."
        )
        display_citation(
            "The centroid method considers existing facilities, the distances between them, "
            "and the volumes of goods to be shipped.",
            "Jacobs & Chase (2024, p. 456)"
        )
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("X Coordinate",
                                 r"C_x = \frac{\sum_{i=1}^{n} d_{ix} \cdot V_i}{\sum_{i=1}^{n} V_i}")
        with col2:
            display_formula_card("Y Coordinate",
                                 r"C_y = \frac{\sum_{i=1}^{n} d_{iy} \cdot V_i}{\sum_{i=1}^{n} V_i}")

        display_key_insight("Assumption",
            "The centroid minimizes total ton-miles (volume × distance). It ignores road "
            "networks, costs per mile, and site-specific constraints — use it as a starting "
            "point, then refine with Factor Rating.")

    with tab2:
        st.markdown("### Centroid Calculator")
        num_locations = st.number_input("Number of Locations", 2, 8, 4)

        default_locs = [
            ("Chicago",     30,  120, 1500),
            ("Atlanta",     75,   60, 1200),
            ("Los Angeles", 10,   80,  800),
            ("New York",   130,  110,  900),
            ("Dallas",      55,   50,  600),
            ("Seattle",      5,  140,  400),
            ("Miami",       90,   30,  700),
            ("Denver",      30,   90, 1000),
        ]

        locations = []
        for i in range(int(num_locations)):
            cols = st.columns(4)
            name, dx, dy, dv = default_locs[i] if i < len(default_locs) else (f"Loc{i+1}", 50+i*10, 70+i*10, 500)
            with cols[0]: name_i = st.text_input(f"Location {i+1}", value=name, key=f"cent_name_{i}")
            with cols[1]: x_i    = st.number_input(f"X {i+1}", value=dx, key=f"cent_x_{i}")
            with cols[2]: y_i    = st.number_input(f"Y {i+1}", value=dy, key=f"cent_y_{i}")
            with cols[3]: v_i    = st.number_input(f"Volume {i+1}", value=dv, step=50, key=f"cent_v_{i}")
            locations.append({"name": name_i, "x": x_i, "y": y_i, "v": v_i})

        total_v = sum(loc["v"] for loc in locations)
        if total_v > 0:
            cx = sum(loc["x"] * loc["v"] for loc in locations) / total_v
            cy = sum(loc["y"] * loc["v"] for loc in locations) / total_v

            col1, col2, col3 = st.columns(3)
            col1.metric("Optimal X (Cx)", f"{cx:.2f}")
            col2.metric("Optimal Y (Cy)", f"{cy:.2f}")
            col3.metric("Total Volume",   f"{total_v:,}")

            # Plotly scatter map
            fig = go.Figure()
            sizes = [loc["v"]/max(l["v"] for l in locations)*40+10 for loc in locations]
            fig.add_trace(go.Scatter(
                x=[loc["x"] for loc in locations],
                y=[loc["y"] for loc in locations],
                mode="markers+text",
                marker=dict(size=sizes, color="#3498db", opacity=0.7,
                            line=dict(width=1, color="white")),
                text=[f"{loc['name']}<br>({loc['v']:,})" for loc in locations],
                textposition="top center", name="Demand Locations"
            ))
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(size=20, color="#e74c3c", symbol="star"),
                text=["⭐ Optimal Location"], textposition="top right",
                name="Centroid (Optimal)"
            ))
            # Lines from centroid to each location
            for loc in locations:
                fig.add_shape(type="line", x0=cx, y0=cy, x1=loc["x"], y1=loc["y"],
                              line=dict(color="gray", width=1, dash="dot"))

            fig.update_layout(title="Centroid Location Map",
                              xaxis_title="X Coordinate",
                              yaxis_title="Y Coordinate",
                              template="plotly_white", height=480,
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Distance from centroid to each location
            locs_df = pd.DataFrame(locations)
            locs_df["Distance to Centroid"] = locs_df.apply(
                lambda r: math.sqrt((r["x"]-cx)**2 + (r["y"]-cy)**2), axis=1)
            locs_df["Ton-Miles"] = locs_df["Distance to Centroid"] * locs_df["v"]
            locs_df = locs_df.rename(columns={"name":"Location","x":"X","y":"Y","v":"Volume"})
            st.dataframe(locs_df.round(2), use_container_width=True)
            st.metric("Total Ton-Miles", f"{locs_df['Ton-Miles'].sum():,.0f}")

    with tab3:
        st.markdown("### Volume Sensitivity Analysis")
        st.write("How does the optimal location shift as volumes at each site change?")
        if 'locations' in dir():
            vol_factor = st.slider("Scale volume at Location 1 (%)", 50, 300, 100, step=10)
            locs_adj = [dict(l) for l in locations]
            locs_adj[0]["v"] = int(locations[0]["v"] * vol_factor / 100)
            tv_adj = sum(l["v"] for l in locs_adj)
            cx_adj = sum(l["x"]*l["v"] for l in locs_adj)/tv_adj if tv_adj>0 else cx
            cy_adj = sum(l["y"]*l["v"] for l in locs_adj)/tv_adj if tv_adj>0 else cy
            col1, col2 = st.columns(2)
            col1.metric("New Cx", f"{cx_adj:.2f}", delta=f"{cx_adj-cx:+.2f}")
            col2.metric("New Cy", f"{cy_adj:.2f}", delta=f"{cy_adj-cy:+.2f}")


# ============================================================
# MODULE 22: FACTOR RATING (Chapter 15) - ENHANCED V5.0
# ============================================================
def module_factor():
    display_header("⚖️", "Chapter 15", "Factor Rating Method",
                   "Multi-criteria weighted scoring for location decisions")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Calculator", "📊 Radar Chart"])

    with tab1:
        st.markdown("### Factor Rating Method")
        st.write(
            "The **Factor Rating Method** applies numeric weights to decision factors "
            "and scores each location candidate. It makes qualitative judgments explicit "
            "and allows sensitivity analysis to see how robust the recommendation is."
        )
        display_formula_card("Weighted Score",
                             r"\text{Score}_j = \sum_{i=1}^{n} w_i \times s_{ij}")
        st.markdown("**Steps:**")
        st.markdown("""
        1. Identify decision **factors** (e.g., labor cost, proximity to market)
        2. Assign **weights** that sum to 1.0 based on relative importance
        3. **Score** each location (0–100) on each factor
        4. Compute **weighted score** = Σ(weight × score)
        5. Select the **highest scoring** location
        """)
        display_key_insight(
            "Weight Sensitivity",
            "Run the analysis multiple times with different weights to test robustness. "
            "If the winner changes with small weight adjustments, collect more data "
            "before finalizing the decision."
        )

    with tab2:
        st.markdown("### Factor Rating Calculator")

        n_factors = st.number_input("Number of Factors", 3, 8, 5)
        n_locs    = st.selectbox("Number of Locations", [2, 3, 4], index=1)

        default_factors = ["Labor Cost", "Market Proximity", "Tax Environment",
                           "Infrastructure Quality", "Land & Building Cost",
                           "Labor Availability", "Quality of Life", "Incentives"]
        default_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.05, 0.02]
        default_scores  = {
            0: [70, 85, 60, 75, 80],
            1: [80, 70, 75, 65, 70],
            2: [60, 90, 80, 80, 65],
            3: [75, 65, 70, 85, 55],
        }
        loc_names = ["Location A", "Location B", "Location C", "Location D"]

        data = []
        st.markdown("#### Weights & Scores (0–100)")
        header_cols = st.columns([2.5, 1.5] + [1.5]*n_locs)
        header_cols[0].write("**Factor**")
        header_cols[1].write("**Weight**")
        for j in range(n_locs):
            header_cols[2+j].write(f"**{loc_names[j]}**")

        for i in range(int(n_factors)):
            row_cols = st.columns([2.5, 1.5] + [1.5]*n_locs)
            with row_cols[0]: st.write(default_factors[i] if i < len(default_factors) else f"Factor {i+1}")
            with row_cols[1]:
                w = st.number_input("w", value=default_weights[i] if i < len(default_weights) else round(1/n_factors,2),
                                    min_value=0.0, max_value=1.0, format="%.2f", key=f"fr_w_{i}",
                                    label_visibility="collapsed")
            scores = []
            for j in range(n_locs):
                with row_cols[2+j]:
                    s = st.number_input("s", value=default_scores.get(j,[75]*8)[i] if i<8 else 70,
                                        min_value=0, max_value=100, key=f"fr_s_{i}_{j}",
                                        label_visibility="collapsed")
                    scores.append(s)
            data.append({"Factor": default_factors[i] if i < len(default_factors) else f"F{i+1}",
                         "Weight": w, **{loc_names[j]: scores[j] for j in range(n_locs)}})

        df_fr = pd.DataFrame(data)
        total_w  = df_fr["Weight"].sum()
        weighted = {loc_names[j]: sum(row["Weight"]*row[loc_names[j]] for _, row in df_fr.iterrows())
                    for j in range(n_locs)}

        st.markdown("---")
        if abs(total_w - 1.0) > 0.05:
            st.warning(f"⚠️ Weights sum to {total_w:.2f} — should sum to 1.00")

        result_cols = st.columns(n_locs + 1)
        result_cols[0].metric("Total Weight", f"{total_w:.2f}")
        for j in range(n_locs):
            result_cols[j+1].metric(f"{loc_names[j]} Score", f"{weighted[loc_names[j]]:.1f}")

        best = max(weighted, key=weighted.get)
        second = sorted(weighted, key=weighted.get, reverse=True)[1] if n_locs > 1 else None
        margin = weighted[best] - weighted[second] if second else 0
        st.success(f"📍 **Recommendation: {best}** (score {weighted[best]:.1f}) — "
                   f"leads {second} by {margin:.1f} points")

        if margin < 3:
            st.warning("⚠️ Small margin — run sensitivity analysis before deciding.")

    with tab3:
        st.markdown("### Radar Chart — Location Comparison")
        if 'data' in dir() and data:
            factor_labels = [d["Factor"] for d in data]
            fig = go.Figure()
            colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
            for j in range(n_locs):
                vals = [d[loc_names[j]] for d in data] + [data[0][loc_names[j]]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=factor_labels+[factor_labels[0]],
                    fill="toself", name=loc_names[j],
                    line_color=colors[j], opacity=0.6
                ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                              title="Factor Score Comparison", template="plotly_white",
                              height=500)
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODULE 23: TRANSPORTATION METHOD (Chapter 15) - ENHANCED V5.0
# ============================================================
def module_transportation():
    display_header("🚚", "Chapter 15", "Transportation Method",
                   "Optimal supply-to-demand allocation minimizing total cost")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Northwest Corner", "💲 Cost Calculator"])

    with tab1:
        st.markdown("### Transportation Problem")
        st.write(
            "The **Transportation Method** allocates supply from multiple sources to "
            "multiple destinations at minimum total cost. It is a special case of linear "
            "programming widely used in distribution network design."
        )

        display_formula_card("Objective",
                             r"\text{Minimize } Z = \sum_{i=1}^{m}\sum_{j=1}^{n} c_{ij} \cdot x_{ij}")
        display_formula_card("Supply Constraint",
                             r"\sum_{j=1}^{n} x_{ij} \leq S_i \quad \forall i")
        display_formula_card("Demand Constraint",
                             r"\sum_{i=1}^{m} x_{ij} \geq D_j \quad \forall j")

        st.markdown("#### Solution Methods")
        methods = pd.DataFrame({
            "Method": ["Northwest Corner", "Minimum Cost", "Vogel's Approximation (VAM)", "MODI"],
            "Type":   ["Initial BFS", "Initial BFS", "Initial BFS", "Optimality Check"],
            "Quality": ["Basic — high cost", "Good starting point", "Best initial BFS", "Optimal"],
            "Use":    ["Quick start", "Improved start", "Best heuristic start", "Final optimization"]
        })
        st.dataframe(methods, use_container_width=True)

    with tab2:
        st.markdown("### Northwest Corner Method")
        st.write("Start in the upper-left cell; allocate as much as possible before moving right or down.")

        n_sources = st.selectbox("Sources (m)", [2, 3], index=1)
        n_dests   = st.selectbox("Destinations (n)", [2, 3, 4], index=1)

        default_supply  = [300, 400, 200]
        default_demand  = [250, 350, 200, 100]
        default_costs   = [[2, 3, 1, 5], [7, 3, 4, 6], [4, 5, 2, 3]]

        st.markdown("#### Supply")
        supply = []
        s_cols = st.columns(n_sources)
        for i in range(n_sources):
            with s_cols[i]:
                supply.append(st.number_input(f"S{i+1}", value=default_supply[i], key=f"tr_s_{i}"))

        st.markdown("#### Demand")
        demand = []
        d_cols = st.columns(n_dests)
        for j in range(n_dests):
            with d_cols[j]:
                demand.append(st.number_input(f"D{j+1}", value=default_demand[j], key=f"tr_d_{j}"))

        st.markdown("#### Unit Costs ($)")
        costs = []
        for i in range(n_sources):
            row_cols = st.columns(n_dests)
            cost_row = []
            for j in range(n_dests):
                with row_cols[j]:
                    c = st.number_input(f"c{i+1}{j+1}", value=default_costs[i][j] if i<3 and j<4 else 5,
                                        key=f"tr_c_{i}_{j}", label_visibility="visible")
                    cost_row.append(c)
            costs.append(cost_row)

        # Balance check
        total_supply = sum(supply)
        total_demand = sum(demand)
        if total_supply != total_demand:
            st.warning(f"⚠️ Unbalanced: Supply={total_supply}, Demand={total_demand}. "
                       "Add a dummy row/column to balance.")

        # Northwest Corner algorithm
        s = supply[:]
        d = demand[:]
        alloc = [[0]*n_dests for _ in range(n_sources)]
        i, j = 0, 0
        while i < n_sources and j < n_dests:
            amt = min(s[i], d[j])
            alloc[i][j] = amt
            s[i] -= amt
            d[j] -= amt
            if s[i] == 0 and i < n_sources-1:
                i += 1
            elif d[j] == 0 and j < n_dests-1:
                j += 1
            else:
                break

        # Display allocation table
        alloc_df = pd.DataFrame(alloc,
                                columns=[f"D{j+1}" for j in range(n_dests)],
                                index=[f"S{i+1}" for i in range(n_sources)])
        alloc_df["Supply"] = supply
        alloc_df.loc["Demand"] = demand + [sum(demand)]
        st.dataframe(alloc_df, use_container_width=True)

        total_cost = sum(alloc[i][j]*costs[i][j]
                         for i in range(n_sources) for j in range(n_dests))
        st.metric("NW Corner Total Cost", f"${total_cost:,}")
        st.info("💡 Northwest Corner gives a feasible but not optimal solution. "
                "Apply MODI method (stepping-stone) to reach optimality.")

    with tab3:
        st.markdown("### Cost Analysis")
        st.write("Enter any allocation to compute total transportation cost.")

        if 'alloc' in dir() and alloc:
            st.markdown("#### Edit Allocation Quantities")
            custom_alloc = []
            for i in range(n_sources):
                row = []
                row_cols = st.columns(n_dests)
                for j in range(n_dests):
                    with row_cols[j]:
                        val = st.number_input(f"x{i+1}{j+1}", value=alloc[i][j],
                                              min_value=0, key=f"ca_{i}_{j}")
                        row.append(val)
                custom_alloc.append(row)

            cost_breakdown = []
            for i in range(n_sources):
                for j in range(n_dests):
                    if custom_alloc[i][j] > 0:
                        cost_breakdown.append({
                            "Route": f"S{i+1}→D{j+1}",
                            "Units": custom_alloc[i][j],
                            "Unit Cost": costs[i][j],
                            "Total Cost": custom_alloc[i][j]*costs[i][j]
                        })
            breakdown_df = pd.DataFrame(cost_breakdown)
            if not breakdown_df.empty:
                st.dataframe(breakdown_df, use_container_width=True)
                st.metric("Custom Total Cost", f"${breakdown_df['Total Cost'].sum():,}")


# ============================================================
# MODULE 24: GLOBAL SOURCING (Chapter 16) - ENHANCED V5.0
# ============================================================
def module_sourcing():
    display_header("🌐", "Chapter 16", "Global Sourcing & Supply Chain Design",
                   "Total cost of ownership, risk, and strategic sourcing decisions")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 TCO Calculator", "⚖️ Risk Framework"])

    with tab1:
        st.markdown("### Strategic Sourcing")
        st.write(
            "**Strategic sourcing** optimizes the entire supply base for lowest total cost "
            "of ownership — not just purchase price. Decisions involve make vs. buy, "
            "domestic vs. offshore, single vs. multiple sourcing."
        )

        st.markdown("#### Functional vs. Innovative Products (Fisher's Framework)")
        df_fi = pd.DataFrame({
            "Characteristic": ["Demand pattern", "Product life cycle", "Profit margin",
                               "Forecast error", "Stockout cost", "Supply focus"],
            "Functional Products": ["Predictable", "Long (years)", "Low (5–20%)",
                                    "Low (<10%)", "Low", "Efficiency/lowest cost"],
            "Innovative Products": ["Unpredictable", "Short (months)", "High (20–60%)",
                                    "High (40–100%)", "High", "Responsiveness/speed"]
        })
        st.dataframe(df_fi, use_container_width=True)

        st.markdown("#### Sourcing Strategy Matrix")
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("🏭", "Make (Vertical Integration)",
                                 "Core competencies; high volume; proprietary. Higher control, higher investment.")
        with col2:
            display_concept_card("🤝", "Buy (Outsourcing)",
                                 "Non-core; commodity; supplier expertise available. Lower cost, less control.")

        display_key_insight("Total Cost of Ownership",
            "Purchase price is typically only 25–40% of TCO. Hidden costs include "
            "transportation, tariffs, quality inspection, inventory holding, lead-time "
            "variability, and supply disruption risk.")

    with tab2:
        st.markdown("### TCO Calculator — Domestic vs. Offshore")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏠 Domestic Supplier")
            dom_price    = st.number_input("Unit Purchase Price ($)",   value=100.0, key="d_price")
            dom_ship     = st.number_input("Shipping Cost/Unit ($)",    value=5.0,   key="d_ship")
            dom_tariff   = st.number_input("Tariff/Import Duty ($)",    value=0.0,   key="d_tariff")
            dom_quality  = st.number_input("Quality Cost/Unit ($)",     value=2.0,   key="d_qual")
            dom_inv      = st.number_input("Inventory Carrying/Unit ($)",value=3.0,  key="d_inv")
            dom_risk     = st.number_input("Risk & Disruption/Unit ($)", value=1.0,  key="d_risk")
            dom_total    = dom_price + dom_ship + dom_tariff + dom_quality + dom_inv + dom_risk

        with col2:
            st.markdown("#### 🌏 Offshore Supplier")
            ovs_price    = st.number_input("Unit Purchase Price ($)",   value=65.0,  key="o_price")
            ovs_ship     = st.number_input("Shipping Cost/Unit ($)",    value=18.0,  key="o_ship")
            ovs_tariff   = st.number_input("Tariff/Import Duty ($)",    value=8.0,   key="o_tariff")
            ovs_quality  = st.number_input("Quality Cost/Unit ($)",     value=6.0,   key="o_qual")
            ovs_inv      = st.number_input("Inventory Carrying/Unit ($)",value=12.0, key="o_inv")
            ovs_risk     = st.number_input("Risk & Disruption/Unit ($)", value=7.0,  key="o_risk")
            ovs_total    = ovs_price + ovs_ship + ovs_tariff + ovs_quality + ovs_inv + ovs_risk

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Domestic TCO",  f"${dom_total:.2f}")
        col2.metric("Offshore TCO",  f"${ovs_total:.2f}")
        savings = abs(dom_total - ovs_total)
        better  = "Domestic" if dom_total < ovs_total else "Offshore"
        col3.metric(f"{better} Saves", f"${savings:.2f}/unit",
                    delta=f"{savings/max(dom_total,ovs_total)*100:.1f}% savings")

        if dom_total < ovs_total:
            st.success(f"📍 **Recommendation: Domestic Supplier** — lower TCO despite higher purchase price")
        else:
            st.info(f"📍 **Recommendation: Offshore Supplier** — lower TCO but verify risk tolerance")

        # Bar chart breakdown
        categories_tco = ["Purchase Price", "Shipping", "Tariff", "Quality", "Inventory", "Risk"]
        dom_vals = [dom_price, dom_ship, dom_tariff, dom_quality, dom_inv, dom_risk]
        ovs_vals = [ovs_price, ovs_ship, ovs_tariff, ovs_quality, ovs_inv, ovs_risk]
        fig = go.Figure(data=[
            go.Bar(name="Domestic", x=categories_tco, y=dom_vals, marker_color="#3498db"),
            go.Bar(name="Offshore", x=categories_tco, y=ovs_vals, marker_color="#e67e22"),
        ])
        fig.update_layout(barmode="group", title="TCO Component Breakdown",
                          yaxis_title="Cost per Unit ($)", template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Supply Chain Risk Assessment")
        st.write("Score each risk factor (1=Low, 5=High) to compare sourcing options.")

        risk_factors = ["Lead Time Variability", "Political Risk", "Currency Risk",
                        "Quality Risk", "Supplier Financial Stability", "Logistics Disruption"]
        risk_data = []
        for rf in risk_factors:
            cols = st.columns([3, 2, 2])
            with cols[0]: st.write(rf)
            with cols[1]: dom_r = st.slider(f"Dom {rf[:8]}", 1, 5, 2, key=f"dr_{rf[:6]}")
            with cols[2]: ovs_r = st.slider(f"Off {rf[:8]}", 1, 5, 4, key=f"or_{rf[:6]}")
            risk_data.append({"Factor": rf, "Domestic": dom_r, "Offshore": ovs_r})

        risk_df = pd.DataFrame(risk_data)
        dom_risk_total = risk_df["Domestic"].sum()
        ovs_risk_total = risk_df["Offshore"].sum()

        col1, col2 = st.columns(2)
        col1.metric("Domestic Risk Score", f"{dom_risk_total}/30")
        col2.metric("Offshore Risk Score", f"{ovs_risk_total}/30",
                    delta=f"{ovs_risk_total - dom_risk_total:+d} vs Domestic")

        if ovs_risk_total > dom_risk_total + 5:
            st.warning("⚠️ Offshore has significantly higher risk. Factor in mitigation costs "
                       "(safety stock, dual sourcing, expediting).")


# ============================================================
# MODULE 25: FORECASTING (Chapter 18) - ENHANCED V5.0
# ============================================================
def module_forecast():
    display_header("📈", "Chapter 18", "Enhanced Forecasting Methods",
                   "WMA, Holt's Trend, Seasonal Index & Tracking Signal")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Weighted MA", "📈 Holt's Method",
                                       "🌊 Seasonal Index", "📡 Tracking Signal"])

    with tab1:
        st.markdown("### Weighted Moving Average (WMA)")
        display_formula_card("WMA",
                             r"F_t = \frac{\sum_{i=1}^{n} w_i \cdot A_{t-i}}{\sum_{i=1}^{n} w_i}")
        st.write("More recent periods receive higher weights — captures trends better than simple MA.")

        col1, col2 = st.columns(2)
        with col1:
            n_periods = st.selectbox("Periods (n)", [3, 4, 5], index=0)
            st.markdown("#### Weights & Actuals")
            weights  = []
            actuals  = []
            defaults_w = [0.5, 0.3, 0.2, 0.1, 0.05]
            defaults_d = [120, 110, 130, 115, 105]
            for i in range(n_periods):
                c = st.columns(2)
                with c[0]: w = st.number_input(f"w(t-{i+1})", value=defaults_w[i], format="%.2f", key=f"wma_w_{i}")
                with c[1]: d = st.number_input(f"A(t-{i+1})", value=defaults_d[i], key=f"wma_d_{i}")
                weights.append(w); actuals.append(d)

        with col2:
            total_w = sum(weights)
            wma     = sum(w*a for w, a in zip(weights, actuals)) / total_w if total_w > 0 else 0
            st.metric("WMA Forecast", f"{wma:.1f}")
            st.metric("Total Weight",  f"{total_w:.2f}")
            if abs(total_w - 1.0) > 0.01:
                st.warning(f"Weights sum to {total_w:.2f} — normalizing automatically")
            st.latex(rf"F_t = \frac{{{'+'.join([f'{w}×{a}' for w,a in zip(weights,actuals)])}}}{{{total_w:.2f}}} = {wma:.1f}")

            # Compare SMA vs WMA
            sma = sum(actuals) / n_periods
            st.metric("Simple MA (for comparison)", f"{sma:.1f}",
                      delta=f"WMA diff: {wma-sma:+.1f}")

    with tab2:
        st.markdown("### Holt's Trend-Adjusted Exponential Smoothing")

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Level", r"L_t = \alpha A_t + (1-\alpha)(L_{t-1}+T_{t-1})")
            display_formula_card("Trend", r"T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}")
            display_formula_card("Forecast m periods ahead", r"F_{t+m} = L_t + m \cdot T_t")

        with col2:
            alpha = st.slider("Alpha α (Level)", 0.05, 0.95, 0.30, 0.05)
            beta  = st.slider("Beta β (Trend)",  0.05, 0.95, 0.20, 0.05)
            L0 = st.number_input("Initial Level L₀", value=100.0)
            T0 = st.number_input("Initial Trend T₀", value=10.0)

        st.markdown("#### Multi-Period Simulation")
        n_sim = st.selectbox("Simulation Periods", [5, 8, 10], index=1)
        default_acts = [112, 125, 118, 140, 135, 150, 148, 165, 158, 172]
        acts_input = []
        sim_cols = st.columns(min(n_sim, 5))
        for i in range(n_sim):
            with sim_cols[i % 5]:
                a = st.number_input(f"A{i+1}", value=default_acts[i] if i<len(default_acts) else 150,
                                    key=f"holt_a_{i}")
                acts_input.append(a)

        # Run Holt's
        L, T = L0, T0
        rows = []
        for i, at in enumerate(acts_input):
            L_new = alpha * at + (1 - alpha) * (L + T)
            T_new = beta * (L_new - L) + (1 - beta) * T
            F_next = L_new + T_new
            rows.append({"Period": i+1, "Actual": at, "Level (L)": round(L_new,2),
                         "Trend (T)": round(T_new,2), "Forecast t+1": round(F_next,2),
                         "Error": round(at - (L + T), 2)})
            L, T = L_new, T_new

        df_holt = pd.DataFrame(rows)
        st.dataframe(df_holt, use_container_width=True)

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_holt["Period"], y=df_holt["Actual"],
                                 mode="lines+markers", name="Actual", line=dict(color="#2c3e50")))
        fig.add_trace(go.Scatter(x=df_holt["Period"], y=df_holt["Forecast t+1"],
                                 mode="lines+markers", name="Holt's Forecast",
                                 line=dict(color="#e74c3c", dash="dash")))
        fig.update_layout(title=f"Holt's Method (α={alpha}, β={beta})",
                          xaxis_title="Period", yaxis_title="Value",
                          template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast future periods
        m = st.number_input("Forecast m periods ahead", value=3, min_value=1, max_value=12)
        for period in range(1, m+1):
            st.write(f"**F(t+{period})** = {L:.2f} + {period} × {T:.2f} = **{L + period*T:.1f}**")

    with tab3:
        st.markdown("### Seasonal Index Calculator")
        display_formula_card("Seasonal Index",
                             r"SI_i = \frac{\text{Avg demand in season } i}{\text{Overall average demand}}")
        st.write("Seasonal index > 1 = above average season; < 1 = below average season.")

        n_seasons = st.selectbox("Periods per Year", [4, 12], format_func=lambda x: "Quarterly (4)" if x==4 else "Monthly (12)")
        n_years   = st.selectbox("Years of Data", [2, 3, 4], index=1)

        labels = [f"Q{i+1}" if n_seasons==4 else f"M{i+1:02d}" for i in range(n_seasons)]
        defaults_s = [80, 100, 120, 100,  90, 110, 130, 115,  85, 105, 125, 110]

        season_data = []
        for s in range(n_seasons):
            year_vals = []
            year_cols = st.columns(n_years + 1)
            with year_cols[0]: st.write(f"**{labels[s]}**")
            for y in range(n_years):
                with year_cols[y+1]:
                    val = st.number_input(f"Y{y+1}", value=defaults_s[s]+y*5 if s<len(defaults_s) else 100,
                                         key=f"si_{s}_{y}", label_visibility="visible")
                    year_vals.append(val)
            avg = sum(year_vals) / n_years
            season_data.append({"Season": labels[s], **{f"Y{y+1}":year_vals[y] for y in range(n_years)},
                                 "Avg": round(avg,1)})

        df_si = pd.DataFrame(season_data)
        overall_avg = df_si["Avg"].mean()
        df_si["SI"] = (df_si["Avg"] / overall_avg).round(3)
        df_si["Interpretation"] = df_si["SI"].apply(
            lambda s: "↑ Peak Season" if s > 1.1 else ("↓ Trough" if s < 0.9 else "≈ Average"))
        st.dataframe(df_si, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Overall Average", f"{overall_avg:.1f}")
        col2.metric("Seasonal Range", f"{df_si['SI'].min():.3f} – {df_si['SI'].max():.3f}")

        # SI bar chart
        fig = go.Figure(go.Bar(x=df_si["Season"], y=df_si["SI"],
                               marker_color=["#e74c3c" if s>1 else "#3498db" for s in df_si["SI"]]))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline SI=1.0")
        fig.update_layout(title="Seasonal Indices", yaxis_title="Seasonal Index",
                          template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Deseasonalized forecast
        st.markdown("#### Apply Seasonal Index to Forecast")
        base_fcst = st.number_input("Base Forecast (deseasonalized)", value=float(round(overall_avg)))
        deseason_df = df_si[["Season","SI"]].copy()
        deseason_df["Adjusted Forecast"] = (deseason_df["SI"] * base_fcst).round(1)
        st.dataframe(deseason_df, use_container_width=True)

    with tab4:
        st.markdown("### Tracking Signal Monitor")
        display_formula_card("Tracking Signal",
                             r"TS = \frac{RSFE}{MAD} = \frac{\sum(A_t - F_t)}{MAD}")
        st.write("|TS| > 4 → forecast bias; revisit the model.")

        st.markdown("#### Enter Actuals & Forecasts")
        n_tr = st.selectbox("Number of Periods", [6, 8, 10], index=1)
        default_act = [100, 110, 105, 115, 120, 125, 130, 128, 135, 140]
        default_fct = [95,  105, 108, 110, 115, 118, 122, 126, 130, 135]

        tr_cols = st.columns(2)
        act_vals, fct_vals = [], []
        with tr_cols[0]:
            st.write("**Actuals**")
            for i in range(n_tr):
                a = st.number_input(f"A{i+1}", value=default_act[i] if i<len(default_act) else 120,
                                    key=f"tr_a_{i}")
                act_vals.append(a)
        with tr_cols[1]:
            st.write("**Forecasts**")
            for i in range(n_tr):
                f = st.number_input(f"F{i+1}", value=default_fct[i] if i<len(default_fct) else 115,
                                    key=f"tr_f_{i}")
                fct_vals.append(f)

        errors   = [a - f for a, f in zip(act_vals, fct_vals)]
        abs_errs = [abs(e) for e in errors]
        rsfe_cum = [sum(errors[:i+1]) for i in range(n_tr)]
        mad_cum  = [sum(abs_errs[:i+1])/(i+1) for i in range(n_tr)]
        ts_vals  = [r/m if m > 0 else 0 for r, m in zip(rsfe_cum, mad_cum)]

        df_ts = pd.DataFrame({
            "Period": list(range(1, n_tr+1)),
            "Actual": act_vals, "Forecast": fct_vals,
            "Error": errors, "|Error|": abs_errs,
            "RSFE": rsfe_cum,
            "MAD":  [round(m,2) for m in mad_cum],
            "TS":   [round(ts,2) for ts in ts_vals]
        })
        st.dataframe(df_ts, use_container_width=True)

        final_mad  = mad_cum[-1]
        final_rsfe = rsfe_cum[-1]
        final_ts   = ts_vals[-1]
        final_mape = sum(abs(e)/a*100 for e,a in zip(errors,act_vals) if a!=0)/n_tr

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAD",             f"{final_mad:.2f}")
        col2.metric("RSFE",            f"{final_rsfe:.0f}")
        col3.metric("Tracking Signal", f"{final_ts:.2f}")
        col4.metric("MAPE",            f"{final_mape:.1f}%")

        # TS chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, n_tr+1)), y=ts_vals,
                                 mode="lines+markers", name="Tracking Signal",
                                 line=dict(color="#3498db", width=2)))
        fig.add_hline(y=4,  line_dash="dash", line_color="red",   annotation_text="+4 (Upper)")
        fig.add_hline(y=-4, line_dash="dash", line_color="red",   annotation_text="-4 (Lower)")
        fig.add_hline(y=0,  line_dash="dot",  line_color="green", annotation_text="0 (Unbiased)")
        fig.update_layout(title="Tracking Signal Over Time", xaxis_title="Period",
                          yaxis_title="Tracking Signal", template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

        if abs(final_ts) > 4:
            st.error(f"⚠️ |TS| = {abs(final_ts):.2f} > 4 — **Forecast bias detected!** "
                     "Forecast is systematically " + ("over-estimating" if final_ts < 0 else "under-estimating"))
        else:
            st.success(f"✅ |TS| = {abs(final_ts):.2f} ≤ 4 — Forecast is performing within control limits")

# ============================================================
# MODULE 26: REGRESSION (Chapter 18) - ENHANCED V5.0
# ============================================================
def module_regression():
    display_header("📈", "Chapter 18", "Linear Regression Trend Line", "Least squares forecasting and fit quality")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Calculator", "🎓 Practice"])

    with tab1:
        st.markdown("### Linear Regression Theory")
        st.write(
            "**Linear regression** fits a straight line that minimizes the sum of squared errors. "
            "It is widely used for trend projection and causal forecasting." 
        )

        display_citation(
            "The least squares principle provides a way of choosing the coefficients effectively by minimising the sum of the squared errors.",
            "Forecasting: Principles and Practice (Least squares estimation)"
        )

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Regression Line", r"\hat{Y} = a + bx")
            display_formula_card("Slope (b)", r"b = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}")
        with col2:
            display_formula_card("Intercept (a)", r"a = \bar{y} - b\bar{x}")
            display_formula_card("Coefficient of Determination", r"R^2 = 1 - \frac{SSE}{SST}")

        display_key_insight(
            "Why least squares works",
            "The fitted line minimizes squared vertical deviations, so large errors count more than small ones. "
            "That makes the method especially useful when you want a single line to summarize a trend."
        )

        display_textbook_content(
            "Interpreting the regression line",
            """The slope shows how much the forecast changes when x increases by 1 unit.
            The intercept is the estimated value when x = 0, though it may not always be meaningful in context.
            R-squared indicates how much of the variation in Y is explained by the line."""
        )

        st.markdown("### Model Fit Summary")
        fit_df = pd.DataFrame({
            "Metric": ["Slope (b)", "Intercept (a)", "R-squared", "Use case"],
            "Meaning": [
                "Change in Y for each 1-unit increase in X",
                "Estimated Y when X = 0",
                "Proportion of variance explained by the model",
                "Forecasting trend or relationship strength"
            ]
        })
        st.dataframe(fit_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Regression Calculator")
        st.write("Enter 12 periods of data, then estimate a future period.")

        x_vals = list(range(1, 13))
        y_vals = []

        cols = st.columns(6)
        for i in range(12):
            with cols[i % 6]:
                y = st.number_input(
                    f"Period {i+1}",
                    value=float(100 + i * 5),
                    key=f"reg_y_{i}"
                )
                y_vals.append(y)

        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x ** 2 for x in x_vals)

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-9:
            st.error("Cannot compute regression because the denominator is zero.")
            return

        b = (n * sum_xy - sum_x * sum_y) / denom
        a = (sum_y - b * sum_x) / n

        y_hat = [a + b * x for x in x_vals]
        y_bar = sum_y / n
        sst = sum((y - y_bar) ** 2 for y in y_vals)
        sse = sum((y - yh) ** 2 for y, yh in zip(y_vals, y_hat))
        r2 = 1 - (sse / sst) if sst > 0 else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            display_metric_card(f"Y = {a:.2f} + {b:.2f}x", "Regression Equation", "highlight")
        with c2:
            display_metric_card(f"{r2:.3f}", "R-squared", "success" if r2 >= 0.7 else "normal")
        with c3:
            display_metric_card(f"{sse:.2f}", "SSE", "normal")

        forecast_x = st.number_input("Forecast for Period", value=13, min_value=1)
        forecast_y = a + b * forecast_x
        st.metric(f"Forecast for Period {forecast_x}", f"{forecast_y:.1f}")

        st.markdown("### Regression Plot")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers", name="Actual",
            marker=dict(size=10, color="#6366f1")
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_hat, mode="lines", name="Fitted Line",
            line=dict(color="#10b981", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[forecast_x], y=[forecast_y], mode="markers+text",
            name="Forecast", marker=dict(size=12, color="#ef4444"),
            text=[f"{forecast_y:.1f}"], textposition="top center"
        ))
        fig.update_layout(**get_plotly_layout("Regression Trend Line", height=420))
        fig.update_layout(xaxis_title="Period", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Trend Line Basics (Easy)"):
            display_practice_problem(
                1, "Easy",
                "What does the slope mean in a linear regression trend line?"
            )
            if st.button("Show Solution", key="reg_p1"):
                display_solution(
                    "The slope tells you how much Y changes for each 1-unit increase in X. "
                    "A positive slope means Y rises as X rises; a negative slope means Y falls."
                )

        with st.expander("🟡 Problem 2: Interpretation (Medium)"):
            display_practice_problem(
                2, "Medium",
                "A regression line is Y = 50 + 4x. What is the forecast for x = 10?"
            )
            user_ans = st.number_input("Your Answer:", key="reg_p2")
            if st.button("Check Answer", key="reg_p2_btn"):
                correct = 50 + 4 * 10
                if check_answer(user_ans, correct):
                    st.success(f"✅ Correct! Forecast = {correct}")
                else:
                    display_solution(f"Y = 50 + 4(10) = <strong>{correct}</strong>")

        with st.expander("🔴 Problem 3: Regression Calculation (Hard)"):
            display_practice_problem(
                3, "Hard",
                "Given x = 1, 2, 3 and y = 2, 4, 5, calculate the regression line."
            )
            if st.button("Show Solution", key="reg_p3"):
                x = np.array([1, 2, 3], dtype=float)
                y = np.array([2, 4, 5], dtype=float)
                b = np.cov(x, y, bias=True)[0, 1] / np.var(x)
                a = y.mean() - b * x.mean()
                display_solution(
                    f"Using least squares: ȳ = {y.mean():.2f}, x̄ = {x.mean():.2f}, "
                    f"b = {b:.2f}, a = {a:.2f}.<br>"
                    f"Regression line: <strong>Y = {a:.2f} + {b:.2f}x</strong>"
                )


# ============================================================
# MODULE 27: AGGREGATE PLANNING (Chapter 19) - ENHANCED V5.0
# ============================================================
def module_aggregate():
    display_header("📋", "Chapter 19", "Aggregate Planning & S&OP", "Balancing supply and demand over time")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator", "🎓 Practice"])

    with tab1:
        st.markdown("### Sales & Operations Planning")
        st.write(
            "Sales and operations planning helps firms keep demand and supply in balance. "
            "Aggregate planning translates that goal into a medium-range production plan."
        )

        display_citation(
            "Sales and operations planning was coined by companies to refer to the process that helps firms keep demand and supply in balance. In operations management, this process traditionally was called aggregate planning.",
            "Jacobs & Chase (2024)"
        )

        st.markdown("#### Pure Planning Strategies")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🏃", "Chase Strategy", "Match output to demand. Low inventory, but workforce changes can be disruptive.")
        with col2:
            display_concept_card("📏", "Level Strategy", "Keep output and workforce stable. Inventory absorbs demand swings.")
        with col3:
            display_concept_card("🔀", "Mixed Strategy", "Blend tactics such as overtime, subcontracting, and inventory.")
        
        display_key_insight(
            "Choosing a strategy",
            "The best plan depends on the cost structure, workforce flexibility, and how volatile demand is."
        )

    with tab2:
        st.markdown("### Aggregate Planning Calculator")

        st.markdown("#### Quarterly Demand Forecast")
        cols = st.columns(4)
        demands = []
        for i in range(4):
            with cols[i]:
                d = st.number_input(f"Q{i+1} Demand", value=1200 + i * 200, key=f"ag_d_{i}")
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

        chase_cost = total_demand * reg_cost
        workers_needed = [d / units_per_worker for d in demands]
        hf_changes = sum(abs(workers_needed[i] - workers_needed[i-1]) for i in range(1, 4))
        chase_hf_cost = hf_changes * hire_fire
        chase_total = chase_cost + chase_hf_cost

        level_prod = avg_demand
        level_cost = total_demand * reg_cost

        inventory = []
        inv = 0
        for d in demands:
            inv = inv + level_prod - d
            inventory.append(max(0, inv))

        level_hold_cost = sum(inventory) * hold_cost
        level_total = level_cost + level_hold_cost

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🏃 Chase Strategy")
            display_metric_card(f"${chase_total:,.0f}", "Total Cost", "normal")
        with c2:
            st.markdown("#### 📏 Level Strategy")
            display_metric_card(f"${level_total:,.0f}", "Total Cost", "normal")

        st.markdown("### Demand and Inventory")
        plan_df = pd.DataFrame({
            "Quarter": ["Q1", "Q2", "Q3", "Q4"],
            "Demand": demands,
            "Level Production": [level_prod] * 4,
            "Inventory End": inventory
        })
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Q1", "Q2", "Q3", "Q4"], y=demands, name="Demand"))
        fig.add_trace(go.Bar(x=["Q1", "Q2", "Q3", "Q4"], y=[level_prod] * 4, name="Level Production"))
        fig.update_layout(**get_plotly_layout("Aggregate Planning Demand vs Production", height=400))
        fig.update_layout(barmode="group", xaxis_title="Quarter", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)

        if chase_total < level_total:
            st.success(f"📊 **Recommendation:** Chase Strategy saves ${level_total - chase_total:,.0f}")
        else:
            st.success(f"📊 **Recommendation:** Level Strategy saves ${chase_total - level_total:,.0f}")

    with tab3:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 Problem 1: Strategy Identification (Easy)"):
            display_practice_problem(
                1, "Easy",
                "Which aggregate planning strategy keeps workforce and production constant?"
            )
            if st.button("Show Solution", key="ag_p1"):
                display_solution("The **level strategy** keeps workforce and production constant while inventory absorbs demand fluctuations.")

        with st.expander("🟡 Problem 2: Planning Choice (Medium)"):
            display_practice_problem(
                2, "Medium",
                "Why might a company prefer a chase strategy?"
            )
            if st.button("Show Solution", key="ag_p2"):
                display_solution(
                    "A chase strategy is attractive when inventory is expensive, demand is volatile, "
                    "and the workforce can be adjusted easily."
                )

        with st.expander("🔴 Problem 3: Cost Comparison (Hard)"):
            display_practice_problem(
                3, "Hard",
                "A firm has stable demand and high holding costs. Which strategy is likely best and why?"
            )
            if st.button("Show Solution", key="ag_p3"):
                display_solution(
                    "The level strategy is often best when demand is stable and holding costs are high, "
                    "because it avoids frequent workforce changes while keeping inventory predictable."
                )


# ============================================================
# MODULE 28: EOQ (Chapter 20) - ENHANCED V5.0
# ============================================================
def module_eoq():
    display_header("📦", "Chapter 20", "Economic Order Quantity (EOQ)", "Minimizing total inventory costs")

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Theory", "🔬 Simulator", "⚙️ EPQ", "🎓 Practice"])

    with tab1:
        st.markdown("### Economic Order Quantity Model")

        display_citation(
            "The EOQ model attempts to find the order quantity that minimizes total annual holding and ordering costs. At the optimal order quantity Q*, the annual ordering cost exactly equals the annual holding cost.",
            "Jacobs & Chase (2024, p. 598)"
        )

        display_formula_card("Total Annual Cost", r"TC = \frac{D}{Q}S + \frac{Q}{2}H")
        display_formula_card("Optimal Order Quantity", r"Q^* = \sqrt{\frac{2DS}{H}}")
        display_formula_card("Minimum Total Cost", r"TC_{\min} = \sqrt{2DSH}")

        display_key_insight(
            "Robustness of EOQ",
            "EOQ is robust around the optimum. Small changes in Q usually do not change total cost very much."
        )

        st.markdown("### EOQ Logic")
        eoq_df = pd.DataFrame({
            "Component": ["Ordering Cost", "Holding Cost", "At Q*"],
            "Behavior": [
                "Falls as Q increases",
                "Rises as Q increases",
                "Ordering cost = Holding cost"
            ]
        })
        st.dataframe(eoq_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### EOQ Calculator")

        col1, col2 = st.columns(2)

        with col1:
            D = st.slider("Annual Demand (D)", 100, 10000, 1000, 100)
            S = st.slider("Order Cost (S) $", 5, 500, 50, 5)
            H = st.slider("Holding Cost (H) $/unit/yr", 1, 50, 2)

        with col2:
            Q_star = math.sqrt(2 * D * S / H)
            annual_ordering = (D / Q_star) * S
            annual_holding = (Q_star / 2) * H
            TC = annual_ordering + annual_holding
            orders_per_year = D / Q_star

            st.metric("Optimal Q*", f"{Q_star:.0f} units")
            st.metric("Total Cost", f"${TC:.0f}")
            st.metric("Orders/Year", f"{orders_per_year:.1f}")

            st.latex(rf"Q^* = \sqrt{{\frac{{2 \times {D} \times {S}}}{{{H}}}}} = {Q_star:.0f}")
            st.latex(rf"Annual ordering cost = {annual_ordering:.0f}")
            st.latex(rf"Annual holding cost = {annual_holding:.0f}")

        st.markdown("### Cost Curve")
        q_vals = np.linspace(max(1, Q_star * 0.25), Q_star * 2.5, 80)
        tc_vals = [(D / q) * S + (q / 2) * H for q in q_vals]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q_vals, y=tc_vals, mode="lines", name="Total Cost"))
        fig.add_vline(x=Q_star, line_dash="dash", line_color="#ef4444")
        fig.update_layout(**get_plotly_layout("EOQ Total Cost Curve", height=400))
        fig.update_layout(xaxis_title="Order Quantity (Q)", yaxis_title="Annual Cost ($)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Production Order Quantity (EPQ)")
        display_formula_card("Optimal Production Lot Size", r"Q_p^* = \sqrt{\frac{2DS}{H(1-d/p)}}")

        col1, col2 = st.columns(2)

        with col1:
            p = st.number_input("Production Rate (p)", value=1000)
            d = st.number_input("Demand Rate (d)", value=200)

        with col2:
            if p > d:
                Q_epq = math.sqrt(2 * D * S / (H * (1 - d / p)))
                st.metric("Optimal Production Lot", f"{Q_epq:.0f}")
                display_key_insight(
                    "EPQ condition",
                    "EPQ requires production rate to exceed demand rate. If not, inventory cannot build properly."
                )
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
                    display_solution(
                        f"Q* = √(2DS / H)<br>"
                        f"Q* = √(2 × 5000 × 100 / 4)<br>"
                        f"Q* = <strong>{correct:.0f} units</strong>"
                    )

        with st.expander("🟡 Problem 2: Ordering vs Holding Cost (Medium)"):
            display_practice_problem(
                2, "Medium",
                "At the EOQ, what is the relationship between annual ordering cost and annual holding cost?"
            )
            if st.button("Show Solution", key="eoq_p2"):
                display_solution(
                    "At the EOQ, annual ordering cost equals annual holding cost. This equality identifies the minimum point of total annual inventory cost."
                )

        with st.expander("🔴 Problem 3: EPQ Condition (Hard)"):
            display_practice_problem(
                3, "Hard",
                "Why must the production rate exceed the demand rate in the EPQ model?"
            )
            if st.button("Show Solution", key="eoq_p3"):
                display_solution(
                    "If production rate does not exceed demand rate, inventory cannot accumulate during production. "
                    "EPQ requires p > d so inventory can be built up in cycles."
                )

# ============================================================
# MODULE 29: SAFETY STOCK (Chapter 20) - ENHANCED V5.0
# ============================================================
def module_safetystock():
    display_header("🛡️", "Chapter 20", "Safety Stock & Reorder Point",
                   "Protecting against demand and lead time variability")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Calculator", "📊 Sensitivity Analysis"])

    with tab1:
        st.markdown("### Safety Stock Theory")

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Safety Stock (Demand Variability)",
                                 r"SS = z \cdot \sigma_d \cdot \sqrt{LT}")
            display_formula_card("Safety Stock (Combined Variability)",
                                 r"SS = z\sqrt{LT \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_{LT}^2}")
        with col2:
            display_formula_card("Reorder Point",
                                 r"ROP = \bar{d} \times LT + SS")
            display_formula_card("Average Inventory",
                                 r"\bar{I} = \frac{Q}{2} + SS")

        st.markdown("#### Common Z-Values by Service Level")
        df_z = pd.DataFrame({
            "Service Level": ["90%", "95%", "97.5%", "99%", "99.9%"],
            "z-value": [1.28, 1.65, 1.96, 2.33, 3.09],
            "Relative SS (vs 95%)": ["–22%", "Base", "+19%", "+41%", "+87%"]
        })
        st.dataframe(df_z, use_container_width=True)

        display_key_insight(
            "Service Level Trade-off",
            "Increasing service level from 95% to 99% requires ~41% more safety stock "
            "(z increases from 1.65 to 2.33). The marginal cost of higher service grows exponentially."
        )

        st.markdown("#### When to Use Combined Formula")
        col1, col2 = st.columns(2)
        with col1:
            display_concept_card("📦", "Simple Formula",
                                 "Use when only demand varies. Lead time is fixed and deterministic.")
        with col2:
            display_concept_card("🔀", "Combined Formula",
                                 "Use when both demand AND lead time vary. Captures full supply-demand uncertainty.")

        display_citation(
            "Safety stock is inventory carried to protect against fluctuations in demand or supply. "
            "The statistical approach sets safety stock based on a desired service level and the "
            "standard deviation of demand during lead time.",
            "Jacobs & Chase (2024, Ch. 20)"
        )

    with tab2:
        st.markdown("### Safety Stock Calculator")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Demand & Lead Time Parameters")
            d_avg    = st.slider("Average Daily Demand (d̄)", 10, 500, 100, step=5)
            sigma_d  = st.slider("Demand Std Dev (σ_d)", 1, 100, 20)
            lt       = st.slider("Lead Time — days (LT)", 1, 30, 7)
            sigma_lt = st.slider("Lead Time Std Dev (σ_LT)", 0, 10, 0,
                                 help="Set > 0 to activate combined formula")

            service_level = st.selectbox("Service Level",
                                         ["90% (z=1.28)", "95% (z=1.65)",
                                          "97.5% (z=1.96)", "99% (z=2.33)", "99.9% (z=3.09)"])
            z_map = {"90% (z=1.28)": 1.28, "95% (z=1.65)": 1.65,
                     "97.5% (z=1.96)": 1.96, "99% (z=2.33)": 2.33, "99.9% (z=3.09)": 3.09}
            z = z_map[service_level]

            order_qty = st.number_input("Order Quantity Q (for avg inventory)", value=200, min_value=1)

        with col2:
            st.markdown("#### Results")
            if sigma_lt > 0:
                ss = z * math.sqrt(lt * sigma_d**2 + d_avg**2 * sigma_lt**2)
                formula_label = "Combined Formula"
                latex_str = (rf"SS = {z}\sqrt{{{lt}\cdot{sigma_d}^2 + "
                             rf"{d_avg}^2\cdot{sigma_lt}^2}} = {ss:.0f}")
            else:
                ss = z * sigma_d * math.sqrt(lt)
                formula_label = "Simple Formula"
                latex_str = rf"SS = {z} \times {sigma_d} \times \sqrt{{{lt}}} = {ss:.0f}"

            rop       = d_avg * lt + ss
            avg_inv   = order_qty / 2 + ss
            cycle_inv = d_avg * lt

            st.metric("Safety Stock", f"{ss:.0f} units", help=formula_label)
            st.metric("Reorder Point (ROP)", f"{rop:.0f} units")
            st.metric("Avg Inventory", f"{avg_inv:.0f} units")
            st.metric("Cycle Stock at ROP", f"{cycle_inv:.0f} units")

            st.latex(latex_str)
            st.latex(rf"ROP = {d_avg} \times {lt} + {ss:.0f} = {rop:.0f}")

            st.info(f"📌 Using **{formula_label}** — "
                    + ("lead time variability included." if sigma_lt > 0
                       else "set σ_LT > 0 for combined formula."))

    with tab3:
        st.markdown("### Safety Stock Sensitivity Analysis")
        st.write("How does safety stock change as service level and demand variability change?")

        levels   = [1.28, 1.65, 1.96, 2.33, 3.09]
        labels   = ["90%", "95%", "97.5%", "99%", "99.9%"]
        sigma_range = [10, 15, 20, 25, 30]

        fig = go.Figure()
        for sig in sigma_range:
            ss_vals = [z_val * sig * math.sqrt(lt) for z_val in levels]
            fig.add_trace(go.Scatter(
                x=labels, y=ss_vals,
                mode="lines+markers",
                name=f"σ_d = {sig}"
            ))

        fig.update_layout(
            title=f"Safety Stock vs Service Level (LT = {lt} days)",
            xaxis_title="Service Level",
            yaxis_title="Safety Stock (units)",
            legend_title="Demand Std Dev",
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Z-Score Impact on Safety Stock")
        base_ss = 1.65 * sigma_d * math.sqrt(lt)
        comparison = pd.DataFrame({
            "Service Level": labels,
            "z-value": levels,
            "Safety Stock": [round(z_v * sigma_d * math.sqrt(lt)) for z_v in levels],
            "% vs 95% Baseline": [f"{((z_v/1.65)-1)*100:+.1f}%" for z_v in levels]
        })
        st.dataframe(comparison, use_container_width=True)


# ============================================================
# MODULE 30: NEWSVENDOR (Chapter 20) - ENHANCED V5.0
# ============================================================
def module_newsvendor():
    display_header("📰", "Chapter 20", "Newsvendor Model",
                   "Single-period inventory optimization under uncertainty")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Calculator", "📊 Profit Curve"])

    with tab1:
        st.markdown("### Newsvendor Model Theory")
        st.write(
            "The **Newsvendor Model** optimizes order quantity for perishable or seasonal items "
            "where demand is stochastic and there is only one ordering opportunity. "
            "It minimizes expected total cost by balancing under- and overstocking risks."
        )

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Cost of Understocking (C_u)",
                                 r"C_u = \text{Price} - \text{Cost}")
            display_formula_card("Critical Ratio (CR)",
                                 r"CR = \frac{C_u}{C_u + C_o}")
        with col2:
            display_formula_card("Cost of Overstocking (C_o)",
                                 r"C_o = \text{Cost} - \text{Salvage}")
            display_formula_card("Optimal Order Quantity",
                                 r"Q^* = \mu + z^* \cdot \sigma")

        display_key_insight(
            "Critical Ratio Intuition",
            "CR is the probability of selling one more unit. If CR = 0.75, order up to the "
            "75th percentile of demand — three-quarters of the time you'll sell that last unit."
        )

        st.markdown("#### Decision Rule")
        st.markdown("""
        1. Calculate **C_u** and **C_o**
        2. Compute **Critical Ratio** = C_u / (C_u + C_o)  
        3. Find **z*** = Φ⁻¹(CR)  
        4. Compute **Q*** = μ + z* · σ
        """)

        display_citation(
            "The single-period model (newsvendor) is applicable to fashion goods, seasonal items, "
            "perishable food, and event tickets. Expected profit is maximized at the critical ratio.",
            "Jacobs & Chase (2024, Ch. 20)"
        )

    with tab2:
        st.markdown("### Newsvendor Calculator")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Cost & Demand Inputs")
            price   = st.number_input("Selling Price ($)",    value=100, min_value=1)
            cost    = st.number_input("Unit Cost ($)",         value=70,  min_value=1)
            salvage = st.number_input("Salvage Value ($)",     value=20,  min_value=0)
            mu      = st.number_input("Mean Demand (μ)",       value=200, min_value=1)
            sigma   = st.number_input("Demand Std Dev (σ)",    value=40,  min_value=1)

        with col2:
            st.markdown("#### Results")
            Cu = price - cost
            Co = cost - salvage

            if Cu + Co <= 0:
                st.error("C_u + C_o must be > 0. Check price, cost, and salvage inputs.")
            else:
                critical_ratio = Cu / (Cu + Co)
                z        = normal_ppf(critical_ratio)
                Q_star   = mu + z * sigma

                # Expected profit approximation
                exp_sales      = mu - sigma * norm.pdf(z) + Q_star * (1 - critical_ratio)
                exp_leftover   = max(0, Q_star - mu)
                exp_profit     = price * min(Q_star, mu) - cost * Q_star + salvage * exp_leftover

                st.metric("Cost of Understocking (Cu)", f"${Cu}")
                st.metric("Cost of Overstocking (Co)",  f"${Co}")
                st.metric("Critical Ratio",             f"{critical_ratio:.4f}")
                st.metric("Z-Score (z*)",               f"{z:.3f}")
                st.metric("Optimal Q*",                 f"{Q_star:.0f} units")

                st.latex(rf"CR = \frac{{{Cu}}}{{{Cu}+{Co}}} = {critical_ratio:.4f}")
                st.latex(rf"Q^* = {mu} + {z:.3f} \times {sigma} = {Q_star:.0f}")

                if Q_star < mu - 2*sigma:
                    st.warning("⚠️ Q* is very low — high stockout risk.")
                elif Q_star > mu + 2*sigma:
                    st.warning("⚠️ Q* is very high — significant leftover risk.")
                else:
                    st.success("✅ Q* is within a reasonable demand range.")

    with tab3:
        st.markdown("### Expected Profit vs Order Quantity")
        st.write("Visualize how profit changes with order quantity around Q*.")

        price_p   = st.number_input("Price ($)",   value=100, key="nv_p")
        cost_p    = st.number_input("Cost ($)",    value=70,  key="nv_c")
        salvage_p = st.number_input("Salvage ($)", value=20,  key="nv_s")
        mu_p      = st.number_input("μ",           value=200, key="nv_mu")
        sigma_p   = st.number_input("σ",           value=40,  key="nv_sig")

        if price_p > cost_p >= salvage_p:
            Cu_p = price_p - cost_p
            Co_p = cost_p - salvage_p
            cr_p = Cu_p / (Cu_p + Co_p)
            z_p  = normal_ppf(cr_p)
            Qopt = mu_p + z_p * sigma_p

            q_range = range(max(1, int(mu_p - 3*sigma_p)), int(mu_p + 3*sigma_p) + 1, max(1, int(sigma_p//5)))
            profits = []
            for q in q_range:
                # E[profit] = price*E[min(D,Q)] - cost*Q + salvage*E[max(Q-D,0)]
                z_q  = (q - mu_p) / sigma_p
                exp_sold     = mu_p * normal_cdf(z_q) + q * (1 - normal_cdf(z_q)) - sigma_p * norm.pdf(z_q)
                exp_leftover = q - exp_sold
                ep = price_p * exp_sold - cost_p * q + salvage_p * exp_leftover
                profits.append(ep)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(q_range), y=profits,
                                     mode="lines", name="Expected Profit",
                                     line=dict(color="royalblue", width=2)))
            fig.add_vline(x=Qopt, line_dash="dash", line_color="red",
                          annotation_text=f"Q* = {Qopt:.0f}", annotation_position="top right")
            fig.update_layout(
                title="Expected Profit vs Order Quantity",
                xaxis_title="Order Quantity (Q)",
                yaxis_title="Expected Profit ($)",
                template="plotly_white", height=420
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Ensure Price > Cost ≥ Salvage to generate chart.")


# ============================================================
# MODULE 31: MRP (Chapter 21) - ENHANCED V5.0
# ============================================================
def module_mrp():
    display_header("🏭", "Chapter 21", "Material Requirements Planning",
                   "Scheduling dependent-demand inventory")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 MRP Matrix", "🌳 BOM Explorer"])

    with tab1:
        st.markdown("### MRP System Structure")

        display_citation(
            "The MRP system uses three sources of information: (1) Demand from the master schedule, "
            "(2) the bill-of-materials identifying what is needed to make each end item, and "
            "(3) the current inventory status managed by the system.",
            "Jacobs & Chase (2024, Ch. 21)"
        )

        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Net Requirements",
                                 r"\text{Net}_t = \max(0,\; \text{Gross}_t - OH_{t-1} - SR_t)")
        with col2:
            display_formula_card("Planned Order Release",
                                 r"\text{Release}_{t} = \text{Receipt}_{t + LT}")

        st.markdown("#### MRP Three-Input Architecture")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("📅", "Master Production Schedule",
                                 "What end items to produce and when — the demand driver.")
        with col2:
            display_concept_card("🌳", "Bill of Materials (BOM)",
                                 "Parent-child product structure with quantities per assembly.")
        with col3:
            display_concept_card("📦", "Inventory Records File",
                                 "On-hand balances, scheduled receipts, and lead times per item.")

        st.markdown("#### MRP Outputs")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("📋", "Planned Order Releases",
                                 "When to release purchase or production orders.")
        with col2:
            display_concept_card("🔄", "Order Rescheduling",
                                 "Expedite, de-expedite, or cancel existing orders.")
        with col3:
            display_concept_card("📊", "Performance Reports",
                                 "Inventory projections, exception messages.")

    with tab2:
        st.markdown("### MRP Record Calculator")

        col1, col2, col3 = st.columns(3)
        with col1:
            lead_time = st.number_input("Lead Time (weeks)", value=1, min_value=1, max_value=6)
            beg_inv   = st.number_input("Beginning On-Hand", value=20, min_value=0)
        with col2:
            lot_rule  = st.selectbox("Lot Sizing Rule",
                                     ["Lot-for-Lot (L4L)", "Fixed Order Qty (FOQ)", "POQ"])
            if lot_rule == "Fixed Order Qty (FOQ)":
                lot_size = st.number_input("Fixed Lot Size", value=50, min_value=1)
            elif lot_rule == "POQ":
                poq_p    = st.number_input("POQ Period (weeks)", value=2, min_value=1)
                lot_size = 0
            else:
                lot_size = 0
        with col3:
            ss_mrp = st.number_input("Safety Stock", value=0, min_value=0)
            n_weeks = st.selectbox("Horizon (weeks)", [6, 8, 10], index=0)

        st.markdown(f"#### Gross Requirements ({n_weeks} weeks)")
        default_gr = [50, 0, 100, 0, 150, 0, 80, 60, 120, 40]
        gross = []
        cols  = st.columns(n_weeks)
        for i in range(n_weeks):
            with cols[i]:
                g = st.number_input(f"Wk {i+1}",
                                    value=default_gr[i] if i < len(default_gr) else 0,
                                    key=f"mrp_g_{i}", min_value=0)
                gross.append(g)

        # ── MRP Calculation ──
        on_hand_proj   = []
        net_req        = []
        planned_rcpt   = []
        planned_rel    = [0] * n_weeks
        current_oh     = beg_inv

        for i in range(n_weeks):
            available = current_oh - gross[i]
            if available >= ss_mrp:
                net_req.append(0)
                receipt = 0
            else:
                needed = ss_mrp - available
                if lot_rule == "Lot-for-Lot (L4L)":
                    receipt = needed
                elif lot_rule == "Fixed Order Qty (FOQ)":
                    receipt = math.ceil(needed / lot_size) * lot_size
                else:  # POQ
                    total_needed = sum(gross[i:i+poq_p])
                    receipt = max(0, total_needed - current_oh + ss_mrp)
                net_req.append(needed)

            planned_rcpt.append(receipt)
            current_oh = current_oh - gross[i] + receipt
            on_hand_proj.append(max(0, current_oh))

            rel_week = i - lead_time
            if rel_week >= 0 and receipt > 0:
                planned_rel[rel_week] = receipt

        df_mrp = pd.DataFrame({
            "Week":             list(range(1, n_weeks + 1)),
            "Gross Req":        gross,
            "Projected OH":     on_hand_proj,
            "Net Req":          net_req,
            "Planned Receipt":  planned_rcpt,
            "Planned Release":  planned_rel
        })

        # Color-code rows where a release fires
        def highlight_release(row):
            return ["background-color: #fff3cd" if row["Planned Release"] > 0
                    else "" for _ in row]

        st.dataframe(df_mrp.style.apply(highlight_release, axis=1),
                     use_container_width=True)

        # Summary metrics
        total_orders  = sum(1 for r in planned_rcpt if r > 0)
        total_ordered = sum(planned_rcpt)
        st.columns(3)[0].metric("Total Orders Placed", total_orders)
        st.columns(3)[1].metric("Total Units Ordered", total_ordered)
        st.columns(3)[2].metric("Avg On-Hand",
                                f"{sum(on_hand_proj)/len(on_hand_proj):.1f}")

    with tab3:
        st.markdown("### Bill of Materials Explorer")

        st.markdown("#### Product Structure Tree")
        st.markdown("""
        ```
        (Level 0)  Product A  ×1
        ├── (Level 1)  Sub-B  ×2
        │   ├── (Level 2)  Part-D  ×4
        │   └── (Level 2)  Part-F  ×1
        └── (Level 1)  Sub-C  ×1
            ├── (Level 2)  Part-D  ×2   ← shared component (low-level code = 2)
            └── (Level 2)  Part-E  ×3
        ```
        """)

        st.info("💡 **Low-Level Coding:** Part-D appears at level 2 under both Sub-B and Sub-C. "
                "MRP processes it at its lowest level to avoid double-counting.")

        end_qty = st.number_input("End Items Required (Product A)", value=100, min_value=1)

        bom_data = {
            "Component":        ["Product A", "Sub-B",      "Sub-C",      "Part-D",           "Part-E",      "Part-F"],
            "Level":            [0,            1,             1,             2,                  2,              2],
            "Qty per Parent":   [1,            2,             1,             "4 (B) + 2 (C)",    3,              1],
            "Gross Requirement":[int(end_qty), int(end_qty*2), int(end_qty), int(end_qty*2*4 + end_qty*2),
                                 int(end_qty*3), int(end_qty*2)]
        }
        st.dataframe(pd.DataFrame(bom_data), use_container_width=True)

        display_key_insight(
            "Shared Components",
            f"Part-D total requirement = (2×4 + 1×2) × {end_qty} = "
            f"{(2*4+1*2)*int(end_qty)} units. Always consolidate shared "
            "components before running MRP."
        )


# ============================================================
# MODULE 32: MRP LOT SIZING (Chapter 21) - ENHANCED V5.0
# ============================================================
def module_mrp_lotsizing():
    display_header("📦", "Chapter 21", "MRP Lot Sizing Comparison",
                   "L4L, EOQ, and POQ techniques with interactive cost analysis")

    tab1, tab2 = st.tabs(["📚 Theory & Case Study", "🔬 Interactive Calculator"])

    with tab1:
        st.markdown("### Lot Sizing Methods")

        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("📦", "Lot-for-Lot (L4L)",
                                 "Order exactly what's needed each period. "
                                 "Minimizes holding cost; maximizes setup cost.")
        with col2:
            display_concept_card("⚖️", "EOQ",
                                 "Fixed quantity from EOQ formula. "
                                 "May create lumpy orders not aligned to needs.")
        with col3:
            display_concept_card("⏱️", "POQ",
                                 "Order for T periods at a time where "
                                 "T* = EOQ ÷ Average Demand.")

        display_formula_card("POQ Period",
                             r"T^* = \frac{EOQ}{\bar{d}} = \sqrt{\frac{2S}{H \bar{d}}}")

        st.markdown("### Textbook Case Study (Exhibit 21.16)")
        st.write("**Parameters:** Setup cost S = $47, Holding cost H = $2/unit/week")

        requirements = [105, 80, 130, 50, 0, 200, 125, 100]
        avg_d = sum(requirements) / len(requirements)
        eoq   = math.sqrt(2 * 47 * sum(requirements) / (2 * len(requirements)))
        poq_T = round(eoq / avg_d) if avg_d > 0 else 1

        df_req = pd.DataFrame({"Week": list(range(1, 9)), "Net Requirements": requirements})
        st.dataframe(df_req, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Weekly Demand", f"{avg_d:.1f}")
        col2.metric("EOQ (computed)", f"{eoq:.0f}")
        col3.metric("POQ Period T*", f"{poq_T} weeks")
        col4.metric("Total Demand", f"{sum(requirements)}")

        col1, col2, col3 = st.columns(3)
        col1.metric("L4L Total Cost", "$329", delta="7 setups, min holding")
        col2.metric("EOQ Total Cost", "$284", delta="Fewer setups, some holding")
        col3.metric("POQ Total Cost", "$261", delta="Best balance ✅", delta_color="off")

        st.success("**Result:** POQ is most cost-effective — it synchronizes order cycles with "
                   "demand periods, reducing both setup frequency and carrying costs.")

    with tab2:
        st.markdown("### Interactive Lot Sizing Calculator")

        col1, col2 = st.columns(2)
        with col1:
            S = st.number_input("Setup / Ordering Cost ($)", value=47, min_value=1)
            H = st.number_input("Holding Cost ($/unit/period)", value=2.0, min_value=0.01,
                                format="%.2f")
        with col2:
            n_p = st.selectbox("Number of Periods", [6, 8, 10], index=1)
            st.markdown(" ")

        st.markdown("#### Enter Net Requirements")
        default_r = [105, 80, 130, 50, 0, 200, 125, 100, 60, 90]
        reqs = []
        cols = st.columns(n_p)
        for i in range(n_p):
            with cols[i]:
                r = st.number_input(f"P{i+1}", value=default_r[i] if i < len(default_r) else 0,
                                    key=f"ls_{i}", min_value=0)
                reqs.append(r)

        avg_demand = sum(reqs) / n_p if n_p else 1
        eoq_calc   = math.sqrt(2 * S * sum(reqs) / H / n_p) if avg_demand > 0 else 50
        poq_calc   = max(1, round(eoq_calc / avg_demand)) if avg_demand > 0 else 1

        # ── L4L ──
        l4l_setups  = sum(1 for r in reqs if r > 0)
        l4l_holding = 0
        l4l_cost    = l4l_setups * S + l4l_holding

        # ── EOQ ──
        eoq_q       = round(eoq_calc)
        eoq_inv     = 0
        eoq_setups  = 0
        carry       = 0
        for r in reqs:
            if carry < r:
                eoq_setups += 1
                carry += eoq_q
            carry -= r
            eoq_inv += carry
        eoq_cost = eoq_setups * S + eoq_inv * H

        # ── POQ ──
        poq_inv    = 0
        poq_setups = 0
        i = 0
        while i < n_p:
            period_demand = sum(reqs[i:i+poq_calc])
            if period_demand > 0:
                poq_setups += 1
                leftover = period_demand
                for j in range(poq_calc):
                    if i + j < n_p:
                        leftover -= reqs[i + j]
                        poq_inv  += max(0, leftover)
            i += poq_calc
        poq_cost = poq_setups * S + poq_inv * H

        results_df = pd.DataFrame({
            "Method":        ["L4L", f"EOQ (Q={eoq_q})", f"POQ (T={poq_calc})"],
            "Setup Cost ($)": [l4l_setups*S, eoq_setups*S, poq_setups*S],
            "Holding Cost ($)":[l4l_holding, eoq_inv*H, poq_inv*H],
            "Total Cost ($)": [l4l_cost, eoq_cost, poq_cost]
        })
        st.dataframe(results_df, use_container_width=True)

        best = results_df.loc[results_df["Total Cost ($)"].idxmin(), "Method"]
        st.success(f"✅ **Best Method: {best}** with ${results_df['Total Cost ($)'].min():.0f} total cost.")

        fig = go.Figure(data=[
            go.Bar(name="Setup Cost",   x=results_df["Method"], y=results_df["Setup Cost ($)"],
                   marker_color="#4C72B0"),
            go.Bar(name="Holding Cost", x=results_df["Method"], y=results_df["Holding Cost ($)"],
                   marker_color="#DD8452"),
        ])
        fig.update_layout(barmode="stack", title="Cost Breakdown by Lot Sizing Method",
                          yaxis_title="Cost ($)", template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# MODULE 33: JOB SCHEDULING (Chapter 22) - ENHANCED V5.0
# ============================================================
def module_scheduling():
    display_header("📅", "Chapter 22", "Job Sequencing & Priority Rules",
                   "Determining job processing order for single-machine scheduling")

    tab1, tab2, tab3 = st.tabs(["📚 Theory", "🔬 Simulator + Gantt", "🎓 Practice"])

    with tab1:
        st.markdown("### Priority Sequencing Rules")

        display_citation(
            "Priority rules are simple heuristics used to select the order in which jobs will be "
            "processed. They are especially important in job-shop environments where different jobs "
            "compete for the same resources.",
            "Jacobs & Chase (2024, p. 672)"
        )

        rule_data = pd.DataFrame({
            "Rule": ["FCFS", "SPT", "EDD", "LPT", "CR", "SLACK"],
            "Name": ["First Come, First Served", "Shortest Processing Time",
                     "Earliest Due Date", "Longest Processing Time",
                     "Critical Ratio", "Least Slack"],
            "Optimizes": ["Fairness / WIP order", "Avg flow time ✅ (provably optimal)",
                          "Max tardiness", "Machine utilization",
                          "Dynamic priority (CR < 1 = late)", "Urgency buffer"],
            "Formula": ["Arrival order", "Min PT", "Min DD",
                        "Max PT", "(DD − Today) / PT", "DD − Today − PT"]
        })
        st.dataframe(rule_data, use_container_width=True)

        display_formula_card("Critical Ratio",
                             r"CR = \frac{D_j - t_{now}}{PT_j} \quad "
                             r"\begin{cases} CR < 1 & \text{behind schedule} \\ "
                             r"CR = 1 & \text{on schedule} \\ "
                             r"CR > 1 & \text{ahead of schedule} \end{cases}")

        display_key_insight(
            "SPT Optimality",
            "SPT is provably optimal for minimizing average flow time, average WIP, and "
            "average lateness on a single machine. However, it can starve long jobs."
        )

        st.markdown("#### Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            display_formula_card("Avg Flow Time",
                                 r"\bar{F} = \frac{\sum C_j}{n}")
            display_formula_card("Makespan",
                                 r"C_{max} = \sum PT_j \quad \text{(same for all rules)}")
        with col2:
            display_formula_card("Avg Tardiness",
                                 r"\bar{T} = \frac{\sum \max(0, C_j - D_j)}{n}")
            display_formula_card("Avg Lateness",
                                 r"\bar{L} = \frac{\sum (C_j - D_j)}{n}")

    with tab2:
        st.markdown("### Job Scheduling Simulator")

        col1, col2 = st.columns([1, 2])
        with col1:
            rule  = st.selectbox("Priority Rule", ["FCFS", "SPT", "EDD", "LPT", "CR", "SLACK"])
            today = st.number_input("Current Day (t_now)", value=0, min_value=0)
            n_jobs = st.selectbox("Number of Jobs", [3, 4, 5, 6], index=1)

        st.markdown("#### Job Data Entry")
        default_pt = [3, 1, 4, 2, 5, 2]
        default_dd = [5, 3, 9, 7, 11, 6]
        jobs = []
        job_cols = st.columns(n_jobs)
        for i in range(n_jobs):
            with job_cols[i]:
                name = chr(65 + i)
                st.markdown(f"**Job {name}**")
                pt = st.number_input(f"PT", value=default_pt[i], key=f"sch_pt_{i}", min_value=1)
                dd = st.number_input(f"DD", value=default_dd[i], key=f"sch_dd_{i}", min_value=1)
                jobs.append({"name": name, "pt": pt, "dd": dd,
                             "cr":    (dd - today) / pt if pt > 0 else float("inf"),
                             "slack": (dd - today - pt)})

        # Sort
        sort_keys = {"FCFS": None, "SPT": "pt", "EDD": "dd",
                     "LPT": "pt", "CR": "cr", "SLACK": "slack"}
        if rule == "FCFS":
            sorted_jobs = jobs[:]
        elif rule == "LPT":
            sorted_jobs = sorted(jobs, key=lambda x: x["pt"], reverse=True)
        else:
            sorted_jobs = sorted(jobs, key=lambda x: x[sort_keys[rule]])

        # Calculate metrics
        results     = []
        current_t   = today
        flow_times  = []
        tardiness   = []
        lateness    = []
        gantt_tasks = []

        for job in sorted_jobs:
            start    = current_t
            finish   = current_t + job["pt"]
            flow     = finish - today
            tardy    = max(0, finish - job["dd"])
            late     = finish - job["dd"]
            flow_times.append(flow)
            tardiness.append(tardy)
            lateness.append(late)
            results.append({
                "Job": job["name"], "Start": start, "PT": job["pt"],
                "Finish": finish, "Due Date": job["dd"],
                "Flow Time": flow, "Tardiness": tardy, "Lateness": late,
                "CR": round(job["cr"], 2), "Slack": job["slack"]
            })
            gantt_tasks.append({"Job": f"Job {job['name']}", "Start": start, "Finish": finish,
                                 "Due": job["dd"], "Late": tardy > 0})
            current_t = finish

        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Flow Time",    f"{sum(flow_times)/len(flow_times):.2f}")
        col2.metric("Avg Tardiness",    f"{sum(tardiness)/len(tardiness):.2f}")
        col3.metric("Avg Lateness",     f"{sum(lateness)/len(lateness):.2f}")
        col4.metric("# Tardy Jobs",     f"{sum(1 for t in tardiness if t > 0)}")

        # Gantt chart
        st.markdown("#### Gantt Chart")
        colors = ["#e74c3c" if t["Late"] else "#2ecc71" for t in gantt_tasks]
        fig = go.Figure()
        for idx, task in enumerate(gantt_tasks):
            fig.add_trace(go.Bar(
                x=[task["Finish"] - task["Start"]], base=[task["Start"]],
                y=[task["Job"]], orientation="h",
                marker_color=colors[idx],
                name=task["Job"],
                text=f"  {task['Job']} (PT={task['Finish']-task['Start']})",
                textposition="inside",
                showlegend=False
            ))
            fig.add_vline(x=task["Due"], line_dash="dot", line_color="orange",
                          line_width=1)

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color="orange", dash="dot"),
                                 name="Due Date"))
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h",
                             marker_color="#2ecc71", name="On Time"))
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h",
                             marker_color="#e74c3c", name="Tardy"))

        fig.update_layout(
            title=f"Gantt Chart — {rule} Rule",
            xaxis_title="Time", barmode="overlay",
            template="plotly_white", height=60 * n_jobs + 120,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rule comparison
        st.markdown("#### Rule Comparison (All Rules)")
        all_rules = ["FCFS", "SPT", "EDD", "LPT", "CR", "SLACK"]
        comparison_rows = []
        sort_map = {"FCFS": (None, False), "SPT": ("pt", False), "EDD": ("dd", False),
                    "LPT": ("pt", True),  "CR": ("cr", False), "SLACK": ("slack", False)}
        for r in all_rules:
            key, rev = sort_map[r]
            sj = jobs[:] if key is None else sorted(jobs, key=lambda x: x[key], reverse=rev)
            ct = today
            ft, td = [], []
            for job in sj:
                ct += job["pt"]
                ft.append(ct - today)
                td.append(max(0, ct - job["dd"]))
            comparison_rows.append({
                "Rule": r,
                "Avg Flow Time": round(sum(ft)/len(ft), 2),
                "Avg Tardiness": round(sum(td)/len(td), 2),
                "# Tardy": sum(1 for t in td if t > 0)
            })
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    with tab3:
        st.markdown("### 📝 Practice Problems")

        with st.expander("🟢 P1: Which rule minimizes average flow time?"):
            display_practice_problem(1, "Easy",
                "You have 5 jobs with varying processing times. Which sequencing rule is "
                "provably optimal for minimizing average flow time?")
            if st.button("Show Answer", key="sch_p1"):
                display_solution(
                    "**SPT (Shortest Processing Time)** is provably optimal for minimizing "
                    "average flow time, average WIP, and average lateness on a single machine."
                )

        with st.expander("🟡 P2: Critical Ratio Classification"):
            display_practice_problem(2, "Medium",
                "Job X has PT = 4 days, due date = Day 10, and today = Day 7. "
                "Calculate CR and classify the job.")
            if st.button("Show Answer", key="sch_p2"):
                display_solution(
                    "CR = (10 − 7) / 4 = **0.75**. Since CR < 1, Job X is **behind schedule** "
                    "and should receive high priority."
                )

        with st.expander("🔴 P3: Compare FCFS vs SPT"):
            display_practice_problem(3, "Hard",
                "Jobs A(4), B(1), C(3), D(2) arrive in that order. Due dates: A=7, B=3, C=6, D=5. "
                "Today = 0. Compare FCFS vs SPT: compute avg flow time for each.")
            if st.button("Show Answer", key="sch_p3"):
                display_solution(
                    "**FCFS** (A→B→C→D): Finish times = 4, 5, 8, 10 → Avg Flow = (4+5+8+10)/4 = **6.75**\n\n"
                    "**SPT** (B→D→C→A): Finish times = 1, 3, 6, 10 → Avg Flow = (1+3+6+10)/4 = **5.0**\n\n"
                    "SPT saves 1.75 days average flow time — a 26% improvement."
                )


# ============================================================
# MODULE 34: POKA-YOKE (Chapter 9) - ENHANCED V5.0
# ============================================================
def module_pokayoke():
    display_header("🛡️", "Chapter 9", "Poka-yoke (Mistake-Proofing) Database",
                   "Prevention and detection techniques for zero-defect operations")

    tab1, tab2 = st.tabs(["📚 Theory & Database", "🔬 Design Tool"])

    with tab1:
        st.markdown("### What is Poka-yoke?")
        st.write(
            "**Poka-yoke** (Japanese: 'mistake-proofing') refers to any mechanism that either "
            "prevents a mistake from being made, or makes a defect immediately obvious once it has occurred. "
            "Developed by Shigeo Shingo as part of the Toyota Production System."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            display_concept_card("🚧", "Prevention (Control)",
                                 "Makes the error physically impossible — e.g., USB-C connector "
                                 "shape, floppy disk notch.")
        with col2:
            display_concept_card("🔔", "Detection (Warning)",
                                 "Allows the error but alerts immediately — e.g., car door-ajar "
                                 "chime, ATM card beep.")
        with col3:
            display_concept_card("🔍", "Shutdown",
                                 "Auto-stops the process when a defect is detected — e.g., "
                                 "Jidoka on a Toyota assembly line.")

        st.markdown("### Poka-yoke Examples Database")

        all_examples = [
            ("Manufacturing", "Wrong part assembled",      "Fixture / sensor rejects wrong part",     "Prevention"),
            ("Manufacturing", "Missing component",         "Shadow board; part-count sensor",          "Detection"),
            ("Manufacturing", "Wrong torque applied",      "Torque wrench with auto-cutoff",           "Prevention"),
            ("Healthcare",    "Wrong medication given",    "Bar-code scanning of patient + medicine",  "Prevention"),
            ("Healthcare",    "Surgical sponge left in",   "Count protocol; RFID sponges",             "Detection"),
            ("Healthcare",    "Wrong patient surgery",     "Pre-surgery time-out checklist",           "Prevention"),
            ("Service",       "Bill is illegible",         "Top copy to customer (carbon copy)",       "Prevention"),
            ("Service",       "Card left in ATM",          "Beep + card-eject before cash dispensed",  "Prevention"),
            ("Service",       "Feedback not obtained",     "Postcard / QR code on receipt",            "Detection"),
            ("Software",      "Incorrect data entry",      "Input validation, dropdowns, masks",       "Prevention"),
            ("Software",      "Unsaved work lost",         "Auto-save & confirmation prompts",         "Detection"),
            ("Aviation",      "Gear not deployed",         "Landing warning horn (gear sensor)",       "Detection"),
        ]

        df_pk = pd.DataFrame(all_examples,
                             columns=["Industry", "Failure Mode", "Poka-yoke Solution", "Type"])

        # Filter
        industries  = ["All"] + sorted(df_pk["Industry"].unique())
        types       = ["All", "Prevention", "Detection", "Shutdown"]
        col1, col2  = st.columns(2)
        sel_ind  = col1.selectbox("Filter by Industry", industries)
        sel_type = col2.selectbox("Filter by Type",     types)

        filtered = df_pk.copy()
        if sel_ind  != "All": filtered = filtered[filtered["Industry"] == sel_ind]
        if sel_type != "All": filtered = filtered[filtered["Type"]     == sel_type]

        def color_type(val):
            colors = {"Prevention": "background-color:#d4edda",
                      "Detection":  "background-color:#fff3cd",
                      "Shutdown":   "background-color:#f8d7da"}
            return colors.get(val, "")

        st.dataframe(filtered.style.applymap(color_type, subset=["Type"]),
                     use_container_width=True)
        st.caption("🟢 Prevention | 🟡 Detection | 🔴 Shutdown")

        display_citation(
            "Poka-yoke devices prevent defects by making errors impossible or immediately visible. "
            "Shingo argued that inspection only finds defects; poka-yoke prevents them.",
            "Jacobs & Chase (2024, Ch. 9)"
        )

    with tab2:
        st.markdown("### Poka-yoke Design Tool")
        st.write("Classify a failure mode and identify the appropriate mistake-proofing strategy.")

        failure = st.text_input("Describe the Failure Mode",
                                placeholder="e.g., Operator installs part backwards")
        stage   = st.selectbox("Process Stage",
                               ["Design", "Incoming Material", "Production", "Assembly",
                                "Inspection", "Shipping", "Service"])
        impact  = st.select_slider("Severity of Error",
                                   options=["Low", "Medium", "High", "Critical"])

        if failure:
            st.markdown("#### Recommended Approach")
            if impact in ["High", "Critical"]:
                st.error(f"⛔ **{impact} severity** — Use **Prevention (Control)** poka-yoke. "
                         "Make the error physically impossible.")
                st.markdown("**Suggestions:** Keyed fixtures, asymmetric connectors, "
                            "sensors that reject wrong parts, automated interlocks.")
            else:
                st.warning(f"⚠️ **{impact} severity** — Use **Detection (Warning)** poka-yoke. "
                           "Alert the operator immediately.")
                st.markdown("**Suggestions:** Andon lights, auditory alerts, color coding, "
                            "checklists, counters.")

            st.info(f"📍 **Stage:** {stage} | Consider adding poka-yoke as close to the error "
                    "source as possible — the later detected, the higher the cost.")


# ============================================================
# MODULE 35: SQC PRACTICE (Chapter 13) - ENHANCED V6.0
# ============================================================
def module_sqc_practice():
    display_header("🎓", "Chapter 13", "SQC Practice Questions",
                   "Statistical Quality Control — exam preparation & interactive tools")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Q&A Bank", "🧮 Interactive Calculators", "📊 Formula Reference", "🏆 Quiz Mode"
    ])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### SQC Question Bank")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            difficulty = st.selectbox("Filter by Difficulty",
                                      ["All", "Easy", "Medium", "Hard"], key="sqc_diff")
        with col_f2:
            topic_filter = st.selectbox("Filter by Topic",
                                        ["All", "Control Charts", "Process Capability",
                                         "Acceptance Sampling", "Six Sigma / DPMO",
                                         "Western Electric Rules", "c-Chart / p-Chart"], key="sqc_topic")

        questions = [
            # (difficulty, topic, title, question, answer)
            ("Easy", "Six Sigma / DPMO",
             "Six Sigma DPMO",
             "A Six Sigma process centered at the mean produces how many DPMO?",
             "**3.4 DPMO.** A Six Sigma process (±6σ from target) with a 1.5σ shift "
             "yields 3.4 defects per million opportunities. This equates to 99.99966% yield."),

            ("Easy", "Six Sigma / DPMO",
             "DMAIC Acronym",
             "What does DMAIC stand for and what is each phase's purpose?",
             "**Define** — identify the problem and project scope.<br>"
             "**Measure** — quantify the current process baseline.<br>"
             "**Analyze** — find root causes of defects.<br>"
             "**Improve** — implement and test solutions.<br>"
             "**Control** — sustain improvements with monitoring plans."),

            ("Easy", "Control Charts",
             "3-Sigma Confidence",
             "A z-value of ±3 gives what percentage of data within control limits?",
             "**99.73%** of data falls within ±3σ, leaving 0.27% (2,700 ppm) outside. "
             "This corresponds to a false-alarm rate (Type I error) of 0.0027 per point."),

            ("Easy", "Control Charts",
             "Common vs Special Cause",
             "What is the difference between common-cause and special-cause variation?",
             "**Common-cause variation** is random, inherent to the process — always present and "
             "predictable within control limits.<br><br>"
             "**Special-cause variation** is assignable — caused by a specific, identifiable event "
             "(machine failure, operator change, bad batch). Points outside control limits signal special causes."),

            ("Easy", "c-Chart / p-Chart",
             "p-Chart vs c-Chart",
             "When do you use a p-chart vs a c-chart?",
             "**p-chart:** Tracks the *fraction defective* per sample. Use when each unit is "
             "either defective or not (binomial). Sample sizes can vary.<br><br>"
             "**c-chart:** Tracks the *number of defects per unit*. Use when counting multiple "
             "defects on a single unit (Poisson). Requires constant sample area/size."),

            ("Easy", "Control Charts",
             "x̄ & R Chart Purpose",
             "Why are the x̄-chart and R-chart always used together?",
             "The **x̄-chart** monitors the process *mean* (center), while the "
             "**R-chart** monitors the process *variability* (spread).<br><br>"
             "A process can have a stable mean but unstable spread, or vice versa. "
             "Using both simultaneously gives a complete picture of process behavior."),

            ("Medium", "Process Capability",
             "Cpk Calculation",
             "USL = 1.255, LSL = 1.245, mean = 1.251, σ = 0.00083. Calculate Cp and Cpk.",
             "Cp = (USL − LSL) / 6σ = (1.255 − 1.245) / (6 × 0.00083) = 0.010 / 0.00498 = **2.01**<br><br>"
             "Cpu = (1.255 − 1.251) / (3 × 0.00083) = **1.606**<br>"
             "Cpl = (1.251 − 1.245) / (3 × 0.00083) = **2.41**<br>"
             "Cpk = min(1.606, 2.41) = **1.61** — Process is very capable but slightly off-center toward USL."),

            ("Medium", "Process Capability",
             "Normal Distribution Tail",
             "Washers: mean = 2.0 mm, σ = 0.2 mm. What fraction exceed 2.4 mm?",
             "Z = (2.4 − 2.0) / 0.2 = **2.0**<br>"
             "P(Z > 2.0) = 1 − Φ(2.0) = 1 − 0.9772 = **0.0228 = 2.28%** of washers exceed 2.4 mm."),

            ("Medium", "c-Chart / p-Chart",
             "p-Chart UCL",
             "Sample size n = 100, average defect proportion p̄ = 0.04. Calculate UCL and LCL.",
             "σ_p = √(p̄(1−p̄)/n) = √(0.04 × 0.96 / 100) = √0.000384 = **0.01960**<br><br>"
             "UCL = 0.04 + 3(0.01960) = **0.0988 ≈ 9.88%**<br>"
             "LCL = max(0, 0.04 − 3(0.01960)) = max(0, −0.0188) = **0** (set to zero)"),

            ("Medium", "c-Chart / p-Chart",
             "c-Chart Control Limits",
             "20 units sampled, total defects = 90. Calculate c̄, UCL, and LCL.",
             "c̄ = 90 / 20 = **4.5 defects/unit**<br><br>"
             "UCL = c̄ + 3√c̄ = 4.5 + 3√4.5 = 4.5 + 6.36 = **10.86**<br>"
             "LCL = max(0, 4.5 − 3√4.5) = max(0, 4.5 − 6.36) = max(0, −1.86) = **0**"),

            ("Medium", "Control Charts",
             "x̄-Chart Constants",
             "Subgroup size n = 5. Given x̄̄ = 12.0 and R̄ = 0.8. Find UCL and LCL for x̄-chart. (A₂ = 0.577)",
             "UCL_x̄ = x̄̄ + A₂ × R̄ = 12.0 + 0.577 × 0.8 = 12.0 + 0.462 = **12.462**<br>"
             "LCL_x̄ = x̄̄ − A₂ × R̄ = 12.0 − 0.462 = **11.538**<br><br>"
             "Interpretation: Any subgroup mean outside [11.538, 12.462] signals a special cause."),

            ("Medium", "Control Charts",
             "R-Chart Limits",
             "n = 5, R̄ = 0.8. Find UCL_R and LCL_R. (D₃ = 0, D₄ = 2.114)",
             "UCL_R = D₄ × R̄ = 2.114 × 0.8 = **1.691**<br>"
             "LCL_R = D₃ × R̄ = 0 × 0.8 = **0** (no lower limit when n < 7)<br><br>"
             "Note: D₃ = 0 for n ≤ 6, meaning only the upper limit is active."),

            ("Medium", "Acceptance Sampling",
             "AQL vs LTPD",
             "Define AQL and LTPD in the context of an OC curve.",
             "**AQL (Acceptable Quality Level):** The worst defect rate the *producer* considers "
             "acceptable. Associated with producer's risk α (probability of rejecting a good lot).<br><br>"
             "**LTPD (Lot Tolerance Percent Defective):** The defect rate the *consumer* considers "
             "unacceptable. Associated with consumer's risk β (probability of accepting a bad lot)."),

            ("Hard", "Process Capability",
             "Cp vs Cpk Interpretation",
             "A process has Cp = 1.5 but Cpk = 0.9. What does this tell you and what action is needed?",
             "**Cp = 1.5** → The process spread is narrow enough to fit within the tolerance "
             "(potential capability is good — 4.5σ headroom).<br><br>"
             "**Cpk = 0.9 < 1.0** → The process *mean is off-center*, too close to one spec limit, "
             "causing defects on that side despite sufficient spread.<br><br>"
             "**Action:** Re-center the process mean. DO NOT reduce variability first — "
             "the spread is already fine. Centering alone will raise Cpk to equal Cp = 1.5."),

            ("Hard", "Acceptance Sampling",
             "OC Curve: α and β Risks",
             "What is the producer's risk (α) and consumer's risk (β) on an OC curve?",
             "**α (Producer's Risk / Type I):** Probability of *rejecting a good lot* "
             "(one at AQL quality). Typically α = 0.05. The producer bears this cost in "
             "unnecessary returns and rework.<br><br>"
             "**β (Consumer's Risk / Type II):** Probability of *accepting a bad lot* "
             "(one at LTPD quality). Typically β = 0.10. The consumer bears this cost in "
             "receiving defective product."),

            ("Hard", "Western Electric Rules",
             "Run Rule: 8 Consecutive Points",
             "Eight consecutive points fall on the same side of the center line, all within "
             "control limits. Is the process in control?",
             "**No — this is an out-of-control signal.** Western Electric Rule 2 states that "
             "8+ consecutive points on one side of the center line indicate the process mean "
             "has shifted, even though no single point exceeds the 3σ limits.<br><br>"
             "Probability of 8 consecutive points on one side by chance alone: "
             "(0.5)⁸ = **0.39%** — highly unlikely to be random."),

            ("Hard", "Western Electric Rules",
             "Zone Rules Explained",
             "Explain the Western Electric Zone A, B, and C rules.",
             "Control limits are divided into 3 equal zones each side of center:<br>"
             "• **Zone C (0–1σ):** 2 of 3 points in Zone A (near limits) on same side → signal<br>"
             "• **Zone B (1–2σ):** 4 of 5 points in Zone B or beyond on same side → signal<br>"
             "• **Zone A (2–3σ):** 1 point in Zone A is a warning; 2 consecutive in Zone A = signal<br><br>"
             "These rules detect gradual shifts that individual points miss."),

            ("Hard", "Six Sigma / DPMO",
             "Sigma Level from DPMO",
             "A process has DPMO = 6,210. What is the approximate sigma level?",
             "Using the approximation formula:<br>"
             "σ ≈ 0.8406 + √(29.37 − 2.221 × ln(DPMO))<br>"
             "= 0.8406 + √(29.37 − 2.221 × ln(6210))<br>"
             "= 0.8406 + √(29.37 − 2.221 × 8.734)<br>"
             "= 0.8406 + √(29.37 − 19.40)<br>"
             "= 0.8406 + √9.97 = 0.8406 + 3.158 = **≈ 4.0σ**<br><br>"
             "Benchmark: 4σ ≈ 6,210 DPMO; 3σ ≈ 66,807 DPMO; 6σ ≈ 3.4 DPMO."),

            ("Hard", "c-Chart / p-Chart",
             "p-Chart with Variable Sample Size",
             "Sample 1: n=200, d=8. Sample 2: n=150, d=9. Overall p̄ = 0.04. "
             "Which sample is OOC? (Check each against its own UCL.)",
             "Each sample gets its own UCL because n varies:<br><br>"
             "**Sample 1 (n=200):** σ_p = √(0.04×0.96/200) = 0.01386 → UCL = 0.04 + 3(0.01386) = **0.0816**<br>"
             "p₁ = 8/200 = 0.040 ✅ Within limits<br><br>"
             "**Sample 2 (n=150):** σ_p = √(0.04×0.96/150) = 0.01600 → UCL = 0.04 + 3(0.01600) = **0.0880**<br>"
             "p₂ = 9/150 = 0.060 ✅ Within limits<br><br>"
             "Both in control — but note that smaller n produces wider limits."),
        ]

        # ── Filter ────────────────────────────────────────────
        filtered_q = questions
        if difficulty != "All":
            filtered_q = [q for q in filtered_q if q[0] == difficulty]
        if topic_filter != "All":
            filtered_q = [q for q in filtered_q if q[1] == topic_filter]

        # ── Stats bar ─────────────────────────────────────────
        easy_n   = sum(1 for q in questions if q[0] == "Easy")
        med_n    = sum(1 for q in questions if q[0] == "Medium")
        hard_n   = sum(1 for q in questions if q[0] == "Hard")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Questions", len(questions))
        c2.metric("🟢 Easy",   easy_n)
        c3.metric("🟡 Medium", med_n)
        c4.metric("🔴 Hard",   hard_n)

        if not filtered_q:
            display_callout("No questions match the selected filters.", "warning")
        else:
            st.markdown(f"**Showing {len(filtered_q)} question(s).**")
            diff_color = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            for i, (diff, topic, title, question, answer) in enumerate(filtered_q):
                with st.expander(
                    f"{diff_color[diff]} **Q{i+1} [{diff}] · {topic}** — {title}"
                ):
                    display_practice_problem(i + 1, diff, question)
                    hint_col, ans_col = st.columns([1, 1])
                    with hint_col:
                        hints = {
                            "Cpk Calculation":         "Hint: Cpk = min(Cpu, Cpl) where each uses 3σ.",
                            "p-Chart UCL":             "Hint: σ_p = √(p̄(1−p̄)/n).",
                            "c-Chart Control Limits":  "Hint: UCL = c̄ + 3√c̄.",
                            "x̄-Chart Constants":       "Hint: UCL = x̄̄ + A₂ × R̄.",
                            "Sigma Level from DPMO":   "Hint: σ ≈ 0.8406 + √(29.37 − 2.221 × ln(DPMO)).",
                        }
                        if title in hints:
                            display_hint(hints[title])
                    with ans_col:
                        if st.button("Show Answer ▼", key=f"sqc_prac_{i}_{diff}_{topic[:4]}",
                                     use_container_width=True):
                            st.session_state[f"sqc_ans_{i}"] = True
                    if st.session_state.get(f"sqc_ans_{i}"):
                        display_solution(answer)

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🧮 Interactive SQC Calculators")

        calc = st.selectbox("Choose Calculator", [
            "p-Chart Builder",
            "c-Chart Builder",
            "Process Capability (Cpk)",
            "DPMO ↔ Sigma Level",
            "x̄ & R Chart Limits",
            "Sample Size for p-Chart",
        ], key="sqc_calc_choice")

        st.markdown("---")

        if calc == "p-Chart Builder":
            st.markdown("#### p-Chart Control Limit Calculator")
            c1, c2 = st.columns(2)
            with c1:
                p_bar_in = st.number_input("Average Proportion Defective (p̄)", value=0.04,
                                           min_value=0.001, max_value=0.999, format="%.4f")
                n_in     = st.number_input("Sample Size (n)", value=100, min_value=5)
                z_in     = st.selectbox("Z-Value (sigma limits)", [2.0, 2.5, 3.0, 3.5], index=2)
            with c2:
                sp       = math.sqrt(p_bar_in * (1 - p_bar_in) / n_in)
                ucl      = p_bar_in + z_in * sp
                lcl      = max(0, p_bar_in - z_in * sp)
                display_metric_card(f"{p_bar_in:.4f} ({p_bar_in:.2%})", "Center Line (p̄)", "normal")
                display_metric_card(f"{ucl:.4f} ({ucl:.2%})", "UCL", "danger")
                display_metric_card(f"{max(0,lcl):.4f} ({max(0,lcl):.2%})", "LCL", "danger")
                display_metric_card(f"{sp:.5f}", "σ_p", "normal")
            display_formula_card("p-Chart UCL/LCL",
                r"\bar{p} \pm z\sqrt{\frac{\bar{p}(1-\bar{p})}{n}}")
            display_callout(
                f"With n={n_in} and p̄={p_bar_in:.3f}, minimum sample size to detect a "
                f"shift of 0.01 with 95% power ≈ {int(math.ceil((z_in*sp/0.01)**2))} units.",
                "info", "Design Tip"
            )

        elif calc == "c-Chart Builder":
            st.markdown("#### c-Chart Control Limit Calculator")
            c1, c2 = st.columns(2)
            with c1:
                c_bar_in = st.number_input("Average Defects per Unit (c̄)", value=4.5,
                                           min_value=0.1, format="%.2f")
                z_c      = st.selectbox("Z-Value", [2.0, 2.5, 3.0, 3.5], index=2,
                                        key="c_z")
            with c2:
                sqrt_c   = math.sqrt(c_bar_in)
                ucl_c    = c_bar_in + z_c * sqrt_c
                lcl_c    = max(0, c_bar_in - z_c * sqrt_c)
                display_metric_card(f"{c_bar_in:.3f}", "Center Line (c̄)", "normal")
                display_metric_card(f"{ucl_c:.3f}", "UCL", "danger")
                display_metric_card(f"{lcl_c:.3f}", "LCL (0 if negative)", "danger")
                display_metric_card(f"{sqrt_c:.4f}", "√c̄ (Std Dev)", "normal")
            display_formula_card("c-Chart UCL/LCL",
                r"\bar{c} \pm 3\sqrt{\bar{c}}")

        elif calc == "Process Capability (Cpk)":
            st.markdown("#### Process Capability Calculator")
            c1, c2 = st.columns(2)
            with c1:
                lsl  = st.number_input("LSL", value=44.0)
                usl  = st.number_input("USL", value=58.0)
                mean = st.number_input("Process Mean (μ)", value=50.0)
                sig  = st.number_input("Process Std Dev (σ)", value=2.0, min_value=0.0001,
                                       format="%.4f")
            with c2:
                result = process_capability(mean, sig, lsl, usl)
                cp, cpk, cpu, cpl = result["Cp"], result["Cpk"], result["Cpu"], result["Cpl"]

                cp_type  = "success" if cp  >= 1.33 else ("warning" if cp  >= 1.0 else "danger")
                cpk_type = "success" if cpk >= 1.33 else ("warning" if cpk >= 1.0 else "danger")

                display_metric_card(f"{cp:.3f}",  "Cp  (Potential)",  cp_type)
                display_metric_card(f"{cpk:.3f}", "Cpk (Actual)",     cpk_type)
                display_metric_card(f"{cpl:.3f}", "Cpl (Lower)",      "normal")
                display_metric_card(f"{cpu:.3f}", "Cpu (Upper)",      "normal")

            sig_lvl = sigma_level(cpk)
            if cp and cpk:
                off_center = abs(cp - cpk) / cp * 100 if cp > 0 else 0
                interp = (
                    "✅ Capable and well-centered" if cpk >= 1.33 and off_center < 10 else
                    "⚠️ Capable but off-center — consider re-centering" if cp >= 1.33 and cpk < cp * 0.9 else
                    "⚠️ Marginally capable — monitor closely" if 1.0 <= cpk < 1.33 else
                    "🚨 Not capable — process produces defects"
                )
                display_callout(
                    f"{interp}<br>Approx. sigma level: **{sig_lvl:.2f}σ** | "
                    f"Off-center: {off_center:.1f}%",
                    "success" if cpk >= 1.33 else "warning" if cpk >= 1.0 else "danger",
                    "Interpretation"
                )
            display_formula_card("Cpk",
                r"C_{pk} = \min\!\left(\frac{USL-\mu}{3\sigma},\frac{\mu-LSL}{3\sigma}\right)")

        elif calc == "DPMO ↔ Sigma Level":
            st.markdown("#### DPMO ↔ Sigma Level Converter")
            mode = st.radio("Direction", ["Defects → DPMO & σ", "Sigma Level → DPMO"],
                            horizontal=True)
            if mode == "Defects → DPMO & σ":
                c1, c2 = st.columns(2)
                with c1:
                    defects_i = st.number_input("Total Defects", value=33, min_value=0)
                    units_i   = st.number_input("Total Units", value=2000, min_value=1)
                    opp_i     = st.number_input("Opportunities / Unit", value=5, min_value=1)
                with c2:
                    dpmo_i = (defects_i / (units_i * opp_i)) * 1_000_000 if units_i * opp_i > 0 else 0
                    if 0 < dpmo_i < 1_000_000:
                        sl = 0.8406 + math.sqrt(max(0, 29.37 - 2.221 * math.log(dpmo_i)))
                    else:
                        sl = 0.0
                    yield_pct = (1 - defects_i / (units_i * opp_i)) * 100 if units_i * opp_i > 0 else 0
                    display_metric_card(f"{dpmo_i:,.1f}", "DPMO", "danger" if dpmo_i > 6210 else "success")
                    display_metric_card(f"{sl:.2f}σ", "Sigma Level", "success" if sl >= 4 else "warning")
                    display_metric_card(f"{yield_pct:.4f}%", "Process Yield", "normal")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    target_sigma = st.slider("Target Sigma Level", 1.0, 6.5, 4.0, 0.1)
                with c2:
                    # Approximate reverse: DPMO from sigma
                    z_eq   = target_sigma - 1.5   # account for 1.5σ shift
                    dpmo_r = (1 - normal_cdf(z_eq)) * 1_000_000
                    display_metric_card(f"{dpmo_r:,.1f}", "Estimated DPMO", "normal")
                    display_metric_card(f"{(1-dpmo_r/1e6)*100:.4f}%", "Yield", "success")

            # Reference table
            with st.expander("📋 Sigma Level Reference Table"):
                ref_df = pd.DataFrame({
                    "Sigma Level": ["1σ", "2σ", "3σ", "4σ", "5σ", "6σ"],
                    "DPMO":        ["691,462", "308,538", "66,807", "6,210", "233", "3.4"],
                    "Yield (%)":   ["30.85", "69.15", "93.32", "99.38", "99.977", "99.99966"],
                    "Example":     [
                        "Very poor process",
                        "Average company",
                        "Industry standard",
                        "Good process",
                        "Near world class",
                        "World class (Six Sigma goal)"
                    ]
                })
                st.dataframe(ref_df, use_container_width=True, hide_index=True)

        elif calc == "x̄ & R Chart Limits":
            st.markdown("#### x̄ & R Chart Limit Calculator")
            a2t = {2:1.880, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
            d3t = {2:0,     3:0,     4:0,     5:0,     6:0,     7:0.076, 8:0.136, 9:0.184, 10:0.223}
            d4t = {2:3.267, 3:2.574, 4:2.282, 5:2.114, 6:2.004, 7:1.924, 8:1.864, 9:1.816, 10:1.777}
            c1, c2 = st.columns(2)
            with c1:
                n_sel   = st.selectbox("Subgroup Size (n)", list(range(2, 11)), index=3)
                xdbl    = st.number_input("Grand Mean (x̄̄)", value=12.0, format="%.4f")
                r_bar   = st.number_input("Average Range (R̄)", value=0.8, min_value=0.0, format="%.4f")
            with c2:
                A2, D3, D4 = a2t[n_sel], d3t[n_sel], d4t[n_sel]
                ucl_x = xdbl + A2 * r_bar
                lcl_x = xdbl - A2 * r_bar
                ucl_r = D4 * r_bar
                lcl_r = D3 * r_bar
                st.markdown("**x̄-Chart**")
                c2a, c2b, c2c = st.columns(3)
                c2a.metric("UCL_x̄", f"{ucl_x:.4f}")
                c2b.metric("CL", f"{xdbl:.4f}")
                c2c.metric("LCL_x̄", f"{lcl_x:.4f}")
                st.markdown("**R-Chart**")
                c2d, c2e, c2f = st.columns(3)
                c2d.metric("UCL_R", f"{ucl_r:.4f}")
                c2e.metric("CL (R̄)", f"{r_bar:.4f}")
                c2f.metric("LCL_R", f"{lcl_r:.4f}" if D3 > 0 else "N/A (n<7)")
                st.write(f"**Constants:** A₂={A2}, D₃={D3}, D₄={D4}")

        elif calc == "Sample Size for p-Chart":
            st.markdown("#### Minimum Sample Size for p-Chart")
            c1, c2 = st.columns(2)
            with c1:
                p_est   = st.number_input("Estimated p̄", value=0.04, min_value=0.001,
                                          max_value=0.5, format="%.3f")
                delta_p = st.number_input("Detectable Shift (Δp)", value=0.02,
                                          min_value=0.001, format="%.3f")
                power   = st.selectbox("Desired Power", ["80% (z=0.84)", "90% (z=1.28)",
                                                         "95% (z=1.65)"])
                z_pow   = {"80% (z=0.84)": 0.84, "90% (z=1.28)": 1.28, "95% (z=1.65)": 1.65}[power]
            with c2:
                z_alpha = 3.0   # standard SPC 3-sigma
                n_min   = math.ceil(
                    ((z_alpha + z_pow) ** 2 * p_est * (1 - p_est)) / delta_p ** 2
                )
                sp_at_n = math.sqrt(p_est * (1 - p_est) / n_min)
                ucl_at_n = p_est + 3 * sp_at_n
                display_metric_card(f"{n_min}", "Min Sample Size", "highlight")
                display_metric_card(f"{sp_at_n:.5f}", "σ_p at this n", "normal")
                display_metric_card(f"{ucl_at_n:.4f}", "UCL at this n", "normal")

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 SQC Formula Quick Reference")

        col_f, col_c = st.columns(2)
        with col_f:
            st.markdown("#### Process Capability")
            display_formula_card("Cp (Potential Capability)",
                r"C_p = \frac{USL - LSL}{6\sigma}")
            display_formula_card("Cpk (Actual Capability)",
                r"C_{pk} = \min\!\left(\frac{USL-\mu}{3\sigma},\frac{\mu-LSL}{3\sigma}\right)")
            display_formula_card("Sigma Level",
                r"\sigma_{level} = 3 \times C_{pk}")

            st.markdown("#### Attribute Charts")
            display_formula_card("p-Chart Std Error",
                r"\sigma_p = \sqrt{\frac{\bar{p}(1-\bar{p})}{n}}")
            display_formula_card("p-Chart UCL/LCL",
                r"\bar{p} \pm 3\sigma_p")
            display_formula_card("c-Chart UCL/LCL",
                r"\bar{c} \pm 3\sqrt{\bar{c}}")

        with col_c:
            st.markdown("#### Variables Charts")
            display_formula_card("x̄-Chart UCL/LCL",
                r"UCL/LCL_{\bar{x}} = \bar{\bar{x}} \pm A_2\bar{R}")
            display_formula_card("R-Chart UCL",
                r"UCL_R = D_4\bar{R}")
            display_formula_card("R-Chart LCL",
                r"LCL_R = D_3\bar{R}")

            st.markdown("#### Six Sigma")
            display_formula_card("DPMO",
                r"DPMO = \frac{Defects}{Units \times Opportunities} \times 10^6")
            display_formula_card("Approx Sigma Level",
                r"\approx 0.8406 + \sqrt{29.37 - 2.221\ln(DPMO)}")

        st.markdown("---")
        st.markdown("#### Control Chart Constants Table")
        const_df = pd.DataFrame({
            "n (subgroup)": [2,3,4,5,6,7,8,9,10],
            "A₂":           [1.880,1.023,0.729,0.577,0.483,0.419,0.373,0.337,0.308],
            "D₃":           [0,0,0,0,0,0.076,0.136,0.184,0.223],
            "D₄":           [3.267,2.574,2.282,2.114,2.004,1.924,1.864,1.816,1.777],
            "d₂":           [1.128,1.693,2.059,2.326,2.534,2.704,2.847,2.970,3.078],
        })
        st.dataframe(const_df, use_container_width=True, hide_index=True)

        display_key_insight("When to Apply Each Chart",
            "p-chart: fraction defective, variable n (binomial) | "
            "np-chart: count defective, constant n | "
            "c-chart: count defects/unit, constant area (Poisson) | "
            "u-chart: defects/unit, variable area | "
            "x̄-R: continuous data, n=2–10 | x̄-s: continuous data, n>10")

        st.markdown("#### Western Electric Rules (4 Main Rules)")
        we_df = pd.DataFrame({
            "Rule": ["Rule 1", "Rule 2", "Rule 3", "Rule 4"],
            "Signal": [
                "1 point beyond ±3σ (Zone A)",
                "8+ consecutive points on one side of CL",
                "6+ points trending in same direction",
                "2 of 3 consecutive points in Zone A (beyond ±2σ)"
            ],
            "Indicates": [
                "Large sudden shift or outlier",
                "Process mean has shifted",
                "Gradual drift or trend",
                "Large sustained shift"
            ]
        })
        st.dataframe(we_df, use_container_width=True, hide_index=True)

        st.markdown("#### Chart Selection Guide")
        chart_guide = pd.DataFrame({
            "Data Type":      ["Variables (x̄ & R)", "Variables (x̄ & s)", "Attributes (defective)",
                               "Attributes (defective)", "Attributes (defects)", "Attributes (defects)"],
            "Chart":          ["x̄-R", "x̄-s", "p-chart", "np-chart", "c-chart", "u-chart"],
            "Sample Size n":  ["2–10", "> 10", "Variable", "Constant", "Constant area", "Variable area"],
            "Distribution":   ["Normal", "Normal", "Binomial", "Binomial", "Poisson", "Poisson"],
            "Measures":       ["Mean & range", "Mean & std dev",
                               "Fraction defective", "Count defective",
                               "Defects per unit", "Defects per unit (adjusted)"]
        })
        st.dataframe(chart_guide, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🏆 SQC Quiz Mode")
        st.write("Test yourself — answer without looking. Track your score below.")

        if "sqc_quiz_score" not in st.session_state:
            st.session_state.sqc_quiz_score  = 0
            st.session_state.sqc_quiz_total  = 0
            st.session_state.sqc_quiz_streak = 0

        quiz_questions = [
            ("What is the UCL formula for a c-chart?",
             r"\bar{c} + 3\sqrt{\bar{c}}",
             "UCL = c̄ + 3√c̄"),
            ("What does Cpk measure?",
             None,
             "Actual process capability — accounts for both spread AND centering of the mean."),
            ("For n=5, A₂=0.577. If R̄=1.2 and x̄̄=10, what is UCL_x̄?",
             None,
             "UCL = 10 + 0.577 × 1.2 = 10.692"),
            ("DPMO = 66,807. What sigma level is this approximately?",
             None,
             "≈ 3 sigma (3σ quality level)"),
            ("What does 'in control but not capable' mean?",
             None,
             "The process is statistically stable (no special causes) but the spread is too wide "
             "to meet specifications — Cpk < 1.0 despite points within control limits."),
            ("p̄ = 0.05, n = 200. Calculate σ_p.",
             r"\sigma_p = \sqrt{\frac{0.05 \times 0.95}{200}} = 0.01541",
             "σ_p = √(0.05 × 0.95 / 200) = √0.0002375 ≈ 0.01541"),
        ]

        q_idx = st.session_state.sqc_quiz_total % len(quiz_questions)
        q_text, q_latex, q_answer = quiz_questions[q_idx]

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Score",  f"{st.session_state.sqc_quiz_score}/{st.session_state.sqc_quiz_total}")
        sc2.metric("Streak", f"🔥 {st.session_state.sqc_quiz_streak}")
        pct = (st.session_state.sqc_quiz_score / max(st.session_state.sqc_quiz_total, 1)) * 100
        sc3.metric("Accuracy", f"{pct:.0f}%")

        display_practice_problem(q_idx + 1, "Medium", q_text)
        if q_latex:
            st.latex(q_latex)

        qa_col1, qa_col2 = st.columns(2)
        with qa_col1:
            if st.button("✅ I Got It Right", key="sqc_quiz_right", use_container_width=True):
                st.session_state.sqc_quiz_score  += 1
                st.session_state.sqc_quiz_total  += 1
                st.session_state.sqc_quiz_streak += 1
                st.session_state.problems_solved += 1
                st.session_state.correct_streak  += 1
                display_solution(q_answer)
        with qa_col2:
            if st.button("❌ Show Answer (I Missed)", key="sqc_quiz_wrong",
                         use_container_width=True):
                st.session_state.sqc_quiz_total  += 1
                st.session_state.sqc_quiz_streak  = 0
                st.session_state.correct_streak   = 0
                display_solution(q_answer)

        if st.button("⏭ Next Question →", key="sqc_quiz_next", use_container_width=True):
            st.session_state.sqc_quiz_total = st.session_state.sqc_quiz_total  # trigger increment next press
            st.rerun()

        if st.session_state.sqc_quiz_total >= 5:
            grade = (
                "🏆 Excellent!" if pct >= 90 else
                "✅ Good work" if pct >= 70 else
                "📖 Review the formula reference tab"
            )
            display_callout(f"{grade} — {pct:.0f}% accuracy over "
                            f"{st.session_state.sqc_quiz_total} questions.", "success")


# ============================================================
# MODULE 36: PRACTICE PROBLEMS (General) - ENHANCED V6.0
# ============================================================
def module_practice():
    display_header("🎓", "Exam Prep", "Comprehensive Practice Problems",
                   "Mixed review across all Operations Management chapters")

    tab1, tab2, tab3 = st.tabs([
        "📝 Problem Bank", "🔬 Quick Calculators", "📊 Formula Cheat Sheet"
    ])

    # ─────────────────────────────────────────────────────────
    with tab1:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            chapter_filter = st.selectbox("Filter by Chapter / Topic", [
                "All", "Quality (Ch 13)", "Inventory (Ch 20)",
                "MRP (Ch 21)", "Scheduling (Ch 22)",
                "Location & Capacity", "Project Mgmt (Ch 4)",
                "Forecasting (Ch 18)", "Queuing (Ch 10)"
            ])
        with col_f2:
            diff_filter = st.selectbox("Filter by Difficulty",
                                       ["All", "Easy", "Medium", "Hard"], key="prac_diff")
        with col_f3:
            sort_opt = st.selectbox("Sort By", ["Default", "Difficulty ↑", "Difficulty ↓"])

        problems = [
            # (chapter, title, difficulty, question, answer)
            ("Quality (Ch 13)", "DPMO Calculation", "Medium",
             "2,000 units produced, 33 total defects, 5 opportunities per unit. Calculate DPMO and approximate sigma level.",
             "DPMO = (Defects / (Units × Opp)) × 10⁶ = (33 / 10,000) × 1,000,000 = **3,300 DPMO**<br><br>"
             "Sigma level ≈ 0.8406 + √(29.37 − 2.221 × ln(3300)) = 0.8406 + √(29.37 − 17.98) = **≈ 4.22σ**"),

            ("Quality (Ch 13)", "Cpk Interpretation", "Hard",
             "Process mean = 50, σ = 2, LSL = 44, USL = 58. Calculate Cp and Cpk. Interpret results.",
             "Cp = (58−44)/(6×2) = 14/12 = **1.17** — potentially capable<br>"
             "Cpl = (50−44)/(3×2) = **1.00**, Cpu = (58−50)/(3×2) = **1.33**<br>"
             "Cpk = min(1.00, 1.33) = **1.00** — marginally capable; mean is closer to LSL.<br><br>"
             "Action: Shift mean upward toward 51 to equalize Cpl and Cpu."),

            ("Quality (Ch 13)", "p-Chart Decision", "Medium",
             "15 samples, n=200 each. Total defectives = 120. Compute p̄, σ_p, UCL, LCL. "
             "Sample 8 has 19 defectives — is it out of control?",
             "p̄ = 120 / (15×200) = **0.040**<br>"
             "σ_p = √(0.040 × 0.960 / 200) = **0.01386**<br>"
             "UCL = 0.040 + 3(0.01386) = **0.0816**; LCL = max(0, 0.040 − 0.0416) = **0**<br><br>"
             "Sample 8: p = 19/200 = 0.095 > UCL = 0.082 → **⚠️ Out of Control!**"),

            ("Inventory (Ch 20)", "EOQ & Annual Cost", "Medium",
             "D = 10,000 units/yr, S = $50/order, H = $2/unit/yr. Find Q*, annual orders, and total inventory cost.",
             "Q* = √(2DS/H) = √(2×10,000×50/2) = √500,000 = **707.1 units**<br>"
             "Annual orders = D/Q* = 10,000/707 = **14.1 orders/yr**<br>"
             "TC = (Q*/2)×H + (D/Q*)×S = 707 + 707 = **$1,414/yr**<br><br>"
             "Note: At EOQ, holding cost = ordering cost — this is a useful check."),

            ("Inventory (Ch 20)", "Safety Stock + ROP", "Medium",
             "d̄ = 40 units/day, σ_d = 8, LT = 9 days, service level = 95% (z=1.65). Find SS, ROP, and cycle stock.",
             "SS = z × σ_d × √LT = 1.65 × 8 × √9 = 1.65 × 8 × 3 = **39.6 ≈ 40 units**<br>"
             "ROP = d̄ × LT + SS = 40×9 + 40 = 360 + 40 = **400 units**<br>"
             "Average cycle stock = Q*/2 (depends on order quantity policy)"),

            ("Inventory (Ch 20)", "Newsvendor Model", "Hard",
             "Price = $120, Cost = $80, Salvage = $30, μ = 500, σ = 60. Find optimal order quantity Q*.",
             "Cu (underage cost) = P − C = 120−80 = **$40**<br>"
             "Co (overage cost)  = C − S = 80−30 = **$50**<br>"
             "CR = Cu/(Cu+Co) = 40/90 = **0.444**<br>"
             "z = Φ⁻¹(0.444) ≈ **−0.14**<br>"
             "Q* = μ + z×σ = 500 + (−0.14)×60 = **492 units**<br><br>"
             "Interpretation: Since CR < 0.5, overages cost more — order below the mean."),

            ("Inventory (Ch 20)", "ABC Classification", "Easy",
             "5 items with annual usage × unit cost: A=$45k, B=$3k, C=$38k, D=$1.5k, E=$12k. "
             "Classify using ABC analysis.",
             "Total = $99.5k. Sort descending: A($45k=45%), C($38k=38%), E($12k=12%), B($3k=3%), D($1.5k=2%)<br><br>"
             "**Class A (≈80%):** Items A + C = $83k (83%) → tight control, frequent review<br>"
             "**Class B (≈15%):** Item E = $12k (12%) → moderate control<br>"
             "**Class C (≈5%):** Items B + D = $4.5k (5%) → minimal control, bulk ordering"),

            ("MRP (Ch 21)", "MRP Net Requirements", "Medium",
             "Gross req = 200, On-Hand = 50, Scheduled Receipt = 75, Safety Stock = 20. "
             "Compute net requirement and planned order.",
             "Available = OH + SR = 50 + 75 = 125<br>"
             "Net = max(0, Gross − Available + SS) = max(0, 200 − 125 + 20) = **95 units**<br>"
             "Planned order receipt = **95 units** (or round up to lot size if applicable)."),

            ("MRP (Ch 21)", "POQ Period", "Medium",
             "S = $50, H = $1/unit/week, average demand = 25 units/week. Calculate T* (POQ interval).",
             "EOQ_weekly = √(2 × S × d / H) = √(2 × 50 × 25 / 1) = √2,500 = **50 units**<br>"
             "T* = EOQ/d = 50/25 = **2 weeks**<br><br>"
             "So order every 2 weeks, covering that period's net requirements."),

            ("MRP (Ch 21)", "L4L vs FOQ", "Hard",
             "Weekly demands: 30, 45, 20, 60, 25. S=$100, H=$2/unit/wk. "
             "Compare Lot-for-Lot vs Fixed Order Qty (FOQ=80) total cost.",
             "**L4L:** Order exactly what is needed each week — no carrying cost.<br>"
             "Setup cost = 5 orders × $100 = $500; Holding cost = $0; **TC = $500**<br><br>"
             "**FOQ=80:** Orders in wk1(80), wk2 carries 5 → needs 40 more in wk3 → wk4 needs 45 → wk5 order 25<br>"
             "Approx holding cost = (5+0+15+0)×$2 = $40; Setup = 5×$100 = $500; **TC ≈ $540**<br><br>"
             "L4L minimizes cost here — best when holding costs dominate setup costs."),

            ("Scheduling (Ch 22)", "SPT Sequence", "Medium",
             "Jobs: A(PT=5,DD=8), B(PT=2,DD=6), C(PT=4,DD=9), D(PT=1,DD=5). "
             "Find SPT sequence, avg flow time, and avg tardiness.",
             "SPT order: **D(1) → B(2) → C(4) → A(5)**<br>"
             "Finish times: 1, 3, 7, 12<br>"
             "Avg Flow Time = (1+3+7+12)/4 = **5.75 days**<br>"
             "Tardiness: D=max(0,1−5)=0, B=max(0,3−6)=0, C=max(0,7−9)=0, A=max(0,12−8)=4<br>"
             "Avg Tardiness = (0+0+0+4)/4 = **1.0 day**"),

            ("Scheduling (Ch 22)", "EDD vs SPT", "Medium",
             "Same jobs: A(5,8), B(2,6), C(4,9), D(1,5). Find EDD sequence and compare avg tardiness to SPT.",
             "EDD order (by due date): **D(DD=5) → B(DD=6) → A(DD=8) → C(DD=9)**<br>"
             "Finish times: 1, 3, 8, 12<br>"
             "Tardiness: D=0, B=0, A=max(0,8−8)=0, C=max(0,12−9)=3<br>"
             "Avg Tardiness = **0.75 days** (better than SPT's 1.0 for minimizing tardiness).<br><br>"
             "EDD minimizes max tardiness; SPT minimizes avg flow time."),

            ("Scheduling (Ch 22)", "Critical Ratio", "Easy",
             "Today = Day 5. Job X: PT=3, Due=Day 10. Job Y: PT=4, Due=Day 9. Rank by CR.",
             "CR_X = (10−5)/3 = **1.67** (ahead of schedule)<br>"
             "CR_Y = (9−5)/4 = **1.00** (exactly on schedule)<br><br>"
             "CR < 1.0 = behind schedule, CR = 1.0 = on schedule, CR > 1.0 = ahead.<br>"
             "Priority: **Job Y first** (lower CR = more urgent)."),

            ("Scheduling (Ch 22)", "Johnson's Rule", "Hard",
             "Two-machine scheduling: Job A(M1=3,M2=5), B(M1=6,M2=2), C(M1=1,M2=4), D(M1=5,M2=6). "
             "Apply Johnson's Rule and calculate makespan.",
             "Step 1 — Find minimum processing time across all jobs and machines:<br>"
             "Min = 1 (Job C, M1) → assign C first. Remaining: A(3,5), B(6,2), D(5,6)<br>"
             "Min = 2 (Job B, M2) → assign B last. Remaining: A(3,5), D(5,6)<br>"
             "Min = 3 (Job A, M1) → assign next. Min = 6 (Job D, either) → D next.<br>"
             "**Sequence: C → A → D → B**<br><br>"
             "Makespan calculation: M1 finish: C=1, A=4, D=9, B=15<br>"
             "M2 start/finish: C=[1,5], A=[5,10], D=[10,16], B=[16,18] → **Makespan = 18**"),

            ("Location & Capacity", "Break-Even Analysis", "Medium",
             "FC = $100,000, Selling Price = $50, VC = $30/unit. "
             "Calculate BEP in units, revenue, and degree of operating leverage at 6,000 units.",
             "CM = $50 − $30 = **$20/unit**<br>"
             "BEP(units) = 100,000 / 20 = **5,000 units**<br>"
             "BEP(revenue) = 5,000 × $50 = **$250,000**<br>"
             "At 6,000 units: Revenue=$300k, VC=$180k, CM=$120k, Profit=$20k<br>"
             "DOL = CM / Profit = $120k / $20k = **6.0×** (1% sales change → 6% profit change)"),

            ("Location & Capacity", "Learning Curve", "Hard",
             "First unit = 100 hrs, 80% learning curve. Find cumulative avg time and total time for 8 units.",
             "Using doubling rule: Y₁=100, Y₂=80, Y₄=64, Y₈=51.2 hrs (cumulative avg)<br>"
             "b = ln(0.8)/ln(2) = −0.2231/0.6931 = **−0.3219**<br>"
             "Y₈ = 100 × 8^(−0.3219) = 100 × 0.512 = **51.2 hrs**<br>"
             "Total hrs for 8 units = 51.2 × 8 = **409.6 hrs**<br>"
             "Unit 8 alone = 409.6 − (64 × 4) = 409.6 − 256 = **153.6 hrs** cumulative check ✅"),

            ("Location & Capacity", "Factor Rating", "Easy",
             "Site A: [75, 80, 70] on factors with weights [0.3, 0.5, 0.2]. "
             "Site B: [85, 70, 80]. Which is preferred?",
             "Score A = 0.3×75 + 0.5×80 + 0.2×70 = 22.5 + 40.0 + 14.0 = **76.5**<br>"
             "Score B = 0.3×85 + 0.5×70 + 0.2×80 = 25.5 + 35.0 + 16.0 = **76.5**<br><br>"
             "**Tie!** Examine qualitative factors or re-weight. Small weight changes can break the tie."),

            ("Location & Capacity", "Indifference Point", "Medium",
             "Option A: FC=$80k, VC=$20/unit. Option B: FC=$120k, VC=$14/unit. "
             "Find indifference volume. Which is better at 5,000 units?",
             "Indifference: 80,000 + 20Q = 120,000 + 14Q → 6Q = 40,000 → **Q = 6,667 units**<br><br>"
             "At 5,000 units:<br>"
             "TC_A = 80,000 + 20×5,000 = $180,000<br>"
             "TC_B = 120,000 + 14×5,000 = $190,000<br>"
             "**Choose Option A** at 5,000 units (below indifference point)."),

            ("Project Mgmt (Ch 4)", "PERT Expected Time", "Easy",
             "Activity E: optimistic=3, most likely=5, pessimistic=9 weeks. "
             "Find expected time and variance.",
             "t_e = (a + 4m + b) / 6 = (3 + 4×5 + 9) / 6 = (3 + 20 + 9) / 6 = **32/6 = 5.33 weeks**<br>"
             "σ² = ((b − a) / 6)² = ((9 − 3) / 6)² = 1² = **1.0 week²**<br>"
             "σ = **1.0 week**"),

            ("Project Mgmt (Ch 4)", "Project Completion Probability", "Hard",
             "Critical path expected duration = 42 weeks, σ²_path = 9 weeks². "
             "What is P(project completes ≤ 45 weeks)?",
             "σ_path = √9 = **3 weeks**<br>"
             "Z = (45 − 42) / 3 = **1.0**<br>"
             "P(Z ≤ 1.0) = Φ(1.0) = **0.8413 = 84.13%**<br><br>"
             "There is an 84.13% probability of finishing by week 45."),

            ("Forecasting (Ch 18)", "Exponential Smoothing", "Medium",
             "Last period forecast = 200, actual demand = 220, α = 0.3. "
             "Find new forecast. What if α = 0.7?",
             "F_new = α × A_last + (1−α) × F_last<br>"
             "α=0.3: F = 0.3×220 + 0.7×200 = 66 + 140 = **206**<br>"
             "α=0.7: F = 0.7×220 + 0.3×200 = 154 + 60 = **214**<br><br>"
             "Higher α reacts faster to recent demand — better for volatile series."),

            ("Forecasting (Ch 18)", "MAD and Bias", "Medium",
             "Actual: [100,110,105,115]. Forecast: [102,108,110,112]. Calculate MAD, MSE, and Bias.",
             "Errors: −2, 2, −5, 3<br>"
             "MAD = (|−2|+|2|+|−5|+|3|) / 4 = 12/4 = **3.0**<br>"
             "MSE = (4+4+25+9) / 4 = 42/4 = **10.5**<br>"
             "Bias = (−2+2−5+3) / 4 = −2/4 = **−0.5** (slight consistent under-forecast)"),

            ("Queuing (Ch 10)", "M/M/1 Queue", "Medium",
             "Arrival rate λ = 4/hr, service rate μ = 6/hr. Find ρ, Lq, Wq, L, W.",
             "ρ = λ/μ = 4/6 = **0.667** (utilization)<br>"
             "Lq = ρ²/(1−ρ) = 0.444/0.333 = **1.333 customers in queue**<br>"
             "Wq = Lq/λ = 1.333/4 = **0.333 hrs = 20 min wait**<br>"
             "L = ρ/(1−ρ) = **2.0 customers in system**<br>"
             "W = L/λ = 2.0/4 = **0.5 hrs = 30 min in system**"),

            ("Queuing (Ch 10)", "Queue Capacity Decision", "Hard",
             "Current M/M/1: λ=8/hr, μ=10/hr. Adding a 2nd server raises effective μ to 20/hr combined. "
             "Should you add the server if hourly cost = $30 and customer wait cost = $50/hr?",
             "**Current (1 server):** ρ=0.8, Lq=3.2, Wq=0.4 hr<br>"
             "Wait cost = 8 customers/hr × 0.4 hr × $50 = **$160/hr total wait cost**<br><br>"
             "**2 servers (M/M/2 approximation):** ρ=0.4 per server, Lq≈0.152, Wq≈0.019 hr<br>"
             "Wait cost ≈ 8 × 0.019 × $50 = **$7.60/hr**<br>"
             "Added server cost = **$30/hr**<br>"
             "Net savings = $160 − $7.60 − $30 = **$122.40/hr → Add the server ✅**"),
        ]

        # ── Filter & sort ─────────────────────────────────────
        filtered_p = problems
        if chapter_filter != "All":
            filtered_p = [p for p in filtered_p if p[0] == chapter_filter]
        if diff_filter != "All":
            filtered_p = [p for p in filtered_p if p[2] == diff_filter]

        diff_order = {"Easy": 0, "Medium": 1, "Hard": 2}
        if sort_opt == "Difficulty ↑":
            filtered_p = sorted(filtered_p, key=lambda x: diff_order[x[2]])
        elif sort_opt == "Difficulty ↓":
            filtered_p = sorted(filtered_p, key=lambda x: -diff_order[x[2]])

        # ── Stats row ─────────────────────────────────────────
        easy_n  = sum(1 for p in problems if p[2] == "Easy")
        med_n   = sum(1 for p in problems if p[2] == "Medium")
        hard_n  = sum(1 for p in problems if p[2] == "Hard")
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Total Problems", len(problems))
        s2.metric("🟢 Easy",   easy_n)
        s3.metric("🟡 Medium", med_n)
        s4.metric("🔴 Hard",   hard_n)

        if not filtered_p:
            display_callout("No problems match the current filters.", "warning")
        else:
            st.markdown(f"**Showing {len(filtered_p)} of {len(problems)} problems.**")
            diff_icons = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            for i, (chapter, title, diff, question, answer) in enumerate(filtered_p):
                with st.expander(
                    f"{diff_icons[diff]} **{title}** — *{chapter}*"
                ):
                    display_practice_problem(i + 1, diff,
                                             f"**{title}:** {question}")
                    key_hint = {
                        "EOQ & Annual Cost":            "Hint: Q* = √(2DS/H). At EOQ, holding = ordering cost.",
                        "Safety Stock + ROP":           "Hint: SS = z × σ_d × √LT, ROP = d̄×LT + SS.",
                        "Newsvendor Model":             "Hint: CR = Cu/(Cu+Co), then z = Φ⁻¹(CR).",
                        "Project Completion Probability": "Hint: Z = (T_D − T_E) / σ_path.",
                        "M/M/1 Queue":                  "Hint: Lq = ρ²/(1−ρ), then use Little's Law.",
                        "Johnson's Rule":               "Hint: Assign smallest time first; if M1 → front, if M2 → back.",
                    }.get(title)
                    if key_hint:
                        display_hint(key_hint)
                    if st.button("Show Solution ▼",
                                 key=f"prac_{i}_{chapter[:4]}_{diff}",
                                 use_container_width=True):
                        st.session_state[f"prac_ans_{i}"] = True
                        st.session_state.problems_solved += 1
                    if st.session_state.get(f"prac_ans_{i}"):
                        display_solution(answer)

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🔬 Quick Formula Calculators")

        calc_choice = st.selectbox("Choose Calculator", [
            "EOQ",
            "Break-Even",
            "Learning Curve",
            "DPMO",
            "Safety Stock & ROP",
            "Newsvendor",
            "PERT",
        ])

        st.markdown("---")

        if calc_choice == "EOQ":
            c1, c2 = st.columns(2)
            with c1:
                D_c = st.number_input("Annual Demand D", value=10000, min_value=1)
                S_c = st.number_input("Setup/Order Cost S ($)", value=50.0, min_value=0.01)
                H_c = st.number_input("Holding Cost H ($/unit/yr)", value=2.0, min_value=0.01)
            with c2:
                Q_c  = math.sqrt(2 * D_c * S_c / H_c)
                TC_c = (Q_c / 2) * H_c + (D_c / Q_c) * S_c
                display_metric_card(f"{Q_c:.1f} units", "EOQ (Q*)", "highlight")
                display_metric_card(f"{D_c/Q_c:.1f} orders/yr", "Annual Orders", "normal")
                display_metric_card(f"${TC_c:.2f}", "Min Annual TC", "success")
                display_metric_card(f"${(D_c/Q_c)*S_c:.2f}", "Annual Order Cost", "normal")
                display_metric_card(f"${(Q_c/2)*H_c:.2f}", "Annual Holding Cost", "normal")
            st.latex(rf"Q^* = \sqrt{{\frac{{2 \times {D_c:,.0f} \times {S_c}}}{{{H_c}}}}} = {Q_c:.1f} \text{{ units}}")
            display_callout("At EOQ, annual ordering cost = annual holding cost — this is a useful self-check.",
                            "info", "Tip")

        elif calc_choice == "Break-Even":
            c1, c2 = st.columns(2)
            with c1:
                FC_c = st.number_input("Fixed Costs ($)", value=100000, min_value=0)
                P_c  = st.number_input("Unit Price ($)", value=50.0, min_value=0.01)
                VC_c = st.number_input("Variable Cost/unit ($)", value=30.0, min_value=0.0)
                vol  = st.number_input("Volume to evaluate (units)", value=6000, min_value=0)
            with c2:
                margin = P_c - VC_c
                if margin > 0:
                    bep_u = FC_c / margin
                    bep_r = bep_u * P_c
                    profit_at_vol = (P_c - VC_c) * vol - FC_c
                    display_metric_card(f"{bep_u:,.0f} units", "BEP (units)", "highlight")
                    display_metric_card(f"${bep_r:,.0f}", "BEP (revenue)", "normal")
                    display_metric_card(f"${margin:.2f}/unit", "Contribution Margin", "normal")
                    display_metric_card(
                        f"${profit_at_vol:,.0f}",
                        f"Profit at {vol:,} units",
                        "success" if profit_at_vol >= 0 else "danger"
                    )
                else:
                    st.error("Price must exceed variable cost.")

        elif calc_choice == "Learning Curve":
            c1, c2 = st.columns(2)
            with c1:
                t1   = st.number_input("Time for Unit 1 (hrs)", value=100.0, min_value=0.1)
                lc   = st.selectbox("Learning Rate", ["70%", "75%", "80%", "85%", "90%"])
                lc_v = float(lc.strip("%")) / 100
                n_u  = st.number_input("Unit Number N", value=8, min_value=1)
            with c2:
                b    = math.log(lc_v) / math.log(2)
                yn   = t1 * (n_u ** b)
                display_metric_card(f"{yn:.2f} hrs", f"Cum Avg Time at Unit {n_u}", "highlight")
                display_metric_card(f"{yn * n_u:.2f} hrs", f"Total Time (1 to {n_u})", "normal")
                display_metric_card(f"{b:.4f}", "Learning Index (b)", "normal")
                if n_u > 1:
                    yn_prev  = t1 * ((n_u - 1) ** b) * (n_u - 1)
                    unit_n_t = yn * n_u - yn_prev
                    display_metric_card(f"{unit_n_t:.2f} hrs", f"Unit {n_u} alone", "normal")
            st.latex(rf"Y_{{{n_u}}} = {t1} \times {n_u}^{{{b:.3f}}} = {yn:.2f} \text{{ hrs}}")

        elif calc_choice == "DPMO":
            c1, c2 = st.columns(2)
            with c1:
                defects_i = st.number_input("Total Defects", value=33, min_value=0)
                units_i   = st.number_input("Total Units", value=2000, min_value=1)
                opp_i     = st.number_input("Opportunities per Unit", value=5, min_value=1)
            with c2:
                dpmo = (defects_i / (units_i * opp_i)) * 1_000_000 if units_i * opp_i > 0 else 0
                if 0 < dpmo < 1_000_000:
                    sl = 0.8406 + math.sqrt(max(0, 29.37 - 2.221 * math.log(dpmo)))
                    yld = (1 - defects_i / (units_i * opp_i)) * 100
                else:
                    sl, yld = 0.0, 100.0
                display_metric_card(f"{dpmo:,.1f}", "DPMO", "danger" if dpmo > 6210 else "success")
                display_metric_card(f"{sl:.2f}σ",   "Approx Sigma Level", "success" if sl >= 4 else "warning")
                display_metric_card(f"{yld:.4f}%",  "Process Yield", "normal")
            st.latex(rf"DPMO = \frac{{{defects_i}}}{{{units_i} \times {opp_i}}} \times 10^6 = {dpmo:,.1f}")

        elif calc_choice == "Safety Stock & ROP":
            c1, c2 = st.columns(2)
            with c1:
                d_bar    = st.number_input("Avg Daily Demand (d̄)", value=40.0, min_value=0.0)
                sigma_d  = st.number_input("Std Dev of Daily Demand (σ_d)", value=8.0, min_value=0.0)
                lt       = st.number_input("Lead Time (days)", value=9, min_value=1)
                sl_pct   = st.selectbox("Service Level",
                                        ["90% (z=1.28)", "95% (z=1.65)", "98% (z=2.05)", "99% (z=2.33)"])
                z_sl     = {"90% (z=1.28)": 1.28, "95% (z=1.65)": 1.65,
                            "98% (z=2.05)": 2.05, "99% (z=2.33)": 2.33}[sl_pct]
            with c2:
                ss  = safety_stock_units(z_sl, sigma_d, lt)
                rop = reorder_point(d_bar, lt, ss)
                display_metric_card(f"{ss:.1f} units",  "Safety Stock (SS)", "highlight")
                display_metric_card(f"{rop:.1f} units", "Reorder Point (ROP)", "normal")
                display_metric_card(f"{d_bar * lt:.1f} units", "Avg Demand over LT", "normal")
            st.latex(
                rf"SS = {z_sl} \times {sigma_d} \times \sqrt{{{lt}}} = {ss:.1f}, "
                rf"\quad ROP = {d_bar} \times {lt} + {ss:.1f} = {rop:.1f}"
            )

        elif calc_choice == "Newsvendor":
            c1, c2 = st.columns(2)
            with c1:
                price_nv = st.number_input("Selling Price ($)", value=120.0)
                cost_nv  = st.number_input("Unit Cost ($)", value=80.0)
                salv_nv  = st.number_input("Salvage Value ($)", value=30.0)
                mu_nv    = st.number_input("Mean Demand (μ)", value=500.0)
                sig_nv   = st.number_input("Std Dev Demand (σ)", value=60.0)
            with c2:
                cu_nv  = price_nv - cost_nv
                co_nv  = cost_nv - salv_nv
                cr_nv  = cu_nv / (cu_nv + co_nv) if (cu_nv + co_nv) > 0 else 0
                z_nv   = normal_ppf(cr_nv)
                q_star = mu_nv + z_nv * sig_nv
                display_metric_card(f"${cu_nv:.2f}", "Underage Cost (Cu)", "normal")
                display_metric_card(f"${co_nv:.2f}", "Overage Cost (Co)",  "normal")
                display_metric_card(f"{cr_nv:.4f}",  "Critical Ratio",     "normal")
                display_metric_card(f"{z_nv:.3f}",   "Z-value",            "normal")
                display_metric_card(f"{q_star:.0f} units", "Q* (Optimal Order)", "highlight")
            st.latex(
                rf"Q^* = \mu + z \cdot \sigma = {mu_nv:.0f} + ({z_nv:.3f}) \times {sig_nv:.0f} = {q_star:.0f}"
            )

        elif calc_choice == "PERT":
            c1, c2 = st.columns(2)
            with c1:
                a_pert = st.number_input("Optimistic (a)", value=3.0)
                m_pert = st.number_input("Most Likely (m)", value=5.0)
                b_pert = st.number_input("Pessimistic (b)", value=9.0)
                deadline = st.number_input("Project Deadline (weeks)", value=45.0)
                path_var = st.number_input("Sum of Path Variances (Σσ²)", value=9.0,
                                           help="Sum of variances of critical path activities")
            with c2:
                te  = pert_te(a_pert, m_pert, b_pert)
                var = pert_variance(a_pert, b_pert)
                sig = pert_sigma(a_pert, b_pert)
                display_metric_card(f"{te:.2f} weeks", "Expected Time (t_e)", "highlight")
                display_metric_card(f"{var:.4f} wk²",  "Activity Variance (σ²)", "normal")
                display_metric_card(f"{sig:.4f} wks",  "Activity Std Dev (σ)", "normal")
                if path_var > 0:
                    path_sig = math.sqrt(path_var)
                    z_dead   = (deadline - te) / path_sig
                    prob     = normal_cdf(z_dead)
                    display_metric_card(f"{prob:.2%}", f"P(done by wk {deadline:.0f})", "success")
            st.latex(
                rf"t_e = \frac{{a+4m+b}}{{6}} = \frac{{{a_pert:.0f}+4({m_pert:.0f})+{b_pert:.0f}}}{{6}} = {te:.2f}"
            )

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📊 Formula Cheat Sheet")

        cheat_cols = st.columns(2)
        sections = [
            ("📦 Inventory", [
                ("EOQ",           r"Q^* = \sqrt{\frac{2DS}{H}}"),
                ("TC at EOQ",     r"TC = \frac{Q}{2}H + \frac{D}{Q}S"),
                ("ROP",           r"ROP = \bar{d} \cdot LT + SS"),
                ("Safety Stock",  r"SS = z \cdot \sigma_d \cdot \sqrt{LT}"),
                ("Newsvendor CR", r"CR = \frac{C_u}{C_u + C_o}"),
            ]),
            ("📈 Forecasting", [
                ("Exp. Smoothing", r"F_{t+1} = \alpha A_t + (1-\alpha)F_t"),
                ("MAD",            r"MAD = \frac{\sum|A_t - F_t|}{n}"),
                ("MSE",            r"MSE = \frac{\sum(A_t-F_t)^2}{n}"),
                ("MAPE",           r"MAPE = \frac{1}{n}\sum\left|\frac{A_t-F_t}{A_t}\right|\times100"),
            ]),
            ("🔗 Project", [
                ("PERT t_e",       r"t_e = \frac{a+4m+b}{6}"),
                ("PERT σ²",        r"\sigma^2 = \left(\frac{b-a}{6}\right)^2"),
                ("P(complete)",    r"Z = \frac{T_D - \mu_{path}}{\sigma_{path}}"),
                ("Crash $/day",    r"\frac{CC - NC}{NT - CT}"),
            ]),
            ("👥 Queuing", [
                ("Utilization",    r"\rho = \frac{\lambda}{\mu}"),
                ("Lq (M/M/1)",     r"L_q = \frac{\rho^2}{1-\rho}"),
                ("Little's Law",   r"L = \lambda W"),
                ("Wq",             r"W_q = \frac{L_q}{\lambda}"),
            ]),
            ("📉 Learning Curve", [
                ("Cum Avg",        r"Y_n = Y_1 \cdot n^b"),
                ("Learning Index", r"b = \frac{\ln(r)}{\ln(2)}"),
            ]),
            ("⚖️ Cost / Break-Even", [
                ("BEP Units",      r"BEP = \frac{FC}{P - VC}"),
                ("Indiff. Point",  r"Q^* = \frac{FC_2 - FC_1}{VC_1 - VC_2}"),
                ("EMV",            r"EMV = P \times \text{Impact}"),
            ]),
        ]

        for i, (section_title, formulas) in enumerate(sections):
            with cheat_cols[i % 2]:
                st.markdown(f"#### {section_title}")
                for name, formula in formulas:
                    display_formula_card(name, formula)

# ============================================================
# MODULE REGISTRY
# ============================================================
# Tuple schema:
#   (key, display_name, icon, chapter_tag, chapter_label, is_new, tags, est_min)
#   tags     – list of searchable keywords beyond name/key/chapter
#   est_min  – estimated completion time in minutes (0 = quick reference)
MODULE_REGISTRY = [
    # key              name                      icon  ch_tag  chapter_label                         new   tags                                          min
    ("risk",          "SC Risk Assessment",      "🛡️", "ch1",  "Ch 1 · Strategy & Risk",             False, ["risk","supply chain","strategy","swot"],     8),
    ("pert",          "PERT Network",            "🔗", "ch4",  "Ch 4 · Project Management",          False, ["pert","network","critical path","cpm","time"], 12),
    ("crashing",      "Project Crashing",        "⚡", "ch4",  "Ch 4 · Project Management",          False, ["crashing","cost","project","compress"],        10),
    ("breakeven",     "Break-Even Analysis",     "📈", "ch5",  "Ch 5 · Capacity Planning",           False, ["breakeven","fixed cost","contribution"],       8),
    ("decision",      "Decision Trees",          "🌳", "ch5",  "Ch 5 · Capacity Planning",           False, ["decision tree","emv","expected value","risk"],  10),
    ("learning",      "Learning Curves",         "📉", "ch6",  "Ch 6 · Learning Curves",             False, ["learning curve","improvement","wright"],        8),
    ("decoupling",    "Decoupling Point",        "🔀", "ch7",  "Ch 7 · Manufacturing",               False, ["decoupling","push pull","inventory","flow"],    7),
    ("linebalance",   "Line Balancing",          "⚖️", "ch8",  "Ch 8 · Layout",                      False, ["line balancing","takt","cycle time","layout"],  10),
    ("service",       "Service Design",          "🎯", "ch9",  "Ch 9 · Service Design",              False, ["service","blueprint","design","customer"],      8),
    ("pokayoke",      "Poka-yoke DB",            "🛡️", "ch9",  "Ch 9 · Service Design",              True,  ["poka yoke","mistake proof","error"],           5),
    ("queuing",       "Queuing Theory",          "👥", "ch10", "Ch 10 · Queuing",                    False, ["queue","waiting","mm1","server","utilization"], 12),
    ("distributions", "Distributions",          "📐", "ch10", "Ch 10 · Queuing",                    False, ["normal","poisson","binomial","distribution"],   8),
    ("littles",       "Little's Law",            "🔄", "ch11", "Ch 11 · Process Analysis",           False, ["little","throughput","wip","flow time"],        6),
    ("dpmo",          "DPMO & DMAIC",            "🎯", "ch12", "Ch 12 · Six Sigma",                  False, ["dpmo","dmaic","defects","six sigma","sigma"],   10),
    ("fmea",          "FMEA Risk",               "⚠️", "ch12", "Ch 12 · Six Sigma",                  False, ["fmea","failure mode","rpn","severity"],         10),
    ("sqc",           "p & c Charts",            "📉", "ch13", "Ch 13 · Quality Control",            False, ["p chart","c chart","control","attribute"],      12),
    ("capability",    "Process Capability",      "🎯", "ch13", "Ch 13 · Quality Control",            False, ["cpk","cp","capability","sigma","process"],      8),
    ("sampling",      "Acceptance Sampling",     "📊", "ch13", "Ch 13 · Quality Control",            False, ["sampling","aql","oc curve","lot"],              8),
    ("pareto",        "Pareto Analysis",         "📊", "ch13", "Ch 13 · Quality Control",            False, ["pareto","80/20","defect","priority"],           6),
    ("fishbone",      "Fishbone Diagram",        "🐟", "ch13", "Ch 13 · Quality Control",            False, ["fishbone","ishikawa","cause effect","root"],    6),
    ("sqc_practice",  "SQC Practice",            "🎓", "ch13", "Ch 13 · Quality Control",            True,  ["practice","quiz","sqc","control chart"],        15),
    ("lean",          "Lean Supply Chains",      "🔄", "ch14", "Ch 14 · Lean",                       False, ["lean","waste","kanban","jit","5s","value"],     10),
    ("centroid",      "Centroid Method",         "📍", "ch15", "Ch 15 · Logistics",                  False, ["centroid","location","facility","coordinates"], 8),
    ("factor",        "Factor Rating",           "⚖️", "ch15", "Ch 15 · Logistics",                  False, ["factor rating","location","weight","score"],    8),
    ("transportation","Transportation",          "🚚", "ch15", "Ch 15 · Logistics",                  False, ["transportation","shipping","lp","network"],     10),
    ("sourcing",      "Global Sourcing",         "🌐", "ch16", "Ch 16 · Sourcing",                   False, ["sourcing","global","supplier","tco","cost"],    8),
    ("forecast",      "Enhanced Forecast",       "📈", "ch18", "Ch 18 · Forecasting",                False, ["forecast","moving average","exponential","mad"],12),
    ("regression",    "Regression+",             "📈", "ch18", "Ch 18 · Forecasting",                False, ["regression","trend","r squared","linear"],     10),
    ("aggregate",     "Aggregate Planning",      "📋", "ch19", "Ch 19 · Aggregate Planning",         False, ["aggregate","chase","level","workforce"],        10),
    ("eoq",           "EOQ Model",               "📦", "ch20", "Ch 20 · Inventory",                  False, ["eoq","order quantity","holding","setup cost"],  10),
    ("safetystock",   "Safety Stock",            "🛡️", "ch20", "Ch 20 · Inventory",                  False, ["safety stock","reorder","lead time","service"], 8),
    ("newsvendor",    "Newsvendor Model",        "📰", "ch20", "Ch 20 · Inventory",                  False, ["newsvendor","single period","overage","cu"],    8),
    ("mrp",           "MRP Matrix",              "🏭", "ch21", "Ch 21 · MRP",                        False, ["mrp","material requirements","bom","lot size"], 12),
    ("mrp_lotsizing", "MRP Lot Sizing",          "📦", "ch21", "Ch 21 · MRP",                        True,  ["lot sizing","eoq","mrp","l4l","pod"],           10),
    ("scheduling",    "Job Scheduling",          "📅", "ch22", "Ch 22 · Scheduling",                 False, ["scheduling","spт","johnson","makespan","jobs"], 10),
    ("practice",      "Practice Problems",       "🎓", "exam", "Exam Prep",                          False, ["practice","exam","quiz","review","all topics"], 20),
]

# ── Fast lookup structures ────────────────────────────────
_KEY_TO_META    = {r[0]: r for r in MODULE_REGISTRY}
_CHAPTER_ORDER  = list(dict.fromkeys(r[4] for r in MODULE_REGISTRY))
_ALL_TAGS       = sorted({tag for r in MODULE_REGISTRY for tag in r[6]})


def _get_chapter_modules() -> dict:
    """Return OrderedDict: chapter_label → list of (key, name, icon, is_new, est_min)."""
    chapters = {}
    for key, name, icon, _, ch_label, is_new, _, est_min in MODULE_REGISTRY:
        chapters.setdefault(ch_label, []).append((key, name, icon, is_new, est_min))
    return chapters


def _get_module_meta(key: str):
    """Return full registry tuple or None."""
    return _KEY_TO_META.get(key)


def _get_chapter_progress(visited: set) -> dict:
    """
    Returns {chapter_label: (visited_count, total_count)} for progress rings.
    """
    chapters = _get_chapter_modules()
    return {
        ch: (sum(1 for k, *_ in mods if k in visited), len(mods))
        for ch, mods in chapters.items()
    }


# ── Session state defaults ────────────────────────────────
def init_session_state():
    """Initialize all required session-state keys once."""
    defaults = {
        "dark_mode":        False,
        "problems_solved":  0,
        "correct_streak":   0,
        "best_streak":      0,
        "modules_visited":  set(),
        "recent_modules":   [],
        "bookmarks":        set(),
        "last_module":      None,
        "selected_module":  None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ============================================================
# WELCOME SCREEN
# ============================================================
def _render_welcome():
    """Full-page welcome shown when no module is selected."""
    p = _get_palette()

    # ── Hero banner ──────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: {p['metric_grad_hi']};
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(99,102,241,0.35);
    ">
        <div style="font-size:3.5rem;margin-bottom:0.5rem;">📊</div>
        <h1 style="margin:0;font-size:2rem;font-weight:800;color:white;">
            OSCM Interactive Simulator
        </h1>
        <p style="margin:0.5rem 0 0;opacity:0.9;font-size:1rem;">
            Based on Jacobs &amp; Chase — <em>Operations and Supply Chain Management</em>, 17th ed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats row ────────────────────────────────────────────
    total_mods     = len(MODULE_REGISTRY)
    new_mods       = sum(1 for r in MODULE_REGISTRY if r[5])
    total_chapters = len(_CHAPTER_ORDER)
    total_mins     = sum(r[7] for r in MODULE_REGISTRY)
    visited_count  = len(st.session_state.modules_visited)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: display_metric_card(str(total_mods),         "Total Modules",       "highlight")
    with c2: display_metric_card("75+",                   "Formulas",            "normal")
    with c3: display_metric_card(str(total_chapters),     "Chapters Covered",    "normal")
    with c4: display_metric_card(str(new_mods),           "New This Version",    "success")
    with c5: display_metric_card(f"~{total_mins//60}h",   "Est. Study Time",     "normal")

    # ── Personal progress (only show if user has started) ────
    if visited_count > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        pct = visited_count / total_mods
        solved  = st.session_state.problems_solved
        streak  = st.session_state.best_streak
        st.markdown("### 📈 Your Progress")
        pc1, pc2, pc3 = st.columns(3)
        with pc1: display_metric_card(f"{visited_count}/{total_mods}", "Modules Explored", "normal")
        with pc2: display_metric_card(str(solved),  "Problems Solved",  "normal" if solved < 10 else "success")
        with pc3: display_metric_card(f"🔥 {streak}", "Best Streak",    "warning" if streak >= 3 else "normal")
        display_progress_bar(visited_count, total_mods, label="Overall completion", bar_type="accent")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick-jump popular modules ────────────────────────────
    st.markdown("### ⚡ Quick Start")
    popular_keys = ["eoq", "pert", "queuing", "sqc", "forecast", "breakeven"]
    qcols = st.columns(len(popular_keys))
    for i, key in enumerate(popular_keys):
        meta = _get_module_meta(key)
        if not meta:
            continue
        _, name, icon, _, ch_label, _, _, est = meta
        with qcols[i]:
            st.markdown(f"""
            <div style="
                background:{p['bg_secondary']};
                border:1px solid {p['border']};
                border-radius:10px;
                padding:0.7rem 0.5rem;
                text-align:center;
                font-size:0.82rem;
                color:{p['text_primary']};
            ">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-weight:700;margin-top:0.3rem;">{name}</div>
                <div style="font-size:0.72rem;color:{p['text_muted']};margin-top:0.15rem;">
                    ~{est} min
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open", key=f"welcome_quick_{key}", use_container_width=True):
                st.session_state.selected_module = key
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature highlights ────────────────────────────────────
    st.markdown("### 🚀 What's Inside")
    features = [
        ("🔬", "Interactive Simulators",
         "Adjust parameters in real time and see results update instantly."),
        ("📐", "Step-by-Step Solutions",
         "Hints, auto-grading, and fully worked solutions for every problem."),
        ("📊", "Plotly Visualizations",
         "Dynamic charts — control charts, Gantt bars, heat maps, and more."),
        ("🎓", "Practice Problems",
         "Easy / Medium / Hard problems with tolerance-based auto-grading."),
        ("🌙", "Dark / Light Theme",
         "Toggle at any time — charts and cards adapt automatically."),
        ("📖", "Textbook Citations",
         "Anchored to Jacobs & Chase (2024) with page references."),
    ]
    feat_cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with feat_cols[i % 3]:
            display_concept_card(icon, title, desc)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Module map ────────────────────────────────────────────
    st.markdown("### 📚 Full Module Map")
    visited    = st.session_state.modules_visited
    ch_prog    = _get_chapter_progress(visited)
    chapters   = _get_chapter_modules()
    ch_cols    = st.columns(3)
    new_badge  = (
        "&nbsp;<span style='font-size:0.7rem;background:#16a34a;color:white;"
        "border-radius:3px;padding:0 4px;'>NEW</span>"
    )

    for idx, (ch_label, mods) in enumerate(chapters.items()):
        v_count, t_count = ch_prog.get(ch_label, (0, len(mods)))
        prog_pct = v_count / t_count if t_count else 0

        with ch_cols[idx % 3]:
            items_html = "".join(
                f"<li style='padding:0.15rem 0;"
                f"color:{'#6366f1' if k in visited else 'inherit'};'>"
                f"{'✓' if k in visited else icon} {name}"
                f"<span style='font-size:0.7rem;color:{p['text_muted']};'> ~{est}m</span>"
                f"{new_badge if is_new else ''}</li>"
                for k, name, icon, is_new, est in mods
            )
            progress_bar_html = (
                f"<div style='background:{p['border_strong']};border-radius:99px;"
                f"height:4px;margin-top:0.5rem;overflow:hidden;'>"
                f"<div style='width:{prog_pct*100:.0f}%;height:100%;"
                f"background:linear-gradient(90deg,#6366f1,#8b5cf6);"
                f"border-radius:99px;'></div></div>"
                f"<div style='font-size:0.68rem;color:{p['text_muted']};margin-top:0.2rem;'>"
                f"{v_count}/{t_count} visited</div>"
                if v_count > 0 else ""
            )
            st.markdown(f"""
            <div style="background:{p['bg_secondary']};border:1px solid {p['border']};
                        border-radius:10px;padding:0.9rem 1rem;margin:0.4rem 0;">
                <div style="font-weight:700;font-size:0.82rem;color:{p['accent']};
                            text-transform:uppercase;letter-spacing:0.04em;
                            margin-bottom:0.5rem;">{ch_label}</div>
                <ul style="margin:0;padding-left:1.1rem;list-style:none;
                           font-size:0.83rem;color:{p['text_primary']};">
                    {items_html}
                </ul>
                {progress_bar_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    display_callout(
        "Select any module from the sidebar or use the Quick Start cards above. "
        "Use the 🔍 search box to jump directly to any topic.",
        callout_type="tip",
        title="Getting Started",
    )


# ============================================================
# BREADCRUMB + MODULE TITLE BAR
# ============================================================
def _render_breadcrumb(key: str):
    """Slim breadcrumb bar with estimated time and bookmark button."""
    meta = _get_module_meta(key)
    if not meta:
        return
    _, name, icon, _, ch_label, is_new, _, est_min = meta
    p = _get_palette()

    bookmarks = st.session_state.get("bookmarks", set())
    is_bookmarked = key in bookmarks

    new_badge = (
        f'&nbsp;<span style="background:#16a34a;color:white;font-size:0.72rem;'
        f'border-radius:4px;padding:0.1rem 0.4rem;font-weight:700;">NEW</span>'
        if is_new else ""
    )
    est_html = (
        f'<span style="margin-left:auto;font-size:0.75rem;color:{p["text_muted"]};">'
        f'⏱ ~{est_min} min</span>'
        if est_min else ""
    )

    st.markdown(f"""
    <div style="
        display:flex; align-items:center; gap:0.5rem;
        padding:0.45rem 0.8rem;
        background:{p['bg_secondary']};
        border:1px solid {p['border']};
        border-radius:8px;
        margin-bottom:0.75rem;
        font-size:0.82rem;
        color:{p['text_secondary']};
    ">
        <span>📊 OSCM</span>
        <span style="color:{p['border_strong']};">›</span>
        <span>{ch_label}</span>
        <span style="color:{p['border_strong']};">›</span>
        <span style="color:{p['accent']};font-weight:700;">{icon} {name}{new_badge}</span>
        {est_html}
    </div>
    """, unsafe_allow_html=True)

    # Bookmark toggle rendered as a native button alongside breadcrumb
    bc1, bc2 = st.columns([10, 1])
    with bc2:
        bm_label = "⭐" if is_bookmarked else "☆"
        if st.button(bm_label, key=f"bm_{key}", help="Bookmark this module",
                     use_container_width=True):
            if is_bookmarked:
                st.session_state.bookmarks.discard(key)
            else:
                st.session_state.bookmarks.add(key)
            st.rerun()


# ============================================================
# SIDEBAR SEARCH
# ============================================================
def _render_search():
    """
    Search box with name, key, chapter, and tag matching.
    Returns key to jump to, or None.
    """
    query = st.sidebar.text_input(
        "🔍 Search modules",
        placeholder="e.g. EOQ, PERT, queuing, cpk…",
        key="module_search",
        label_visibility="collapsed",
    )
    if not query or len(query) < 2:
        return None

    q = query.lower().strip()
    matches = []
    for key, name, icon, _, ch_label, _, tags, _ in MODULE_REGISTRY:
        score = 0
        if q == key.lower():          score += 10   # exact key hit
        if q in name.lower():         score += 5    # name substring
        if any(q in t for t in tags): score += 3    # tag hit
        if q in ch_label.lower():     score += 1    # chapter hit
        if score:
            matches.append((score, key, name, icon))

    matches.sort(key=lambda x: -x[0])   # best matches first

    if not matches:
        st.sidebar.caption("No modules match your search.")
        return None

    st.sidebar.caption(f"{len(matches)} result(s):")
    for _, key, name, icon in matches[:6]:
        if st.sidebar.button(f"{icon} {name}", key=f"search_nav_{key}",
                             use_container_width=True):
            return key

    if len(matches) > 6:
        st.sidebar.caption(f"…and {len(matches)-6} more. Refine your search.")

    return None


# ============================================================
# BOOKMARKS PANEL
# ============================================================
def _render_bookmarks():
    """Render bookmarked modules in the sidebar."""
    bookmarks = st.session_state.get("bookmarks", set())
    if not bookmarks:
        return

    with st.sidebar.expander(f"⭐ Bookmarks ({len(bookmarks)})", expanded=False):
        for key in list(bookmarks):
            meta = _get_module_meta(key)
            if not meta:
                continue
            _, name, icon, *_ = meta
            if st.button(f"{icon} {name}", key=f"bm_nav_{key}",
                         use_container_width=True):
                st.session_state.selected_module = key
                st.rerun()


# ============================================================
# RECENTLY VISITED
# ============================================================
def _push_recent(key: str):
    """Add key to the front of recently-visited list (max 5, no duplicates)."""
    recent = st.session_state.get("recent_modules", [])
    recent = [k for k in recent if k != key]
    recent.insert(0, key)
    st.session_state.recent_modules = recent[:5]


def _render_recent():
    """Render Recently Visited in sidebar."""
    recent = st.session_state.get("recent_modules", [])
    if len(recent) < 2:
        return

    with st.sidebar.expander("🕐 Recently Visited", expanded=False):
        for key in recent:
            meta = _get_module_meta(key)
            if not meta:
                continue
            _, name, icon, *_ = meta
            if st.button(f"{icon} {name}", key=f"recent_{key}",
                         use_container_width=True):
                st.session_state.selected_module = key
                st.rerun()


# ============================================================
# SIDEBAR STATS PANEL
# ============================================================
def _render_stats_panel():
    """Dynamic stats card in the sidebar."""
    visited = len(st.session_state.get("modules_visited", set()))
    total   = len(MODULE_REGISTRY)
    solved  = st.session_state.get("problems_solved", 0)
    streak  = st.session_state.get("correct_streak", 0)
    best    = st.session_state.get("best_streak", 0)
    p       = _get_palette()

    pct_visited = visited / total
    streak_html = (
        f'<div style="margin-top:0.5rem;text-align:center;font-size:0.78rem;'
        f'color:#f59e0b;">🔥 {streak}-answer streak! (best: {best})</div>'
        if streak >= 3 else ""
    )

    st.sidebar.markdown(f"""
    <div style="
        background:{p['bg_secondary']};
        border:1px solid {p['border']};
        border-radius:10px;
        padding:0.8rem;
        margin:0.5rem 0;
    ">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                    gap:0.4rem;text-align:center;">
            <div>
                <div style="font-size:1.2rem;font-weight:800;color:{p['accent']};">
                    {visited}/{total}
                </div>
                <div style="font-size:0.68rem;color:{p['text_secondary']};">Modules</div>
            </div>
            <div>
                <div style="font-size:1.2rem;font-weight:800;color:{p['accent']};">
                    {solved}
                </div>
                <div style="font-size:0.68rem;color:{p['text_secondary']};">Solved</div>
            </div>
            <div>
                <div style="font-size:1.2rem;font-weight:800;
                    color:{'#f59e0b' if streak >= 3 else p['accent']};">
                    🔥{streak}
                </div>
                <div style="font-size:0.68rem;color:{p['text_secondary']};">Streak</div>
            </div>
        </div>
        <div style="margin-top:0.6rem;">
            <div style="font-size:0.7rem;color:{p['text_secondary']};margin-bottom:0.2rem;">
                Progress: {pct_visited:.0%}
            </div>
            <div style="background:{p['border_strong']};border-radius:99px;
                        height:6px;overflow:hidden;">
                <div style="
                    width:{pct_visited*100:.1f}%;height:100%;border-radius:99px;
                    background:linear-gradient(90deg,#6366f1,#8b5cf6);
                    transition:width 0.6s ease;
                "></div>
            </div>
        </div>
        {streak_html}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CHAPTER NAVIGATION
# ============================================================
def _render_chapter_nav():
    """
    Chapter-grouped navigation with per-chapter progress indicators.
    Returns the key of a clicked module, or None.
    """
    chapters   = _get_chapter_modules()
    current    = st.session_state.get("selected_module", "risk")
    visited    = st.session_state.get("modules_visited", set())
    bookmarks  = st.session_state.get("bookmarks", set())
    ch_prog    = _get_chapter_progress(visited)
    p          = _get_palette()
    jump_to    = None

    for ch_label, mods in chapters.items():
        is_active = any(k == current for k, *_ in mods)
        v_c, t_c  = ch_prog.get(ch_label, (0, len(mods)))

        # Chapter label with inline progress fraction
        ch_display = (
            f"{ch_label}  ✓{v_c}/{t_c}"
            if v_c > 0 else ch_label
        )

        with st.sidebar.expander(ch_display, expanded=is_active):
            for key, name, icon, is_new, est_min in mods:
                is_current    = (key == current)
                was_visited   = (key in visited)
                is_bookmarked = (key in bookmarks)

                new_tag  = " 🆕" if is_new else ""
                bm_tag   = " ⭐" if is_bookmarked else ""
                dot      = "● " if was_visited and not is_current else "○ "
                est_tag  = f" ({est_min}m)" if est_min else ""
                btn_label = f"{dot}{icon} {name}{new_tag}{bm_tag}"

                if is_current:
                    p_ = _get_palette()
                    st.markdown(f"""
                    <div style="
                        background:{p_['accent_soft']};
                        border:1px solid {p_['accent']};
                        border-radius:6px;
                        padding:0.35rem 0.6rem;
                        font-size:0.83rem;
                        font-weight:700;
                        color:{p_['accent']};
                        margin:0.15rem 0;
                    ">▶ {icon} {name}{new_tag}{bm_tag}
                    <span style="font-size:0.7rem;opacity:0.75;">{est_tag}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if st.button(btn_label, key=f"nav_{key}",
                                 use_container_width=True,
                                 help=f"~{est_min} min"):
                        jump_to = key

    return jump_to


# ============================================================
# NEXT / PREV MODULE FOOTER
# ============================================================
def _render_module_footer(current_key: str):
    """← Prev | chapter · module index | Next → navigation row."""
    keys = [r[0] for r in MODULE_REGISTRY]
    idx  = keys.index(current_key) if current_key in keys else -1
    if idx < 0:
        return

    meta     = _get_module_meta(current_key)
    ch_label = meta[4] if meta else ""
    p        = _get_palette()

    prev_key = keys[idx - 1] if idx > 0          else None
    next_key = keys[idx + 1] if idx < len(keys) - 1 else None

    st.markdown(f"""
    <div style="
        border-top:1px solid {p['border']};
        margin-top:2rem; padding-top:1rem;
        display:flex; justify-content:space-between; align-items:center;
        flex-wrap:wrap; gap:0.5rem;
        font-size:0.82rem; color:{p['text_secondary']};
    ">
        <div>{ch_label}</div>
        <div style="font-size:0.76rem;opacity:0.6;">
            Module {idx+1} of {len(keys)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_cols = st.columns([1, 2, 1])
    with nav_cols[0]:
        if prev_key:
            prev_meta = _get_module_meta(prev_key)
            if prev_meta and st.button(
                f"← {prev_meta[2]} {prev_meta[1]}",
                key="footer_prev", use_container_width=True,
            ):
                st.session_state.selected_module = prev_key
                st.rerun()

    with nav_cols[1]:
        # Related modules suggestion (same chapter)
        if meta:
            same_ch = [
                r for r in MODULE_REGISTRY
                if r[4] == ch_label and r[0] != current_key
            ]
            if same_ch:
                p_ = _get_palette()
                links = "  ·  ".join(
                    f"<span style='color:{p_['accent']};cursor:pointer;'>"
                    f"{r[2]} {r[1]}</span>"
                    for r in same_ch[:3]
                )
                st.markdown(
                    f"<div style='text-align:center;font-size:0.76rem;"
                    f"color:{p_['text_muted']};'>Also in this chapter: {links}</div>",
                    unsafe_allow_html=True,
                )

    with nav_cols[2]:
        if next_key:
            next_meta = _get_module_meta(next_key)
            if next_meta and st.button(
                f"{next_meta[2]} {next_meta[1]} →",
                key="footer_next", use_container_width=True,
            ):
                st.session_state.selected_module = next_key
                st.rerun()


# ============================================================
# MODULE → FUNCTION DISPATCH TABLE
# ============================================================
def _get_module_functions() -> dict:
    """Key → callable dispatch. One place to add new modules."""
    return {
        "risk":           module_risk,
        "pert":           module_pert,
        "crashing":       module_crashing,
        "breakeven":      module_breakeven,
        "decision":       module_decision,
        "learning":       module_learning,
        "decoupling":     module_decoupling,
        "linebalance":    module_linebalance,
        "service":        module_service,
        "pokayoke":       module_pokayoke,
        "queuing":        module_queuing,
        "distributions":  module_distributions,
        "littles":        module_littles,
        "dpmo":           module_dpmo,
        "fmea":           module_fmea,
        "sqc":            module_sqc,
        "capability":     module_capability,
        "sampling":       module_sampling,
        "pareto":         module_pareto,
        "fishbone":       module_fishbone,
        "sqc_practice":   module_sqc_practice,
        "lean":           module_lean,
        "centroid":       module_centroid,
        "factor":         module_factor,
        "transportation": module_transportation,
        "sourcing":       module_sourcing,
        "forecast":       module_forecast,
        "regression":     module_regression,
        "aggregate":      module_aggregate,
        "eoq":            module_eoq,
        "safetystock":    module_safetystock,
        "newsvendor":     module_newsvendor,
        "mrp":            module_mrp,
        "mrp_lotsizing":  module_mrp_lotsizing,
        "scheduling":     module_scheduling,
        "practice":       module_practice,
    }


# ============================================================
# MAIN APP
# ============================================================
def main():
    module_functions = _get_module_functions()
    p = _get_palette()

    # ── Sidebar header ────────────────────────────────────────
    st.sidebar.markdown(f"""
    <div style="
        text-align:center;
        padding:1.2rem 0.5rem 0.8rem;
        border-bottom:1px solid {p['border']};
        margin-bottom:0.5rem;
    ">
        <div style="font-size:2.4rem;">📊</div>
        <div style="font-weight:800;font-size:1.15rem;
                    color:{p['text_primary']};letter-spacing:-0.01em;">
            OSCM Simulator
        </div>
        <div style="font-size:0.72rem;color:{p['text_secondary']};margin-top:0.2rem;">
            Interactive Learning Edition
        </div>
        <div style="margin-top:0.5rem;">
            <span style="
                background:{p['accent_soft']};
                color:{p['accent_hover']};
                border:1px solid {p['accent']};
                border-radius:20px;
                padding:0.18rem 0.65rem;
                font-size:0.72rem;
                font-weight:700;
            ">{len(MODULE_REGISTRY)} Modules · 75+ Formulas</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme toggle ──────────────────────────────────────────
    render_theme_toggle()

    # ── Stats panel ───────────────────────────────────────────
    _render_stats_panel()
    st.sidebar.markdown("---")

    # ── Search ────────────────────────────────────────────────
    search_jump = _render_search()
    if search_jump:
        st.session_state.selected_module = search_jump
        st.rerun()

    # ── Bookmarks ─────────────────────────────────────────────
    _render_bookmarks()

    # ── Recently visited ──────────────────────────────────────
    _render_recent()

    st.sidebar.markdown("---")

    # ── Chapter navigation ────────────────────────────────────
    nav_jump = _render_chapter_nav()
    if nav_jump:
        st.session_state.selected_module = nav_jump
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "📖 Jacobs & Chase (2024). *Operations and Supply Chain Management*, "
        "17th ed. McGraw-Hill."
    )

    # ── Main panel ────────────────────────────────────────────
    selected = st.session_state.selected_module

    if selected:
        visited = st.session_state.get("modules_visited", set())
        visited.add(selected)
        st.session_state.modules_visited = visited
        _push_recent(selected)
        # Update best streak
        if st.session_state.correct_streak > st.session_state.best_streak:
            st.session_state.best_streak = st.session_state.correct_streak

    # ── Dispatch ──────────────────────────────────────────────
    if selected is None:
        _render_welcome()
    elif selected in module_functions:
        _render_breadcrumb(selected)
        module_functions[selected]()
        _render_module_footer(selected)
    else:
        st.error(
            f"Module `{selected}` not found. Please select a module from the sidebar.",
            icon="🚨",
        )
        if st.button("← Return to Welcome", key="error_home"):
            st.session_state.selected_module = None
            st.rerun()


if __name__ == "__main__":
    main()