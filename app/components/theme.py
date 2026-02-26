"""Theme utilities for AI Artist 2.0 Streamlit app."""

import streamlit as st
from pathlib import Path

# CSS variable sets for each theme
DARK_VARS = """
:root {
    --bg-primary:        #0f0f0f;
    --bg-secondary:      #171717;
    --bg-tertiary:       #1e1e1e;
    --bg-chat-user:      #1e1e1e;
    --bg-chat-assistant: #141414;
    --bg-sidebar:        #141414;
    --bg-input:          #1e1e1e;
    --bg-card:           #171717;
    --bg-hover:          #252525;

    --text-primary:      #ececec;
    --text-secondary:    #a1a1a1;
    --text-tertiary:     #6b6b6b;
    --text-inverse:      #0f0f0f;

    --border-primary:    #2e2e2e;
    --border-secondary:  #222222;

    --accent:            #818cf8;
    --accent-hover:      #6366f1;
    --accent-success:    #34d399;
    --accent-warning:    #fbbf24;
    --accent-error:      #f87171;

    --shadow-sm:         0 1px 2px rgba(0,0,0,0.3);
    --shadow-md:         0 4px 6px rgba(0,0,0,0.4);
    --radius:            8px;
    --radius-lg:         12px;
}
"""

LIGHT_VARS = """
:root {
    --bg-primary:        #ffffff;
    --bg-secondary:      #f8f9fa;
    --bg-tertiary:       #f0f1f3;
    --bg-chat-user:      #f0f1f3;
    --bg-chat-assistant: #ffffff;
    --bg-sidebar:        #f4f4f5;
    --bg-input:          #ffffff;
    --bg-card:           #ffffff;
    --bg-hover:          #f0f1f3;

    --text-primary:      #1a1a1a;
    --text-secondary:    #4b5563;
    --text-tertiary:     #9ca3af;
    --text-inverse:      #ffffff;

    --border-primary:    #e5e7eb;
    --border-secondary:  #f0f1f3;

    --accent:            #6366f1;
    --accent-hover:      #4f46e5;
    --accent-success:    #10b981;
    --accent-warning:    #f59e0b;
    --accent-error:      #ef4444;

    --shadow-sm:         0 1px 2px rgba(0,0,0,0.05);
    --shadow-md:         0 4px 6px rgba(0,0,0,0.07);
    --radius:            8px;
    --radius-lg:         12px;
}
"""

# ── Dark mode: override every Streamlit native element ──────────────────
DARK_ST_OVERRIDES = """
/* === Layout shells === */
.stApp,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"],
[data-testid="stVerticalBlock"] {
    background-color: #0f0f0f !important;
    color: #ececec !important;
}

/* === Sidebar === */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #141414 !important;
    color: #ececec !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ececec !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #a1a1a1 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #6b6b6b !important;
}
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] a span {
    color: #a1a1a1 !important;
}

/* === All text in main area === */
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span,
[data-testid="stMainBlockContainer"] li,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4,
[data-testid="stMainBlockContainer"] td,
[data-testid="stMainBlockContainer"] th {
    color: #ececec !important;
}

/* === Chat messages === */
[data-testid="stChatMessage"] {
    background-color: #141414 !important;
    border: 1px solid #2e2e2e !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] code {
    color: #ececec !important;
}

/* === Chat input === */
[data-testid="stChatInput"] {
    background-color: #0f0f0f !important;
    border-top: 1px solid #2e2e2e !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    color: #ececec !important;
    caret-color: #ececec !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6b6b6b !important;
}
/* Chat send button */
[data-testid="stChatInput"] button,
[data-testid="stChatInputSubmitButton"] {
    background-color: transparent !important;
    color: #a1a1a1 !important;
}
[data-testid="stChatInput"] button svg {
    fill: #a1a1a1 !important;
    stroke: #a1a1a1 !important;
}

/* === Selectbox / Dropdowns === */
[data-baseweb="select"],
[data-baseweb="select"] > div {
    background-color: #1e1e1e !important;
    border-color: #2e2e2e !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] input {
    color: #ececec !important;
}
/* Dropdown menu */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #1e1e1e !important;
    color: #ececec !important;
}
[role="option"]:hover {
    background-color: #252525 !important;
}
[data-baseweb="menu"] li {
    color: #ececec !important;
}

/* === Slider === */
[data-testid="stSlider"] label {
    color: #a1a1a1 !important;
}
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #6b6b6b !important;
}
[data-testid="stThumbValue"] {
    color: #ececec !important;
}

/* === Number input === */
.stNumberInput input {
    background-color: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    color: #ececec !important;
}
.stNumberInput button {
    background-color: #1e1e1e !important;
    border-color: #2e2e2e !important;
    color: #ececec !important;
}

/* === Toggle === */
[data-testid="stToggle"] span {
    color: #ececec !important;
}

/* === Checkbox === */
[data-testid="stCheckbox"] label span {
    color: #ececec !important;
}

/* === Buttons (non-primary keep themed) === */
.stButton > button {
    border-color: #2e2e2e !important;
}

/* === Expanders === */
[data-testid="stExpander"] {
    background-color: #171717 !important;
    border: 1px solid #2e2e2e !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #a1a1a1 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #0f0f0f !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] div {
    color: #ececec !important;
}

/* === Alerts (keep default alert colors but fix text visibility) === */
[data-testid="stAlert"] p {
    color: inherit !important;
}

/* === Status widget === */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] span,
[data-testid="stStatusWidget"] label {
    color: #ececec !important;
}

/* === Tabs === */
.stTabs [data-baseweb="tab"] {
    color: #a1a1a1 !important;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color: #0f0f0f !important;
}

/* === Tables === */
[data-testid="stTable"] table,
[data-testid="stTable"] th,
[data-testid="stTable"] td,
.stMarkdown table,
.stMarkdown th,
.stMarkdown td {
    color: #ececec !important;
    border-color: #2e2e2e !important;
}
.stMarkdown th {
    background-color: #171717 !important;
}

/* === Code blocks === */
pre, code {
    background-color: #1e1e1e !important;
    color: #ececec !important;
}

/* === Captions === */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #6b6b6b !important;
}

/* === Metrics === */
[data-testid="stMetric"] {
    background-color: #171717 !important;
    border: 1px solid #2e2e2e !important;
}
[data-testid="stMetricValue"] {
    color: #ececec !important;
}
[data-testid="stMetricLabel"] {
    color: #a1a1a1 !important;
}

/* === Tooltips === */
[data-baseweb="tooltip"] {
    background-color: #1e1e1e !important;
    color: #ececec !important;
}

/* === Popover / icon buttons === */
[data-testid="stHeaderActionElements"] button {
    color: #a1a1a1 !important;
}
"""

# ── Light mode: override Streamlit defaults for light look ──────────────
LIGHT_ST_OVERRIDES = """
/* === Layout shells === */
.stApp,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stBottomBlockContainer"],
[data-testid="stVerticalBlock"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* === Sidebar === */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background-color: #f4f4f5 !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #4b5563 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #9ca3af !important;
}
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] a span {
    color: #4b5563 !important;
}

/* === All text in main area === */
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span,
[data-testid="stMainBlockContainer"] li,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4,
[data-testid="stMainBlockContainer"] td,
[data-testid="stMainBlockContainer"] th {
    color: #1a1a1a !important;
}

/* === Chat messages === */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] code {
    color: #1a1a1a !important;
}

/* === Chat input === */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border-top: 1px solid #e5e7eb !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    color: #1a1a1a !important;
    caret-color: #1a1a1a !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
}
[data-testid="stChatInput"] button,
[data-testid="stChatInputSubmitButton"] {
    background-color: transparent !important;
    color: #4b5563 !important;
}

/* === Selectbox / Dropdowns === */
[data-baseweb="select"],
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #e5e7eb !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] input {
    color: #1a1a1a !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}
[role="option"]:hover {
    background-color: #f0f1f3 !important;
}
[data-baseweb="menu"] li {
    color: #1a1a1a !important;
}

/* === Slider === */
[data-testid="stSlider"] label {
    color: #4b5563 !important;
}
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #9ca3af !important;
}
[data-testid="stThumbValue"] {
    color: #1a1a1a !important;
}

/* === Number input === */
.stNumberInput input {
    background-color: #f8f9fa !important;
    border: 1px solid #e5e7eb !important;
    color: #1a1a1a !important;
}
.stNumberInput button {
    background-color: #f8f9fa !important;
    border-color: #e5e7eb !important;
    color: #1a1a1a !important;
}

/* === Toggle === */
[data-testid="stToggle"] span {
    color: #1a1a1a !important;
}

/* === Checkbox === */
[data-testid="stCheckbox"] label span {
    color: #1a1a1a !important;
}

/* === Expanders === */
[data-testid="stExpander"] {
    background-color: #f8f9fa !important;
    border: 1px solid #e5e7eb !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #4b5563 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: #ffffff !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] span,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] div {
    color: #1a1a1a !important;
}

/* === Alerts === */
[data-testid="stAlert"] p {
    color: inherit !important;
}

/* === Status widget === */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] span,
[data-testid="stStatusWidget"] label {
    color: #1a1a1a !important;
}

/* === Tabs === */
.stTabs [data-baseweb="tab"] {
    color: #4b5563 !important;
}
.stTabs [aria-selected="true"] {
    color: #6366f1 !important;
}

/* === Tables === */
[data-testid="stTable"] table,
[data-testid="stTable"] th,
[data-testid="stTable"] td,
.stMarkdown table,
.stMarkdown th,
.stMarkdown td {
    color: #1a1a1a !important;
    border-color: #e5e7eb !important;
}
.stMarkdown th {
    background-color: #f8f9fa !important;
}

/* === Code blocks === */
pre, code {
    background-color: #f0f1f3 !important;
    color: #1a1a1a !important;
}

/* === Captions === */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #9ca3af !important;
}

/* === Metrics === */
[data-testid="stMetric"] {
    background-color: #f8f9fa !important;
    border: 1px solid #e5e7eb !important;
}
[data-testid="stMetricValue"] {
    color: #1a1a1a !important;
}
[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
}

/* === Tooltips === */
[data-baseweb="tooltip"] {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}

/* === Popover / icon buttons === */
[data-testid="stHeaderActionElements"] button {
    color: #4b5563 !important;
}
"""


def init_theme():
    """Initialize theme in session state if not set."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"


def load_css_and_theme():
    """Load the custom CSS file and inject theme-specific CSS variables."""
    # 1. Load base stylesheet (layout, selectors using var(--*) references)
    css_path = Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # 2. Inject the correct CSS variable set + native overrides
    is_dark = st.session_state.get("theme", "dark") == "dark"
    theme_vars = DARK_VARS if is_dark else LIGHT_VARS
    overrides = DARK_ST_OVERRIDES if is_dark else LIGHT_ST_OVERRIDES

    st.markdown(f"<style>{theme_vars}\n{overrides}</style>", unsafe_allow_html=True)


def render_theme_toggle():
    """Render the dark/light mode toggle. Call inside a `with st.sidebar:` block."""
    is_dark = st.toggle(
        "Dark mode",
        value=st.session_state.get("theme", "dark") == "dark",
    )
    new_theme = "dark" if is_dark else "light"
    if new_theme != st.session_state.get("theme", "dark"):
        st.session_state.theme = new_theme
        st.rerun()
