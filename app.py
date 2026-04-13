"""
DAVL Analytics Suite — Weather Pattern Classification
Automated analysis of datasets: EDA, PCA, LDA, Factor Analysis, Clustering & Insights
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DAVL Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS — DAVL UI ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@300;400;600&family=DM+Sans:wght@400;500;700&display=swap');

:root {
    --ivory: #F8F6F1;
    --charcoal: #2C2C2C;
    --slate: #4A4A4A;
    --sage: #8B9D83;
    --forest: #3D5A3C;
    --copper: #B87D4B;
    --gold: #D4AF37;
    --shadow: rgba(0, 0, 0, 0.08);
    --shadow-strong: rgba(0, 0, 0, 0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #F8F6F1 !important;
    color: #2C2C2C !important;
}

/* ── Grain overlay hack ────────*/
.grain-overlay {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image: url('data:image/svg+xml;utf8,<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><filter id="noiseFilter"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" stitchTiles="stitch"/></filter><rect width="100%" height="100%" filter="url(%23noiseFilter)" opacity="0.03"/></svg>');
    pointer-events: none;
    z-index: 9999;
    opacity: 0.4;
}

/* ── Hide Sidebar ──────────────*/
[data-testid="stSidebar"]      { display: none !important; }
[data-testid="collapsedControl"]{ display: none !important; }

/* ── Main container ─────────── */
.block-container {
    padding-top: 0 !important;
    padding-left: 5% !important;
    padding-right: 5% !important;
    max-width: 1400px !important;
}

/* ── Header area ────────────── */
.davl-header-wrapper {
    margin: 0 -5.5% 3rem -5.5%;
    padding: 3rem 5%;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    background: linear-gradient(135deg, rgba(139, 157, 131, 0.03) 0%, rgba(184, 125, 75, 0.02) 100%);
    position: relative;
    overflow: hidden;
}
.davl-brand {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.davl-brand h1 {
    font-family: 'Crimson Pro', serif;
    font-size: 3.5rem;
    font-weight: 300;
    letter-spacing: -0.02em;
    color: #2C2C2C;
    margin:0; padding:0; line-height:1;
}
.davl-subtitle {
    font-size: 0.95rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #B87D4B;
    font-weight: 500;
}
.davl-tagline {
    font-family: 'Crimson Pro', serif;
    font-size: 1.25rem;
    color: #4A4A4A;
    font-style: italic;
    margin-top: 0.5rem;
}

/* ── Stats Bar ──────────────── */
.stats-bar {
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.06);
    padding: 2rem;
    margin-bottom: 3rem;
    display: flex;
    justify-content: space-around;
    align-items: center;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
    padding: 1rem;
    position: relative;
    flex: 1;
}
.stat-item:not(:last-child)::after {
    content: '';
    position: absolute;
    right: 0;
    top: 20%;
    height: 60%;
    width: 1px;
    background: rgba(0, 0, 0, 0.08);
}
.stat-value {
    font-family: 'Crimson Pro', serif;
    font-size: 2.5rem;
    font-weight: 300;
    color: #3D5A3C;
    display: block;
    margin-bottom: 0.25rem;
}
.stat-label {
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4A4A4A;
    font-weight: 500;
}

/* ── Buttons & UI Elements ──── */
.stButton > button {
    background: #2C2C2C;
    color: white;
    border: none;
    border-radius: 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.85rem 1.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: all 0.3s ease;
    width: auto !important;
}
.stButton > button:hover {
    background: #3D5A3C;
    color: white;
    transform: translateX(2px);
}
.stDownloadButton > button {
    background: white;
    color: #2C2C2C;
    border: 1px solid rgba(0,0,0,0.1);
}
.stDownloadButton > button:hover { background: #f0f0f0; border-color: #8B9D83; }

/* ── Overrides for forms & dataframes */
[data-testid="stFileUploader"] {
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.08);
    padding: 1.5rem;
    border-radius: 8px;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #8B9D83;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] > div {
    border-radius: 0;
    border: 1px solid rgba(0,0,0,0.06);
    background: white;
}
.insight-item {
    background: white;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 4px solid #3D5A3C;
}
.insight-warn { border-left-color: #B87D4B; background: #FFFdfa;}
.insight-good { border-left-color: #8B9D83; background: #fbfdf9;}
.wl-info { border-left: 4px solid #4A4A4A; background: white; padding:15px; margin: 15px 0;}
.wl-success { border-left: 4px solid #3D5A3C; background: white; padding:15px; margin: 15px 0;}

/* Footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
<div class="grain-overlay"></div>
""", unsafe_allow_html=True)


# ── Module imports ─────────────────────────────────────────────────────────────
from utils.data_loader   import load_dataset, get_dataset_info, convert_df_to_csv, detect_target_column
from utils.overview      import render_overview
from utils.quality       import render_quality
from utils.preprocessing import render_preprocessing
from utils.eda           import render_eda
from utils.visualization import render_visualizations
from utils.stats         import render_stats
from utils.pca_analysis  import render_pca
from utils.lda_analysis  import render_lda
from utils.factor_analysis import render_factor_analysis
from utils.clustering    import render_clustering
from utils.insights      import render_insights
from utils.split         import render_train_test_split


# ── State Initialization ────────────────────────────────────────────────────────
if "current_module" not in st.session_state:
    st.session_state.current_module = "home"

def launch_module(module_name: str):
    st.session_state.current_module = module_name

def go_home():
    st.session_state.current_module = "home"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="davl-header-wrapper">
    <div class="davl-brand">
        <h1>DAVL</h1>
        <span class="davl-subtitle">Analytics Suite</span>
    </div>
    <p class="davl-tagline">Sophisticated data analysis and visualization laboratory</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload dataset to begin analysis:",
    type=["csv", "xlsx", "xls"],
    key="main_upload",
)
df   = load_dataset(uploaded_file) if uploaded_file else None
info = get_dataset_info(df)        if df is not None else None

# Helper to show stats bar
def show_stats_bar():
    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-value">12</span><span class="stat-label">Analysis Modules</span></div>
        <div class="stat-item"><span class="stat-value">∞</span><span class="stat-label">Data Points</span></div>
        <div class="stat-item"><span class="stat-value">8</span><span class="stat-label">Visualization Types</span></div>
        <div class="stat-item"><span class="stat-value">100%</span><span class="stat-label">Precision</span></div>
    </div>
    """, unsafe_allow_html=True)


# Modular mapping for ease of scaling
MODULE_DEF = {
    "Overview": {"icon": "📋", "desc": "Quick summary statistics, data types, and initial exploration of your dataset.", "func": render_overview},
    "Quality": {"icon": "✨", "desc": "Automated data quality assessment with completeness, consistency, and metrics.", "func": render_quality},
    "Preprocessing": {"icon": "🔧", "desc": "Clean, transform, and prepare your data with missing value imputation.", "func": render_preprocessing},
    "EDA": {"icon": "📈", "desc": "Comprehensive statistical analysis with distribution and outlier detection.", "func": render_eda},
    "Visualizations": {"icon": "📊", "desc": "Create publication-quality charts with customizable themes.", "func": render_visualizations},
    "Statistics": {"icon": "🎲", "desc": "Comprehensive hypothesis testing including t-tests, ANOVA, and chi-square.", "func": render_stats},
    "Train/Test": {"icon": "✂️", "desc": "Intelligently partition your dataset for model training and validation.", "func": render_train_test_split},
    "PCA": {"icon": "🎯", "desc": "Dimensionality reduction and feature extraction using PCA.", "func": render_pca},
    "LDA": {"icon": "📝", "desc": "Latent Dirichlet Allocation for discovering abstract topics in collections.", "func": render_lda},
    "Factor": {"icon": "🔍", "desc": "Identify latent variables and underlying structure in your data.", "func": render_factor_analysis},
    "Clustering": {"icon": "🎨", "desc": "Unsupervised learning with K-means, hierarchical, and DBSCAN clustering.", "func": render_clustering},
    "Insights": {"icon": "💡", "desc": "AI-powered insights generation that highlights patterns, and anomalies.", "func": render_insights},
}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VIEW LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if df is None:
    show_stats_bar()
    st.markdown("<p style='color:#4A4A4A; font-family:\"DM Sans\"; text-align: center; margin-top: 2rem;'>👆 Please upload a CSV or Excel dataset above to load the modules.</p>", unsafe_allow_html=True)
    st.stop()

else:
    # Render Dashboard Grid if in home
    if st.session_state.current_module == "home":
        show_stats_bar()
        target = detect_target_column(df)
        st.markdown(f"""
        <div class="wl-success">
          ✅ &nbsp;<b>{uploaded_file.name if uploaded_file else 'Data Loaded'}</b> — {info['rows']:,} rows × {info['cols']} columns
          &nbsp;|&nbsp; Memory: {info['memory_mb']} MB
          {f"&nbsp;|&nbsp; 🎯 Detected target: <b>{target}</b>" if target else ""}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid layout (3 cols)
        cols = st.columns(3)
        for i, (m_id, m_data) in enumerate(MODULE_DEF.items()):
            col = cols[i % 3]
            with col:
                st.markdown(f"""
                <div style="background: white; border: 1px solid rgba(0,0,0,0.06); padding: 2rem; position: relative; margin-bottom: 2rem; height: 280px;">
                    <div style="width:48px; height:48px; background:linear-gradient(135deg, #8B9D83, #3D5A3C); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.5rem; margin-bottom:1.5rem; box-shadow:0 4px 12px rgba(139,157,131,0.2);">{m_data['icon']}</div>
                    <h3 style="font-family:'Crimson Pro',serif; font-size:1.5rem; font-weight:400; margin-bottom:0.5rem; color:#2C2C2C;">{m_id}</h3>
                    <p style="color:#4A4A4A; font-size:0.9rem; line-height:1.6; margin-bottom:1rem;">{m_data['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                # Streamlit button styled over the markdown
                st.button(f"Launch {m_id} →", key=f"btn_{m_id}", on_click=launch_module, args=(m_id,))

    # Render Specific Module
    else:
        m_id = st.session_state.current_module
        if m_id in MODULE_DEF:
            st.button("← Back to Dashboard", on_click=go_home, key="back_btn")
            st.markdown("<hr style='border-top:1px solid rgba(0,0,0,0.08); margin: 1rem 0 2rem;'>", unsafe_allow_html=True)
            
            st.markdown(f"<h2 style='font-family:\"Crimson Pro\"; font-weight:400; color:#3D5A3C; margin-bottom:1.5rem;'>{MODULE_DEF[m_id]['icon']} {m_id}</h2>", unsafe_allow_html=True)
            MODULE_DEF[m_id]["func"](df, info)
        else:
            st.error(f"Module {m_id} not found.")
            st.button("Return Home", on_click=go_home)

