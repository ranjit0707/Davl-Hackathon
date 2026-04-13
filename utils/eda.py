"""
EDA Module — Univariate, Bivariate & Correlation analysis for weather data.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_style import PALETTE, BLUE, INDIGO, EMERALD, AMBER, base_layout, heatmap_layout
from utils.data_loader import detect_target_column

# Typical weather feature keywords
WEATHER_FEATURES = ["temp", "humidity", "pressure", "wind", "rain", "rainfall",
                     "precipitation", "cloud", "sunshine", "visibility", "dew", "uv"]


def _find_weather_cols(df: pd.DataFrame, num_cols: list) -> list:
    """Return numeric cols that match weather keywords."""
    matched = []
    for col in num_cols:
        for kw in WEATHER_FEATURES:
            if kw in col.lower():
                matched.append(col)
                break
    return matched if matched else num_cols[:6]


def render_eda(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 📊 Exploratory Data Analysis")

    num_cols = info["numeric_cols"]
    cat_cols = info["categorical_cols"]
    target   = detect_target_column(df)

    if not num_cols:
        st.warning("No numeric columns found for EDA.")
        return

    _sel_eda = st.radio('Select View', ["Univariate Analysis", "Bivariate Analysis", "Correlation Analysis"], horizontal=True, label_visibility='collapsed', key='radio_eda_ca2b')

    # ── Univariate ────────────────────────────────────────────────────────────
    if _sel_eda == "Univariate Analysis":
        st.markdown("##### Distribution of Numeric Features")
        weather_cols = _find_weather_cols(df, num_cols)

        selected_col = st.selectbox("Select Column", num_cols, key="eda_uni_col")
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.histogram(df, x=selected_col,
                               color_discrete_sequence=[BLUE],
                               nbins=40, opacity=0.85)
            fig.update_layout(**base_layout(f"Distribution — {selected_col}"))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.violin(df, y=selected_col, box=True, points="outliers",
                             color_discrete_sequence=[INDIGO])
            fig2.update_layout(**base_layout(f"Violin — {selected_col}"))
            st.plotly_chart(fig2, use_container_width=True)

        # Grid of histograms for weather cols
        st.markdown("##### Feature Histograms (Weather Variables)")
        w_cols = _find_weather_cols(df, num_cols)
        n = len(w_cols)
        if n > 0:
            cols_per_row = 3
            rows = [(w_cols[i:i+cols_per_row]) for i in range(0, n, cols_per_row)]
            for row in rows:
                cs = st.columns(cols_per_row)
                for i, col in enumerate(row):
                    with cs[i]:
                        fig_h = px.histogram(df, x=col,
                                             color_discrete_sequence=[PALETTE[i % len(PALETTE)]],
                                             nbins=30, opacity=0.85)
                        fig_h.update_layout(**base_layout(col, height=260))
                        st.plotly_chart(fig_h, use_container_width=True)

        # Categorical distributions
        if cat_cols:
            st.markdown("##### Categorical Feature Distributions")
            sel_cat = st.selectbox("Select Categorical Column", cat_cols, key="eda_cat")
            vc = df[sel_cat].value_counts().head(20).reset_index()
            vc.columns = ["Value", "Count"]
            fig_c = px.bar(vc, x="Value", y="Count",
                           color="Count",
                           color_continuous_scale=[[0, "#dbeafe"], [1, BLUE]])
            fig_c.update_layout(**base_layout(f"Value Counts — {sel_cat}", height=320))
            fig_c.update_traces(marker_line_width=0)
            st.plotly_chart(fig_c, use_container_width=True)

    # ── Bivariate ─────────────────────────────────────────────────────────────
    if _sel_eda == "Bivariate Analysis":
        st.markdown("##### Bivariate Relationships")
        col_a, col_b = st.columns(2)
        with col_a:
            x_col = st.selectbox("X Axis", num_cols, index=0, key="eda_biv_x")
        with col_b:
            y_col = st.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1), key="eda_biv_y")

        color_col = target if target and target in df.columns else None

        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                         opacity=0.7,
                         color_discrete_sequence=PALETTE,
                         title=f"{x_col} vs {y_col}")
        fig.update_layout(**base_layout(f"{x_col} vs {y_col}", height=430))
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot by target
        if target and target in df.columns and num_cols:
            st.markdown("##### Boxplot by Weather Class")
            box_col = st.selectbox("Numeric column", num_cols, key="eda_box_col")
            fig_b = px.box(df, x=target, y=box_col,
                           color=target, color_discrete_sequence=PALETTE)
            fig_b.update_layout(**base_layout(f"{box_col} by {target}", height=380))
            st.plotly_chart(fig_b, use_container_width=True)

        # Scatter pair plot (top 4 weather cols)
        st.markdown("##### Scatter Matrix (Top Weather Features)")
        best4 = _find_weather_cols(df, num_cols)[:4]
        if len(best4) >= 2:
            fig_pm = px.scatter_matrix(df, dimensions=best4, color=color_col,
                                       color_discrete_sequence=PALETTE, opacity=0.6)
            fig_pm.update_traces(diagonal_visible=False)
            fig_pm.update_layout(height=550, paper_bgcolor="white",
                                  font=dict(family="Inter", size=10))
            st.plotly_chart(fig_pm, use_container_width=True)

    # ── Correlation ───────────────────────────────────────────────────────────
    if _sel_eda == "Correlation Analysis":
        if len(num_cols) < 2:
            st.info("Need ≥ 2 numeric columns for correlation.")
            return

        corr = df[num_cols].corr()

        fig_hm = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig_hm.update_layout(**heatmap_layout("Correlation Heatmap", height=550))
        st.plotly_chart(fig_hm, use_container_width=True)

        # Top correlations table
        upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs  = upper.stack().reset_index()
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["Abs Corr"] = pairs["Correlation"].abs()
        pairs = pairs.sort_values("Abs Corr", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔴 Strongest Correlations (|r| > 0.7)**")
            strong = pairs[pairs["Abs Corr"] > 0.7].head(15)
            if strong.empty:
                st.info("No pairs with |r| > 0.7 found.")
            else:
                st.dataframe(strong[["Feature A", "Feature B", "Correlation"]].round(4),
                             use_container_width=True)
        with col_b:
            st.markdown("**📊 Top 10 Correlation Pairs**")
            st.dataframe(pairs[["Feature A", "Feature B", "Correlation"]].head(10).round(4),
                         use_container_width=True)
