"""
Data Quality Module — missing values, duplicates, outliers, cardinality.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_style import PALETTE, BLUE, AMBER, RED, base_layout


def render_quality(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 🔍 Data Quality Report")

    num_cols = info["numeric_cols"]
    n_rows   = info["rows"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_missing = sum(info["null_counts"].values())
    dup_count     = int(df.duplicated().sum())
    miss_pct      = round(total_missing / (n_rows * info["cols"]) * 100, 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing Values",   total_missing)
    c2.metric("Missing %",        f"{miss_pct}%")
    c3.metric("Duplicate Rows",   dup_count)
    c4.metric("Complete Rows",    f"{n_rows - df[df.isnull().any(axis=1)].shape[0]:,}")

    st.markdown('<hr style="border-top:1px solid #e2e8f0;margin:14px 0">', unsafe_allow_html=True)

    _sel_quality = st.radio('Select View', ["Missing Values", "Duplicates", "Outliers", "High Cardinality"], horizontal=True, label_visibility='collapsed', key='radio_quality_73a6')

    # ── Missing Values ────────────────────────────────────────────────────────
    if _sel_quality == "Missing Values":
        miss_df = pd.DataFrame({
            "Column":      df.columns.tolist(),
            "Missing":     df.isnull().sum().tolist(),
            "Missing %":   (df.isnull().sum() / n_rows * 100).round(2).tolist(),
            "Data Type":   df.dtypes.astype(str).tolist(),
        }).sort_values("Missing", ascending=False)

        cols_with_miss = miss_df[miss_df["Missing"] > 0]

        if cols_with_miss.empty:
            st.markdown('<div class="wl-success">✅ No missing values detected in the dataset.</div>',
                        unsafe_allow_html=True)
        else:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.dataframe(cols_with_miss, use_container_width=True)
            with col_b:
                fig = px.bar(
                    cols_with_miss.head(15), x="Missing %", y="Column",
                    orientation="h", color="Missing %",
                    color_continuous_scale=[[0, "#fef9c3"], [0.5, AMBER], [1, RED]],
                )
                fig.update_layout(**base_layout("Missing Values (%)", height=350))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

            # Heatmap of missingness
            if len(cols_with_miss) > 0:
                miss_matrix = df[cols_with_miss["Column"].tolist()].isnull().astype(int)
                fig2 = px.imshow(
                    miss_matrix.head(100).T,
                    color_continuous_scale=[[0, "#f0fdf4"], [1, "#dc2626"]],
                    labels=dict(color="Missing"),
                    aspect="auto",
                )
                fig2.update_layout(**base_layout("Missingness Heatmap (first 100 rows)", height=300))
                st.plotly_chart(fig2, use_container_width=True)

    # ── Duplicates ────────────────────────────────────────────────────────────
    if _sel_quality == "Duplicates":
        if dup_count == 0:
            st.markdown('<div class="wl-success">✅ No duplicate rows found.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="wl-warn">⚠️ {dup_count} duplicate rows detected '
                        f'({round(dup_count/n_rows*100,2)}% of data).</div>',
                        unsafe_allow_html=True)
            dupes = df[df.duplicated(keep=False)]
            st.dataframe(dupes.head(30), use_container_width=True)

    # ── Outliers (IQR method) ─────────────────────────────────────────────────
    if _sel_quality == "Outliers":
        if not num_cols:
            st.info("No numeric columns for outlier detection.")
        else:
            outlier_data = []
            for col in num_cols:
                s = df[col].dropna()
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR     = Q3 - Q1
                n_out   = int(((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum())
                outlier_data.append({
                    "Column":    col,
                    "Q1":        round(Q1, 3),
                    "Q3":        round(Q3, 3),
                    "IQR":       round(IQR, 3),
                    "Outliers":  n_out,
                    "Outlier %": round(n_out / len(s) * 100, 2),
                })
            out_df = pd.DataFrame(outlier_data).sort_values("Outlier %", ascending=False)

            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.dataframe(out_df, use_container_width=True)
            with col_b:
                fig = px.bar(
                    out_df[out_df["Outliers"] > 0].head(15),
                    x="Outlier %", y="Column", orientation="h",
                    color="Outlier %",
                    color_continuous_scale=[[0, "#fef3c7"], [1, "#f59e0b"]],
                )
                fig.update_layout(**base_layout("Outlier Rate by Column (%)", height=350))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

            # Box plots for top outlier columns
            top_out_cols = out_df[out_df["Outliers"] > 0]["Column"].head(6).tolist()
            if top_out_cols:
                st.markdown("##### 📦 Boxplots — Top Outlier Columns")
                n = len(top_out_cols)
                cols_row = st.columns(min(n, 3))
                for i, col in enumerate(top_out_cols):
                    with cols_row[i % 3]:
                        fig_b = px.box(df, y=col, color_discrete_sequence=[BLUE])
                        fig_b.update_layout(**base_layout(col, height=260))
                        st.plotly_chart(fig_b, use_container_width=True)

    # ── High Cardinality ──────────────────────────────────────────────────────
    if _sel_quality == "High Cardinality":
        card_df = pd.DataFrame({
            "Column":      df.columns.tolist(),
            "Unique":      [df[c].nunique() for c in df.columns],
            "Cardinality %": [(df[c].nunique() / n_rows * 100).__round__(2) for c in df.columns],
            "Type":        df.dtypes.astype(str).tolist(),
        }).sort_values("Unique", ascending=False)

        high_card = card_df[card_df["Cardinality %"] > 50]
        st.dataframe(card_df, use_container_width=True)
        if not high_card.empty:
            st.markdown(f'<div class="wl-warn">⚠️ {len(high_card)} column(s) with cardinality > 50%: '
                        f'{high_card["Column"].tolist()}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="wl-success">✅ No high-cardinality columns detected.</div>',
                        unsafe_allow_html=True)
