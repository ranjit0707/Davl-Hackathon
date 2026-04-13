"""
Overview Module — data types, unique values, memory, target detection.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from utils.chart_style import PALETTE, BLUE, base_layout
from utils.data_loader import detect_target_column


def render_overview(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 🗂️ Dataset Overview")

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows",       f"{info['rows']:,}")
    c2.metric("Total Columns",    info['cols'])
    c3.metric("Numeric Features", len(info['numeric_cols']))
    c4.metric("Categorical Cols", len(info['categorical_cols']))
    c5.metric("Memory (MB)",      info['memory_mb'])

    st.markdown('<hr style="border-top:1px solid #e2e8f0;margin:16px 0">', unsafe_allow_html=True)

    # ── Column info table ─────────────────────────────────────────────────────
    col_info = []
    for c in df.columns:
        col_info.append({
            "Column":        c,
            "Data Type":     str(df[c].dtype),
            "Non-Null":      int(df[c].notnull().sum()),
            "Null Count":    int(df[c].isnull().sum()),
            "Unique Values": int(df[c].nunique()),
            "Sample Value":  str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "—",
        })
    info_df = pd.DataFrame(col_info)

    st.markdown("##### 📋 Column Information")
    st.dataframe(info_df, use_container_width=True, height=300)

    # ── Dtype distribution chart ──────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        dtype_counts = info_df["Data Type"].value_counts().reset_index()
        dtype_counts.columns = ["Type", "Count"]
        fig = px.pie(dtype_counts, values="Count", names="Type",
                     color_discrete_sequence=PALETTE, hole=0.45)
        fig.update_layout(**base_layout("Data Type Distribution", height=300))
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### 🎯 Target Column Detection")
        target = detect_target_column(df)
        if target:
            st.markdown(f"""
            <div style="background:#eff6ff;border-left:4px solid #3b82f6;border-radius:8px;
                        padding:14px 18px;margin:8px 0;">
              <b>Detected:</b> <code>{target}</code><br>
              <b>Unique classes:</b> {df[target].nunique()}<br>
              <b>Class values:</b> {", ".join(str(v) for v in df[target].unique()[:10])}
            </div>
            """, unsafe_allow_html=True)

            vc = df[target].value_counts().head(15).reset_index()
            vc.columns = ["Class", "Count"]
            fig2 = px.bar(vc, x="Class", y="Count",
                          color="Count",
                          color_continuous_scale=[[0, "#dbeafe"], [1, BLUE]])
            fig2.update_layout(**base_layout("Class Distribution", height=260))
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No clear target column detected. Analysis will treat all columns as features.")

    # ── Memory by column ──────────────────────────────────────────────────────
    with st.expander("📦 Memory Usage by Column"):
        mem = df.memory_usage(deep=True).drop("Index").reset_index()
        mem.columns = ["Column", "Bytes"]
        mem["KB"] = (mem["Bytes"] / 1024).round(2)
        mem = mem.sort_values("KB", ascending=False)
        fig3 = px.bar(mem.head(20), x="Column", y="KB",
                      color="KB", color_continuous_scale=[[0, "#e0f2fe"], [1, "#0369a1"]])
        fig3.update_layout(**base_layout("Memory Usage per Column (KB)", height=300))
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)
