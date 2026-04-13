"""
Visualization Module — weather-specific charts: seasonal trends, wind roses,
temperature/humidity/rainfall plots.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_style import PALETTE, BLUE, EMERALD, AMBER, INDIGO, base_layout
from utils.data_loader import detect_target_column

SEASON_KW   = ["season", "month", "date", "time", "year", "quarter"]
TEMP_KW     = ["temp", "temperature"]
HUMID_KW    = ["humidity", "humid"]
RAIN_KW     = ["rain", "rainfall", "precipitation"]
PRESSURE_KW = ["pressure", "baro"]
WIND_KW     = ["wind"]


def _find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None


def render_visualizations(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 📈 Weather Visualizations")

    num_cols = info["numeric_cols"]
    cat_cols = info["categorical_cols"]
    target   = detect_target_column(df)

    temp_col     = _find_col(df, TEMP_KW)
    humid_col    = _find_col(df, HUMID_KW)
    rain_col     = _find_col(df, RAIN_KW)
    pressure_col = _find_col(df, PRESSURE_KW)
    wind_col     = _find_col(df, WIND_KW)
    season_col   = _find_col(df, SEASON_KW)

    _sel_visualization = st.radio('Select View', ["Temperature", "Humidity & Rain", "Pressure & Wind", "Seasonal Trends", "Custom Charts"], horizontal=True, label_visibility='collapsed', key='radio_visualization_36c0')

    # ── Temperature ───────────────────────────────────────────────────────────
    if _sel_visualization == "Temperature":
        if temp_col:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=temp_col, nbins=40,
                                   color_discrete_sequence=[BLUE], opacity=0.85)
                fig.add_vline(x=df[temp_col].mean(), line_dash="dash",
                              line_color="#ef4444", annotation_text="Mean")
                fig.update_layout(**base_layout(f"Temperature Distribution — {temp_col}"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df, y=temp_col, x=target if target else None,
                              color=target if target else None,
                              color_discrete_sequence=PALETTE)
                fig2.update_layout(**base_layout(f"Temperature Boxplot by Weather Type"))
                st.plotly_chart(fig2, use_container_width=True)

            # Temp vs Humidity scatter
            if humid_col:
                fig3 = px.scatter(df, x=temp_col, y=humid_col,
                                  color=target if target else None,
                                  color_discrete_sequence=PALETTE,
                                  opacity=0.65)
                fig3.update_layout(**base_layout(f"Temperature vs Humidity", height=420))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No temperature column detected. Select a column below.")
            col_sel = st.selectbox("Temperature column", num_cols, key="viz_temp")
            if col_sel:
                fig = px.histogram(df, x=col_sel, nbins=40,
                                   color_discrete_sequence=[BLUE])
                fig.update_layout(**base_layout(f"Distribution — {col_sel}"))
                st.plotly_chart(fig, use_container_width=True)

    # ── Humidity & Rain ───────────────────────────────────────────────────────
    if _sel_visualization == "Humidity & Rain":
        plot_cols = [c for c in [humid_col, rain_col] if c]
        if plot_cols:
            for col in plot_cols:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=col, nbins=35,
                                       color_discrete_sequence=[EMERALD], opacity=0.85)
                    fig.update_layout(**base_layout(f"Distribution — {col}"))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig2 = px.box(df, y=col, color=target if target else None,
                                  color_discrete_sequence=PALETTE)
                    fig2.update_layout(**base_layout(f"Boxplot — {col}"))
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No humidity/rainfall columns detected.")
            cols_pick = st.multiselect("Select columns to plot", num_cols, default=num_cols[:2], key="viz_hum")
            for col in cols_pick:
                fig = px.histogram(df, x=col, nbins=35, color_discrete_sequence=[EMERALD])
                fig.update_layout(**base_layout(f"Distribution — {col}"))
                st.plotly_chart(fig, use_container_width=True)

        # Rain vs humidity scatter
        if humid_col and rain_col:
            fig3 = px.scatter(df, x=humid_col, y=rain_col,
                              color=target if target else None,
                              color_discrete_sequence=PALETTE, opacity=0.65)
            fig3.update_layout(**base_layout("Humidity vs Rainfall", height=380))
            st.plotly_chart(fig3, use_container_width=True)

    # ── Pressure & Wind ───────────────────────────────────────────────────────
    if _sel_visualization == "Pressure & Wind":
        pw_cols = [c for c in [pressure_col, wind_col] if c]
        if pw_cols:
            cs = st.columns(len(pw_cols))
            for i, col in enumerate(pw_cols):
                with cs[i]:
                    fig = px.violin(df, y=col, box=True, points="outliers",
                                    color_discrete_sequence=[AMBER])
                    fig.update_layout(**base_layout(f"Violin — {col}", height=350))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pressure/wind columns detected.")

        # Pressure vs Temperature
        if pressure_col and temp_col:
            fig_sp = px.scatter(df, x=pressure_col, y=temp_col,
                                color=target if target else None,
                                color_discrete_sequence=PALETTE, opacity=0.65)
            fig_sp.update_layout(**base_layout("Pressure vs Temperature", height=380))
            st.plotly_chart(fig_sp, use_container_width=True)

    # ── Seasonal Trends ───────────────────────────────────────────────────────
    if _sel_visualization == "Seasonal Trends":
        if season_col:
            st.markdown(f"**Seasonal grouping column:** `{season_col}`")
            if target:
                fig_s = px.histogram(df, x=season_col, color=target,
                                     barmode="group", color_discrete_sequence=PALETTE)
                fig_s.update_layout(**base_layout("Weather Type by Season/Period", height=380))
                st.plotly_chart(fig_s, use_container_width=True)

            if num_cols:
                trend_col = st.selectbox("Numeric feature for trend", num_cols, key="viz_trend")
                agg = df.groupby(season_col)[trend_col].mean().reset_index()
                fig_t = px.line(agg, x=season_col, y=trend_col,
                                markers=True, color_discrete_sequence=[BLUE])
                fig_t.update_layout(**base_layout(f"Average {trend_col} by {season_col}", height=360))
                st.plotly_chart(fig_t, use_container_width=True)

                # Grouped boxplot
                fig_gb = px.box(df, x=season_col, y=trend_col,
                                color=season_col, color_discrete_sequence=PALETTE)
                fig_gb.update_layout(**base_layout(f"{trend_col} by Season", height=360))
                st.plotly_chart(fig_gb, use_container_width=True)
        else:
            st.info("No date/season/month column detected. Using target-based grouping.")
            if target and num_cols:
                sel = st.selectbox("Numeric column to aggregate", num_cols, key="viz_sea2")
                agg = df.groupby(target)[sel].mean().reset_index()
                fig_a = px.bar(agg, x=target, y=sel,
                               color=target, color_discrete_sequence=PALETTE)
                fig_a.update_layout(**base_layout(f"Average {sel} by {target}", height=360))
                st.plotly_chart(fig_a, use_container_width=True)

    # ── Custom Charts ─────────────────────────────────────────────────────────
    if _sel_visualization == "Custom Charts":
        st.markdown("##### 🎨 Custom Chart Builder")
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_type = st.selectbox("Chart Type",
                                      ["Scatter", "Histogram", "Box", "Violin", "Line (grouped mean)"],
                                      key="viz_custom_type")
        with c2:
            x_col = st.selectbox("X Axis", df.columns.tolist(), key="viz_cx")
        with c3:
            y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns.tolist(),
                                 key="viz_cy")

        color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="viz_color")
        color_col = None if color_col == "None" else color_col

        try:
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                 color_discrete_sequence=PALETTE, opacity=0.7)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_col,
                                   color_discrete_sequence=PALETTE, nbins=40, opacity=0.85)
            elif chart_type == "Box":
                fig = px.box(df, x=color_col, y=y_col, color=color_col,
                             color_discrete_sequence=PALETTE)
            elif chart_type == "Violin":
                fig = px.violin(df, x=color_col, y=y_col, color=color_col,
                                box=True, color_discrete_sequence=PALETTE)
            elif chart_type == "Line (grouped mean)":
                agg = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.line(agg, x=x_col, y=y_col, markers=True,
                              color_discrete_sequence=[INDIGO])

            fig.update_layout(**base_layout(f"{chart_type}: {x_col} vs {y_col}", height=420))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")
