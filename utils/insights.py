"""
Weather Pattern Insights Module — auto-generates climate trend insights,
seasonal analysis, correlation findings, and analysis recommendations.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import VarianceThreshold
from utils.data_loader          import detect_target_column, convert_df_to_csv
from utils.chart_style          import PALETTE, BLUE, EMERALD, AMBER, INDIGO, base_layout

# Weather keyword categories
TEMP_KW     = ["temp", "temperature"]
HUMID_KW    = ["humidity", "humid"]
RAIN_KW     = ["rain", "rainfall", "precipitation"]
PRESSURE_KW = ["pressure", "baro"]
WIND_KW     = ["wind"]
SEASON_KW   = ["season", "month", "date", "year", "quarter"]


def _find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None


def render_insights(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("### 💡 Auto-Generated Climate Insights")

    num_cols = info["numeric_cols"]
    cat_cols = info["categorical_cols"]
    n_rows, n_cols = info["rows"], info["cols"]
    target = detect_target_column(df)

    # Detect key weather columns
    temp_col     = _find_col(df, TEMP_KW)
    humid_col    = _find_col(df, HUMID_KW)
    rain_col     = _find_col(df, RAIN_KW)
    pressure_col = _find_col(df, PRESSURE_KW)
    wind_col     = _find_col(df, WIND_KW)
    season_col   = _find_col(df, SEASON_KW)

    # ── Build insight lists ───────────────────────────────────────────────────
    observations  = []
    warnings      = []
    climate_notes = []
    suggestions   = []

    # Size
    observations.append(f"📦 Dataset: **{n_rows:,} rows × {n_cols} cols** ({info['memory_mb']} MB)")

    # Missing
    miss = [(c, v) for c, v in info["null_pct"].items() if v > 0]
    if miss:
        high = [(c, v) for c, v in miss if v > 30]
        warnings.append(f"⚠️ **{len(miss)} column(s)** have missing values.")
        if high:
            warnings.append(f"❌ High missing (>30%): {[c for c, _ in high]}")
            suggestions.append("Consider **dropping** columns with >50% missing or use **KNN imputation**.")
        else:
            suggestions.append("Use **SimpleImputer (median)** — more robust for weather data with outliers.")
    else:
        observations.append("✅ **No missing values** found.")

    # Duplicates
    dup = int(df.duplicated().sum())
    if dup > 0:
        warnings.append(f"⚠️ **{dup} duplicate rows** ({round(dup/n_rows*100,2)}%). Drop before modelling.")
    else:
        observations.append("✅ **No duplicate rows** detected.")

    # Target column
    if target:
        n_cls = df[target].nunique()
        classes = df[target].value_counts()
        observations.append(f"🎯 Target column: **{target}** with **{n_cls} classes**: "
                            f"{', '.join(str(c) for c in classes.index[:6])}")
        if n_cls >= 2:
            ratio = classes.max() / classes.min()
            if ratio > 5:
                warnings.append(f"⚖️ **Severe class imbalance** in '{target}' (ratio={ratio:.1f}). "
                                "Use SMOTE or class_weight='balanced'.")
            elif ratio > 2:
                warnings.append(f"⚖️ **Moderate class imbalance** in '{target}' (ratio={ratio:.1f}).")
            else:
                observations.append(f"✅ Classes in '{target}' are roughly balanced.")

    # Temperature insights
    if temp_col:
        s = df[temp_col].dropna()
        mean_t, std_t = s.mean(), s.std()
        climate_notes.append(
            f"🌡️ **Temperature ({temp_col})**: Mean = **{mean_t:.1f}**, Std = {std_t:.1f}. "
            f"Range: [{s.min():.1f}, {s.max():.1f}]."
        )
        if abs(s.skew()) > 1:
            climate_notes.append(f"📉 Temperature is **{'positively' if s.skew()>0 else 'negatively'} skewed** "
                                   "(|skew| > 1). Consider log/Box-Cox transform.")

    # Humidity insights
    if humid_col:
        s = df[humid_col].dropna()
        mean_h = s.mean()
        climate_notes.append(
            f"💧 **Humidity ({humid_col})**: Mean = **{mean_h:.1f}%**. "
            f"High humidity months likely indicate rainy/cloudy patterns."
        )

    # Rainfall insights
    if rain_col:
        s = df[rain_col].dropna()
        zero_pct = round((s == 0).sum() / len(s) * 100, 1)
        climate_notes.append(
            f"🌧️ **Rainfall ({rain_col})**: Mean = {s.mean():.2f}, "
            f"Zero-rain days = **{zero_pct}%** of record."
        )
        if zero_pct > 60:
            climate_notes.append("☀️ Majority of days are dry. Dataset likely covers an arid or semi-arid region.")

    # Pressure insights
    if pressure_col:
        s = df[pressure_col].dropna()
        climate_notes.append(
            f"🔵 **Pressure ({pressure_col})**: Mean = {s.mean():.1f} hPa. "
            "Low pressure typically associates with storms/rain; high = clear weather."
        )

    # Seasonal insights
    if season_col and num_cols:
        climate_notes.append(
            f"📅 **Seasonal column detected:** `{season_col}`. "
            "Seasonal trend analysis available in the Visualizations tab."
        )

    # Correlation
    high_corr_pairs = []
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack()
        high  = pairs[pairs > 0.80]
        if not high.empty:
            high_corr_pairs = [(f"{a} ↔ {b}", round(v, 3)) for (a, b), v in high.items()]
            warnings.append(f"🔗 **{len(high_corr_pairs)} highly correlated pair(s)** (|r| > 0.80):")
            for pair, val in high_corr_pairs[:5]:
                warnings.append(f"   • {pair} = {val}")
            suggestions.append("Remove or merge **highly correlated features** to reduce multicollinearity. "
                                "PCA can collapse them into principal components.")

    # Outliers
    high_out_cols = []
    for col in num_cols:
        s = df[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = ((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum()
        if n_out / len(s) > 0.05:
            high_out_cols.append((col, round(n_out / len(s) * 100, 1)))
    if high_out_cols:
        warnings.append(f"📈 Columns with >5% outliers: {high_out_cols}")
        suggestions.append("Use **RobustScaler** or **Winsorization** for outlier-heavy weather features.")
    else:
        observations.append("✅ Outlier levels are within acceptable range (<5% per column).")

    # Skewness
    skewed = [(c, round(df[c].skew(), 2)) for c in num_cols if abs(df[c].skew()) > 1]
    if skewed:
        observations.append(f"📉 **{len(skewed)} skewed column(s)** (|skew|>1): "
                            f"{[c for c, _ in skewed[:5]]}")
        suggestions.append("Apply **log**, **sqrt**, or **Box-Cox** transformation to normalize skewed features.")

    # High dimensionality
    if n_cols > 30:
        suggestions.append(f"⚡ High dimensionality ({n_cols} cols). Use **PCA** or **SelectKBest** for reduction.")

    # Model suggestion
    if target:
        suggestions.append(
            "🤖 For **weather classification**, recommended models: "
            "**Random Forest**, **Gradient Boosting (XGBoost)**, or **SVM**. "
            "For regression tasks (predicting temperature/rainfall): **Ridge**, **SVR**, **LightGBM**."
        )
        suggestions.append(
            "📊 Run **PCA** to understand variance structure, **LDA** for class separation, "
            "**KMeans** to detect hidden weather clusters, and **Factor Analysis** to find latent climate factors."
        )

    # ── Render UI ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Observations", "⚠️ Warnings", "🌍 Climate Notes", "🛠️ Recommendations"]
    )

    with tab1:
        for item in observations:
            st.markdown(f"""
            <div class="insight-item insight-good">
              {item}
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        if warnings:
            for item in warnings:
                st.markdown(f"""
                <div class="insight-item insight-warn">
                  {item}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="wl-success">✅ No critical warnings found.</div>',
                        unsafe_allow_html=True)

    with tab3:
        if climate_notes:
            for item in climate_notes:
                st.markdown(f"""
                <div class="insight-item">
                  {item}
                </div>
                """, unsafe_allow_html=True)

            # Visualize if seasonal col found
            if season_col and num_cols:
                st.markdown("---")
                st.markdown("##### 📅 Quick Seasonal Trend")
                trend_col = st.selectbox("Pick numeric feature", num_cols, key="ins_trend")
                agg = df.groupby(season_col)[trend_col].mean().reset_index()
                fig = px.line(agg, x=season_col, y=trend_col,
                              markers=True, color_discrete_sequence=[BLUE])
                fig.update_layout(**base_layout(f"Avg {trend_col} by {season_col}", height=340))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No weather columns auto-detected. Rename columns with keywords like 'temperature', 'humidity', etc.")

    with tab4:
        for item in suggestions:
            st.markdown(f"""
            <div class="insight-item">
              {item}
            </div>
            """, unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown('<hr style="border-top:1px solid #e2e8f0;margin:18px 0">', unsafe_allow_html=True)
    all_text = (
        "## Weather Pattern Analysis — Auto Insights\n\n"
        "### Observations\n" + "\n".join(f"- {i}" for i in observations) + "\n\n"
        "### Warnings\n" + "\n".join(f"- {w}" for w in warnings) + "\n\n"
        "### Climate Notes\n" + "\n".join(f"- {c}" for c in climate_notes) + "\n\n"
        "### Recommendations\n" + "\n".join(f"- {s}" for s in suggestions)
    )
    st.download_button(
        "⬇️ Export Insights (.txt)",
        data=all_text.encode("utf-8"),
        file_name="weather_insights.txt",
        mime="text/plain",
        key="ins_dl",
    )
