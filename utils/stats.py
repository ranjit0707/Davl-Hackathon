"""
Statistics Module — mean, median, mode, std, variance, skewness, kurtosis,
normality tests, grouped statistics, and 80/20 train-test model evaluation.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.model_selection  import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from collections import Counter
from utils.chart_style import PALETTE, BLUE, AMBER, EMERALD, INDIGO, base_layout, heatmap_layout


def render_stats(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 📐 Statistical Summary")

    num_cols = info["numeric_cols"]
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    _sel_stats = st.radio('Select View', ["Descriptive Stats", "Distribution Tests", "Grouped Statistics", "🧪 Model Comparison (80/20)"], horizontal=True, label_visibility='collapsed', key='radio_stats_9d3e')

    # ── Descriptive ───────────────────────────────────────────────────────────
    if _sel_stats == "Descriptive Stats":
        desc = df[num_cols].describe().T
        desc["variance"]  = df[num_cols].var()
        desc["skewness"]  = df[num_cols].skew()
        desc["kurtosis"]  = df[num_cols].kurtosis()
        desc["mode"]      = df[num_cols].mode().iloc[0] if not df[num_cols].mode().empty else np.nan

        # Rename describe columns
        desc = desc.rename(columns={
            "count": "Count", "mean": "Mean", "std": "Std Dev",
            "min": "Min", "25%": "Q1", "50%": "Median",
            "75%": "Q3", "max": "Max",
        })

        st.dataframe(desc.round(4), use_container_width=True)

        # Highlight metrics
        c1, c2, c3 = st.columns(3)
        col_sel = st.selectbox("Inspect column", num_cols, key="stats_col")
        col     = df[col_sel].dropna()

        c1.metric("Mean",     round(col.mean(), 4))
        c2.metric("Median",   round(col.median(), 4))
        c3.metric("Std Dev",  round(col.std(), 4))
        c4, c5, c6 = st.columns(3)
        c4.metric("Variance", round(col.var(), 4))
        c5.metric("Skewness", round(col.skew(), 4))
        c6.metric("Kurtosis", round(col.kurtosis(), 4))

        # Skewness bar chart
        skew_df = pd.DataFrame({
            "Column":   num_cols,
            "Skewness": [df[c].skew() for c in num_cols],
        }).sort_values("Skewness", key=abs, ascending=False)

        fig = px.bar(skew_df, x="Column", y="Skewness",
                     color="Skewness",
                     color_continuous_scale="RdBu_r")
        fig.add_hline(y=0, line_color="#64748b", line_dash="solid", line_width=1)
        fig.add_hline(y=1,  line_dash="dash", line_color=AMBER, annotation_text="+1 (high skew)")
        fig.add_hline(y=-1, line_dash="dash", line_color=AMBER, annotation_text="-1 (high skew)")
        fig.update_layout(**base_layout("Skewness by Column", height=360))
        st.plotly_chart(fig, use_container_width=True)

        # Kurtosis bar chart
        kurt_df = pd.DataFrame({
            "Column":   num_cols,
            "Kurtosis": [df[c].kurtosis() for c in num_cols],
        }).sort_values("Kurtosis", ascending=False)

        fig2 = px.bar(kurt_df, x="Column", y="Kurtosis",
                      color="Kurtosis",
                      color_continuous_scale=[[0, "#ede9fe"], [1, "#8b5cf6"]])
        fig2.add_hline(y=3, line_dash="dash", line_color="#ef4444",
                       annotation_text="Normal kurtosis = 3")
        fig2.update_layout(**base_layout("Kurtosis by Column", height=360))
        st.plotly_chart(fig2, use_container_width=True)

        # Download
        st.download_button(
            "⬇️ Download Stats CSV",
            desc.round(4).to_csv().encode("utf-8"),
            "statistical_summary.csv", "text/csv", key="stats_dl",
        )

    # ── Distribution Tests ────────────────────────────────────────────────────
    if _sel_stats == "Distribution Tests":
        st.markdown("##### Normality Tests (Shapiro-Wilk)")
        norm_results = []
        for col in num_cols:
            s = df[col].dropna()
            if len(s) > 3:
                sample = s if len(s) <= 5000 else s.sample(5000, random_state=42)
                try:
                    stat, p = scipy_stats.shapiro(sample)
                    norm_results.append({
                        "Column":    col,
                        "W-stat":    round(stat, 4),
                        "p-value":   round(p, 6),
                        "Normal?":   "✅ Yes" if p > 0.05 else "❌ No",
                    })
                except Exception:
                    pass

        if norm_results:
            norm_df = pd.DataFrame(norm_results)
            st.dataframe(norm_df, use_container_width=True)
            n_normal = sum(1 for r in norm_results if "Yes" in r["Normal?"])
            st.info(f"📌 {n_normal}/{len(norm_results)} columns are normally distributed (p > 0.05).")

    # ── Grouped Statistics ────────────────────────────────────────────────────
    if _sel_stats == "Grouped Statistics":
        from utils.data_loader import detect_target_column
        target = detect_target_column(df)
        if not target:
            st.info("No categorical target column detected for grouping.")
        else:
            st.markdown(f"**Grouping by:** `{target}`")
            sel_num = st.multiselect("Select numeric columns",
                                      num_cols[:8], default=num_cols[:4], key="stats_grp")
            if sel_num:
                grouped = df.groupby(target)[sel_num].agg(["mean", "std", "median"]).round(3)
                grouped.columns = ["_".join(c) for c in grouped.columns]
                st.dataframe(grouped, use_container_width=True)

                # Grouped bar for mean
                for col in sel_num[:3]:
                    mean_col = f"{col}_mean"
                    if mean_col in grouped.columns:
                        fig = px.bar(grouped.reset_index(), x=target, y=mean_col,
                                     color=target, color_discrete_sequence=PALETTE)
                        fig.update_layout(**base_layout(f"Mean {col} by {target}", height=320))
                        st.plotly_chart(fig, use_container_width=True)

    # ── Model Comparison (80/20 Train-Test) ───────────────────────────────────
    if _sel_stats == "🧪 Model Comparison (80/20)":
        st.markdown("##### 🧪 Multi-Model Comparison — 80/20 Train-Test Split")

        from utils.data_loader import detect_target_column
        target = detect_target_column(df)

        if not target or target not in df.columns:
            st.warning("⚠️ No target column detected. Cannot train models. "
                       "Ensure your dataset has a categorical target column.")
            return

        feat_cols = [c for c in num_cols if c != target]
        if len(feat_cols) < 1:
            st.warning("Need at least 1 numeric feature column.")
            return

        with st.expander("⚙️ Model Settings", expanded=True):
            mc1, mc2 = st.columns(2)
            with mc1:
                test_pct = st.slider("Test Size (%)", 10, 40, 20, step=5, key="stats_test") / 100
            with mc2:
                selected_models = st.multiselect(
                    "Models to compare",
                    ["Logistic Regression", "Random Forest", "Gradient Boosting", "KNN", "SVM"],
                    default=["Logistic Regression", "Random Forest", "KNN"],
                    key="stats_models",
                )

        if not selected_models:
            st.info("Select at least one model to compare.")
            return

        try:
            sub = df[feat_cols + [target]].copy()
            imp = SimpleImputer(strategy="mean")
            sub[feat_cols] = imp.fit_transform(sub[feat_cols])
            sub = sub.dropna(subset=[target])

            X = StandardScaler().fit_transform(sub[feat_cols].values)
            le = LabelEncoder()
            y = le.fit_transform(sub[target].values)

            class_counts = Counter(y)
            min_class_count = min(class_counts.values())
            use_stratify = min_class_count >= 2

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_pct, random_state=42,
                stratify=y if use_stratify else None,
            )

            st.markdown(f"""
            <div class="wl-info">
              📊 <b>Split:</b> {len(X_train):,} train ({(1-test_pct)*100:.0f}%) · 
              {len(X_test):,} test ({test_pct*100:.0f}%) · 
              {len(feat_cols)} features · {len(le.classes_)} classes
            </div>
            """, unsafe_allow_html=True)

            # Model definitions
            model_map = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
                "KNN":                 KNeighborsClassifier(n_neighbors=5),
                "SVM":                 SVC(kernel="rbf", random_state=42),
            }

            results = []
            best_model_name = None
            best_test_acc = -1
            best_y_pred = None

            with st.spinner("Training models…"):
                for name in selected_models:
                    clf = model_map[name]
                    clf.fit(X_train, y_train)
                    y_pred_tr = clf.predict(X_train)
                    y_pred_te = clf.predict(X_test)

                    tr_acc = accuracy_score(y_train, y_pred_tr)
                    te_acc = accuracy_score(y_test, y_pred_te)
                    prec   = precision_score(y_test, y_pred_te, average="macro", zero_division=0)
                    rec    = recall_score(y_test, y_pred_te, average="macro", zero_division=0)
                    f1     = f1_score(y_test, y_pred_te, average="macro", zero_division=0)

                    results.append({
                        "Model":              name,
                        "Train Accuracy":     round(tr_acc, 4),
                        "Test Accuracy":      round(te_acc, 4),
                        "Precision (macro)":  round(prec, 4),
                        "Recall (macro)":     round(rec, 4),
                        "F1 Score (macro)":   round(f1, 4),
                    })

                    if te_acc > best_test_acc:
                        best_test_acc = te_acc
                        best_model_name = name
                        best_y_pred = y_pred_te

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Highlight best model
            st.markdown(f"""
            <div class="wl-success">
              🏆 <b>Best Model:</b> {best_model_name} — Test Accuracy: <b>{best_test_acc:.4f}</b>
            </div>
            """, unsafe_allow_html=True)

            # Accuracy comparison chart
            fig_comp = px.bar(
                results_df, x="Model", y=["Train Accuracy", "Test Accuracy"],
                barmode="group",
                color_discrete_sequence=[BLUE, EMERALD],
                text_auto=".4f",
            )
            fig_comp.update_layout(**base_layout("Model Accuracy Comparison — Train vs Test", height=400))
            st.plotly_chart(fig_comp, use_container_width=True)

            # F1 score comparison
            fig_f1 = px.bar(
                results_df, x="Model", y="F1 Score (macro)",
                color="F1 Score (macro)",
                color_continuous_scale=[[0, "#fef3c7"], [1, AMBER]],
                text_auto=".4f",
            )
            fig_f1.update_layout(**base_layout("F1 Score Comparison (macro)", height=340))
            fig_f1.update_traces(marker_line_width=0)
            st.plotly_chart(fig_f1, use_container_width=True)

            # Best model — classification report & confusion matrix
            st.markdown(f"##### 📋 Classification Report — {best_model_name} (Test Set)")
            report = classification_report(
                y_test, best_y_pred,
                target_names=le.classes_.astype(str),
                output_dict=True, zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

            st.markdown(f"##### 🔲 Confusion Matrix — {best_model_name}")
            cm = confusion_matrix(y_test, best_y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[f"True: {c}" for c in le.classes_],
                columns=[f"Pred: {c}" for c in le.classes_],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto",
                               color_continuous_scale="Blues")
            fig_cm.update_layout(**heatmap_layout(
                f"Confusion Matrix — {best_model_name}", height=450
            ))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Download
            dl_df = pd.DataFrame({
                "Actual":    le.inverse_transform(y_test),
                "Predicted": le.inverse_transform(best_y_pred),
            })
            st.download_button(
                "⬇️ Download Best Model Predictions CSV",
                dl_df.to_csv(index=False).encode("utf-8"),
                "model_predictions.csv", "text/csv", key="stats_pred_dl",
            )

        except Exception as exc:
            st.error(f"Model comparison error: {exc}")
