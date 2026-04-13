"""
LDA Analysis Module — sklearn LinearDiscriminantAnalysis for weather classification
with 80/20 train-test split, classification metrics, and confusion matrix.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing          import LabelEncoder, StandardScaler
from sklearn.impute                  import SimpleImputer
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from collections import Counter
from utils.data_loader              import detect_target_column
from utils.chart_style              import PALETTE, INDIGO, BLUE, base_layout, heatmap_layout


def render_lda(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 🟣 Linear Discriminant Analysis (LDA)")

    num_cols = info["numeric_cols"]
    if len(num_cols) < 2:
        st.info("Need ≥ 2 numeric columns for LDA.")
        return

    target = detect_target_column(df)
    if target is None:
        target = st.selectbox("Select weather type / target column", df.columns.tolist(), key="lda_tgt")
    else:
        st.info(f"🎯 Target column detected: **{target}**")

    if target not in df.columns:
        st.warning("Target column not found.")
        return

    n_classes = df[target].nunique()
    if n_classes < 2:
        st.warning("Need ≥ 2 classes for LDA.")
        return

    feat_cols = [c for c in num_cols if c != target]
    n_comp    = min(n_classes - 1, len(feat_cols))
    if n_comp < 1:
        st.warning("Not enough components. Add more features or classes.")
        return

    with st.expander("⚙️ LDA Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            test_pct = st.slider("Test Size (%)", 10, 40, 20, step=5, key="lda_test") / 100
        with c2:
            solver = st.selectbox("LDA Solver", ["svd", "lsqr", "eigen"], key="lda_solver")

    try:
        sub = df[feat_cols + [target]].copy()
        # Impute numeric features
        imp = SimpleImputer(strategy="mean")
        sub[feat_cols] = imp.fit_transform(sub[feat_cols])

        # Drop rows with NaN target
        sub = sub.dropna(subset=[target])

        n_samples      = len(sub)
        actual_classes = sub[target].nunique()

        if n_samples <= actual_classes:
            st.warning(
                f"⚠️ LDA needs more rows ({n_samples}) than classes ({actual_classes}). "
                "Run Preprocessing → Imputation first."
            )
            return

        n_comp = min(actual_classes - 1, len(feat_cols))

        X  = sub[feat_cols].values
        X  = StandardScaler().fit_transform(X)

        le = LabelEncoder()
        y  = le.fit_transform(sub[target].values)

        # ── Train-Test Split ──────────────────────────────────────────────────
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        use_stratify = min_class_count >= 2

        if not use_stratify:
            st.info("⚠️ Some classes have very few samples — stratified split disabled.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_pct,
            random_state=42,
            stratify=y if use_stratify else None,
        )

        # ── Fit LDA on Training Data ──────────────────────────────────────────
        lda_solver = solver if solver != "svd" else "svd"
        lda = LinearDiscriminantAnalysis(
            n_components=n_comp,
            solver=lda_solver,
        )
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda  = lda.transform(X_test)

        # Also transform full dataset for visualization
        X_full_lda = lda.transform(X)

        lda_df = pd.DataFrame(
            X_full_lda,
            columns=[f"LD{i+1}" for i in range(n_comp)],
        )
        lda_df[target] = le.inverse_transform(y)

        st.markdown(f"""
        <div class="wl-info">
          📊 <b>Split:</b> {len(X_train):,} train ({(1-test_pct)*100:.0f}%) · 
          {len(X_test):,} test ({test_pct*100:.0f}%) · 
          {n_comp} LDA component(s) · {actual_classes} classes
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Class Separation", "Explained Variance", "3D View",
             "🧪 Train/Test Results", "Transformed Data"]
        )

        # ── Class Separation ──────────────────────────────────────────────────
        with tab1:
            if n_comp >= 2:
                fig = px.scatter(lda_df, x="LD1", y="LD2", color=target,
                                 opacity=0.75, color_discrete_sequence=PALETTE)
                fig.update_layout(**base_layout("LDA — Class Separation (LD1 vs LD2)", height=460))
            else:
                fig = px.histogram(lda_df, x="LD1", color=target,
                                   barmode="overlay", opacity=0.7,
                                   color_discrete_sequence=PALETTE)
                fig.update_layout(**base_layout("LDA — Class Separation (LD1)", height=400))
            st.plotly_chart(fig, use_container_width=True)

        # ── Explained Variance ────────────────────────────────────────────────
        with tab2:
            evr    = lda.explained_variance_ratio_
            ev_df  = pd.DataFrame({
                "Component":            [f"LD{i+1}" for i in range(len(evr))],
                "Explained Variance %": (evr * 100).round(2),
                "Cumulative %":         (np.cumsum(evr) * 100).round(2),
            })
            st.dataframe(ev_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(ev_df, x="Component", y="Explained Variance %",
                             color="Explained Variance %",
                             color_continuous_scale=[[0, "#ede9fe"], [1, "#8b5cf6"]])
                fig.update_layout(**base_layout("LDA Explained Variance"))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.line(ev_df, x="Component", y="Cumulative %",
                               markers=True, color_discrete_sequence=[INDIGO])
                fig2.add_hline(y=95, line_dash="dash", line_color="#f43f5e",
                               annotation_text="95% threshold")
                fig2.update_layout(**base_layout("Cumulative Explained Variance"))
                st.plotly_chart(fig2, use_container_width=True)

        # ── 3D View ───────────────────────────────────────────────────────────
        with tab3:
            if n_comp >= 3:
                fig3d = px.scatter_3d(lda_df, x="LD1", y="LD2", z="LD3",
                                       color=target, opacity=0.75,
                                       color_discrete_sequence=PALETTE)
                fig3d.update_layout(paper_bgcolor="white", margin=dict(t=50, b=20),
                                    height=560)
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.info("3D view requires ≥ 3 LDA components (need more classes or features).")

        # ── Train/Test Results ────────────────────────────────────────────────
        with tab4:
            st.markdown("##### 🧪 LDA Classification — Train/Test Evaluation")

            # LDA itself is a classifier — use its predict method
            y_pred_train = lda.predict(X_train)
            y_pred_test  = lda.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc  = accuracy_score(y_test, y_pred_test)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train Accuracy", f"{train_acc:.4f}")
            m2.metric("Test Accuracy",  f"{test_acc:.4f}")
            m3.metric("Precision (macro)", f"{precision_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")
            m4.metric("F1 Score (macro)",  f"{f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")

            m5, m6 = st.columns(2)
            m5.metric("Recall (macro)", f"{recall_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")
            m6.metric("Components Used", n_comp)

            # Accuracy comparison bar
            acc_df = pd.DataFrame({
                "Set": ["Training", "Testing"],
                "Accuracy": [train_acc, test_acc],
            })
            fig_acc = px.bar(acc_df, x="Set", y="Accuracy", color="Set",
                             color_discrete_sequence=[BLUE, "#8b5cf6"],
                             text=acc_df["Accuracy"].apply(lambda x: f"{x:.4f}"))
            fig_acc.update_layout(**base_layout("LDA Accuracy — Train vs Test", height=320))
            fig_acc.update_traces(textposition="outside")
            st.plotly_chart(fig_acc, use_container_width=True)

            # Classification report
            st.markdown("##### 📋 Classification Report (Test Set)")
            report = classification_report(
                y_test, y_pred_test,
                target_names=le.classes_.astype(str),
                output_dict=True, zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

            # Confusion matrix
            st.markdown("##### 🔲 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred_test)
            cm_df = pd.DataFrame(
                cm,
                index=[f"True: {c}" for c in le.classes_],
                columns=[f"Pred: {c}" for c in le.classes_],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto",
                               color_continuous_scale="Blues")
            fig_cm.update_layout(**heatmap_layout("Confusion Matrix — LDA Classifier", height=450))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Download predictions
            results_df = pd.DataFrame({
                "Actual": le.inverse_transform(y_test),
                "Predicted": le.inverse_transform(y_pred_test),
            })
            for i in range(min(n_comp, 5)):
                results_df[f"LD{i+1}"] = X_test_lda[:, i]
            st.download_button(
                "⬇️ Download LDA Predictions CSV",
                results_df.to_csv(index=False).encode("utf-8"),
                "lda_predictions.csv", "text/csv", key="lda_pred_dl",
            )

        # ── Transformed Data ──────────────────────────────────────────────────
        with tab5:
            st.dataframe(lda_df.head(40), use_container_width=True)
            st.download_button("⬇️ Download LDA CSV",
                               lda_df.to_csv(index=False).encode("utf-8"),
                               "lda_transformed.csv", "text/csv", key="lda_dl")

    except Exception as exc:
        st.error(f"LDA error: {exc}")
