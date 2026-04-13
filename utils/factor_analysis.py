"""
Factor Analysis Module — sklearn FactorAnalysis with scree plot,
loadings heatmap, communalities, variance explained, and 80/20 train-test classification.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition   import FactorAnalysis
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.impute            import SimpleImputer
from sklearn.model_selection   import train_test_split
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from collections import Counter
from utils.data_loader         import detect_target_column
from utils.chart_style         import PALETTE, INDIGO, AMBER, BLUE, base_layout, heatmap_layout


def render_factor_analysis(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 🔶 Factor Analysis")

    num_cols = info["numeric_cols"]
    if len(num_cols) < 3:
        st.info("Need ≥ 3 numeric columns for Factor Analysis.")
        return

    target = detect_target_column(df)
    max_factors = min(len(num_cols) - 1, 15)

    with st.expander("⚙️ Factor Analysis Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            n_factors = st.slider("Number of Factors", 1, max_factors,
                                  min(3, max_factors), key="fa_n")
            rotation  = st.selectbox(
                "Rotation Method",
                ["varimax", "quartimax", None],
                format_func=lambda x: x if x else "None (no rotation)",
                key="fa_rot",
            )
        with c2:
            test_pct = st.slider("Test Size (%)", 10, 40, 20, step=5, key="fa_test") / 100
            model_choice = st.selectbox(
                "Classifier for FA evaluation",
                ["Logistic Regression", "Random Forest"],
                key="fa_model",
            )

    try:
        feat_cols = [c for c in num_cols if c != target] if target else num_cols

        # Impute & scale
        imp      = SimpleImputer(strategy="mean")
        X_imp    = imp.fit_transform(df[feat_cols])
        X_scaled = StandardScaler().fit_transform(X_imp)

        fa     = FactorAnalysis(n_components=n_factors, rotation=rotation, random_state=42)
        X_fa   = fa.fit_transform(X_scaled)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Scree Plot", "Factor Loadings", "Communalities",
             "Variance Explained", "🧪 Train/Test Model"]
        )

        # ── Scree Plot ────────────────────────────────────────────────────────
        with tab1:
            corr_matrix = np.corrcoef(X_scaled.T)
            eigenvalues = np.linalg.eigvalsh(corr_matrix)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]

            scree_df = pd.DataFrame({
                "Factor":     [f"F{i+1}" for i in range(len(eigenvalues))],
                "Eigenvalue": eigenvalues.round(4),
            })
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scree_df["Factor"], y=scree_df["Eigenvalue"],
                mode="lines+markers",
                marker=dict(size=8, color=AMBER, line=dict(color="white", width=2)),
                line=dict(color=AMBER, width=2.5),
                name="Eigenvalue",
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="#ef4444",
                          annotation_text="Kaiser criterion (λ=1)")
            fig.update_layout(**base_layout("Scree Plot — Factor Analysis", height=420))
            st.plotly_chart(fig, use_container_width=True)

            n_sig = int((eigenvalues > 1).sum())
            st.info(f"📌 **{n_sig} factor(s)** retain eigenvalue > 1 (Kaiser criterion).")
            st.dataframe(scree_df.head(max_factors), use_container_width=True)

        # ── Factor Loadings ───────────────────────────────────────────────────
        with tab2:
            loadings = pd.DataFrame(
                fa.components_.T,
                index=feat_cols,
                columns=[f"Factor {i+1}" for i in range(n_factors)],
            ).round(4)

            fig = px.imshow(loadings, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(**heatmap_layout("Factor Loadings Heatmap", height=500))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(loadings, use_container_width=True)
            st.download_button("⬇️ Download Loadings CSV",
                               loadings.to_csv().encode("utf-8"),
                               "fa_loadings.csv", "text/csv", key="fa_dl")

        # ── Communalities ─────────────────────────────────────────────────────
        with tab3:
            communalities = (fa.components_.T ** 2).sum(axis=1)
            comm_df = pd.DataFrame({
                "Variable":    feat_cols,
                "Communality": communalities.round(4),
            }).sort_values("Communality", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                st.dataframe(comm_df, use_container_width=True)
            with c2:
                fig = px.bar(comm_df, x="Variable", y="Communality",
                             color="Communality",
                             color_continuous_scale=[[0, "#fef9c3"], [1, AMBER]])
                fig.update_layout(**base_layout("Communalities per Variable", height=350))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

        # ── Variance Explained ────────────────────────────────────────────────
        with tab4:
            ss_loadings = (fa.components_.T ** 2).sum(axis=0)
            total_var   = len(feat_cols)
            pct_var     = ss_loadings / total_var * 100
            cum_pct     = np.cumsum(pct_var)

            var_df = pd.DataFrame({
                "Factor":       [f"Factor {i+1}" for i in range(n_factors)],
                "SS Loadings":  ss_loadings.round(4),
                "% Variance":   pct_var.round(2),
                "Cumulative %": cum_pct.round(2),
            })
            st.dataframe(var_df, use_container_width=True)

            fig = px.bar(var_df, x="Factor", y="% Variance",
                         color="% Variance",
                         color_continuous_scale=[[0, "#ede9fe"], [1, INDIGO]])
            fig.update_layout(**base_layout("Variance Explained per Factor"))
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)

        # ── Train/Test Model (80/20) ──────────────────────────────────────────
        with tab5:
            st.markdown("##### 🧪 Factor Analysis + Classification — Train/Test Evaluation")

            if not target or target not in df.columns:
                st.warning("⚠️ No target column detected. Cannot train a classification model. "
                           "Ensure your dataset has a categorical target column.")
                return

            # Prepare target
            y_raw = df[target].copy()
            valid_mask = y_raw.notna()
            X_valid = X_fa[valid_mask.values[:len(X_fa)]]
            y_valid = y_raw[valid_mask].reset_index(drop=True)

            if y_valid.nunique() < 2:
                st.warning("Need ≥ 2 classes in target column for classification.")
                return

            le = LabelEncoder()
            y_enc = le.fit_transform(y_valid.values)

            class_counts = Counter(y_enc)
            min_class_count = min(class_counts.values())
            use_stratify = min_class_count >= 2

            if not use_stratify:
                st.info("⚠️ Some classes have very few samples — stratified split disabled.")

            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_enc,
                test_size=test_pct,
                random_state=42,
                stratify=y_enc if use_stratify else None,
            )

            st.markdown(f"""
            <div class="wl-info">
              📊 <b>Split:</b> {len(X_train):,} train ({(1-test_pct)*100:.0f}%) · 
              {len(X_test):,} test ({test_pct*100:.0f}%) · 
              {n_factors} factors · {y_valid.nunique()} classes
            </div>
            """, unsafe_allow_html=True)

            # Train model
            with st.spinner(f"Training {model_choice} on Factor Analysis features…"):
                if model_choice == "Logistic Regression":
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                else:
                    clf = RandomForestClassifier(n_estimators=100, random_state=42)

                clf.fit(X_train, y_train)
                y_pred_train = clf.predict(X_train)
                y_pred_test  = clf.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc  = accuracy_score(y_test, y_pred_test)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train Accuracy", f"{train_acc:.4f}")
            m2.metric("Test Accuracy",  f"{test_acc:.4f}")
            m3.metric("Precision (macro)", f"{precision_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")
            m4.metric("F1 Score (macro)",  f"{f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}")

            # Accuracy comparison bar
            acc_df = pd.DataFrame({
                "Set": ["Training", "Testing"],
                "Accuracy": [train_acc, test_acc],
            })
            fig_acc = px.bar(acc_df, x="Set", y="Accuracy", color="Set",
                             color_discrete_sequence=[AMBER, "#8b5cf6"],
                             text=acc_df["Accuracy"].apply(lambda x: f"{x:.4f}"))
            fig_acc.update_layout(**base_layout("FA Accuracy — Train vs Test", height=320))
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
            fig_cm.update_layout(**heatmap_layout(
                "Confusion Matrix — FA + " + model_choice, height=450
            ))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Download predictions
            results_df = pd.DataFrame({
                "Actual": le.inverse_transform(y_test),
                "Predicted": le.inverse_transform(y_pred_test),
            })
            for i in range(min(n_factors, 5)):
                results_df[f"Factor_{i+1}"] = X_test[:, i]
            st.download_button(
                "⬇️ Download FA Predictions CSV",
                results_df.to_csv(index=False).encode("utf-8"),
                "fa_predictions.csv", "text/csv", key="fa_pred_dl",
            )

    except Exception as exc:
        st.error(f"Factor Analysis error: {exc}")
