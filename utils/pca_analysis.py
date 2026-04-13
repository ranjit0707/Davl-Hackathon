"""
PCA Analysis Module — dimensionality reduction with explained variance,
scree plot, scatter, loadings heatmap, and 80/20 train-test classification.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from utils.data_loader       import detect_target_column
from utils.chart_style       import PALETTE, INDIGO, BLUE, base_layout, heatmap_layout


def render_pca(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 🔵 Principal Component Analysis (PCA)")

    num_cols = info["numeric_cols"]
    if len(num_cols) < 2:
        st.info("Need ≥ 2 numeric columns for PCA.")
        return

    target = detect_target_column(df)

    with st.expander("⚙️ PCA Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            n_comp = st.slider(
                "Number of Components", 2, min(len(num_cols), 20),
                min(5, len(num_cols)), key="pca_n",
            )
            use_scaler = st.checkbox("Apply StandardScaler", value=True, key="pca_sc")
        with c2:
            test_pct = st.slider("Test Size (%)", 10, 40, 20, step=5, key="pca_test") / 100
            model_choice = st.selectbox(
                "Classifier for PCA evaluation",
                ["Logistic Regression", "Random Forest"],
                key="pca_model",
            )

    try:
        feat_cols = [c for c in num_cols if c != target] if target else num_cols
        X   = df[feat_cols].copy()
        imp = SimpleImputer(strategy="mean")
        X_imp = imp.fit_transform(X)
        if use_scaler:
            X_imp = StandardScaler().fit_transform(X_imp)

        n_comp = min(n_comp, X_imp.shape[0], X_imp.shape[1])
        pca    = PCA(n_components=n_comp, random_state=42)
        X_pca  = pca.fit_transform(X_imp)
        evr    = pca.explained_variance_ratio_
        ev     = pca.explained_variance_

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Explained Variance", "Scree Plot", "PCA Scatter", "Loadings Heatmap",
             "🧪 Train/Test Model"]
        )

        # ── Explained Variance ────────────────────────────────────────────────
        with tab1:
            cumvar = np.cumsum(evr)
            ev_df  = pd.DataFrame({
                "Component":            [f"PC{i+1}" for i in range(n_comp)],
                "Explained Variance %": (evr * 100).round(2),
                "Cumulative %":         (cumvar * 100).round(2),
            })
            st.dataframe(ev_df, use_container_width=True)

            n95 = int(np.searchsorted(cumvar, 0.95)) + 1
            st.info(f"📌 **{n95} component(s)** explain ≥ 95% of total variance.")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(ev_df, x="Component", y="Explained Variance %",
                             color="Explained Variance %",
                             color_continuous_scale=[[0, "#dbeafe"], [1, BLUE]])
                fig.update_layout(**base_layout("Individual Explained Variance"))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.line(ev_df, x="Component", y="Cumulative %",
                               markers=True, color_discrete_sequence=[INDIGO])
                fig2.add_hline(y=95, line_dash="dash", line_color="#ef4444",
                               annotation_text="95% threshold")
                fig2.update_layout(**base_layout("Cumulative Explained Variance"))
                st.plotly_chart(fig2, use_container_width=True)

        # ── Scree Plot ────────────────────────────────────────────────────────
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(n_comp)], y=ev,
                mode="lines+markers",
                marker=dict(size=8, color=INDIGO, line=dict(color="white", width=2)),
                line=dict(color=INDIGO, width=2.5),
                name="Eigenvalue",
            ))
            fig.add_hline(y=1, line_dash="dash", line_color="#f43f5e",
                          annotation_text="Kaiser criterion (λ=1)")
            fig.update_layout(**base_layout("Scree Plot — Eigenvalues", height=420))
            st.plotly_chart(fig, use_container_width=True)

        # ── PCA Scatter ───────────────────────────────────────────────────────
        with tab3:
            n_show = min(n_comp, 10)
            pca_df = pd.DataFrame(X_pca[:, :n_show],
                                   columns=[f"PC{i+1}" for i in range(n_show)])
            if target and target in df.columns:
                pca_df[target] = df[target].values[:len(pca_df)]
                color_arg = target
            else:
                color_arg = None

            c1, c2 = st.columns(2)
            pc_x = c1.selectbox("X Component", pca_df.columns[:n_show].tolist(), key="pca_cx")
            pc_y = c2.selectbox("Y Component", pca_df.columns[:n_show].tolist(),
                                index=min(1, n_show-1), key="pca_cy")

            fig = px.scatter(pca_df, x=pc_x, y=pc_y, color=color_arg,
                             opacity=0.7, color_discrete_sequence=PALETTE)
            fig.update_layout(**base_layout(f"PCA — {pc_x} vs {pc_y}", height=450))
            st.plotly_chart(fig, use_container_width=True)

            if n_comp >= 3:
                fig3d = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3",
                                       color=color_arg, opacity=0.65,
                                       color_discrete_sequence=PALETTE)
                fig3d.update_layout(paper_bgcolor="white", margin=dict(t=50, b=20),
                                    height=500)
                st.plotly_chart(fig3d, use_container_width=True)

        # ── Loadings ──────────────────────────────────────────────────────────
        with tab4:
            loadings = pd.DataFrame(
                pca.components_.T,
                index=feat_cols,
                columns=[f"PC{i+1}" for i in range(n_comp)],
            ).round(4)
            fig = px.imshow(loadings, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig.update_layout(**heatmap_layout("PCA Loadings Heatmap", height=520))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(loadings, use_container_width=True)
            st.download_button("⬇️ Download Loadings CSV",
                               loadings.to_csv().encode("utf-8"),
                               "pca_loadings.csv", "text/csv", key="pca_dl")

        # ── Train/Test Model (80/20) ──────────────────────────────────────────
        with tab5:
            st.markdown("##### 🧪 PCA + Classification — Train/Test Evaluation")

            if not target or target not in df.columns:
                st.warning("⚠️ No target column detected. Cannot train a classification model. "
                           "Ensure your dataset has a categorical target column.")
                return

            # Prepare target
            y_raw = df[target].copy()
            valid_mask = y_raw.notna()
            X_valid = X_pca[valid_mask.values[:len(X_pca)]]
            y_valid = y_raw[valid_mask].reset_index(drop=True)

            if y_valid.nunique() < 2:
                st.warning("Need ≥ 2 classes in target column for classification.")
                return

            le = LabelEncoder()
            y_enc = le.fit_transform(y_valid.values)

            # Check minimum class size for stratified split
            from collections import Counter
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
              {n_comp} PCA components · {y_valid.nunique()} classes
            </div>
            """, unsafe_allow_html=True)

            # Train model
            with st.spinner(f"Training {model_choice} on PCA features…"):
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
            fig_cm.update_layout(**heatmap_layout("Confusion Matrix — PCA + " + model_choice, height=450))
            st.plotly_chart(fig_cm, use_container_width=True)

            # Download results
            results_df = pd.DataFrame({
                "Actual": le.inverse_transform(y_test),
                "Predicted": le.inverse_transform(y_pred_test),
            })
            for i in range(min(n_comp, 5)):
                results_df[f"PC{i+1}"] = X_test[:, i]
            st.download_button(
                "⬇️ Download Predictions CSV",
                results_df.to_csv(index=False).encode("utf-8"),
                "pca_predictions.csv", "text/csv", key="pca_pred_dl",
            )

    except Exception as exc:
        st.error(f"PCA error: {exc}")
