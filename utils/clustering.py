"""
Clustering Module — KMeans weather pattern detection with cluster
visualizations, profiles, and elbow/silhouette analysis.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster        import KMeans
from sklearn.preprocessing  import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.decomposition  import PCA
from sklearn.metrics         import silhouette_score, davies_bouldin_score
from utils.chart_style       import PALETTE, BLUE, EMERALD, AMBER, INDIGO, base_layout, heatmap_layout
from utils.data_loader       import detect_target_column, convert_df_to_csv


def render_clustering(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### 🔴 KMeans Clustering — Weather Pattern Detection")

    num_cols = info["numeric_cols"]
    if len(num_cols) < 2:
        st.info("Need ≥ 2 numeric columns for clustering.")
        return

    target = detect_target_column(df)

    with st.expander("⚙️ Clustering Settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            k = st.slider("Number of Clusters (K)", 2, 12, 4, key="km_k")
        with c2:
            n_init = st.slider("n_init", 5, 30, 10, key="km_ninit")
        with c3:
            max_iter = st.slider("Max Iterations", 100, 500, 300, key="km_iter")

        feat_cols = st.multiselect(
            "Feature Columns",
            num_cols,
            default=num_cols[:min(6, len(num_cols))],
            key="km_feats",
        )

    if len(feat_cols) < 2:
        st.warning("Select at least 2 feature columns.")
        return

    try:
        # Prepare data
        imp      = SimpleImputer(strategy="mean")
        X_imp    = imp.fit_transform(df[feat_cols])
        X_scaled = StandardScaler().fit_transform(X_imp)

        # ── Elbow + Silhouette ────────────────────────────────────────────────
        _sel_clustering = st.radio('Select View', ["Elbow & Silhouette", "Cluster Visualization", "Cluster Profiles", "vs True Labels", "Download"], horizontal=True, label_visibility='collapsed', key='radio_clustering_48ce')

        if _sel_clustering == "Elbow & Silhouette":
            st.markdown("##### 📐 Optimal K Selection")
            inertias    = []
            sil_scores  = []
            db_scores   = []
            k_range     = range(2, min(12, len(df)//5 + 2))

            with st.spinner("Calculating elbow curve…"):
                # Downsample for performance on large datasets
                if len(X_scaled) > 3000:
                    np.random.seed(42)
                    idx = np.random.choice(len(X_scaled), 3000, replace=False)
                    X_eval = X_scaled[idx]
                else:
                    X_eval = X_scaled
                
                for ki in k_range:
                    km = KMeans(n_clusters=ki, n_init=3, max_iter=100, random_state=42)
                    km.fit(X_eval)
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(X_eval, km.labels_))
                    db_scores.append(davies_bouldin_score(X_eval, km.labels_))

            elbow_df = pd.DataFrame({
                "K":            list(k_range),
                "Inertia":      inertias,
                "Silhouette":   [round(s, 4) for s in sil_scores],
                "Davies-Bouldin": [round(d, 4) for d in db_scores],
            })

            best_k = int(elbow_df.loc[elbow_df["Silhouette"].idxmax(), "K"])
            st.info(f"📌 Best K by Silhouette Score: **K = {best_k}**")

            c1, c2 = st.columns(2)
            with c1:
                fig_elbow = px.line(elbow_df, x="K", y="Inertia",
                                    markers=True, color_discrete_sequence=[BLUE])
                fig_elbow.update_layout(**base_layout("Elbow Curve (Inertia)"))
                st.plotly_chart(fig_elbow, use_container_width=True)
            with c2:
                fig_sil = px.line(elbow_df, x="K", y="Silhouette",
                                  markers=True, color_discrete_sequence=[EMERALD])
                fig_sil.add_vline(x=best_k, line_dash="dash", line_color="#ef4444",
                                   annotation_text=f"Best K={best_k}")
                fig_sil.update_layout(**base_layout("Silhouette Score by K"))
                st.plotly_chart(fig_sil, use_container_width=True)

            st.dataframe(elbow_df, use_container_width=True)

        # ── Run KMeans with selected K ────────────────────────────────────────
        km     = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=42)
        labels = km.fit_predict(X_scaled)

        cluster_df            = df[feat_cols].copy()
        cluster_df["Cluster"] = [f"Cluster {l+1}" for l in labels]
        if target and target in df.columns:
            cluster_df[target] = df[target].values[:len(cluster_df)]

        sil = silhouette_score(X_scaled, labels)
        db  = davies_bouldin_score(X_scaled, labels)

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters (K)",      k)
        c2.metric("Silhouette Score",  round(sil, 4))
        c3.metric("Davies-Bouldin",    round(db, 4))

        # ── Cluster Visualization ─────────────────────────────────────────────
        if _sel_clustering == "Cluster Visualization":
            # PCA reduce to 2D/3D for visualization
            pca2  = PCA(n_components=min(3, len(feat_cols)), random_state=42)
            X_pca = pca2.fit_transform(X_scaled)

            vis_df = pd.DataFrame(X_pca[:, :2], columns=["PCA 1", "PCA 2"])
            vis_df["Cluster"] = cluster_df["Cluster"].values

            fig = px.scatter(vis_df, x="PCA 1", y="PCA 2",
                             color="Cluster",
                             color_discrete_sequence=PALETTE,
                             opacity=0.75,
                             title=f"KMeans Clusters (K={k}) — PCA Projection")
            fig.update_layout(**base_layout(f"KMeans K={k} — 2D PCA", height=480))
            st.plotly_chart(fig, use_container_width=True)

            if X_pca.shape[1] >= 3:
                vis3d = pd.DataFrame(X_pca[:, :3], columns=["PCA 1", "PCA 2", "PCA 3"])
                vis3d["Cluster"] = cluster_df["Cluster"].values
                fig3d = px.scatter_3d(vis3d, x="PCA 1", y="PCA 2", z="PCA 3",
                                       color="Cluster",
                                       color_discrete_sequence=PALETTE, opacity=0.7)
                fig3d.update_layout(paper_bgcolor="white", margin=dict(t=50, b=20), height=560)
                st.plotly_chart(fig3d, use_container_width=True)

            # Cluster size pie
            size_df = cluster_df["Cluster"].value_counts().reset_index()
            size_df.columns = ["Cluster", "Count"]
            fig_pie = px.pie(size_df, values="Count", names="Cluster",
                             color_discrete_sequence=PALETTE, hole=0.4)
            fig_pie.update_layout(**base_layout("Cluster Size Distribution", height=340))
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Cluster Profiles ──────────────────────────────────────────────────
        if _sel_clustering == "Cluster Profiles":
            st.markdown("##### 📊 Cluster Mean Profiles")
            profile = cluster_df.groupby("Cluster")[feat_cols].mean().round(3)
            st.dataframe(profile, use_container_width=True)

            # Heatmap of cluster centroids
            profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
            fig_hm = px.imshow(
                profile_norm.T, text_auto=".2f", aspect="auto",
                color_continuous_scale="Blues",
            )
            fig_hm.update_layout(**heatmap_layout("Normalized Cluster Centroids", height=400))
            st.plotly_chart(fig_hm, use_container_width=True)

            # Bar chart per feature per cluster
            st.markdown("##### Feature Comparison Across Clusters")
            sel_feat = st.selectbox("Select Feature", feat_cols, key="km_feat_bar")
            bar_df = cluster_df.groupby("Cluster")[sel_feat].mean().reset_index()
            fig_bar = px.bar(bar_df, x="Cluster", y=sel_feat,
                              color="Cluster", color_discrete_sequence=PALETTE)
            fig_bar.update_layout(**base_layout(f"Mean {sel_feat} by Cluster", height=340))
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Compare with True Labels ──────────────────────────────────────────
        if _sel_clustering == "vs True Labels":
            if target and target in df.columns:
                cross_df = pd.crosstab(cluster_df["Cluster"], df[target])
                st.markdown(f"##### Cluster vs True Label: `{target}`")
                st.dataframe(cross_df, use_container_width=True)

                fig_cross = px.imshow(cross_df, text_auto=True, aspect="auto",
                                       color_continuous_scale="Blues")
                fig_cross.update_layout(**heatmap_layout("Cluster vs True Label Heatmap"))
                st.plotly_chart(fig_cross, use_container_width=True)
            else:
                st.info("No target column detected for comparison.")

        # ── Download ──────────────────────────────────────────────────────────
        if _sel_clustering == "Download":
            # Attach cluster labels to original df
            result_df           = df.copy()
            result_df["KMeans_Cluster"] = [f"Cluster {l+1}" for l in labels]
            st.dataframe(result_df.head(30), use_container_width=True)
            st.download_button(
                "⬇️ Download Clustered Dataset",
                convert_df_to_csv(result_df),
                "weather_clustered.csv", "text/csv", key="km_dl",
            )

    except Exception as exc:
        st.error(f"Clustering error: {exc}")
