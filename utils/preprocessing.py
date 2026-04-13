"""
Preprocessing Module — sklearn Pipeline with SimpleImputer, OneHotEncoder,
StandardScaler, ColumnTransformer, VarianceThreshold, SelectKBest.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.pipeline           import Pipeline
from sklearn.impute             import SimpleImputer
from sklearn.preprocessing      import StandardScaler, OneHotEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.feature_selection  import SelectKBest, VarianceThreshold, f_classif, f_regression
from utils.data_loader          import detect_target_column, convert_df_to_csv


def render_preprocessing(df: pd.DataFrame, info: dict, label: str = "Dataset"):
    st.markdown("#### ⚙️ Preprocessing Pipeline")

    num_cols = info["numeric_cols"].copy()
    cat_cols = info["categorical_cols"].copy()
    target   = detect_target_column(df)

    with st.expander("🔧 Configure Pipeline", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            remove_dups  = st.checkbox("Remove Duplicate Rows",          value=True,  key="pp_dups")
            impute_num   = st.checkbox("Impute Missing — Numeric",        value=True,  key="pp_impn")
            impute_cat   = st.checkbox("Impute Missing — Categorical",    value=True,  key="pp_impc")
            scale        = st.checkbox("StandardScaler (Numeric)",        value=True,  key="pp_sc")
        with col2:
            encode       = st.checkbox("OneHotEncode (Categorical)",      value=True,  key="pp_enc")
            var_thresh   = st.checkbox("Remove Constant Cols (VarThresh)",value=True,  key="pp_vt")
            feat_select  = st.checkbox("Feature Selection (SelectKBest)", value=False, key="pp_fs")

        num_strategy = st.selectbox("Numeric imputation strategy",
                                    ["mean", "median", "most_frequent", "constant"],
                                    key="pp_ns")
        cat_strategy = st.selectbox("Categorical imputation strategy",
                                    ["most_frequent", "constant"], key="pp_cs")

        k_best = 5
        if feat_select:
            max_k  = max(1, len(num_cols) - 1)
            k_best = st.slider("SelectKBest — k features", 1, max_k, min(5, max_k), key="pp_kb")

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    steps = []
    if remove_dups:   steps.append("① Remove Duplicates")
    if impute_num:    steps.append(f"② Impute Numeric ({num_strategy})")
    if impute_cat:    steps.append(f"③ Impute Categorical ({cat_strategy})")
    if scale:         steps.append("④ StandardScaler")
    if encode:        steps.append("⑤ OneHotEncoder")
    if var_thresh:    steps.append("⑥ VarianceThreshold")
    if feat_select:   steps.append(f"⑦ SelectKBest (k={k_best})")

    if steps:
        st.markdown("**Pipeline Steps:** " + " → ".join(steps))

    if st.button("▶ Run Preprocessing Pipeline", key="pp_run"):
        with st.spinner("Running sklearn pipeline…"):
            result_df = _run_pipeline(
                df, num_cols, cat_cols, target,
                remove_dups=remove_dups, impute_num=impute_num, impute_cat=impute_cat,
                scale=scale, encode=encode, var_thresh=var_thresh,
                feat_select=feat_select, k_best=k_best,
                num_strategy=num_strategy, cat_strategy=cat_strategy,
            )
            if result_df is not None:
                st.session_state["processed_df"] = result_df

    if "processed_df" in st.session_state:
        proc_df = st.session_state["processed_df"]
        st.success(f"✅ Processed shape: {proc_df.shape[0]:,} rows × {proc_df.shape[1]} columns")
        st.dataframe(proc_df.head(25), use_container_width=True)
        st.download_button(
            "⬇️ Download Processed CSV",
            data=convert_df_to_csv(proc_df),
            file_name="weather_processed.csv",
            mime="text/csv",
            key="pp_dl",
        )


def _run_pipeline(df, num_cols, cat_cols, target,
                  remove_dups, impute_num, impute_cat, scale, encode,
                  var_thresh, feat_select, k_best,
                  num_strategy, cat_strategy) -> pd.DataFrame | None:
    try:
        work = df.copy()

        if remove_dups:
            before = len(work)
            work   = work.drop_duplicates()
            st.info(f"🔁 Removed {before - len(work)} duplicate rows.")

        feat_num = [c for c in num_cols if c != target]
        feat_cat = [c for c in cat_cols if c != target]

        # Build numeric pipeline
        num_steps = []
        if impute_num: num_steps.append(("imputer", SimpleImputer(strategy=num_strategy)))
        if scale:      num_steps.append(("scaler",  StandardScaler()))
        num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

        # Build categorical pipeline
        cat_steps = []
        if impute_cat: cat_steps.append(("imputer", SimpleImputer(strategy=cat_strategy, fill_value="missing")))
        if encode:     cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        cat_pipe = Pipeline(cat_steps) if cat_steps else "passthrough"

        # ColumnTransformer
        transformers = []
        if feat_num:              transformers.append(("num", num_pipe, feat_num))
        if feat_cat and cat_steps: transformers.append(("cat", cat_pipe, feat_cat))

        if not transformers:
            st.warning("No transformers configured — returning cleaned data.")
            return work

        ct     = ColumnTransformer(transformers=transformers, remainder="drop")
        X_in   = work[feat_num + feat_cat]
        X_out  = ct.fit_transform(X_in)

        # Column names
        out_cols = feat_num.copy()
        if feat_cat and cat_steps and encode:
            enc       = ct.named_transformers_["cat"].named_steps["encoder"]
            out_cols += enc.get_feature_names_out(feat_cat).tolist()
        elif feat_cat:
            out_cols += feat_cat

        if X_out.shape[1] != len(out_cols):
            out_cols = [f"feat_{i}" for i in range(X_out.shape[1])]

        result = pd.DataFrame(X_out, columns=out_cols)

        # VarianceThreshold
        if var_thresh:
            vt = VarianceThreshold(threshold=0.0)
            try:
                X_vt  = vt.fit_transform(result)
                kept  = result.columns[vt.get_support()].tolist()
                removed = set(result.columns) - set(kept)
                result  = pd.DataFrame(X_vt, columns=kept)
                if removed:
                    st.info(f"🔒 Removed constant columns: {list(removed)}")
            except Exception as e:
                st.warning(f"VarianceThreshold skipped: {e}")

        # SelectKBest
        if feat_select and target and target in work.columns:
            y       = work[target].values[:len(result)]
            fn      = f_classif if (work[target].dtype == object or work[target].nunique() <= 20) else f_regression
            k       = min(k_best, result.shape[1])
            sel     = SelectKBest(score_func=fn, k=k)
            X_sel   = sel.fit_transform(result, y)
            kept    = result.columns[sel.get_support()].tolist()
            scores  = pd.DataFrame({
                "Feature": kept,
                "Score":   sel.scores_[sel.get_support()].round(3),
            }).sort_values("Score", ascending=False)
            result  = pd.DataFrame(X_sel, columns=kept)
            st.markdown("**SelectKBest Feature Scores:**")
            st.dataframe(scores, use_container_width=True)

        # Re-attach target
        if target and target in work.columns:
            result[target] = work[target].values[:len(result)]

        return result

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        return None
