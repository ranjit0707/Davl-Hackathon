"""
Train-Test Split Module — prepares dataset for machine learning.
Uses sklearn train_test_split.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from utils.data_loader import detect_target_column, convert_df_to_csv


def render_train_test_split(df: pd.DataFrame, info: dict):
    st.markdown("#### ✂️ Train-Test Split")

    if df is None:
        st.warning("Please upload a dataset first.")
        return

    target = detect_target_column(df)
    
    with st.expander("⚙️ Split Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            selected_target = st.selectbox(
                "Select Target Column (Y)",
                df.columns.tolist(),
                index=df.columns.get_loc(target) if target in df.columns else 0,
                key="split_target"
            )
            test_size = st.slider("Test Size (%)", 5, 50, 20, step=5, key="split_size") / 100
        
        with col2:
            random_state = st.number_input("Random State", value=42, step=1, key="split_seed")
            shuffle = st.checkbox("Shuffle Data", value=True, key="split_shuffle")
            stratify = st.checkbox("Stratify (keep class proportions)", value=False, key="split_strat")

    if st.button("🚀 Perform Split", key="run_split"):
        try:
            y = df[selected_target]
            X = df.drop(columns=[selected_target])
            
            stratify_y = y if stratify else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state, 
                shuffle=shuffle,
                stratify=stratify_y
            )
            
            st.session_state["split_data"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "target": selected_target
            }
            st.success("✅ Dataset successfully split into training and testing sets.")

        except Exception as e:
            st.error(f"Error during split: {e}")
            if "stratify" in str(e).lower():
                st.info("💡 Stratify might fail if some classes have too few samples. Try disabling 'Stratify'.")

    # ── Display Split Results ─────────────────────────────────────────────────
    if "split_data" in st.session_state:
        data = st.session_state["split_data"]
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]
        
        st.markdown("##### 📊 Split Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Samples", f"{len(X_train):,}")
        c2.metric("Testing Samples", f"{len(X_test):,}")
        c3.metric("Test Ratio", f"{test_size*100:.0f}%")

        _sel_split = st.radio('Select View', ["Training Set", "Testing Set", "Downloads"], horizontal=True, label_visibility='collapsed', key='radio_split_5ced')
        
        if _sel_split == "Training Set":
            st.markdown(f"**X_train** ({X_train.shape[0]}x{X_train.shape[1]})")
            st.dataframe(X_train.head(10), use_container_width=True)
            st.markdown(f"**y_train** ({y_train.shape[0]} samples)")
            st.dataframe(y_train.head(10), use_container_width=True)

        if _sel_split == "Testing Set":
            st.markdown(f"**X_test** ({X_test.shape[0]}x{X_test.shape[1]})")
            st.dataframe(X_test.head(10), use_container_width=True)
            st.markdown(f"**y_test** ({y_test.shape[0]} samples)")
            st.dataframe(y_test.head(10), use_container_width=True)

        if _sel_split == "Downloads":
            col_a, col_b = st.columns(2)
            
            train_full = pd.concat([X_train, y_train], axis=1)
            test_full = pd.concat([X_test, y_test], axis=1)
            
            with col_a:
                st.download_button(
                    "⬇️ Download Train Set (CSV)",
                    convert_df_to_csv(train_full),
                    "weather_train.csv",
                    "text/csv",
                    key="dl_train"
                )
            with col_b:
                st.download_button(
                    "⬇️ Download Test Set (CSV)",
                    convert_df_to_csv(test_full),
                    "weather_test.csv",
                    "text/csv",
                    key="dl_test"
                )
