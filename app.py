import streamlit as st
import pandas as pd
from processor import MachineLearningRepairKit

# Page Config
st.set_page_config(page_title="ML Repair Pipeline", page_icon="🛡️", layout="wide")

st.title("🛡️ Automated ML Data Repair Pipeline")
st.markdown("### Data Cleaning: Iterative Imputation & Outlier Removal")

# 1. File Upload
uploaded_file = st.file_uploader("Upload the CSV", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df_raw = pd.read_csv(uploaded_file)
    
    # Layout: Split screen for Before vs After
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Raw Data")
        st.write(f"Shape: {df_raw.shape}")
        # Show null counts
        null_counts = df_raw.isnull().sum()
        st.dataframe(df_raw.head(10))
        st.caption(f"Total Missing Values: {null_counts.sum()}")

    # 2. The Execution Trigger
    if st.button("🚀 Run Repair Pipeline", type="primary"):
        with st.spinner("Initializing Iterative Imputer & Isolation Forest..."):
            
            # Instantiate the Engine
            repair_kit = MachineLearningRepairKit()
            
            # Run the Logic
            df_clean, report = repair_kit.repair(df_raw)
            
            # Success Message
            st.success("Pipeline Execution Complete")
            
            with col2:
                st.subheader("✅ Repaired Data")
                st.write(f"Shape: {df_clean.shape}")
                st.dataframe(df_clean.head(10))
                
                # Metrics Display
                m1, m2, m3 = st.columns(3)
                m1.metric("Missing Fixed", report['missing_fixed'])
                m2.metric("Outliers Removed", report['outliers_detected'])
                m3.metric("Rows Retained", f"{report['final_shape'][0]}")

            # 3. Validation Visualization (Simple histogram)
            st.divider()
            st.subheader("📊 Distribution Check (First Numeric Column)")
            
            # Pick the first numeric column to plot
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
                
                chart_col1, chart_col2 = st.columns(2)
                
                # We need to force numeric on raw data just for plotting comparison
                # (The pipeline does this internally, but we do it here for the 'Before' chart)
                raw_numeric = pd.to_numeric(df_raw[target_col], errors='coerce')
                
                with chart_col1:
                    st.caption(f"Before: {target_col}")
                    st.bar_chart(raw_numeric)
                
                with chart_col2:
                    st.caption(f"After: {target_col}")
                    st.bar_chart(df_clean[target_col])

            # 4. Download
            csv = df_clean.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Clean CSV",
                csv,
                "clean_data.csv",
                "text/csv",
                key='download-csv'
            )