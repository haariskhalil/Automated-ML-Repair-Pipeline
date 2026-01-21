import streamlit as st
import pandas as pd
import plotly.express as px
from processor import MachineLearningRepairKit

# Page Config
st.set_page_config(page_title="ML Repair Pipeline", page_icon="🛡️", layout="wide")

st.title("🛡️ Automated ML Repair Pipeline")
st.markdown("### Data Cleaning: Iterative Imputation & Outlier Removal")

# Initialize Session State for persistence
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "report" not in st.session_state:
    st.session_state.report = None

# 1. File Upload
uploaded_file = st.file_uploader("Upload your messy CSV", type=["csv"])

if uploaded_file is not None:

    df_raw = pd.read_csv(uploaded_file)
    
    # Split screen for Before vs After
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Raw Data")
        st.write(f"Shape: {df_raw.shape}")
        # Show null counts
        null_counts = df_raw.isnull().sum()
        st.dataframe(df_raw, height=400)
        st.caption(f"Total Missing Values: {null_counts.sum()}")

    # 2. The Execution Trigger
    if st.button("🚀 Run Repair Pipeline", type="primary"):
        with st.spinner("Initializing Random Forest Imputer & Isolation Forest..."):
            
            # Instantiate the Engine
            repair_kit = MachineLearningRepairKit()
            
            # Run the Logic
            df_clean, report = repair_kit.repair(df_raw)
            
            # Save to Session State
            st.session_state.df_clean = df_clean
            st.session_state.report = report
            st.session_state.repair_kit = repair_kit 
            
            st.success("Pipeline Execution Complete")

    # 3. Persistent Display Logic
    if st.session_state.df_clean is not None:
        
        # Retrieve from memory
        df_clean = st.session_state.df_clean
        report = st.session_state.report
        
        if "repair_kit" in st.session_state:
            repair_kit = st.session_state.repair_kit
        else:
            repair_kit = MachineLearningRepairKit()

        with col2:
            st.subheader("✅ Repaired Data")
            st.write(f"Shape: {df_clean.shape}")
            st.dataframe(df_clean, height=400)
            
            # Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Missing Fixed", report['missing_fixed'])
            m2.metric("Outliers Removed", report['outliers_detected'])
            m3.metric("Rows Retained", f"{report['final_shape'][0]}")

        # Outlier Audit Section
        if report['outliers_detected'] > 0:
            with st.expander("🔍 Audit & Restore: Detected Outliers", expanded=True):
                
                # To make buttons inside this specific expander Green
                st.markdown("""
                    <style>
                    /* Target buttons inside the Expander Details */
                    [data-testid="stExpanderDetails"] button {
                        background-color: #28a745 !important;
                        color: white !important;
                        border: 1px solid #28a745 !important;
                    }
                    [data-testid="stExpanderDetails"] button:hover {
                        background-color: #218838 !important;
                        border-color: #1e7e34 !important;
                        color: white !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                # ----------------------------------------------------------------

                st.warning("The rows below were flagged as anomalies. Check the box and click 'Restore' to force-add them back to the clean dataset.")
                
                # 1. Prepare data for the editor
                audit_df = report['outliers'].copy()
                if "Restore" not in audit_df.columns:
                    audit_df.insert(0, "Restore", False)
                
                # 2. Display the Editable Table
                edited_df = st.data_editor(
                    audit_df, 
                    column_config={"Restore": st.column_config.CheckboxColumn(required=True)},
                    disabled=audit_df.columns.drop("Restore"), 
                    use_container_width=True
                )
                
                # 3. The Restore Button Logic
                if st.button("Restore Selected Rows"):
                    rows_to_restore = edited_df[edited_df["Restore"] == True].copy()
                    
                    if not rows_to_restore.empty:
                        # Clean up
                        cols_to_drop = ["Restore", "Likely_Reason", "Anomaly_Score"]
                        cols_to_drop = [c for c in cols_to_drop if c in rows_to_restore.columns]
                        clean_rows_to_add = rows_to_restore.drop(columns=cols_to_drop)
                        
                        # Add back
                        st.session_state.df_clean = pd.concat(
                            [st.session_state.df_clean, clean_rows_to_add], 
                            ignore_index=True
                        )
                        
                        # Remove from Report
                        restore_indices = rows_to_restore.index
                        st.session_state.report['outliers'] = st.session_state.report['outliers'].drop(restore_indices)
                        
                        # Metrics Update
                        st.session_state.report['outliers_detected'] -= len(rows_to_restore)
                        st.session_state.report['final_shape'] = st.session_state.df_clean.shape
                        
                        st.success(f"Restored {len(rows_to_restore)} rows!")
                        st.rerun()
                    else:
                        st.info("Please check at least one box to restore.")

        # 4. Benchmarking
        st.divider()
        st.subheader("🧪 Model Comparison: Baseline vs. Advanced Pipeline")
        st.info("Two models will be trained to predict a target variable. Lower Error (MAE) is better.")
        
       # Select Target (Dynamic & Numeric Only)
        # check st.session_state.df_clean to ensure only columns that ended up numeric are picked
        if 'df_clean' in st.session_state:
            numeric_cols = st.session_state.df_clean.select_dtypes(include=['number']).columns.tolist()
        else:
            # Fallback if clean data isn't generated yet
            numeric_cols = df_raw.select_dtypes(include=['number']).columns.tolist()

        if not numeric_cols:
            st.error("No numeric columns found to predict! The 'Showdown' requires at least one number column (like Salary).")
        else:
            target_col = st.selectbox("Select Target Variable to Predict", numeric_cols)
            
            if st.button("⚔️ Fight!", type="primary"):
                if len(df_raw) < 10:
                    st.warning("⚠️ Dataset is very small. Results might be volatile!")
                
                with st.spinner("Training models..."):
                    # pass df_raw (dirty data) so the function can compare "Baseline" vs "Repaired Data".
                    metrics = repair_kit.evaluate_model(df_raw, target_col)
                
                if "error" in metrics:
                    st.error(metrics["error"])
                else:
                    # Display metrics side-by-side
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown("### Baseline Model")
                        st.caption("Strategy: Mean Imputation + Linear Regression")
                        st.metric("Mean Absolute Error", f"{metrics['baseline_mae']:.2f}")
                        st.metric("R² Score", f"{metrics['baseline_r2']:.2f}")
                    
                    with res_col2:
                            st.markdown("### Advanced Pipeline")
                            st.caption("Strategy: Iterative RF Imputation + Outlier Removal + Random Forest")
                            
                            delta_mae = metrics['advanced_mae'] - metrics['baseline_mae']
                            st.metric(
                                "Mean Absolute Error", 
                                f"{metrics['advanced_mae']:.2f}", 
                                delta=f"{delta_mae:.2f}", 
                                delta_color="inverse"
                            )
                            
                            delta_r2 = metrics['advanced_r2'] - metrics['baseline_r2']
                            st.metric(
                                "R² Score", 
                                f"{metrics['advanced_r2']:.2f}", 
                                delta=f"{delta_r2:.2f}"
                            )
                
                # Chart
                st.markdown("#### Performance Comparison")
                chart_data = pd.DataFrame({
                    "Model": ["Baseline", "Advanced Pipeline"],
                    "Error (MAE) - Lower is Better": [metrics['baseline_mae'], metrics['advanced_mae']]
                })
                fig = px.bar(chart_data, x="Model", y="Error (MAE) - Lower is Better", color="Model", 
                             color_discrete_map={"Baseline": "#EF553B", "Advanced Pipeline": "#00CC96"})
                st.plotly_chart(fig, use_container_width=True)
        
        # 5. Download
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Clean CSV",
            csv,
            "clean_data.csv",
            "text/csv",
            key='download-csv'
        )