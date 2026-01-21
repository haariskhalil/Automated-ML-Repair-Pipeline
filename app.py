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
        st.dataframe(df_raw.head(10))
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
            st.session_state.repair_kit = repair_kit # Save the object itself to reuse methods
            
            st.success("Pipeline Execution Complete")

    # 3. Persistent Display Logic
    # Check if data exists in memory, regardless of button press
    if st.session_state.df_clean is not None:
        
        # Retrieve from memory
        df_clean = st.session_state.df_clean
        report = st.session_state.report
        # We need to re-instantiate or retrieve the kit to run evaluation
        if "repair_kit" in st.session_state:
            repair_kit = st.session_state.repair_kit
        else:
            repair_kit = MachineLearningRepairKit()

        with col2:
            st.subheader("✅ Repaired Data")
            st.write(f"Shape: {df_clean.shape}")
            st.dataframe(df_clean.head(10))
            
            # Metrics Display
            m1, m2, m3 = st.columns(3)
            m1.metric("Missing Fixed", report['missing_fixed'])
            m2.metric("Outliers Removed", report['outliers_detected'])
            m3.metric("Rows Retained", f"{report['final_shape'][0]}")

        # 4. Benchmarking
        st.divider()
        st.subheader("🧪 Model Comprison: Baseline vs. Advanced Pipeline")
        st.info("Two models will be trained to predict a target variable. Lower Error (MAE) is better.")
        
        # Select Target
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        target_col = st.selectbox("Select Target Variable to Predict (e.g., Salary)", numeric_cols)
        
        if st.button("⚔️ Fight!", type="primary"):
            if len(df_raw) < 10:
                st.warning("⚠️ Dataset is very small. Results might be volatile!")
            
            with st.spinner("Training models..."):
                # the original raw data to the evaluate function
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