
# Analog IC Fault Detection Dashboard
# Ashwin Kumar Singh | IIT Kanpur


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# - File Directories -
BASE_DIR   = BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"

# The 23 features extracted from our transient analysis data
FEATURES = [
    'mean_diff', 'std_diff', 'max_diff', 'min_diff', 'peak_to_peak', 'rms_diff', 
    'skewness', 'kurtosis', 'mean_cm', 'std_cm', 'cm_deviation', 'std_voutp', 
    'std_voutn', 'mean_voutp', 'mean_voutn', 'output_asymmetry', 'mean_asymmetry', 
    'fundamental', 'harmonic2', 'harmonic3', 'thd', 'spectral_energy', 'high_freq_ratio'
]

# -Helper Functions for Model Storage 
@st.cache_resource
def load_sklearn_artifacts():
    rf      = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb     = joblib.load(MODELS_DIR / "xgboost.pkl")
    scaler  = joblib.load(MODELS_DIR / "scaler.pkl")
    le      = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return rf, xgb, scaler, le

@st.cache_data
def load_raw_dataset():
    return pd.read_csv(DATA_DIR / "dataset_final.csv")

# - UI Configuration -
st.set_page_config(
    page_title="FDA OTA Fault Diagnosis",
    page_icon="⚡",
    layout="wide"
)

# - Sidebar Layout -
st.sidebar.title("⚡ Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go To:",
    ["🏠 Project Overview", "🔍 Live Testing Engine", "📊 Model Comparison Matrix", "🔌 Circuit Specifications"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Ashwin Kumar Singh")
st.sidebar.markdown("**Institute:** IIT Kanpur")
st.sidebar.markdown("**Branch:** Electrical Engineering")

# Load persistent states
rf, xgb, scaler, le = load_sklearn_artifacts()
df = load_raw_dataset()
num_features = len(FEATURES)


# 1. PROJECT OVERVIEW (HOME)

if page == "🏠 Project Overview":
    st.title("⚡ Automated Fault Diagnosis for Analog ICs")
    st.markdown("### Machine Learning Framework for Fully Differential OTA Diagnostics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Circuit Architecture", "FDA OTA")
    col2.metric("Dataset Rows", f"{len(df)}")
    col3.metric("Target Profiles", f"{len(le.classes_)}")
    col4.metric("Peak Test Accuracy", "99.83%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Project Focus")
        st.markdown("""
        Parametric and geometric faults at the internal nodes of analog integrated circuits are notoriously hard to isolate. Traditional testing methods require slow, complex setups and deep manual verification.
        
        **This automated pipeline:**
        - Models a fully differential operational transconductance amplifier (FDA OTA) in a **180nm CMOS process**.
        - Simulates the layout under **6 specific target configurations** (including aspect ratio mismatches, thermal variance, and voltage supply rails).
        - Tracks a rich **23-dimensional feature space** derived directly from raw transient voltage outputs.
        - Isolates and maps anomalies instantly with up to **100.00% accuracy** using optimized classification models.
        """)

    with col2:
        st.subheader("🔧 Core Operational Constraints")
        specs = {
            'Design Metric': ['Technology Node', 'Supply Voltage (VDD)', 'Gain Bandwidth (GBW)', 'Phase Margin', 'DC Open-Loop Gain', 'Nominal THD'],
            'Target Value': ['TSMC 180nm CMOS', '1.8V', '10 MHz', '90°', '34 dB', '0.136%']
        }
        st.dataframe(pd.DataFrame(specs), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("⚡ Modeled Circuit Conditions")

    fault_info = {
        'Condition Profile': ['Healthy', 'M1 Width Fault', 'M2 Width Fault', 'M3 Mirror Mismatch', 'High Temperature', 'VDD Overvoltage'],
        'Simulation Parameters': [
            'Circuit running entirely within normal operational boundaries.',
            'M1 differential input channel width falls by 15-25%.',
            'M2 differential input channel width falls by 15-25%.',
            'Active PMOS current mirror balancing ratio scales off by +10-20%.',
            'Circuit core exposed to localized heat spikes (100°C to 125°C).',
            'Main supply voltage rail spikes by +5-12% above nominal parameters.'
        ],
        'Physical Output Behavior': [
            'Baseline reference markers for all extraction vectors.',
            'Degrades total differential gain and introduces structural signal asymmetry.',
            'Causes structural output distortion complementary but opposite to M1 failures.',
            'Displaces target common-mode points and adds stress to the CMFB loop.',
            'Lowers overall carrier mobility, degrading open-loop voltage margins.',
            'Forces common-mode levels upward, shrinking linear output swing limits.'
        ]
    }
    st.dataframe(pd.DataFrame(fault_info), hide_index=True, use_container_width=True)



# 2. LIVE TESTING ENGINE (DETECTOR)

elif page == "🔍 Live Testing Engine":
    st.title("🔍 Evaluation & Inference Engine")
    st.markdown("Verify extraction logs by uploading custom data or dragging random samples directly from your test splits.")
    st.markdown("---")

    model_choice = st.selectbox(
        "Select Active Classification Model",
        ["XGBoost (99.83%)", "Random Forest (99.67%)", "KNN k=5 (99.00%)", "SVM RBF (97.83%)"]
    )

    st.markdown("---")
    tab1, tab2 = st.tabs(["📁 Parse Custom Log File (.csv)", "🎲 Pull Random Test Samples"])

    with tab1:
        st.markdown("#### Load Simulation Logs")
        uploaded = st.file_uploader("Upload target CSV feature data", type=['csv'])

        if uploaded:
            try:
                input_df = pd.read_csv(uploaded)
                if all(c in input_df.columns for c in FEATURES):
                    X_input = input_df[FEATURES].values
                else:
                    X_input = input_df.select_dtypes(include=np.number).values[:, :num_features]

                if X_input.shape[1] != num_features:
                    st.error(f"Feature count mismatch. Expected {num_features}, found {X_input.shape[1]}")
                    st.stop()

                X_scaled = scaler.transform(X_input)
                st.success(f"Parsed {len(X_input)} test entries successfully.")

                if st.button("Execute Diagnostic Test Run"):
                    if "XGBoost" in model_choice:
                        preds = xgb.predict(X_scaled)
                        probs = xgb.predict_proba(X_scaled)
                    else:
                        preds = rf.predict(X_scaled)
                        probs = rf.predict_proba(X_scaled)

                    pred_names = le.inverse_transform(preds)

                    st.markdown("### Classification Report")
                    for i, (pred, prob) in enumerate(zip(pred_names, probs)):
                        conf = prob.max() * 100
                        tag_color = "green" if pred.lower() == "healthy" else "red"
                        st.markdown(f"**Sample {i+1}:** :{tag_color}[{pred}] — System Confidence: **{conf:.2f}%**")

            except Exception as e:
                st.error(f"Error handling file execution: {e}")

    with tab2:
        st.markdown("#### Fetch Random Logs from Simulation Base")
        sample_fault = st.selectbox("Choose Target Condition Profile to Sample", list(le.classes_))
        n_samples = st.slider("Number of samples to grab", 1, 10, 3)

        if st.button("Sample Database and Run Inference"):
            subset = df[df['fault_name'] == sample_fault]
            if subset.empty:
                st.error(f"No experimental rows found for target `{sample_fault}`.")
                st.stop()
                
            samples = subset.sample(min(n_samples, len(subset)), random_state=42)
            X_sample = samples[FEATURES].values
            X_scaled = scaler.transform(X_sample)

            if "XGBoost" in model_choice:
                preds = xgb.predict(X_scaled)
                probs = xgb.predict_proba(X_scaled)
            else:
                preds = rf.predict(X_scaled)
                probs = rf.predict_proba(X_scaled)

            pred_names = le.inverse_transform(preds)

            st.markdown(f"### Evaluation Outputs for Label: `{sample_fault}`")
            col_inf, col_chart = st.columns(2)

            with col_inf:
                for i, (pred, prob) in enumerate(zip(pred_names, probs)):
                    conf = prob.max() * 100
                    matched = (pred.lower() == sample_fault.lower())
                    status = "✅ Prediction Match" if matched else "❌ Class Variance"
                    st.markdown(f"{status} — **Sample {i+1}:** Classified as `{pred}` ({conf:.1f}%)")

            with col_chart:
                avg_probs = probs.mean(axis=0)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#2ecc71' if c.lower() == sample_fault.lower() else '#bdc3c7' for c in le.classes_]
                ax.barh(le.classes_, avg_probs * 100, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_xlabel('Probability Score (%)')
                ax.set_title('Mean Network Classification Confidence')
                st.pyplot(fig)


# 3. MODEL COMPARISON MATRIX

elif page == "📊 Model Comparison Matrix":
    st.title("📊 Model Benchmarks")
    st.markdown("Comparative results across all tested architectures using stratified validation sets.")
    st.markdown("---")
    st.info("""
    📊 **Note on Accuracy:** Peak accuracy reaches 100% 
    (Random Forest on 90-sample test set). XGBoost at 
    99.83% is selected as production model — better 
    generalization across validation splits.
    """)

    # This mirrors your Jupyter Notebook Summary Ranking exactly
    model_results = {
        'Classifier Model': [
            'Logistic Regression', 
            'KNN (k=5)', 
            'Random Forest', 
            'XGBoost', 
            'Decision Tree', 
            'SVM (RBF)', 
            'Naive Bayes Baseline',
	    'MLP(PyTorch)'
        ],
        'Test Accuracy (%)': [97.67, 99.00, 99.67, 99.83, 94.83, 97.83, 93.17,84.44],
        'Model Family': [
            'Linear Decision Boundary', 
            'Instance-Based Proximity Model', 
            'Bagged Ensemble Trees', 
            'Gradient Boosted Decision Trees', 
            'Single Node Partitioning', 
            'Kernelized Vector Projections', 
            'Feature Independence Classifier',
            'Feedforward Neural Network'
        ]
    }
    results_df = pd.DataFrame(model_results)

    # Plotting the notebook accurate results
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = ['#2ecc71' if acc == 100 else '#3498db' if acc >= 99 else '#f1c40f' if acc >= 97 else '#e74c3c' for acc in results_df['Test Accuracy (%)']]
    bars = ax.bar(results_df['Classifier Model'], results_df['Test Accuracy (%)'], color=bar_colors, edgecolor='black', linewidth=0.7)

    for bar, acc in zip(bars, results_df['Test Accuracy (%)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(85, 105)
    ax.set_ylabel('Validation Set Accuracy (%)', fontsize=11)
    ax.set_title('Comparative Classifier Performance Benchmark (Notebook Metrics)', fontsize=13, fontweight='bold')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, label='90% Target Baseline')
    ax.legend(loc='lower left')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Experimental Performance Ledger")
    st.dataframe(results_df, hide_index=True, use_container_width=True)


# 4. CIRCUIT SPECIFICATIONS

elif page == "🔌 Circuit Specifications":
    st.title("🔌 FDA OTA Circuit Architecture & Parameter Layout")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Operational Target Specifications")
        specs = {
            'Design Attribute': ['Process Technology Node', 'Nominal Supply (VDD)', 'Total Current Budget', 'Gain Bandwidth (GBW)', 'Target Phase Margin', 'Open-Loop DC Voltage Gain', 'Total Harmonic Distortion (THD)', 'Common-Mode Isolation', 'Target Output Common-Mode'],
            'Engineering Value': ['TSMC 180nm CMOS', '1.8 V', '100 µA', '10 MHz', '90°', '34.2 dB', '0.136%', '>60.4 dB', '0.9 V']
        }
        st.dataframe(pd.DataFrame(specs), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Transistor Sizing Matrix")
        sizing = {
            'Transistor Groups': ['M1, M2 (Differential Input Core Pair)', 'M3, M4 (Active PMOS Mirror Loads)', 'M5 (Tail Current Source Regulator)', 'M6 through M10 (Switched-Capacitor CMFB Arrays)'],
            'Aspect Ratio (W / L)': ['10.0 µm / 0.18 µm', '28.0 µm / 0.18 µm', '20.0 µm / 0.18 µm', 'Sized to match tracking tolerances']
        }
        st.dataframe(pd.DataFrame(sizing), hide_index=True, use_container_width=True)
