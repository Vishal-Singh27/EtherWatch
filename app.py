import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ml_engine import EthereumFraudDetector, DATA_PATH

st.set_page_config(
    page_title="EtherWatch — Ethereum Fraud & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .title-banner {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0.2rem;
    }
    .sub-banner {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-metric {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Detector in Session State
if "detector" not in st.session_state:
    st.session_state.detector = EthereumFraudDetector()
    st.session_state.detector.train_all_models()

detector = st.session_state.detector
df = detector.df

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")
    st.markdown(f"**Dataset Size:** `{len(df)} transactions`")
    st.markdown(f"**Features Analyzed:** `{len(detector.feature_names)} on-chain metrics`")
    
    st.markdown("---")
    st.markdown("### 🎯 Class Distribution")
    counts = df["FLAG"].value_counts()
    st.write(f"- Legitimate (0): **{counts.get(0, 0)}** ({counts.get(0, 0)/len(df)*100:.1f}%)")
    st.write(f"- Fraudulent (1): **{counts.get(1, 0)}** ({counts.get(1, 0)/len(df)*100:.1f}%)")
    
    st.markdown("---")
    st.caption("EtherWatch • Anomaly Detection on Ethereum Transaction Mempool & Accounts")

# Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="title-banner">⚡ EtherWatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-banner">Machine Learning & Anomaly Detection on Ethereum Blockchain Transactions</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("● Pipeline Ready (XGBoost + RF)")

tabs = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🤖 Model Benchmark & Metrics", "🔍 Feature Importance", "⚡ Live Wallet Simulator", "🏗️ Architecture"])

# ================= TAB 1: EDA =================
with tabs[0]:
    st.subheader("📊 Exploratory Data Analysis (On-Chain Transaction Features)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Transactions", f"{len(df):,}")
    with c2:
        st.metric("Fraud Rate", f"{(df['FLAG'].sum() / len(df) * 100):.1f}%")
    with c3:
        st.metric("Avg ETH Sent (Legit)", f"{df[df['FLAG']==0]['Total_Ether_Sent'].mean():.1f} ETH")
    with c4:
        st.metric("Avg ETH Sent (Fraud)", f"{df[df['FLAG']==1]['Total_Ether_Sent'].mean():.1f} ETH")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### 1. Account Lifespan vs. Fraud Probability")
        fig_lifetime = px.box(
            df, x="FLAG", y="Time_Diff_First_Last_Mins",
            color="FLAG",
            labels={"FLAG": "Fraud Flag (0=Legit, 1=Fraud)", "Time_Diff_First_Last_Mins": "Account Lifetime (Mins)"},
            log_y=True,
            title="Account Lifespan (Log Scale): Burner Wallets vs. Long-term Users",
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"}
        )
        st.plotly_chart(fig_lifetime, use_container_width=True)

    with col_g2:
        st.markdown("#### 2. Total Ether Received vs. Sent")
        fig_scatter = px.scatter(
            df.sample(min(500, len(df))),
            x="Total_Ether_Received", y="Total_Ether_Sent",
            color="FLAG",
            labels={"FLAG": "Fraud"},
            title="Received vs. Sent Volume: Draining Patterns in Malicious Wallets",
            color_discrete_map={0: "#3B82F6", 1: "#EF4444"},
            log_x=True, log_y=True
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("📄 View Raw Dataset Sample"):
        st.dataframe(df.head(15), use_container_width=True)

# ================= TAB 2: MODEL BENCHMARK =================
with tabs[1]:
    st.subheader("🤖 Machine Learning Model Benchmarking")
    st.markdown("Comparison across **Random Forest (Balanced)**, **XGBoost (Cost-Sensitive)**, **Logistic Regression**, and **Isolation Forest (Unsupervised)**.")

    results = detector.results
    
    # Table of Results
    summary_data = []
    for m_name, metrics in results.items():
        summary_data.append({
            "Model": m_name,
            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Precision": f"{metrics['precision']*100:.2f}%",
            "Recall": f"{metrics['recall']*100:.2f}%",
            "F1-Score": f"{metrics['f1']*100:.2f}%",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}"
        })
    df_results = pd.DataFrame(summary_data)
    st.dataframe(df_results, use_container_width=True)

    st.markdown("---")

    col_roc, col_cm = st.columns([1, 1])
    
    with col_roc:
        st.markdown("#### ROC Curves Comparison")
        fig_roc = go.Figure()
        colors = {"Random Forest (Balanced)": "#2563EB", "XGBoost": "#10B981", "Logistic Regression": "#F59E0B", "Isolation Forest": "#8B5CF6"}
        for m_name, metrics in results.items():
            fig_roc.add_trace(go.Scatter(
                x=metrics["fpr"], y=metrics["tpr"],
                mode='lines',
                name=f"{m_name} (AUC={metrics['roc_auc']:.2f})",
                line=dict(color=colors.get(m_name, "#000"), width=2)
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash', color='gray')))
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate (Recall)", height=400)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown("#### Confusion Matrix (XGBoost)")
        cm = results["XGBoost"]["confusion_matrix"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=["Legitimate (0)", "Fraud (1)"],
            y=["Legitimate (0)", "Fraud (1)"],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

# ================= TAB 3: FEATURE IMPORTANCE =================
with tabs[2]:
    st.subheader("🔍 On-Chain Predictive Feature Importance")
    st.markdown("Which blockchain behavioral patterns most reliably distinguish malicious accounts from genuine users?")

    df_imp = detector.get_feature_importance("XGBoost")
    
    fig_imp = px.bar(
        df_imp,
        x="Importance", y="Feature",
        orientation="h",
        title="XGBoost Normalized Feature Importances",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=480)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    ### 💡 Technical Insights for Fraud Investigators:
    1. **`Time_Diff_First_Last_Mins` (Top Predictor):** Sybil wallets and phishing distributors operate with extremely short operational lifespans (<12 hours) to avoid blacklisting.
    2. **`Total_Ether_Balance` (Balance Drain):** Fraudulent accounts typically maintain near-zero balances after laundering through mixers or external wallets.
    3. **`Avg_min_between_sent_tnx` (Burst Velocity):** Automated bot exploitation yields millisecond/second-level burst transaction velocities rarely seen in human-operated wallets.
    """)

# ================= TAB 4: LIVE SIMULATOR =================
with tabs[3]:
    st.subheader("⚡ Real-Time Wallet Risk & Fraud Probability Scanner")
    st.markdown("Adjust wallet behavioral parameters below to evaluate fraud risk in real-time.")

    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("##### ⏱️ Temporal Behavior")
        sim_time_diff = st.slider("Account Lifespan (Minutes)", min_value=1, max_value=60000, value=350, help="Time between first and last recorded transaction")
        sim_avg_sent_min = st.slider("Avg Minutes Between Sent Txns", min_value=0.1, max_value=2000.0, value=12.0)
        sim_avg_recv_min = st.slider("Avg Minutes Between Recv Txns", min_value=0.1, max_value=2000.0, value=25.0)

    with col_s2:
        st.markdown("##### 💰 Transaction Volumes")
        sim_eth_recv = st.number_input("Total Ether Received", min_value=0.0, max_value=10000.0, value=85.0)
        sim_eth_sent = st.number_input("Total Ether Sent", min_value=0.0, max_value=10000.0, value=84.5)
        sim_balance = st.number_input("Current Balance (ETH)", min_value=0.0, max_value=5000.0, value=0.02)

    with col_s3:
        st.markdown("##### 📜 Contract & ERC-20 Interactions")
        sim_sent_tnx = st.slider("Sent Transaction Count", min_value=1, max_value=500, value=18)
        sim_recv_tnx = st.slider("Received Transaction Count", min_value=1, max_value=500, value=12)
        sim_contracts = st.selectbox("Number of Created Contracts", [0, 1, 2, 5, 10], index=1)
        sim_erc20 = st.number_input("ERC20 Ether Volume", min_value=0.0, max_value=500.0, value=40.0)

    sim_features = {
        "Time_Diff_First_Last_Mins": sim_time_diff,
        "Sent_tnx": sim_sent_tnx,
        "Received_tnx": sim_recv_tnx,
        "Avg_min_between_sent_tnx": sim_avg_sent_min,
        "Avg_min_between_received_tnx": sim_avg_recv_min,
        "Total_Ether_Sent": sim_eth_sent,
        "Total_Ether_Received": sim_eth_recv,
        "Total_Ether_Balance": sim_balance,
        "Number_of_Created_Contracts": sim_contracts,
        "ERC20_Total_Ether_Received": sim_erc20
    }

    st.markdown("---")
    
    risk_out = detector.predict_wallet(sim_features)
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 2px solid {risk_out['color']}; border-radius: 12px; padding: 20px; text-align: center;">
            <h4 style="margin: 0; color: #1E293B;">Calculated Fraud Risk</h4>
            <h1 style="margin: 10px 0; font-size: 3rem; color: {risk_out['color']};">{risk_out['risk_score_100']}%</h1>
            <span style="background-color: {risk_out['color']}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                {risk_out['tier']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_res2:
        st.markdown("#### 🚨 Risk Indicators Detected")
        for flag in risk_out["flags"]:
            st.markdown(f"- ⚠️ **{flag}**")

# ================= TAB 5: ARCHITECTURE =================
with tabs[4]:
    st.subheader("🏗️ System Architecture & ML Governance")
    st.markdown("""
    ```
    [ Ethereum Node / Mempool Telemetry ]
                   │
                   ▼
    [ On-Chain Feature Engineering Pipeline ]
      (Temporal deltas, Balance drain ratio, Contract creation velocity)
                   │
                   ▼
    [ Leakage-Safe Stratified Split & Standard Scaling ]
                   │
                   ├──► [ Cost-Sensitive XGBoost (Primary Classifier) ]
                   │
                   ├──► [ Balanced Random Forest (Ensemble Verification) ]
                   │
                   └──► [ Isolation Forest (Zero-Day Anomaly Detection) ]
                                   │
                                   ▼
                    [ Real-Time Risk Score (0-100%) ] ──► [ Emergency Smart Contract Pause / Alert ]
    ```
    ### Practical Applications & Security Architecture:
    High-throughput machine learning models monitoring on-chain mempool velocity allow decentralized protocols and exchanges to trigger automated circuit breakers before malicious transactions are confirmed on-chain.
    """)
