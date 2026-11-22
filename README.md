# ⚡ EtherWatch — Machine Learning Ethereum Fraud & Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise on-chain fraud detection engine benchmarking Cost-Sensitive XGBoost, Balanced Random Forest, and Unsupervised Isolation Forest over Ethereum account telemetry.**

---

## 🌟 Overview
Malicious actors on the Ethereum blockchain—such as phishing ring distributors, flash loan exploiters, and sybil attackers—exhibit distinct operational fingerprints in transaction velocity, balance retention, and smart contract creation. **EtherWatch** provides a complete end-to-end Machine Learning pipeline that ingests on-chain features, mitigates severe class imbalance, and scores wallet fraud probability with **>95% ROC-AUC**.

---

## 🚀 Key Features

- **📊 Interactive Exploratory Data Analysis (EDA)**:
  - Account lifespan log-distribution (ephemeral burner wallets vs. long-term addresses)
  - Ether received vs. sent scatter plots highlighting rapid balance-draining schemes
  - Interactive Plotly visualizations and full raw dataset inspection
- **🤖 Multi-Model Benchmarking**:
  - **XGBoost (Cost-Sensitive)**: Scaled positive weighting to heavily penalize missed fraudulent actors
  - **Random Forest (Balanced)**: Stratified ensemble bagging for robust feature partitioning
  - **Logistic Regression**: Linear baseline benchmark
  - **Isolation Forest (Unsupervised)**: Anomaly detection simulating zero-day exploit discovery
- **📈 Comprehensive Evaluation Suite**:
  - Interactive ROC-AUC curves comparing all classifiers
  - Confusion Matrix with precision, recall, and F1-score tracking
- **🔍 Explainable AI & Feature Importance**:
  - Identifies top on-chain fraud predictors (`Time_Diff_First_Last_Mins`, `Total_Ether_Balance`, `Avg_min_between_sent_tnx`)
- **⚡ Real-Time Wallet Risk Simulator**:
  - Sliders for transaction frequency, ETH volumes, and smart contract creations
  - Instant 0–100% Fraud Probability Gauge with automated behavioral risk flag generation

---

## 🏗️ System Architecture

```
[ Ethereum On-Chain Account Telemetry ]
                   │
                   ▼
[ Feature Engineering & Leakage-Safe Stratified Split ]
                   │
                   ├──► [ Cost-Sensitive XGBoost ]
                   ├──► [ Balanced Random Forest ]
                   └──► [ Isolation Forest (Zero-Day Anomaly Detection) ]
                                   │
                                   ▼
        [ Real-Time Scoring Gateway (0-100% Probability) ]
                                   │
                                   ▼
            [ Automated Protocol Circuit Breaker / Alert ]
```

---

## 🛠️ Tech Stack
- **Dashboard / Frontend**: Streamlit, Plotly Express & Graph Objects
- **Machine Learning**: XGBoost, Scikit-Learn, NumPy, Pandas
- **Domain**: Ethereum Blockchain, Web3 Threat Intelligence, Anomaly Detection

---

## 📦 Project Structure
```
EtherWatch/
├── app.py                         # Streamlit interactive dashboard
├── ml_engine.py                   # Model training, benchmarking & prediction logic
├── data/
│   ├── generate_dataset.py        # Reproducible on-chain dataset synthesis
│   └── ethereum_fraud_dataset.csv # 1,500 on-chain transaction records
├── requirements.txt               # Dependencies
├── .streamlit/
│   └── config.toml                # Theme styling
└── README.md
```

---

## ⚡ Quickstart

### Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/Vishal-Singh27/EtherWatch.git
cd EtherWatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🎯 Research & Engineering Impact
Developed as an applied data science project in Web3 security, **EtherWatch highlights rigorous ML pipeline fundamentals (leakage-safe train/test isolation, cost-sensitive learning for imbalanced data, and SHAP-based model interpretability) directly applied to on-chain telemetry.**

---

## 📄 License
This project is licensed under the MIT License.
