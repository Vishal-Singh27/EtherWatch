import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ethereum_fraud_dataset.csv")

class EthereumFraudDetector:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.feature_names = [col for col in self.df.columns if col != "FLAG"]
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.trained = False

    def train_all_models(self, test_size=0.25, random_state=42):
        X = self.df[self.feature_names]
        y = self.df["FLAG"]

        # Leakage-safe Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled

        # 1. Random Forest (Balanced)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=random_state)
        rf.fit(X_train_scaled, y_train)
        self.models["Random Forest (Balanced)"] = rf

        # 2. XGBoost
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        xgb_model = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            random_state=random_state,
            eval_metric="logloss"
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models["XGBoost"] = xgb_model

        # 3. Logistic Regression Baseline
        lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state)
        lr.fit(X_train_scaled, y_train)
        self.models["Logistic Regression"] = lr

        # 4. Isolation Forest (Unsupervised Anomaly Detection)
        iso = IsolationForest(contamination=0.20, random_state=random_state)
        iso.fit(X_train_scaled)
        self.models["Isolation Forest"] = iso

        # Evaluate Models
        for name, model in self.models.items():
            if name == "Isolation Forest":
                # Maps 1 -> 0 (normal), -1 -> 1 (anomaly/fraud)
                raw_pred = model.predict(X_test_scaled)
                y_pred = np.where(raw_pred == -1, 1, 0)
                # Decision function as proxy for probability
                y_prob = -model.decision_function(X_test_scaled)
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-6)
            else:
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)

            self.results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist()
            }

        self.trained = True
        return self.results

    def get_feature_importance(self, model_name="Random Forest (Balanced)"):
        if not self.trained:
            self.train_all_models()
        model = self.models.get(model_name, self.models["Random Forest (Balanced)"])
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            df_imp = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance": imp
            }).sort_values(by="Importance", ascending=False)
            return df_imp
        return pd.DataFrame()

    def predict_wallet(self, features_dict):
        if not self.trained:
            self.train_all_models()
            
        input_df = pd.DataFrame([features_dict])[self.feature_names]
        scaled = self.scaler.transform(input_df)
        
        # Use XGBoost probability as primary
        best_model = self.models["XGBoost"]
        prob = float(best_model.predict_proba(scaled)[0][1])
        
        # Risk tiering
        if prob > 0.70:
            tier = "High Risk (Likely Malicious)"
            color = "#E02424"
        elif prob > 0.35:
            tier = "Medium Risk (Suspicious Pattern)"
            color = "#E3A008"
        else:
            tier = "Low Risk (Legitimate Activity)"
            color = "#0E9F6E"
            
        # Top contributing risk flags
        flags = []
        if features_dict.get("Time_Diff_First_Last_Mins", 1000) < 500:
            flags.append("Ephemeral Account: Wallet lifetime < 8 hours (sybil/burner trait).")
        if features_dict.get("Total_Ether_Balance", 10) < 0.1 and features_dict.get("Total_Ether_Received", 10) > 20:
            flags.append("Rapid Siphoning: 99%+ of received ETH immediately drained.")
        if features_dict.get("Avg_min_between_sent_tnx", 100) < 5:
            flags.append("High-Frequency Automation: Automated script burst behavior.")
        if features_dict.get("Number_of_Created_Contracts", 0) > 1:
            flags.append("Multiple Contract Deployments: Possible phishing contract factory.")
            
        if not flags:
            flags.append("Normal transactional velocity and steady balance retention.")
            
        return {
            "fraud_probability": prob,
            "risk_score_100": round(prob * 100, 1),
            "tier": tier,
            "color": color,
            "flags": flags
        }
