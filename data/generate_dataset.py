import numpy as np
import pandas as pd
import os

def create_ethereum_fraud_dataset(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    
    # Approx 80% normal (0), 20% fraud (1) - realistic on-chain skew
    n_fraud = int(n_samples * 0.20)
    n_legit = n_samples - n_fraud
    
    # 1. Normal wallets
    # Older accounts, steady activity, balanced sent/received, low contract creation anomalies
    time_diff_legit = np.random.exponential(scale=50000, size=n_legit) + 1000
    sent_tnx_legit = np.random.poisson(lam=45, size=n_legit) + 1
    recv_tnx_legit = np.random.poisson(lam=55, size=n_legit) + 1
    avg_min_between_sent_legit = np.random.exponential(scale=800, size=n_legit) + 10
    avg_min_between_recv_legit = np.random.exponential(scale=600, size=n_legit) + 10
    total_ether_sent_legit = np.random.lognormal(mean=2.5, sigma=1.2, size=n_legit)
    total_ether_recv_legit = total_ether_sent_legit + np.random.lognormal(mean=1.5, sigma=0.8, size=n_legit)
    total_ether_balance_legit = np.maximum(0, total_ether_recv_legit - total_ether_sent_legit)
    created_contracts_legit = np.random.choice([0, 1], size=n_legit, p=[0.92, 0.08])
    erc20_total_ether_recv_legit = np.random.exponential(scale=15, size=n_legit)
    flag_legit = np.zeros(n_legit, dtype=int)
    
    # 2. Fraudulent wallets (phishers, flash-loan bots, sybils)
    # Very short lifetime, high burst frequency, sudden huge balance draining, zero balance left
    time_diff_fraud = np.random.exponential(scale=800, size=n_fraud) + 20
    sent_tnx_fraud = np.random.poisson(lam=8, size=n_fraud) + 1
    recv_tnx_fraud = np.random.poisson(lam=12, size=n_fraud) + 1
    avg_min_between_sent_fraud = np.random.exponential(scale=15, size=n_fraud) + 1
    avg_min_between_recv_fraud = np.random.exponential(scale=25, size=n_fraud) + 1
    total_ether_recv_fraud = np.random.lognormal(mean=4.0, sigma=1.5, size=n_fraud)
    total_ether_sent_fraud = total_ether_recv_fraud * np.random.uniform(0.95, 1.05, size=n_fraud)  # Drained quickly!
    total_ether_balance_fraud = np.maximum(0, total_ether_recv_fraud - total_ether_sent_fraud) * 0.02
    created_contracts_fraud = np.random.choice([0, 1, 2, 5], size=n_fraud, p=[0.70, 0.15, 0.10, 0.05])
    erc20_total_ether_recv_fraud = np.random.exponential(scale=85, size=n_fraud)
    flag_fraud = np.ones(n_fraud, dtype=int)
    
    # Concatenate
    df = pd.DataFrame({
        "Time_Diff_First_Last_Mins": np.concatenate([time_diff_legit, time_diff_fraud]),
        "Sent_tnx": np.concatenate([sent_tnx_legit, sent_tnx_fraud]),
        "Received_tnx": np.concatenate([recv_tnx_legit, recv_tnx_fraud]),
        "Avg_min_between_sent_tnx": np.concatenate([avg_min_between_sent_legit, avg_min_between_sent_fraud]),
        "Avg_min_between_received_tnx": np.concatenate([avg_min_between_recv_legit, avg_min_between_recv_fraud]),
        "Total_Ether_Sent": np.concatenate([total_ether_sent_legit, total_ether_sent_fraud]),
        "Total_Ether_Received": np.concatenate([total_ether_recv_legit, total_ether_recv_fraud]),
        "Total_Ether_Balance": np.concatenate([total_ether_balance_legit, total_ether_balance_fraud]),
        "Number_of_Created_Contracts": np.concatenate([created_contracts_legit, created_contracts_fraud]),
        "ERC20_Total_Ether_Received": np.concatenate([erc20_total_ether_recv_legit, erc20_total_ether_recv_fraud]),
        "FLAG": np.concatenate([flag_legit, flag_fraud])
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    df = create_ethereum_fraud_dataset(1500)
    out_path = os.path.join(out_dir, "ethereum_fraud_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated dataset with {len(df)} rows at: {out_path}")
    print("Class distribution:\n", df['FLAG'].value_counts())
