#!/usr/init/env python3
"""
Customer Churn Processing Pipeline
Intentionally written to load the entire dataset into memory at once
to provide a clear target for a pandas chunksize refactor.
"""

import os
import pandas as pd
import numpy as np

def generate_sample_data(filepath="customer_transactions.csv", num_rows=20000):
    """Generates a realistic sample CSV if none exists."""
    np.random.seed(42)
    data = {
        "customer_id": np.random.randint(1000, 9999, size=num_rows),
        "signup_date": pd.date_range(start="2024-01-01", periods=num_rows, freq="min"),
        "total_spend": np.random.exponential(scale=150.0, size=num_rows),
        "support_calls": np.random.poisson(lam=1.5, size=num_rows),
        "churned": np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15])
    }
    df = pd.DataFrame(data)
    # Inject some missing values to make cleaning necessary
    df.loc[np.random.choice(num_rows, 100), "total_spend"] = np.nan
    df.to_csv(filepath, index=False)
    print(f"Generated sample dataset at {filepath} with {num_rows} rows.")

def process_data(filepath):
    print(f"Loading entire dataset from {filepath}...")
    # Intentional full-file memory load (The bottleneck for Turn 1)
    df = pd.read_csv(filepath)
    
    # Cleaning steps
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['total_spend'] = df['total_spend'].fillna(df['total_spend'].mean())
    df = df.dropna(subset=['customer_id'])
    
    # Feature engineering
    df['spend_per_call'] = df['total_spend'] / (df['support_calls'] + 1)
    
    summary = {
        "total_rows": len(df),
        "churn_rate": df['churned'].mean(),
        "avg_spend": df['total_spend'].mean()
    }
    return summary

if __name__ == "__main__":
    csv_path = "customer_transactions.csv"
    if not os.path.exists(csv_path):
        generate_sample_data(csv_path)
    
    metrics = process_data(csv_path)
    print("Processing Complete. Metrics:", metrics)