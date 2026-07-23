import os
import time
import subprocess
import pickle
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from api.api_client import APIClient
from features.feature_engineering import generate_feature_vector

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def start_mock_backend():
    """Start the mock backend FastAPI server in a background process."""
    print("[Train] Starting mock backend server on http://127.0.0.1:8080...")
    # Run mock_backend.py
    proc = subprocess.Popen(
        [".venv/Scripts/python.exe", "api/mock_backend.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2.0)  # Wait for server to boot
    return proc

def build_dataset(client: APIClient):
    """
    Fetch transactions, simulate receipt inputs programmatically (without OCR/images),
    run feature engineering, and save the dataset.
    """
    print("[Train] Fetching transactions from mock backend...")
    transactions = client.fetch_all_transactions()
    if not transactions:
        raise RuntimeError("No transactions fetched from backend API. Make sure backend is running.")
        
    print(f"[Train] Fetched {len(transactions)} transactions.")
    
    # Store list of expected UTRs to compile UTR list for duplicate check simulation
    all_completed_utrs = [
        tx["transaction_details"]["expected_utr"]
        for tx in transactions
        if tx["payment_status"] == "COMPLETED" and tx["transaction_details"]["expected_utr"]
    ]
    
    dataset_rows = []
    
    # We will generate screenshots for each transaction, introducing different labels
    # To build a realistic ML model, let's distribute valid and fraudulent payments:
    # fraud = YES (1) or NO (0)
    random.seed(42)
    
    print("[Train] Simulating input transaction features programmatically...")
    for i, tx in enumerate(transactions):
        payment_id = tx["payment_id"]
        status = tx["payment_status"]
        
        # Decide fraud scenario and image mode
        # By default, completed backend payments are valid.
        # But we will inject fraud cases to train the model.
        if status == "COMPLETED":
            # 75% valid, 25% fraud
            rand_val = random.random()
            if rand_val < 0.75:
                mode = "valid"
                fraud_label = "NO"
            elif rand_val < 0.82:
                mode = "mismatch_amount"
                fraud_label = "YES"
            elif rand_val < 0.88:
                mode = "mismatch_receiver"
                fraud_label = "YES"
            elif rand_val < 0.94:
                mode = "mismatch_time"
                fraud_label = "YES"
            else:
                mode = "duplicate_utr"
                fraud_label = "YES"
        else:
            # FAILED or PENDING payments
            # If backend is failed/pending, but screenshot says success -> fraud
            # If screenshot matches failed status -> not fraud, just a failed payment transaction
            rand_val = random.random()
            if rand_val < 0.5:
                mode = "failed_status"
                fraud_label = "NO"  # It failed correctly in both places, not a payment fraud, just a failed transaction
            else:
                mode = "valid"  # Image claims success, but backend is failed/pending -> Fraud!
                fraud_label = "YES"
                
        # Calculate expected upload time
        tx_time = datetime.fromisoformat(tx["purchase_date"])
        # Upload time is usually 2-5 minutes after transaction, unless time mismatch is injected
        if mode == "mismatch_time":
            upload_time = tx_time.isoformat()
        else:
            upload_time = (tx_time + timedelta(minutes=random.randint(2, 4))).isoformat()
            
        # Programmatic receipt details simulation
        expected_amount = float(tx["expected_amount"])
        paid_amount = expected_amount
        if mode == "mismatch_amount":
            paid_amount = round(expected_amount * random.choice([0.1, 0.5, 0.9, 1.1]), 2)
            if paid_amount == expected_amount:
                paid_amount += 10.0
                
        screenshot_time = tx_time + timedelta(minutes=random.randint(1, 5))
        if mode == "mismatch_time":
            screenshot_time = tx_time + timedelta(minutes=random.choice([15, 60, 1440, -1440]))
            
        receiver_name = tx["transaction_details"]["receiver_name"]
        receiver_upi = tx["transaction_details"]["receiver_upi"]
        if mode == "mismatch_receiver":
            receiver_name = "PERSONAL WALLET"
            receiver_upi = "personalaccount@okaxis"
            
        utr = tx["transaction_details"]["expected_utr"]
        if not utr:
            utr = "".join([str(random.randint(0, 9)) for _ in range(12)])
        if mode == "duplicate_utr" and all_completed_utrs:
            utr = random.choice(all_completed_utrs)
            
        receipt_status = "SUCCESS"
        if mode == "failed_status" or tx["payment_status"] == "FAILED":
            receipt_status = "FAILED"
        elif tx["payment_status"] == "PENDING":
            receipt_status = "PENDING"
            
        simulated_ocr_details = {
            "paid_amount": float(paid_amount),
            "payment_time": screenshot_time.isoformat(),
            "receiver_name": receiver_name,
            "receiver_upi": receiver_upi,
            "utr": utr,
            "payment_status": receipt_status,
            "ocr_confidence": round(random.uniform(0.95, 0.99), 4)
        }
        
        # Generate feature vector
        features = generate_feature_vector(
            backend_tx=tx,
            ocr_data=simulated_ocr_details,
            upload_time=upload_time,
            existing_utrs=all_completed_utrs
        )
        
        # Add target label
        features["fraud"] = fraud_label
        dataset_rows.append(features)
        
    df = pd.DataFrame(dataset_rows)
    
    # Save CSV
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/transactions.csv", index=False)
    print(f"[Train] Dataset compiled with {len(df)} records and saved to dataset/transactions.csv.")
    return df

def preprocess_data(df: pd.DataFrame):
    """Convert categorical fields to numerical values for model training."""
    # Features required for ML training
    # Convert YES/NO columns to 1/0
    processed_df = df.copy()
    
    binary_cols = ["amount_match", "time_check", "receiver_match", "duplicate_utr"]
    for col in binary_cols:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].map({"YES": 1, "NO": 0}).fillna(0).astype(int)
            
    # Normalize payment status (OCR status)
    # SUCCESS -> 1, PENDING -> 0.5, FAILED -> 0
    if "payment_status" in processed_df.columns:
        processed_df["payment_status_val"] = processed_df["payment_status"].map({
            "SUCCESS": 1.0, "COMPLETED": 1.0, "PENDING": 0.5, "FAILED": 0.0
        }).fillna(0.0)
        
    # Normalize backend status
    if "backend_status" in processed_df.columns:
        processed_df["backend_status_val"] = processed_df["backend_status"].map({
            "COMPLETED": 1.0, "PENDING": 0.5, "FAILED": 0.0
        }).fillna(0.0)
        
    # Columns to use as input features
    feature_cols = [
        "fraction_count",
        "fraction_price",
        "expected_amount",
        "paid_amount",
        "amount_match",
        "time_difference",
        "time_check",
        "receiver_match",
        "duplicate_utr",
        "ocr_confidence",
        "payment_status_val",
        "backend_status_val"
    ]
    
    X = processed_df[feature_cols]
    y = processed_df["fraud"].map({"YES": 1, "NO": 0})
    
    return X, y, feature_cols

def train_and_evaluate(X, y, feature_cols):
    """Train Random Forest and XGBoost, compare performance, and save the best one."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_f1 = f1_score(y_test, rf_preds)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Random Forest F1-Score: {rf_f1:.4f}")
    print(classification_report(y_test, rf_preds))
    
    print("\n--- Training XGBoost ---")
    # Using eval_metric='logloss' to avoid warning
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"XGBoost F1-Score: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_preds))
    
    # Choose best model based on F1-Score (balance of precision and recall for fraud)
    best_model = None
    best_name = ""
    best_score = 0.0
    
    if rf_f1 >= xgb_f1:
        best_model = rf_model
        best_name = "Random Forest"
        best_score = rf_f1
    else:
        best_model = xgb_model
        best_name = "XGBoost"
        best_score = xgb_f1
        
    print(f"\n[Train] Best Model Selected: {best_name} with F1-Score = {best_score:.4f}")
    
    # Save the best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/fraud_model.pkl"
    
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": feature_cols,
        "trained_date": datetime.now().isoformat(),
        "metrics": {
            "rf": {"accuracy": rf_acc, "f1": rf_f1},
            "xgb": {"accuracy": xgb_acc, "f1": xgb_f1}
        }
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"[Train] Model saved to {model_path}")
    
    # Save metrics JSON for the dashboard
    with open("models/model_metrics.json", "w") as f:
        json.dump(payload["metrics"], f, indent=2)
        
    return best_name

def main():
    # Start mock backend
    backend_proc = start_mock_backend()
    
    try:
        # Initialize client
        client = APIClient()
        
        # Compile dataset
        df = build_dataset(client)
        
        # Preprocess features
        X, y, feature_cols = preprocess_data(df)
        
        # Train and evaluate models
        train_and_evaluate(X, y, feature_cols)
        
    finally:
        # Guarantee server shutdown
        print("[Train] Shutting down mock backend server...")
        backend_proc.terminate()
        backend_proc.wait()
        print("[Train] Backend server stopped.")

if __name__ == "__main__":
    main()
