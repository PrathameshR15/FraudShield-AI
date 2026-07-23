import os
import sys
from pipeline.live_pipeline import LivePipelineProcessor
from utils.csv_loader import load_transactions_csv
from utils.price_fetcher import fetch_live_fraction_price

def main():
    print("=== Testing Live Transaction Data Fraud Detection Pipeline ===")
    
    csv_path = os.path.join("dataset", "purchase_request.csv")
    if not os.path.exists(csv_path):
        print(f"FAILED: CSV file not found at relative path '{csv_path}'")
        sys.exit(1)
        
    print(f"[1] Verified relative CSV file presence: {csv_path}")
    df = load_transactions_csv(csv_path)
    print(f"[2] Total records loaded from CSV: {len(df)}")
    
    print("[3] Testing Live Fraction Price API...")
    live_price = fetch_live_fraction_price()
    print(f"    Current Node Fraction Price: Rs. {live_price:.2f}")
    assert live_price > 0, "Live price must be greater than zero"
    
    print("\n[4] Running Live Pipeline Processor on first 3 transactions...")
    processor = LivePipelineProcessor(csv_path=csv_path)
    summary = processor.process_all_live_transactions(limit=3)
    
    print("\n=== Live Test Execution Summary ===")
    print(f"Status           : {summary.get('status')}")
    print(f"Total Processed  : {summary.get('total_processed')}")
    print(f"Valid Payments   : {summary.get('valid_count')}")
    print(f"Flagged Payments : {summary.get('flagged_count')}")
    print(f"Errors          : {summary.get('error_count')}")
    
    assert summary.get("total_processed") == 3, "Expected 3 processed records"
    assert summary.get("error_count") == 0, "Expected 0 errors"
    
    print("\nSUCCESS: Live transaction pipeline integration verified successfully!")

if __name__ == "__main__":
    main()
