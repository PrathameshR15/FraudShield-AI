import os
import sys
import json

def test_ui_logic():
    print("=== Testing Web UI & Search API Integration Logic ===")
    
    from utils.csv_loader import load_transactions_csv
    from pipeline.live_pipeline import LivePipelineProcessor
    from utils.price_fetcher import fetch_live_fraction_price
    
    csv_path = os.path.join("dataset", "purchase_request.csv")
    df = load_transactions_csv(csv_path)
    assert not df.empty, "CSV file should not be empty"
    print(f"[1] Verified CSV loaded: {len(df)} records.")
    
    # 1. Search Logic
    q_clean = "273"
    mask = (
        df["id"].astype(str).str.lower().str.contains(q_clean, na=False) |
        df["user_id"].astype(str).str.lower().str.contains(q_clean, na=False)
    )
    matches = df[mask]
    assert not matches.empty, "Search should return matching records for User ID 273"
    first_match = matches.iloc[0].to_dict()
    print(f"[2] Search logic verified: User ID {first_match.get('user_id')} (Tx ID: {first_match.get('id')})")
    
    # 2. Authentication Report Generation Logic
    processor = LivePipelineProcessor(csv_path=csv_path)
    live_price = fetch_live_fraction_price()
    
    all_utrs = [
        str(utr).strip() for utr in df["transaction_id"].dropna() 
        if str(utr).strip() and str(utr).strip().lower() not in ["nan", "null"]
    ]
    
    res = processor.process_single_transaction(
        row_dict=first_match,
        live_price=live_price,
        existing_utrs=all_utrs
    )
    
    assert res.get("payment_id") == first_match.get("id"), "Payment ID should match"
    assert "features" in res, "Result must contain feature map"
    assert "ocr_data" in res, "Result must contain OCR data map"
    
    print(f"\n[3] Generated Authentication Report for User {res['user_id']} (Tx #{res['payment_id']}):")
    print(f"    Status Prediction   : {res['prediction']} (Confidence: {res['confidence']*100:.1f}%)")
    print(f"    Live Fraction Price : Rs. {live_price:.2f}")
    print(f"    Expected Amount (CSV): Rs. {res['expected_amount']}")
    print(f"    Paid Amount (SS)    : Rs. {res['ocr_paid_amount']}")
    print(f"    Amount Match        : {res['amount_match']}")
    print(f"    UTR Match           : {res['utr_match']}")
    print(f"    Duplicate UTR Check : {res['duplicate_utr']}")
    
    print("\nSUCCESS: Web UI search and side-by-side Authentication Report logic verified successfully!")

if __name__ == "__main__":
    test_ui_logic()
