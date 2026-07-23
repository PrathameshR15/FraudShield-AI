import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.csv_loader import load_transactions_csv, map_row_to_backend_tx
from utils.downloader import download_screenshot
from utils.price_fetcher import fetch_live_fraction_price
from ocr.ocr_engine import extract_fields, _empty_extracted
from features.feature_engineering import generate_feature_vector
from models.predict import FraudPredictor
from utils.suspicious_db import log_suspicious_activity

class LivePipelineProcessor:
    """
    Live Transaction Pipeline Processor.
    Integrates live CSV records, screenshot downloads, live fraction prices,
    OCR analysis, feature engineering, ML predictions, and audit logging.
    """
    
    def __init__(self, csv_path: str = os.path.join("dataset", "purchase_request.csv")):
        self.csv_path = csv_path
        self.predictor = FraudPredictor()
        
    def process_single_transaction(
        self,
        row_dict: Dict[str, Any],
        live_price: float,
        existing_utrs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Processes a single transaction row from the CSV.
        """
        # 1. Map row to backend transaction dictionary & calculate Expected Amount dynamically
        backend_tx = map_row_to_backend_tx(row_dict, live_price)
        upload_time = backend_tx.get("purchase_date") or datetime.now().isoformat()
        
        # 2. Download corresponding payment screenshot automatically
        screenshot_filename = backend_tx.get("payment_screenshot", "")
        local_image_path = None
        if screenshot_filename:
            local_image_path = download_screenshot(screenshot_filename)
            
        # 3. Run OCR on downloaded screenshot (or use fallback empty OCR if image missing/failed)
        if local_image_path and os.path.exists(local_image_path):
            try:
                ocr_data = extract_fields(local_image_path, csv_context=backend_tx)
            except Exception as e:
                print(f"[LivePipeline Error] OCR failed for {screenshot_filename}: {e}")
                ocr_data = _empty_extracted()
        else:
            print(f"[LivePipeline Warning] Screenshot missing/failed to download for tx {backend_tx['payment_id']}. Using default OCR structure.")
            ocr_data = _empty_extracted()
            
        # 4. Generate feature vector using existing feature engineering module
        features = generate_feature_vector(
            backend_tx=backend_tx,
            ocr_data=ocr_data,
            upload_time=upload_time,
            existing_utrs=existing_utrs or []
        )
        
        # 5. Run prediction using existing ML Model + Rule Engine
        prediction, confidence = self.predictor.predict(features)
        
        # 6. Log suspicious activity using existing audit logging module
        if prediction == "NO":
            live_check_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_suspicious_activity(
                backend_tx_dict=backend_tx,
                ocr_data=ocr_data,
                features=features,
                upload_time=live_check_timestamp
            )
            
        return {
            "payment_id": backend_tx["payment_id"],
            "user_id": backend_tx["user_id"],
            "screenshot": screenshot_filename,
            "fractions_count": backend_tx["fraction_count"],
            "fraction_price": live_price,
            "expected_amount": backend_tx["expected_amount"],
            "ocr_paid_amount": ocr_data.get("paid_amount"),
            "amount_match": features.get("amount_match"),
            "receiver_match": features.get("receiver_match"),
            "utr_match": features.get("utr_match"),
            "duplicate_utr": features.get("duplicate_utr"),
            "prediction": prediction,
            "confidence": confidence,
            "ocr_data": ocr_data,
            "features": features,
            "backend_tx": backend_tx
        }

    def process_all_live_transactions(
        self,
        limit: Optional[int] = None,
        start_index: int = 0
    ) -> Dict[str, Any]:
        """
        Reads CSV transactions, fetches live fraction price, downloads screenshots,
        runs OCR + Feature Engineering + Prediction + Logging for each row.
        """
        print(f"[LivePipeline] Starting live transaction processing from {self.csv_path}...")
        
        # Step A: Read CSV
        df = load_transactions_csv(self.csv_path)
        if df.empty:
            return {"status": "error", "message": "CSV dataframe is empty or missing."}
            
        total_rows = len(df)
        print(f"[LivePipeline] Loaded {total_rows} records from CSV.")
        
        # Step B: Fetch live node fraction price from API
        live_price = fetch_live_fraction_price()
        print(f"[LivePipeline] Current Node Fraction Price: Rs. {live_price:.2f}")
        
        # Compile existing completed UTRs from CSV for duplicate check
        existing_utrs = [
            str(utr).strip() for utr in df["transaction_id"].dropna() 
            if str(utr).strip() and str(utr).strip().lower() not in ["nan", "null"]
        ]
        
        end_index = total_rows if limit is None else min(start_index + limit, total_rows)
        subset_df = df.iloc[start_index:end_index]
        
        results = []
        passed_count = 0
        flagged_count = 0
        error_count = 0
        
        for idx, row in subset_df.iterrows():
            try:
                row_dict = row.to_dict()
                print(f"\n--- Processing Row {idx+1}/{total_rows} (Tx ID: {row_dict.get('id')}) ---")
                
                res = self.process_single_transaction(
                    row_dict=row_dict,
                    live_price=live_price,
                    existing_utrs=existing_utrs
                )
                
                if res["prediction"] == "YES":
                    passed_count += 1
                else:
                    flagged_count += 1
                    
                results.append(res)
                
            except Exception as e:
                print(f"[LivePipeline Error] Failed to process row {idx}: {e}")
                error_count += 1
                
        summary = {
            "status": "success",
            "total_processed": len(results),
            "total_csv_records": total_rows,
            "live_fraction_price": live_price,
            "valid_count": passed_count,
            "flagged_count": flagged_count,
            "error_count": error_count,
            "results": results
        }
        
        print("\n=== Live Pipeline Batch Processing Summary ===")
        print(f"Total Processed : {len(results)}")
        print(f"Valid Payments  : {passed_count}")
        print(f"Flagged (Fraud) : {flagged_count}")
        print(f"Errors Encountered: {error_count}")
        
        return summary
