import os
from datetime import datetime, timedelta
import random
from features.feature_engineering import generate_feature_vector
from models.predict import FraudPredictor

def run_tests():
    print("=== Starting Payment Fraud System Integration Tests ===")
    
    try:
        predictor = FraudPredictor()
        
        # 2. Mock a backend transaction
        tx_time = datetime.now() - timedelta(minutes=3)  # Transaction made 3 mins ago
        backend_tx = {
            "payment_id": "PAY_TEST_999",
            "user_id": "USER_TEST_1",
            "fraction_count": 10,
            "fraction_price": 250.0,
            "expected_amount": 2500.0,
            "purchase_date": tx_time.isoformat(),
            "payment_status": "COMPLETED",
            "transaction_details": {
                "receiver_name": "FRACTION INVEST INC",
                "receiver_upi": "fractioninvest@ybl",
                "expected_utr": "987654321012"
            }
        }
        
        # List of completed UTRs (for duplicate checks)
        completed_utrs = ["987654321012", "111122223333"]
        
        # 3. Test Cases Configuration
        test_cases = [
            {
                "name": "Perfect Valid Payment",
                "mode": "valid",
                "upload_delay_mins": 4,  # upload 4 mins after tx (valid)
                "expected_prediction": "YES",
                "utr_list": completed_utrs
            },
            {
                "name": "Fraud: Mismatched Amount",
                "mode": "mismatch_amount",
                "upload_delay_mins": 4,
                "expected_prediction": "NO",
                "utr_list": completed_utrs
            },
            {
                "name": "Fraud: Mismatched Receiver",
                "mode": "mismatch_receiver",
                "upload_delay_mins": 4,
                "expected_prediction": "NO",
                "utr_list": completed_utrs
            },
            {
                "name": "Fraud: Time Exceeded (> 10 mins)",
                "mode": "mismatch_time",
                "upload_delay_mins": 15,  # upload 15 mins after tx (expired)
                "expected_prediction": "NO",
                "utr_list": completed_utrs
            },
            {
                "name": "Fraud: Duplicate UTR Reuse",
                "mode": "duplicate_utr",
                "upload_delay_mins": 4,
                "expected_prediction": "NO",
                "utr_list": completed_utrs
            },
            {
                "name": "Fraud: Incorrect Payment Status",
                "mode": "failed_status",
                "upload_delay_mins": 4,
                "expected_prediction": "NO",
                "utr_list": completed_utrs
            }
        ]
        
        # 4. Execute test cases
        failed_tests = 0
        for tc in test_cases:
            print(f"\nRunning test: {tc['name']}")
            
            # Simulate receipt details dictionary programmatically
            expected_amount = float(backend_tx["expected_amount"])
            paid_amount = expected_amount
            if tc["mode"] == "mismatch_amount":
                paid_amount = expected_amount - 100.0  # Mismatched amount
                
            screenshot_time = tx_time + timedelta(minutes=random.randint(1, 5))
            if tc["mode"] == "mismatch_time":
                screenshot_time = tx_time + timedelta(minutes=15)
                
            receiver_name = backend_tx["transaction_details"]["receiver_name"]
            receiver_upi = backend_tx["transaction_details"]["receiver_upi"]
            if tc["mode"] == "mismatch_receiver":
                receiver_name = "WRONG RECEIVER CO"
                receiver_upi = "wrongreceiver@upi"
                
            utr = backend_tx["transaction_details"]["expected_utr"]
            if tc["mode"] == "duplicate_utr":
                utr = "111122223333"  # Existing completed UTR
                
            receipt_status = "SUCCESS"
            if tc["mode"] == "failed_status":
                receipt_status = "FAILED"
                
            simulated_ocr_res = {
                "paid_amount": float(paid_amount),
                "payment_time": screenshot_time.isoformat(),
                "receiver_name": receiver_name,
                "receiver_upi": receiver_upi,
                "utr": utr,
                "payment_status": receipt_status,
                "ocr_confidence": 0.98
            }
            
            print(f"  Simulated receipt amount: {simulated_ocr_res['paid_amount']} | UTR: {simulated_ocr_res['utr']} | Status: {simulated_ocr_res['payment_status']}")
            
            # Run Feature Engineering
            upload_time = (tx_time + timedelta(minutes=tc["upload_delay_mins"])).isoformat()
            features = generate_feature_vector(
                backend_tx=backend_tx,
                ocr_data=simulated_ocr_res,
                upload_time=upload_time,
                existing_utrs=tc["utr_list"]
            )
            
            # Run Predictor
            prediction, confidence = predictor.predict(features)
            print(f"  Model Prediction: {prediction} (Confidence: {confidence:.2%})")
            
            # Validate output matches expectation
            if prediction == tc["expected_prediction"]:
                print(f"  [PASS] {tc['name']}")
            else:
                print(f"  [FAIL] {tc['name']} (Expected {tc['expected_prediction']}, got {prediction})")
                failed_tests += 1
                
        # 5. Print Results
        print("\n=== Test Results Summary ===")
        total_tests = len(test_cases)
        passed_tests = total_tests - failed_tests
        print(f"Passed: {passed_tests}/{total_tests}")
        
        if failed_tests == 0:
            print("SUCCESS: All pipeline integration tests passed!")
            return True
        else:
            print(f"FAILURE: {failed_tests} integration tests failed.")
            return False
            
    except Exception as e:
        print(f"Error executing test pipeline: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
