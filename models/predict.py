import os
import pickle
import pandas as pd
from typing import Dict, Any, Tuple

class FraudPredictor:
    """Predicts whether a payment is valid (YES) or suspicious (NO) using the trained model."""
    
    def __init__(self, model_path: str = "models/fraud_model.pkl"):
        self.model_path = model_path
        self.model_payload = None
        self.model = None
        self.feature_cols = []
        self.load_model()
        
    def load_model(self):
        """Load the trained model and associated feature info."""
        if not os.path.exists(self.model_path):
            print(f"[Predictor] Model file {self.model_path} not found. Needs to be trained first.")
            return
            
        try:
            with open(self.model_path, "rb") as f:
                self.model_payload = pickle.load(f)
            self.model = self.model_payload["model"]
            self.feature_cols = self.model_payload["feature_cols"]
            print(f"[Predictor] Successfully loaded {self.model_payload['model_name']} model.")
        except Exception as e:
            print(f"[Predictor] Error loading model: {e}")
            
    def predict(self, raw_features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Runs prediction on raw features engineered from backend and OCR.
        
        Returns:
            Tuple[str, float]: ("YES" if valid, "NO" if suspicious, probability of validity)
        """
        if self.model is None:
            # Fallback rule-based evaluation if model is not trained/loaded yet
            print("[Predictor Warning] Model not loaded. Falling back to rule-based verification.")
            return self._rule_based_check(raw_features)
            
        try:
            # Process single row into DataFrame matching training layout
            df_row = pd.DataFrame([raw_features])
            
            # Map categorical YES/NO to 1/0
            binary_cols = ["amount_match", "time_check", "receiver_match", "duplicate_utr"]
            for col in binary_cols:
                if col in df_row.columns:
                    df_row[col] = df_row[col].map({"YES": 1, "NO": 0}).fillna(0).astype(int)
                    
            # Normalize statuses
            if "payment_status" in df_row.columns:
                df_row["payment_status_val"] = df_row["payment_status"].astype(str).str.upper().map({
                    "SUCCESS": 1.0, "COMPLETED": 1.0, "ACCEPTED": 1.0, "APPROVED": 1.0, "PAID": 1.0, "SUCCESSFUL": 1.0, "PENDING": 0.5, "FAILED": 0.0, "REJECTED": 0.0
                }).fillna(1.0)
            else:
                df_row["payment_status_val"] = 0.0
                
            if "backend_status" in df_row.columns:
                df_row["backend_status_val"] = df_row["backend_status"].map({
                    "COMPLETED": 1.0, "PENDING": 0.5, "FAILED": 0.0
                }).fillna(0.0)
            else:
                df_row["backend_status_val"] = 0.0
                
            # Filter and order features
            X = df_row[self.feature_cols]
            
            # Predict probability of fraud (class 1)
            # fraud_prob is probability of class 1 (fraud)
            # valid_prob is 1 - fraud_prob
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                fraud_prob = float(probs[1])
            else:
                # Fallback for models without predict_proba
                pred = self.model.predict(X)[0]
                fraud_prob = float(pred)
                
            valid_prob = 1.0 - fraud_prob
            
            # Prediction:
            # YES -> Payment is valid (fraud_prob < 0.5)
            # NO -> Payment is suspicious (fraud_prob >= 0.5)
            # We can also add strict check: if time_difference > 10 or amount_match == "NO", we override or keep it
            # But the ML model should learn this if trained properly.
            # To be 100% robust against edge cases, we can combine ML prediction with a strict rule safety net:
            is_suspicious_by_rules = (
                raw_features.get("amount_match") == "NO" or
                raw_features.get("time_check") == "NO" or
                raw_features.get("receiver_match") == "NO" or
                raw_features.get("utr_match") == "NO" or
                raw_features.get("status_match") == "NO" or
                raw_features.get("sender_match") == "NO" or
                raw_features.get("duplicate_utr") == "YES" or
                raw_features.get("image_genuine") is False or
                raw_features.get("is_duplicate_upload") is True
            )
            
            if is_suspicious_by_rules:
                # Strict security override: if critical check fails, mark suspicious
                return "NO", min(valid_prob, 0.10)
            else:
                # If everything matches, it is 100% valid
                return "YES", max(valid_prob, 0.95)
                
        except Exception as e:
            print(f"[Predictor] Error running inference: {e}. Using rule-based fallback.")
            return self._rule_based_check(raw_features)
            
    def _rule_based_check(self, raw_features: Dict[str, Any]) -> Tuple[str, float]:
        """Strict rule-based verification when ML model is unavailable."""
        amount_ok = raw_features.get("amount_match") == "YES"
        time_ok = raw_features.get("time_check") == "YES"
        receiver_ok = raw_features.get("receiver_match") == "YES"
        utr_ok = raw_features.get("utr_match") == "YES"
        status_ok = raw_features.get("status_match") == "YES"
        sender_ok = raw_features.get("sender_match") == "YES"
        duplicate_utr = raw_features.get("duplicate_utr") == "YES"
        image_ok = raw_features.get("image_genuine", True) is not False
        duplicate_upload = raw_features.get("is_duplicate_upload") is True
        
        valid = amount_ok and time_ok and receiver_ok and utr_ok and status_ok and sender_ok and not duplicate_utr and image_ok and not duplicate_upload
        
        if valid:
            return "YES", 0.95
        else:
            return "NO", 0.05

