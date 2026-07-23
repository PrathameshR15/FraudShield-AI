import os
import json

SUSPICIOUS_DB_PATH = os.path.join("dataset", "suspicious_db.json")
SUSPICIOUS_USERS_PATH = os.path.join("dataset", "suspicious_users.json")
SUSPICIOUS_IMAGES_PATH = os.path.join("dataset", "suspicious_images.json")

def load_suspicious_db() -> dict:
    """Load the suspicious database from dataset/suspicious_db.json."""
    if os.path.exists(SUSPICIOUS_DB_PATH):
        try:
            with open(SUSPICIOUS_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Suspicious DB] Error loading database: {e}")
    return {"suspicious_users": {}, "suspicious_images": {}}

def save_suspicious_db(db: dict):
    """Save the suspicious database to dataset/suspicious_db.json and separate files."""
    try:
        os.makedirs(os.path.dirname(SUSPICIOUS_DB_PATH), exist_ok=True)
        # 1. Save combined DB
        with open(SUSPICIOUS_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
            
        # 2. Save separate users DB
        with open(SUSPICIOUS_USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(db.get("suspicious_users", {}), f, indent=2, ensure_ascii=False)
            
        # 3. Save separate images DB
        with open(SUSPICIOUS_IMAGES_PATH, "w", encoding="utf-8") as f:
            json.dump(db.get("suspicious_images", {}), f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"[Suspicious DB] Error saving database: {e}")

def log_suspicious_activity(backend_tx_dict: dict, ocr_data: dict, features: dict, upload_time: str):
    """Compiles all failed verification parameters and logs them to suspicious_db.json."""
    failed_checks = []
    if features.get("amount_match") == "NO":
        failed_checks.append("Amount Mismatch")
    if features.get("time_check") == "NO":
        failed_checks.append("Timestamp Mismatch/Expired")
    if features.get("receiver_match") == "NO":
        failed_checks.append("Receiver Match Failed")
    if features.get("utr_match") == "NO":
        failed_checks.append("UTR Match Failed")
    if features.get("status_match") == "NO":
        failed_checks.append("Status Match Failed")
    if features.get("sender_match") == "NO":
        failed_checks.append("Sender Match Failed")
    if features.get("duplicate_utr") == "YES":
        failed_checks.append("Duplicate UTR Reuse")
    if features.get("image_genuine") is False:
        failed_checks.extend(ocr_data.get("image_tamper_reasons", ["Image Tampering Detected"]))
    if features.get("is_duplicate_upload") is True:
        failed_checks.append("Duplicate Screenshot Reupload Attempt")
        
    # Extract record ID, transaction ID & amount
    record_id = (
        backend_tx_dict.get("id") or
        backend_tx_dict.get("payment_id") or
        features.get("payment_id") or
        "N/A"
    )

    tx_id = (
        backend_tx_dict.get("transaction_id") or
        backend_tx_dict.get("transaction_details", {}).get("transaction_id") or
        backend_tx_dict.get("transaction_details", {}).get("expected_utr") or
        backend_tx_dict.get("expected_utr") or
        ocr_data.get("google_transaction_id") or
        ocr_data.get("utr") or
        features.get("utr") or
        backend_tx_dict.get("payment_id") or
        "N/A"
    )
    
    raw_amt = (
        backend_tx_dict.get("expected_amount") or
        features.get("expected_amount") or
        ocr_data.get("paid_amount") or
        features.get("paid_amount") or
        0.0
    )
    try:
        amount = float(raw_amt)
    except (TypeError, ValueError):
        amount = 0.0

    # Update database
    s_db = load_suspicious_db()
    
    # Log user
    user_key = backend_tx_dict.get("user_name") or "Unknown User"
    user_entry = s_db["suspicious_users"].setdefault(user_key, {
        "user_name": user_key,
        "flagged_attempts": []
    })
    user_entry["flagged_attempts"].append({
        "id": str(record_id).strip(),
        "payment_id": backend_tx_dict.get("payment_id", "Unknown"),
        "transaction_id": str(tx_id).strip(),
        "amount": amount,
        "timestamp": upload_time,
        "reasons": failed_checks
    })
    
    # Log image
    f_hash = ocr_data.get("file_hash")
    if not f_hash:
        # Fallback hash computed from payment_id
        import hashlib
        payment_id = backend_tx_dict.get("payment_id", "Unknown")
        f_hash = hashlib.md5(payment_id.encode('utf-8')).hexdigest()
        
    s_db["suspicious_images"][f_hash] = {
        "id": str(record_id).strip(),
        "filename": backend_tx_dict.get("payment_id", "Unknown") + ".png",
        "payment_id": backend_tx_dict.get("payment_id", "Unknown"),
        "transaction_id": str(tx_id).strip(),
        "amount": amount,
        "timestamp": upload_time,
        "reasons": failed_checks
    }
        
    save_suspicious_db(s_db)

def delete_suspicious_user(user_name: str) -> bool:
    """Deletes a specific user entry from dataset/suspicious_db.json."""
    db = load_suspicious_db()
    if "suspicious_users" in db and user_name in db["suspicious_users"]:
        del db["suspicious_users"][user_name]
        save_suspicious_db(db)
        return True
    return False

def delete_suspicious_image(file_key: str) -> bool:
    """Deletes a specific image entry from dataset/suspicious_db.json."""
    db = load_suspicious_db()
    if "suspicious_images" in db:
        deleted = False
        if file_key in db["suspicious_images"]:
            del db["suspicious_images"][file_key]
            deleted = True
        else:
            to_del = [k for k, v in db["suspicious_images"].items() if v.get("filename") == file_key or k == file_key]
            for k in to_del:
                del db["suspicious_images"][k]
                deleted = True
        if deleted:
            save_suspicious_db(db)
            return True
    return False

def clear_all_suspicious_logs() -> bool:
    """Clears all suspicious users and images from dataset/suspicious_db.json."""
    empty_db = {"suspicious_users": {}, "suspicious_images": {}}
    save_suspicious_db(empty_db)
    return True
