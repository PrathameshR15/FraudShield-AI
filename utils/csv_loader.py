import os
import re
import pandas as pd
from typing import List, Dict, Any, Optional

DEFAULT_CSV_PATH = os.path.join("dataset", "purchase_request.csv")

def clean_string_value(val: Any) -> str:
    """
    Sanitizes raw cell values from CSV:
    - Strips leading/trailing spaces, non-breaking spaces (\xa0), zero-width spaces (\u200b).
    - Removes trailing '.0' if pandas parsed an integer column as float text.
    - Standardizes 'nan', 'null', 'none', 'n/a' to empty string.
    """
    if pd.isna(val) or val is None:
        return ""
        
    s = str(val).replace("\xa0", " ").replace("\u200b", "").strip()
    s_clean = s.strip("'\"")
    
    if s_clean.lower() in ["nan", "null", "none", "n/a", "undefined", "nat"]:
        return ""
        
    if s_clean.endswith(".0") and re.match(r"^\d+\.0$", s_clean):
        s_clean = s_clean[:-2]
        
    return s_clean

def clean_currency_amount(val: Any) -> Optional[float]:
    """
    Parses currency values like '₹ 1,190.00', 'INR 1190', '$1,250.50' accurately into float.
    Returns None if missing or invalid.
    """
    s = clean_string_value(val)
    if not s:
        return None
        
    # Remove currency symbols, commas, spaces
    cleaned = re.sub(r"[^\d.-]", "", s)
    try:
        if cleaned:
            return round(float(cleaned), 2)
    except (ValueError, TypeError):
        pass
    return None

def get_aliased_value(row_dict: Dict[str, Any], aliases: List[str]) -> str:
    """
    Retrieves the value for the first matching column name from a list of aliases (case-insensitive).
    """
    lower_map = {str(k).strip().lower(): v for k, v in row_dict.items()}
    for alias in aliases:
        a_lower = alias.lower()
        if a_lower in lower_map:
            val = clean_string_value(lower_map[a_lower])
            if val:
                return val
    return ""

def map_row_to_backend_tx(row_dict: Dict[str, Any], live_price: float = 4953.50) -> Dict[str, Any]:
    """
    Maps a raw CSV row dictionary into a standardized backend transaction object.
    Calculates expected amount dynamically using fraction count and live fraction price.
    """
    rec_id = get_aliased_value(row_dict, ["id", "transaction_id", "rec_id", "payment_id"]) or "1"
    user_id = get_aliased_value(row_dict, ["user_id", "userid", "user"]) or "202"
    sender_name = get_aliased_value(row_dict, ["sender_name", "user_name", "name", "paid_by", "banking_name"]) or "Standard User"
    
    fractions_str = get_aliased_value(row_dict, ["fraction_count", "fractions", "quantity", "count", "num_fractions"])
    try:
        fraction_count = int(fractions_str) if fractions_str else 2
    except ValueError:
        fraction_count = 2

    raw_amt_str = get_aliased_value(row_dict, ["paid_amount", "expected_amount", "amount", "total_price", "price"])
    parsed_amt = clean_currency_amount(raw_amt_str)
    if parsed_amt is not None and parsed_amt > 0:
        expected_amount = parsed_amt
    else:
        expected_amount = round(fraction_count * live_price + 1000, 2)

    status = get_aliased_value(row_dict, ["payment_status", "status", "tx_status"]).upper() or "COMPLETED"
    date_str = get_aliased_value(row_dict, ["created_at", "purchase_date", "date", "timestamp", "time"])
    
    # Extract transaction_id and utr specifically from CSV columns
    tx_id_val = get_aliased_value(row_dict, ["transaction_id", "tx_id", "txn_id", "google_transaction_id", "ref_no"])
    utr_val = get_aliased_value(row_dict, ["utr", "utr_number", "expected_utr", "upi_ref_no", "rrn"])
    
    # Single parameter fallback rule:
    # If Transaction ID is missing but UTR is available, use UTR.
    # If UTR is missing but Transaction ID is available, use Transaction ID.
    if not tx_id_val and utr_val:
        effective_ref = utr_val
    elif not utr_val and tx_id_val:
        effective_ref = tx_id_val
    else:
        effective_ref = utr_val if utr_val else tx_id_val

    final_tx_id = tx_id_val or effective_ref or rec_id
    final_utr = utr_val or effective_ref

    screenshot_fn = get_aliased_value(row_dict, ["payment_screenshot", "screenshot", "image", "file", "receipt"])
    rcv_name = get_aliased_value(row_dict, ["receiver_name", "merchant_name", "to_name"]) or "MASTERSTROKE TECHNOSOFT PRIVATE LIMITED"
    rcv_upi = get_aliased_value(row_dict, ["receiver_upi", "merchant_upi", "to_upi"]) or "merchantaumb100011870@aubank"
    sender_bank = get_aliased_value(row_dict, ["sender_bank", "bank_name", "bank"]) or "Union Bank of India"
    last4 = get_aliased_value(row_dict, ["account_last4", "account_last_4", "last4"]) or ""

    return {
        "id": rec_id,
        "payment_id": rec_id,
        "transaction_id": final_tx_id,
        "user_id": user_id,
        "sender_name": sender_name,
        "fraction_count": fraction_count,
        "fraction_price": live_price,
        "expected_amount": expected_amount,
        "payment_status": status,
        "purchase_date": date_str,
        "transaction_details": {
            "expected_utr": final_utr,
            "transaction_id": final_tx_id
        },
        "expected_utr": final_utr,
        "payment_screenshot": screenshot_fn,
        "receiver_name": rcv_name,
        "receiver_upi": rcv_upi,
        "sender_bank": sender_bank,
        "account_last4": last4
    }

def validate_transaction_id(utr_val: Any) -> Dict[str, Any]:
    """
    Validates transaction_id/utr completeness before passing to fraud detection.
    Checks if value is missing, formatted in scientific notation, or complete.
    """
    clean_utr = clean_string_value(utr_val)
    if not clean_utr:
        return {"is_valid": False, "reason": "Missing transaction_id", "clean_utr": ""}
        
    # Check for scientific notation format (e.g. 4.67806E+11)
    if re.search(r'[eE][+-]?\d+', clean_utr):
        return {
            "is_valid": False,
            "reason": f"Corrupted by scientific notation ('{clean_utr}') - precision lost before import",
            "clean_utr": clean_utr
        }
        
    return {"is_valid": True, "reason": "Complete string transaction_id", "clean_utr": clean_utr}

def check_scientific_notation_utr(df: pd.DataFrame) -> List[str]:
    """
    Detects if any transaction_id or utr values in the dataframe are formatted in scientific notation
    (e.g., 4.67806E+11) indicating precision loss prior to CSV import.
    Logs warnings detailing necessary re-export steps.
    """
    corrupted_ids = []
    utr_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["transaction_id", "utr", "utr_number", "ref_no", "upi_ref_no"]:
            utr_col = c
            break
            
    if not utr_col:
        return corrupted_ids
        
    sci_pattern = re.compile(r'^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$')
    
    for idx, val in df[utr_col].items():
        str_val = clean_string_value(val)
        if sci_pattern.match(str_val):
            corrupted_ids.append(str_val)
            
    if corrupted_ids:
        print(f"[CSVLoader WARNING] Detected {len(corrupted_ids)} UTR(s) in scientific notation format (e.g., '{corrupted_ids[0]}').")
        print("[CSVLoader WARNING] The original UTR may have been corrupted before import (e.g. opened & saved in Excel).")
        print("[CSVLoader CRITICAL] Precision was lost prior to CSV loading. The exact UTR cannot be reconstructed algorithmically.")
        print("[CSVLoader RECOVERY] The CSV must be re-exported directly from the original database as text to preserve exact UTR strings.")
        
    return corrupted_ids

def load_transactions_csv(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Reads transactions from a CSV file with high accuracy:
    - Multi-encoding fallback (utf-8-sig for Excel UTF-8 BOM, utf-8, latin1, cp1252).
    - Forces all columns to str dtype (dtype=str, keep_default_na=False) to prevent numeric conversion or scientific notation loss.
    - Merges dataset/verified_purchases.csv records if present.
    """
    if not os.path.exists(csv_path):
        df_main = pd.DataFrame()
    else:
        df_main = _load_single_csv(csv_path)
        
    verified_path = os.path.join("dataset", "verified_purchases.csv")
    if os.path.exists(verified_path) and os.path.abspath(verified_path) != os.path.abspath(csv_path):
        df_ver = _load_single_csv(verified_path)
        if not df_ver.empty:
            df_main = pd.concat([df_main, df_ver], ignore_index=True)
            
    return df_main

def _load_single_csv(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    df = None
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(
                csv_path,
                dtype=str,
                encoding=enc,
                keep_default_na=False,
                skipinitialspace=True,
                low_memory=False
            )
            break
        except Exception as e:
            last_err = e
            continue
            
    if df is None:
        return pd.DataFrame()

    try:
        df.columns = [str(col).strip().replace("\ufeff", "").strip("'\"") for col in df.columns]
        for col in df.columns:
            df[col] = df[col].apply(clean_string_value)
        return df
    except Exception:
        return pd.DataFrame()

def append_transaction_to_csv(new_row: Dict[str, Any], csv_path: str = DEFAULT_CSV_PATH) -> bool:
    """
    Appends a newly verified genuine transaction record to purchase_request.csv accurately.
    """
    try:
        df = _load_single_csv(csv_path)
        next_id = 1
        if not df.empty and "id" in df.columns:
            try:
                numeric_ids = pd.to_numeric(df["id"], errors="coerce").dropna()
                if not numeric_ids.empty:
                    next_id = int(numeric_ids.max()) + 1
            except Exception:
                next_id = len(df) + 1
                
        new_row["id"] = str(next_id)
        if "created_at" not in new_row or not new_row["created_at"]:
            from datetime import datetime
            new_row["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        new_df = pd.DataFrame([new_row])
        if not df.empty:
            cols = list(df.columns)
            for c in cols:
                if c not in new_df.columns:
                    new_df[c] = ""
            new_df = new_df[cols]
            new_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
        else:
            new_df.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8')
            
        print(f"[CSVLoader] Successfully appended new genuine transaction ID #{next_id} to {csv_path}")
        
        # Also append to dataset/verified_purchases.csv
        append_to_verified_purchases_csv(new_row)
        return True
    except Exception as e:
        print(f"[CSVLoader Error] Failed to append transaction to CSV: {e}")
        return False

def append_to_verified_purchases_csv(new_row: Dict[str, Any]) -> bool:
    """
    Appends a newly verified genuine transaction to dataset/verified_purchases.csv (creating file if needed).
    """
    try:
        verified_path = os.path.join("dataset", "verified_purchases.csv")
        file_exists = os.path.exists(verified_path)
        
        df = _load_single_csv(verified_path) if file_exists else pd.DataFrame()
        new_df = pd.DataFrame([new_row])
        
        if file_exists and not df.empty:
            cols = list(df.columns)
            for c in cols:
                if c not in new_df.columns:
                    new_df[c] = ""
            new_df = new_df[cols]
            new_df.to_csv(verified_path, mode='a', header=False, index=False, encoding='utf-8')
        else:
            new_df.to_csv(verified_path, mode='w', header=True, index=False, encoding='utf-8')
            
        print(f"[CSVLoader] Verified record written to {verified_path}")
        return True
    except Exception as e:
        print(f"[CSVLoader Error] Failed writing to verified_purchases.csv: {e}")
        return False
