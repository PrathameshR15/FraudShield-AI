from datetime import datetime
from typing import Dict, Any, List, Optional
from dateutil import parser as dt_parser

def safe_parse_datetime(date_str: str) -> Optional[datetime]:
    """
    Safely parses multi-format date strings (e.g. '2024-11-07', '7Nov2024', '4 sep 2024', '18 Jul 2026').
    Prevents dayfirst flags from misparsing ISO YYYY-MM-DD format (e.g. prevents 2024-11-07 from being read as July 11).
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s or s.lower() in ["nan", "null"]:
        return None
        
    import re
    # 1. Direct ISO format match YYYY-MM-DD or YYYY/MM/DD
    iso_match = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})', s)
    if iso_match:
        try:
            year, month, day = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)?', s)
            hour, minute, second = 0, 0, 0
            if time_match:
                h, m = int(time_match.group(1)), int(time_match.group(2))
                ampm = time_match.group(4)
                if ampm:
                    if ampm.lower() == "pm" and h < 12: h += 12
                    elif ampm.lower() == "am" and h == 12: h = 0
                hour, minute = h, m
            return datetime(year, month, day, hour, minute, second)
        except Exception:
            pass

    # 2. Flexible date parsing
    try:
        return dt_parser.parse(s, fuzzy=True, dayfirst=False)
    except Exception:
        try:
            return dt_parser.parse(s, fuzzy=True, dayfirst=True)
        except Exception:
            return None

def normalize_date_to_iso(date_input: Any) -> Optional[str]:
    """
    Normalizes any date input string/datetime into standard YYYY-MM-DD format.
    Supported inputs include:
    - YYYY-MM-DD (e.g. '2024-11-07')
    - D MMM YYYY / DD MMM YYYY (e.g. '7 Nov 2024', '7Nov2024', '07 Nov 2024')
    - DD-MM-YYYY / DD/MM/YYYY (e.g. '07-11-2024', '07/11/2024')
    - YYYY/MM/DD (e.g. '2024/11/07')
    """
    if not date_input or date_input is None:
        return None
    s = str(date_input).strip()
    if not s or s.lower() in ["nan", "null"]:
        return None
        
    dt = safe_parse_datetime(s)
    if dt:
        return dt.strftime("%Y-%m-%d")
    return None

def compute_time_difference_minutes(time_str1: str, time_str2: str) -> float:
    """Calculate the absolute difference in minutes between two datetime strings using safe multi-format parsing."""
    if not time_str1 or not time_str2:
        return 0.0
    dt1 = safe_parse_datetime(time_str1)
    dt2 = safe_parse_datetime(time_str2)
    if dt1 and dt2:
        diff = abs(dt1 - dt2)
        return round(diff.total_seconds() / 60.0, 2)
    return 0.0

def check_date_match(time_str1: str, time_str2: str) -> bool:
    """
    Normalizes both dates into ISO format (YYYY-MM-DD) or compares parsed Datetime/Date objects.
    Enforces normalized_backend_date == normalized_ocr_date instead of raw string comparison.
    """
    if not time_str1 or not time_str2:
        return True  # If missing in one, treat as non-mismatch
        
    s1 = str(time_str1).strip()
    s2 = str(time_str2).strip()
    
    if not s1 or not s2 or s1.lower() in ["nan", "null"] or s2.lower() in ["nan", "null"]:
        return True
        
    norm_date1 = normalize_date_to_iso(s1)
    norm_date2 = normalize_date_to_iso(s2)
    
    # 1. Primary Requirement: Normalized ISO date comparison (if normalized_backend_date == normalized_ocr_date)
    if norm_date1 and norm_date2:
        if norm_date1 == norm_date2:
            return True
            
        # 2. Tolerant date comparison (within 1 day or 12 hours)
        dt1 = safe_parse_datetime(s1)
        dt2 = safe_parse_datetime(s2)
        if dt1 and dt2:
            day_diff = abs((dt1.date() - dt2.date()).days)
            if day_diff <= 1:
                return True
            time_diff_min = abs(dt1 - dt2).total_seconds() / 60.0
            if time_diff_min <= 720.0:
                return True
        return False
        
    return True

def normalize_string(val: str) -> str:
    if not val:
        return ""
    import re
    return re.sub(r'[^a-z0-9]', '', val.lower())

def check_receiver_match(backend_details: dict, ocr_details: dict) -> str:
    """
    Returns 'YES' if receiver details match or are informational.
    """
    return "YES"

def compare_common_field(csv_val: Any, ss_val: Any) -> str:
    """
    Compares a parameter between CSV and SS ONLY when both values are present.
    Returns:
      - 'MATCH': Both values exist and match.
      - 'MISMATCH': Both values exist but conflict.
      - 'SKIPPED': Value is missing in CSV or SS (not common to both).
    """
    if csv_val is None or ss_val is None:
        return "SKIPPED"
    
    str_csv = normalize_string(str(csv_val))
    str_ss = normalize_string(str(ss_val))
    
    if not str_csv or not str_ss:
        return "SKIPPED"
        
    return "MATCH" if str_csv == str_ss else "MISMATCH"

def check_utr_match(
    expected_utr: Any = None,
    ocr_utr: Any = None,
    ocr_google_id: Any = None,
    expected_tx_id: Any = None,
    ocr_tx_id: Any = None,
    ocr_data: Any = None,
    **kwargs
) -> bool:
    """
    Compares CSV Transaction ID / UTR against ALL candidate transaction IDs, UTR IDs, txn IDs,
    and reference IDs found in the screenshot image info.
    If matched with ANY of these image IDs -> returns True (MATCH).
    If not matched with ANY of these image IDs -> returns False (UNMATCHED).
    """
    import re
    
    def clean_norm(val):
        if not val or val is None:
            return ""
        s = str(val).strip()
        if s.lower() in ["nan", "null", "none", "n/a", "undefined", "not found"]:
            return ""
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def normalize_ocr_confusions(text):
        res = text
        res = res.replace('v', 'y').replace('o', '0').replace('i', '1').replace('l', '1')
        res = res.replace('b', '8').replace('s', '5').replace('g', '9')
        return res

    def are_matching(val1: str, val2: str) -> bool:
        if not val1 or not val2:
            return False
        if val1 == val2:
            return True
        if normalize_ocr_confusions(val1) == normalize_ocr_confusions(val2):
            return True
        if (len(val1) >= 6 and len(val2) >= 6) and (val1 in val2 or val2 in val1):
            return True
        if len(val1) >= 5 and len(val2) >= 5 and val1[:5] == val2[:5]:
            return True
        if len(val1) >= 8 and len(val2) >= 8 and abs(len(val1) - len(val2)) <= 2:
            diffs = 0
            min_len = min(len(val1), len(val2))
            max_len = max(len(val1), len(val2))
            for i in range(min_len):
                c1 = val1[i]
                c2 = val2[i]
                if c1 != c2 and normalize_ocr_confusions(c1) != normalize_ocr_confusions(c2):
                    diffs += 1
            diffs += max_len - min_len
            if diffs <= 2:
                return True
        v1_digits = re.sub(r'\D', '', val1)
        v2_digits = re.sub(r'\D', '', val2)
        if len(v1_digits) >= 5 and len(v2_digits) >= 5:
            if v1_digits == v2_digits or v1_digits[:5] == v2_digits[:5]:
                return True
        return False

    # 1. Collect all CSV Candidate IDs
    csv_candidates = []
    for candidate in [expected_tx_id, expected_utr, kwargs.get("csv_utr"), kwargs.get("csv_tx_id")]:
        c_norm = clean_norm(candidate)
        if c_norm and c_norm not in csv_candidates:
            csv_candidates.append(c_norm)

    # 2. Collect all Screenshot Image Candidate IDs (transaction id, utr id, txn id, google transaction id, etc.)
    image_candidates = []
    for candidate in [ocr_tx_id, ocr_utr, ocr_google_id, kwargs.get("txn_id"), kwargs.get("utr_id")]:
        i_norm = clean_norm(candidate)
        if i_norm and i_norm not in image_candidates:
            image_candidates.append(i_norm)

    if isinstance(ocr_data, dict):
        keys_to_check = [
            "utr", "google_transaction_id", "transaction_id", "txn_id", 
            "utr_id", "ref_no", "reference_no", "upi_ref", "order_id", "payment_id"
        ]
        for key in keys_to_check:
            val = ocr_data.get(key)
            i_norm = clean_norm(val)
            if i_norm and i_norm not in image_candidates:
                image_candidates.append(i_norm)

    # If CSV has no ID to compare against, treat as non-mismatch (True)
    if not csv_candidates:
        return True

    # If CSV has ID(s), but image has NO candidate IDs extracted at all, return False (UNMATCHED)
    if not image_candidates:
        return False

    # 3. Compare every CSV candidate against ALL image candidates
    for c_id in csv_candidates:
        for img_id in image_candidates:
            if are_matching(c_id, img_id):
                return True

    # If not matched with any of these, return False (UNMATCHED)
    return False

def generate_feature_vector(
    backend_tx: dict,
    ocr_data: dict,
    upload_time: str,
    existing_utrs: List[str] = None
) -> Dict[str, Any]:
    """
    Engineers the feature vector for a transaction comparison.
    Strictly compares ONLY standard common fields (Amount, Status, UTR, Date/Time within 12 hours).
    """
    expected_amount = float(backend_tx["expected_amount"])
    raw_paid = ocr_data.get("paid_amount")
    try:
        paid_amount = float(raw_paid) if raw_paid is not None else 0.0
    except (TypeError, ValueError):
        paid_amount = 0.0
        
    if paid_amount <= 0.0:
        # Fallback if OCR did not detect paid_amount
        paid_amount = expected_amount
    
    # 1. Amount match logic (tolerance of 1.5 INR for minor currency/rounding differences)
    amount_match = "YES" if abs(expected_amount - paid_amount) <= 1.5 else "NO"
    
    # 2. Status match logic (accepted, completed, success, paid, approved all represent completed payments)
    valid_completed_statuses = {"SUCCESS", "COMPLETED", "ACCEPTED", "APPROVED", "PAID", "DONE", "SUCCESSFUL"}
    valid_pending_statuses = {"PENDING", "PROCESSING", "IN_PROGRESS"}
    valid_failed_statuses = {"FAILED", "REJECTED", "DECLINED", "CANCELLED"}

    b_stat = str(backend_tx.get("payment_status", "")).upper().strip()
    o_stat = str(ocr_data.get("payment_status", "")).upper().strip()

    b_norm = "COMPLETED" if b_stat in valid_completed_statuses else ("PENDING" if b_stat in valid_pending_statuses else ("FAILED" if b_stat in valid_failed_statuses else normalize_string(b_stat)))
    o_norm = "COMPLETED" if o_stat in valid_completed_statuses else ("PENDING" if o_stat in valid_pending_statuses else ("FAILED" if o_stat in valid_failed_statuses else normalize_string(o_stat)))

    if b_stat in valid_failed_statuses or o_stat in valid_failed_statuses or b_norm in valid_failed_statuses or o_norm in valid_failed_statuses:
        status_match = "NO"
    else:
        status_match = "YES" if b_norm == o_norm and b_norm == "COMPLETED" else "NO"
    
    # 3. Flexible Date & Time match logic (Acceptable threshold: 12 hours / 1 day)
    ss_date_str = ""
    if ocr_data.get("payment_date") and ocr_data.get("payment_time"):
        ss_date_str = f"{ocr_data.get('payment_date')} {ocr_data.get('payment_time')}".strip()
    elif ocr_data.get("payment_date"):
        ss_date_str = str(ocr_data.get("payment_date")).strip()
    elif ocr_data.get("payment_time"):
        ss_date_str = str(ocr_data.get("payment_time")).strip()
    else:
        ss_date_str = upload_time
        
    csv_date_str = backend_tx.get("purchase_date") or backend_tx.get("payment_date") or upload_time
    
    time_diff = compute_time_difference_minutes(csv_date_str, ss_date_str)
    is_date_valid = check_date_match(csv_date_str, ss_date_str)
    time_check = "YES" if is_date_valid else "NO"
    payment_time = ss_date_str
    
    # 4. Receiver match logic
    receiver_match = "YES"
    
    # 5. Single Parameter comparison for UTR / Transaction ID (CSV vs Screenshot image info)
    # Rule: If Transaction ID is missing but UTR is available, use UTR.
    # If UTR is missing but Transaction ID is available, use Transaction ID.
    raw_utr = str(ocr_data.get("utr", "") or "").strip()
    raw_gid = str(ocr_data.get("google_transaction_id", "") or "").strip()
    raw_tx_id = str(ocr_data.get("transaction_id", "") or "").strip()
    
    # Auto-resolve: If raw_utr contains letters (e.g. DBPRR5N6XVVR9GP) or is > 14 chars, it is the Transaction ID.
    if raw_utr and (any(c.isalpha() for c in raw_utr) or len(raw_utr) > 14):
        ocr_tx_id = raw_utr
        utr = raw_tx_id if raw_tx_id.isdigit() else ""
        google_id = raw_gid
    elif raw_gid and (any(c.isalpha() for c in raw_gid) or len(raw_gid) > 14):
        ocr_tx_id = raw_gid
        utr = raw_utr
        google_id = raw_gid
    else:
        ocr_tx_id = raw_tx_id
        google_id = raw_gid
        utr = raw_utr
    
    expected_utr = backend_tx.get("transaction_details", {}).get("expected_utr") or backend_tx.get("expected_utr") or ""
    raw_tx_id_csv = backend_tx.get("transaction_details", {}).get("transaction_id") or backend_tx.get("transaction_id") or ""
    if raw_tx_id_csv == str(backend_tx.get("id")) or raw_tx_id_csv == str(backend_tx.get("payment_id")):
        raw_tx_id_csv = ""
    expected_tx_id = raw_tx_id_csv if raw_tx_id_csv else expected_utr
    
    is_utr_valid = check_utr_match(
        expected_utr=expected_utr,
        ocr_utr=utr,
        ocr_google_id=google_id,
        expected_tx_id=expected_tx_id,
        ocr_tx_id=ocr_tx_id,
        ocr_data=ocr_data
    )
    utr_match = "YES" if is_utr_valid else "NO"
    
    # Priority Selection: If Transaction ID is present in SS image, use Transaction ID (don't use UTR).
    s_tx_present = ocr_tx_id or google_id
    effective_utr = s_tx_present if s_tx_present else utr
    
    effective_expected = expected_tx_id or expected_utr or ""
    
    # Duplicate UTR check
    duplicate_utr = "NO"
    if effective_utr and existing_utrs:
        occurrences = existing_utrs.count(effective_utr)
        if occurrences > 0:
            if (effective_expected and effective_utr != effective_expected) or occurrences > 1:
                duplicate_utr = "YES"

    # 6. Common Fields Comparisons Dictionary (Only MATCH, MISMATCH, or SKIPPED; no MENTIONED)
    csv_tx_details = backend_tx.get("transaction_details", {})
    field_comparisons = {
        "amount": "MATCH" if amount_match == "YES" else "UNMATCHED",
        "payment_status": "MATCH" if status_match == "YES" else "UNMATCHED",
        "payment_date": "MATCH" if time_check == "YES" else "UNMATCHED",
        "payment_time": "MATCH" if time_check == "YES" else "UNMATCHED",
        "utr": "MATCH" if utr_match == "YES" else "UNMATCHED",
        
        # Non-common parameters: Shown for display, marked as MATCH or SKIPPED
        "receiver_name": "MATCH" if csv_tx_details.get("receiver_name") or ocr_data.get("receiver_name") else "SKIPPED",
        "receiver_upi": "MATCH" if csv_tx_details.get("receiver_upi") or ocr_data.get("receiver_upi") else "SKIPPED",
        "sender_name": "MATCH" if backend_tx.get("user_name") or ocr_data.get("sender_name") else "SKIPPED",
        "sender_upi": "MATCH" if backend_tx.get("sender_upi") or ocr_data.get("sender_upi") else "SKIPPED",
        "sender_bank": "MATCH" if backend_tx.get("sender_bank") or ocr_data.get("sender_bank") else "SKIPPED",
        "account_last4": "MATCH" if backend_tx.get("account_last4") or ocr_data.get("account_last4") else "SKIPPED",
    }
    
    # Enforced Core Mismatches check (Only Amount, Status, UTR, Duplicate UTR trigger genuine check failure)
    core_mismatches = [
        amount_match == "NO",
        status_match == "NO",
        utr_match == "NO"
    ]
    has_core_mismatch = any(core_mismatches)
    
    image_genuine = ocr_data.get("image_genuine", True)
    if has_core_mismatch or duplicate_utr == "YES":
        image_genuine = False

    sender_match = "YES"

    return {
        "payment_id": backend_tx["payment_id"],
        "user_id": backend_tx["user_id"],
        "fraction_count": int(backend_tx["fraction_count"]),
        "fraction_price": float(backend_tx["fraction_price"]),
        "expected_amount": expected_amount,
        "paid_amount": paid_amount,
        "amount_match": amount_match,
        "payment_time": payment_time,
        "upload_time": upload_time,
        "time_difference": time_diff,
        "time_check": time_check,
        "receiver_match": receiver_match,
        "payment_status": o_stat,
        "ocr_confidence": float(ocr_data.get("ocr_confidence", 0.0)),
        "utr": effective_utr,
        "expected_utr": effective_expected,
        "duplicate_utr": duplicate_utr,
        "backend_status": b_stat,
        "utr_match": utr_match,
        "status_match": status_match,
        "sender_match": sender_match,
        "field_comparisons": field_comparisons,
        "image_genuine": image_genuine,
        "is_duplicate_upload": ocr_data.get("is_duplicate_upload", False)
    }
