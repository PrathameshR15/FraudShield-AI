import os
import sys

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath("."))

def test_csv_matching():
    print("=== Testing CSV Record Matching & REUPLOADED / UNIQUE Classification ===")
    from main import find_matching_csv_record
    
    # 1. Test UTR Match for existing row ID #2 (UTR: 467806124565, User ID: 273)
    match_utr = find_matching_csv_record(extracted_utr="467806124565")
    assert match_utr is not None, "Should find matching record for UTR 467806124565"
    assert match_utr["matched_id"] == "2", f"Matched ID should be 2, got {match_utr['matched_id']}"
    assert match_utr["matched_user_id"] == "273", f"Matched User ID should be 273, got {match_utr['matched_user_id']}"
    upload_status = "REUPLOADED" if match_utr else "UNIQUE"
    assert upload_status == "REUPLOADED", "Existing CSV UTR should be classified as REUPLOADED"
    print(f"[SUCCESS] UTR Match: Found Record ID #{match_utr['matched_id']} (User ID: {match_utr['matched_user_id']}, UTR: {match_utr['matched_transaction_id']}) -> Classified as '{upload_status}'")
    
    # 2. Test Screenshot Filename Match for existing row ID #1 (File: 0cfa33e506744f9bda43599c6fbcb436.jpg, User ID: 202)
    match_file = find_matching_csv_record(uploaded_filename="0cfa33e506744f9bda43599c6fbcb436.jpg")
    assert match_file is not None, "Should find matching record for screenshot filename"
    assert match_file["matched_id"] == "1", f"Matched ID should be 1, got {match_file['matched_id']}"
    assert match_file["matched_user_id"] == "202", f"Matched User ID should be 202, got {match_file['matched_user_id']}"
    upload_status_file = "REUPLOADED" if match_file else "UNIQUE"
    assert upload_status_file == "REUPLOADED", "Existing filename should be classified as REUPLOADED"
    print(f"[SUCCESS] Filename Match: Found Record ID #{match_file['matched_id']} (User ID: {match_file['matched_user_id']}, File: {match_file['matched_screenshot']}) -> Classified as '{upload_status_file}'")
    
    # 3. Test Non-matching UTR (Unique upload)
    unique_match = find_matching_csv_record(extracted_utr="99998888777766665555")
    assert unique_match is None, "Unique UTR should return None"
    upload_status_unique = "REUPLOADED" if unique_match else "UNIQUE"
    assert upload_status_unique == "UNIQUE", "Non-existent UTR should be classified as UNIQUE"
    print(f"[SUCCESS] Unique Record Check: Returned None for non-existent UTR -> Classified as '{upload_status_unique}'")

    print("\nALL CSV MATCHING & REUPLOADED/UNIQUE CLASSIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_csv_matching()
