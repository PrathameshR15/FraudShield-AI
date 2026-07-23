import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.downloader import download_screenshot
from ocr.ocr_engine import extract_fields
from features.feature_engineering import generate_feature_vector
from utils.csv_loader import load_transactions_csv, map_row_to_backend_tx

img_path = download_screenshot("0a4b762342979bae299fe0013fdcf171.jpg")
print("Downloaded screenshot path:", img_path)

ocr_data = extract_fields(img_path)
print("\n--- Raw OCR Extracted Data ---")
print(json.dumps(ocr_data, indent=2))

df = load_transactions_csv()
row_800 = df[df["id"] == "800"].iloc[0].to_dict()
backend_tx = map_row_to_backend_tx(row_800)

print("\n--- Backend Tx ---")
print(json.dumps(backend_tx, indent=2))

features = generate_feature_vector(backend_tx, ocr_data, upload_time="2025-01-16 05:19:57")
print("\n--- Feature Vector ---")
print(json.dumps(features, indent=2))
