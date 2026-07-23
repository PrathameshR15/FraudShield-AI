import os
import json
from ocr.ocr_engine import extract_fields

image_path = os.path.join("temp_uploads", "WhatsApp Image 2026-07-17 at 5.31.03 PM.jpeg")
extracted = extract_fields(image_path)

print("\n--- EXTRACTED PARSED FIELDS ---")
print(json.dumps(extracted, indent=2))
