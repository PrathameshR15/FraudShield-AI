import os
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
image_path = os.path.join("temp_uploads", "WhatsApp Image 2026-07-17 at 5.31.03 PM.jpeg")

print(f"Running OCR on: {image_path}")
result, elapse = ocr(image_path)

if result:
    print("\n--- RAW OCR LINES ---")
    for idx, line in enumerate(result):
        print(f"{idx}: '{line[1]}' (conf: {line[2]})")
else:
    print("No text detected or failed to load image.")
