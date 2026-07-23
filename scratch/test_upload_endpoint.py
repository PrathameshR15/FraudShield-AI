import os
import requests
import json

url = "http://127.0.0.1:8001/api/verify-uploaded-screenshot"
sample_img = os.path.join("dataset", "screenshots", "3e5c2ea8de7383ee2a16f3f6dedffde3.jpg")

if not os.path.exists(sample_img):
    _sdir = os.path.join("dataset", "screenshots")
    if os.path.exists(_sdir):
        for f in os.listdir(_sdir):
            if f.endswith(".jpg") or f.endswith(".png"):
                sample_img = os.path.join(_sdir, f)
                break

print(f"Testing POST to {url} with image: {sample_img}")

with open(sample_img, "rb") as f:
    files = {"file": (os.path.basename(sample_img), f, "image/jpeg")}
    response = requests.post(url, files=files)

print(f"Response Status Code: {response.status_code}")
data = response.json()
print("Keys in response:", list(data.keys()))
print("Success:", data.get("success"))
print("Prediction:", data.get("status_prediction"))
print("Is Duplicate:", data.get("is_duplicate"))
if data.get("matched_record"):
    print("Matched Record:", data.get("matched_record"))
