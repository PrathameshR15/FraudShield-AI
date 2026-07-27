import os
import requests
import json

def load_dotenv(dotenv_path=".env"):
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY", "")

# Test Gemini models list
models_to_test = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]

for model_name in models_to_test:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": "Ping"}]}]}
    res = requests.post(url, json=payload, headers=headers, timeout=10)
    print(f"Model [{model_name}]: HTTP {res.status_code}")
    if res.status_code == 200:
        print(f"  -> SUCCESS! Working response: {res.json()['candidates'][0]['content']['parts'][0]['text'].strip()}")
    else:
        print(f"  -> {res.text[:150]}")
