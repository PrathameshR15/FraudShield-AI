import json
import os
import urllib.request
from urllib.error import URLError, HTTPError

# Base URL for the blockchain API
BASE_URL = "https://api.mstblockchain.com"

# Directory to store fetched API data
DATA_DIR = os.path.join(os.path.dirname(__file__), "api_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping of endpoint paths to output filenames
ENDPOINTS = {
    "/fractions": "fractions.json",
    "/fractions/purchase/request/status": "purchase_status.json",
    "/fractions/reward": "reward.json",
    "/fractions/price": "price.json",
    "/fractions/list": "list.json",
    # POST endpoints are placeholders – they need request bodies
    "/fractions/revert": "revert_response.json",
    "/fractions/revert-logs": "revert_logs.json",
    "/fractions/assign-by-admin": "assign_by_admin.json",
    "/fractions/added-by-admin-logs": "added_by_admin_logs.json",
    "/fractions/payement-remove/fraction-revert": "payment_remove_revert.json",
    "/fractions/remove-payment-by-admin-logs": "remove_payment_by_admin_logs.json",
}

def fetch_endpoint(path: str) -> dict | None:
    """Fetch JSON data from a GET endpoint.

    Returns the parsed JSON dictionary, or None on error.
    """
    url = BASE_URL + path
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "FraudShield-Client/1.0",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8").strip()
            if "application/json" in content_type:
                return json.loads(raw)
            else:
                try:
                    return {"value": float(raw)}
                except ValueError:
                    return {"value": raw}
    except (URLError, HTTPError) as e:
        print(f"[API Fetch] Failed to fetch {path}: {e}")
        return None
    except Exception as e:
        print(f"[API Fetch] Unexpected error for {path}: {e}")
        return None

def save_json(data: dict, filename: str) -> None:
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[API Fetch] Saved {filename} to {file_path}")

def fetch_all_and_save() -> None:
    for path, filename in ENDPOINTS.items():
        if path in {"/fractions/revert", "/fractions/assign-by-admin", "/fractions/payement-remove/fraction-revert"}:
            print(f"[API Fetch] Skipping POST endpoint {path} – manual implementation required.")
            continue
        data = fetch_endpoint(path)
        if data is not None:
            save_json(data, filename)
        else:
            print(f"[API Fetch] No data saved for {path}")

if __name__ == "__main__":
    fetch_all_and_save()
