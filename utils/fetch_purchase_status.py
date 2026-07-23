import json
import urllib.request
from urllib.error import URLError, HTTPError

# URL provided by the user (includes pagination and paymentId filter)
URL = "https://api.mstblockchain.com/fractions/purchase/request/status?skip=1&take=2&cursor=3&direction=prev&paymentId=123456"

def fetch_purchase_status():
    """Fetch purchase request status from the MST blockchain API.
    Returns the parsed JSON (or None on error) and prints a short summary.
    """
    try:
        req = urllib.request.Request(
            URL,
            headers={
                "User-Agent": "FraudShield-Client/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            print("[PurchaseStatus] Fetched data successfully.")
            # Pretty‑print a small excerpt for quick verification
            print(json.dumps(data, indent=2)[:500] + ("..." if len(json.dumps(data)) > 500 else ""))
            return data
    except HTTPError as e:
        print(f"[PurchaseStatus] HTTP error {e.code}: {e.reason}")
    except URLError as e:
        print(f"[PurchaseStatus] URL error: {e.reason}")
    except Exception as e:
        print(f"[PurchaseStatus] Unexpected error: {e}")
    return None

if __name__ == "__main__":
    fetch_purchase_status()
