import json
import os
import urllib.request
from urllib.error import URLError, HTTPError

# URL for the /fractions endpoint (no auth required)
URL = "https://api.mstblockchain.com/fractions"

# Directory to store the fetched data
DATA_DIR = os.path.join(os.path.dirname(__file__), "api_data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_fractions():
    """Fetch the list of fractions from the MST blockchain API.
    Returns the parsed JSON dictionary (or None on error) and saves the
    response to ``api_data/fractions.json``.
    """
    try:
        req = urllib.request.Request(
            URL,
            headers={"User-Agent": "FraudShield-Client/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            # Save the full response
            file_path = os.path.join(DATA_DIR, "fractions.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[Fractions] Saved response to {file_path}")
            # Print a short preview for quick verification
            preview = json.dumps(data, indent=2)[:500]
            if len(json.dumps(data, indent=2)) > 500:
                preview += "..."
            print(preview)
            return data
    except HTTPError as e:
        print(f"[Fractions] HTTP error {e.code}: {e.reason}")
    except URLError as e:
        print(f"[Fractions] URL error: {e.reason}")
    except Exception as e:
        print(f"[Fractions] Unexpected error: {e}")
    return None

if __name__ == "__main__":
    fetch_fractions()
