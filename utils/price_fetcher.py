import urllib.request
import json

API_URL = "https://api.mstblockchain.com/purchase/node-fraction-price"
DEFAULT_PRICE = 4000.0

def fetch_live_fraction_price() -> float:
    """
    Fetches the live fraction price from the blockchain API via POST.
    Falls back to a default price of 4000.0 if the API call fails.
    """
    try:
        req = urllib.request.Request(
            API_URL,
            data=b"{}",
            headers={
                "User-Agent": "FraudShield-Shield-Agent/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        # Timeout after 5 seconds to keep the application responsive
        with urllib.request.urlopen(req, timeout=5) as response:
            body = response.read().decode("utf-8").strip()
            data = json.loads(body)
            price = float(data["currentNodeFractionPrice"])
            print(f"[PriceFetcher] Fetched live fraction price: Rs. {price:.2f}")
            return price
    except Exception as e:
        print(f"[PriceFetcher] API call failed: {e}. Falling back to default Rs. {DEFAULT_PRICE:.2f}")
        return DEFAULT_PRICE

if __name__ == "__main__":
    # Test execution
    live_price = fetch_live_fraction_price()
    print(f"Live Price: {live_price}")
