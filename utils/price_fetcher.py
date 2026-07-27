import urllib.request
import json

API_URL = "https://api.mstblockchain.com/purchase/node-fraction-price"
DEFAULT_PRICE = 4000.0

_cached_price = None
_cached_price_time = 0.0

def fetch_live_fraction_price() -> float:
    """
    Fetches the live fraction price from the blockchain API via POST with a 60s in-memory TTL cache.
    Falls back to a default price of 4000.0 if the API call fails.
    """
    global _cached_price, _cached_price_time
    import time
    now = time.time()
    if _cached_price is not None and (now - _cached_price_time) < 60:
        return _cached_price

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
        # Timeout after 3 seconds to keep the application ultra-responsive
        with urllib.request.urlopen(req, timeout=3) as response:
            body = response.read().decode("utf-8").strip()
            data = json.loads(body)
            price = float(data["currentNodeFractionPrice"])
            print(f"[PriceFetcher] Fetched live fraction price: Rs. {price:.2f}")
            _cached_price = price
            _cached_price_time = now
            return price
    except Exception as e:
        print(f"[PriceFetcher] API call failed: {e}. Falling back to default Rs. {DEFAULT_PRICE:.2f}")
        if _cached_price is not None:
            return _cached_price
        return DEFAULT_PRICE

if __name__ == "__main__":
    # Test execution
    live_price = fetch_live_fraction_price()
    print(f"Live Price: {live_price}")
