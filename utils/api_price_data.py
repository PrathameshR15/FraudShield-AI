import json
import os
from price_fetcher import fetch_live_fraction_price

# Directory to store API data
DATA_DIR = os.path.join(os.path.dirname(__file__), "api_data")
os.makedirs(DATA_DIR, exist_ok=True)

def save_price_to_file(filename: str = "fraction_price.json") -> str:
    """Fetch the live price and save it to a JSON file.

    Returns the absolute path of the written file.
    """
    price = fetch_live_fraction_price()
    data = {"price": price}
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[APIData] Saved live fraction price to {file_path}")
    return file_path

if __name__ == "__main__":
    save_price_to_file()
