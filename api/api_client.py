import requests
from typing import List, Dict, Optional

class APIClient:
    """Client utility to retrieve transaction data from the payment backend."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url

    def fetch_all_transactions(self) -> List[Dict]:
        """Fetch the full list of transactions from the backend database (for dataset compilation)."""
        try:
            response = requests.get(f"{self.base_url}/api/transactions", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching all transactions: {e}")
            return []

    def fetch_transaction(self, payment_id: str) -> Optional[Dict]:
        """Fetch details of a single transaction by ID (for live verification)."""
        try:
            response = requests.get(f"{self.base_url}/api/transactions/{payment_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching transaction {payment_id}: {e}")
            return None
