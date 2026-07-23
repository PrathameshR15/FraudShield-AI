import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mock Payments & Fractions Backend")

import os
import json

# In-memory storage for transaction records
transactions_db: Dict[str, dict] = {}

class TransactionDetails(BaseModel):
    receiver_name: str
    receiver_upi: str
    expected_utr: Optional[str] = None

class TransactionModel(BaseModel):
    payment_id: str
    user_id: str
    user_name: Optional[str] = "Unknown"
    fraction_count: int
    fraction_price: float
    expected_amount: float
    purchase_date: str
    payment_status: str
    transaction_details: TransactionDetails

def init_mock_data(n: int = 150):
    """Load mock transactions from JSON database or generate them as fallback."""
    global transactions_db
    json_path = os.path.join("dataset", "transactions.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            transactions_db.update(data)
            print(f"[Backend] Successfully loaded {len(transactions_db)} transactions from {json_path}")
            return
        except Exception as e:
            print(f"[Backend] Error loading transactions.json: {e}. Generating fallback data...")
            
    random.seed(42)  # For reproducibility
    receiver_names = ["FRACTION INVEST INC", "FRACTIONS CO", "FRACTION PAY"]
    receiver_upis = ["fractioninvest@ybl", "fractions@paytm", "fractionco@okaxis"]
    statuses = ["COMPLETED", "FAILED", "PENDING"]
    
    base_time = datetime(2026, 7, 16, 10, 0, 0)
    
    for i in range(1, n + 1):
        payment_id = f"PAY_{1000 + i}"
        user_id = f"USER_{500 + random.randint(1, 30)}"
        fraction_count = random.choice([5, 10, 20, 50, 100, 250])
        fraction_price = random.choice([100.0, 250.0, 500.0])
        expected_amount = float(fraction_count * fraction_price)
        purchase_date = (base_time + timedelta(minutes=15 * i)).isoformat()
        payment_status = random.choices(statuses, weights=[0.85, 0.10, 0.05], k=1)[0]
        
        utr = None
        if payment_status == "COMPLETED":
            utr = "".join([str(random.randint(0, 9)) for _ in range(12)])
            
        rec_idx = random.randint(0, len(receiver_names) - 1)
        
        transactions_db[payment_id] = {
            "payment_id": payment_id,
            "user_id": user_id,
            "user_name": f"User {user_id.split('_')[1]}",
            "fraction_count": fraction_count,
            "fraction_price": fraction_price,
            "expected_amount": expected_amount,
            "purchase_date": purchase_date,
            "payment_status": payment_status,
            "transaction_details": {
                "receiver_name": receiver_names[rec_idx],
                "receiver_upi": receiver_upis[rec_idx],
                "expected_utr": utr
            }
        }

# Generate mock records upon initialization
init_mock_data()

@app.get("/api/transactions", response_model=List[TransactionModel])
def get_all_transactions():
    """Retrieve all backend transaction records."""
    return list(transactions_db.values())

@app.get("/api/transactions/{payment_id}", response_model=TransactionModel)
def get_transaction(payment_id: str):
    """Retrieve transaction record for a specific payment ID."""
    if payment_id not in transactions_db:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transactions_db[payment_id]

@app.post("/api/add-transaction")
def add_transaction(transaction: TransactionModel):
    """Add a new mock transaction manually for testing."""
    transactions_db[transaction.payment_id] = transaction.dict()
    try:
        json_path = os.path.join("dataset", "transactions.json")
        os.makedirs("dataset", exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(transactions_db, f, indent=2)
    except Exception as e:
        print(f"[Backend Warning] Could not save transaction to json: {e}")
    return {"success": True, "payment_id": transaction.payment_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
