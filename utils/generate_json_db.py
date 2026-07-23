import os
import json
import random
from datetime import datetime, timedelta

def main():
    names = [
        "Aarav Sharma", "Aditya Patel", "Vikram Singh", "Ananya Rao", 
        "Siddharth Nair", "Neha Gupta", "Rohan Mehta", "Priya Iyer", 
        "Karan Malhotra", "Ishaan Verma", "Diya Sen", "Kabir Bose",
        "Meera Joshi", "Rahul Kapoor", "Riya Dutta", "Arjun Reddy"
    ]
    receiver_names = ["FRACTION INVEST INC", "FRACTIONS CO", "FRACTION PAY"]
    receiver_upis = ["fractioninvest@ybl", "fractions@paytm", "fractionco@okaxis"]
    statuses = ["COMPLETED", "FAILED", "PENDING"]
    
    base_time = datetime(2026, 7, 16, 10, 0, 0)
    db = {}
    
    random.seed(42)
    for i in range(1, 101):
        payment_id = f"PAY_{1000 + i}"
        user_id = f"USER_{500 + random.randint(1, 20)}"
        user_name = random.choice(names)
        fraction_count = random.choice([5, 10, 20, 50, 100, 250])
        fraction_price = random.choice([100.0, 250.0, 500.0])
        expected_amount = float(fraction_count * fraction_price)
        
        # Stagger transaction times
        purchase_date = (base_time + timedelta(minutes=15 * i)).isoformat()
        payment_status = random.choices(statuses, weights=[0.85, 0.10, 0.05], k=1)[0]
        
        utr = None
        if payment_status == "COMPLETED":
            utr = "".join([str(random.randint(0, 9)) for _ in range(12)])
            
        rec_idx = random.randint(0, len(receiver_names) - 1)
        
        db[payment_id] = {
            "payment_id": payment_id,
            "user_id": user_id,
            "user_name": user_name,
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
        
    os.makedirs("dataset", exist_ok=True)
    with open("dataset/transactions.json", "w") as f:
        json.dump(db, f, indent=2)
        
    print(f"Successfully generated dataset/transactions.json with {len(db)} records.")

if __name__ == "__main__":
    main()
