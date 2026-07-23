# 🛡️ FraudShield AI - Payment Fraud Detection System

**FraudShield AI** is an advanced real-time payment verification engine designed to detect payment receipt fraud, fake UPI screenshots, and duplicate UTR reuploads using Multimodal AI Vision OCR, Machine Learning classification, and automated backend record matching.

---

## ✨ Features
- 🔍 **Multimodal Vision OCR**: Powered by Groq Llama-3.3 70B & Google Gemini Vision API to extract UTR/Transaction IDs, Paid Amounts, Receiver UPIs, Timestamps, and Payment Statuses from screenshots.
- ⚡ **Real-Time Backend Cross-Matching**: Automated pairwise comparison against CSV records (`purchase_request.csv`) to instantly detect mismatched amounts, failed payment statuses, or tampered receipts.
- 🚫 **Duplicate UTR & Screenshot Reupload Detection**: Multi-candidate matching engine that flags any reused transaction ID or reuploaded screenshot image.
- 🧠 **Machine Learning & Rule Engine**: Hybrid Random Forest classifier with dynamic genuineness scoring that scales penalty rates based on amount discrepancies.
- 📊 **Interactive Dashboard UI**: Dynamic side-by-side comparison report, live fraction price ticker, and real-time 🚨 Flagged Suspicious Activity audit log table sorted by latest occurrence.

---

## 🛠️ Tech Stack
- **Backend Framework**: Python, FastAPI, Uvicorn
- **AI & OCR Engine**: Groq Llama-3.3 70B, Google Gemini Multimodal Vision API, RapidOCR
- **Machine Learning**: Scikit-Learn (Random Forest, XGBoost), Pandas, NumPy
- **Frontend**: Vanilla HTML5, Modern CSS Design Tokens, Async JavaScript (Fetch API)

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd Fraud-Detection
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Run the Development Server
```bash
python main.py
```
Open [http://localhost:8001](http://localhost:8001) in your browser.
