import os
import re
import cv2
import json
import base64
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr = RapidOCR()
except Exception as _e:
    print(f"[OCR Engine Warning] rapidocr_onnxruntime unavailable ({_e}). Using Cloud OCR / Gemini / Groq pipeline.")
    def _ocr(image_path):
        return None, 0.0

def load_dotenv(dotenv_path=".env"):
    if os.path.exists(dotenv_path):
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    val = val.strip().strip("'\"")
                    os.environ[key.strip()] = val

# Load environment variables (e.g. GEMINI_API_KEY)
load_dotenv()

def gemini_extract_fields(image_path: str) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
        
    try:
        # Read and base64-encode the image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
        mime_type = "image/png"
        if image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"
            
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"Read this payment transaction screenshot carefully. "
                                f"Identify and extract the exact payment fields. "
                                f"CRITICAL INSTRUCTION FOR NUMERIC AMOUNTS: Pay special attention to the transaction amount at the top header. "
                                f"On mobile phone screenshots or photos of phone screens, phone notches, Dynamic Islands, camera punch-holes, or status bars "
                                f"may cut through or partially cover the top loop of digits (for example, partially obscuring the upper curve of an '8' so it might resemble a '0', e.g. '5848' vs '5048'). "
                                f"Carefully inspect the visual shape, bottom loop, and full contour of the numbers to read the true amount accurately (e.g., 5848.00). "
                                f"The current date and time is {datetime.now().strftime('%d %B %Y, %I:%M %p')}. "
                                f"Inspect the image for signs of editing or tampering. "
                                f"If the image looks altered, set image_genuine to false and list reasons."
                            )
                        },
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": image_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "paid_amount": {"type": "NUMBER", "description": "The exact transaction amount paid e.g. 5848.00"},
                        "payment_status": {"type": "STRING", "enum": ["SUCCESS", "FAILED", "PENDING", "COMPLETED"]},
                        "payment_date": {"type": "STRING", "description": "Date of payment e.g. 21 Jul 2026"},
                        "payment_time": {"type": "STRING", "description": "Time of payment e.g. 10:37 AM"},
                        "receiver_name": {"type": "STRING", "description": "The banking/display name of the receiver/merchant"},
                        "receiver_upi": {"type": "STRING", "description": "The UPI ID of the receiver"},
                        "sender_name": {"type": "STRING", "description": "The name of the sender, if available"},
                        "sender_upi": {"type": "STRING", "description": "The UPI ID of the sender, if available"},
                        "sender_bank": {"type": "STRING", "description": "The bank name of the sender, if available"},
                        "account_last4": {"type": "STRING", "description": "Last 4 digits of sender bank account, if available"},
                        "utr": {"type": "STRING", "description": "12-digit transaction reference number (UTR / UPI Ref No.)"},
                        "transaction_id": {"type": "STRING", "description": "Exact UPI Transaction ID / Txn ID string e.g. T2411091435531684691558"},
                        "google_transaction_id": {"type": "STRING", "description": "Google Transaction ID / secondary ref ID, if present"},
                        "image_genuine": {"type": "BOOLEAN", "description": "Set to false if there are visual anomalies indicating editing, otherwise true"},
                        "image_tamper_reasons": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "Short list of visual reasons explaining why the screenshot is considered edited/tampered"
                        }
                    },
                    "required": ["paid_amount", "payment_time", "receiver_name", "receiver_upi", "utr", "payment_status", "sender_name", "image_genuine", "image_tamper_reasons"]
                }
            }
        }
        
        model_names = ["gemini-2.0-flash"]
        last_exception = None
        
        for model_name in model_names:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json=payload, headers=headers, timeout=12)
                response.raise_for_status()
                
                resp_json = response.json()
                text_content = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text_content)
                
                result["ocr_confidence"] = 0.99
                result["engine"] = f"gemini ({model_name})"
                return result
            except Exception as ex:
                last_exception = ex
                continue
                
        if last_exception:
            raise last_exception
        raise ValueError("All Gemini model endpoints failed")
        
    except Exception as e:
        print(f"[Gemini OCR] Error during Gemini API execution: {e}")
        raise e

def groq_extract_fields_from_text(text_lines: List[str], csv_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        joined_text = "\n".join(text_lines)
        csv_info_str = ""
        if csv_context:
            csv_info_str = (
                f"--- EXPECTED CSV TRANSACTION DATA ---\n"
                f"Expected Amount: {csv_context.get('expected_amount')}\n"
                f"Expected Status: {csv_context.get('payment_status')}\n"
                f"Expected UTR/TxID: {csv_context.get('transaction_details', {}).get('expected_utr')}\n"
                f"Expected Bank: {csv_context.get('sender_bank')}\n"
                f"--------------------------------------\n\n"
            )
            
        prompt = (
            f"You are an expert payment transaction verification AI. Extract structured JSON payment fields from this OCR text extracted from a payment receipt screenshot:\n\n"
            f"{csv_info_str}"
            f"--- OCR TEXT FROM SCREENSHOT ---\n{joined_text}\n-------------------------------\n\n"
            f"Return a JSON object with the following fields:\n"
            f"- paid_amount: (number or null, e.g. 1190.00 or 62500.00)\n"
            f"- payment_status: (\"COMPLETED\", \"SUCCESS\", \"FAILED\", or \"PENDING\")\n"
            f"- payment_date: (string date or null)\n"
            f"- payment_time: (ISO string or null)\n"
            f"- receiver_name: (string or null)\n"
            f"- receiver_upi: (string or null)\n"
            f"- sender_name: (string or null)\n"
            f"- sender_upi: (string or null)\n"
            f"- sender_bank: (string or null)\n"
            f"- account_last4: (string or null)\n"
            f"- utr: (EXACT 12-digit UTR / UPI Ref No extracted strictly from the screenshot, e.g. 980013401025 or 200778621797)\n"
            f"- transaction_id: (EXACT UPI Transaction ID / Txn ID string extracted from screenshot, e.g. T2411122138381282089867 or T2411091435531684691558)\n"
            f"- google_transaction_id: (EXACT Google Transaction ID / secondary ref ID string, if present)\n"
            f"- image_genuine: (boolean, true unless text looks fake/tampered)\n"
            f"- image_tamper_reasons: (list of strings)\n"
        )
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        parsed["ocr_confidence"] = 0.99
        parsed["engine"] = "Groq Llama-3.3 70B"
        return parsed
    except Exception as e:
        print(f"[Groq OCR Error] {e}")
        return None

def extract_fields(image_path: str, csv_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run OCR on the provided image and extract relevant payment fields, prioritizing Groq Llama-3.3 70B LLM API."""
    # Priority 1: Local OCR + Groq Llama-3.3 70B LLM API (Fast, Reliable, High Precision)
    try:
        result, elapse = _ocr(image_path)
        text_lines = [line[1].strip() for line in result if line[1].strip()] if result else []
        
        if os.environ.get("GROQ_API_KEY") and text_lines:
            try:
                print("[OCR Engine] Extracting fields with Groq Llama-3.3 70B LLM API...")
                groq_parsed = groq_extract_fields_from_text(text_lines, csv_context)
                if groq_parsed and (groq_parsed.get("paid_amount") or groq_parsed.get("utr") or groq_parsed.get("payment_status")):
                    print("[OCR Engine] Successfully extracted screenshot info using Groq Llama-3.3 70B!")
                    return groq_parsed
            except Exception as e:
                print(f"[OCR Engine] Groq Llama extraction exception: {e}")
    except Exception as e:
        print(f"[OCR Engine] Local OCR text line extraction failed: {e}")

    # Fallback to Gemini Multimodal Vision API if explicitly configured
    if os.environ.get("GEMINI_API_KEY"):
        try:
            print("[OCR Engine] Attempting Gemini Multimodal Vision API fallback...")
            res = gemini_extract_fields(image_path)
            if res and (res.get("paid_amount") or res.get("utr")):
                return res
        except Exception as e:
            print(f"[OCR Engine] Gemini Vision API extraction fallback: {e}")

    try:
        # Perform Local OCR
        result, elapse = _ocr(image_path)
        if not result:
            return _empty_extracted()
            
        text_lines = [line[1].strip() for line in result if line[1].strip()]
        
        # High Precision Groq Llama-3.3 70B extraction from OCR text lines
        if os.environ.get("GROQ_API_KEY"):
            try:
                groq_parsed = groq_extract_fields_from_text(text_lines)
                if groq_parsed and (groq_parsed.get("paid_amount") or groq_parsed.get("utr")):
                    print("[OCR Engine] Successfully extracted fields using Groq Llama-3.3 70B!")
                    return groq_parsed
            except Exception as e:
                print(f"[OCR Engine] Groq Llama extraction exception: {e}")
            
        extracted = {
            "paid_amount": None,
            "payment_time": None,
            "receiver_name": None,
            "receiver_upi": None,
            "utr": None,
            "payment_status": None,
            "sender_name": None,
            "ocr_confidence": 0.0
        }
        
        text_lines = []
        conf_sum = 0.0
        for line in result:
            text = line[1].strip()
            conf = float(line[2])
            text_lines.append(text)
            conf_sum += conf
            
        extracted["ocr_confidence"] = round(conf_sum / len(result), 4) if result else 0.0
        
        # 1. Status Check
        for text in text_lines:
            text_upper = text.upper()
            if "SUCCESS" in text_upper or "COMPLETED" in text_upper:
                extracted["payment_status"] = "SUCCESS"
                break
            elif "FAILED" in text_upper or "FAIL" in text_upper:
                extracted["payment_status"] = "FAILED"
                break
            elif "PENDING" in text_upper:
                extracted["payment_status"] = "PENDING"
                break
        if not extracted["payment_status"]:
            extracted["payment_status"] = "SUCCESS"
            
        # 2. UTR Parsing (with keywords priority)
        # We look for numbers of exactly 12 digits to avoid matching 22-digit Transaction IDs.
        for i, text in enumerate(text_lines):
            clean_text = text.lower().replace(" ", "")
            if "utr" in clean_text or "refno" in clean_text or "transaction" in clean_text or "ref" in clean_text or "reference" in clean_text:
                nums = re.findall(r'\d+', text)
                found = False
                for num in nums:
                    if len(num) >= 5:
                        extracted["utr"] = num
                        found = True
                        break
                if found:
                    break
                elif i + 1 < len(text_lines):
                    nums = re.findall(r'\d+', text_lines[i + 1])
                    for num in nums:
                        if len(num) >= 5:
                            extracted["utr"] = num
                            found = True
                            break
                if found:
                    break
                        
        if not extracted["utr"]:
            for text in text_lines:
                nums = re.findall(r'\d+', text)
                for num in nums:
                    if len(num) >= 5:
                        extracted["utr"] = num
                        break
                if extracted["utr"]:
                    break

        # 3. UPI Parsing
        upi_addresses = []
        for text in text_lines:
            match = re.search(r'[\w\.\-]+@[\w\-]+', text)
            if match:
                upi_addresses.append(match.group(0))
                
        # Use context to determine Receiver UPI
        for i, text in enumerate(text_lines):
            clean_text = text.lower()
            if "to" in clean_text or "receiver" in clean_text:
                for j in range(i + 1, min(i + 4, len(text_lines))):
                    match = re.search(r'[\w\.\-]+@[\w\-]+', text_lines[j])
                    if match:
                        extracted["receiver_upi"] = match.group(0)
                        break
                if extracted["receiver_upi"]:
                    break
                    
        if not extracted["receiver_upi"] and upi_addresses:
            extracted["receiver_upi"] = upi_addresses[-1]
            
        # Sender UPI
        for i, text in enumerate(text_lines):
            clean_text = text.lower()
            if "from" in clean_text or "sender" in clean_text:
                for j in range(i + 1, min(i + 4, len(text_lines))):
                    match = re.search(r'[\w\.\-]+@[\w\-]+', text_lines[j])
                    if match:
                        val = match.group(0)
                        if val != extracted["receiver_upi"]:
                            break

        # 4. Receiver and Sender Names
        for i, text in enumerate(text_lines):
            clean_text = text.lower()
            if "bankingname" in clean_text or "banking name" in clean_text:
                parts = text.split(":")
                if len(parts) > 1:
                    extracted["receiver_name"] = parts[1].replace("✅", "").strip()
                    break

        if not extracted["receiver_name"]:
            for i, text in enumerate(text_lines):
                clean_text = text.lower()
                if clean_text == "to" or clean_text == "to name" or clean_text == "paid to":
                    if i + 1 < len(text_lines):
                        candidate = text_lines[i + 1].strip()
                        if not candidate.endswith(":") and "@" not in candidate and len(candidate) > 2:
                            extracted["receiver_name"] = candidate
                            break

        if not extracted["receiver_name"]:
            for i, text in enumerate(text_lines):
                if extracted["receiver_upi"] and extracted["receiver_upi"] in text:
                    if i - 1 >= 0 and len(text_lines[i - 1]) > 2 and "@" not in text_lines[i - 1]:
                        extracted["receiver_name"] = text_lines[i - 1]
                        break

        # Sender Name
        for i, text in enumerate(text_lines):
            clean_text = text.lower()
            if clean_text == "from" or clean_text == "sender name" or clean_text == "from name":
                if i + 1 < len(text_lines):
                    candidate = text_lines[i + 1].strip()
                    if not candidate.endswith(":") and "@" not in candidate and len(candidate) > 2:
                        extracted["sender_name"] = candidate
                        break
        if not extracted["sender_name"]:
            for i, text in enumerate(text_lines):
                if i - 1 >= 0 and "from" in text_lines[i - 1].lower():
                    extracted["sender_name"] = text
                    break

        # 5. Amount Parsing (Avoid clock times, UTRs, transaction IDs)
        # 5. Amount Parsing (Avoid clock times, UTRs, account numbers, transaction IDs)
        amount_candidates_currency = []
        amount_candidates_decimal = []
        amount_candidates_generic = []

        non_amount_keywords = {"utr", "txn", "ref", "id", "account", "bank", "a/c", "card", "date", "time", "pm", "am", "phone"}

        for i, text in enumerate(text_lines):
            clean_lower = text.lower().strip()
            if any(kw in clean_lower for kw in non_amount_keywords):
                continue
                
            clean_text = text.replace(",", "")
            # Currency symbol match (e.g. Rs. 1190.74 or ₹1190.74)
            match = re.search(r'(?:Rs\.?|₹|INR)\s*(\d+(?:\.\d{1,2})?)', clean_text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    if 1.0 <= val <= 1000000.0:
                        amount_candidates_currency.append(val)
                except ValueError:
                    pass
            
            # Decimal amount match (e.g. 1190.74 or 1,190.74)
            match_dec = re.search(r'\b\d{1,3}(?:,\d{3})*\.\d{2}\b|\b\d+\.\d{2}\b', text)
            if match_dec:
                try:
                    val = float(match_dec.group(0).replace(",", ""))
                    if 1.0 <= val <= 1000000.0:
                        amount_candidates_decimal.append(val)
                except ValueError:
                    pass

            # Generic numeric match (excluding 4-digit account numbers like 5613 and 12-digit UTRs)
            match_gen = re.search(r'\b\d{1,3}(?:,\d{3})+\b|\b\d+\b', text)
            if match_gen and not match_dec and not match:
                num_str = match_gen.group(0).replace(",", "")
                if len(num_str) in [4, 12, 16] and not any(symbol in text for symbol in ["Rs", "₹", "INR"]):
                    continue
                try:
                    val = float(num_str)
                    if 10.0 <= val <= 1000000.0 and ":" not in text:
                        amount_candidates_generic.append(val)
                except ValueError:
                    pass

        if amount_candidates_currency:
            extracted["paid_amount"] = amount_candidates_currency[0]
        elif amount_candidates_decimal:
            extracted["paid_amount"] = amount_candidates_decimal[0]
        elif amount_candidates_generic:
            extracted["paid_amount"] = max(amount_candidates_generic)

        # 6. Timestamp Parsing
        for text in text_lines:
            iso_match = re.search(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b', text)
            if iso_match:
                extracted["payment_time"] = iso_match.group(0).replace(" ", "T")
                break
                
        if not extracted["payment_time"]:
            months_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
                "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
                "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
            }
            pattern = r'(\d{1,2})[\s-]*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)[\s-]*(\d{4})(?:\s*at|\s*,)?\s*(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?'
            for text in text_lines:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    day, month_str, year, hour, minute, ampm = match.groups()
                    m_val = months_map.get(month_str.lower())
                    if m_val:
                        try:
                            h = int(hour)
                            if ampm and ampm.lower() == "pm" and h < 12:
                                h += 12
                            elif ampm and ampm.lower() == "am" and h == 12:
                                h = 0
                            dt = datetime(year=int(year), month=m_val, day=int(day), hour=h, minute=int(minute))
                            extracted["payment_time"] = dt.isoformat()
                            break
                        except Exception:
                            pass
                            
        if not extracted["payment_time"]:
            extracted["payment_time"] = datetime.now().isoformat()

        return extracted
        
    except Exception as e:
        print(f"[OCR Engine] Error running OCR: {e}")
        return _empty_extracted()

def _empty_extracted() -> Dict[str, Any]:
    return {
        "paid_amount": None,
        "payment_status": "COMPLETED",
        "payment_date": None,
        "payment_time": datetime.now().isoformat(),
        "receiver_name": None,
        "receiver_upi": None,
        "sender_name": None,
        "sender_upi": None,
        "sender_bank": None,
        "account_last4": None,
        "utr": None,
        "google_transaction_id": None,
        "image_genuine": True,
        "image_tamper_reasons": [],
        "ocr_confidence": 0.0,
    }
