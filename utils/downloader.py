import os
import urllib.request
import urllib.error
from typing import Optional

BASE_SCREENSHOT_URL = "https://api.mstblockchain.com/storage/purchase-request/screenshot/"
DEFAULT_DOWNLOAD_DIR = os.path.join("temp_uploads", "live_screenshots")

def download_screenshot(
    filename: str,
    output_dir: str = DEFAULT_DOWNLOAD_DIR,
    overwrite: bool = False,
    timeout_seconds: int = 4
) -> Optional[str]:
    """
    Downloads a payment screenshot from the live server storage URL.
    
    URL: https://api.mstblockchain.com/storage/purchase-request/screenshot/{filename}
    
    Returns the relative path to the downloaded image file, or None if the download fails.
    """
    if not filename or filename.strip().lower() in ["nan", "null", "none"]:
        print("[Downloader Warning] Empty or invalid screenshot filename provided.")
        return None
        
    filename = filename.strip()
    os.makedirs(output_dir, exist_ok=True)
    local_path = os.path.join(output_dir, filename)
    
    # Check if image is already downloaded locally
    if os.path.exists(local_path) and not overwrite:
        if os.path.getsize(local_path) > 0:
            # Re-use cached copy
            return local_path
            
    download_url = f"{BASE_SCREENSHOT_URL}{filename}"
    print(f"[Downloader] Downloading screenshot from: {download_url}")
    
    try:
        req = urllib.request.Request(
            download_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PaymentFraudDetector/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            if response.status == 200:
                content = response.read()
                with open(local_path, "wb") as f:
                    f.write(content)
                print(f"[Downloader] Successfully saved screenshot ({len(content)} bytes) to {local_path}")
                return local_path
            else:
                print(f"[Downloader Error] HTTP status {response.status} for {download_url}")
                return None
    except urllib.error.HTTPError as e:
        print(f"[Downloader Error] HTTP error {e.code} ({e.reason}) downloading {filename}")
        return None
    except urllib.error.URLError as e:
        print(f"[Downloader Error] URL error downloading {filename}: {e.reason}")
        return None
    except Exception as e:
        print(f"[Downloader Error] Failed to download {filename}: {e}")
        return None
