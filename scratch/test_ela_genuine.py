import os
import gc
from PIL import Image, ImageChops
import numpy as np

def run_ela(image_path: str, quality: int = 90):
    temp_ela_path = image_path + f".ela_temp.jpg"
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.save(temp_ela_path, "JPEG", quality=quality)
            
            with Image.open(temp_ela_path) as resaved:
                diff = ImageChops.difference(img, resaved)
                # Use uint8 to save memory!
                diff_arr = np.array(diff, dtype=np.uint8)
                mean_diff = float(np.mean(diff_arr))
                max_diff = int(np.max(diff_arr))
                std_diff = float(np.std(diff_arr))
                
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)
        return mean_diff, max_diff, std_diff
    except Exception as e:
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)
        print(f"Error for {image_path}: {e}")
        return None

# Check files
files = [
    r"C:\Users\Masterstroke\.gemini\antigravity-ide\brain\494b6e98-dec3-49c4-8bc4-025953c92d7a\media__1784290072654.jpg",
    r"temp_uploads\itp edited.jpeg"
]

for f in files:
    if os.path.exists(f):
        print(f"ELA details for {os.path.basename(f)}:")
        for q in [95, 90, 85]:
            res = run_ela(f, q)
            if res:
                mean_d, max_d, std_d = res
                ratio = max_d / (mean_d + 1e-5)
                print(f"  Quality {q}: Mean={mean_d:.4f}, Max={max_d}, Std={std_d:.4f}, Ratio={ratio:.2f}")
    else:
        print(f"File not found: {f}")
