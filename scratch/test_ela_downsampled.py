import os
from PIL import Image, ImageChops
import numpy as np

def run_ela_downsampled(image_path: str, quality: int = 90):
    temp_ela_path = image_path + f".ela_ds.jpg"
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            # Downsample to a max dimension of 400
            max_size = 400
            if img.size[0] > max_size or img.size[1] > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            img.save(temp_ela_path, "JPEG", quality=quality)
            
            with Image.open(temp_ela_path) as resaved:
                diff = ImageChops.difference(img, resaved)
                diff = diff.convert("L")  # Convert diff to grayscale
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
        print(f"Error: {e}")
        return None

files = [
    r"C:\Users\Masterstroke\.gemini\antigravity-ide\brain\494b6e98-dec3-49c4-8bc4-025953c92d7a\media__1784290072654.jpg",
    r"temp_uploads\itp edited.jpeg"
]

for f in files:
    if os.path.exists(f):
        print(f"Downsampled ELA details for {os.path.basename(f)}:")
        for q in [95, 90, 85]:
            res = run_ela_downsampled(f, q)
            if res:
                mean_d, max_d, std_d = res
                print(f"  Quality {q}: Mean={mean_d:.4f}, Max={max_d}, Std={std_d:.4f}")
    else:
        print(f"File not found: {f}")
