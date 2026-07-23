import os
from PIL import Image, ImageChops
import numpy as np

def check_local_ela_anomalies(image_path: str, quality: int = 90):
    temp_ela_path = image_path + ".ela_local_temp.jpg"
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((400, 800), Image.Resampling.LANCZOS)
            img.save(temp_ela_path, "JPEG", quality=quality)
            
            with Image.open(temp_ela_path) as resaved:
                diff = ImageChops.difference(img, resaved)
                diff = diff.convert("L")
                diff_arr = np.array(diff, dtype=np.uint8)
                
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)
            
        h, w = diff_arr.shape
        patch_size = 16
        local_stds = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = diff_arr[y:y+patch_size, x:x+patch_size]
                local_stds.append(float(np.std(patch)))
                
        variance_of_stds = float(np.var(local_stds))
        max_std = float(np.max(local_stds))
        return variance_of_stds, max_std
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
        res = check_local_ela_anomalies(f)
        if res:
            var_std, max_std = res
            print(f"{os.path.basename(f)}: Local Var={var_std:.4f}, Max Std={max_std:.4f}")
    else:
        print(f"File not found: {f}")
