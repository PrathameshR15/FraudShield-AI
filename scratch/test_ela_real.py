import os
from PIL import Image, ImageChops
import numpy as np

def check_ela_forgery(image_path: str, quality: int = 90) -> tuple:
    temp_ela_path = image_path + ".ela.jpg"
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.save(temp_ela_path, "JPEG", quality=quality)
            
            with Image.open(temp_ela_path) as resaved:
                diff = ImageChops.difference(img, resaved)
                diff_arr = np.array(diff)
                mean_diff = float(np.mean(diff_arr))
                max_diff = int(np.max(diff_arr))
                
        if os.path.exists(temp_ela_path):
            os.remove(temp_ela_path)
        return mean_diff, max_diff
    except Exception as e:
        return 0.0, 0

upload_dir = "temp_uploads"
print("Analyzing ELA values for uploaded files:")
for fname in os.listdir(upload_dir):
    if fname.endswith(".jpeg") or fname.endswith(".png") or fname.endswith(".jpg"):
        fpath = os.path.join(upload_dir, fname)
        mean_d, max_d = check_ela_forgery(fpath)
        print(f"- {fname}: Mean={mean_d:.4f}, Max={max_d}")
