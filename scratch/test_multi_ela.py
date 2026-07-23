import os
from PIL import Image, ImageChops
import numpy as np

def analyze_ela_levels(image_path: str):
    print(f"Analyzing {image_path}:")
    for q in [95, 92, 90, 88, 85, 80]:
        temp_ela_path = image_path + f".ela_{q}.jpg"
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img.save(temp_ela_path, "JPEG", quality=q)
                
                with Image.open(temp_ela_path) as resaved:
                    diff = ImageChops.difference(img, resaved)
                    diff_arr = np.array(diff)
                    mean_diff = float(np.mean(diff_arr))
                    max_diff = int(np.max(diff_arr))
                    std_diff = float(np.std(diff_arr))
                    print(f"  Quality {q}: Mean={mean_diff:.4f}, Max={max_diff}, Std={std_diff:.4f}, Max/Mean Ratio={max_diff/(mean_diff+1e-5):.1f}")
            if os.path.exists(temp_ela_path):
                os.remove(temp_ela_path)
        except Exception as e:
            print(f"Error for quality {q}: {e}")

img_path = os.path.join("temp_uploads", "itp edited.jpeg")
if os.path.exists(img_path):
    analyze_ela_levels(img_path)
else:
    print("itp edited.jpeg not found.")
