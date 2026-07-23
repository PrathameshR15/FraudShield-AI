import os
from PIL import Image
import numpy as np

img_path = os.path.join("temp_uploads", "itp edited.jpeg")
if os.path.exists(img_path):
    with Image.open(img_path) as img:
        print(f"Image Format: {img.format}")
        print(f"Image Size: {img.size}")
        print(f"Image Mode: {img.mode}")
        
        # Check unique colors in the image
        arr = np.array(img)
        unique_colors = len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))
        print(f"Number of unique colors: {unique_colors}")
else:
    print("File not found.")
