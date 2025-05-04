import os
from PIL import Image

# 原始資料夾與輸出資料夾
input_dir = './data/inputs'
output_dir = './data/outputs'

os.makedirs(output_dir, exist_ok=True)

# 目標尺寸
target_size = (1280, 720)
exts = ('.jpg', '.jpeg', '.png')
for fname in os.listdir(input_dir):
    if fname.lower().endswith(exts):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        try:
            img = Image.open(in_path)
            if img.width > target_size[0] or img.height > target_size[1]:
                img = img.resize(target_size, Image.LANCZOS)
            else:
                img = img.copy()
            img.save(out_path)
            print(f"已處理: {fname}")
        except Exception as e:
            print(f"處理 {fname} 時發生錯誤: {e}")