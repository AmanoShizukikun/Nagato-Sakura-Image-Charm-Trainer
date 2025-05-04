import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import shutil
import datetime
import io
import multiprocessing
from tqdm import tqdm

# 圖片輸入及輸出路徑
input_directory = "./data/inputs"
output_directory = "./data/outputs"

# 設定使用的處理器核心數
NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))
QUALITY_SETTINGS = {
    100: {"quality": 100, "subsampling": 0},  # 原始品質
    90: {"quality": 90, "subsampling": 0},    # 高品質
    80: {"quality": 80, "subsampling": 0},    # 良好品質
    70: {"quality": 70, "subsampling": 0},    # 中高品質
    60: {"quality": 60, "subsampling": 1},    # 中等品質
    50: {"quality": 50, "subsampling": 1},    # 中低品質
    40: {"quality": 40, "subsampling": 1},    # 較低品質
    30: {"quality": 30, "subsampling": 2},    # 低品質
    20: {"quality": 20, "subsampling": 2},    # 很低品質
    10: {"quality": 10, "subsampling": 2}     # 極低品質
}

def calculate_psnr(original_arr, compressed_arr):
    mse = np.mean((original_arr - compressed_arr) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def process_images(input_dir, output_dir):
    quality_levels = sorted(QUALITY_SETTINGS.keys())
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff'))]
    print(f"開始處理 {len(image_files)} 張圖片，使用 {NUM_WORKERS} 個處理程序...")
    log_path = os.path.join(output_dir, f"quality_metrics_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, "w") as log_file:
        log_file.write("圖片名稱,品質等級,PSNR(dB)\n")
    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_image, file_name, input_dir, output_dir, quality_levels): file_name 
                 for file_name in image_files}
        for future in tqdm(futures, total=len(image_files), desc="處理圖片"):
            result = future.result()
            if result:
                all_results.extend(result)
    with open(log_path, "a") as log_file:
        for entry in all_results:
            log_file.write(f"{entry}\n")
    with open(os.path.join(output_dir, "compression_settings.txt"), "w") as f:
        f.write("JPEG壓縮品質參數設定:\n")
        f.write("品質等級,Quality值,Chroma Subsampling,DCT方法\n")
        for q in sorted(QUALITY_SETTINGS.keys(), reverse=True):
            settings = QUALITY_SETTINGS[q]
            dct_method = "FLOAT" if q >= 50 else "FAST"
            f.write(f"{q},{settings['quality']},{settings['subsampling']},{dct_method}\n")

def process_single_image(file_name, input_dir, output_dir, quality_levels):
    log_entries = []
    input_path = os.path.join(input_dir, file_name)
    try:
        with Image.open(input_path) as original_img:
            if original_img.mode != "RGB":
                original_img = original_img.convert("RGB")
            original_img_array = np.array(original_img)
            base_name, ext = os.path.splitext(file_name)
            psnr_results = {}
            for quality in quality_levels:
                output_file = f"{base_name}_q{quality}.jpg"
                output_path = os.path.join(output_dir, output_file)
                settings = QUALITY_SETTINGS[quality]
                if quality == 100:
                    original_img.save(
                        output_path,
                        "JPEG",
                        quality=settings["quality"],
                        subsampling=settings["subsampling"],
                        dct_method="FLOAT"
                    )
                    psnr_results[quality] = float('inf')
                else:
                    buffer = io.BytesIO()
                    original_img.save(
                        buffer,
                        "JPEG",
                        quality=settings["quality"],
                        subsampling=settings["subsampling"],
                        dct_method="FLOAT" if quality >= 50 else "FAST"
                    )
                    buffer.seek(0)
                    with Image.open(buffer) as compressed_img:
                        compressed_array = np.array(compressed_img)
                        psnr = calculate_psnr(original_img_array, compressed_array)
                        psnr_results[quality] = psnr
                        compressed_img.save(output_path)
            for q in sorted(psnr_results.keys(), reverse=True):
                log_entries.append(f"{file_name},{q},{psnr_results[q]:.2f}")
            return log_entries
        
    except Exception as e:
        print(f"處理圖片 {file_name} 時發生錯誤: {e}")
        return [f"{file_name},錯誤,{str(e)}"]

# 執行
if __name__ == "__main__":
    if os.path.exists(output_directory):
        print(f"警告: 輸出目錄 {output_directory} 已存在，將清空重新生成")
        shutil.rmtree(output_directory)
    start_time = datetime.datetime.now()
    process_images(input_directory, output_directory)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"圖片處理完成！總耗時: {elapsed_time}")