import os
import numpy as np
import cv2 
from concurrent.futures import ProcessPoolExecutor
import shutil
import datetime
import multiprocessing
from tqdm import tqdm
import sys

input_directory = "./data/input_images_01"
output_directory = "./data/quality_dataset_01_P" 

# 設定使用的處理器核心數，預設為可用核心數的75%
NUM_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.75))
PIXELATION_LEVELS = list(range(100, 0, -10))

def calculate_psnr(original_arr, processed_arr):
    original_arr = original_arr.astype(np.float64)
    processed_arr = processed_arr.astype(np.float64)
    mse = np.mean((original_arr - processed_arr) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def create_pixel_art(img, pixel_size):
    """根據給定的 pixel_size 創建像素化圖像"""
    if pixel_size <= 1: 
        return img
    height, width = img.shape[:2]
    small_width = max(1, width // pixel_size)
    small_height = max(1, height // pixel_size)
    small_img = cv2.resize(img, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    pixel_art = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixel_art

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    if not image_files:
        print(f"在 {input_dir} 中找不到任何圖片文件。")
        return
    print(f"開始處理 {len(image_files)} 張圖片，使用 {NUM_WORKERS} 個處理程序...")
    log_path = os.path.join(output_dir, f"pixelation_metrics_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, "w", encoding='utf-8') as log_file:
        log_file.write("圖片名稱,像素區塊大小,PSNR(dB)\n") 
    all_results = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_image, file_name, input_dir, output_dir, PIXELATION_LEVELS): file_name
                 for file_name in image_files}
        for future in tqdm(futures, total=len(image_files), desc="處理圖片"):
            result = future.result()
            if result:
                all_results.extend(result) 
    with open(log_path, "a", encoding='utf-8') as log_file:
        for entry in all_results:
            log_file.write(f"{entry}\n")
    with open(os.path.join(output_dir, "pixelation_settings.txt"), "w", encoding='utf-8') as f:
        f.write("像素化等級說明:\n")
        f.write("原始等級,像素區塊大小 (Pixel Size)\n")
        for q in sorted(PIXELATION_LEVELS, reverse=True):
            pixel_size = (100 - q) // 10 + 1
            f.write(f"q{q},{pixel_size}\n")

def process_single_image(file_name, input_dir, output_dir, quality_levels):
    log_entries = []
    input_path = os.path.join(input_dir, file_name)
    try:
        original_img_array = cv2.imread(input_path)
        if original_img_array is None:
            raise ValueError(f"無法讀取圖像文件: {file_name}")
        base_name, _ = os.path.splitext(file_name)
        psnr_results = {} # 用於存儲每個像素大小的 PSNR
        quality_map = {} # 用於儲存 pixel_size 到 q 的映射
        for q in quality_levels:
            pixel_size = (100 - q) // 10 + 1
            quality_map[pixel_size] = q 
            pixelated_array = create_pixel_art(original_img_array, pixel_size)
            psnr = calculate_psnr(original_img_array, pixelated_array)
            psnr_results[pixel_size] = psnr
            output_file = f"{base_name}_q{q}.png"
            output_path = os.path.join(output_dir, output_file)
            cv2.imwrite(output_path, pixelated_array)
        for pixel_size in sorted(psnr_results.keys()):
            psnr_str = f"{psnr_results[pixel_size]:.2f}" if psnr_results[pixel_size] != float('inf') else "inf"
            log_entries.append(f"{file_name},{pixel_size},{psnr_str}")
        return log_entries
    except Exception as e:
        print(f"處理圖片 {file_name} 時發生錯誤: {e}")
        return [f"{file_name},錯誤,{str(e)}"]

if __name__ == "__main__":
    if os.path.exists(output_directory):
        print(f"警告: 輸出目錄 {output_directory} 已存在，將清空重新生成。")
        try:
            shutil.rmtree(output_directory)
        except OSError as e:
            print(f"錯誤: 無法刪除目錄 {output_directory}: {e}")
            sys.exit(1)
    start_time = datetime.datetime.now()
    process_images(input_directory, output_directory)
    elapsed_time = datetime.datetime.now() - start_time
    print(f"圖片像素化處理完成！總耗時: {elapsed_time}")
    print(f"處理結果已保存至: {output_directory}")
    print(f"詳細 PSNR 指標已記錄在 {os.path.join(output_directory, 'pixelation_metrics_log_*.csv')}")