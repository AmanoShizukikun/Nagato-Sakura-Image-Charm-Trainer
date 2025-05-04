import cv2
import os
import numpy as np
import pandas as pd
import re
from glob import glob
from pathlib import Path

def calculate_psnr(img1, img2):
    """計算兩張圖片的PSNR值"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def extract_frame(video_path, frame_num):
    """從影片中提取特定幀"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num >= total_frames:
        print(f"警告: {video_path} 只有 {total_frames} 幀，無法擷取第 {frame_num} 幀")
        frame_num = total_frames - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"無法讀取 {video_path} 的第 {frame_num} 幀")
        return None
    return frame

def get_quality_from_filename(filename):
    """從檔案名稱中提取品質數值 (q100, q99 等)"""
    match = re.search(r'_q(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main(input_folder, frame_num, output_csv):
    if not os.path.exists(input_folder):
        print(f"資料夾 {input_folder} 不存在")
        return
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob(os.path.join(input_folder, ext)))
    if not video_files:
        print(f"在 {input_folder} 中找不到影片檔案")
        return
    video_groups = {}
    for video_path in video_files:
        filename = os.path.basename(video_path)
        base_name = re.sub(r'_q\d+', '', filename) 
        if base_name not in video_groups:
            video_groups[base_name] = []
        quality = get_quality_from_filename(filename)
        if quality is not None:
            video_groups[base_name].append((quality, video_path))
    results = []
    for base_name, videos in video_groups.items():
        if not videos:
            continue
        videos.sort(reverse=True)
        reference_video = None
        for quality, video_path in videos:
            if quality == 100:
                reference_video = video_path
                break
        if not reference_video:
            print(f"找不到 {base_name} 的 q100 參考影片，跳過此組")
            continue
        reference_frame = extract_frame(reference_video, frame_num)
        if reference_frame is None:
            continue
        for quality, video_path in videos:
            if quality == 100:
                continue
            current_frame = extract_frame(video_path, frame_num)
            if current_frame is None:
                continue
            if current_frame.shape != reference_frame.shape:
                print(f"警告: {os.path.basename(video_path)} 的幀大小與參考影片不同，正在調整大小...")
                current_frame = cv2.resize(current_frame, (reference_frame.shape[1], reference_frame.shape[0]))
            psnr_value = calculate_psnr(reference_frame, current_frame)
            results.append({
                '影片名稱': base_name,
                '品質設定': f'q{quality}',
                'PSNR': psnr_value
            }) 
            print(f"已處理: {base_name} - q{quality} - PSNR: {psnr_value:.2f}dB")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"結果已儲存至 {output_csv}")
    else:
        print("沒有結果可供輸出")

if __name__ == "__main__":
    input_folder = os.path.join(os.path.dirname(__file__), "output")
    frame_num = 50  # 要擷取的幀號 (從0開始計數)
    output_csv = os.path.join(os.path.dirname(__file__), "psnr_results.csv")
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"已建立 input 資料夾: {input_folder}")
    main(input_folder, frame_num, output_csv)