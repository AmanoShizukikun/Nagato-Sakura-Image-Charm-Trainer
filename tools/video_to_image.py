import cv2
import os
import argparse
import glob
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
import re
import numpy as np
import psutil

def sanitize_filename(filename):
    """處理檔名中的特殊字元，確保檔名安全有效"""
    safe_name = re.sub(r'[\\/*?:"<>|＃＆％＠！？]', "_", filename)
    safe_name = re.sub(r'[\s　]', "_", safe_name)
    if len(safe_name) > 100:
        safe_name = safe_name[:100]
    return safe_name

def get_image_quality_params(output_format):
    """根據輸出格式返回適當的圖片品質參數"""
    if output_format.lower() == 'jpg':
        return [cv2.IMWRITE_JPEG_QUALITY, 100]
    elif output_format.lower() == 'png':
        return [cv2.IMWRITE_PNG_COMPRESSION, 1]
    return []

def process_frame_range(video_path, start_frame, end_frame, output_dir, frame_interval=1, output_format='jpg'):
    """處理影片中指定範圍的幀"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_video_name = sanitize_filename(video_name)
    img_quality_params = get_image_quality_params(output_format)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"無法開啟影片檔案: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    saved_count = 0
    current_frame = start_frame
    error_count = 0
    try:
        try:
            process_id = multiprocessing.current_process()._identity[0]
        except IndexError:
            process_id = np.random.randint(0, 100)
        pbar = tqdm(total=end_frame-start_frame,
                    desc=f"處理 {safe_video_name} 幀 {start_frame}-{end_frame}",
                    position=process_id % 5, 
                    leave=False)
        first_extract_frame = frame_interval 
        if start_frame > 0:
            remainder = start_frame % frame_interval
            if remainder == 0:
                first_extract_frame = start_frame 
            else:
                first_extract_frame = start_frame + (frame_interval - remainder)
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame >= first_extract_frame and (current_frame - first_extract_frame) % frame_interval == 0:
                output_filename = f"{safe_video_name}_frame_{current_frame:06d}.{output_format}"
                output_path = os.path.join(output_dir, output_filename)
                try:
                    cv2.imwrite(output_path, frame, img_quality_params)
                    saved_count += 1
                except Exception as e:
                    error_count += 1
                    if error_count < 5: 
                        print(f"警告: 無法保存幀 {current_frame} 到 {output_path}: {e}")
            current_frame += 1
            pbar.update(1)
        pbar.close()
    finally:
        cap.release()
    return saved_count, error_count

def extract_frames_multicore(video_path, output_dir, frame_interval=1, output_format='jpg'):
    """使用多核心處理單一影片檔案"""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_video_name = sanitize_filename(video_name)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        print(f"錯誤: 無法開啟影片檔案: {video_path}")
        return 0, 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"警告: 無法讀取影片 {safe_video_name} 的總幀數或影片為空。")
        return 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"影片資訊: {video_name}")
    print(f"  • 解析度: {width}x{height} 像素")
    print(f"  • 幀數: {total_frames} 幀")
    print(f"  • 幀率: {fps:.2f} FPS")
    print(f"  • 時長: {duration/60:.1f} 分鐘 ({duration:.2f} 秒)")
    print(f"  • 檔案大小: {file_size_mb:.1f} MB")
    cap.release()
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024) 
    estimated_memory_per_worker = (width * height * 3) / (1024 * 1024 * 10) 
    cpu_count = multiprocessing.cpu_count()
    memory_based_workers = max(1, int(available_memory / max(0.1, estimated_memory_per_worker)))
    num_workers = min(cpu_count, memory_based_workers, max(1, int(cpu_count * 0.75)))
    print(f"使用 {num_workers} 個處理程序進行多核心處理 (系統有 {cpu_count} 個核心, 可用記憶體 {available_memory:.1f}GB)")
    frames_per_worker = max(1, total_frames // num_workers)
    frame_ranges = []
    for i in range(num_workers):
        start_frame = i * frames_per_worker
        if start_frame >= total_frames:
            break
        end_frame = min((i + 1) * frames_per_worker, total_frames)
        frame_ranges.append((start_frame, end_frame))
    start_time = time.time()
    total_saved_frames = 0
    total_error_frames = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for start_frame, end_frame in frame_ranges:
            if start_frame < end_frame:
                future = executor.submit(
                    process_frame_range,
                    video_path,
                    start_frame,
                    end_frame,
                    output_dir,
                    frame_interval,
                    output_format
                )
                futures.append(future)
        for future in futures:
            try:
                saved_count, error_count = future.result()
                total_saved_frames += saved_count
                total_error_frames += error_count
            except Exception as e:
                print(f"錯誤: 子進程執行時發生錯誤: {e}")
    elapsed_time = time.time() - start_time
    print(f"影片處理完成！共提取了 {total_saved_frames} 幀，耗時: {elapsed_time:.2f} 秒")
    print(f"平均處理速度: {total_saved_frames/elapsed_time:.1f} 幀/秒")
    first_frame = frame_interval
    expected_frames = (total_frames - first_frame) // frame_interval + 1
    success_rate = total_saved_frames / max(1, expected_frames) * 100
    return total_saved_frames, elapsed_time, success_rate

def process_videos_in_directory(input_dir, output_dir, frame_interval=1, output_format='jpg'):
    """處理目錄中的所有影片檔案"""
    os.makedirs(output_dir, exist_ok=True)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    video_files = []
    print(f"正在搜尋 {input_dir} 中的影片檔案...")
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, extension)))
    if not video_files:
        print(f"在 {input_dir} 中找不到任何支援的影片檔案")
        return
    video_files.sort(key=lambda x: os.path.getsize(x))
    print(f"找到 {len(video_files)} 個影片檔案")
    overall_start_time = time.time()
    total_processed_frames = 0
    processed_videos = 0
    failed_videos = 0
    video_results = []
    for video_idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print("\n" + "=" * 50)
        print(f"【{video_idx}/{len(video_files)}】開始處理影片: {video_name}")
        try:
            frames, process_time, success_rate = extract_frames_multicore(
                video_path,
                output_dir,
                frame_interval,
                output_format
            )
            total_processed_frames += frames
            if frames > 0:
                processed_videos += 1
                video_results.append({
                    'name': video_name,
                    'frames': frames,
                    'time': process_time,
                    'success_rate': success_rate
                })
            else:
                failed_videos += 1        
        except Exception as e:
            print(f"處理影片時發生錯誤: {e}")
            failed_videos += 1
        print(f"目前進度: {video_idx}/{len(video_files)} ({video_idx/len(video_files)*100:.1f}%)")
    overall_elapsed_time = time.time() - overall_start_time
    print("\n" + "=" * 50)
    print(f"📊 處理結果報告")
    print(f"=" * 50)
    print(f"✅ 成功處理 {processed_videos} 個影片")
    print(f"❌ 失敗 {failed_videos} 個影片")
    print(f"🖼️ 共提取了 {total_processed_frames} 幀圖像")
    print(f"⏱️ 總耗時: {overall_elapsed_time:.2f} 秒 ({overall_elapsed_time/60:.2f} 分鐘)")
    if video_results:
        print("\n🔍 各影片處理詳情:")
        for idx, result in enumerate(video_results, 1):
            print(f"{idx}. {result['name']}")
            print(f"   提取幀數: {result['frames']} 幀")
            print(f"   處理時間: {result['time']:.2f} 秒")
            print(f"   成功率: {result['success_rate']:.1f}%")
            print(f"   平均速度: {result['frames']/max(0.1, result['time']):.1f} 幀/秒")
    
    print(f"\n📂 輸出路徑: {os.path.abspath(output_dir)}")
    print("=" * 50)
    print("執行完畢！")

def main():
    parser = argparse.ArgumentParser(description="從多個影片中按間隔提取幀")
    parser.add_argument("--input-dir", default="output", help="輸入影片的目錄 (預設: output)")
    parser.add_argument("--output-dir", "-o", default="frames", help="輸出圖片的目錄 (預設: frames)")
    parser.add_argument("--interval", "-i", type=int, default=150, help="抽取幀的間隔 (預設: 150)")
    parser.add_argument("--format", "-f", default="jpg", choices=["jpg", "png", "bmp"], help="輸出圖片格式 (預設: jpg)")
    args = parser.parse_args()
    print("\n🎬 影片幀擷取工具 - 多核心版本")
    print("=" * 50)
    print(f"📂 輸入目錄: {os.path.abspath(args.input_dir)}")
    print(f"📂 輸出目錄: {os.path.abspath(args.output_dir)}")
    print(f"⏱️ 幀提取間隔: 每 {args.interval} 幀擷取一次 (從第 {args.interval} 幀開始)")
    print(f"🖼️ 輸出格式: {args.format}")
    print("=" * 50)
    try:
        if not os.path.exists(args.input_dir):
            print(f"❌ 錯誤: 輸入目錄 '{args.input_dir}' 不存在！")
            return 1 
        process_videos_in_directory(
            args.input_dir,
            args.output_dir,
            args.interval,
            args.format
        )
    except KeyboardInterrupt:
        print("\n⚠️ 使用者中斷處理")
        return 130
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return 1
    return 0

if __name__ == "__main__":
    multiprocessing.freeze_support()
    exit_code = main()
    exit(exit_code)