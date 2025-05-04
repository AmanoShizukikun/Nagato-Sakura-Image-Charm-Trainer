import os
import subprocess
from pathlib import Path
import time
import logging
import sys
import shutil 
import traceback 
import re


log_file = "video_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

QUALITY_MAPPING = {
    "q100": 0,   
    "q90": 24,
    "q80": 27,
    "q70": 30,
    "q60": 32,
    "q50": 34,
    "q40": 36,
    "q30": 38,
    "q20": 40,
    "q10": 42
}

# 支援的影片檔案副檔名
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.ts', '.mpg', '.mpeg']

# FFmpeg 設定
FFMPEG_PATH = "ffmpeg" 
NVENC_PRESET = "p7" 
CPU_PRESET = "medium" 
MAX_RETRIES = 3 
RETRY_DELAY = 5 

# --- 輔助函數 ---
def check_ffmpeg():
    """檢查ffmpeg是否已安裝並可執行"""
    ffmpeg_executable = shutil.which(FFMPEG_PATH)
    if not ffmpeg_executable:
        logger.error(f"找不到 FFmpeg 執行檔: '{FFMPEG_PATH}'")
        logger.error("請確保 FFmpeg 已安裝並已添加到系統 PATH，或者在腳本中指定正確的 FFMPEG_PATH。")
        return None 

    logger.info(f"找到 FFmpeg 執行檔: {ffmpeg_executable}")
    try:
        result = subprocess.run(
            [ffmpeg_executable, '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', 
            errors='replace'
        )
        if result.returncode == 0:
            version_info = result.stdout.splitlines()[0] if result.stdout else "無法獲取版本信息"
            logger.info(f"FFmpeg 版本: {version_info}")
            return ffmpeg_executable 
        else:
            logger.error(f"執行 'ffmpeg -version' 失敗。錯誤碼: {result.returncode}")
            logger.error(f"FFmpeg stderr: {result.stderr}")
            return None
    except FileNotFoundError:
        logger.error(f"找不到 FFmpeg 執行檔: '{FFMPEG_PATH}'")
        return None
    except Exception as e:
        logger.error(f"檢查 FFmpeg 時發生未預期的錯誤: {e}")
        return None

def check_nvidia_gpu_and_nvenc(ffmpeg_exec):
    """檢查NVIDIA GPU和FFmpeg中的NVENC編碼器"""
    try:
        nvidia_smi_path = shutil.which('nvidia-smi')
        if not nvidia_smi_path:
            logger.warning("找不到 'nvidia-smi' 命令。無法確認 NVIDIA GPU 是否存在。")
        else:
            logger.info(f"找到 nvidia-smi: {nvidia_smi_path}")
            result_smi = subprocess.run([nvidia_smi_path], capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result_smi.returncode != 0:
                logger.warning(f"'nvidia-smi' 執行失敗 (錯誤碼: {result_smi.returncode})。可能沒有 NVIDIA GPU 或驅動程式有問題。")
                logger.debug(f"nvidia-smi stderr: {result_smi.stderr}")
            else:
                logger.info("nvidia-smi 執行成功，檢測到 NVIDIA GPU。")
                logger.debug(f"nvidia-smi output:\n{result_smi.stdout[:500]}...")
        logger.info("正在檢查 FFmpeg 中的 h264_nvenc 編碼器...")
        result_enc = subprocess.run(
            [ffmpeg_exec, '-encoders'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result_enc.returncode != 0:
            logger.error(f"執行 'ffmpeg -encoders' 失敗 (錯誤碼: {result_enc.returncode})。")
            logger.error(f"FFmpeg stderr: {result_enc.stderr}")
            return False
        encoders_output = result_enc.stdout
        if 'h264_nvenc' in encoders_output:
            logger.info("在 FFmpeg 中找到 'h264_nvenc' 編碼器。")
            return True
        else:
            logger.warning("在 FFmpeg 的可用編碼器列表中找不到 'h264_nvenc'。")
            logger.debug(f"FFmpeg -encoders output:\n{encoders_output}")
            return False
    except FileNotFoundError as e:
        logger.error(f"執行檢查命令時出錯: {e}。請確保相關程式 (ffmpeg, nvidia-smi) 在 PATH 中。")
        return False
    except Exception as e:
        logger.error(f"檢查 NVIDIA GPU 或 NVENC 時發生未預期的錯誤: {e}")
        return False

def check_video_properties(ffmpeg_exec, video_path):
    """檢查視頻是否為 10-bit 格式"""
    try:
        result = subprocess.run(
            [ffmpeg_exec, '-i', str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        output = result.stderr
        is_10bit = False
        bit_depth = 8 
        if re.search(r'(yuv420p10le|10\s?bit|Main 10)', output):
            is_10bit = True
            bit_depth = 10
        is_hevc = 'hevc' in output
        return {
            'is_10bit': is_10bit,
            'bit_depth': bit_depth,
            'is_hevc': is_hevc,
        }
    except Exception as e:
        logger.error(f"檢查視頻屬性時出錯: {e}")
        return {
            'is_10bit': False,
            'bit_depth': 8,
            'is_hevc': False,
        }

def process_video(ffmpeg_exec, video_path, quality_name, qp_value, output_folder, attempt=0):
    """
    處理單一影片檔案的單一畫質。
    返回: 'success', 'skipped', 'failed'
    """
    input_file = video_path
    filename = video_path.stem
    output_file = output_folder / f"{filename}_{quality_name}.mp4"
    error_log_folder = Path("error_logs")
    error_log_folder.mkdir(exist_ok=True) 
    error_log_path = error_log_folder / f"error_{filename}_{quality_name}_attempt{attempt}.log"
    if output_file.exists() and output_file.stat().st_size > 0:
        logger.info(f"檔案已存在且非空，跳過處理: {output_file.name}")
        return "skipped"
    elif output_file.exists():
        logger.warning(f"檔案已存在但大小為 0，將嘗試覆蓋: {output_file.name}")
        try:
            output_file.unlink()
        except OSError as e:
            logger.error(f"刪除現有空檔案失敗: {output_file.name} - {e}")
            return "failed"
    video_props = check_video_properties(ffmpeg_exec, input_file)
    is_10bit = video_props['is_10bit']
    
    # --- 構建 FFmpeg 命令 ---
    common_options = [
        '-i', str(input_file),
        '-c:a', 'copy',
        '-map_metadata', '-1',
        '-y',
    ]
    try_nvenc = True
    if is_10bit:
        logger.info(f"檢測到 10-bit 視頻: {input_file.name}")
        common_options.extend([
            '-pix_fmt', 'yuv420p', 
        ])
    if try_nvenc:
        encoder_options = [
            '-c:v', 'h264_nvenc',
            '-preset', NVENC_PRESET,
            '-rc', 'constqp',       
            '-qp', str(qp_value),   
            '-profile:v', 'high',
        ]
        processing_mode = "NVENC"
    else:
        encoder_options = [
            '-c:v', 'libx264',
            '-preset', CPU_PRESET,
            '-crf', str(qp_value), 
            '-profile:v', 'high',
        ]
        processing_mode = "CPU (libx264)"
    cmd = [ffmpeg_exec] + common_options + encoder_options + [str(output_file)]
    logger.info(f"[{processing_mode}] 開始處理: {output_file.name} (QP/CRF: {qp_value}, 嘗試: {attempt+1}/{MAX_RETRIES})")
    logger.debug(f"執行命令: {' '.join(cmd)}")

    # --- 執行 FFmpeg ---
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace', 
            creationflags=subprocess.CREATE_NO_WINDOW 
        )
        stdout, stderr = process.communicate() 

        # --- 檢查結果 ---
        if process.returncode == 0:
            if output_file.exists() and output_file.stat().st_size > 0:
                logger.info(f"[{processing_mode}] 成功處理: {output_file.name}")
                return "success"
            else:
                logger.error(f"[{processing_mode}] FFmpeg 返回成功碼 0，但輸出檔案不存在或為空: {output_file.name}")
                logger.error(f"請檢查 FFmpeg 的輸出以獲取詳細信息。")
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(f"FFmpeg Command:\n{' '.join(cmd)}\n\n")
                    f.write(f"Return Code: {process.returncode}\n\n")
                    f.write(f"STDOUT:\n{stdout}\n\n")
                    f.write(f"STDERR:\n{stderr}\n")
                logger.error(f"FFmpeg 輸出已儲存至: {error_log_path}")
                return "failed"
        else:
            if try_nvenc and processing_mode == "NVENC" and ("No capable devices found" in stderr or "10 bit encode not supported" in stderr):
                logger.warning(f"NVENC 編碼失敗，嘗試使用 CPU 編碼（libx264）...")
                cpu_encoder_options = [
                    '-c:v', 'libx264',
                    '-preset', CPU_PRESET,
                    '-crf', str(qp_value), 
                    '-profile:v', 'high',
                ]
                cpu_cmd = [ffmpeg_exec] + common_options + cpu_encoder_options + [str(output_file)]
                logger.info(f"[CPU (libx264)] 重試處理: {output_file.name} (CRF: {qp_value})")
                logger.debug(f"執行命令: {' '.join(cpu_cmd)}")
                cpu_process = subprocess.Popen(
                    cpu_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                cpu_stdout, cpu_stderr = cpu_process.communicate()
                if cpu_process.returncode == 0:
                    if output_file.exists() and output_file.stat().st_size > 0:
                        logger.info(f"[CPU (libx264)] 成功處理: {output_file.name}")
                        return "success"
                    else:
                        logger.error(f"[CPU (libx264)] FFmpeg 返回成功碼，但輸出檔案不存在或為空: {output_file.name}")
                else:
                    logger.error(f"[CPU (libx264)] 處理失敗: {output_file.name}")
                    logger.error(f"FFmpeg 錯誤碼: {cpu_process.returncode}")
                    with open(error_log_path, "w", encoding="utf-8") as f:
                        f.write(f"Original FFmpeg Command (NVENC):\n{' '.join(cmd)}\n\n")
                        f.write(f"NVENC Error:\n{stderr}\n\n")
                        f.write(f"CPU FFmpeg Command:\n{' '.join(cpu_cmd)}\n\n")
                        f.write(f"CPU Return Code: {cpu_process.returncode}\n\n")
                        f.write(f"CPU STDOUT:\n{cpu_stdout}\n\n")
                        f.write(f"CPU STDERR:\n{cpu_stderr}\n")
                    return "failed"
            else:
                logger.error(f"[{processing_mode}] 處理失敗: {output_file.name}")
                logger.error(f"FFmpeg 錯誤碼: {process.returncode}")
                with open(error_log_path, "w", encoding="utf-8") as f:
                    f.write(f"FFmpeg Command:\n{' '.join(cmd)}\n\n")
                    f.write(f"Return Code: {process.returncode}\n\n")
                    f.write(f"STDOUT:\n{stdout}\n\n")
                    f.write(f"STDERR:\n{stderr}\n")
                logger.error(f"完整錯誤已儲存至: {error_log_path}")
                return "failed"
    except FileNotFoundError:
        logger.error(f"執行 FFmpeg 失敗: 找不到執行檔 '{ffmpeg_exec}'。")
        return "failed" 
    except Exception as e:
        logger.error(f"處理 {input_file.name} 時發生未預期的 Python 錯誤: {e}")
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Python Exception during FFmpeg execution for command:\n{' '.join(cmd)}\n\n")
            f.write(traceback.format_exc())
        logger.error(f"Python 錯誤追蹤已儲存至: {error_log_path}")
        return "failed"

def process_single_video(ffmpeg_exec, video_path, output_folder):
    """處理單一影片的所有畫質版本，包含重試邏輯"""
    logger.info(f"\n===== 開始處理影片: {video_path.name} =====")
    success_count = 0
    skipped_count = 0
    failed_count = 0
    total_qualities = len(QUALITY_MAPPING)
    for i, (quality_name, qp_value) in enumerate(QUALITY_MAPPING.items()):
        logger.info(f"--- 處理畫質 {i+1}/{total_qualities}: {quality_name} (QP/CRF: {qp_value}) ---")
        attempt = 0
        success = False
        while attempt < MAX_RETRIES and not success:
            status = process_video(ffmpeg_exec, video_path, quality_name, qp_value, output_folder, attempt)

            if status == "success":
                success_count += 1
                success = True
                break
            elif status == "skipped":
                skipped_count += 1
                success = True
                break 
            else: 
                logger.error(f"處理 {quality_name} 失敗 (嘗試 {attempt+1}/{MAX_RETRIES})。")
                attempt += 1
                if attempt < MAX_RETRIES:
                    logger.info(f"將在 {RETRY_DELAY} 秒後重試...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"處理 {quality_name} 達到最大重試次數，放棄。")
                    failed_count += 1
                    break 
    logger.info(f"===== 影片 {video_path.name} 處理完成 =====")
    logger.info(f"結果: {success_count} 個成功, {skipped_count} 個跳過, {failed_count} 個失敗 (共 {total_qualities} 個畫質)")
    return success_count, skipped_count, failed_count

# --- 主函數 ---
def main():
    start_time = time.time()
    logger.info("=========================================")
    logger.info("===   批量影片轉換器 v1.3   ===")
    logger.info("=========================================")

    # --- 檢查 FFmpeg ---
    ffmpeg_executable = check_ffmpeg()
    if not ffmpeg_executable:
        sys.exit(1) 

    # --- 檢查 GPU 和 NVENC ---
    can_use_nvenc = check_nvidia_gpu_and_nvenc(ffmpeg_executable)
    if not can_use_nvenc:
        logger.warning("未檢測到可用的 NVENC 編碼器。將使用 CPU 編碼（libx264）作為備用方案。")
        logger.warning("這可能會導致處理速度變慢。")
    else:        
        logger.info("檢測到 NVIDIA GPU 和可用的 NVENC 編碼器，優先使用 NVENC 進行編碼。")
    
    # --- 設定資料夾路徑 ---
    input_folder = Path("input")
    output_folder = Path("output")

    # --- 檢查輸入資料夾 ---
    if not input_folder.exists() or not input_folder.is_dir():
        logger.error(f"輸入資料夾 '{input_folder}' 不存在或不是一個有效的資料夾。")
        try:
            input_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"已創建輸入資料夾 '{input_folder}'。")
            logger.info(f"請將要處理的影片放入 '{input_folder}' 資料夾，然後重新執行程式。")
        except OSError as e:
            logger.error(f"創建輸入資料夾 '{input_folder}' 失敗: {e}")
        sys.exit(1)
    elif not any(input_folder.iterdir()):
        logger.warning(f"輸入資料夾 '{input_folder}' 為空。")
        logger.info(f"請將要處理的影片放入 '{input_folder}' 資料夾，然後重新執行程式。")
        sys.exit(0)

    # --- 確保輸出資料夾存在 ---
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"輸出資料夾: '{output_folder}'")
    except OSError as e:
        logger.error(f"創建輸出資料夾 '{output_folder}' 失敗: {e}")
        sys.exit(1)

    # --- 獲取所有影片檔案 ---
    logger.info(f"正在從 '{input_folder}' 搜尋影片檔案...")
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(list(input_folder.glob(f"*{ext}")))
        video_files.extend(list(input_folder.glob(f"*{ext.upper()}")))
    video_files = sorted(list(set(video_files)))
    if not video_files:
        logger.warning(f"在 '{input_folder}' 資料夾中找不到支援的影片檔案。")
        logger.warning(f"支援的副檔名: {', '.join(VIDEO_EXTENSIONS)}")
        sys.exit(0)

    logger.info(f"找到 {len(video_files)} 個影片檔案準備處理。")

    # --- 處理所有影片 ---
    total_processed_files = 0
    total_success_versions = 0
    total_skipped_versions = 0
    total_failed_versions = 0
    total_expected_versions = len(video_files) * len(QUALITY_MAPPING)
    for index, video_file in enumerate(video_files, 1):
        logger.info(f"\n---<<< 開始處理第 {index}/{len(video_files)} 個影片 >>>---")
        success_count, skipped_count, failed_count = process_single_video(
            ffmpeg_executable,
            video_file,
            output_folder
        )
        total_processed_files += 1
        total_success_versions += success_count
        total_skipped_versions += skipped_count
        total_failed_versions += failed_count
        progress = (index / len(video_files)) * 100
        logger.info(f"---<<< 影片 {index}/{len(video_files)} ({video_file.name}) 處理完畢 >>>---")
        logger.info(f"---<<< 總體進度: {progress:.1f}% >>>---")

    # --- 顯示最終處理結果 ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("\n=========================================")
    logger.info("===      影片批量處理完成         ===")
    logger.info("=========================================")
    logger.info(f"總共掃描了 {len(video_files)} 個影片檔案。")
    logger.info(f"預期生成 {total_expected_versions} 個轉檔版本。")
    logger.info(f"  - 成功生成: {total_success_versions}")
    logger.info(f"  - 因已存在而跳過: {total_skipped_versions}")
    logger.info(f"  - 處理失敗: {total_failed_versions}")
    logger.info(f"處理總時間: {elapsed_time:.2f} 秒 ({time.strftime('%H:%M:%S', time.gmtime(elapsed_time))})")
    logger.info(f"詳細日誌已儲存至: {log_file}")
    logger.info(f"錯誤日誌（如果有的話）儲存在: error_logs 資料夾")
    logger.info("=========================================")

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        try:
           application_path = os.path.dirname(os.path.realpath(__file__))
        except NameError:
           application_path = os.getcwd()
    os.chdir(application_path)
    main()
    if os.name == 'nt':
        input("\n處理完成，按 Enter 鍵退出...")