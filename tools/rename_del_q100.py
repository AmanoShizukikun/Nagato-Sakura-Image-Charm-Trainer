import os
import sys

def rename_files(directory):
    """
    重命名指定資料夾中的圖片檔案，移除檔名中的 "q100" 部分
    Args:
        directory: 包含圖片的資料夾路徑
    """
    if not os.path.isdir(directory):
        print(f"錯誤: 資料夾 '{directory}' 不存在")
        return
    renamed_count = 0
    for filename in os.listdir(directory):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            base_name, ext = os.path.splitext(filename)
            if "q100" in base_name:
                new_base_name = base_name.replace("_q100", "")
                new_filename = new_base_name + ext
                new_path = os.path.join(directory, new_filename)
                if os.path.exists(new_path):
                    print(f"警告: 無法重命名 {filename} - 目標檔案 {new_filename} 已存在")
                    continue
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                    print(f"已重命名: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"錯誤: 無法重命名 {filename}: {e}")
    print(f"\n完成! 已重命名 {renamed_count} 個檔案")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_outputs_dir = os.path.join(script_dir, "outputs") # 請改成你的資料夾名稱
    directory = sys.argv[1] if len(sys.argv) > 1 else video_outputs_dir
    print(f"正在處理資料夾: {directory}")
    rename_files(directory)