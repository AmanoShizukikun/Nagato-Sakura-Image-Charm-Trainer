import os
import re

def batch_rename_frames(folder, output_format='jpg'):
    """
    重命名指定資料夾中的圖片檔案，調換檔名中的 "frame" 和 "name" 部分
    Args:
        directory: 包含圖片的資料夾路徑
    """
    pattern = re.compile(r'^(?P<name>.+)_frame_(?P<frame>\d{6})\.' + re.escape(output_format) + '$')
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            name = match.group('name')
            frame = match.group('frame')
            new_name = f"frame_{frame}_{name}.{output_format}"
            src = os.path.join(folder, filename)
            dst = os.path.join(folder, new_name)
            print(f"重命名: {filename} → {new_name}")
            os.rename(src, dst)

if __name__ == "__main__":
    folder = r"outputs"  # 請改成你的資料夾路徑
    batch_rename_frames(folder, output_format='jpg')