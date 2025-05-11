import torch
import os

try:
    from src.IQE import ImageQualityEnhancer 
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from src.IQE import ImageQualityEnhancer
    except ImportError as e:
        print(f"錯誤：無法導入 ImageQualityEnhancer。請確保模型定義可訪問。")
        print(f"詳細錯誤: {e}")
        sys.exit(1)

# --- 設定路徑 ---
checkpoint_path = './models/train_v9_Kairitsu_best_psnr.pth.tar' # 或者使用最佳 PSNR 的檢查點
output_pth_path = './models/train_v9_Kairitsu_generator_extracted.pth' # 設定輸出的 .pth 檔案路徑
if os.path.isfile(checkpoint_path):
    print(f"正在加載檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_state_dict' in checkpoint:
        generator_state_dict = checkpoint['generator_state_dict']
        print("已成功提取生成器狀態字典。")
        try:
            model = ImageQualityEnhancer(num_rrdb_blocks=16, features=64)
            model.load_state_dict(generator_state_dict)
            model.eval()
            print("狀態字典已成功加載到模型實例。")
        except Exception as e:
            print(f"警告：加載狀態字典到模型時出錯 (可能是模型結構不匹配): {e}")
        try:
            torch.save(generator_state_dict, output_pth_path)
            print(f"生成器權重已成功保存至: {output_pth_path}")
        except Exception as e:
            print(f"錯誤：保存 .pth 文件時出錯: {e}")
    else:
        print(f"錯誤：在檢查點 '{checkpoint_path}' 中找不到 'generator_state_dict'。")
else:
    print(f"錯誤：找不到檢查點文件: {checkpoint_path}")
