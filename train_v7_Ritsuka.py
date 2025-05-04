import os
import sys
import time
import math
import random
import json
import argparse
import gc 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from sklearn.model_selection import train_test_split
from PIL import Image

try:
    from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator
    except ImportError as e:
        print(f"錯誤：無法導入 src.IQE。請確保 src 目錄與此腳本在同一父目錄下。詳細錯誤: {e}")
        sys.exit(1)

# 設置CUDNN優化
torch.backends.cudnn.benchmark = True
# 為了可重複性，可以設置 deterministic = True，但可能會稍微降低速度
# torch.backends.cudnn.deterministic = True

# --- 損失函數 ---
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def _gaussian(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        pad = self.window_size // 2 
        if channel != self.channel or self.window.device != img1.device or self.window.dtype != img1.dtype:
            window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        else:
            window = self.window
        img1_padded = F.pad(img1, (pad, pad, pad, pad), mode='reflect')
        img2_padded = F.pad(img2, (pad, pad, pad, pad), mode='reflect')
        mu1 = F.conv2d(img1_padded, window, padding='valid', groups=channel)
        mu2 = F.conv2d(img2_padded, window, padding='valid', groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1_padded * img1_padded, window, padding='valid', groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2_padded * img2_padded, window, padding='valid', groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1_padded * img2_padded, window, padding='valid', groups=channel) - mu1_mu2
        sigma1_sq = F.relu(sigma1_sq) + 1e-8
        sigma2_sq = F.relu(sigma2_sq) + 1e-8
        C1 = (0.01 * 1)**2
        C2 = (0.03 * 1)**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        loss = 1.0 - ssim_map
        if self.size_average:
            return loss.mean()
        else:
            return loss.mean(dim=[1, 2, 3])

class PixelRestoreLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True, channel=3)
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        # 拉普拉斯算子用於高頻細節 (可選)
        # laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        # laplacian = F.pad(laplacian, (1, 1, 1, 1), "constant", 0) # 5x5
        # self.register_buffer('laplacian_kernel', laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))

    def extract_sobel_edges(self, x):
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(x_padded, self.sobel_x, padding='valid', groups=3)
        grad_y = F.conv2d(x_padded, self.sobel_y, padding='valid', groups=3)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude

    # def extract_high_freq(self, x):
    #     # 使用 reflect padding
    #     x_padded = F.pad(x, (2, 2, 2, 2), mode='reflect')
    #     return F.conv2d(x_padded, self.laplacian_kernel, padding='valid', groups=3)

    def forward(self, x, target):
        l1_pixel_loss = self.l1_loss(x, target)
        mse_pixel_loss = self.mse_loss(x, target) 
        structure_loss = self.ssim_loss(x, target) 
        x_edges = self.extract_sobel_edges(x)
        target_edges = self.extract_sobel_edges(target)
        edge_loss = self.l1_loss(x_edges, target_edges)

        # 高頻細節損失 ( 可選)
        # x_high_freq = self.extract_high_freq(x)
        # target_high_freq = self.extract_high_freq(target)
        # high_freq_loss = self.l1_loss(x_high_freq, target_high_freq)

        color_loss = self.l1_loss(x.mean(dim=[2, 3]), target.mean(dim=[2, 3]))

        # --- 組合損失 ---
        total_loss = (
            l1_pixel_loss * 1.5 +      
            mse_pixel_loss * 0.2 +      
            structure_loss * 2.5 +     
            edge_loss * 3.0 +          
            # high_freq_loss * 0.5 +    
            color_loss * 0.1           
        )
        loss_dict = {
            'total': total_loss,
            'l1_pixel': l1_pixel_loss,
            'mse_pixel': mse_pixel_loss,
            'ssim': structure_loss,
            'edge': edge_loss,
            # 'high_freq': high_freq_loss,
            'color': color_loss
        }
        return loss_dict


# --- 資料集類別 ---
class QualityDataset(Dataset):
    def __init__(self, image_dir, transform=None, crop_size=256, augment=True, cache_images=False, 
                 min_quality=10, max_quality=90):
        self.image_dir = image_dir
        self.transform = transform
        self.crop_size = crop_size
        self.augment = augment
        self.cache_images = cache_images
        self.image_cache = {}
        self.image_groups = {}
        self.epoch = 0 
        self.min_quality = min_quality
        self.max_quality = max_quality
        print(f"初始化資料集，目錄: {image_dir}")
        print(f"訓練品質範圍: q{min_quality} - q{max_quality}")
        print(f"圖像裁剪大小: {crop_size}x{crop_size}")
        print(f"數據增強: {'啟用' if augment else '禁用'}")
        print(f"圖像快取: {'啟用' if cache_images else '禁用'}")
        all_image_paths = []
        print("掃描圖像文件中...")
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                    all_image_paths.append(os.path.join(root, f))
        print(f"找到 {len(all_image_paths)} 個圖像文件。")
        for path in all_image_paths:
            try:
                base_name_part = os.path.splitext(os.path.basename(path))[0]
                parts = base_name_part.rsplit('_', 1)
                if len(parts) == 2 and parts[1].startswith('q') and parts[1][1:].isdigit():
                    base_name = parts[0]
                    quality_str = parts[1]
                    if base_name not in self.image_groups:
                        self.image_groups[base_name] = {}
                    self.image_groups[base_name][quality_str] = path
                # else:
                #     print(f"忽略無法解析的檔名: {path}")
            except Exception as e:
                print(f"處理路徑時出錯: {path}, {e}")
                continue
        self.valid_groups = []
        for base_name, qualities in self.image_groups.items():
            if 'q100' in qualities:
                has_valid_low_quality = False
                for q_str in qualities.keys():
                    if q_str != 'q100':
                        try:
                            q_val = int(q_str[1:])
                            if self.min_quality <= q_val <= self.max_quality:
                                has_valid_low_quality = True
                                break
                        except ValueError:
                            continue 
                if has_valid_low_quality:
                    self.valid_groups.append(base_name)
        print(f"找到 {len(self.valid_groups)} 組有效的圖像 (包含 q100 和 q{min_quality}-q{max_quality} 範圍內的圖像)")
        if not self.valid_groups:
             raise ValueError(f"在目錄 {image_dir} 中找不到有效的圖像組 (需要 q100 和 q{min_quality}-q{max_quality} 範圍內的圖像)")
        if self.cache_images and len(self.valid_groups) > 0:
            print(f"預載入 {len(self.valid_groups)} 組圖像到記憶體...")
            loaded_count = 0
            skipped_groups = 0
            for i, base_name in enumerate(self.valid_groups):
                qualities = self.image_groups.get(base_name)
                if not qualities: continue
                try:
                    high_quality_path = qualities.get('q100')
                    if not high_quality_path: continue
                    if high_quality_path not in self.image_cache:
                         img = Image.open(high_quality_path).convert("RGB")
                         self.image_cache[high_quality_path] = img
                    loaded_low = False
                    for q_name, low_quality_path in qualities.items():
                        if q_name != 'q100':
                             try:
                                 q_val = int(q_name[1:])
                                 if self.min_quality <= q_val <= self.max_quality:
                                     if low_quality_path not in self.image_cache:
                                         img_low = Image.open(low_quality_path).convert("RGB")
                                         self.image_cache[low_quality_path] = img_low
                                     loaded_low = True
                             except ValueError:
                                 continue
                    if loaded_low:
                        loaded_count +=1
                    else:
                        if high_quality_path in self.image_cache:
                            del self.image_cache[high_quality_path]
                        skipped_groups += 1
                        continue 
                    if (i + 1) % 200 == 0:
                        print(f"  已處理 {i+1}/{len(self.valid_groups)} 組，實際載入 {loaded_count} 組有效圖像...")
                except FileNotFoundError:
                    print(f"警告：預載入時找不到文件，組 '{base_name}' 將在需要時載入。")
                    if high_quality_path and high_quality_path in self.image_cache: del self.image_cache[high_quality_path]
                    for q_name, low_quality_path in qualities.items():
                         if low_quality_path in self.image_cache: del self.image_cache[low_quality_path]
                    skipped_groups += 1
                    continue
                except Exception as e:
                    print(f"預載入圖像時出錯: {base_name}, {e}. 組將在需要時載入。")
                    if high_quality_path and high_quality_path in self.image_cache: del self.image_cache[high_quality_path]
                    for q_name, low_quality_path in qualities.items():
                         if low_quality_path in self.image_cache: del self.image_cache[low_quality_path]
                    skipped_groups += 1
                    continue
            if skipped_groups > 0:
                print(f"移除了 {skipped_groups} 個在預載入時發現無效或出錯的組。")
                self.valid_groups = [bn for bn in self.valid_groups if any(self.min_quality <= int(q[1:]) <= self.max_quality for q, p in self.image_groups.get(bn, {}).items() if q != 'q100')]
                print(f"更新後有效組數量: {len(self.valid_groups)}")
            print(f"圖像預載入完成！實際載入 {len(self.image_cache)} 張圖像 ({loaded_count} 組)。")
            if not self.valid_groups:
                 raise ValueError("預載入後沒有有效的圖像組了，請檢查數據或緩存設置。")

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        if idx >= len(self.valid_groups):
             raise IndexError("索引超出範圍")
        base_name = self.valid_groups[idx]
        qualities = self.image_groups.get(base_name)
        if not qualities or 'q100' not in qualities:
            print(f"警告：組 {base_name} 數據不完整，返回空張量。")
            return self._get_empty_tensors()
        high_quality_path = qualities['q100']
        low_quality_options = []
        for q_str, path in qualities.items():
            if q_str != 'q100':
                try:
                    q_val = int(q_str[1:])
                    if self.min_quality <= q_val <= self.max_quality:
                        if self.cache_images and path in self.image_cache:
                             low_quality_options.append((q_str, path))
                        elif not self.cache_images and os.path.exists(path):
                             low_quality_options.append((q_str, path))
                        # else: 
                        #     print(f"警告: 文件 {path} 不存在或不在緩存中，跳過選項。")
                except ValueError:
                    continue
                except FileNotFoundError:
                    # print(f"警告: 文件 {path} 未找到，跳過選項。")
                    continue
        if not low_quality_options:
            # print(f"警告：組 {base_name} 找不到範圍內有效的低品質圖像文件，返回空張量。")
            return self._get_empty_tensors()
        weights = []
        options_paths = []
        for q_name, path in low_quality_options:
            q_num = int(q_name[1:])
            weight = 1.0 / (q_num + 1e-6)
            weights.append(weight)
            options_paths.append(path)
        sum_weight = sum(weights)
        if sum_weight > 0:
            normalized_weights = [w / sum_weight for w in weights]
        else:
             normalized_weights = [1.0 / len(options_paths)] * len(options_paths)
        try:
            low_quality_path = random.choices(options_paths, weights=normalized_weights, k=1)[0]
        except IndexError:
             print(f"警告：組 {base_name} 權重計算後選項列表為空，返回空張量。")
             return self._get_empty_tensors()
        try:
            if self.cache_images:
                if low_quality_path not in self.image_cache or high_quality_path not in self.image_cache:
                     print(f"警告：組 {base_name} 的圖像不在緩存中（可能預載入失敗），嘗試重新載入...")
                     try:
                         low_quality_image = Image.open(low_quality_path).convert("RGB")
                         high_quality_image = Image.open(high_quality_path).convert("RGB")
                         # 可選：如果現場加載成功，放入緩存
                         # self.image_cache[low_quality_path] = low_quality_image.copy()
                         # self.image_cache[high_quality_path] = high_quality_image.copy()
                     except FileNotFoundError:
                         print(f"錯誤：現場載入失敗，文件未找到: {low_quality_path} 或 {high_quality_path}")
                         return self._get_empty_tensors()
                     except Exception as load_err:
                         print(f"錯誤：現場載入時發生錯誤: {load_err}")
                         return self._get_empty_tensors()
                else:
                    low_quality_image = self.image_cache[low_quality_path].copy()
                    high_quality_image = self.image_cache[high_quality_path].copy()
            else:
                low_quality_image = Image.open(low_quality_path).convert("RGB")
                high_quality_image = Image.open(high_quality_path).convert("RGB")

            # --- 尺寸對齊與裁剪 ---
            width, height = low_quality_image.size
            if high_quality_image.size != (width, height):
                 high_quality_image = high_quality_image.resize((width, height), Image.LANCZOS) 
            crop_w = min(self.crop_size, width)
            crop_h = min(self.crop_size, height)
            if width >= crop_w and height >= crop_h:
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_quality_image, output_size=(crop_h, crop_w)
                )
                low_quality_image = transforms.functional.crop(low_quality_image, i, j, h, w)
                high_quality_image = transforms.functional.crop(high_quality_image, i, j, h, w)
            else:
                low_quality_image = transforms.functional.resize(low_quality_image, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BILINEAR)
                high_quality_image = transforms.functional.resize(high_quality_image, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.LANCZOS)
                crop_h, crop_w = self.crop_size, self.crop_size
            if self.augment:
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.hflip(low_quality_image)
                    high_quality_image = transforms.functional.hflip(high_quality_image)

                # 垂直翻轉
                if random.random() > 0.7: 
                    low_quality_image = transforms.functional.vflip(low_quality_image)
                    high_quality_image = transforms.functional.vflip(high_quality_image)

                # 旋轉
                if random.random() > 0.8:
                    angle = random.choice([90, 180, 270])
                    low_quality_image = transforms.functional.rotate(low_quality_image, angle)
                    high_quality_image = transforms.functional.rotate(high_quality_image, angle)

                # 輕微顏色抖動
                if random.random() > 0.8:
                    color_jitter = transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02 
                    )
                    low_quality_image = color_jitter(low_quality_image)

                # 隨機高斯噪聲
                # if random.random() > 0.95 and self.epoch < 30: 
                #     temp_transform = transforms.ToTensor()
                #     low_quality_tensor = temp_transform(low_quality_image)
                #     noise_level = random.uniform(0.005, 0.015) 
                #     noise = torch.randn_like(low_quality_tensor) * noise_level
                #     low_quality_tensor = torch.clamp(low_quality_tensor + noise, 0, 1)
                #     low_quality_image = transforms.ToPILImage()(low_quality_tensor)

            # --- 轉換為張量 ---
            if self.transform:
                low_quality_tensor = self.transform(low_quality_image)
                high_quality_tensor = self.transform(high_quality_image)
            else:
                low_quality_tensor = transforms.functional.to_tensor(low_quality_image)
                high_quality_tensor = transforms.functional.to_tensor(high_quality_image)
            if low_quality_tensor.shape != high_quality_tensor.shape:
                 print(f"警告：最終張量尺寸不匹配 {low_quality_tensor.shape} vs {high_quality_tensor.shape} for {base_name}. 嘗試調整...")
                 try:
                      h, w = low_quality_tensor.shape[1:]
                      high_quality_tensor = transforms.functional.resize(high_quality_tensor, [h, w], interpolation=transforms.InterpolationMode.LANCZOS)
                      if low_quality_tensor.shape != high_quality_tensor.shape:
                           print("錯誤：調整後尺寸仍然不匹配。返回空張量。")
                           return self._get_empty_tensors()
                 except Exception as resize_err:
                      print(f"錯誤：調整尺寸時出錯: {resize_err}。返回空張量。")
                      return self._get_empty_tensors()
            if torch.isnan(low_quality_tensor).any() or torch.isinf(low_quality_tensor).any() or \
               torch.isnan(high_quality_tensor).any() or torch.isinf(high_quality_tensor).any():
                print(f"警告：組 {base_name} 生成的張量包含 NaN/Inf 值。返回空張量。")
                return self._get_empty_tensors()
            return low_quality_tensor, high_quality_tensor
        except FileNotFoundError:
             print(f"錯誤：處理組 {base_name} 時找不到文件: {low_quality_path} 或 {high_quality_path}")
             return self._get_empty_tensors()
        except Exception as e:
            print(f"處理圖像時發生嚴重錯誤: {base_name} (Low: {low_quality_path}, High: {high_quality_path}), Error: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()
            return self._get_empty_tensors()

    def _get_empty_tensors(self):
        """返回一對空的、符合 crop_size 的張量"""
        target_shape = (3, self.crop_size, self.crop_size)
        return torch.zeros(target_shape), torch.zeros(target_shape)

# --- 工具函數 ---
def format_time(seconds):
    """格式化時間顯示"""
    if seconds < 0: seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def calculate_psnr(img1, img2, data_range=1.0):
    """計算峰值信噪比(PSNR)，假設輸入範圍為 [0, data_range]"""
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    if mse < 1e-10: mse = 1e-10
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()

# --- 大圖像處理 ---
def process_large_images(model, images, max_size=512, overlap=64):
    """處理大尺寸圖像，使用分塊處理和無縫拼接"""
    b, c, h, w = images.shape
    device = images.device
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    if h <= max_size and w <= max_size:
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
             with torch.no_grad():
                 return model(images.to(dtype)).to(images.dtype)
    result = torch.zeros_like(images, dtype=images.dtype) 
    weights = torch.zeros_like(images, dtype=torch.float32) 
    stride = max_size - overlap
    h_steps = math.ceil(max(0, h - max_size) / stride) + 1 if h > max_size else 1
    w_steps = math.ceil(max(0, w - max_size) / stride) + 1 if w > max_size else 1
    window_h = torch.hann_window(max_size, periodic=False, device=device).unsqueeze(1)
    window_w = torch.hann_window(max_size, periodic=False, device=device).unsqueeze(0)
    smooth_window = (window_h * window_w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1).to(torch.float32) 
    for i in range(h_steps):
        for j in range(w_steps):
            h_start = min(i * stride, h - max_size) if h > max_size else 0
            w_start = min(j * stride, w - max_size) if w > max_size else 0
            h_end = h_start + max_size
            w_end = w_start + max_size
            block = images[:, :, h_start:h_end, w_start:w_end]
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                 with torch.no_grad():
                     output_block = model(block.to(dtype)).to(images.dtype) 
            result[:, :, h_start:h_end, w_start:w_end] += output_block * smooth_window.to(images.dtype)
            weights[:, :, h_start:h_end, w_start:w_end] += smooth_window
    result = torch.where(weights > 1e-6, result / weights.to(images.dtype), result)
    return result

# --- 驗證函數 ---
def validate(generator, val_loader, device, max_validate_batches=None, return_images=False, crop_size=256):
    """驗證模型性能"""
    generator.eval() 
    val_psnr_list = []
    val_mse_list = []
    validation_images = []
    total_samples = 0
    processed_batches = 0
    with torch.no_grad(): 
        for i, (images, targets) in enumerate(val_loader):
            if max_validate_batches is not None and i >= max_validate_batches:
                break
            if images.nelement() == 0 or targets.nelement() == 0:
                print(f"\n警告：驗證時遇到空 Batch {i+1}，跳過。")
                continue
            images, targets = images.to(device), targets.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                fake_images = generator(images)
            fake_images = torch.clamp(fake_images, 0.0, 1.0)
            batch_size = images.size(0)
            for j in range(batch_size):
                psnr = calculate_psnr(fake_images[j], targets[j], data_range=1.0)
                if not math.isinf(psnr) and not math.isnan(psnr):
                    val_psnr_list.append(psnr)
                    mse = F.mse_loss(fake_images[j], targets[j]).item()
                    val_mse_list.append(mse)
                else:
                    print(f"\n警告：驗證時計算 PSNR 得到無效值 (inf/nan)，跳過樣本 {j}。")
                if return_images and i == 0 and j < 4:
                    try:
                        validation_images.append((
                            transforms.ToPILImage()(images[j].cpu()),
                            transforms.ToPILImage()(fake_images[j].cpu()),
                            transforms.ToPILImage()(targets[j].cpu())
                        ))
                    except Exception as img_err:
                         print(f"\n警告：保存驗證圖像時出錯: {img_err}")
            total_samples += batch_size
            processed_batches += 1
            if max_validate_batches is None and (i + 1) % 50 == 0: 
                 print(f"\r  驗證進度: Batch {i+1}/{len(val_loader)}...", end="")
    avg_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
    avg_mse = np.mean(val_mse_list) if val_mse_list else float('inf')
    print(f"\n  驗證完成 ({processed_batches} 批次, {total_samples} 樣本). 平均 PSNR: {avg_psnr:.4f} dB, 平均 MSE: {avg_mse:.6f}")
    if return_images:
        return avg_psnr, validation_images
    return avg_psnr

# --- 模型保存 ---
def save_model_with_metadata(model, path, metadata=None):
    """保存模型狀態字典並附加元數據"""
    try:
        torch.save(model.state_dict(), path)
        print(f"模型已保存至: {path}")
        if metadata:
            metadata_path = os.path.splitext(path)[0] + "_info.json"
            try:
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        cleaned_metadata[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor):
                                cleaned_metadata[key][sub_key] = sub_value.item()
                            elif isinstance(sub_value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                                cleaned_metadata[key][sub_key] = int(sub_value)
                            elif isinstance(sub_value, (np.float_, np.float16, np.float32, np.float64)):
                                cleaned_metadata[key][sub_key] = float(sub_value)
                            elif isinstance(sub_value, np.ndarray):
                                cleaned_metadata[key][sub_key] = sub_value.tolist() 
                            else:
                                cleaned_metadata[key][sub_key] = sub_value
                    elif isinstance(value, torch.Tensor):
                         cleaned_metadata[key] = value.item()
                    elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        cleaned_metadata[key] = int(value)
                    elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                        cleaned_metadata[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        cleaned_metadata[key] = value.tolist()
                    else:
                        cleaned_metadata[key] = value
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_metadata, f, ensure_ascii=False, indent=4)
                print(f"元數據已保存至: {metadata_path}")
            except TypeError as json_err:
                 print(f"警告：保存元數據時發生 JSON 序列化錯誤: {json_err}。元數據可能未完全保存。")
            except Exception as e:
                print(f"保存元數據時出錯: {e}")
    except Exception as save_err:
        print(f"錯誤：保存模型狀態時出錯: {save_err}")

# --- 核心訓練函數 ---
def train_model(generator, discriminator, train_loader, val_loader, criterion,
                g_optimizer, d_optimizer, scheduler_g, scheduler_d, num_epochs, device,
                save_dir="./models", log_dir="./logs", model_name="NS-IC-PixelRestore",
                gradient_accumulation_steps=4, checkpoint_interval=25,
                validation_interval=1,
                fast_validation=False,
                max_grad_norm=1.0,
                args=None):
    """針對馬賽克還原優化的訓練函數"""
    generator.to(device)
    discriminator.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    start_epoch = args.start_epoch if hasattr(args, 'start_epoch') and args.start_epoch else 0
    best_g_loss = args.best_g_loss if hasattr(args, 'best_g_loss') and args.best_g_loss else float('inf')
    best_psnr = args.best_psnr if hasattr(args, 'best_psnr') and args.best_psnr else 0.0
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"training_log_{model_name}_{timestamp}.csv")
    log_header = "Epoch,G_Loss,D_Loss,G_L1_Pixel,G_MSE_Pixel,G_SSIM,G_Edge,G_Color,G_Adv,PSNR,Time_Epoch,Time_Total,LR_G,LR_D\n"
    if not os.path.exists(log_file) or start_epoch == 0:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write(log_header)
    training_start_time = time.time()
    adversarial_weight = 0.001 
    pixel_restore_criterion = criterion.to(device)
    total_epochs_to_run = num_epochs 
    if hasattr(args, 'metadata') and args.metadata:
         model_metadata = args.metadata
         model_metadata["training_info"]["start_time_current_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
         model_metadata["training_info"]["total_epochs_planned"] = start_epoch + total_epochs_to_run
         model_metadata["training_args"] = vars(args) 
    else:
        model_metadata = {
            "model_name": model_name,
            "version": "NS-IC-v7", 
            "description": args.model_description if args and args.model_description else f"二次元馬賽克還原模型 (q{args.min_quality}-q{args.max_quality})",
            "architecture": {
                "type": "ImageQualityEnhancer",
                "num_rrdb_blocks": len(generator.rrdb_blocks) if hasattr(generator, 'rrdb_blocks') else 'Unknown',
                "features": generator.conv_first.out_channels if hasattr(generator, 'conv_first') else 'Unknown'
            },
            "training_args": vars(args) if args else {},
            "training_info": {
                "start_time_initial": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_time_current_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(train_loader.dataset) + len(val_loader.dataset) if train_loader and val_loader else 'Unknown',
                "total_epochs_planned": start_epoch + total_epochs_to_run,
                "quality_range": f"q{args.min_quality}-q{args.max_quality}" if args else "N/A",
                "loss_function": "PixelRestoreLoss",
                "adversarial_weight": adversarial_weight,
            },
            "performance": {
                "best_psnr": best_psnr,
                "best_g_loss": best_g_loss,
            }
        }
    print(f"將從 Epoch {start_epoch + 1} 開始訓練，共 {total_epochs_to_run} 個 Epochs。")
    print(f"使用損失函數: PixelRestoreLoss")
    print(f"對抗損失權重: {adversarial_weight}")
    print(f"梯度累積步數: {gradient_accumulation_steps}")
    print(f"有效批量大小: {args.batch_size * gradient_accumulation_steps}")

    # --- 訓練循環 ---
    for epoch in range(start_epoch, start_epoch + total_epochs_to_run):
        generator.train() 
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_g_losses_detailed = {'l1_pixel': 0.0, 'mse_pixel': 0.0, 'ssim': 0.0, 'edge': 0.0, 'color': 0.0, 'adv': 0.0}
        epoch_start_time = time.time()
        batch_count_accum = 0
        processed_batches_in_epoch = 0 
        current_dataset = train_loader.dataset
        if isinstance(current_dataset, torch.utils.data.Subset):
            current_dataset = current_dataset.dataset
        if hasattr(current_dataset, 'epoch'):
             current_dataset.epoch = epoch

        # --- 批次訓練 ---
        for i, (images, targets) in enumerate(train_loader):
            if images.nelement() == 0 or targets.nelement() == 0:
                # print(f"\n警告：訓練時遇到空 Batch {i+1}，跳過。")
                continue
            images, targets = images.to(device), targets.to(device)
            if torch.isnan(images).any() or torch.isinf(images).any() or \
               torch.isnan(targets).any() or torch.isinf(targets).any():
                print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 輸入數據包含 NaN/Inf 值。跳過此批次。")
                continue

            # ===== 訓練判別器 =====
            if batch_count_accum == 0:
                d_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                fake_images = generator(images)
                if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                    print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 生成器輸出包含 NaN/Inf 值。跳過此批次。")
                    if batch_count_accum > 0:
                         d_optimizer.zero_grad(set_to_none=True)
                         g_optimizer.zero_grad(set_to_none=True)
                         batch_count_accum = 0
                    continue
                fake_images_detached = fake_images.detach()
                real_outputs = discriminator(targets)
                fake_outputs = discriminator(fake_images_detached)
                d_loss_real = 0
                d_loss_fake = 0
                num_outputs = len(real_outputs)
                for scale_idx in range(num_outputs):
                    d_loss_real += torch.mean((real_outputs[scale_idx] - 1.0) ** 2)
                    d_loss_fake += torch.mean((fake_outputs[scale_idx] - 0.0) ** 2)
                d_loss = 0.5 * (d_loss_real + d_loss_fake) / num_outputs
                d_loss_scaled = d_loss / gradient_accumulation_steps
            if torch.isnan(d_loss_scaled).any() or torch.isinf(d_loss_scaled).any():
                print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 判別器損失為 NaN/Inf。跳過判別器梯度計算。")
            else:
                scaler.scale(d_loss_scaled).backward()
                epoch_d_loss += d_loss.item() 

            # ===== 訓練生成器 =====
            if batch_count_accum == 0:
                g_optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                     print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 生成器輸出 NaN/Inf，跳過生成器訓練。")
                     if batch_count_accum > 0:
                          d_optimizer.zero_grad(set_to_none=True)
                          g_optimizer.zero_grad(set_to_none=True)
                          batch_count_accum = 0
                     continue
                fake_outputs_for_g = discriminator(fake_images)
                loss_components = pixel_restore_criterion(fake_images, targets)
                g_loss_pixel_restore = loss_components['total'] 
                adversarial_g_loss = 0
                num_outputs_g = len(fake_outputs_for_g)
                for scale_idx in range(num_outputs_g):
                    adversarial_g_loss += torch.mean((fake_outputs_for_g[scale_idx] - 1.0) ** 2)
                adversarial_g_loss = adversarial_g_loss / num_outputs_g
                g_loss = g_loss_pixel_restore + adversarial_weight * adversarial_g_loss
                g_loss_scaled = g_loss / gradient_accumulation_steps
            if torch.isnan(g_loss_scaled).any() or torch.isinf(g_loss_scaled).any():
                 print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 生成器損失為 NaN/Inf。跳過生成器梯度計算。")
            else:
                scaler.scale(g_loss_scaled).backward()
                epoch_g_loss += g_loss.item()
                epoch_g_losses_detailed['l1_pixel'] += loss_components['l1_pixel'].item()
                epoch_g_losses_detailed['mse_pixel'] += loss_components['mse_pixel'].item()
                epoch_g_losses_detailed['ssim'] += loss_components['ssim'].item()
                epoch_g_losses_detailed['edge'] += loss_components['edge'].item()
                epoch_g_losses_detailed['color'] += loss_components['color'].item()
                # epoch_g_losses_detailed['high_freq'] += loss_components['high_freq'].item() # 如果啟用
                epoch_g_losses_detailed['adv'] += adversarial_g_loss.item()
            batch_count_accum += 1
            processed_batches_in_epoch += 1 
            if batch_count_accum == gradient_accumulation_steps or (i == len(train_loader) - 1):
                if not (torch.isnan(d_loss_scaled).any() or torch.isinf(d_loss_scaled).any()):
                    scaler.unscale_(d_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
                    scaler.step(d_optimizer)
                if not (torch.isnan(g_loss_scaled).any() or torch.isinf(g_loss_scaled).any()):
                    scaler.unscale_(g_optimizer)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
                    scaler.step(g_optimizer)
                scaler.update()
                batch_count_accum = 0
                # g_optimizer.zero_grad(set_to_none=True)
                # d_optimizer.zero_grad(set_to_none=True)


            # --- 顯示訓練進度 ---
            if (i + 1) % 20 == 0 or i == len(train_loader) - 1: 
                progress = (i + 1) / len(train_loader)
                percentage = progress * 100
                elapsed_time = time.time() - epoch_start_time
                eta_seconds = (elapsed_time / progress - elapsed_time) if progress > 0 else 0
                fill_length = int(30 * progress)
                space_length = 30 - fill_length
                current_g_loss = g_loss.item() if not torch.isnan(g_loss).any() else 0
                current_d_loss = d_loss.item() if not torch.isnan(d_loss).any() else 0
                print(f"\rEpoch [{epoch+1}/{start_epoch + total_epochs_to_run}] "
                      f"{percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| "
                      f"[{format_time(elapsed_time)}<{format_time(eta_seconds)}] "
                      f"G:{current_g_loss:.4f} D:{current_d_loss:.4f}", end="")

        # --- Epoch 結束 ---
        epoch_time = time.time() - epoch_start_time
        total_training_time = time.time() - training_start_time
        num_batches_processed = processed_batches_in_epoch
        if num_batches_processed == 0:
             print(f"\n警告：Epoch {epoch+1} 沒有處理任何有效的批次，跳過此 Epoch 的統計和保存。")
             continue 
        avg_g_loss = epoch_g_loss / num_batches_processed
        avg_d_loss = epoch_d_loss / num_batches_processed
        avg_g_losses_detailed = {k: v / num_batches_processed for k, v in epoch_g_losses_detailed.items()}

        # --- 執行驗證 ---
        val_psnr = 0.0
        validation_images = []
        if (epoch + 1) % validation_interval == 0:
            print(f"\n--- 驗證輪數 {epoch+1} ---")
            validate_batches = args.validate_batches if fast_validation else None
            val_psnr, validation_images = validate(generator, val_loader, device,
                                                 max_validate_batches=validate_batches,
                                                 return_images=True, 
                                                 crop_size=args.crop_size)
            print(f"--- 驗證完成 ---")
            if validation_images:
                 val_img_dir = os.path.join(log_dir, "validation_images", f"epoch_{epoch+1}")
                 os.makedirs(val_img_dir, exist_ok=True)
                 for idx, (img_in, img_out, img_tgt) in enumerate(validation_images):
                     try:
                         img_in.save(os.path.join(val_img_dir, f"{idx}_input.png"))
                         img_out.save(os.path.join(val_img_dir, f"{idx}_output_psnr{val_psnr:.2f}.png"))
                         img_tgt.save(os.path.join(val_img_dir, f"{idx}_target.png"))
                     except Exception as img_save_err:
                         print(f"保存驗證圖像時出錯: {img_save_err}")

        # --- 更新學習率 ---
        current_lr_g = g_optimizer.param_groups[0]['lr']
        current_lr_d = d_optimizer.param_groups[0]['lr']

        lr_updated = False
        if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_psnr > 0: 
                 scheduler_g.step(val_psnr)
                 scheduler_d.step(val_psnr) 
                 lr_updated = True
            else:
                 print("警告：無有效 PSNR，跳過 ReduceLROnPlateau step")
        else:
            scheduler_g.step()
            scheduler_d.step()
            lr_updated = True
        new_lr_g = g_optimizer.param_groups[0]['lr']
        new_lr_d = d_optimizer.param_groups[0]['lr']
        if lr_updated and (new_lr_g != current_lr_g or new_lr_d != current_lr_d) :
            print(f"\n學習率更新 -> G: {new_lr_g:.7f}, D: {new_lr_d:.7f}")
        print(f"\nEpoch [{epoch+1}/{start_epoch + total_epochs_to_run}] Summary | Time: {format_time(epoch_time)} "
              f"| Total Time: {format_time(total_training_time)}")
        print(f"  平均損失 -> G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}")
        loss_detail_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_g_losses_detailed.items()])
        print(f"  G Loss 細項 -> {loss_detail_str}")
        if val_psnr > 0:
            print(f"  驗證 PSNR: {val_psnr:.4f} dB")
        print(f"  當前學習率 -> G: {new_lr_g:.7f}, D: {new_lr_d:.7f}")
        try:
            with open(log_file, "a", encoding='utf-8') as f:
                log_line = (f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},"
                            f"{avg_g_losses_detailed.get('l1_pixel', 0.0):.6f},"
                            f"{avg_g_losses_detailed.get('mse_pixel', 0.0):.6f},"
                            f"{avg_g_losses_detailed.get('ssim', 0.0):.6f},"
                            f"{avg_g_losses_detailed.get('edge', 0.0):.6f},"
                            f"{avg_g_losses_detailed.get('color', 0.0):.6f},"
                            # f"{avg_g_losses_detailed.get('high_freq', 0.0):.6f}," # 如果啟用
                            f"{avg_g_losses_detailed.get('adv', 0.0):.6f},"
                            f"{val_psnr:.6f},{epoch_time:.2f},{total_training_time:.2f},"
                            f"{new_lr_g:.8f},{new_lr_d:.8f}\n")
                f.write(log_line)
        except IOError as log_err:
             print(f"錯誤：無法寫入日誌文件 {log_file}: {log_err}")

        # --- 更新模型元數據 ---
        model_metadata["training_info"]["last_completed_epoch"] = epoch + 1
        model_metadata["training_info"]["total_time_seconds"] = total_training_time
        model_metadata["performance"]["current_psnr"] = val_psnr
        model_metadata["performance"]["current_g_loss"] = avg_g_loss

        # --- 保存模型 ---
        is_best_psnr = False
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            is_best_psnr = True
            model_metadata["performance"]["best_psnr"] = best_psnr
            model_metadata["performance"]["best_psnr_epoch"] = epoch + 1
            best_psnr_path = os.path.join(save_dir, f"{model_name}_best_psnr.pth")
            save_model_with_metadata(generator, best_psnr_path, model_metadata)
            print(f"*** 新的最佳 PSNR 模型已保存 (Epoch {epoch+1}, PSNR: {val_psnr:.4f} dB) ***")

        # 可選：保存最佳損失模型
        is_best_loss = False
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            is_best_loss = True
            model_metadata["performance"]["best_g_loss"] = best_g_loss
            model_metadata["performance"]["best_g_loss_epoch"] = epoch + 1
            # best_loss_path = os.path.join(save_dir, f"{model_name}_best_loss.pth")
            # save_model_with_metadata(generator, best_loss_path, model_metadata)
            # print(f"--- 新的最佳 G_Loss 模型已儲存 (Epoch {epoch+1}, G_Loss: {avg_g_loss:.4f}) ---")

        # --- 保存檢查點 ---
        save_checkpoint_flag = (epoch + 1) % checkpoint_interval == 0 or epoch == (start_epoch + total_epochs_to_run - 1)
        if save_checkpoint_flag or is_best_psnr:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': scheduler_g.state_dict(),
                'd_scheduler_state_dict': scheduler_d.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
                'psnr': val_psnr,
                'best_g_loss': best_g_loss, 
                'best_psnr': best_psnr, 
                'metadata': model_metadata,
                'args': args
            }
            latest_checkpoint_path = os.path.join(save_dir, f"{model_name}_latest.pth.tar") 
            try:
                torch.save(checkpoint, latest_checkpoint_path)
                print(f"檢查點已保存至: {latest_checkpoint_path}")
            except Exception as ckpt_save_err:
                 print(f"錯誤：保存檢查點時出錯: {ckpt_save_err}")

            # 可選：如果需要，可以額外保存特定 epoch 的檢查點
            # if save_checkpoint_flag:
            #     epoch_checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth.tar")
            #     try:
            #         torch.save(checkpoint, epoch_checkpoint_path)
            #         print(f"Epoch {epoch+1} 檢查點已保存至: {epoch_checkpoint_path}")
            #     except Exception as ckpt_save_err:
            #          print(f"錯誤：保存 Epoch {epoch+1} 檢查點時出錯: {ckpt_save_err}")

        # --- 清理記憶體 ---
        if (epoch + 1) % 10 == 0:
            del images, targets, fake_images, fake_images_detached, real_outputs, fake_outputs, fake_outputs_for_g
            del loss_components, g_loss_pixel_restore, adversarial_g_loss, g_loss, d_loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # print(f"Epoch {epoch+1} 結束，清理 CUDA 緩存。")

    # --- 訓練結束 ---
    final_gen_path = os.path.join(save_dir, f"{model_name}_final_generator_epoch_{start_epoch + total_epochs_to_run}.pth")
    model_metadata["training_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_metadata["training_info"]["total_time_formatted"] = format_time(time.time() - training_start_time)
    model_metadata["performance"]["final_psnr"] = val_psnr
    model_metadata["performance"]["final_g_loss"] = avg_g_loss
    save_model_with_metadata(generator, final_gen_path, model_metadata)
    print("\n" + "="*30 + " 訓練結束 " + "="*30)
    print(f"最終生成器模型已保存至: {final_gen_path}")
    print(f"最佳 PSNR: {best_psnr:.4f} dB (在 Epoch {model_metadata['performance'].get('best_psnr_epoch', 'N/A')})")
    print(f"最佳 G_Loss: {best_g_loss:.4f} (在 Epoch {model_metadata['performance'].get('best_g_loss_epoch', 'N/A')})")
    print(f"訓練日誌文件保存在: {log_file}")
    return generator, discriminator 

# --- 主執行區塊 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門影像品質增強訓練器 v7.0 - 馬賽克還原優化版')
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01_P', help='包含 q100 和像素化圖像 (q10-q90) 的訓練集路徑')
    parser.add_argument('--save_dir', type=str, default='./models_pixel', help='模型保存路徑')
    parser.add_argument('--log_dir', type=str, default='./logs_pixel', help='日誌保存路徑')
    parser.add_argument('--model_name', type=str, default='NS-IC-Ritsuka-HQ', help='模型名稱')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點路徑 (.pth.tar)')
    parser.add_argument('--model_description', type=str, default='二次元馬賽克還原模型', help='模型描述')
    parser.add_argument('--num_epochs', type=int, default=1000, help='總訓練輪數 (如果 resume，則為從檢查點開始的總目標輪數)')
    parser.add_argument('--batch_size', type=int, default=4, help='訓練批量 (根據 VRAM 調整，馬賽克還原可能需要更大模型或更高分辨率，建議減小)')
    parser.add_argument('--crop_size', type=int, default=256, help='訓練圖像裁剪大小')
    parser.add_argument('--quality_range', type=str, default='70-90', help='訓練使用的品質範圍')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='生成器初始學習率 (可適當降低)') 
    parser.add_argument('--d_lr_factor', type=float, default=0.5, help='判別器學習率相對於生成器的係數')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='優化器類型')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW 的權重衰減 (可適當增加)') 
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'step'], help='學習率調度器類型')
    parser.add_argument('--plateau_patience', type=int, default=10, help='ReduceLROnPlateau 的耐心值 (epochs)') 
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='ReduceLROnPlateau 的學習率衰減因子')
    parser.add_argument('--cosine_t_max', type=int, default=1000, help='CosineAnnealingLR 的 T_max (建議設為總 epochs)') 
    parser.add_argument('--step_size', type=int, default=100, help='StepLR 的步長 (epochs)')
    parser.add_argument('--step_gamma', type=float, default=0.5, help='StepLR 的衰減因子')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='最小學習率下限')
    parser.add_argument('--grad_accum', type=int, default=8, help='梯度累積步數 (有效批量 = batch_size * grad_accum)') 
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪範數上限')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作線程數 (根據 CPU 和內存調整)')
    parser.add_argument('--cache_images', action='store_true', default=False, help='將圖片快取到記憶體以加速 (需要大量內存)')
    parser.add_argument('--no_cache_images', action='store_false', dest='cache_images', help='禁用圖片快取 (默認)')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='啟用 pin_memory 加速數據轉移 (如果內存充足)')
    parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory', help='禁用 pin_memory')
    parser.add_argument('--validation_interval', type=int, default=1, help='多少個 epoch 執行一次驗證')
    parser.add_argument('--fast_validation', action='store_true', default=False, help='啟用快速驗證 (只驗證部分批次)')
    parser.add_argument('--validate_batches', type=int, default=30, help='快速驗證使用的批次數 (減少數量)')
    parser.add_argument('--checkpoint_interval', type=int, default=25, help='檢查點保存間隔 (epoch)') 
    args = parser.parse_args()
    try:
        min_q_str, max_q_str = args.quality_range.split('-')
        args.min_quality = int(min_q_str)
        args.max_quality = int(max_q_str)
        if not (0 < args.min_quality <= args.max_quality <= 100):
            raise ValueError("品質範圍必須在 1-100 之間，且最小值不大於最大值")
        print(f"設定訓練品質範圍: q{args.min_quality} - q{args.max_quality}")
    except ValueError as e:
        parser.error(f"無效的 quality_range 格式或值: {args.quality_range}. 請使用 'min-max' 格式，例如 '10-90'. 錯誤: {e}")

    # --- 設定隨機種子 ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # 確保 deterministic (如果需要完全可重複性，但可能影響性能)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # 禁用 benchmark 以確保 deterministic

    # --- 創建目錄 ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    try:
        cpu_count = os.cpu_count()
        if cpu_count:
             num_workers = min(args.num_workers, max(1, cpu_count // 2))
        else:
             num_workers = args.num_workers
    except NotImplementedError:
        num_workers = args.num_workers
    print(f"使用 {num_workers} 個資料載入工作線程")
    transform = transforms.Compose([transforms.ToTensor()])

    # --- 載入數據集 ---
    try:
        dataset = QualityDataset(
            args.data_dir,
            transform=transform,
            crop_size=args.crop_size,
            augment=True, 
            cache_images=args.cache_images,
            min_quality=args.min_quality,
            max_quality=args.max_quality
        )
    except ValueError as e:
        print(f"錯誤：無法初始化資料集: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"錯誤：找不到資料目錄: {args.data_dir}")
        sys.exit(1)
    except Exception as ds_err:
         print(f"錯誤：初始化資料集時發生未知錯誤: {ds_err}")
         sys.exit(1)

    # --- 劃分訓練集和驗證集 ---
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("錯誤：資料集為空，請檢查 data_dir 和 quality_range 設置。")
        sys.exit(1)

    val_split = 0.05 
    val_size = max(1, int(val_split * dataset_size)) 
    train_size = dataset_size - val_size
    print(f"數據集總大小: {dataset_size}, 訓練集: {train_size}, 驗證集: {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed) 
    )
    
    # --- 數據載入器 ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False, 
        prefetch_factor=2 if num_workers > 0 else None, 
        drop_last=True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, 
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=True if max(1, num_workers // 2) > 0 else False
    )

    # --- 初始化模型 ---
    generator = ImageQualityEnhancer(num_rrdb_blocks=16, features=64)
    discriminator = MultiScaleDiscriminator(num_scales=3, input_channels=3)
    criterion = PixelRestoreLoss() 

    # --- 創建優化器 ---
    optimizer_choice = torch.optim.AdamW if args.optimizer == 'AdamW' else torch.optim.Adam
    g_optimizer = optimizer_choice(
        generator.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay if args.optimizer == 'AdamW' else 0
    )
    d_optimizer = optimizer_choice(
        discriminator.parameters(),
        lr=args.learning_rate * args.d_lr_factor,
        betas=(0.9, 0.999), 
        weight_decay=args.weight_decay if args.optimizer == 'AdamW' else 0
    )
    start_epoch_for_scheduler = 0 
    args.start_epoch = 0
    args.best_psnr = 0.0
    args.best_g_loss = float('inf')
    args.metadata = None 
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"載入檢查點: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location='cpu')
                args.start_epoch = checkpoint['epoch']
                generator.load_state_dict(checkpoint['generator_state_dict'])
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint:
                     temp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
                     temp_scaler.load_state_dict(checkpoint['scaler_state_dict'])
                     # scaler = temp_scaler # 直接使用加載後的 scaler
                     print("GradScaler 狀態已恢復。")
                args.best_psnr = checkpoint.get('best_psnr', 0.0)
                args.best_g_loss = checkpoint.get('best_g_loss', float('inf'))
                args.metadata = checkpoint.get('metadata', None)
                # 恢復命令行參數 (可選)
                # loaded_args = checkpoint.get('args', None)
                start_epoch_for_scheduler = args.start_epoch 
                if 'g_scheduler_state_dict' in checkpoint:
                    pass 
                if 'd_scheduler_state_dict' in checkpoint:
                    pass #
                print(f"檢查點載入成功，將從 Epoch {args.start_epoch + 1} 繼續訓練。")
                print(f"  恢復的最佳 PSNR: {args.best_psnr:.4f} dB")
                print(f"  恢復的最佳 G_Loss: {args.best_g_loss:.4f}")
            except FileNotFoundError:
                 print(f"錯誤：找不到檢查點文件 '{args.resume}'。")
                 args.resume = None 
                 # sys.exit(1) 
            except Exception as e:
                print(f"錯誤：載入檢查點 '{args.resume}' 時發生錯誤: {e}")
                print("將從頭開始訓練。")
                args.resume = None
                args.start_epoch = 0
                args.best_psnr = 0.0
                args.best_g_loss = float('inf')
                args.metadata = None
        else:
            print(f"警告：指定的檢查點文件 '{args.resume}' 不是一個有效文件。將從頭開始訓練。")
            args.resume = None

    # --- 創建學習率調度器 ---
    if args.scheduler == 'cosine':
        total_target_epochs = args.num_epochs
        scheduler_g = CosineAnnealingLR(
            g_optimizer,
            T_max=total_target_epochs - start_epoch_for_scheduler if total_target_epochs > start_epoch_for_scheduler else 1,
            eta_min=args.min_lr,
            last_epoch=start_epoch_for_scheduler - 1 if start_epoch_for_scheduler > 0 else -1 
        )
        scheduler_d = CosineAnnealingLR(
            d_optimizer,
            T_max=total_target_epochs - start_epoch_for_scheduler if total_target_epochs > start_epoch_for_scheduler else 1,
            eta_min=args.min_lr * args.d_lr_factor,
            last_epoch=start_epoch_for_scheduler - 1 if start_epoch_for_scheduler > 0 else -1
        )
    elif args.scheduler == 'step':
         scheduler_g = StepLR(
             g_optimizer,
             step_size=args.step_size,
             gamma=args.step_gamma,
             last_epoch=start_epoch_for_scheduler - 1 if start_epoch_for_scheduler > 0 else -1
         )
         scheduler_d = StepLR(
             d_optimizer,
             step_size=args.step_size,
             gamma=args.step_gamma,
             last_epoch=start_epoch_for_scheduler - 1 if start_epoch_for_scheduler > 0 else -1
         )
    else:  # 'plateau'
        scheduler_g = ReduceLROnPlateau(
            g_optimizer,
            mode='max', 
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True,
            min_lr=args.min_lr,
            cooldown=max(1, args.plateau_patience // 2)
        )
        scheduler_d = ReduceLROnPlateau(
            d_optimizer,
            mode='max',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True,
            min_lr=args.min_lr * args.d_lr_factor,
            cooldown=max(1, args.plateau_patience // 2)
        )
    if args.resume and 'checkpoint' in locals():
         try:
             if 'g_scheduler_state_dict' in checkpoint:
                 scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
                 print("生成器調度器狀態已恢復。")
             if 'd_scheduler_state_dict' in checkpoint:
                 scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
                 print("判別器調度器狀態已恢復。")
         except Exception as scheduler_load_err:
              print(f"警告：恢復調度器狀態時出錯: {scheduler_load_err}。調度器將從當前 epoch 開始。")
    remaining_epochs = args.num_epochs - args.start_epoch
    if remaining_epochs <= 0:
         print(f"起始 epoch ({args.start_epoch}) 已達到或超過目標 epoch ({args.num_epochs})。無需繼續訓練。")
         sys.exit(0)
    else:
         print(f"將訓練 {remaining_epochs} 個 epochs (從 {args.start_epoch+1} 到 {args.num_epochs})")
    print("\n" + "="*30 + " 開始訓練 " + "="*30)
    train_model(
        generator,
        discriminator,
        train_loader,
        val_loader,
        criterion,
        g_optimizer,
        d_optimizer,
        scheduler_g,
        scheduler_d,
        num_epochs=remaining_epochs, 
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=args.validation_interval,
        fast_validation=args.fast_validation,
        max_grad_norm=args.max_grad_norm,
        args=args 
    )
