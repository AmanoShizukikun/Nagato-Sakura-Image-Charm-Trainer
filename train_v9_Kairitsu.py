import os
import sys
import argparse
import time
import math
import random
import json
import gc
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.amp import GradScaler, autocast
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- 模型導入 ---
try:
    from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator
    except ImportError as e:
        print(f"錯誤：無法從 src.IQE 導入模型。請確保 src 目錄存在於 {parent_dir} 下。")
        print(f"詳細錯誤: {e}")
        sys.exit(1)

# --- CUDNN 優化 ---
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# --- 損失函數 ---
class SSIMLoss(nn.Module):
    """結構相似性指數測量 (SSIM) 損失函數"""
    def __init__(self, window_size=11, size_average=True, channel=3, data_range=1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.data_range = data_range
        self.register_buffer('window', self._create_window(window_size, self.channel))
        self.C1 = (0.01 * self.data_range) ** 2
        self.C2 = (0.03 * self.data_range) ** 2

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
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        loss = 1.0 - ssim_map
        if self.size_average:
            return loss.mean()
        else:
            return loss.mean([1, 2, 3])

class JPEGRestorationLoss(nn.Module):
    """
    為寫實圖像 JPEG 壓縮修復設計的綜合損失函數。
    重點在於 L1、SSIM、邊緣和高頻細節。
    """
    def __init__(self, ssim_window_size=11, device='cpu'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=ssim_window_size, size_average=True, channel=3).to(device)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer('sobel_x', sobel_x.to(device))
        self.register_buffer('sobel_y', sobel_y.to(device))
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).to(device))

    def _extract_sobel_edges(self, x):
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        grad_x = F.conv2d(x_padded, self.sobel_x, padding='valid', groups=3)
        grad_y = F.conv2d(x_padded, self.sobel_y, padding='valid', groups=3)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude

    def _extract_high_freq(self, x):
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        return F.conv2d(x_padded, self.laplacian_kernel, padding='valid', groups=3)

    def forward(self, generated, target):
        """
        計算總損失。
        Args:
            generated (torch.Tensor): 生成器輸出的圖像。
            target (torch.Tensor): 目標（高質量）圖像
        Returns:
            dict: 包含總損失 ('total') 和各分項損失的字典。
        """
        loss_l1 = self.l1_loss(generated, target)
        loss_mse = self.mse_loss(generated, target)
        loss_ssim = self.ssim_loss(generated, target)
        gen_edges = self._extract_sobel_edges(generated)
        target_edges = self._extract_sobel_edges(target)
        loss_edge = self.l1_loss(gen_edges, target_edges)
        gen_hf = self._extract_high_freq(generated)
        target_hf = self._extract_high_freq(target)
        loss_hf = self.l1_loss(gen_hf, target_hf)
        loss_color = self.l1_loss(generated.mean(dim=[2, 3]), target.mean(dim=[2, 3]))

        # --- 組合損失 ---
        total_loss = (
            loss_l1 * 1.5 +
            loss_mse * 0.05 +
            loss_ssim * 2.0 +
            loss_edge * 1.2 +
            loss_hf * 1.0 +
            loss_color * 0.1
        )

        loss_dict = {
            'total': total_loss,
            'l1': loss_l1,
            'mse': loss_mse,
            'ssim': loss_ssim,
            'edge': loss_edge,
            'hf': loss_hf,
            'color': loss_color
        }
        return loss_dict

# --- 資料集類別 ---
class QualityDataset(Dataset):
    """
    加載低質量/高質量圖像對用於訓練。
    支持圖像緩存、數據增強和動態採樣低質量版本。
    假定文件名格式為 'basename_qXX.ext' 和 'basename_q100.ext'。
    會遞迴掃描指定目錄及其子目錄。
    """
    def __init__(self, image_dir, transform=None, crop_size=256, augment=True, cache_images=False,
                 min_quality=10, max_quality=90):
        self.image_dir = image_dir
        self.transform = transform
        self.crop_size = crop_size
        self.augment = augment
        self.cache_images = cache_images
        self.image_cache = {}
        self.image_groups = {}
        self.min_quality = min_quality
        self.max_quality = max_quality
        print(f"初始化資料集，遞迴掃描路徑: {image_dir}")
        print(f"訓練品質範圍: q{min_quality} - q{max_quality}")
        print(f"圖像裁剪大小: {crop_size}x{crop_size}")
        print(f"數據增強: {'啟用' if augment else '禁用'}")
        print(f"圖像快取: {'啟用' if cache_images else '禁用'}")

        # --- 遞迴掃描目錄並分組圖像 ---
        all_image_paths = []
        print("遞迴掃描圖像文件中...")
        try:
            for root, _, files in os.walk(image_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        all_image_paths.append(os.path.join(root, f))
        except FileNotFoundError:
            print(f"錯誤：找不到資料目錄 {image_dir}")
            sys.exit(1)
        print(f"找到 {len(all_image_paths)} 個圖像文件。")
        parsed_count = 0
        ignored_count = 0
        # --- 文件名解析和分組邏輯 ---
        for path in all_image_paths:
            try:
                base_name, quality_str = os.path.splitext(os.path.basename(path))[0].rsplit('_q', 1)
                unique_base_name = base_name
                if unique_base_name not in self.image_groups:
                    self.image_groups[unique_base_name] = {}
                self.image_groups[unique_base_name]['q' + quality_str] = path
                parsed_count += 1
            except ValueError:
                ignored_count += 1
            except Exception as e:
                print(f"解析檔名時發生錯誤 {os.path.basename(path)}: {e}")
                ignored_count += 1
        print(f"檔名解析完成：成功解析 {parsed_count} 個，忽略 {ignored_count} 個。")

        # --- 過濾有效組 ---
        self.valid_groups = []
        for base_name, qualities in self.image_groups.items():
            has_q100 = 'q100' in qualities
            has_low_q = any(
                q_str != 'q100' and q_str[1:].isdigit() and self.min_quality <= int(q_str[1:]) <= self.max_quality
                for q_str in qualities
            )
            if has_q100 and has_low_q:
                self.valid_groups.append(base_name)
        print(f"找到 {len(self.valid_groups)} 組有效的圖像 (包含 q100 和 q{min_quality}-q{max_quality} 範圍內的圖像)")
        if not self.valid_groups:
             raise ValueError(f"在目錄 {image_dir} 及其子目錄中找不到有效的圖像組。請檢查文件名格式和 quality_range。")
        if self.cache_images and len(self.valid_groups) > 0:
            print(f"預載入 {len(self.valid_groups)} 組圖像到記憶體...")
            loaded_count = 0
            for base_name in tqdm(self.valid_groups, desc="預載入圖像", unit="組"):
                 qualities = self.image_groups[base_name]
                 for q_str, path in qualities.items():
                     if path not in self.image_cache:
                         img = self._load_image(path)
                         if img:
                             self.image_cache[path] = img
                             loaded_count += 1
                 # if (i + 1) % 100 == 0:
                 #     print(f"  已處理 {i+1}/{len(self.valid_groups)} 組...")
            print(f"圖像預載入完成！實際載入 {loaded_count} 張圖像。")

        # --- 定義數據增強 ---
        if self.augment:
            self.augmentation_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=5),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
        else:
            self.augmentation_transform = None

    def __len__(self):
        return len(self.valid_groups)

    def _load_image(self, path):
        """從緩存或磁盤加載圖像"""
        if self.cache_images and path in self.image_cache:
            return self.image_cache[path].copy()
        else:
            try:
                img = Image.open(path).convert('RGB')
                if self.cache_images:
                    self.image_cache[path] = img.copy()
                return img
            except FileNotFoundError:
                 print(f"警告：找不到圖像文件 {path}")
                 return None
            except Exception as e:
                 print(f"警告：加載圖像 {path} 時出錯: {e}")
                 return None

    def _get_paired_images(self, idx):
        """獲取一對低質量和高質量圖像"""
        base_name = self.valid_groups[idx]
        qualities = self.image_groups[base_name]
        high_quality_path = qualities['q100']
        low_quality_options = [
            (q_str, path) for q_str, path in qualities.items()
            if q_str != 'q100' and q_str[1:].isdigit() and self.min_quality <= int(q_str[1:]) <= self.max_quality
        ]
        if not low_quality_options:
            raise RuntimeError(f"組 {base_name} 缺少範圍內的低質量圖像。")
        weights = [1.0] * len(low_quality_options)
        options_paths = [p for q, p in low_quality_options]
        sum_weight = sum(weights)
        normalized_weights = [w / sum_weight for w in weights] if sum_weight > 0 else [1.0 / len(options_paths)] * len(options_paths)
        low_quality_path = random.choices(options_paths, weights=normalized_weights, k=1)[0]
        low_img = self._load_image(low_quality_path)
        high_img = self._load_image(high_quality_path)
        return low_img, high_img

    def __getitem__(self, idx):
        try:
            low_img, high_img = self._get_paired_images(idx)
            if low_img is None or high_img is None:
                 print(f"警告：索引 {idx} 的圖像對加載失敗，返回空張量。")
                 return self._get_empty_tensors()
            if low_img.size[0] < self.crop_size or low_img.size[1] < self.crop_size or \
               high_img.size[0] < self.crop_size or high_img.size[1] < self.crop_size:
                low_img = transforms.functional.resize(low_img, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC)
                high_img = transforms.functional.resize(high_img, (self.crop_size, self.crop_size), interpolation=transforms.InterpolationMode.BICUBIC)
                i, j, h, w = 0, 0, self.crop_size, self.crop_size
            else:
                i, j, h, w = transforms.RandomCrop.get_params(low_img, output_size=(self.crop_size, self.crop_size))
            low_img_cropped = transforms.functional.crop(low_img, i, j, h, w)
            high_img_cropped = transforms.functional.crop(high_img, i, j, h, w)
            if self.augmentation_transform:
                seed = random.randint(0, 2**32 - 1)
                random.seed(seed); torch.manual_seed(seed)
                low_img_aug = self.augmentation_transform(low_img_cropped)
                random.seed(seed); torch.manual_seed(seed)
                high_img_aug = self.augmentation_transform(high_img_cropped)
            else:
                low_img_aug = low_img_cropped
                high_img_aug = high_img_cropped
            if self.transform:
                low_tensor = self.transform(low_img_aug)
                high_tensor = self.transform(high_img_aug)
            else:
                low_tensor = transforms.functional.to_tensor(low_img_aug)
                high_tensor = transforms.functional.to_tensor(high_img_aug)
            return low_tensor, high_tensor
        except FileNotFoundError as e:
            print(f"錯誤：找不到圖像文件 {e.filename} (索引 {idx})。返回空張量。")
            return self._get_empty_tensors()
        except RuntimeError as e:
             print(f"錯誤：處理索引 {idx} 時發生運行時錯誤: {e}。返回空張量。")
             return self._get_empty_tensors()
        except Exception as e:
            print(f"錯誤：處理索引 {idx} 時發生未知異常: {e}。返回空張量。")
            traceback.print_exc()
            return self._get_empty_tensors()

    def _get_empty_tensors(self):
        """返回一對空的、符合 crop_size 的張量"""
        shape = (3, self.crop_size, self.crop_size)
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)

# --- 工具函數 ---
def format_time(seconds):
    """格式化時間顯示"""
    if seconds < 0: seconds = 0
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

def calculate_psnr(img1, img2, data_range=1.0):
    """計算峰值信噪比(PSNR)，假設輸入範圍為 [0, data_range]"""
    if not torch.is_tensor(img1) or not torch.is_tensor(img2): return 0.0
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    if mse < 1e-10: mse = 1e-10 # 防止 mse 過小導致 log10 出錯
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()

def process_large_images(model, images, max_size=512, overlap=64):
    """處理大尺寸圖像，使用分塊處理和無縫拼接"""
    model.eval()
    b, c, h, w = images.shape
    device = images.device
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    if h <= max_size and w <= max_size:
        with torch.no_grad(), autocast(device.type, dtype=dtype, enabled=(device.type == 'cuda')):
            output = model(images)
        return output.to(images.dtype)
    result = torch.zeros_like(images, dtype=images.dtype)
    weights = torch.zeros_like(images, dtype=torch.float32)
    stride = max_size - overlap
    h_steps = math.ceil(max(0, h - max_size) / stride) + 1 if h > max_size else 1
    w_steps = math.ceil(max(0, w - max_size) / stride) + 1 if w > max_size else 1
    window_h = torch.hann_window(max_size, periodic=False, device=device).unsqueeze(1)
    window_w = torch.hann_window(max_size, periodic=False, device=device).unsqueeze(0)
    smooth_window = (window_h * window_w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1).to(torch.float32)
    with torch.no_grad(), autocast(device.type, dtype=dtype, enabled=(device.type == 'cuda')):
        for i in range(h_steps):
            for j in range(w_steps):
                y1 = i * stride
                x1 = j * stride
                y2 = min(y1 + max_size, h)
                x2 = min(x1 + max_size, w)
                y1 = max(0, y2 - max_size)
                x1 = max(0, x2 - max_size)
                patch = images[:, :, y1:y2, x1:x2]
                patch_result = model(patch)
                current_window = smooth_window[:, :, :(y2-y1), :(x2-x1)]
                result[:, :, y1:y2, x1:x2] += patch_result.to(images.dtype) * current_window.to(images.dtype)
                weights[:, :, y1:y2, x1:x2] += current_window
    result = torch.where(weights > 1e-6, result / weights.to(images.dtype), result)
    return result

# --- 驗證函數 ---
def validate(generator, val_loader, device, criterion, max_validate_batches=None, crop_size=256):
    """驗證模型性能"""
    generator.eval()
    val_psnr_list = []
    val_loss_list = []
    total_samples = 0
    processed_batches = 0

    # --- 計算驗證批次數 ---
    num_val_batches = len(val_loader)
    if max_validate_batches:
        num_val_batches = min(max_validate_batches, num_val_batches)

    with torch.no_grad():
        val_pbar = tqdm(enumerate(val_loader), total=num_val_batches, desc="Validation", unit="batch", leave=False)
        for i, batch_data in val_pbar:
            if max_validate_batches and i >= max_validate_batches:
                break
            if batch_data is None:
                # print(f"警告：驗證時跳過無效的批次 {i}") # tqdm 會覆蓋
                continue
            images, targets = batch_data
            if images.numel() == 0 or targets.numel() == 0:
                # print(f"警告：驗證時跳過空的批次 {i}") # tqdm 會覆蓋
                continue
            images, targets = images.to(device), targets.to(device)
            batch_size = images.size(0)
            total_samples += batch_size
            processed_batches += 1
            with autocast(device.type, enabled=(device.type == 'cuda')):
                if images.size(2) > crop_size or images.size(3) > crop_size:
                    outputs = process_large_images(generator, images, max_size=crop_size, overlap=crop_size // 4)
                else:
                    outputs = generator(images)
                loss_dict = criterion(outputs, targets)
                val_loss = loss_dict['total']
            val_loss_list.append(val_loss.item())
            outputs_clamped = torch.clamp(outputs, 0.0, 1.0)
            targets_clamped = torch.clamp(targets, 0.0, 1.0)
            batch_psnr = calculate_psnr(outputs_clamped, targets_clamped, data_range=1.0)
            val_psnr_list.append(batch_psnr)

            # --- 更新驗證進度條後綴 ---
            val_pbar.set_postfix({
                'Loss': f"{val_loss.item():.4f}",
                'PSNR': f"{batch_psnr:.2f}"
            })
            # val_pbar.update(1)
        val_pbar.close()

    avg_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
    avg_loss = np.mean(val_loss_list) if val_loss_list else float('inf')
    # print(f"\n  驗證完成 ({processed_batches} 批次, {total_samples} 樣本). 平均 Loss: {avg_loss:.4f}, 平均 PSNR: {avg_psnr:.4f} dB") # 移到 train_model 中打印
    generator.train()
    return avg_psnr, avg_loss

# --- 模型保存 ---
def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, scheduler_g, scheduler_d, scaler, best_psnr, best_g_loss, metadata, is_best_psnr, save_dir, model_name):
    """保存訓練檢查點，包含所有必要狀態"""
    state = {
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict() if scheduler_g else None,
        'scheduler_d_state_dict': scheduler_d.state_dict() if scheduler_d else None,
        'scaler_state_dict': scaler.state_dict(),
        'best_psnr': best_psnr,
        'best_g_loss': best_g_loss,
        'metadata': metadata,
    }
    latest_path = os.path.join(save_dir, f"{model_name}_latest.pth.tar")
    torch.save(state, latest_path)
    # print(f"檢查點已保存至: {latest_path} (Epoch {epoch+1})") # 移到 train_model 中打印

    if is_best_psnr:
        best_psnr_path = os.path.join(save_dir, f"{model_name}_best_psnr.pth.tar")
        torch.save(state, best_psnr_path)
        # print(f"*** 最佳 PSNR 模型已更新並保存至: {best_psnr_path} (PSNR: {best_psnr:.4f} dB) ***") # 移到 train_model 中打印

    if (epoch + 1) % 50 == 0:
        epoch_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth.tar")
        torch.save(state, epoch_path)
        # print(f"定期檢查點已保存至: {epoch_path}") # 移到 train_model 中打印

def save_final_model(generator, metadata, save_dir, model_name, final_epoch):
    """僅保存最終的生成器模型狀態字典和元數據"""
    final_gen_path = os.path.join(save_dir, f"{model_name}_generator_final_epoch_{final_epoch}.pth")
    torch.save(generator.state_dict(), final_gen_path)
    print(f"最終生成器模型已保存至: {final_gen_path}")
    metadata_path = os.path.join(save_dir, f"{model_name}_generator_final_epoch_{final_epoch}_info.json")
    try:
        if 'training_args' in metadata and isinstance(metadata['training_args'], argparse.Namespace):
             metadata['training_args'] = vars(metadata['training_args'])
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4, default=lambda o: '<not serializable>')
        print(f"最終模型元數據已保存至: {metadata_path}")
    except Exception as e:
        print(f"警告：保存最終模型元數據時出錯: {e}")

# --- 核心訓練函數 ---
def train_model(generator, discriminator, train_loader, val_loader, criterion,
                g_optimizer, d_optimizer, scheduler_g, scheduler_d, scaler, num_epochs, device,
                save_dir="./models", log_dir="./logs", model_name="NS-IC-JPEG-Restoration",
                gradient_accumulation_steps=4, checkpoint_interval=25,
                validation_interval=1,
                max_grad_norm=1.0,
                adversarial_weight=0.002,
                args=None):
    """JPEG 修復模型的訓練循環"""
    generator.to(device)
    discriminator.to(device)
    criterion.to(device)
    start_epoch = args.start_epoch if hasattr(args, 'start_epoch') else 0
    best_psnr = args.best_psnr if hasattr(args, 'best_psnr') else 0.0
    best_g_loss = args.best_g_loss if hasattr(args, 'best_g_loss') else float('inf')
    model_metadata = args.metadata if hasattr(args, 'metadata') and args.metadata else {}

    # --- 日誌設定 ---
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"training_log_{model_name}_{timestamp}.csv")
    log_header = "Epoch,G_Loss,D_Loss,G_L1,G_MSE,G_SSIM,G_Edge,G_HF,G_Color,G_Adv,Val_PSNR,Val_Loss,Time_Epoch,Time_Total,LR_G,LR_D\n"
    if not os.path.exists(log_file) or start_epoch == 0:
        with open(log_file, "w", encoding='utf-8') as f:
            f.write(log_header)
    training_start_time = time.time()
    if not model_metadata:
        try: num_rrdb = len(generator.rrdb_blocks); features = generator.conv_first.out_channels
        except AttributeError: num_rrdb, features = 'Unknown', 'Unknown'
        model_metadata = {
            "model_name": model_name,
            "version": "NS-IC-JPEG-Restoration-v9",
            "description": args.model_description if args and args.model_description else "寫實圖像 JPEG 壓縮修復模型 v9",
            "architecture": {"type": "ImageQualityEnhancer", "num_rrdb_blocks": num_rrdb, "features": features},
            "training_args": vars(args) if args else {},
            "training_info": {
                "start_time_initial": time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_time_current_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_size": len(train_loader.dataset) + len(val_loader.dataset) if train_loader and val_loader else 'Unknown',
                "total_epochs_planned": start_epoch + num_epochs,
                "quality_range": f"q{args.min_quality}-q{args.max_quality}" if args else "N/A",
                "loss_function": "JPEGRestorationLoss",
                "adversarial_weight": adversarial_weight,
            },
            "performance": {"best_psnr": best_psnr, "best_g_loss": best_g_loss}
        }
    else:
        model_metadata["training_info"]["start_time_current_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
        model_metadata["training_info"]["total_epochs_planned"] = start_epoch + num_epochs
        model_metadata["training_args"] = vars(args) if args else {}
        model_metadata["training_info"]["adversarial_weight"] = adversarial_weight
        model_metadata["loss_function"] = "JPEGRestorationLoss"

    print(f"將從 Epoch {start_epoch + 1} 開始訓練，共 {num_epochs} 個 Epochs。")
    print(f"使用損失函數: JPEGRestorationLoss")
    print(f"對抗損失權重: {adversarial_weight}")
    print(f"梯度累積步數: {gradient_accumulation_steps}")
    print(f"有效批量大小: {args.batch_size * gradient_accumulation_steps}")

    # --- 訓練循環 ---
    total_epochs_target = start_epoch + num_epochs
    for epoch in range(start_epoch, total_epochs_target):
        generator.train()
        discriminator.train()
        epoch_g_loss_accum = 0.0
        epoch_d_loss_accum = 0.0
        epoch_g_losses_detailed = {'l1': 0.0, 'mse': 0.0, 'ssim': 0.0, 'edge': 0.0, 'hf': 0.0, 'color': 0.0, 'adv': 0.0}
        epoch_start_time = time.time()
        processed_batches_in_epoch = 0
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        num_batches_total = len(train_loader)
        num_optim_steps_per_epoch = num_batches_total // gradient_accumulation_steps
        pbar = tqdm(enumerate(train_loader), total=num_optim_steps_per_epoch, desc=f"Epoch {epoch+1}/{total_epochs_target} Train", unit="step")
        for i, batch_data in pbar:
            if batch_data is None:
                # print(f"警告：訓練時跳過無效的批次 {i}")
                continue
            low_res_imgs, high_res_imgs = batch_data
            if low_res_imgs.numel() == 0 or high_res_imgs.numel() == 0:
                # print(f"警告：訓練時跳過空的批次 {i}")
                continue
            batch_size = low_res_imgs.size(0)
            low_res_imgs = low_res_imgs.to(device)
            high_res_imgs = high_res_imgs.to(device)

            # --- 訓練判別器 ---
            with autocast(device.type, enabled=(device.type == 'cuda')):
                fake_imgs = generator(low_res_imgs).detach()
                real_outputs = discriminator(high_res_imgs)
                fake_outputs = discriminator(fake_imgs)
                d_loss_real = 0
                d_loss_fake = 0
                for real_out, fake_out in zip(real_outputs, fake_outputs):
                    real_labels = torch.ones_like(real_out) * random.uniform(0.9, 1.0)
                    fake_labels = torch.zeros_like(fake_out) * random.uniform(0.0, 0.1)
                    d_loss_real += F.binary_cross_entropy_with_logits(real_out, real_labels)
                    d_loss_fake += F.binary_cross_entropy_with_logits(fake_out, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2.0
                d_loss = d_loss / gradient_accumulation_steps
            scaler.scale(d_loss).backward()

            # --- 訓練生成器 ---
            with autocast(device.type, enabled=(device.type == 'cuda')):
                generated_imgs = generator(low_res_imgs)
                core_loss_dict = criterion(generated_imgs, high_res_imgs)
                core_loss = core_loss_dict['total']
                adv_fake_outputs = discriminator(generated_imgs)
                adv_loss = 0
                for fake_out in adv_fake_outputs:
                    real_labels_for_g = torch.ones_like(fake_out, device=device)
                    adv_loss += F.binary_cross_entropy_with_logits(fake_out, real_labels_for_g)
                g_loss = core_loss + adversarial_weight * adv_loss
                g_loss = g_loss / gradient_accumulation_steps
            scaler.scale(g_loss).backward()

            # --- 梯度累積與優化器步驟 ---
            if (i + 1) % gradient_accumulation_steps == 0:
                processed_batches_in_epoch += 1
                if max_grad_norm > 0:
                    scaler.unscale_(d_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                scaler.step(d_optimizer)
                d_optimizer.zero_grad()
                if max_grad_norm > 0:
                    scaler.unscale_(g_optimizer)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
                scaler.step(g_optimizer)
                g_optimizer.zero_grad()
                scaler.update()
                current_d_loss_val = d_loss.item() * gradient_accumulation_steps
                current_g_loss_val = g_loss.item() * gradient_accumulation_steps
                epoch_d_loss_accum += current_d_loss_val
                epoch_g_loss_accum += current_g_loss_val
                for key in epoch_g_losses_detailed:
                    if key == 'adv':
                        epoch_g_losses_detailed[key] += adv_loss.item()
                    elif key in core_loss_dict:
                        epoch_g_losses_detailed[key] += core_loss_dict[key].item()
                pbar.set_postfix({
                    'G_Loss': f"{current_g_loss_val:.4f}",
                    'D_Loss': f"{current_d_loss_val:.4f}",
                    'LR_G': f"{g_optimizer.param_groups[0]['lr']:.1e}",
                    'LR_D': f"{d_optimizer.param_groups[0]['lr']:.1e}"
                })
                pbar.update(1)
                # if processed_batches_in_epoch % 50 == 0:
                #     print(...) 
        pbar.close() 

        # --- Epoch 結束 ---
        epoch_time = time.time() - epoch_start_time
        total_training_time = time.time() - training_start_time

        num_optim_steps = max(1, processed_batches_in_epoch) 
        avg_g_loss = epoch_g_loss_accum / num_optim_steps
        avg_d_loss = epoch_d_loss_accum / num_optim_steps
        avg_g_losses_detailed = {k: v / num_optim_steps for k, v in epoch_g_losses_detailed.items()}


        # --- 執行驗證 ---
        val_psnr = 0.0
        val_loss = float('inf')
        if (epoch + 1) % validation_interval == 0:
            val_psnr, val_loss = validate(generator, val_loader, device, criterion,
                                          max_validate_batches=args.validate_batches if args.fast_validation else None,
                                          crop_size=args.crop_size)

        # --- 更新學習率 ---
        current_lr_g = g_optimizer.param_groups[0]['lr']
        current_lr_d = d_optimizer.param_groups[0]['lr']
        lr_updated = False
        if isinstance(scheduler_g, ReduceLROnPlateau):
            scheduler_g.step(val_psnr)
            if scheduler_d: scheduler_d.step(val_psnr)
            lr_updated = True
        elif scheduler_g is not None:
            scheduler_g.step()
            if scheduler_d: scheduler_d.step()
            lr_updated = True
        new_lr_g = g_optimizer.param_groups[0]['lr']
        new_lr_d = d_optimizer.param_groups[0]['lr'] if d_optimizer.param_groups else 0
        if lr_updated and (new_lr_g != current_lr_g or new_lr_d != current_lr_d):
            print(f"學習率已更新 -> G: {new_lr_g:.7f}, D: {new_lr_d:.7f}")
        print(f"\nEpoch [{epoch+1}/{total_epochs_target}] Summary | Time: {format_time(epoch_time)} "
              f"| Total Time: {format_time(total_training_time)}")
        print(f"  平均損失 -> G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}")
        loss_detail_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_g_losses_detailed.items()])
        print(f"  G Loss 細項 -> {loss_detail_str}")
        if val_psnr > 0 or val_loss != float('inf'):
            print(f"  驗證結果 -> Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f} dB")
        print(f"  當前學習率 -> G: {new_lr_g:.7f}, D: {new_lr_d:.7f}")
        try:
            with open(log_file, "a", encoding='utf-8') as f:
                log_data = [
                    epoch + 1, avg_g_loss, avg_d_loss,
                    avg_g_losses_detailed['l1'], avg_g_losses_detailed['mse'], avg_g_losses_detailed['ssim'],
                    avg_g_losses_detailed['edge'], avg_g_losses_detailed['hf'], avg_g_losses_detailed['color'],
                    avg_g_losses_detailed['adv'],
                    val_psnr, val_loss,
                    epoch_time, total_training_time, new_lr_g, new_lr_d
                ]
                f.write(",".join(map(str, log_data)) + "\n")
        except IOError as log_err:
             print(f"警告：無法寫入日誌文件 {log_file}: {log_err}")

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
            print(f"*** 新的最佳 PSNR: {best_psnr:.4f} dB ***")
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            model_metadata["performance"]["best_g_loss"] = best_g_loss
            model_metadata["performance"]["best_g_loss_epoch"] = epoch + 1
        save_checkpoint_flag = (epoch + 1) % checkpoint_interval == 0 or epoch == (total_epochs_target - 1)
        if save_checkpoint_flag or is_best_psnr:
            save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer,
                            scheduler_g, scheduler_d, scaler, best_psnr, best_g_loss,
                            model_metadata, is_best_psnr, save_dir, model_name)
            latest_path = os.path.join(save_dir, f"{model_name}_latest.pth.tar")
            print(f"檢查點已保存至: {latest_path} (Epoch {epoch+1})")
            if is_best_psnr:
                best_psnr_path = os.path.join(save_dir, f"{model_name}_best_psnr.pth.tar")
                print(f"*** 最佳 PSNR 模型已更新並保存至: {best_psnr_path} (PSNR: {best_psnr:.4f} dB) ***")
            if (epoch + 1) % 50 == 0 and save_checkpoint_flag: 
                 epoch_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth.tar")
                 print(f"定期檢查點已保存至: {epoch_path}")


        # --- 清理記憶體 ---
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- 訓練結束 ---
    print("\n" + "="*30 + " 訓練結束 " + "="*30)
    save_final_model(generator, model_metadata, save_dir, model_name, total_epochs_target)
    print(f"最終生成器模型已保存。")
    print(f"最佳 PSNR: {best_psnr:.4f} dB (在 Epoch {model_metadata['performance'].get('best_psnr_epoch', 'N/A')})")
    print(f"最佳 G_Loss: {best_g_loss:.4f} (在 Epoch {model_metadata['performance'].get('best_g_loss_epoch', 'N/A')})")
    print(f"訓練日誌文件保存在: {log_file}")
    return generator, discriminator

def collate_fn(batch):
    """
    處理數據集中可能返回的 None 或空張量 (由錯誤處理產生)。
    過濾掉無效樣本對，如果整個批次都無效，則返回 None。
    """
    batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None and x[0].numel() > 0 and x[1].numel() > 0, batch))
    if not batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"錯誤：在 collate_fn 中堆疊批次時出錯: {e}")
        return None

# --- 主執行區塊 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門影像品質增強訓練器 v9.0 (寫實圖像 JPEG 壓縮修復)')
    # --- 數據與路徑 ---
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01_R_J', help='訓練數據目錄 (包含寫實圖像的低/高質量對, 會遞迴掃描)')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存路徑')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日誌保存路徑')
    parser.add_argument('--model_name', type=str, default='NS-IC-JPEG-Restoration-v9', help='模型名稱')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點路徑 (.pth.tar)')
    parser.add_argument('--model_description', type=str, default='寫實圖像 JPEG 壓縮修復模型 v9.1', help='模型描述')
    # --- 訓練參數 ---
    parser.add_argument('--num_epochs', type=int, default=1000, help='總訓練輪數 (如果 resume，則為目標總輪數)')
    parser.add_argument('--batch_size', type=int, default=6, help='訓練批量 (根據 VRAM 調整)')
    parser.add_argument('--crop_size', type=int, default=256, help='訓練圖像裁剪大小')
    parser.add_argument('--quality_range', type=str, default='10-90', help='訓練使用的低品質範圍 (min-max)')
    # --- 優化器與學習率 ---
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='生成器初始學習率')
    parser.add_argument('--d_lr_factor', type=float, default=0.5, help='判別器學習率相對於生成器的係數')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='優化器類型')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='AdamW 的權重衰減')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'step'], help='學習率調度器類型')
    parser.add_argument('--plateau_patience', type=int, default=10, help='ReduceLROnPlateau 的耐心值 (epochs)')
    parser.add_argument('--plateau_factor', type=float, default=0.5, help='ReduceLROnPlateau 的學習率衰減因子')
    parser.add_argument('--cosine_t_max', type=int, default=500, help='CosineAnnealingLR 的 T_max (建議設為總 epochs)')
    parser.add_argument('--step_size', type=int, default=100, help='StepLR 的步長 (epochs)')
    parser.add_argument('--step_gamma', type=float, default=0.5, help='StepLR 的衰減因子')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='最小學習率下限')
    # --- 損失權重 ---
    parser.add_argument('--adversarial_weight', type=float, default=0.002, help='對抗損失權重')
    # --- 訓練技巧 ---
    parser.add_argument('--grad_accum', type=int, default=4, help='梯度累積步數')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪範數上限 (0 表示不裁剪)')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作線程數')
    parser.add_argument('--cache_images', action='store_true', default=False, help='將圖片快取到記憶體以加速')
    parser.add_argument('--no_cache_images', action='store_false', dest='cache_images')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='啟用 pin_memory 加速數據轉移')
    parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory')
    parser.add_argument('--augment', action='store_true', default=True, help='啟用數據增強')
    parser.add_argument('--no_augment', action='store_false', dest='augment')
    # --- 驗證與保存 ---
    parser.add_argument('--validation_interval', type=int, default=1, help='多少個 epoch 執行一次驗證')
    parser.add_argument('--fast_validation', action='store_true', default=False, help='啟用快速驗證 (只驗證部分批次)')
    parser.add_argument('--validate_batches', type=int, default=50, help='快速驗證使用的批次數')
    parser.add_argument('--checkpoint_interval', type=int, default=25, help='檢查點保存間隔 (epoch)')
    args = parser.parse_args()

    # --- 解析品質範圍 ---
    try:
        min_q_str, max_q_str = args.quality_range.split('-')
        args.min_quality = int(min_q_str)
        args.max_quality = int(max_q_str)
        if not (0 < args.min_quality <= args.max_quality <= 100):
            raise ValueError("品質範圍必須在 1-100 之間，且最小值不大於最大值。")
        print(f"設定訓練品質範圍: q{args.min_quality} - q{args.max_quality}")
    except ValueError as e:
        parser.error(f"無效的 quality_range 格式或值: {args.quality_range}. 請使用 'min-max' 格式，例如 '10-90'. 錯誤: {e}")

    # --- 設定隨機種子 ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- 創建目錄 ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    try:
        cpu_count = os.cpu_count()
        num_workers = min(args.num_workers, cpu_count // 2 if cpu_count and cpu_count > 1 else 1)
        num_workers = max(0, num_workers)
    except NotImplementedError:
        num_workers = args.num_workers
    print(f"使用 {num_workers} 個資料載入工作線程")
    base_transform = transforms.Compose([transforms.ToTensor()])
    try:
        dataset = QualityDataset(
            args.data_dir,
            transform=base_transform,
            crop_size=args.crop_size,
            augment=args.augment,
            cache_images=args.cache_images,
            min_quality=args.min_quality,
            max_quality=args.max_quality
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"錯誤：無法初始化資料集: {e}")
        sys.exit(1)
    except Exception as ds_err:
         print(f"錯誤：初始化資料集時發生未知錯誤: {ds_err}")
         sys.exit(1)

    # --- 劃分訓練集和驗證集 ---
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("錯誤：資料集為空。")
        sys.exit(1)
    val_split = 0.1
    val_size = max(1, int(val_split * dataset_size))
    train_size = dataset_size - val_size
    print(f"數據集總大小: {dataset_size}, 訓練集: {train_size}, 驗證集: {val_size}")
    if train_size <= 0 or val_size <= 0:
        print(f"錯誤：數據集大小不足以劃分訓練集 ({train_size}) 和驗證集 ({val_size})。")
        sys.exit(1)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    # --- 數據載入器 ---
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=max(1, num_workers // 2),
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=max(1, num_workers // 2) > 0,
        collate_fn=collate_fn
    )

    # --- 初始化模型 ---
    generator = ImageQualityEnhancer(num_rrdb_blocks=16, features=64)
    discriminator = MultiScaleDiscriminator(num_scales=3, input_channels=3)

    # --- 創建損失函數 ---
    criterion = JPEGRestorationLoss(device=device)

    # --- 創建優化器 ---
    optimizer_choice = optim.AdamW if args.optimizer == 'AdamW' else optim.Adam
    g_optimizer = optimizer_choice(
        generator.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
        weight_decay=args.weight_decay if args.optimizer == 'AdamW' else 0
    )
    d_optimizer = optimizer_choice(
        discriminator.parameters(), lr=args.learning_rate * args.d_lr_factor, betas=(0.9, 0.999),
        weight_decay=args.weight_decay if args.optimizer == 'AdamW' else 0
    )

    # --- 初始化 GradScaler ---
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # --- 恢復訓練狀態 ---
    args.start_epoch = 0
    args.best_psnr = 0.0
    args.best_g_loss = float('inf')
    args.metadata = None
    scheduler_g_state = None
    scheduler_d_state = None

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"正在從檢查點恢復: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                generator.to(device)
                discriminator.to(device)
                generator.load_state_dict(checkpoint['generator_state_dict'])
                g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                # d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                for state in g_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                for state in d_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("GradScaler 狀態已恢復。")
                else:
                    print("警告：檢查點中未找到 GradScaler 狀態。")
                args.start_epoch = checkpoint.get('epoch', 0)
                args.best_psnr = checkpoint.get('best_psnr', 0.0)
                args.best_g_loss = checkpoint.get('best_g_loss', float('inf'))
                args.metadata = checkpoint.get('metadata', None)
                scheduler_g_state = checkpoint.get('scheduler_g_state_dict', None)
                scheduler_d_state = checkpoint.get('scheduler_d_state_dict', None)
                print(f"成功從 Epoch {args.start_epoch} 恢復。") 
                print(f"  恢復的最佳 PSNR: {args.best_psnr:.4f} dB")
                print(f"  恢復的最佳 G Loss: {args.best_g_loss:.4f}")
            except KeyError as e:
                print(f"錯誤：檢查點文件缺少鍵 '{e}'。可能是不兼容的檢查點。將從頭開始訓練。")
                args.start_epoch = 0
                args.best_psnr = 0.0
                args.best_g_loss = float('inf')
                args.metadata = None
                scheduler_g_state = None
                scheduler_d_state = None
            except Exception as e:
                print(f"錯誤：加載檢查點時出錯: {e}")
                args.start_epoch = 0
        else:
            print(f"警告：找不到檢查點文件: {args.resume}。將從頭開始訓練。")

        # --- 創建學習率調度器 ---
        total_target_epochs = args.num_epochs
        remaining_epochs_for_scheduler = max(1, total_target_epochs - args.start_epoch)
        if args.scheduler == 'cosine':
            scheduler_g = CosineAnnealingLR(g_optimizer, T_max=remaining_epochs_for_scheduler, eta_min=args.min_lr, last_epoch=args.start_epoch-1 if args.resume else -1)
            scheduler_d = CosineAnnealingLR(d_optimizer, T_max=remaining_epochs_for_scheduler, eta_min=args.min_lr * args.d_lr_factor, last_epoch=args.start_epoch-1 if args.resume else -1)
        elif args.scheduler == 'step':
            scheduler_g = StepLR(g_optimizer, step_size=args.step_size, gamma=args.step_gamma, last_epoch=args.start_epoch-1 if args.resume else -1)
            scheduler_d = StepLR(d_optimizer, step_size=args.step_size, gamma=args.step_gamma, last_epoch=args.start_epoch-1 if args.resume else -1)
        elif args.scheduler == 'plateau':
            scheduler_g = ReduceLROnPlateau(g_optimizer, mode='max', factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr, verbose=True)
            scheduler_d = ReduceLROnPlateau(d_optimizer, mode='max', factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr * args.d_lr_factor, verbose=True)
            
            if scheduler_g and scheduler_g_state:
                scheduler_g.load_state_dict(scheduler_g_state)
                # 如果是恢復訓練且已知的 best_psnr > 0，但 scheduler 的 best 仍為 -inf，則手動校準
                if args.resume and args.best_psnr > 0 and hasattr(scheduler_g, 'best') and scheduler_g.best == float('-inf'):
                    print(f"警告：G Scheduler (Plateau) 的 best 為 -inf，但恢復的 best_psnr 為 {args.best_psnr:.4f}。手動設定 scheduler.best。")
                    scheduler_g.best = args.best_psnr 
                print(f"已加載 G Scheduler (Plateau) 狀態: best={scheduler_g.best if hasattr(scheduler_g, 'best') else 'N/A':.4f}, num_bad_epochs={scheduler_g.num_bad_epochs if hasattr(scheduler_g, 'num_bad_epochs') else 'N/A'}")
            if scheduler_d and scheduler_d_state:
                scheduler_d.load_state_dict(scheduler_d_state)
                if args.resume and args.best_psnr > 0 and hasattr(scheduler_d, 'best') and scheduler_d.best == float('-inf'): 
                     print(f"警告：D Scheduler (Plateau) 的 best 為 -inf，但恢復的 best_psnr 為 {args.best_psnr:.4f}。手動設定 scheduler.best。")
                     scheduler_d.best = args.best_psnr 
                print(f"已加載 D Scheduler (Plateau) 狀態: best={scheduler_d.best if hasattr(scheduler_d, 'best') else 'N/A':.4f}, num_bad_epochs={scheduler_d.num_bad_epochs if hasattr(scheduler_d, 'num_bad_epochs') else 'N/A'}")
        remaining_epochs = total_target_epochs - args.start_epoch
        if remaining_epochs <= 0:
            print(f"起始 epoch ({args.start_epoch}) 已達到或超過目標 epoch ({total_target_epochs})。無需繼續訓練。")
            sys.exit(0)
        else:
            print(f"將訓練 {remaining_epochs} 個 epochs (從 {args.start_epoch+1} 到 {total_target_epochs})")

    # --- 開始訓練 ---
    print("\n" + "="*30 + " 開始訓練 " + "="*30)
    try:
        train_model(
            generator, discriminator, train_loader, val_loader, criterion,
            g_optimizer, d_optimizer, scheduler_g, scheduler_d, scaler,
            num_epochs=remaining_epochs,
            device=device, save_dir=args.save_dir, log_dir=args.log_dir,
            model_name=args.model_name,
            gradient_accumulation_steps=args.grad_accum,
            checkpoint_interval=args.checkpoint_interval,
            validation_interval=args.validation_interval,
            max_grad_norm=args.max_grad_norm,
            adversarial_weight=args.adversarial_weight,
            args=args
        )
    except KeyboardInterrupt:
        print("\n訓練被手動中斷。")
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {e}")
        traceback.print_exc()
    finally:
        print("清理資源...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("程序結束。")