import os
import gc
import argparse
import time
import random
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

try:
    from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator

# 設置CUDNN優化
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# --- 損失函數 ---
class EnhancedPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1,1,3,3).repeat(3,1,1,1))
        self.register_buffer('sobel_y', sobel_y.view(1,1,3,3).repeat(3,1,1,1))
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32)
        self.register_buffer('laplacian', lap.view(1,1,3,3).repeat(3,1,1,1))

    def _multi_scale(self, x, levels=3):
        xs = [x]
        for _ in range(levels-1):
            x = self.avg_pool(x)
            xs.append(x)
        return xs

    def _sobel_edges(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def _laplacian_edges(self, x):
        return F.conv2d(x, self.laplacian, padding=1, groups=3)

    def _local_variance(self, x, kernel_size=5):
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)
        mean_sq = F.avg_pool2d(x**2, kernel_size, stride=1, padding=kernel_size//2)
        return mean_sq - mean**2

    def _local_contrast(self, x, kernel_size=7):
        max_pool = F.max_pool2d(x, kernel_size, stride=1, padding=kernel_size//2)
        min_pool = -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size//2)
        return max_pool - min_pool

    def _fft_energy(self, x):
        B,C,H,W = x.shape
        x = x.to(torch.float32)
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft = torch.fft.fftshift(x_fft)
        mag = torch.abs(x_fft)
        mag = mag / (mag.amax(dim=(-2,-1), keepdim=True) + 1e-6)
        return mag

    def _color_stats(self, x):
        mean = x.mean(dim=[2,3])
        std = x.std(dim=[2,3])
        x_perm = x.permute(0,2,3,1)
        mx, _ = x_perm.max(-1)
        mn, _ = x_perm.min(-1)
        sat = torch.where(mx==0, torch.zeros_like(mx), (mx-mn)/(mx+1e-6))
        sat_mean = sat.mean(dim=[1,2])
        return mean, std, sat_mean

    def forward(self, x, target):
        xs = self._multi_scale(x)
        ts = self._multi_scale(target)
        loss_mse = sum(self.mse_loss(a, b) for a, b in zip(xs, ts)) / len(xs)
        loss_l1 = sum(self.l1_loss(a, b) for a, b in zip(xs, ts)) / len(xs)
        sobel_x = self._sobel_edges(x)
        sobel_t = self._sobel_edges(target)
        loss_sobel = self.l1_loss(sobel_x, sobel_t)
        lap_x = self._laplacian_edges(x)
        lap_t = self._laplacian_edges(target)
        loss_lap = self.l1_loss(lap_x, lap_t)
        var_x = self._local_variance(x)
        var_t = self._local_variance(target)
        loss_var = self.l1_loss(var_x, var_t)
        cont_x = self._local_contrast(x)
        cont_t = self._local_contrast(target)
        loss_cont = self.l1_loss(cont_x, cont_t)
        fft_x = self._fft_energy(x)
        fft_t = self._fft_energy(target)
        loss_fft = self.l1_loss(fft_x, fft_t)
        mean_x, std_x, sat_x = self._color_stats(x)
        mean_t, std_t, sat_t = self._color_stats(target)
        loss_color = self.l1_loss(mean_x, mean_t) + self.l1_loss(std_x, std_t) + self.l1_loss(sat_x, sat_t)
        loss_vec = torch.stack([
            loss_mse, loss_l1, loss_sobel, loss_lap, loss_var, loss_cont, loss_fft, loss_color
        ])
        inv = 1.0 / (loss_vec.detach() + 1e-6).sqrt()
        weights = inv / inv.sum()
        total_loss = (loss_vec * weights).sum()
        return total_loss

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, self.channel))

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        pad = self.window_size // 2 
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        mu1 = F.conv2d(F.pad(img1, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel)
        mu2 = F.conv2d(F.pad(img2, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(F.pad(img1*img1, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(F.pad(img2*img2, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel) - mu2_sq
        sigma12 = F.conv2d(F.pad(img1*img2, (pad, pad, pad, pad), mode='reflect'), window, padding=0, groups=channel) - mu1_mu2
        sigma1_sq = F.relu(sigma1_sq) + 1e-8
        sigma2_sq = F.relu(sigma2_sq) + 1e-8
        C1 = (0.01 * 1)**2 
        C2 = (0.03 * 1)**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        loss = 1.0 - ssim_map
        if self.size_average:
            return loss.mean()
        else:
            return loss.mean(1).mean(1).mean(1)

# --- 資料集類別 ---
class QualityDataset(Dataset):
    def __init__(self, image_dir, transform=None, crop_size=256, augment=True, cache_images=True,
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
        print(f"初始化資料集，品質範圍: q{min_quality} - q{max_quality}")
        all_image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                           if f.lower().endswith(('jpg', 'jpeg', 'png'))]
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
            for i, base_name in enumerate(self.valid_groups):
                qualities = self.image_groups[base_name]
                try:
                    high_quality_path = qualities['q100']
                    if high_quality_path not in self.image_cache:
                         self.image_cache[high_quality_path] = Image.open(high_quality_path).convert("RGB")
                    for q_name, low_quality_path in qualities.items():
                        if q_name != 'q100':
                             q_val = int(q_name[1:])
                             if self.min_quality <= q_val <= self.max_quality:
                                 if low_quality_path not in self.image_cache:
                                     self.image_cache[low_quality_path] = Image.open(low_quality_path).convert("RGB")
                    loaded_count +=1
                    if loaded_count % 200 == 0:
                        print(f"已載入 {loaded_count}/{len(self.valid_groups)} 組圖像")
                except Exception as e:
                    print(f"預載入圖像時出錯: {base_name}, {e}. 將在需要時載入。")
                    if high_quality_path in self.image_cache: del self.image_cache[high_quality_path]
                    for q_name, low_quality_path in qualities.items():
                         if low_quality_path in self.image_cache: del self.image_cache[low_quality_path]
                    continue 
            print(f"圖像預載入完成！實際載入 {len(self.image_cache)} 張圖像。")

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        base_name = self.valid_groups[idx]
        qualities = self.image_groups[base_name]
        low_quality_options = []
        for q_str, path in qualities.items():
            if q_str != 'q100':
                try:
                    q_val = int(q_str[1:])
                    if self.min_quality <= q_val <= self.max_quality:
                        low_quality_options.append((q_str, path))
                except ValueError:
                    continue
        if not low_quality_options:
            print(f"警告：組 {base_name} 找不到範圍內的低品質圖像，返回空張量。")
            return torch.zeros(3, self.crop_size, self.crop_size), torch.zeros(3, self.crop_size, self.crop_size)
        weights = []
        options_paths = []
        for q_name, path in low_quality_options:
            q_num = int(q_name[1:])
            weight = 1.0 / (q_num + 1e-6) 
            weights.append(weight)
            options_paths.append(path)
        sum_weight = sum(weights)
        if sum_weight > 0:
            weights = [w / sum_weight for w in weights]
        else:
             weights = [1.0 / len(options_paths)] * len(options_paths)
        low_quality_path = random.choices(options_paths, weights=weights, k=1)[0]
        high_quality_path = qualities['q100']
        try:
            if self.cache_images and low_quality_path in self.image_cache:
                low_quality_image = self.image_cache[low_quality_path].copy()
            else:
                low_quality_image = Image.open(low_quality_path).convert("RGB")
                if self.cache_images:
                    self.image_cache[low_quality_path] = low_quality_image.copy()
            if self.cache_images and high_quality_path in self.image_cache:
                high_quality_image = self.image_cache[high_quality_path].copy()
            else:
                high_quality_image = Image.open(high_quality_path).convert("RGB")
                if self.cache_images:
                    self.image_cache[high_quality_path] = high_quality_image.copy()
            width, height = low_quality_image.size
            if high_quality_image.size != (width, height):
                 high_quality_image = high_quality_image.resize((width, height), Image.LANCZOS)

            # --- 數據增強 ---
            crop_w = min(self.crop_size, width)
            crop_h = min(self.crop_size, height)
            if width >= crop_w and height >= crop_h:
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_quality_image, output_size=(crop_h, crop_w)
                )
                low_quality_image = transforms.functional.crop(low_quality_image, i, j, h, w)
                high_quality_image = transforms.functional.crop(high_quality_image, i, j, h, w)
            else:
                low_quality_image = transforms.functional.resize(low_quality_image, (crop_h, crop_w), interpolation=transforms.InterpolationMode.BILINEAR)
                high_quality_image = transforms.functional.resize(high_quality_image, (crop_h, crop_w), interpolation=transforms.InterpolationMode.LANCZOS)
            if self.augment:
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.hflip(low_quality_image)
                    high_quality_image = transforms.functional.hflip(high_quality_image)
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.vflip(low_quality_image)
                    high_quality_image = transforms.functional.vflip(high_quality_image)
                if random.random() > 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    if angle != 0:
                        low_quality_image = transforms.functional.rotate(low_quality_image, angle)
                        high_quality_image = transforms.functional.rotate(high_quality_image, angle)
                if random.random() > 0.9:
                    color_jitter = transforms.ColorJitter(
                        brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01
                    )
                    low_quality_image = color_jitter(low_quality_image)
                if random.random() > 0.95 and self.epoch < 50:
                    low_quality_tensor = transforms.ToTensor()(low_quality_image)
                    noise = torch.randn_like(low_quality_tensor) * random.uniform(0.001, 0.008)
                    low_quality_tensor = torch.clamp(low_quality_tensor + noise, 0, 1)
                    low_quality_image = transforms.ToPILImage()(low_quality_tensor)
            if self.transform:
                low_quality_image = self.transform(low_quality_image)
                high_quality_image = self.transform(high_quality_image)
            if low_quality_image.shape[1:] != high_quality_image.shape[1:]:
                 print(f"警告：最終張量尺寸不匹配 {low_quality_image.shape} vs {high_quality_image.shape} for {base_name}. 跳過此樣本。")
                 target_shape = (3, self.crop_size, self.crop_size)
                 return torch.zeros(target_shape), torch.zeros(target_shape)
            return low_quality_image, high_quality_image
        except Exception as e:
            print(f"處理圖像時發生嚴重錯誤: {base_name} (Low: {low_quality_path}, High: {high_quality_path}), Error: {e}")
            target_shape = (3, self.crop_size, self.crop_size)
            return torch.zeros(target_shape), torch.zeros(target_shape)

# --- 工具函數 ---
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def calculate_psnr(img1, img2, data_range=1.0):
    """計算峰值信噪比(PSNR)，假設輸入範圍為 [0, data_range]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()

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
    h_steps = math.ceil(max(0, h - overlap) / stride) + 1
    w_steps = math.ceil(max(0, w - overlap) / stride) + 1
    window_h = torch.hann_window(max_size, periodic=False).unsqueeze(1)
    window_w = torch.hann_window(max_size, periodic=False).unsqueeze(0)
    # 更平滑的 cosine window
    # t = torch.linspace(0, math.pi, max_size)
    # window_h = (0.5 - 0.5 * torch.cos(t)).unsqueeze(1)
    # window_w = (0.5 - 0.5 * torch.cos(t)).unsqueeze(0)
    smooth_window = (window_h * window_w).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, 1).to(device, dtype=torch.float32)
    for i in range(h_steps):
        for j in range(w_steps):
            h_start = min(i * stride, h - max_size)
            w_start = min(j * stride, w - max_size)
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
    """驗證模型性能，處理大尺寸圖像"""
    generator.eval()
    val_psnr_list = []
    val_mse_list = []
    validation_images = []
    total_samples = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if max_validate_batches is not None and i >= max_validate_batches:
                break
            images, targets = images.to(device), targets.to(device)
            # 如果驗證集圖像尺寸固定且不大於模型訓練尺寸，可以直接推理
            # fake_images = process_large_images(generator, images)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                fake_images = generator(images)
            fake_images = torch.clamp(fake_images, 0.0, 1.0)
            batch_size = images.size(0)
            for j in range(batch_size):
                psnr = calculate_psnr(fake_images[j], targets[j], data_range=1.0)
                if not math.isinf(psnr) and not math.isnan(psnr):
                    val_psnr_list.append(psnr)
                    val_mse_list.append(F.mse_loss(fake_images[j], targets[j]).item())
                if return_images and i < 1 and j < 4:
                    validation_images.append((
                        transforms.ToPILImage()(images[j].cpu()),
                        transforms.ToPILImage()(fake_images[j].cpu()),
                        transforms.ToPILImage()(targets[j].cpu())
                    ))
            total_samples += batch_size
            if max_validate_batches is None and i % 50 == 0:
                 print(f"\r驗證進度: Batch {i+1}/{len(val_loader)}", end="")
    avg_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
    avg_mse = np.mean(val_mse_list) if val_mse_list else float('inf')
    print(f"\n驗證完成. 平均 PSNR: {avg_psnr:.4f} dB, 平均 MSE: {avg_mse:.6f}")
    if return_images:
        return avg_psnr, validation_images
    return avg_psnr

# --- 模型保存 ---
def save_model_with_metadata(model, path, metadata=None):
    """保存模型狀態字典並附加元數據"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存至: {path}")
    if metadata:
        metadata_path = os.path.splitext(path)[0] + "_info.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                cleaned_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        cleaned_metadata[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor):
                                cleaned_metadata[key][sub_key] = sub_value.item()
                            else:
                                cleaned_metadata[key][sub_key] = sub_value
                    elif isinstance(value, torch.Tensor):
                         cleaned_metadata[key] = value.item()
                    else:
                        cleaned_metadata[key] = value
                json.dump(cleaned_metadata, f, ensure_ascii=False, indent=4)
            print(f"元數據已保存至: {metadata_path}")
        except Exception as e:
            print(f"保存元數據時出錯: {e}")

# --- 核心訓練函數 ---
def train_model(generator, discriminator, train_loader, val_loader, criterion_dict,
                g_optimizer, d_optimizer, scheduler_g, scheduler_d, num_epochs, device,
                save_dir="./models", log_dir="./logs", model_name="NS-IC",
                gradient_accumulation_steps=4, checkpoint_interval=25,
                validation_interval=1,
                fast_validation=False,
                max_grad_norm=1.0,
                args=None):
    """改進的訓練函數，專注於訓練穩定性和PSNR最大化"""
    generator.to(device)
    discriminator.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_g_loss = float('inf')
    best_psnr = 0.0
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_file, "w") as f:
        f.write("Epoch,G_Loss,D_Loss,MSE_Loss,L1_Loss,Perceptual_Loss,SSIM_Loss,Adv_Loss,PSNR,Time_Epoch,Time_Total,LR_G,LR_D\n")
    training_start_time = time.time()
    mse_weight = 1.0              # PSNR，提高像素精度
    l1_weight = 0.8               # 輔助 MSE，有助於銳度
    perceptual_weight = 2.0       # 感知損失權重
    ssim_weight = 1.2             # SSIM，強調結構
    adversarial_weight = 0.001   # 對抗權重
    
    # 獲取損失函數實例
    perceptual_criterion = criterion_dict['perceptual']
    ssim_criterion = criterion_dict['ssim']

    # 訓練狀態變量
    start_epoch = args.start_epoch if hasattr(args, 'start_epoch') else 0
    total_epochs_to_run = num_epochs
    
    # 模型元數據
    model_metadata = {
        "model_name": model_name,
        "version": "NS-IC-v6",
        "architecture": {
            "type": "ImageQualityEnhancer",
            "num_rrdb_blocks": len(generator.rrdb_blocks) if hasattr(generator, 'rrdb_blocks') else 'Unknown',
            "features": generator.conv_first.out_channels if hasattr(generator, 'conv_first') else 'Unknown'
        },
        "training_args": vars(args) if args else {},
        "training_info": {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(train_loader.dataset) + len(val_loader.dataset),
            "total_epochs_planned": start_epoch + total_epochs_to_run,
            "quality_range": f"q{args.min_quality}-q{args.max_quality}" if args else "N/A",
            "loss_weights": {
                "mse": mse_weight,
                "l1": l1_weight,
                "perceptual": perceptual_weight,
                "ssim": ssim_weight,
                "adversarial": adversarial_weight
            }
        },
        "performance": {
            "best_psnr": best_psnr,
            "best_g_loss": best_g_loss,
        }
    }

    # --- 訓練循環 ---
    for epoch in range(start_epoch, start_epoch + total_epochs_to_run):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_start_time = time.time()
        batch_count = 0
        if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'epoch'):
             train_loader.dataset.dataset.epoch = epoch
        elif hasattr(train_loader.dataset, 'epoch'):
             train_loader.dataset.epoch = epoch

        # --- 批次訓練 ---
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            if torch.isnan(images).any() or torch.isinf(images).any() or \
               torch.isnan(targets).any() or torch.isinf(targets).any():
                print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 輸入數據包含 NaN/Inf 值。跳過此批次。")
                continue

            # ===== 訓練判別器 =====
            if batch_count % gradient_accumulation_steps == 0:
                d_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                fake_images = generator(images)
                if torch.isnan(fake_images).any() or torch.isinf(fake_images).any():
                    print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 生成器輸出包含 NaN/Inf 值。跳過此批次。")
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
                print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 判別器損失為 NaN/Inf。跳過判別器更新。")
            else:
                scaler.scale(d_loss_scaled).backward()
                epoch_d_loss += d_loss.item()

            # ===== 訓練生成器 =====
            if batch_count % gradient_accumulation_steps == 0:
                g_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                fake_outputs_for_g = discriminator(fake_images)
                mse_loss = F.mse_loss(fake_images, targets)
                l1_loss = F.l1_loss(fake_images, targets)
                perceptual_loss = perceptual_criterion(fake_images, targets)
                ssim_loss = ssim_criterion(fake_images, targets)
                adversarial_g_loss = 0
                num_outputs_g = len(fake_outputs_for_g)
                for scale_idx in range(num_outputs_g):
                    adversarial_g_loss += torch.mean((fake_outputs_for_g[scale_idx] - 1.0) ** 2)
                adversarial_g_loss /= num_outputs_g

                # --- 組合生成器損失 ---
                g_loss = (mse_weight * mse_loss +
                          l1_weight * l1_loss +
                          perceptual_weight * perceptual_loss +
                          ssim_weight * ssim_loss +
                          adversarial_weight * adversarial_g_loss)
                g_loss_scaled = g_loss / gradient_accumulation_steps

            # 檢查生成器損失
            if torch.isnan(g_loss_scaled).any() or torch.isinf(g_loss_scaled).any():
                 print(f"\n警告：Epoch {epoch+1}, Batch {i+1}: 生成器損失為 NaN/Inf。跳過生成器更新。")
            else:
                scaler.scale(g_loss_scaled).backward()
                epoch_g_loss += g_loss.item()
                epoch_mse_loss += mse_loss.item()
                epoch_l1_loss += l1_loss.item()
                epoch_perceptual_loss += perceptual_loss.item()
                epoch_ssim_loss += ssim_loss.item()
                epoch_adv_loss += adversarial_g_loss.item()
            batch_count += 1

            # --- 梯度更新 ---
            if batch_count % gradient_accumulation_steps == 0 or (i == len(train_loader) - 1):
                if not (torch.isnan(d_loss_scaled).any() or torch.isinf(d_loss_scaled).any()):
                    scaler.unscale_(d_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
                    scaler.step(d_optimizer)
                if not (torch.isnan(g_loss_scaled).any() or torch.isinf(g_loss_scaled).any()):
                    scaler.unscale_(g_optimizer)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
                    scaler.step(g_optimizer)
                scaler.update()

            # --- 顯示訓練進度 ---
            if i % 10 == 0 or i == len(train_loader) - 1: 
                progress = (i + 1) / len(train_loader)
                percentage = progress * 100
                elapsed_time = time.time() - epoch_start_time
                eta = (elapsed_time / progress - elapsed_time) if progress > 0 else 0
                fill_length = int(30 * progress) 
                space_length = 30 - fill_length
                current_g_loss = g_loss.item() if not torch.isnan(g_loss).any() else 0
                current_d_loss = d_loss.item() if not torch.isnan(d_loss).any() else 0
                print(f"\rEpoch [{epoch+1}/{start_epoch + total_epochs_to_run}] "
                      f"{percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| "
                      f"[{format_time(elapsed_time)}<{format_time(eta)}] "
                      f"G:{current_g_loss:.4f} D:{current_d_loss:.4f}", end="")

        # --- Epoch 結束 ---
        epoch_time = time.time() - epoch_start_time
        total_training_time = time.time() - training_start_time

        # 計算平均損失
        num_batches = len(train_loader)
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        avg_mse_loss = epoch_mse_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        avg_perceptual_loss = epoch_perceptual_loss / num_batches
        avg_ssim_loss = epoch_ssim_loss / num_batches
        avg_adv_loss = epoch_adv_loss / num_batches

        # --- 執行驗證 ---
        val_psnr = 0.0
        if (epoch + 1) % validation_interval == 0:
            print(f"\n--- 驗證輪數 {epoch+1} ---")
            validate_batches = args.validate_batches if fast_validation else None
            val_psnr, validation_images = validate(generator, val_loader, device,
                                                 max_validate_batches=validate_batches,
                                                 return_images=True,
                                                 crop_size=args.crop_size)
            print(f"--- 驗證完成 ---")

        # --- 更新學習率 ---
        current_lr_g = g_optimizer.param_groups[0]['lr']
        current_lr_d = d_optimizer.param_groups[0]['lr']
        if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if val_psnr > 0:
                 scheduler_g.step(val_psnr)
                 scheduler_d.step(val_psnr)
            else:
                 print("警告：無有效 PSNR，跳過 ReduceLROnPlateau step")
        else:
            scheduler_g.step()
            scheduler_d.step()
        new_lr_g = g_optimizer.param_groups[0]['lr']
        new_lr_d = d_optimizer.param_groups[0]['lr']
        if new_lr_g != current_lr_g:
            print(f"\nGenerator LR updated to {new_lr_g:.7f}")
        if new_lr_d != current_lr_d:
            print(f"Discriminator LR updated to {new_lr_d:.7f}")

        # --- 打印 Epoch 摘要 ---
        print(f"\nEpoch [{epoch+1}/{start_epoch + total_epochs_to_run}] Summary | Time: {format_time(epoch_time)} "
              f"| Total Time: {format_time(total_training_time)}")
        print(f"  平均損失 -> G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}")
        print(f"  G Loss 細項 -> MSE: {avg_mse_loss:.4f}, L1: {avg_l1_loss:.4f}, Perc: {avg_perceptual_loss:.4f}, SSIM: {avg_ssim_loss:.4f}, Adv: {avg_adv_loss:.4f}")
        if val_psnr > 0:
            print(f"  驗證 PSNR: {val_psnr:.4f} dB")
        print(f"  當前學習率 -> G: {new_lr_g:.7f}, D: {new_lr_d:.7f}")

        # --- 記錄訓練日誌 ---
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},"
                    f"{avg_mse_loss:.6f},{avg_l1_loss:.6f},{avg_perceptual_loss:.6f},{avg_ssim_loss:.6f},{avg_adv_loss:.6f},"
                    f"{val_psnr:.6f},{epoch_time:.2f},{total_training_time:.2f},{new_lr_g:.8f},{new_lr_d:.8f}\n")

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

        is_best_loss = False
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            is_best_loss = True
            model_metadata["performance"]["best_g_loss"] = best_g_loss
            model_metadata["performance"]["best_g_loss_epoch"] = epoch + 1
            # 保存最佳損失模型
            # best_loss_path = os.path.join(save_dir, f"{model_name}_best_loss.pth")
            # save_model_with_metadata(generator, best_loss_path, model_metadata)
            # print(f"--- 新的最佳 G_Loss 模型已儲存 (Epoch {epoch+1}, G_Loss: {avg_g_loss:.4f}) ---")

        # 定期保存檢查點
        save_checkpoint = (epoch + 1) % checkpoint_interval == 0 or epoch == (start_epoch + total_epochs_to_run - 1)
        if save_checkpoint or is_best_psnr: 
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

            # 保存最新檢查點
            latest_checkpoint_path = os.path.join(save_dir, f"{model_name}_latest.pth")
            torch.save(checkpoint, latest_checkpoint_path)
            print(f"Latest checkpoint saved to: {latest_checkpoint_path}")

            if save_checkpoint:
                checkpoint_path = os.path.join(save_dir, f"{model_name}_checkpoint_epoch_{epoch+1}.pth")
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved for Epoch {epoch+1} to: {checkpoint_path}")

        # --- 清理記憶體 ---
        if (epoch + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- 訓練結束 ---
    final_path = os.path.join(save_dir, f"{model_name}_final_epoch_{start_epoch + total_epochs_to_run}.pth")
    model_metadata["training_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_metadata["training_info"]["total_time_formatted"] = format_time(time.time() - training_start_time)
    model_metadata["performance"]["final_psnr"] = val_psnr
    model_metadata["performance"]["final_g_loss"] = avg_g_loss
    save_model_with_metadata(generator, final_path, model_metadata)
    print("訓練完成！最終模型已保存。")
    print(f"最佳 PSNR: {best_psnr:.4f} dB (Epoch {model_metadata['performance'].get('best_psnr_epoch', 'N/A')})")
    print(f"日誌文件保存在: {log_file}")
    return generator, discriminator

# --- 主執行區塊 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門影像品質增強訓練器 v8.0')
    # 數據與路徑
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01_R', help='訓練集路徑')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存路徑')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日誌保存路徑')
    parser.add_argument('--model_name', type=str, default='NS-IC-v8', help='模型名稱')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點路徑 (.pth)')

    # 訓練參數
    parser.add_argument('--num_epochs', type=int, default=1000, help='總訓練輪數 (如果 resume，則為剩餘輪數)')
    parser.add_argument('--batch_size', type=int, default=6, help='訓練批量 (根據 VRAM 調整)')
    parser.add_argument('--crop_size', type=int, default=256, help='訓練圖像裁剪大小')
    parser.add_argument('--quality_range', type=str, default='10-90', help='訓練使用的品質範圍 (例如 "10-40", "40-70", "10-90")')

    # 優化器與學習率
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='生成器初始學習率')
    parser.add_argument('--d_lr_factor', type=float, default=0.5, help='判別器學習率相對於生成器的係數')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'], help='優化器類型')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='AdamW 的權重衰減')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['cosine', 'plateau', 'step'], help='學習率調度器類型')
    parser.add_argument('--plateau_patience', type=int, default=15, help='ReduceLROnPlateau 的耐心值 (epochs)')
    parser.add_argument('--plateau_factor', type=float, default=0.2, help='ReduceLROnPlateau 的學習率衰減因子')
    parser.add_argument('--cosine_t_max', type=int, default=1000, help='CosineAnnealingLR 的 T_max (通常設為總 epochs)')
    parser.add_argument('--step_size', type=int, default=100, help='StepLR 的步長 (epochs)')
    parser.add_argument('--step_gamma', type=float, default=0.5, help='StepLR 的衰減因子')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='最小學習率下限')

    # 訓練技巧
    parser.add_argument('--grad_accum', type=int, default=4, help='梯度累積步數 (有效批量 = batch_size * grad_accum)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪範數上限')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作線程數')
    parser.add_argument('--cache_images', action='store_true', default=False, help='將圖片快取到記憶體以加速')
    parser.add_argument('--no_cache_images', action='store_false', dest='cache_images', help='禁用圖片快取')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='啟用 pin_memory 加速數據轉移')
    parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory', help='禁用 pin_memory')

    # 驗證與保存
    parser.add_argument('--validation_interval', type=int, default=1, help='多少個 epoch 執行一次驗證')
    parser.add_argument('--fast_validation', action='store_true', default=False, help='啟用快速驗證 (只驗證部分批次)')
    parser.add_argument('--validate_batches', type=int, default=50, help='快速驗證使用的批次數')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='檢查點保存間隔 (epoch)')

    # 模型描述
    parser.add_argument('--model_description', type=str, default='精度優化版', help='模型描述')
    args = parser.parse_args()

    # --- 解析品質範圍 ---
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

    # --- 優化工作線程數 ---
    num_workers = min(args.num_workers, os.cpu_count() // 2 if os.cpu_count() else 4) 
    print(f"使用 {num_workers} 個資料載入工作線程")

    # --- 定義數據轉換 ---
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        dataset = QualityDataset(
            args.data_dir,
            transform=transform,
            crop_size=args.crop_size,
            cache_images=args.cache_images,
            min_quality=args.min_quality,
            max_quality=args.max_quality
        )
    except ValueError as e:
        print(f"錯誤：無法初始化資料集: {e}")
        exit(1)
    except FileNotFoundError:
        print(f"錯誤：找不到資料目錄: {args.data_dir}")
        exit(1)
    dataset_size = len(dataset)
    if dataset_size == 0:
        print("錯誤：資料集為空，請檢查 data_dir 和 quality_range 設置。")
        exit(1)
    val_split = 0.1
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
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )

    # --- 初始化模型 ---
    generator = ImageQualityEnhancer(num_rrdb_blocks=16, features=64)
    discriminator = MultiScaleDiscriminator(num_scales=3, input_channels=3)
    criterion_dict = {
        'perceptual': EnhancedPerceptualLoss().to(device),
        'ssim': SSIMLoss().to(device)
    }

    # --- 創建優化器 ---
    if args.optimizer == 'AdamW':
        g_optimizer = torch.optim.AdamW(
            generator.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.learning_rate * args.d_lr_factor,
            betas=(0.9, 0.999), 
            weight_decay=args.weight_decay
        )
    else: # Adam
         g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999)
        )
         d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.learning_rate * args.d_lr_factor,
            betas=(0.9, 0.999)
        )

    # --- 創建學習率調度器 ---
    if args.scheduler == 'cosine':
        g_t_max = args.cosine_t_max if args.resume is None else args.cosine_t_max - args.start_epoch
        scheduler_g = CosineAnnealingLR(
            g_optimizer,
            T_max=max(1, g_t_max),
            eta_min=args.min_lr
        )
        scheduler_d = CosineAnnealingLR(
            d_optimizer,
            T_max=max(1, g_t_max),
            eta_min=args.min_lr * args.d_lr_factor
        )
    elif args.scheduler == 'step':
         scheduler_g = torch.optim.lr_scheduler.StepLR(
             g_optimizer,
             step_size=args.step_size,
             gamma=args.step_gamma
         )
         scheduler_d = torch.optim.lr_scheduler.StepLR(
             d_optimizer,
             step_size=args.step_size,
             gamma=args.step_gamma
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

    # --- 恢復訓練 ---
    start_epoch = 0
    best_psnr = 0.0
    best_g_loss = float('inf')
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"載入檢查點: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                if 'generator_state_dict' in checkpoint:
                    generator.load_state_dict(checkpoint['generator_state_dict'])
                else:
                    generator.load_state_dict(checkpoint)
                    print("警告：檢查點只包含生成器狀態。")
                if 'discriminator_state_dict' in checkpoint:
                    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                loaded_optimizer_type = checkpoint.get('args', {}).get('optimizer', 'AdamW')
                if 'g_optimizer_state_dict' in checkpoint and args.optimizer == loaded_optimizer_type:
                    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
                else:
                     print(f"警告：無法載入生成器優化器狀態 (檢查點優化器: {loaded_optimizer_type}, 當前: {args.optimizer})。")
                if 'd_optimizer_state_dict' in checkpoint and args.optimizer == loaded_optimizer_type:
                    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
                else:
                     print(f"警告：無法載入判別器優化器狀態 (檢查點優化器: {loaded_optimizer_type}, 當前: {args.optimizer})。")
                loaded_scheduler_type = checkpoint.get('args', {}).get('scheduler', 'plateau')
                if 'g_scheduler_state_dict' in checkpoint and args.scheduler == loaded_scheduler_type:
                    try:
                        scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
                    except Exception as e:
                        print(f"警告：無法載入生成器調度器狀態: {e}。將使用新的調度器。")
                else:
                     print(f"警告：無法載入生成器調度器狀態 (檢查點調度器: {loaded_scheduler_type}, 當前: {args.scheduler})。")
                if 'd_scheduler_state_dict' in checkpoint and args.scheduler == loaded_scheduler_type:
                     try:
                        scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
                     except Exception as e:
                        print(f"警告：無法載入判別器調度器狀態: {e}。將使用新的調度器。")
                else:
                     print(f"警告：無法載入判別器調度器狀態 (檢查點調度器: {loaded_scheduler_type}, 當前: {args.scheduler})。")
                if 'scaler_state_dict' in checkpoint and device.type == 'cuda':
                    scaler = torch.cuda.amp.GradScaler(enabled=True)
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("GradScaler 狀態已載入。")
                start_epoch = checkpoint.get('epoch', 0)
                best_psnr = checkpoint.get('best_psnr', 0.0)
                best_g_loss = checkpoint.get('best_g_loss', float('inf'))
                if start_epoch >= args.num_epochs:
                     print(f"檢查點 epoch ({start_epoch}) 已達到或超過目標 epoch ({args.num_epochs})。無需繼續訓練。")
                     exit(0)
                print(f"成功從 Epoch {start_epoch} 恢復訓練。")
                print(f"  恢復時最佳 PSNR: {best_psnr:.4f} dB")
                print(f"  恢復時最佳 G Loss: {best_g_loss:.4f}")
                args.start_epoch = start_epoch

            except Exception as e:
                print(f"錯誤：無法載入檢查點 '{args.resume}': {e}")
                print("將從頭開始訓練。")
                start_epoch = 0
                best_psnr = 0.0
                best_g_loss = float('inf')
        else:
            print(f"警告：找不到檢查點文件 '{args.resume}'。將從頭開始訓練。")
            start_epoch = 0
            best_psnr = 0.0
            best_g_loss = float('inf')

    # 計算剩餘需要訓練的 epoch 數
    remaining_epochs = args.num_epochs - start_epoch
    if remaining_epochs <= 0:
         print(f"起始 epoch ({start_epoch}) 已達到或超過目標 epoch ({args.num_epochs})。無需繼續訓練。")
         exit(0)
    else:
         print(f"將訓練 {remaining_epochs} 個 epochs (從 {start_epoch+1} 到 {args.num_epochs})")

    # --- 開始訓練 ---
    print("\n" + "="*30 + " 開始訓練 " + "="*30)
    generator, discriminator = train_model(
        generator,
        discriminator,
        train_loader,
        val_loader,
        criterion_dict,
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
    print("="*30 + " 訓練結束 " + "="*30 + "\n")
