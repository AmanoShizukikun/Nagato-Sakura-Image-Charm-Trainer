import os
import gc
import argparse
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split

from src.IQE import RRDB, AttentionBlock, ImageQualityEnhancer, MultiScaleDiscriminator

torch.backends.cudnn.benchmark = True


# 感知損失函數
class EnhancedPerceptualLoss(nn.Module):
    def __init__(self):
        super(EnhancedPerceptualLoss, self).__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.register_buffer('high_freq_kernel', self._create_high_freq_kernel())
        
    def _create_high_freq_kernel(self):
        kernel_size = 5
        sigma = 1.0
        # 建立高斯濾波器
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        x = x.view(1, -1).repeat(kernel_size, 1)
        y = x.transpose(0, 1)
        gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        # 拉普拉斯算子（近似高頻濾波）
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        laplacian = F.pad(laplacian, (1, 1, 1, 1))
        # 轉為適合卷積的形狀 [out_channels, in_channels/groups, H, W]
        kernel = laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel
    
    def extract_high_freq(self, x):
        # 使用拉普拉斯算子提取高頻成分
        return F.conv2d(x, self.high_freq_kernel, padding=2, groups=3)
        
    def forward(self, x, target):
        mse_loss = self.mse_loss(x, target)
        # 多尺度感知損失 - 降低低分辨率的影響
        x_down1 = self.avg_pool(x)
        target_down1 = self.avg_pool(target)
        mse_down1 = self.mse_loss(x_down1, target_down1)
        # 二級下採樣
        x_down2 = self.avg_pool(x_down1)
        target_down2 = self.avg_pool(target_down1)
        mse_down2 = self.mse_loss(x_down2, target_down2)
        # 邊緣感知損失 - 更強調輪廓
        x_grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        x_grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        #  邊緣損失 - 使用MSE計算邊緣差異
        edge_loss = (self.mse_loss(x_grad_x, target_grad_x) + 
                    self.mse_loss(x_grad_y, target_grad_y))
        # 使用拉普拉斯核心提取高頻成分，替代FFT
        x_high_freq = self.extract_high_freq(x)
        target_high_freq = self.extract_high_freq(target)
        high_freq_loss = self.mse_loss(x_high_freq, target_high_freq)
        # 顏色一致性損失
        color_loss = self.l1_loss(x.mean(dim=[2, 3]), target.mean(dim=[2, 3]))
        # 組合各類損失，權重調整為增強PSNR
        total_loss = (mse_loss * 1.5 +           # 提高MSE權重
                        0.5 * mse_down1 +           # 降低低分辨率MSE權重
                        0.3 * mse_down2 +           # 更低的二級下採樣權重
                        1.2 * edge_loss +           # 保持邊緣一致性很重要
                        0.6 * high_freq_loss +      # 增強高頻細節重建
                        0.1 * color_loss)           # 降低顏色損失權重
        return total_loss

# SSIM損失
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size)
        
    def _create_window(self, window_size):
        _1D_window = torch.Tensor([1.0 / window_size] * window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        if img1.is_cuda:
            self.window = self.window.to(img1.device)
            
        window = self.window
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

# 資料集類別，支援質量級別採樣策略
class QualityDataset(Dataset):
    def __init__(self, image_dir, transform=None, crop_size=256, augment=True, cache_images=True):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                          if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform
        self.crop_size = crop_size
        self.augment = augment
        self.cache_images = cache_images
        self.image_cache = {} 
        self.image_groups = {}
        for path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(path))[0].rsplit('_', 1)[0]
            quality = os.path.splitext(os.path.basename(path))[0].rsplit('_', 1)[1]
            if base_name not in self.image_groups:
                self.image_groups[base_name] = {}
            self.image_groups[base_name][quality] = path
        self.valid_groups = []
        # 擴充有效的品質組合範圍 - 支援更多品質級別
        for base_name, qualities in self.image_groups.items():
            if 'q100' in qualities and any(f'q{q}' in qualities for q in range(10, 41, 10)): # 訓練範圍
                self.valid_groups.append(base_name)
        
        if self.cache_images and len(self.valid_groups) > 0:
            print(f"預載入 {len(self.valid_groups)} 組圖像到記憶體...")
            for i, base_name in enumerate(self.valid_groups):
                if i % 100 == 0:
                    print(f"已載入 {i}/{len(self.valid_groups)} 組圖像")
                qualities = self.image_groups[base_name]
                high_quality_path = qualities['q100']
                self.image_cache[high_quality_path] = Image.open(high_quality_path).convert("RGB")
                # 預載入所有品質版本以提升訓練效率
                for q_name in qualities.keys():
                    if q_name != 'q100':
                        low_quality_path = qualities[q_name]
                        self.image_cache[low_quality_path] = Image.open(low_quality_path).convert("RGB")

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        base_name = self.valid_groups[idx]
        qualities = self.image_groups[base_name]
        low_quality_options = [q for q in qualities.keys() if q != 'q100']
        epoch_portion = min(1.0, getattr(self, 'epoch', 0) / 100)  # 0~1之間
        weights = []
        for q in low_quality_options:
            q_num = int(q[1:])
            # 訓練初期給予低品質更高權重，隨訓練進行逐漸平衡
            if epoch_portion < 0.5:
                weight = 1.0 / (q_num ** 0.5)  # 較陡的權重曲線
            else:
                weight = 1.0 / (q_num ** 0.3)  # 較平緩的權重曲線
            weights.append(weight)
            
        # 歸一化權重
        sum_weight = sum(weights)
        weights = [w/sum_weight for w in weights]
        low_quality = random.choices(low_quality_options, weights=weights, k=1)[0]
        low_quality_path = qualities[low_quality]
        high_quality_path = qualities['q100']
        
        try:
            if self.cache_images and low_quality_path in self.image_cache and high_quality_path in self.image_cache:
                low_quality_image = self.image_cache[low_quality_path].copy()
                high_quality_image = self.image_cache[high_quality_path].copy()
            else:
                low_quality_image = Image.open(low_quality_path).convert("RGB")
                high_quality_image = Image.open(high_quality_path).convert("RGB")
            width, height = low_quality_image.size
            high_quality_image = high_quality_image.resize((width, height), Image.LANCZOS)
            # 確保裁剪大小不超過圖像尺寸
            crop_size = min(min(self.crop_size, width), min(self.crop_size, height))
            if self.augment:
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_quality_image, 
                    output_size=(crop_size, crop_size)
                )
                low_quality_image = transforms.functional.crop(low_quality_image, i, j, h, w)
                high_quality_image = transforms.functional.crop(high_quality_image, i, j, h, w)
                # 增強數據增強
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.hflip(low_quality_image)
                    high_quality_image = transforms.functional.hflip(high_quality_image)
                if random.random() > 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    low_quality_image = transforms.functional.rotate(low_quality_image, angle)
                    high_quality_image = transforms.functional.rotate(high_quality_image, angle)
                if random.random() > 0.7:
                    color_jitter = transforms.ColorJitter(
                        brightness=0.05, 
                        contrast=0.05, 
                        saturation=0.05, 
                        hue=0.02
                    )
                    low_quality_image = color_jitter(low_quality_image)
                if random.random() > 0.85:
                    low_quality_tensor = transforms.ToTensor()(low_quality_image)
                    noise = torch.randn_like(low_quality_tensor) * random.uniform(0.005, 0.02)
                    low_quality_tensor = torch.clamp(low_quality_tensor + noise, 0, 1)
                    low_quality_image = transforms.ToPILImage()(low_quality_tensor)
            if self.transform:
                low_quality_image = self.transform(low_quality_image)
                high_quality_image = self.transform(high_quality_image)
            return low_quality_image, high_quality_image
            
        except Exception as e:
            print(f"處理圖像時出錯: {base_name}, {e}")
            return torch.zeros(3, self.crop_size, self.crop_size), torch.zeros(3, self.crop_size, self.crop_size)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def process_large_images(model, images, max_size=512, overlap=64):
    """處理大尺寸圖像，使用分塊處理和無縫拼接"""
    b, c, h, w = images.shape
    result = torch.zeros_like(images)
    if h <= max_size and w <= max_size:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            return model(images)
    h_blocks = max(1, (h - overlap) // (max_size - overlap))
    w_blocks = max(1, (w - overlap) // (max_size - overlap))
    h_size = min(max_size, h)
    w_size = min(max_size, w)
    weights = torch.zeros_like(images)
    for i in range(h_blocks):
        for j in range(w_blocks):
            h_start = min(i * (max_size - overlap), h - h_size)
            w_start = min(j * (max_size - overlap), w - w_size)
            block = images[:, :, h_start:h_start+h_size, w_start:w_start+w_size]
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(block)
            weight = torch.ones_like(output)
            if i > 0:  # 頂部有重疊
                for h_idx in range(overlap):
                    weight_val = (h_idx / overlap) ** 2  # 使用二次函數使過渡更平滑
                    weight[:, :, h_idx:h_idx+1, :] *= weight_val    
            if i < h_blocks - 1:  # 底部有重疊
                for h_idx in range(overlap):
                    weight_val = (1 - h_idx / overlap) ** 2
                    weight[:, :, h_size-overlap+h_idx:h_size-overlap+h_idx+1, :] *= weight_val
            if j > 0:  # 左側有重疊
                for w_idx in range(overlap):
                    weight_val = (w_idx / overlap) ** 2
                    weight[:, :, :, w_idx:w_idx+1] *= weight_val  
            if j < w_blocks - 1:  # 右側有重疊
                for w_idx in range(overlap):
                    weight_val = (1 - w_idx / overlap) ** 2
                    weight[:, :, :, w_size-overlap+w_idx:w_size-overlap+w_idx+1] *= weight_val
            result[:, :, h_start:h_start+h_size, w_start:w_start+w_size] += output * weight
            weights[:, :, h_start:h_start+h_size, w_start:w_start+w_size] += weight
    
    # 除以權重以獲得最終結果（避免除零）
    mask = weights > 0
    result[mask] = result[mask] / weights[mask]
    
    return result

def validate(generator, val_loader, device, max_validate_batches=None, return_images=False):
    """驗證模型性能，處理大尺寸圖像"""
    generator.eval()
    val_psnr = 0.0
    val_mse = 0.0
    total_samples = 0
    validation_images = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if max_validate_batches is not None and i >= max_validate_batches:
                break
                
            images, targets = images.to(device), targets.to(device)
            
            # 處理大尺寸圖像
            fake_images = process_large_images(generator, images)
            
            batch_size = images.size(0)
            for j in range(batch_size):
                # 計算PSNR
                this_psnr = calculate_psnr(fake_images[j], targets[j])
                val_psnr += this_psnr
                val_mse += F.mse_loss(fake_images[j], targets[j]).item()
                
                # 保存部分圖像用於可視化
                if return_images and i < 2 and j < 2:  # 只保存前兩個批次中各兩張圖片
                    validation_images.append((images[j].cpu(), fake_images[j].cpu(), targets[j].cpu()))
                    
            total_samples += batch_size
    
    avg_psnr = val_psnr / total_samples
    avg_mse = val_mse / total_samples
    
    if return_images:
        return avg_psnr, validation_images
    return avg_psnr

def save_model_with_metadata(model, path, metadata=None):
    """保存模型並附加元數據"""
    torch.save(model.state_dict(), path)
    if metadata:
        metadata_path = os.path.splitext(path)[0] + "_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

def train_model(generator, discriminator, train_loader, val_loader, criterion_dict, 
                g_optimizer, d_optimizer, scheduler_g, scheduler_d, num_epochs, device, 
                save_dir="./models", log_dir="./logs", model_name="NS_ICE", 
                gradient_accumulation_steps=4, checkpoint_interval=100,
                fast_validation=False):
    """改進的訓練函數"""
    generator.to(device)
    discriminator.to(device)
    scaler = torch.amp.GradScaler()
    best_g_loss = float('inf')
    best_psnr = 0.0
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,G_Loss,D_Loss,PSNR,Time,Learning_Rate\n")
    training_start_time = time.time()
    
    # 更新損失權重配置 - 大幅增強PSNR導向的權重
    mse_weight = 1.8              # 顯著提高MSE權重，直接增強PSNR
    perceptual_weight = 0.4       # 降低感知損失，更關注像素重建而非視覺效果
    ssim_weight = 0.7             # 提高SSIM權重，增強結構保存能力
    adversarial_weight = 0.0005   # 極大降低對抗性損失，專注於重建精度
    
    # 紀錄耐心值用於動態調整學習率
    patience = 0
    best_val_psnr = 0
    patience_limit = 5
    
    # 更新品質資料集的epoch參數，用於動態採樣策略
    for dataset in [train_loader.dataset, val_loader.dataset]:
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'epoch'):
            dataset.dataset.epoch = 0
    
    # 模型元數據
    model_metadata = {
        "version": "3.0.0",
        "architecture": {
            "type": "ImageQualityEnhancer",
            "num_rrdb_blocks": len(generator.rrdb_blocks),
            "features": 64
        },
        "training_info": {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(train_loader.dataset) + len(val_loader.dataset),
            "epochs": num_epochs,
            "parameters": {
                "mse_weight": mse_weight,
                "perceptual_weight": perceptual_weight,
                "ssim_weight": ssim_weight,
                "adversarial_weight": adversarial_weight
            }
        },
        "performance": {
            "best_psnr": 0.0,
            "best_g_loss": float('inf'),
        }
    }
    
    # 確保驗證一致性
    validate_batches = 40
    
    # 存取損失函數
    perceptual_criterion = criterion_dict['perceptual']
    ssim_criterion = criterion_dict['ssim']
    
    # 衰減係數 - 用於動態調整損失權重
    decay_factor = 0.9999
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_start_time = time.time()
        batch_count = 0
        
        # 更新資料集epoch參數
        for dataset in [train_loader.dataset, val_loader.dataset]:
            if hasattr(dataset, 'dataset'):
                if hasattr(dataset.dataset, 'epoch'):
                    dataset.dataset.epoch = epoch
            elif hasattr(dataset, 'epoch'):
                dataset.epoch = epoch
                
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # ===== 訓練判別器 =====
            if batch_count % gradient_accumulation_steps == 0:
                d_optimizer.zero_grad(set_to_none=True)
                
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                fake_images = generator(images)
                real_outputs = discriminator(targets)
                fake_outputs = discriminator(fake_images.detach())
                
                # 改進的WGAN風格判別器損失
                d_loss_real = 0
                d_loss_fake = 0
                
                for real_output, fake_output in zip(real_outputs, fake_outputs):
                    d_loss_real += torch.mean((1 - real_output) ** 2)
                    d_loss_fake += torch.mean(fake_output ** 2)
                    
                d_loss = 0.5 * (d_loss_real + d_loss_fake) / len(real_outputs)
                d_loss = d_loss / gradient_accumulation_steps 
                
            scaler.scale(d_loss).backward()
            
            # ===== 訓練生成器 =====
            if batch_count % gradient_accumulation_steps == 0:
                g_optimizer.zero_grad(set_to_none=True)
                
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                fake_outputs = discriminator(fake_images)
                
                # 像素重建損失(MSE) - 核心損失
                mse_loss = F.mse_loss(fake_images, targets)
                
                # 增強感知損失 - 保留細節和紋理
                perceptual_loss = perceptual_criterion(fake_images, targets)
                
                # SSIM損失(結構相似性) - 保留結構信息
                ssim_loss = ssim_criterion(fake_images, targets)
                
                # L1損失(絕對誤差) - 補充MSE，更好處理邊緣
                l1_loss = F.l1_loss(fake_images, targets)
                
                # 對抗性損失 - 微量，讓圖像看起來更真實
                adversarial_loss = 0
                for fake_output in fake_outputs:
                    adversarial_loss += torch.mean((1 - fake_output) ** 2)
                adversarial_loss /= len(fake_outputs)
                
                # 晚期損失權重調整 - 隨訓練進行調整損失權重
                epoch_progress = min(1.0, epoch / 100)
                # 後期稍微增加MSE權重，強化PSNR
                dynamic_mse_weight = mse_weight * (1 + 0.2 * epoch_progress)
                
                # 組合損失
                g_loss = (dynamic_mse_weight * mse_loss + 
                          0.3 * l1_loss +  # 保持L1損失相對穩定
                          perceptual_weight * perceptual_loss + 
                          ssim_weight * ssim_loss +
                          adversarial_weight * adversarial_loss)
                          
                g_loss = g_loss / gradient_accumulation_steps 
                
            scaler.scale(g_loss).backward()
            
            batch_count += 1
            epoch_g_loss += g_loss.item() * gradient_accumulation_steps
            epoch_d_loss += d_loss.item() * gradient_accumulation_steps
            
            # 梯度更新
            if batch_count % gradient_accumulation_steps == 0 or (i == len(train_loader) - 1):
                # 梯度裁剪增強穩定性
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                
                scaler.step(d_optimizer)
                scaler.step(g_optimizer)
                scaler.update()
            
            # 顯示訓練進度
            if i % 5 == 0 or i == len(train_loader) - 1:
                progress = (i + 1) / len(train_loader)
                percentage = progress * 100
                elapsed_time = time.time() - epoch_start_time
                eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
                fill_length = int(50 * progress)
                space_length = 50 - fill_length
                
                print(f"\rEpoch [{epoch+1}/{num_epochs}] "
                      f"Progress: {percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| "
                      f"[{format_time(elapsed_time)}<{format_time(eta)}] "
                      f"G: {g_loss.item() * gradient_accumulation_steps:.4f}, D: {d_loss.item() * gradient_accumulation_steps:.4f} ", end="")
        
        # 更新學習率
        scheduler_g.step()
        scheduler_d.step()
        current_lr = scheduler_g.get_last_lr()[0]
        
        # 計算平均損失
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        # 執行驗證
        if epoch % 5 == 0:
            val_psnr, validation_images = validate(generator, val_loader, device, 
                                                 max_validate_batches=None, 
                                                 return_images=True)
        else:
            val_psnr = validate(generator, val_loader, device, 
                               max_validate_batches=validate_batches)
        
        total_training_time = time.time() - training_start_time
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}, "
              f"PSNR: {val_psnr:.2f} dB, LR: {current_lr:.7f} [{format_time(total_training_time)} elapsed]")
        
        # 記錄訓練日誌
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},{val_psnr:.6f},{total_training_time:.2f},{current_lr:.7f}\n")
        
        # 更新模型元數據
        model_metadata["training_info"]["last_epoch"] = epoch + 1
        model_metadata["training_info"]["total_time_seconds"] = total_training_time
        model_metadata["performance"]["current_psnr"] = val_psnr
        model_metadata["performance"]["current_g_loss"] = avg_g_loss
        model_metadata["performance"]["current_d_loss"] = avg_d_loss
        
        # 檢查PSNR是否有改善
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience = 0  # 重設耐心值
            model_metadata["performance"]["best_psnr"] = best_psnr
            model_metadata["performance"]["best_psnr_epoch"] = epoch + 1
            best_psnr_path = os.path.join(save_dir, f"{model_name}_best_psnr.pth")
            save_model_with_metadata(generator, best_psnr_path, model_metadata)
            print(f"最佳PSNR模型已保存於 Epoch {epoch+1}，PSNR: {val_psnr:.2f} dB")
        else:
            patience += 1
            
        # 檢查損失是否有改善
        is_best_loss = False
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            is_best_loss = True
            model_metadata["performance"]["best_g_loss"] = best_g_loss
            model_metadata["performance"]["best_g_loss_epoch"] = epoch + 1
            best_loss_path = os.path.join(save_dir, f"{model_name}_best_loss.pth")
            save_model_with_metadata(generator, best_loss_path, model_metadata)
            print(f"最佳損失模型已保存於 Epoch {epoch+1}，G Loss: {avg_g_loss:.4f}")
            
        # 定期保存檢查點
        save_checkpoint = (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1
        if is_best_loss or save_checkpoint:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_scheduler_state_dict': scheduler_g.state_dict(),
                'd_scheduler_state_dict': scheduler_d.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
                'psnr': val_psnr,
                'best_g_loss': best_g_loss,
                'best_psnr': best_psnr,
                'metadata': model_metadata
            }
            
            if save_checkpoint:
                checkpoint_path = os.path.join(save_dir, f"{model_name}_checkpoint_epoch_{epoch+1}.pth")
                torch.save(checkpoint, checkpoint_path)
                print(f"檢查點已保存至 Epoch {epoch+1}")
                
            torch.save(checkpoint, os.path.join(save_dir, f"{model_name}_latest.pth")) 
            
        # 動態調整學習率 - 如果PSNR連續多個epoch沒有改善
        if patience >= patience_limit:
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
            print(f"PSNR {patience}個epoch沒有改善，學習率降低為 {g_optimizer.param_groups[0]['lr']:.7f}")
            patience = 0  # 重設耐心值
            
        # 每5個epoch清理一次記憶體
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
    # 訓練結束，保存最終模型
    final_path = os.path.join(save_dir, f"{model_name}_final.pth")
    model_metadata["training_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_metadata["training_info"]["total_epochs"] = num_epochs
    model_metadata["training_info"]["total_time"] = format_time(time.time() - training_start_time)
    model_metadata["performance"]["final_psnr"] = val_psnr
    model_metadata["performance"]["final_g_loss"] = avg_g_loss
    save_model_with_metadata(generator, final_path, model_metadata)
    print("訓練完成！")
    return generator, discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門影像品質增強訓練器 v3.0')
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01', help='訓練集路徑')
    parser.add_argument('--num_epochs', type=int, default=10000, help='訓練步數')
    parser.add_argument('--batch_size', type=int, default=6, help='訓練批量')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='學習率')
    parser.add_argument('--crop_size', type=int, default=256, help='訓練裁剪大小')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存路徑')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日誌保存路徑')
    parser.add_argument('--model_name', type=str, default='NS-IC-Kyouka-LQ-10K', help='模型名稱')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點路徑')
    parser.add_argument('--grad_accum', type=int, default=4, help='梯度累積步數')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='檢查點保存間隔(epoch)')
    parser.add_argument('--cache_images', action='store_true', help='是否將圖片快取到記憶體')
    parser.add_argument('--fast_validation', action='store_true', default=True, help='啟用快速驗證')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作線程數')
    parser.add_argument('--model_description', type=str, default='', help='模型描述')
    parser.add_argument('--d_lr_factor', type=float, default=0.5, help='判別器學習率係數')
    parser.add_argument('--validate_batches', type=int, default=40, help='驗證批次數')
    parser.add_argument('--max_lr', type=float, default=4e-4, help='最大學習率')
    parser.add_argument('--min_lr', type=float, default=5e-7, help='最小學習率')
    parser.add_argument('--mixup', action='store_true', help='啟用mixup數據增強')
    args = parser.parse_args()
    
    # 設定隨機種子以確保實驗可重複性
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # 創建保存目錄
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 優化工作線程數
    num_workers = min(args.num_workers, os.cpu_count() or 4)
    
    # 定義數據轉換
    transform = transforms.Compose([transforms.ToTensor()])
    
    # 載入數據集
    dataset = QualityDataset(
        args.data_dir, 
        transform=transform, 
        crop_size=args.crop_size,
        cache_images=args.cache_images
    )
    
    # 劃分訓練集和驗證集
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    print(f"數據集總大小: {dataset_size}, 訓練集: {train_size}, 驗證集: {val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 設置每個數據集的epoch屬性，用於動態採樣策略
    train_dataset.dataset.epoch = 0
    val_dataset.dataset.epoch = 0
    
    # 數據載入器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 初始化模型
    generator = ImageQualityEnhancer(num_rrdb_blocks=16)
    discriminator = MultiScaleDiscriminator(num_scales=3)
    
    # 創建各類損失函數
    criterion_dict = {
        'perceptual': EnhancedPerceptualLoss().to(device),
        'ssim': SSIMLoss().to(device)
    }
    
    # 使用AdamW優化器，提供更好的權重衰減
    g_optimizer = torch.optim.AdamW(
        generator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.999),
        weight_decay=1e-5  # 適當的權重衰減以避免過擬合
    )
    
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), 
        lr=args.learning_rate * args.d_lr_factor, 
        betas=(0.9, 0.95),  # 低β2有助於判別器更新
        weight_decay=1e-5
    )
    
    # 使用CosineAnnealingWarmRestarts學習率調度，實現週期性的學習率重置
    scheduler_g = CosineAnnealingWarmRestarts(
        g_optimizer, 
        T_0=20,                  # 初始週期長度
        T_mult=2,                # 每個週期後將週期長度乘以此值
        eta_min=args.min_lr      # 最小學習率
    )
    
    scheduler_d = CosineAnnealingWarmRestarts(
        d_optimizer, 
        T_0=20, 
        T_mult=2, 
        eta_min=args.min_lr * args.d_lr_factor
    )
    
    # 恢復訓練檢查點（如果提供）
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"載入檢查點: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
            scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
            best_g_loss = checkpoint.get('best_g_loss', float('inf'))
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"繼續訓練自 epoch {start_epoch}, 目前最佳 PSNR: {best_psnr:.2f}dB")
            
            # 更新數據集epoch
            train_dataset.dataset.epoch = start_epoch
            val_dataset.dataset.epoch = start_epoch
    
    # 訓練模型
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
        args.num_epochs - start_epoch, 
        device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_interval=args.checkpoint_interval,
        fast_validation=args.fast_validation
    )
    
    # 保存模型資訊
    model_info = {
        "name": args.model_name,
        "description": args.model_description or f"{args.model_name} 圖像增強模型",
        "version": "3.0.0",
        "author": "Nagato-Sakura-Image-Charm",
        "date_created": time.strftime("%Y-%m-%d"),
        "framework": "PyTorch",
        "type": "ImageEnhancement",
        "input_format": "RGB",
        "output_format": "RGB",
        "model_architecture": {
            "type": "ImageQualityEnhancer",
            "num_rrdb_blocks": 16,
            "features": 64
        }
    }
    
    model_info_path = os.path.join(args.save_dir, f"{args.model_name}_info.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)