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
from sklearn.model_selection import train_test_split


from src.IQE import ImageQualityEnhancer, MultiScaleDiscriminator

# 設置CUDNN優化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True  # 提高訓練穩定性

# 感知損失函數 - 改進版本，更重視細節和上下文關係
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
        
        # 多尺度感知損失 - 更好地捕捉不同層級的特徵
        x_down1 = self.avg_pool(x)
        target_down1 = self.avg_pool(target)
        mse_down1 = self.mse_loss(x_down1, target_down1)
        
        # 二級下採樣 - 捕捉更大範圍的上下文
        x_down2 = self.avg_pool(x_down1)
        target_down2 = self.avg_pool(target_down1)
        mse_down2 = self.mse_loss(x_down2, target_down2)
        
        # 邊緣感知損失 - 提升輪廓清晰度
        x_grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        x_grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        edge_loss = (self.mse_loss(x_grad_x, target_grad_x) + 
                     self.mse_loss(x_grad_y, target_grad_y))
        
        # 高頻細節損失 - 增強細節表現
        x_high_freq = self.extract_high_freq(x)
        target_high_freq = self.extract_high_freq(target)
        high_freq_loss = self.mse_loss(x_high_freq, target_high_freq)
        
        # 顏色一致性損失 - 確保整體色調正確
        color_loss = self.l1_loss(x.mean(dim=[2, 3]), target.mean(dim=[2, 3]))
        
        # 針對RRDB架構優化的損失權重
        total_loss = (mse_loss * 2.5 +           # 更高的MSE權重，增強PSNR
                      0.3 * mse_down1 +          # 適度降低低分辨率MSE權重
                      0.1 * mse_down2 +          # 更低的二級下採樣權重
                      1.0 * edge_loss +          # 增強邊緣一致性
                      0.5 * high_freq_loss +     # 強調高頻細節
                      0.05 * color_loss)         # 輕微約束顏色
        
        return total_loss


# SSIM損失 - 結構相似性評估
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size)
        
    def _create_window(self, window_size):
        _1D_window = torch.Tensor([1.0/window_size] * window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == 3:
            # 分開計算並平均三個通道
            sum_ssim = 0.0
            for i in range(3):
                sum_ssim += self._ssim(img1[:, i:i+1, :, :], img2[:, i:i+1, :, :])
            ssim_loss = sum_ssim / 3.0
        else:
            ssim_loss = self._ssim(img1, img2)
        
        return 1.0 - ssim_loss  # 轉換為損失（越小越好）
    
    def _ssim(self, img1, img2):
        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=1) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


# 改進的資料集類別 - 動態均衡採樣策略
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
        self.epoch = 0  # 初始化epoch計數器
        
        # 根據檔案名組織圖像
        for path in self.image_paths:
            try:
                base_name = os.path.splitext(os.path.basename(path))[0].rsplit('_', 1)[0]
                quality = os.path.splitext(os.path.basename(path))[0].rsplit('_', 1)[1]
                if base_name not in self.image_groups:
                    self.image_groups[base_name] = {}
                self.image_groups[base_name][quality] = path
            except Exception as e:
                print(f"處理路徑時出錯: {path}, {e}")
                continue
                
        self.valid_groups = []
        # 擴充有效的品質組合範圍 - 支援更多品質級別
        for base_name, qualities in self.image_groups.items():
            if 'q100' in qualities and any(f'q{q}' in qualities for q in range(10, 91, 10)): # 擴展到所有品質級別
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
            print("圖像預載入完成！")

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        base_name = self.valid_groups[idx]
        qualities = self.image_groups[base_name]
        low_quality_options = [q for q in qualities.keys() if q != 'q100']
        
        # 根據當前epoch調整採樣策略，實現課程學習
        epoch_phase = min(1.0, self.epoch / 200)  # 0~1之間，延長過渡期
        
        # 調整採樣權重 - 訓練前期聚焦於更低品質，後期均衡
        weights = []
        for q in low_quality_options:
            q_num = int(q[1:])
            if epoch_phase < 0.3:  # 第一階段：重點學習最低品質(q10-q30)
                if q_num <= 30:
                    weight = 5.0 / (q_num + 1)
                else:
                    weight = 0.5 / q_num
            elif epoch_phase < 0.6:  # 第二階段：均衡學習中低品質(q10-q50)
                if q_num <= 50:
                    weight = 2.0 / (q_num + 1)
                else:
                    weight = 1.0 / q_num
            else:  # 第三階段：均衡學習所有品質
                weight = 1.0 / math.sqrt(q_num)
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
                
            # 確保尺寸一致
            width, height = low_quality_image.size
            high_quality_image = high_quality_image.resize((width, height), Image.LANCZOS)
            
            # 確保裁剪大小不超過圖像尺寸
            crop_size = min(min(self.crop_size, width), min(self.crop_size, height))
            
            if self.augment:
                # 隨機裁剪
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_quality_image, 
                    output_size=(crop_size, crop_size)
                )
                low_quality_image = transforms.functional.crop(low_quality_image, i, j, h, w)
                high_quality_image = transforms.functional.crop(high_quality_image, i, j, h, w)
                
                # 基本增強 - 水平翻轉
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.hflip(low_quality_image)
                    high_quality_image = transforms.functional.hflip(high_quality_image)
                
                # 基本增強 - 旋轉
                if random.random() > 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    low_quality_image = transforms.functional.rotate(low_quality_image, angle)
                    high_quality_image = transforms.functional.rotate(high_quality_image, angle)
                
                # 顏色抖動 - 僅應用於輸入圖像
                if random.random() > 0.7:
                    color_jitter = transforms.ColorJitter(
                        brightness=0.05, 
                        contrast=0.05, 
                        saturation=0.05, 
                        hue=0.02
                    )
                    low_quality_image = color_jitter(low_quality_image)
                
                # 隨機加噪 - 僅訓練早期使用
                if random.random() > 0.85 and self.epoch < 50:
                    low_quality_tensor = transforms.ToTensor()(low_quality_image)
                    noise = torch.randn_like(low_quality_tensor) * random.uniform(0.003, 0.015)
                    low_quality_tensor = torch.clamp(low_quality_tensor + noise, 0, 1)
                    low_quality_image = transforms.ToPILImage()(low_quality_tensor)
            
            # 轉換為張量
            if self.transform:
                low_quality_tensor = self.transform(low_quality_image)
                high_quality_tensor = self.transform(high_quality_image)
                
            # 為了與原先的代碼相容，修改返回格式
            return {'input': low_quality_tensor, 'target': high_quality_tensor}
            
        except Exception as e:
            print(f"處理圖像時出錯: {base_name}, {e}")
            # 出錯時返回全零張量
            return {'input': torch.zeros(3, self.crop_size, self.crop_size), 
                    'target': torch.zeros(3, self.crop_size, self.crop_size)}


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def calculate_psnr(img1, img2):
    """計算峰值信噪比(PSNR)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def process_large_images(model, images, max_size=512, overlap=96):
    """處理大尺寸圖像，使用分塊處理和無縫拼接"""
    b, c, h, w = images.shape
    result = torch.zeros_like(images)
    
    # 如果圖像較小，直接處理
    if h <= max_size and w <= max_size:
        return model(images)
    
    # 計算分塊處理
    h_blocks = max(1, (h - overlap) // (max_size - overlap))
    w_blocks = max(1, (w - overlap) // (max_size - overlap))
    
    h_size = min(max_size, h)
    w_size = min(max_size, w)
    
    # 創建權重累積張量用於平均重疊區域
    weights = torch.zeros_like(images)
    
    # 分塊處理並拼接
    for i in range(h_blocks):
        h_start = i * (h_size - overlap) if i > 0 else 0
        h_end = min(h_start + h_size, h)
        
        for j in range(w_blocks):
            w_start = j * (w_size - overlap) if j > 0 else 0
            w_end = min(w_start + w_size, w)
            
            # 提取塊
            block = images[:, :, h_start:h_end, w_start:w_end]
            
            # 處理塊
            processed_block = model(block)
            
            # 創建漸變權重矩陣用於平滑拼接邊緣
            weight = torch.ones_like(processed_block)
            
            # 應用漸變邊界權重
            if i > 0:  # 頂部邊緣
                for k in range(overlap):
                    weight[:, :, k, :] = k / overlap
            if i < h_blocks - 1 and h_end == h_start + h_size:  # 底部邊緣
                for k in range(overlap):
                    weight[:, :, -(k+1), :] = k / overlap
            if j > 0:  # 左側邊緣
                for k in range(overlap):
                    weight[:, :, :, k] = k / overlap
            if j < w_blocks - 1 and w_end == w_start + w_size:  # 右側邊緣
                for k in range(overlap):
                    weight[:, :, :, -(k+1)] = k / overlap
            
            # 累積結果和權重
            result[:, :, h_start:h_end, w_start:w_end] += processed_block * weight
            weights[:, :, h_start:h_end, w_start:w_end] += weight
    
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
        for i, batch in enumerate(val_loader):
            if max_validate_batches is not None and i >= max_validate_batches:
                break
                
            input_imgs = batch['input'].to(device)
            target_imgs = batch['target'].to(device)
            
            # 處理可能的大尺寸圖像
            output_imgs = process_large_images(generator, input_imgs)
            
            # 計算性能指標
            batch_size = input_imgs.size(0)
            for j in range(batch_size):
                total_samples += 1
                psnr = calculate_psnr(output_imgs[j], target_imgs[j])
                val_psnr += psnr
                
                # 記錄MSE
                mse = F.mse_loss(output_imgs[j], target_imgs[j]).item()
                val_mse += mse
            
            # 保存一些圖像用於視覺化
            if return_images and i == 0:
                for j in range(min(3, batch_size)):
                    validation_images.append({
                        'input': input_imgs[j].cpu(),
                        'output': output_imgs[j].cpu(),
                        'target': target_imgs[j].cpu()
                    })
    
    avg_psnr = val_psnr / total_samples if total_samples > 0 else 0
    avg_mse = val_mse / total_samples if total_samples > 0 else 0
    
    if return_images:
        return avg_psnr, validation_images
    return avg_psnr


def save_model_with_metadata(model, path, metadata=None):
    """保存模型並附加元數據"""
    torch.save(model.state_dict(), path)
    if metadata:
        metadata_path = path.replace('.pth', '_meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


def train_model(generator, discriminator, train_loader, val_loader, criterion_dict, 
                g_optimizer, d_optimizer, num_epochs, device, 
                save_dir="./models", log_dir="./logs", model_name="NS_ICE", 
                gradient_accumulation_steps=4, checkpoint_interval=50,
                fast_validation=False):
    """簡化的訓練函數，使用固定學習率"""
    generator.to(device)
    discriminator.to(device)
    scaler = torch.amp.GradScaler()
    
    # 訓練日誌設定
    best_g_loss = float('inf')
    best_psnr = 0.0
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, "w") as f:
        f.write("Epoch,G_Loss,D_Loss,PSNR,Time\n")
    training_start_time = time.time()
    
    # 更新資料集epoch參數
    for dataset in [train_loader.dataset, val_loader.dataset]:
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'epoch'):
            dataset.dataset.epoch = 0
    
    # 為RRDB模型優化的固定損失權重配置
    mse_weight = 3.0              # 高MSE權重優先確保PSNR
    perceptual_weight = 0.25      # 較低的感知損失權重
    ssim_weight = 0.7             # 較高的SSIM權重保持結構
    adversarial_weight = 0.0001   # 極低的GAN權重僅用於增強細節
    
    # RRDB架構的模型元數據
    model_metadata = {
        "version": "5.0.0",
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
    
    # 驗證配置
    validate_batches = 40 if fast_validation else None
    
    # 損失函數
    perceptual_criterion = criterion_dict['perceptual']
    ssim_criterion = criterion_dict['ssim']
    
    # NaN檢測和防止
    nan_detected = False
    
    # BCEWithLogitsLoss 用於替換 binary_cross_entropy
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        # 更新資料集的epoch參數
        for dataset in [train_loader.dataset, val_loader.dataset]:
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'epoch'):
                dataset.dataset.epoch = epoch
                
        generator.train()
        discriminator.train()
        epoch_start_time = time.time()
        
        # 追蹤損失
        total_g_loss = 0
        total_d_loss = 0
        total_batches = 0
        
        # 訓練迴圈
        for i, batch in enumerate(train_loader):
            input_imgs = batch['input'].to(device)
            target_imgs = batch['target'].to(device)
            batch_size = input_imgs.size(0)
            
            # 更新判別器
            d_optimizer.zero_grad()
            
            # 使用新的 autocast 語法
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # 生成器輸出
                fake_imgs = generator(input_imgs)
                
                # 判別器預測
                real_pred = discriminator(target_imgs)
                fake_pred = discriminator(fake_imgs.detach())
                
                # 判別器損失 - 使用標籤平滑化技術
                real_label = torch.ones_like(real_pred[0]) * 0.9  # 標籤平滑化
                fake_label = torch.zeros_like(fake_pred[0])
                
                d_real_loss = 0
                d_fake_loss = 0
                
                # 使用 sigmoid 後的值計算 BCE 損失
                for scale_idx in range(len(real_pred)):
                    d_real_loss += F.binary_cross_entropy(torch.sigmoid(real_pred[scale_idx]), real_label)
                    d_fake_loss += F.binary_cross_entropy(torch.sigmoid(fake_pred[scale_idx]), fake_label)
                
                d_loss = (d_real_loss + d_fake_loss) / len(real_pred) / 2
            
            # 梯度累積
            if i % gradient_accumulation_steps == 0:
                scaler.scale(d_loss).backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(d_optimizer)
                scaler.update()
            else:
                with scaler.scale(d_loss):
                    d_loss.backward()
            
            # 更新生成器
            if i % 2 == 0:  # 每2步更新一次生成器
                g_optimizer.zero_grad()
                
                # 使用新的 autocast 語法
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    # 重新計算生成器輸出
                    fake_imgs = generator(input_imgs)
                    
                    # 判別器對生成圖像的預測
                    fake_pred = discriminator(fake_imgs)
                    
                    # 各種損失
                    pixel_loss = F.mse_loss(fake_imgs, target_imgs)
                    perceptual_loss = perceptual_criterion(fake_imgs, target_imgs)
                    ssim_loss = ssim_criterion(fake_imgs, target_imgs)
                    
                    # 生成器對抗損失 - 使用 sigmoid 後的值計算 BCE 損失
                    g_adv_loss = 0
                    for pred in fake_pred:
                        g_adv_loss += F.binary_cross_entropy(torch.sigmoid(pred), torch.ones_like(torch.sigmoid(pred)))
                    g_adv_loss /= len(fake_pred)
                    
                    # 總生成器損失
                    g_loss = (
                        mse_weight * pixel_loss +
                        perceptual_weight * perceptual_loss +
                        ssim_weight * ssim_loss +
                        adversarial_weight * g_adv_loss
                    )
                    
                    # 檢查NaN損失
                    if torch.isnan(g_loss):
                        print(f"警告: 偵測到NaN損失在批次 {i}，跳過此批次")
                        nan_detected = True
                        continue
                
                # 梯度累積
                if i % gradient_accumulation_steps == 0:
                    scaler.scale(g_loss).backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    scaler.step(g_optimizer)
                    scaler.update()
                else:
                    with scaler.scale(g_loss):
                        g_loss.backward()
            
            # 更新追蹤
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_batches += 1
            
            # 定期顯示進度
            if i % 50 == 0:
                avg_g_loss = total_g_loss / max(1, total_batches)
                avg_d_loss = total_d_loss / max(1, total_batches)
                elapsed = time.time() - epoch_start_time
                remaining = elapsed / (i + 1) * (len(train_loader) - i - 1)
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(train_loader)}] "
                      f"G_Loss: {avg_g_loss:.4f} D_Loss: {avg_d_loss:.4f} "
                      f"Time: {format_time(elapsed)} ETA: {format_time(remaining)}")
        
        # 計算epoch平均損失
        avg_g_loss = total_g_loss / total_batches if total_batches > 0 else 0
        avg_d_loss = total_d_loss / total_batches if total_batches > 0 else 0
        
        # NaN檢測
        if nan_detected:
            print("警告: 此epoch檢測到NaN損失，考慮降低學習率或檢查模型")
        
        # 驗證
        val_psnr = validate(generator, val_loader, device, max_validate_batches=validate_batches)
        
        # 顯示訓練和驗證結果
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"G_Loss: {avg_g_loss:.4f} D_Loss: {avg_d_loss:.4f} "
              f"PSNR: {val_psnr:.2f} "
              f"Time: {format_time(epoch_time)}")
        
        # 記錄到檔案
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f},{val_psnr:.6f},{epoch_time:.2f}\n")
        
        # 檢查是否需要保存模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            model_metadata["performance"]["best_psnr"] = best_psnr
            best_path = os.path.join(save_dir, f"{model_name}_best_psnr.pth")
            save_model_with_metadata(generator, best_path, model_metadata)
            print(f"保存最佳PSNR模型 ({best_psnr:.2f}dB) 到 {best_path}")
            
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            model_metadata["performance"]["best_g_loss"] = best_g_loss
            best_g_path = os.path.join(save_dir, f"{model_name}_best_gloss.pth")
            save_model_with_metadata(generator, best_g_path, model_metadata)
            print(f"保存最佳G_Loss模型 ({best_g_loss:.4f}) 到 {best_g_path}")
        
        # 定期保存檢查點
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1 == num_epochs):
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': avg_g_loss,
                'psnr': val_psnr
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"保存檢查點到 {checkpoint_path}")
        
        # 釋放記憶體
        torch.cuda.empty_cache()
        gc.collect()
            
    # 訓練結束，保存最終模型
    final_path = os.path.join(save_dir, f"{model_name}_final.pth")
    model_metadata["training_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_metadata["training_info"]["total_epochs"] = epoch + 1
    model_metadata["training_info"]["total_time"] = format_time(time.time() - training_start_time)
    model_metadata["performance"]["final_psnr"] = val_psnr
    model_metadata["performance"]["final_g_loss"] = avg_g_loss
    save_model_with_metadata(generator, final_path, model_metadata)
    print("訓練完成！")
    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門影像品質增強訓練器 v5.0')
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01', help='訓練集路徑')
    parser.add_argument('--num_epochs', type=int, default=3000, help='訓練步數')
    parser.add_argument('--batch_size', type=int, default=6, help='訓練批量')
    parser.add_argument('--learning_rate', type=float, default=3e-6, help='學習率')
    parser.add_argument('--crop_size', type=int, default=256, help='訓練裁剪大小')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存路徑')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日誌保存路徑')
    parser.add_argument('--model_name', type=str, default='NS-IC-RRDB-GAN', help='模型名稱')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點路徑')
    parser.add_argument('--grad_accum', type=int, default=2, help='梯度累積步數')
    parser.add_argument('--checkpoint_interval', type=int, default=250, help='檢查點保存間隔(epoch)')
    parser.add_argument('--cache_images', action='store_true', help='將圖片快取到記憶體')
    parser.add_argument('--fast_validation', action='store_true', default=True, help='快速驗證模式')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作線程數')
    parser.add_argument('--model_description', type=str, default='RRDB-GAN圖像品質增強模型', help='模型描述')
    parser.add_argument('--d_lr_factor', type=float, default=0.8, help='判別器學習率係數')
    parser.add_argument('--validate_batches', type=int, default=40, help='驗證批次數')
    
    # RRDB相關參數
    parser.add_argument('--num_rrdb_blocks', type=int, default=16, help='RRDB區塊數量')
    parser.add_argument('--features', type=int, default=64, help='基礎特徵數')
    
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
    
    # 初始化RRDB為基礎的模型
    generator = ImageQualityEnhancer(num_rrdb_blocks=args.num_rrdb_blocks, features=args.features)
    discriminator = MultiScaleDiscriminator(num_scales=3, input_channels=3)
    
    # 創建各類損失函數
    criterion_dict = {
        'perceptual': EnhancedPerceptualLoss().to(device),
        'ssim': SSIMLoss().to(device)
    }
    
    # 使用固定學習率的AdamW優化器
    g_optimizer = torch.optim.AdamW(
        generator.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), 
        lr=args.learning_rate * args.d_lr_factor, 
        betas=(0.9, 0.95),
        weight_decay=1e-4
    )
                                      
    # 從檢查點恢復訓練
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"從檢查點恢復訓練，當前周期: {start_epoch}")
    
    # 開始訓練
    generator, discriminator = train_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_dict=criterion_dict,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        gradient_accumulation_steps=args.grad_accum,
        checkpoint_interval=args.checkpoint_interval,
        fast_validation=args.fast_validation
    )