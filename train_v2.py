import os
import gc
import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True

#----------------- 模型定義 -----------------#
class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super(RRDB, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dense2 = nn.Sequential(
            nn.Conv2d(in_channels + growth_channels, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dense3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Conv2d(in_channels + 3 * growth_channels, in_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out1 = self.dense1(x)
        out = torch.cat([x, out1], dim=1)
        out2 = self.dense2(out)
        out = torch.cat([out, out2], dim=1)
        out3 = self.dense3(out)
        out = torch.cat([out, out3], dim=1)
        out = self.final_conv(out)
        return self.lrelu(residual + out * 0.2)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class ImageQualityEnhancer(nn.Module):
    def __init__(self, num_rrdb_blocks=16, features=64, model_path=None, device=None):
        """初始化圖像品質增強器
        
        參數:
            num_rrdb_blocks: RRDB 區塊數量
            features: 基礎特徵通道數
            model_path: 預訓練模型路徑，如果提供則自動載入權重
            device: 計算裝置 (cuda/cpu)
        """
        super(ImageQualityEnhancer, self).__init__()
        self.conv_first = nn.Conv2d(3, features, kernel_size=3, stride=1, padding=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.rrdb_blocks = nn.ModuleList([RRDB(features * 4) for _ in range(num_rrdb_blocks)])
        self.attention = AttentionBlock(features * 4)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        if model_path and os.path.exists(model_path):
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            self.to(self.device)
            self.eval()

    def forward(self, x):
        initial_features = self.conv_first(x)
        encoder_out = self.encoder(initial_features)
        rrdb_out = encoder_out
        for rrdb in self.rrdb_blocks:
            rrdb_out = rrdb(rrdb_out)
        attention_out = self.attention(rrdb_out)
        upsampled = self.upsample(attention_out)
        out = self.final_conv(upsampled)
        return out

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_scales=3, input_channels=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_scales):
            self.discriminators.append(self._create_discriminator(input_channels))
            
    def _create_discriminator(self, input_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
            x = F.avg_pool2d(x, kernel_size=2)
        return outputs


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, target):
        loss = self.mse_loss(x, target)
        x_down1 = self.avg_pool(x)
        target_down1 = self.avg_pool(target)
        loss += self.mse_loss(x_down1, target_down1)
        x_down2 = self.avg_pool(x_down1)
        target_down2 = self.avg_pool(target_down1)
        loss += self.mse_loss(x_down2, target_down2)
        x_grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        x_grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        loss += 0.5 * (self.mse_loss(x_grad_x, target_grad_x) + self.mse_loss(x_grad_y, target_grad_y))
        return loss

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
        for base_name, qualities in self.image_groups.items():
            if 'q100' in qualities and any(f'q{q}' in qualities for q in [10, 20, 30, 40]):
                self.valid_groups.append(base_name)
        if self.cache_images and len(self.valid_groups) > 0:
            print(f"預載入 {len(self.valid_groups)} 組圖像到記憶體...")
            for i, base_name in enumerate(self.valid_groups):
                if i % 100 == 0:
                    print(f"已載入 {i}/{len(self.valid_groups)} 組圖像")
                qualities = self.image_groups[base_name]
                high_quality_path = qualities['q100']
                self.image_cache[high_quality_path] = Image.open(high_quality_path).convert("RGB")
                low_quality_options = [q for q in qualities.keys() if q != 'q100']
                if low_quality_options:
                    low_quality = random.choice(low_quality_options)
                    low_quality_path = qualities[low_quality]
                    self.image_cache[low_quality_path] = Image.open(low_quality_path).convert("RGB")

    def __len__(self):
        return len(self.valid_groups)

    def __getitem__(self, idx):
        base_name = self.valid_groups[idx]
        qualities = self.image_groups[base_name]
        low_quality_options = [q for q in qualities.keys() if q != 'q100']
        low_quality = random.choice(low_quality_options)
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
            crop_size = min(min(self.crop_size, width), min(self.crop_size, height))
            if self.augment:
                i, j, h, w = transforms.RandomCrop.get_params(
                    low_quality_image, 
                    output_size=(crop_size, crop_size)
                )
                low_quality_image = transforms.functional.crop(low_quality_image, i, j, h, w)
                high_quality_image = transforms.functional.crop(high_quality_image, i, j, h, w)
                if random.random() > 0.5:
                    low_quality_image = transforms.functional.hflip(low_quality_image)
                    high_quality_image = transforms.functional.hflip(high_quality_image)
                if random.random() > 0.5:
                    angle = random.choice([0, 90, 180, 270])
                    low_quality_image = transforms.functional.rotate(low_quality_image, angle)
                    high_quality_image = transforms.functional.rotate(high_quality_image, angle)
                if random.random() > 0.7:
                    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
                    low_quality_image = color_jitter(low_quality_image)
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

class EMA():
    """指數移動平均模型追踪器"""
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.model.state_dict()

def validate(generator, val_loader, device, max_validate_batches=None, return_images=False):
    """驗證模型性能"""
    generator.eval()
    val_psnr = 0.0
    total_samples = 0
    validation_images = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            if max_validate_batches is not None and i >= max_validate_batches:
                break
            images, targets = images.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                fake_images = generator(images)
            batch_size = images.size(0)
            for j in range(batch_size):
                val_psnr += calculate_psnr(fake_images[j], targets[j])
                if return_images and i < 2 and j < 2: 
                    validation_images.append((images[j].cpu(), fake_images[j].cpu(), targets[j].cpu()))
            total_samples += batch_size
    if return_images:
        return val_psnr / total_samples, validation_images
    return val_psnr / total_samples

def save_model_with_metadata(model, path, metadata=None):
    """保存模型並附加元數據以與GUI程序兼容"""
    torch.save(model.state_dict(), path)
    if metadata:
        metadata_path = os.path.splitext(path)[0] + "_info.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

def train_model(generator, discriminator, train_loader, val_loader, perceptual_criterion, 
                g_optimizer, d_optimizer, scheduler_g, scheduler_d, num_epochs, device, 
                save_dir="./models", log_dir="./logs", model_name="NS_ICE", 
                gradient_accumulation_steps=4, checkpoint_interval=100,
                fast_validation=False):
    """訓練模型的主函數"""
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
    mse_weight = 1.0
    perceptual_weight = 0.4  
    adversarial_weight = 0.05 
    ema_generator = EMA(generator, decay=0.9995)
    
    model_metadata = {
        "version": "1.0.0",
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
                "adversarial_weight": adversarial_weight
            }
        },
        "performance": {
            "best_psnr": 0.0,
            "best_g_loss": float('inf'),
        }
    }
    validate_batches = 20
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_start_time = time.time()
        batch_count = 0
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            if batch_count % gradient_accumulation_steps == 0:
                d_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                fake_images = generator(images)
                real_outputs = discriminator(targets)
                fake_outputs = discriminator(fake_images.detach())
                d_loss_real = 0
                d_loss_fake = 0
                for real_output, fake_output in zip(real_outputs, fake_outputs):
                    d_loss_real += torch.mean((1 - real_output) ** 2) 
                    d_loss_fake += torch.mean(fake_output ** 2) 
                d_loss = 0.5 * (d_loss_real + d_loss_fake) / len(real_outputs)
                d_loss = d_loss / gradient_accumulation_steps 
            scaler.scale(d_loss).backward()
            if batch_count % gradient_accumulation_steps == 0:
                g_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                fake_outputs = discriminator(fake_images)
                mse_loss = F.mse_loss(fake_images, targets)
                perceptual_loss = perceptual_criterion(fake_images, targets)
                adversarial_loss = 0
                for fake_output in fake_outputs:
                    adversarial_loss += torch.mean((1 - fake_output) ** 2)
                adversarial_loss /= len(fake_outputs)
                g_loss = (mse_weight * mse_loss + 
                          perceptual_weight * perceptual_loss + 
                          adversarial_weight * adversarial_loss)     
                g_loss = g_loss / gradient_accumulation_steps 
            scaler.scale(g_loss).backward()
            batch_count += 1
            epoch_g_loss += g_loss.item() * gradient_accumulation_steps
            epoch_d_loss += d_loss.item() * gradient_accumulation_steps
            if batch_count % gradient_accumulation_steps == 0 or (i == len(train_loader) - 1):
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                scaler.step(d_optimizer)
                scaler.step(g_optimizer)
                scaler.update()
                ema_generator.update(generator)
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
        scheduler_g.step()
        scheduler_d.step()
        current_lr = scheduler_g.get_last_lr()[0]
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        if epoch % 10 == 0:
            val_psnr, validation_images = validate(ema_generator.model, val_loader, device, 
                                                  max_validate_batches=None, 
                                                  return_images=True)
        else:
            val_psnr = validate(ema_generator.model, val_loader, device, 
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
        
        is_best_loss = False
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            is_best_loss = True
            model_metadata["performance"]["best_g_loss"] = best_g_loss
            model_metadata["performance"]["best_g_loss_epoch"] = epoch + 1
            best_loss_path = os.path.join(save_dir, f"{model_name}_best_loss.pth")
            save_model_with_metadata(generator, best_loss_path, model_metadata)
            best_loss_ema_path = os.path.join(save_dir, f"{model_name}_best_loss_ema.pth")
            save_model_with_metadata(ema_generator.model, best_loss_ema_path, model_metadata)
            print(f"最佳損失模型已保存於 Epoch {epoch+1}，G Loss: {avg_g_loss:.4f}")
            
        is_best_psnr = False
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            is_best_psnr = True
            model_metadata["performance"]["best_psnr"] = best_psnr
            model_metadata["performance"]["best_psnr_epoch"] = epoch + 1
            best_psnr_path = os.path.join(save_dir, f"{model_name}_best_psnr.pth")
            save_model_with_metadata(generator, best_psnr_path, model_metadata)
            best_psnr_ema_path = os.path.join(save_dir, f"{model_name}_best_psnr_ema.pth")
            save_model_with_metadata(ema_generator.model, best_psnr_ema_path, model_metadata)
            print(f"最佳PSNR模型已保存於 Epoch {epoch+1}，PSNR: {val_psnr:.2f} dB")
            
        save_checkpoint = (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1
        if is_best_loss or is_best_psnr or save_checkpoint:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'ema_generator_state_dict': ema_generator.state_dict(),
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
            
        # 每5個epoch清理一次記憶體
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
    # 訓練結束，保存最終模型
    final_path = os.path.join(save_dir, f"{model_name}_final.pth")
    final_ema_path = os.path.join(save_dir, f"{model_name}_final_ema.pth")
    model_metadata["training_info"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    model_metadata["training_info"]["total_epochs"] = num_epochs
    model_metadata["training_info"]["total_time"] = format_time(time.time() - training_start_time)
    model_metadata["performance"]["final_psnr"] = val_psnr
    model_metadata["performance"]["final_g_loss"] = avg_g_loss
    save_model_with_metadata(generator, final_path, model_metadata)
    save_model_with_metadata(ema_generator.model, final_ema_path, model_metadata)
    print("訓練完成！")
    return generator, discriminator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門影像品質增強訓練器 v2.0')
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01_hf', help='訓練集路徑')
    parser.add_argument('--num_epochs', type=int, default=1000, help='訓練步數')
    parser.add_argument('--batch_size', type=int, default=6, help='訓練批量')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='學習率')
    parser.add_argument('--crop_size', type=int, default=256, help='訓練裁剪大小')
    parser.add_argument('--save_dir', type=str, default='./models', help='模型保存路徑')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日誌保存路徑')
    parser.add_argument('--model_name', type=str, default='NS_ICE', help='模型名稱')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    parser.add_argument('--resume', type=str, default=None, help='恢復訓練的檢查點路徑')
    parser.add_argument('--grad_accum', type=int, default=4, help='梯度累積步數')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='檢查點保存間隔(時代)')
    parser.add_argument('--cache_images', action='store_true', help='是否將圖片快取到記憶體')
    parser.add_argument('--fast_validation', action='store_true', default=True, help='啟用快速驗證')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作線程數')
    parser.add_argument('--model_description', type=str, default='', help='模型描述')
    parser.add_argument('--d_lr_factor', type=float, default=0.5, help='判別器學習率係數')
    parser.add_argument('--validate_batches', type=int, default=20, help='驗證批次數')
    args = parser.parse_args()
    
    # 設定隨機種子
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
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 初始化模型
    generator = ImageQualityEnhancer(num_rrdb_blocks=16)
    discriminator = MultiScaleDiscriminator(num_scales=3)
    perceptual_criterion = PerceptualLoss()
    
    # 使用針對判別器的較低學習率
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate * args.d_lr_factor, betas=(0.9, 0.999))
    
    # 使用OneCycleLR提供更好的收斂
    scheduler_g = OneCycleLR(g_optimizer, max_lr=args.learning_rate, total_steps=args.num_epochs,
                             pct_start=0.05, anneal_strategy='cos', div_factor=25,
                             final_div_factor=1000)
    scheduler_d = OneCycleLR(d_optimizer, max_lr=args.learning_rate * args.d_lr_factor, total_steps=args.num_epochs,
                             pct_start=0.05, anneal_strategy='cos', div_factor=25,
                             final_div_factor=1000)
    
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"載入檢查點: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            ema_generator_state_dict = checkpoint.get('ema_generator_state_dict')
            g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            scheduler_g.load_state_dict(checkpoint['g_scheduler_state_dict'])
            scheduler_d.load_state_dict(checkpoint['d_scheduler_state_dict'])
            best_g_loss = checkpoint.get('best_g_loss', float('inf'))
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"繼續訓練自 epoch {start_epoch}, 目前最佳 PSNR: {best_psnr:.2f}dB")
    
    # 訓練模型
    generator, discriminator = train_model(
        generator, 
        discriminator, 
        train_loader, 
        val_loader,
        perceptual_criterion,
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
        "version": "2.0.0",
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