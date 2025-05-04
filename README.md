# Nagato-Sakura-Image-Charm-Trainer

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer/blob/main/assets/docs/README_jp.md) \]

## 簡介
「長門櫻-影像魅影訓練器」是「長門櫻-影像魅影」的模型訓練器，由於模型訓練要針對不同場景做優化在「長門櫻-影像魅影」中進行模型訓練是件吃力不討好的事，故將模型訓練拆分到這個倉庫進行模型的迭代。

## 公告
由於開發者過於忙碌無法即時更新「長門櫻-影像魅影」的模型訓練器版本，導致模型訓練器會跟「長門櫻-影像魅影訓練器」不同步，如果要進行模型訓練請優先使用本倉庫的代碼訓練。

## 快速開始
> [!NOTE]
> 訓練模型以下皆為必要環境。
### 環境設置
- **Python 3**
  - 下載: https://www.python.org/downloads/windows/
- **PyTorch**
  - 下載: https://pytorch.org/
- NVIDIA GPU驅動程式
  - 下載: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
  - 下載: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - 下載: https://developer.nvidia.com/cudnn
> [!TIP]
> 請按照當前 PyTorch 支援安裝對應的 CUDA 版本。

### 安裝倉庫
> [!IMPORTANT]
> 此為必要步驟。
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm-Trainer.git
cd Nagato-Sakura-Image-Charm-Trainer
pip install -r requirements.txt
```