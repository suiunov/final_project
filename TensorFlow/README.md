# LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement

<div align="center">
  
![Logo](../figs/Logo.png)

[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/abs/2401.15204)
</div>

## Description
This is the TensorFlow version of LYT-Net.

## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n LYTNet python=3.10
conda activate LYTNet
```
- Install Dependencies
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
pip install tensorflow==2.10 "numpy<2" "opencv-python<4.9" tqdm matplotlib lpips
```

### 2. Prepare Datasets
Download the LOLv1 and LOLv2 datasets:

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

**Note:** Under the main directory, create a folder called ```data``` and place the dataset folders inside it.
<details>
  <summary>
  <b>Datasets should be organized as follows:</b>
  </summary>

  ```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
  ```

</details>

**Note:** ```data``` directory should be placed under the ```TensorFlow``` implementation folder.

### 3. Test
You can test the model using the following commands. Pre-trained weights are available at [Google Drive](https://drive.google.com/drive/folders/1LgLUXGy-7fQXVnxyEeyBolkZ5ZX1f_em?usp=sharing). GT Mean evaluation can be done with the ```--gtmean``` argument.

```bash
# Test on LOLv1
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOLv1.h5
# Test on LOLv1 using GT Mean
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOLv1.h5 --gtmean

# Test on LOLv2 Real
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOLv2_Real.h5
# Test on LOLv2 Real using GT Mean
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOLv2_Real.h5 --gtmean

# Test on LOLv2 Synthetic
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOLv2_Synthetic.h5
# Test on LOLv2 Synthetic using GT Mean
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOLv2_Synthetic.h5 --gtmean
```

### 4. Compute Complexity
You can test the model complexity (FLOPS/Params) using the following command:
```bash
# To run FLOPS check with default (1,256,256,3)
python main.py --complexity

# To run FLOPS check with custom (1,H,W,C)
python main.py --complexity --shape '(H,W,C)'
```

### 5. Train
You can train the model using the following commands:

```bash
# Train on LOLv1
python main.py --train --dataset LOLv1

# Train on LOLv2 Real
python main.py --train --dataset LOLv2_Real

# Train on LOLv2 Synthetic
python main.py --train --dataset LOLv2_Synthetic
```

---

## Extended Scripts（擴充工具腳本）

以下為本專案額外開發的三支工具腳本，涵蓋 **單張推理**、**LoLI-Street 訓練**、**模型對比視覺化**。

> **注意**：所有指令皆在 `TensorFlow/` 目錄下執行，並確保已啟用 conda 環境：
> ```bash
> conda activate LYTNet
> cd TensorFlow
> ```

---

### 6. 單張推理（`scripts/infer.py`）

使用指定的模型權重對單張低光照片進行增強推理，輸出增強後的圖片。

#### 基本用法

```bash
python scripts/infer.py --weights <權重路徑.h5> --image <圖片路徑>
```

#### 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|:----:|--------|------|
| `--weights` | ✅ | — | 模型權重檔路徑（`.h5` 格式） |
| `--image` | ✅ | — | 輸入圖片路徑（支援 jpg / png / bmp） |
| `--output` | ❌ | `<原檔名>_enhanced.png` | 輸出圖片儲存路徑 |
| `--max_dim` | ❌ | `0`（不縮放） | 推論時的最大邊長限制，避免記憶體不足（建議 512 或 640） |
| `--compare` | ❌ | `false` | 額外輸出一張原圖 vs 增強圖的並排對比圖 |

#### 使用範例

```bash
# 使用預訓練權重推理單張圖片
python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./test.jpg

# 使用自訓練權重，指定輸出路徑
python scripts/infer.py --weights ./experiments/loli_scratch_v1/best.h5 --image ./dark.png --output result.png

# 加上 --compare 產生並排對比圖
python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./test.jpg --compare

# 限制最大邊長為 640（大圖避免 OOM）
python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./big.jpg --max_dim 640
```

#### 輸出說明
- 增強圖片：儲存至 `--output` 指定路徑（或自動命名為 `<原檔名>_enhanced.png`）
- 對比圖（`--compare` 啟用時）：儲存為 `<原檔名>_enhanced_comparison.png`

> **注意**：圖片尺寸會自動調整為 8 的倍數（因模型內部有 8x pooling），推論完成後會 resize 回原始尺寸。

---

### 7. LoLI-Street 從零訓練（`scripts/train_loli.py`）

從零開始在 LoLI-Street 33K 配對街景資料集上訓練 LYT-Net。

#### 資料集準備

LoLI-Street 資料集應放置於 `dataset/LoLI-Street Dataset/` 目錄下（與 `TensorFlow/` 同層），結構如下：

```
|--dataset
|    |--LoLI-Street Dataset
|    |    |--Train
|    |    |    |--low
|    |    |    |    |-- *.jpg
|    |    |    |--high
|    |    |    |    |-- *.jpg
|    |    |--Val
|    |    |    |--low
|    |    |    |    |-- *.jpg
|    |    |    |--high
|    |    |    |    |-- *.jpg
```

#### 基本用法

```bash
python scripts/train_loli.py --epochs 200 --tag loli_scratch_v1
```

#### 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|:----:|--------|------|
| `--tag` | ❌ | `loli_scratch_v1` | 實驗名稱，權重儲存於 `./experiments/<tag>/` |
| `--epochs` | ❌ | `200` | 訓練輪數 |
| `--bs` | ❌ | `2` | Batch size |
| `--crop` | ❌ | `256` | 訓練裁剪尺寸（隨機裁 crop × crop） |
| `--lr` | ❌ | `2e-4` | 初始學習率（與原始論文相同） |
| `--no_gtmean` | ❌ | `false` | 停用 GT-Mean PSNR 驗證（加速驗證，跳過 gamma 搜尋） |
| `--train_low` | ❌ | 自動偵測 | 訓練集低光影像路徑（glob pattern） |
| `--train_high` | ❌ | 自動偵測 | 訓練集正常影像路徑（glob pattern） |
| `--val_low` | ❌ | 自動偵測 | 驗證集低光影像路徑（glob pattern） |
| `--val_high` | ❌ | 自動偵測 | 驗證集正常影像路徑（glob pattern） |

#### 使用範例

```bash
# 基本訓練（200 epochs，自動偵測 LoLI-Street 路徑）
python scripts/train_loli.py --epochs 200 --tag loli_scratch_v1

# 快速測試訓練（較少 epochs、關閉 GT-Mean）
python scripts/train_loli.py --epochs 10 --tag quick_test --no_gtmean

# 自訂資料集路徑
python scripts/train_loli.py --epochs 100 --tag custom_v1 \
    --train_low "./my_data/train/low/*.jpg" \
    --train_high "./my_data/train/high/*.jpg" \
    --val_low "./my_data/val/low/*.jpg" \
    --val_high "./my_data/val/high/*.jpg"
```

#### 輸出說明
- 權重儲存於 `./experiments/<tag>/` 目錄
- `best.h5`：驗證 GT-PSNR 最佳的權重（方便後續引用）
- `checkpoint_ep<N>.h5`：每 50 個 epoch 自動存檔
- `final.h5`：訓練結束時的最終權重
- 每個 epoch 結束時輸出 `loss`、`PSNR`、`GT-PSNR`、`SSIM` 指標

---

### 8. 模型對比視覺化（`scripts/gen_comparison.py`）

將「自訓練模型」與「原始預訓練模型」的推理結果並排對比，支援兩種模式：

| 模式 | 說明 | 輸出格式 |
|------|------|----------|
| **資料集模式** | 使用 LOLv1 / LOLv2 測試集（有 Ground Truth） | 4 格對比：Input ｜ Ours ｜ Original ｜ GT（附 PSNR / SSIM / LPIPS） |
| **自訂圖片模式** | 指定任意圖片路徑（無 Ground Truth） | 3 格對比：Input ｜ Ours ｜ Original |

#### 基本用法

```bash
# 模式 1：資料集模式
python scripts/gen_comparison.py --dataset LOLv1

# 模式 2：自訂圖片模式
python scripts/gen_comparison.py --images path1.jpg path2.jpg path3.jpg
```

#### 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|:----:|--------|------|
| `--dataset` | ✅* | — | 使用預設資料集：`LOLv1` / `LOLv2_Real` / `LOLv2_Synthetic` |
| `--images` | ✅* | — | 指定圖片路徑（支援 glob，如 `*.jpg`） |
| `--ours` | ❌ | `./experiments/loli_scratch_v1/best.h5` | 自訓練模型權重路徑 |
| `--orig` | ❌ | `./pretrained_weights/LOLv1.h5` | 原始模型權重路徑（作為 baseline） |
| `--out` | ❌ | 自動命名 | 輸出資料夾路徑 |
| `--max_dim` | ❌ | `512` | 推論時最大解析度（避免 OOM） |
| `--num` | ❌ | 全部 | 最多處理幾張圖片 |

> *`--dataset` 與 `--images` 擇一使用，不可同時指定。

#### 使用範例

```bash
# 用 LOLv1 測試集對比（取 10 張均勻取樣）
python scripts/gen_comparison.py --dataset LOLv1 --num 10

# 用 LOLv2_Real 測試集對比，並使用自訂權重
python scripts/gen_comparison.py --dataset LOLv2_Real \
    --ours ./experiments/loli_scratch_v1/best.h5

# 自訂圖片對比（支援 glob 展開）
python scripts/gen_comparison.py --images ./my_images/*.jpg --max_dim 640

# 指定輸出目錄
python scripts/gen_comparison.py --dataset LOLv1 --out ./my_results
```

#### 輸出說明
- 每張圖片產生一張 `<檔名>_comparison.png` 對比圖
- 資料集模式輸出至 `./results/comparison_<dataset>/`
- 自訂圖片模式輸出至 `./results/comparison_custom/`
- 資料集模式下，對比圖包含 PSNR、SSIM、LPIPS 量化指標

---

## Citation
Preprint Citation
```
@article{brateanu2024,
  title={LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image Enhancement},
  author={Brateanu, Alexandru and Balmez, Raul and Avram, Adrian and Orhei, Ciprian},
  journal={arXiv preprint arXiv:2401.15204},
  year={2024}
}
```
