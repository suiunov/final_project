# LYT-Net 遷移訓練研究提案：低光照交通號誌影像增強（TensorFlow 版）

> **主題**：以 LYT-Net（SPL 2025）TensorFlow 實作為骨幹，建立一套面向「低光照交通號誌即時偵測管線前處理」的遷移訓練流程
> **目標情境**：即時偵測管線前處理（enhancement-as-preprocessing for YOLO / Detector）
> **資料集現況**：完全沒有自有資料，需從零開始建立
> **撰寫日期**：2026-04-23
> **作者**：Sayone

---

## 0. 摘要 (Executive Summary)

本提案針對「低光照交通號誌（traffic signal/traffic light）即時偵測」的前處理需求，規劃以 **LYT-Net**（45K 參數、3.49 GFLOPS 的輕量化 YUV Transformer LLIE 網路）作為增強骨幹，進行 **domain-specific transfer learning**。由於偵測目標為「會主動發光的紅/黃/綠號誌」，其在低光下易出現 **高光飽和（highlight clipping）、光暈（halo / blooming）、色偏（chromatic shift）**，與一般 LOL 資料集中「全黑室內」分佈差異明顯。因此遷移訓練的核心在於：

1. 將 LYT-Net 預訓練權重（`LOLv1.h5`、`LOLv2_Real.h5`、`LOLv2_Synthetic.h5`）作為初始化。
2. 建立 **交通領域配對資料集**（真實配對 + 合成配對 + 零參考訓練資料）。
3. 修改 Loss：抑制過度增亮導致的號誌顏色飽和，加入 **偵測感知（detection-aware）** 的 joint loss（可選）。
4. 以 **PSNR/SSIM/LPIPS/NIQE** 量化畫質、以 **mAP on night detection** 量化下游收益。

整體可行性：★★★★☆（高）。LYT-Net 架構輕量（適合前處理）、已開源 TensorFlow 版訓練 / 測試腳本，且同質任務（YOLO-LLTS、En-YOLO、FE-YOLO）在 2025 年已有多篇成功案例驗證此路線。

---

## 1. LYT-Net 架構深度分析（TensorFlow 版）

閱讀 `TensorFlow/model/arch.py`、`losses.py`、`scripts/train.py` 後，整理關鍵設計如下。

### 1.1 核心思想：YUV 雙路徑

```
RGB ──tf.image.rgb_to_yuv──▶ Y (亮度) ┐
                              U (色度) ├── 各自獨立處理，最後 Conv 融合
                              V (色度) ┘
```

**為什麼這對交通號誌有利？**
號誌的「紅/黃/綠」訊息主要承載於 U、V 通道。LYT-Net 將 U、V 獨立送入 `Denoiser`（CWD, Channel-Wise Denoiser），可避免亮度增強時「把紅燈洗白、把綠燈變成青白」—— 這正是一般 RGB-space LLIE 最容易犯的錯。此設計對「保留號誌類別色相」有天然優勢。

### 1.2 主要模組（參見 `arch.py`）

| 模組 | 行數 | 作用 | 對交通號誌的意義 |
|------|------|------|------------------|
| `Denoiser`（即 CWD） | 76–100 | 對 U/V 通道 4 層下採樣 + MHSA bottleneck + U-Net skip | 在色度通道壓雜訊，避免紅綠燈在黑夜中被噪點吞噬 |
| `MultiHeadSelfAttention` | 35–74 | 4-head self-attention | 對 Y 通道的全域照度建模（夜景中局部很暗、局部過曝的街景特別有用） |
| `MSEFBlock` | 7–20 | LayerNorm + DepthwiseConv × SEBlock 融合 | 將亮度 cue 注入色度分支，有助於恢復暗部號誌的邊緣 |
| `SEBlock` | 22–33 | Global Avg Pool + Dense(relu/tanh) | 對亮/色做 channel-wise gating |
| `LYT` (top) | 102–151 | `filters=32`，總計 ~45K 參數 | 單張 256×256 僅 3.49 GFLOPS，**可直接串接於 YOLO 前端做 live preprocessing** |

### 1.3 Loss 設計（參見 `losses.py`）

LYT-Net 採 **6 項加權複合 loss**：

```
L_total = 1.00·SmoothL1 + 0.50·MS-SSIM + 0.06·Perceptual(VGG19-block3) 
        + 0.25·ColorLoss + 0.05·HistogramLoss + 0.0083·PSNRLoss
```

- **SmoothL1 / MS-SSIM**：基礎畫質
- **Perceptual Loss**：使用 ImageNet pre-trained VGG19 `block3_conv3`。**注意：遷移時若目標域是夜間街景，VGG 對自然光譜的感知可能偏離，但由於其係「高層語意約束」，一般仍保留**
- **ColorLoss**：`|mean(y_true) - mean(y_pred)|` —— 全圖色調一致性
- **HistogramLoss**：直方圖 soft-matching（σ=0.01 的高斯核）—— **關鍵：可能造成號誌局部高亮被「強行壓平」，遷移時建議降權**
- **PSNRLoss**：`40 - PSNR`

### 1.4 訓練流程關鍵設定（參見 `scripts/train.py`）

- 輸入解析度：`256×256`（random crop，見 `data_loading.py:61`）
- 輸出範圍：`tanh` → `[-1, 1]`，在 loss 計算前 rescale 回 `[0,1]`
- Batch size：**1**（由於模型極小且訓練影像較大，原作採 BS=1）
- Optimizer：Adam + `CosineDecayWithRestartsLearningRateSchedule` (`initial_lr=2e-4, min_lr=1e-6, first_decay_steps=150·steps_per_epoch`)
- 總 epochs：1000（含 cosine restart）
- 資料增強：隨機左右翻、上下翻、90° 旋轉（`data_loading.py:11-30`）
- 評估時的 `gt_mean` 技巧：對預測圖做 γ-scan 找到最佳 γ 再算 PSNR —— 論文 ICLR 2025 spotlight **GT-Mean Loss** 也在討論這個「亮度匹配」問題

### 1.5 預訓練權重清單

| 檔案 | 訓練資料 | TF PSNR | TF SSIM |
|------|----------|---------|---------|
| `pretrained_weights/LOLv1.h5`          | LOL v1（500 對室內 low/normal） | 27.23 | 0.853 |
| `pretrained_weights/LOLv2_Real.h5`     | LOL v2 Real（真實拍攝配對）      | 27.80 | 0.873 |
| `pretrained_weights/LOLv2_Synthetic.h5`| LOL v2 Synthetic（合成配對）    | 29.39 | 0.939 |

**建議起點**：以 `LOLv2_Real.h5` 作為 fine-tune 起始——因真實相機噪訊分佈更接近車載相機的夜拍情境；若後續要餵合成資料，再補一次從 `LOLv2_Synthetic.h5` 的實驗做 ablation。

---

## 2. 可用資料集盤點

### 2.1 交通號誌 / 交通號誌專用資料集

| 資料集 | 規模 | 夜間佔比 | 是否配對 low/normal | 標註 | 授權 | 建議用途 |
|--------|------|----------|----------------------|------|------|----------|
| **LISA Traffic Light** | 43,007 frames / 113,888 燈泡 | 高（有夜間子集） | ✗（僅偵測標註） | bbox + 狀態 | 公開 | **目標域偵測資料**；合成低光來源 |
| **Bosch Small Traffic Lights (BSTLD)** | 13,427 frames / ~24K lights | 低（日間為主） | ✗ | bbox + 狀態 | Research | **日間乾淨影像** → 合成低光來源 |
| **S2TLD** (SJTU) | 5,786 frames / 14,130 instances | 含夜間 / 閃爍情境 | ✗ | bbox + 5 類 | 公開 (GitHub) | 目標域偵測 |
| **DriveU Traffic Light (DTLD)** | 230K+ annotations / 11 城市 | 含夜間 | ✗ | bbox + rich meta | Research | 最大規模，適合做小物件評估 |
| **TN-TLD**（2025） | — | 夜間為主 | ✗ | bbox | 公開 | LNT-YOLO 作者釋出，為最新夜間號誌 benchmark |
| **CNTSSS**（2025，中國夜間交通標誌） | 4,062 | 100% 夜間 | ✗ | bbox + 類別 | YOLO-LLTS 附帶 | 目標域偵測，亞洲場景貼近台灣 |

### 2.2 低光照增強專用（做 LLIE 預訓練 / 繼續訓練）

| 資料集 | 規模 | 配對 | 場景 | 備註 |
|--------|------|------|------|------|
| **LOL v1** | 500 對 | ✓ | 室內 | LYT-Net 已預訓練 |
| **LOL v2 Real** | 689 對 | ✓ | 室內 / 簡單戶外 | LYT-Net 已預訓練 |
| **LOL v2 Synthetic** | 1000 對 | ✓ | 合成 | LYT-Net 已預訓練 |
| **SID (See-in-the-Dark)** | ~5000 對 | ✓ | RAW 極暗 | 車載相機相近的 sensor noise |
| **LoLI-Street** (ACCV 2024) | 街景 LLIE benchmark | ✓ | 街景（駕駛角度） | **最貼近本題的 LLIE benchmark** |
| **ExDark** | 7,363 張 / 12 類 | ✗（unpaired） | 多類低光 | 適合 no-ref 評估 |

### 2.3 夜間駕駛 / 惡劣天氣駕駛（提供 unpaired 夜間樣本）

| 資料集 | 規模 | 任務 | 備註 |
|--------|------|------|------|
| **BDD100K** | 100K 影片 | 偵測 / 分割 | 含 night 子集，可作 domain validation |
| **Dark Zurich** | 8,779 張 | 分割 | day/twilight/night 三時段 |
| **ACDC** | 4,006 張 | 語意分割 | fog/night/rain/snow 四類 |

### 2.4 資料策略建議（因完全沒有自有資料）

三段式堆疊：

```
Stage A：LOL v2 Real 預訓練  ─(已完成，直接載權重)─▶  LYT-Net 基礎畫質能力
         │
         ▼
Stage B：合成配對「日→夜」  ────  以 BSTLD / DTLD / LISA 日間影像為 GT，以
         (gamma + Poisson-Gaussian noise + CRF) 合成 low-light 作為輸入
         │
         ▼
Stage C：真實夜間無參考微調  ──  LISA night / TN-TLD / CNTSSS / Dark Zurich
         以 contrastive / Zero-DCE-style 零參考 loss，或用 enhancement→
         detection 的 joint loss（detector-aware fine-tuning）
```

---

## 3. 相關文獻回顧

### 3.1 LLIE 骨幹演進（選對 backbone）

- **RetinexNet (BMVC 2018)** —— 首個將 Retinex 拆分導入 DNN 的方法，分 Decom-Net + Enhance-Net。
- **Zero-DCE / Zero-DCE++ (CVPR 2020, TPAMI 2021)** —— 無需配對 GT，學習像素級曲線函式；**遷移到駕駛場景常被採用**（見 drone 論文使用 Zero-DCE++ + YOLOv5）。
- **Retinexformer (ICCV 2023 → ICCV 2025 extended)** —— One-stage Retinex + Transformer，在 15 個 benchmarks SOTA；相較 LYT-Net 更重量級（參數百萬級）。
- **LYT-Net (SPL 2025)** —— **本專案骨幹**，0.045M 參數，YUV 雙路徑，對色彩保真度特別有利。
- **ModalFormer (2025，同作者)** —— 多模態 LLIE，可作為日後擴充（若加入 depth / semantic 作為條件）。
- **MambaLLIE (2024)** —— Retinex-aware + State Space Model，比 Transformer 類更省記憶體，可作為 ablation 對照組。

### 3.2 低光照偵測（YOLO 家族與 enhancement 整合）

- **YOLO-LLTS (IEEE TIM 2025)** —— **與本題最接近的工作**。提出 HRFM-SOD（高解析小物件 FPN）、MFIA（多分支特徵交互注意力）、**PGFE（Prior-Guided Feature Enhancement，對應 LYT-Net 的角色）**，並釋出 **CNTSSS** 夜間標誌資料集。
- **LNT-YOLO (Smart Cities 2025)** —— YOLOv7-tiny 為基礎的輕量化夜間號誌偵測，提出 **TN-TLD** 夜間號誌資料集。
- **En-YOLO / FE-YOLO / YOLO-AS / ELS-YOLO (2025)** —— 均採 **end-to-end joint training**：enhancement module 的梯度由 detection loss 回傳。此為 **階段 C 的技術路線**，可作為本專案延伸。
- **WTEFNet (arXiv 2025)** —— 針對 ADAS 的低光即時偵測。

### 3.3 合成低光資料（因為本專案沒有真實配對）

- **物理式相機雜訊建模（Wei et al., 2021）** —— 以 CMOS sensor 為基準，建模 photon shot noise、dark current、pixel circuit noise；在極暗場景下只用合成也能 match 真實表現。
- **Gamma + Poisson-Gaussian + CRF inverse** —— 社群常見配方：
  1. 反轉 camera response function (CRF) 回到線性域
  2. 乘以 exposure ratio（0.05–0.3）
  3. 加 Poisson（shot） + Gaussian（read） noise
  4. re-apply CRF + quantize 回 8-bit
- **Noise Synthesis with Diffusion (2025)** —— 用 diffusion 模型產生逼真 sensor-noise，比傳統 parametric 模型更貼近真實分佈。

### 3.4 評估指標趨勢（2025）

- **GT-Mean Loss (ICLR 2025 spotlight)** —— 指出傳統 PSNR 易被亮度 bias，提出在 loss / metric 兩端做亮度匹配；LYT-Net `train.py` 已有 `find_optimal_gamma` 的雛形實作。
- **下游任務導向的評估** —— 趨勢是**除了 PSNR/SSIM/LPIPS，必須加入 night-time mAP、mAP50-95** 驗證 enhancement 真的對 detector 有幫助。

---

## 4. 可行性評估

### 4.1 優勢（GO）

1. **架構輕量**：45K 參數、3.49 GFLOPS，256×256 輸入在 GTX 1060 級顯卡即可 >50 FPS，完全能當偵測前處理。
2. **YUV 色彩保真**：對「紅/黃/綠」這三類號誌，色彩分離架構先天降低色偏風險。
3. **預訓練權重齊全**：三組 LOL 權重可作為強起點，**小資料即可收斂**。
4. **TensorFlow 原版腳本完整**：`train.py`、`test.py`、`data_loading.py` 已含資料 pipeline、cosine restart scheduler、多 loss；僅需替換資料路徑與微調即可。
5. **近期文獻背書**：YOLO-LLTS、LNT-YOLO、En-YOLO 三篇 2025 年論文均驗證「enhancement + detection」管線在夜間號誌上的增益，方向正確。

### 4.2 風險與緩解

| 風險 | 影響 | 緩解策略 |
|------|------|----------|
| **無真實配對資料** | 無法監督式微調 | 採三段式（§2.4），先合成配對 + 再零參考微調 |
| **號誌過曝 / halo** | LOL 訓練資料以「欠曝」為主，過曝較少；LYT-Net 可能把號誌燈芯進一步洗白 | 加 `highlight-preserving loss`（對 Y 通道 > 0.9 區域強加 L1），或在合成資料中刻意模擬號誌過曝 |
| **VGG perceptual loss 對夜景特徵弱** | block3 特徵來自白天 ImageNet | 可替換為 **夜景預訓練 CLIP image encoder** 或降低 `alpha2` 權重 |
| **HistogramLoss 壓扁號誌高光** | 直方圖對齊會犧牲局部對比 | 遷移時將 `alpha3` 從 0.05 降到 0.01 或遮罩號誌 bbox 不算入 |
| **256×256 random crop 可能剪掉小號誌** | 夜間號誌常僅 10–30 px | 改 crop size 或使用 detection-aware crop（含 bbox 採樣） |
| **RGB→YUV 使用 `tf.image.rgb_to_yuv` 為 BT.601** | 與部分 dataset 的 sRGB/BT.709 不一致 | 若量化觀察到色偏，可改用自定 BT.709 矩陣 |

### 4.3 成本估算

- **算力**：單卡 RTX 3060/4060 即可，預訓練 fine-tune 200 epochs 約 8–15 小時（依資料量）
- **儲存**：LISA 已含 ~14 GB，BSTLD ~11 GB，DTLD ~100 GB（可只取子集）
- **人力**：標註階段若需自拍微調資料，約 500 張夜間台灣路口即可，單人約 1 週

### 4.4 結論

> **整體可行性：高**。以 LYT-Net TF 版為骨架、以 LOLv2_Real 為起點、以合成「日→夜」配對作主訓練、以真實夜間資料作零參考微調，是一條風險可控、算力低、文獻背書充分的路線。

---

## 5. 遷移訓練具體步驟（含 TensorFlow 程式碼）

### Step 0：環境準備

```bash
# 建議 Python 3.10, TF 2.13+
pip install tensorflow==2.13.0 keras matplotlib tqdm opencv-python numpy pillow
# 評估用
pip install lpips torch torchvision     # LPIPS 需要 torch
pip install scikit-image piq            # NIQE / BRISQUE / PSNR / SSIM
```

確認預訓練權重存在：

```bash
ls TensorFlow/pretrained_weights/
# LOLv1.h5  LOLv2_Real.h5  LOLv2_Synthetic.h5
```

### Step 1：建立交通號誌資料集目錄結構

```
TensorFlow/data/TrafficSignal/
├── Synth/                     # 階段 B：日間→合成低光
│   ├── Train/
│   │   ├── input/   (合成低光)
│   │   └── target/  (原日間)
│   └── Test/...
├── RealNight_Paired/          # 若日後採集到少量對齊配對
│   └── Train / Test
└── RealNight_Unpaired/        # 階段 C：TN-TLD / CNTSSS / LISA night
    └── input/
```

### Step 2：合成「日→夜」配對資料（核心技巧）

新建 `TensorFlow/scripts/synthesize_lowlight.py`：

```python
import glob, os, cv2, numpy as np
from tqdm import tqdm

def inverse_crf(x, gamma=2.2):
    return np.power(x, gamma)

def forward_crf(x, gamma=2.2):
    return np.power(np.clip(x, 0, 1), 1.0/gamma)

def add_shot_read_noise(lin, iso_k=0.01, read_sigma=0.005):
    # Poisson (photon shot) + Gaussian (read)
    noisy = np.random.poisson(lin / iso_k) * iso_k
    noisy = noisy + np.random.normal(0, read_sigma, lin.shape)
    return np.clip(noisy, 0, 1)

def synthesize(img_uint8, exposure_ratio=None):
    if exposure_ratio is None:
        exposure_ratio = np.random.uniform(0.03, 0.25)  # 隨機曝光不足
    x = img_uint8.astype(np.float32) / 255.0
    lin = inverse_crf(x)                      # sRGB → linear
    lin_low = lin * exposure_ratio            # 降曝光
    lin_low = add_shot_read_noise(lin_low,
                                  iso_k=np.random.uniform(0.005, 0.02),
                                  read_sigma=np.random.uniform(0.003, 0.01))
    low = forward_crf(lin_low)                # linear → sRGB
    return (low * 255).astype(np.uint8)

def main(src_dir, dst_input, dst_target):
    os.makedirs(dst_input, exist_ok=True)
    os.makedirs(dst_target, exist_ok=True)
    for p in tqdm(sorted(glob.glob(os.path.join(src_dir, '*.png')) +
                         glob.glob(os.path.join(src_dir, '*.jpg')))):
        img = cv2.imread(p)                   # BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        low = synthesize(img_rgb)
        name = os.path.splitext(os.path.basename(p))[0] + '.png'
        cv2.imwrite(os.path.join(dst_target, name),
                    cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(dst_input,  name),
                    cv2.cvtColor(low,    cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # 以 BSTLD 日間影像為 GT，合成低光
    main(src_dir='./data/BSTLD/day_images',
         dst_input='./data/TrafficSignal/Synth/Train/input',
         dst_target='./data/TrafficSignal/Synth/Train/target')
```

### Step 3：修改資料載入（含 bbox-aware crop 選項）

在 `data_loading.py` 新增一條 path 並把 random crop 擴大到 320，避免剪掉小號誌：

```python
# data_loading.py 末端追加
def get_traffic_datasets(raw_glob, gt_glob, crop=320, bs=2):
    tf.random.set_seed(100)
    raw_files = sorted(glob.glob(raw_glob))
    gt_files  = sorted(glob.glob(gt_glob))
    ds = tf.data.Dataset.from_tensor_slices((raw_files, gt_files))

    def _load(raw_p, gt_p):
        raw = tf.image.decode_png(tf.io.read_file(raw_p), channels=3)
        gt  = tf.image.decode_png(tf.io.read_file(gt_p),  channels=3)
        raw = tf.cast(raw, tf.float32); gt = tf.cast(gt, tf.float32)
        stacked = tf.stack([raw, gt], 0)
        stacked = tf.image.random_crop(stacked, size=[2, crop, crop, 3])
        raw, gt = stacked[0], stacked[1]
        raw = (raw / 255.0) * 2 - 1.0
        gt  = (gt  / 255.0) * 2 - 1.0
        return raw, gt

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds
```

### Step 4：載入預訓練權重 + 凍結策略

新建 `TensorFlow/scripts/finetune_traffic.py`：

```python
import os, sys, argparse, datetime, numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

import tensorflow as tf
import data_loading as dl
from model.arch import LYT, Denoiser
from model.losses import load_vgg, smooth_l1_loss, multiscale_ssim_loss, \
                          perceptual_loss, histogram_loss, psnr_loss, color_loss
from model.scheduler import CosineDecayWithRestartsLearningRateSchedule
from find_gamma import find_optimal_gamma, adjust_gamma

tf.random.set_seed(1)

# ------------------ 自訂 loss（壓低 HistogramLoss，加 highlight-preserve） ------------------
def highlight_preserve_loss(y_true, y_pred, thresh=0.85):
    # 只懲罰「真實本來就亮」的地方被預測壓暗
    mask = tf.cast(y_true > thresh, tf.float32)
    return tf.reduce_sum(tf.abs(y_true - y_pred) * mask) / (tf.reduce_sum(mask) + 1e-6)

def traffic_loss(y_true, y_pred, loss_model):
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0
    l1   = smooth_l1_loss(y_true, y_pred)
    ssim = multiscale_ssim_loss(y_true, y_pred)
    perc = perceptual_loss(y_true, y_pred, loss_model)
    hist = histogram_loss(y_true, y_pred)
    psnr = psnr_loss(y_true, y_pred)
    col  = color_loss(y_true, y_pred)
    hl   = highlight_preserve_loss(y_true, y_pred)
    # 遷移用權重：↓hist ↓psnr ↑color ↑highlight
    return (1.00*l1 + 0.50*ssim + 0.06*perc
            + 0.01*hist + 0.0083*psnr + 0.50*col
            + 0.30*hl)

# ------------------ 建模 + 載入預訓練 ------------------
def build_and_load(weight_path):
    denoiser_cb = Denoiser(16); denoiser_cr = Denoiser(16)
    denoiser_cb.build((None, None, None, 1))
    denoiser_cr.build((None, None, None, 1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build((None, None, None, 3))
    if weight_path and os.path.exists(weight_path):
        model.load_weights(weight_path)
        print(f'[OK] Loaded pretrained: {weight_path}')
    return model

# ------------------ 凍結策略（small-data safeguard） ------------------
def apply_freeze(model, freeze_denoisers=True, freeze_process_rgb=False):
    """
    策略說明：
      - 初期（<30 epochs）建議 freeze CWD，只訓 MHSA + MSEF + final_adjustments
        以免小資料下把色度去雜訊能力毀掉
      - 若畫質飽和後再 unfreeze 整網 end-to-end（discriminative fine-tuning）
    """
    if freeze_denoisers:
        model.denoiser_cb.trainable = False
        model.denoiser_cr.trainable = False
    if freeze_process_rgb:
        model.process_y.trainable  = False
        model.process_cb.trainable = False
        model.process_cr.trainable = False
    n_train = sum([np.prod(v.shape) for v in model.trainable_variables])
    print(f'[Freeze] trainable params = {int(n_train)}')

# ------------------ 訓練 step ------------------
@tf.function
def train_step(x, y, model, loss_model, opt):
    with tf.GradientTape() as tape:
        pred = model(x, training=True)
        lv   = traffic_loss(pred, y, loss_model)
    grads = tape.gradient(lv, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return lv

# ------------------ 主流程 ------------------
def main(args):
    # 1. 資料
    tr = dl.get_traffic_datasets(args.train_input, args.train_gt,
                                 crop=args.crop, bs=args.bs)
    te = dl.get_datasets_metrics(args.test_input, args.test_gt, 0)

    # 2. 模型 + 預訓練
    model = build_and_load(args.pretrained)
    apply_freeze(model, freeze_denoisers=args.freeze_cwd)

    # 3. Loss helper
    vgg = load_vgg()

    # 4. Scheduler（因為是 fine-tune，lr 降一個量級）
    steps_per_epoch = sum(1 for _ in tr)
    total_steps     = args.epochs * steps_per_epoch
    sched = CosineDecayWithRestartsLearningRateSchedule(
        initial_lr=2e-5, min_lr=1e-7,
        total_steps=total_steps,
        first_decay_steps=30 * steps_per_epoch)
    opt = tf.keras.optimizers.Adam(learning_rate=sched)

    # 5. Train loop
    os.makedirs(f'./experiments/{args.tag}', exist_ok=True)
    best_psnr = 0.0
    for ep in range(1, args.epochs + 1):
        # --- 第 30 ep 後解凍 CWD，做 discriminative fine-tuning
        if ep == 30 and args.freeze_cwd:
            print('[Unfreeze] CWD denoisers now trainable')
            model.denoiser_cb.trainable = True
            model.denoiser_cr.trainable = True

        for x, y in tr:
            loss_v = train_step(x, y, model, vgg, opt)

        # --- 驗證
        tp = ts = 0.0; n = 0
        for x, y in te:
            p = model(x)
            p = (p + 1.0) / 2.0; y2 = (y + 1.0) / 2.0
            tp += float(tf.reduce_mean(tf.image.psnr(y2, p, 1.0)))
            ts += float(tf.reduce_mean(tf.image.ssim(y2, p, 1.0)))
            n  += 1
        ps, sm = tp / n, ts / n
        print(f'[{datetime.datetime.now():%m-%d %H:%M}] ep{ep:03d} '
              f'loss={loss_v:.4f}  PSNR={ps:.2f}  SSIM={sm:.3f}')

        if ps > best_psnr:
            best_psnr = ps
            model.save_weights(
              f'./experiments/{args.tag}/lyt_traffic_ep{ep}_psnr{ps:.2f}.h5')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag',        default='traffic_synth_v1')
    p.add_argument('--pretrained', default='./pretrained_weights/LOLv2_Real.h5')
    p.add_argument('--train_input',default='./data/TrafficSignal/Synth/Train/input/*.png')
    p.add_argument('--train_gt',   default='./data/TrafficSignal/Synth/Train/target/*.png')
    p.add_argument('--test_input', default='./data/TrafficSignal/Synth/Test/input/*.png')
    p.add_argument('--test_gt',    default='./data/TrafficSignal/Synth/Test/target/*.png')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--bs',     type=int, default=2)
    p.add_argument('--crop',   type=int, default=320)
    p.add_argument('--freeze_cwd', action='store_true', default=True)
    args = p.parse_args()
    main(args)
```

執行：

```bash
cd TensorFlow
python scripts/finetune_traffic.py --tag traffic_synth_v1 \
       --pretrained ./pretrained_weights/LOLv2_Real.h5 \
       --epochs 200
```

### Step 5：無參考（unpaired）真實夜間微調（選配，階段 C）

對於 TN-TLD / LISA night / CNTSSS 這類無 GT 的真實夜間資料，採 **Zero-DCE 風格零參考 loss**：

```python
# scripts/zero_ref_losses.py
def exposure_loss(pred, E=0.6, patch=16):
    # 讓輸出亮度區域平均亮度接近 E
    pred = (pred + 1.0)/2.0
    avg  = tf.nn.avg_pool2d(tf.reduce_mean(pred, -1, keepdims=True),
                            ksize=patch, strides=patch, padding='VALID')
    return tf.reduce_mean(tf.abs(avg - E))

def tv_loss(pred):
    return tf.reduce_mean(tf.abs(pred[:,1:] - pred[:,:-1])) + \
           tf.reduce_mean(tf.abs(pred[:,:,1:] - pred[:,:,:-1]))

def color_const_loss(pred):
    pred = (pred + 1.0)/2.0
    r, g, b = tf.split(pred, 3, axis=-1)
    return (tf.reduce_mean((tf.reduce_mean(r,[1,2]) - tf.reduce_mean(g,[1,2]))**2)
          + tf.reduce_mean((tf.reduce_mean(r,[1,2]) - tf.reduce_mean(b,[1,2]))**2)
          + tf.reduce_mean((tf.reduce_mean(g,[1,2]) - tf.reduce_mean(b,[1,2]))**2))
```

### Step 6：Detector-aware Joint Fine-tune（選配，進階）

若有 YOLO 權重可用，將 LYT-Net 串到 detector 前，固定 detector、用 detection loss 回傳梯度到 LYT-Net（做法同 En-YOLO / FE-YOLO）。虛擬碼：

```python
lyt = build_and_load('./experiments/traffic_synth_v1/best.h5')
yolo = tf.keras.models.load_model('./yolov5_night_tl.h5'); yolo.trainable=False
def det_aware_step(raw_night, bboxes, cls):
    with tf.GradientTape() as tape:
        enh = (lyt(raw_night) + 1.0)/2.0
        pred = yolo(enh)                   # yolo 須可微
        det_l = yolo_loss(pred, bboxes, cls)
    grads = tape.gradient(det_l, lyt.trainable_variables)
    opt.apply_gradients(zip(grads, lyt.trainable_variables))
```

---

## 6. 評估指標與驗證策略

### 6.1 畫質層（LLIE metric）

| 指標 | 測試資料 | 說明 |
|------|----------|------|
| **PSNR / SSIM** | 合成配對 test set | 標準監督指標 |
| **GT-Mean PSNR** | 合成配對 test set | `find_gamma.py` 已提供；抵銷亮度 bias |
| **LPIPS (AlexNet)** | 合成配對 test set | 感知相似度；`scripts/test.py` 已實作 |
| **NIQE / BRISQUE** | 真實夜間 unpaired（TN-TLD, CNTSSS） | No-reference，數值越低越好 |
| **Color Fidelity (CIEDE2000)** | 號誌 bbox 區域（需 bbox） | 檢查紅/綠保真度 —— 本題核心關鍵 |

### 6.2 下游任務層（detection metric）

以相同 YOLO 權重，比較「raw 夜間圖 vs. LYT-Net 增強後圖」：

- **mAP@0.5 / mAP@0.5:0.95**（TN-TLD、CNTSSS、BDD100K-night）
- **Recall @ small object (< 32 px)** —— 夜間號誌多為小物件
- **Per-class AP**：Red / Yellow / Green 分開看，驗證色彩保真度是否轉化為偵測收益

### 6.3 推論速度

```python
import time, numpy as np, tensorflow as tf
x = tf.random.uniform([1, 720, 1280, 3])
for _ in range(5): model(x)   # warmup
t0 = time.time()
for _ in range(50): model(x)
print('FPS:', 50/(time.time()-t0))
```

目標：**Jetson Xavier NX 上 ≥ 30 FPS @ 720p** 才算符合「即時前處理」。

### 6.4 驗證實驗矩陣（ablation）

| 實驗 | Pretrained | 資料 | 凍結策略 | Loss 版本 | 預期結論 |
|------|------------|------|----------|-----------|----------|
| A0 (baseline) | LOLv1 | 無微調 | — | 原始 | 確認未微調基線 |
| B1 | LOLv2_Real | Synth only | freeze CWD 30 ep | traffic_loss | 主要訓練 |
| B2 | LOLv2_Synthetic | Synth only | 同上 | 同上 | 起點對照 |
| B3 | LOLv2_Real | Synth + Zero-ref | unfreeze 全網 | traffic + zero-ref | 最佳 |
| C1 | 以上最佳 | + YOLO joint | — | + det-loss | 極佳但成本高 |

---

## 7. 時程建議（6 週）

| 週次 | 任務 |
|------|------|
| W1 | 下載 LISA / BSTLD / S2TLD，跑通原 LYT-Net `test.py` 驗證環境 |
| W2 | 合成「日→夜」資料（§Step 2），訓練 B2 baseline |
| W3 | 調整 loss 權重與凍結策略（§Step 4），完成 B1、B2 對照 |
| W4 | 加入 Zero-ref 微調（§Step 5），產出 B3 |
| W5 | 串接 YOLO 做 detection mAP 評估（§6.2） |
| W6 | 撰寫實驗報告、優化推論速度（TFLite / TensorRT） |

---

## 8. 延伸方向

1. **多曝光融合**：借鑒 ModalFormer，將短曝/長曝雙輸入送入 LYT-Net
2. **知識蒸餾**：以 Retinexformer（大模型）為 teacher、LYT-Net 為 student
3. **領域自適應**：加上 `Cityscapes → DarkZurich` 風格的 domain adaptation loss
4. **不確定性輸出**：讓 LYT-Net 額外輸出 uncertainty map 供下游 detector 做 soft gating
5. **TFLite / TensorRT 部署**：因 model 僅 45K params，量化到 INT8 幾乎無損

---

## 9. 參考資源（Sources）

### 程式碼與模型
- [LYT-Net GitHub（官方）](https://github.com/albrateanu/LYT-Net)
- [LYT-Net HuggingFace](https://huggingface.co/albrateanu/LYT-Net)
- [LYT-Net arXiv](https://arxiv.org/abs/2401.15204) ｜ [IEEE SPL 2025](https://ieeexplore.ieee.org/document/10972228)
- [YOLO-LLTS GitHub](https://github.com/linzy88/YOLO-LLTS) ｜ [YOLO-LLTS arXiv](https://arxiv.org/abs/2503.13883)
- [Retinexformer GitHub](https://github.com/caiyuanhao1998/Retinexformer)
- [Awesome LLIE 資源整理](https://github.com/zhihongz/awesome-low-light-image-enhancement)

### 交通號誌資料集
- [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset) ｜ [Dataset Ninja 說明](https://datasetninja.com/lisa-traffic-light)
- [Bosch Small Traffic Lights (BSTLD)](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)
- [S2TLD GitHub](https://github.com/Thinklab-SJTU/S2TLD)
- [DriveU Traffic Light (DTLD)](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/) ｜ [ICRA 2018 論文](https://ieeexplore.ieee.org/document/8460737)
- [LNT-YOLO 論文（含 TN-TLD）](https://www.mdpi.com/2624-6511/8/3/95)

### 夜間駕駛 / LLIE Benchmark
- [ACDC 資料集](https://arxiv.org/html/2104.13395v5)
- [Dark Zurich / Nighttime Datasets 清單](https://github.com/aasharma90/NightTime_Datasets)
- [ExDark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- [LoLI-Street (ACCV 2024)](https://openaccess.thecvf.com/content/ACCV2024/papers/Islam_LoLI-Street_Benchmarking_Low-light_Image_Enhancement_and_Beyond_ACCV_2024_paper.pdf)

### 相關 2025 年論文
- [En-YOLO (Springer 2025)](https://link.springer.com/article/10.1007/s00530-025-01820-7)
- [FE-YOLO (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/abs/pii/S105120042500377X)
- [WTEFNet (ADAS, arXiv 2025)](https://arxiv.org/html/2505.23201v2)
- [Low-Light Image & Video Enhancement Survey (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12027663/)
- [Physics-based Noise Modeling (Wei et al.)](https://ar5iv.labs.arxiv.org/html/2108.02158)
- [Noise Synthesis with Diffusion (2025)](https://arxiv.org/html/2503.11262v1)

### 遷移學習資源
- [TensorFlow 官方 Transfer Learning 指南](https://www.tensorflow.org/guide/keras/transfer_learning)
- [Keras Transfer Learning & Fine-tuning](https://keras.io/guides/transfer_learning/)

---

*本提案以 LYT-Net TensorFlow 原始碼（`arch.py` / `losses.py` / `train.py` / `data_loading.py`）為分析依據，結合 2024–2025 年 LLIE 與夜間交通號誌偵測最新文獻，提出一條可落地的遷移訓練路徑。如需進一步細分某步驟（例如 detector-aware joint training 的詳細程式、或是 LISA 夜間子集自動篩選腳本），可以再展開。*
