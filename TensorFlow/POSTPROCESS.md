# Post-Processing Module 後處理模組說明

## Overview 概述

本模組在 LYT-Net 模型輸出之後，額外施加兩階段影像後處理，以進一步提升增強品質。
後處理**不影響模型架構和訓練**，僅在推論/測試階段使用。

## Pipeline 流程

```
Model Output (RGB, [0,1])
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Stage 1: CLAHE (Y channel only)    │
   │  ─ Adaptive local contrast boost    │
   │  ─ Only affects luminance           │
   │  ─ Preserves color fidelity         │
   └─────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │  Stage 2: Guided Filter (full RGB)  │
   │  ─ Edge-preserving smoothing        │
   │  ─ Removes residual noise           │
   │  ─ Self-guided (uses output itself) │
   └─────────────────────────────────────┘
        │
        ▼
   Final Output (RGB, [0,1])
```

## Stage 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)

### What it does
對影像的 **Y（亮度）通道** 進行局部自適應直方圖均衡化。

### Why only Y channel?
- 與 LYT-Net 的 YUV 分離設計理念一致
- 只增強亮度對比度，**不改變色彩**，避免色偏
- 低光場景中，暗部細節主要損失在亮度通道

### Parameters
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `clip_limit` | 1.5 | 對比度限制閾值，越大增強越強（建議 1.0~2.0） |
| `tile_grid_size` | (8, 8) | 局部區域的網格大小 |

### Why clip_limit = 1.5?
模型已經做過亮度增強，CLAHE 只是微調。太大的 `clip_limit` 會導致過曝。

### Implementation
```python
ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
y, cr, cb = cv2.split(ycrcb)
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
y_enhanced = clahe.apply(y)
# Merge back and convert to RGB
```

---

## Stage 2: Guided Filter (Edge-Preserving Smoothing)

### What it does
使用影像自身作為引導圖，進行**邊緣保留平滑**。

### Why after CLAHE?
- CLAHE 增強對比度時可能放大殘留雜訊
- Guided Filter 可以平滑這些雜訊，同時保持邊緣銳利
- 「先增強、再清理」是合理的處理順序

### Parameters
| 參數 | 預設值 | 說明 |
|------|--------|------|
| `radius` | 4 | 濾波視窗半徑，越大平滑越強（建議 2~8） |
| `eps` | 0.01 | 正則化參數，越小越銳利（建議 0.001~0.1） |

### How it works
Guided Filter 在每個局部視窗中，擬合一個線性模型：
```
output(i) = a * guide(i) + b
```
- 在平坦區域：`a ≈ 0`，輸出 ≈ 常數 → 平滑效果
- 在邊緣區域：`a ≈ 1`，輸出 ≈ 原始值 → 邊緣保留

### Implementation
```python
filtered = cv2.ximgproc.guidedFilter(
    guide=image,  # Self-guided
    src=image,
    radius=4,
    eps=0.01
)
```

---

## Usage 使用方式

### Testing (test.py)
```bash
# 無後處理
python main.py --test --dataset LOLv1 --weights <weights.h5> --gtmean

# 有後處理
python main.py --test --dataset LOLv1 --weights <weights.h5> --gtmean --postprocess
```

### Inference (infer.py)
```bash
# 無後處理
python scripts/infer.py --weights <weights.h5> --image <input.jpg>

# 有後處理
python scripts/infer.py --weights <weights.h5> --image <input.jpg> --postprocess
```

---

## File Structure 檔案結構

```
scripts/
├── postprocess.py     # 後處理模組（CLAHE + Guided Filter）
├── test.py            # 測試腳本（已加入 --postprocess flag）
├── infer.py           # 推論腳本（已加入 --postprocess flag）
└── ...
```

---

## Dependencies 依賴

```
opencv-contrib-python   # 包含 cv2.ximgproc.guidedFilter
```

安裝：
```bash
pip install opencv-contrib-python
```

---

## References 參考文獻

1. **CLAHE**: Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." *Graphics Gems IV*, 1994.
2. **Guided Filter**: He, K., Sun, J., Tang, X. "Guided Image Filtering." *IEEE TPAMI*, 2013.
