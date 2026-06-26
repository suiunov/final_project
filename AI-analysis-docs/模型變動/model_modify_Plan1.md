# LYT-Net 模型改進方案

## 現況分析

| 指標 | 目前訓練結果 (1000 epochs) | 預訓練權重 (with gtmean) |
|------|---------------------------|--------------------------|
| PSNR | ~21.7 | 26.26 |
| GT-PSNR | ~25.7 | — |
| SSIM | ~0.850 | 0.845 |
| LPIPS | — | 0.083 |

> [!NOTE]
> 目標：做出「能解釋」的小幅改進，不是大幅改架構。每個改動都要能在報告中說明「為什麼這樣改、改了什麼效果」。

---

## 問題診斷

從訓練 log 可以觀察到：
1. **GT-PSNR 在 ~25.7 飽和**，SSIM 在 0.850 不再提升 → 模型表達力可能不足
2. **`_create_processing_layers` 只有 1 層 Conv2D** → 每個 YUV 通道只用 1 層 3×3 卷積提取特徵，感受野太小
3. **亮度注入比例 `0.2` 是硬編碼** → 不同圖片可能需要不同程度的亮度引導
4. **Denoiser 沒有正則化** → 容易在色度通道過擬合

---

## 提出的改進（共 3 項）

### 改進 1：加深 YUV 通道的特徵提取層

**目前**：每個通道 (Y, Cb, Cr) 只有 **1 層** Conv2D → 感受野 3×3

**改為**：每個通道用 **2 層** Conv2D → 感受野擴大到 5×5

```python
# 原本
def _create_processing_layers(self, filters):
    layerz = [layers.Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(1)]
    return keras.Sequential(layerz)

# 改為
def _create_processing_layers(self, filters):
    layerz = [layers.Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(2)]
    return keras.Sequential(layerz)
```

**解釋**：
- 更多的卷積層可以捕捉更大範圍的空間特徵，等效感受野從 3×3 擴大到 5×5
- 參數量增加很少（只多了 3 個 3×3×32×32 的卷積核 ≈ 27.6K 參數）
- 這是最常見且有效的「加深網路」改進策略

---

### 改進 2：Denoiser 加入 Dropout 正則化

**目前**：Denoiser 的 encoder-decoder 沒有任何正則化

**改為**：在 bottleneck 之後加入 Dropout(0.1)

```python
# 在 Denoiser.__init__ 中加入
self.dropout = layers.Dropout(0.1)

# 在 Denoiser.call 中，bottleneck 之後
x = self.bottleneck(x4)
x = self.dropout(x)  # 新增
```

**解釋**：
- Dropout 在訓練時隨機丟棄 10% 的神經元，迫使網路學習更魯棒的特徵
- 放在 bottleneck（最低解析度）後面，不會影響 skip connection 的細節傳遞
- 0.1 的比率很保守，不會大幅影響收斂速度

---

### 改進 3：可學習的亮度注入比例

**目前**：`ref = ref + 0.2 * self.lum_conv(lum)` 中的 `0.2` 是硬編碼常數

**改為**：用一個可學習的參數取代，初始值設為 0.2

```python
# 在 LYT.__init__ 中
self.lum_weight = tf.Variable(0.2, trainable=True, name='lum_injection_weight')

# 在 LYT.call 中
ref = ref + self.lum_weight * self.lum_conv(lum)
```

**解釋**：
- 讓模型自動學習最佳的亮度-色度融合比例
- 初始值保持 0.2 確保訓練初期行為不變
- 這是一個極小的改動（只增加 1 個可訓練參數），但允許模型自適應地調整融合強度

---

## Proposed Changes

### Model Architecture

#### [MODIFY] [arch.py](file:///d:/人工智慧課程%201142/LYT-Net-main/TensorFlow/model/arch.py)
1. `_create_processing_layers`: `range(1)` → `range(2)`（加深特徵提取）
2. `Denoiser.__init__`: 加入 `self.dropout = layers.Dropout(0.1)`
3. `Denoiser.call`: bottleneck 後加入 `x = self.dropout(x)`
4. `LYT.__init__`: 加入 `self.lum_weight = tf.Variable(0.2, ...)`
5. `LYT.call`: `0.2` → `self.lum_weight`

> [!IMPORTANT]
> `model_modify/arch.py`（無註解版本）也要同步修改，因為 `train.py` import 的是 `from model.arch import ...`（有註解版本），但若你之後要用 `model_modify` 的版本也需要一致。

### 不修改的檔案
- **losses.py** — 損失函數不動，保持可比較性
- **scheduler.py** — 學習率排程不動
- **train.py** — 訓練腳本不動
- **data_loading.py** — 資料載入不動

---

## 改動量統計

| 項目 | 新增參數量 | 改動行數 |
|------|-----------|---------|
| 加深特徵提取 (×3 通道) | ~27,648 | 1 行 |
| Dropout | 0 | 2 行 |
| 可學習亮度比例 | 1 | 2 行 |
| **合計** | ~27,649 | ~5 行 |

> [!TIP]
> 原始 LYT-Net 的總參數量約 270K，新增 ~28K 參數只增加 ~10%，是非常輕量的改動。

---

## Open Questions

> [!IMPORTANT]
> 1. **是否要同步修改 `model_modify/arch.py`？** 目前 `train.py` 用的是 `from model.arch import ...`，如果你未來可能切換到 `model_modify` 版本，需要同步。
> 2. **訓練多少 epochs？** 建議先跑 300-500 epochs 看趨勢，如果有提升再跑滿 1000。
> 3. **是否需要從預訓練權重 fine-tune？** 還是從頭開始訓練？由於架構有改動，必須從頭訓練。

---

## Verification Plan

### 自動測試
```bash
python main.py --train --dataset LOLv1    # 訓練
python main.py --test --dataset LOLv1 --weights <best_weights.h5> --gtmean    # 測試
```

### 比較基準
- 原始模型 (gtmean): **PSNR 26.26, SSIM 0.845, LPIPS 0.083**
- 目標：PSNR > 26.3 或 SSIM > 0.850 即算有改進
