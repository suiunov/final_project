# Results 章節數據總結

> 產生時間：2026-05-14
> 評估協議：標準直接評估（無 GT-mean gamma 調整），兩模型使用相同協議，比較具內部一致性。
> 此檔記錄所有實驗數據與圖片路徑，由使用者手動貼回 `results.tex`。

---

## 1. Table I (`tab:cross_dataset`) — 量化結果

| Method | LOL-v1 PSNR↑ | LOL-v1 SSIM↑ | LOL-v1 LPIPS↓ | LOL-v2-Real PSNR↑ | LOL-v2-Real SSIM↑ | LOL-v2-Real LPIPS↓ | LOLI-Street PSNR↑ | LOLI-Street SSIM↑ | LOLI-Street LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| LYT-Net (baseline) | 22.38 | 0.826 | 0.076 | 22.25 | 0.854 | 0.082 | 14.64 | 0.774 | 0.106 |
| **Ours** | 22.20 | **0.831** | 0.077 | **23.11** | **0.860** | **0.069** | **17.22** | **0.842** | **0.080** |

### LaTeX 替換用片段（直接複製貼到 results.tex 的 tabular 內）

```latex
LYT-Net (baseline) & 22.38 & 0.826 & 0.076
                   & 22.25 & 0.854 & 0.082
                   & 14.64 & 0.774 & 0.106 \\
Ours               & 22.20 & \textbf{0.831} & 0.077
                   & \textbf{23.11} & \textbf{0.860} & \textbf{0.069}
                   & \textbf{17.22} & \textbf{0.842} & \textbf{0.080} \\
```

> **注意**：LOLv1 的 Ours PSNR 略低（-0.18 dB），但 SSIM 更高（+0.005）。
> LOLv2-Real 與 LOLI-Street 兩個 cross-domain 資料集 Ours 全面優勝，
> 尤其 LOLI-Street **+2.58 dB PSNR / +0.068 SSIM / -0.027 LPIPS**，是論文的核心貢獻。

---

## 2. 段落中的 delta 數字

### In-Domain (LOL-v1)
- Ours: PSNR=22.20, SSIM=0.831, LPIPS=0.077
- Baseline: PSNR=22.38, SSIM=0.826, LPIPS=0.076
- ΔPSNR = **−0.18 dB**（略低，可用 "comparable" 描述）
- ΔSSIM = **+0.005**
- ΔLPIPS = **+0.001**（幾乎相同）

### LOL-v2-Real delta (Ours − Baseline)
- ΔPSNR = **+0.86 dB**
- ΔSSIM = **+0.006**
- ΔLPIPS = **−0.013**（LPIPS 越低越好）

### LOLI-Street delta (Ours − Baseline)
- ΔPSNR = **+2.58 dB**
- ΔSSIM = **+0.068**
- ΔLPIPS = **−0.027**

---

## 3. Figure 1：`fig:qualitative` 定性比較圖

**檔案實際路徑**：`D:\人工智慧課程 1142\LYT-Net-main\docs\figures\qualitative_traffic_sign.pdf`

**選用的 3 個場景（均來自 LOLI-Street Val 集）**：
1. 場景 1（紅色標誌）：來源 `dense_30001.jpg`
2. 場景 2（藍色標誌）：來源 `dense_30401.jpg`
3. 場景 3（Ours 優勝場景）：來源 `light_30801.jpg`

**LaTeX 替換**：把 `results.tex` 中 `fig:qualitative` 那段的佔位 `\rule`/`\textit` 區塊改為：
```latex
\includegraphics[width=\linewidth]{figures/qualitative_traffic_sign.pdf}
```

---

## 4. Figure 2：`fig:failure` 失敗案例圖

**檔案實際路徑**：`D:\人工智慧課程 1142\LYT-Net-main\docs\figures\failure_cases.pdf`

**選用的 2 個失敗案例**：
- (a) 過暗失色：來源 `light_30772.jpg`，Ours PSNR=9.90 dB（最低）
- (b) 強光 bloom：來源 `light_30044.jpg`，Ours PSNR=10.96 dB

**LaTeX 替換**：把 `results.tex` 中 `fig:failure` 那段的佔位改為：
```latex
\includegraphics[width=\linewidth]{figures/failure_cases.pdf}
```

---

## 5. 完整檔案清單（給使用者）

**需要複製到主論文目錄的檔案**（建議放到 `D:\人工智慧課程 1142\人工智慧課程 1142\figures\`）：

- [ ] `docs/figures/qualitative_traffic_sign.pdf` → `figures/qualitative_traffic_sign.pdf`
- [ ] `docs/figures/failure_cases.pdf` → `figures/failure_cases.pdf`

**需要在 `results.tex` 中手動修改的位置**：

- [ ] tabular 內的數字 × 18（見第 1 節 LaTeX 片段）
- [ ] `\paragraph{In-Domain Performance}` 內的數字（見第 2 節 In-Domain）
- [ ] `\paragraph{Cross-Dataset Generalization}` 內的 delta 數字（見第 2 節 delta）
- [ ] `fig:qualitative` 的 `\rule` 佔位 → `\includegraphics`
- [ ] `fig:failure` 的 `\rule` 佔位 → `\includegraphics`

---

## 6. 原始 CSV 路徑（給開發者驗證用）

- `D:\人工智慧課程 1142\LYT-Net-main\TensorFlow\experiments\eval_original.csv`
- `D:\人工智慧課程 1142\LYT-Net-main\TensorFlow\experiments\eval_modified.csv`
- `D:\人工智慧課程 1142\LYT-Net-main\TensorFlow\experiments\per_image_LOLI-Street.csv`
