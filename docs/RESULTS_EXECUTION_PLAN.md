# Results 章節實驗執行計畫 (for Claude Code)

> **目的**：為 IEEE 論文 "Road sign recognition in low-light conditions" 的 Results 章節產生所有量化數據與視覺化圖片。
>
> **本計畫位置**：`D:\人工智慧課程 1142\LYT-Net-main\docs\RESULTS_EXECUTION_PLAN.md`
>
> **論文檔位置（唯讀，不可修改）**：`D:\人工智慧課程 1142\人工智慧課程 1142\results.tex`
>
> **程式碼根目錄**：`D:\人工智慧課程 1142\LYT-Net-main\TensorFlow`

---

## 標準資料夾結構

執行完本計畫後，整體結構如下：

```
D:\人工智慧課程 1142\LYT-Net-main\
├── TensorFlow\                          [程式碼與訓練/評估]
│   ├── model\                           原版 LYT-Net 架構
│   ├── model_modify\                    改良版 (D1/D2/D3) 架構
│   ├── scripts\
│   │   ├── eval_all.py                  [新增] 統一評估腳本（雙架構 + 3 資料集）
│   │   ├── compose_qualitative.py       [新增] 組合 3 場景 → 1 張 PDF
│   │   ├── compose_failure.py           [新增] 組合失敗案例 → 1 張 PDF
│   │   ├── gen_comparison.py            [可能需 patch 支援 modified arch]
│   │   ├── train.py / test.py / ...     既有腳本
│   │   └── *.bak                        修改前的備份
│   ├── data\                            LOLv1, LOLv2 資料集
│   ├── pretrained_weights\              LOLv1.h5 等原作者權重
│   └── experiments\                     [評估產出]
│       ├── LOLv1\                       Ours 訓練好的 .h5
│       ├── eval_original.csv            baseline 三資料集指標
│       ├── eval_modified.csv            Ours 三資料集指標
│       └── per_image_LOLI-Street.csv    每張圖的 PSNR（用於挑失敗案例）
│
├── dataset\
│   └── LoLI-Street Dataset\             外部資料集
│       ├── Val\low\*.jpg
│       └── Val\high\*.jpg
│
└── docs\                                [本計畫 + 文件交付]
    ├── paper-master-baseline.md         （既有，由 paper-master skill 產生）
    ├── RESULTS_EXECUTION_PLAN.md        本檔
    ├── RESULTS_DATA.md                  [Phase 6 產出] 給使用者貼回 results.tex 用
    ├── figures\                         [最終要插入論文的圖]
    │   ├── qualitative_traffic_sign.pdf
    │   ├── failure_cases.pdf
    │   └── sources\                     挑選圖片的原檔備份
    │       ├── scene1_red_sign.jpg
    │       ├── scene2_blue_sign.jpg
    │       ├── scene3_text_sign.jpg
    │       ├── failure_a_dark.jpg
    │       └── failure_b_glare.jpg
    └── intermediate\                    [候選圖，方便回頭挑]
        ├── qualitative_all\             30 張候選對比圖
        └── failure_candidates\          PSNR 最低的 10 張對比圖
```

> ⚠️ 論文檔（`D:\人工智慧課程 1142\人工智慧課程 1142\` 底下的所有 `.tex` / `.txt`）**完全不要動**。

---

## 任務總覽

最終要交付 18 個量化數字 + 2 張視覺化圖片，全部寫入 `docs/RESULTS_DATA.md`：

**量化數據 (Table I `tab:cross_dataset`)**：

| 模型 | LOL-v1 | LOL-v2-Real | LOLI-Street |
|---|---|---|---|
| LYT-Net (baseline，用 `model/arch.py` + `LOLv1.h5`) | PSNR/SSIM/LPIPS | PSNR/SSIM/LPIPS | PSNR/SSIM/LPIPS |
| Ours (用 `model_modify/arch.py` + 自訓 LOLv1 權重) | PSNR/SSIM/LPIPS | PSNR/SSIM/LPIPS | PSNR/SSIM/LPIPS |

**圖片**：
- `docs/figures/qualitative_traffic_sign.pdf` — 3 個含交通標誌場景 × 4 欄並排 (Input | LYT-Net | Ours | GT)
- `docs/figures/failure_cases.pdf` — 2 個失敗案例 (a) 過暗失色 + (b) 強光 bloom

---

## Phase 0：環境檢查（必做）

執行下列檢查並輸出狀態：

```bash
cd "D:\人工智慧課程 1142\LYT-Net-main\TensorFlow"
conda activate LYTNet   # 或專案使用的 env，若失敗則回報
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
python -c "import torch, lpips; print('torch:', torch.__version__); print('lpips OK')"
```

**逐項確認下列檔案是否存在：**

- [ ] `TensorFlow/model/arch.py` — 原版 LYT-Net 架構
- [ ] `TensorFlow/model_modify/arch.py` — 改良版 (D1/D2/D3) 架構
- [ ] `TensorFlow/pretrained_weights/LOLv1.h5` — 原作者 baseline 權重
- [ ] `TensorFlow/data/LOLv1/Test/input/*.png` (15 張)
- [ ] `TensorFlow/data/LOLv1/Test/target/*.png` (15 張)
- [ ] `TensorFlow/data/LOLv2/Real_captured/Test/Low/*.png` (100 張) — **若不存在，需告知使用者下載**
- [ ] `TensorFlow/data/LOLv2/Real_captured/Test/Normal/*.png`
- [ ] `dataset/LoLI-Street Dataset/Val/low/*.jpg` — **若不存在，需告知使用者下載**
- [ ] `dataset/LoLI-Street Dataset/Val/high/*.jpg`

**確認 docs 結構：**

```bash
mkdir -p "D:\人工智慧課程 1142\LYT-Net-main\docs\figures\sources"
mkdir -p "D:\人工智慧課程 1142\LYT-Net-main\docs\intermediate\qualitative_all"
mkdir -p "D:\人工智慧課程 1142\LYT-Net-main\docs\intermediate\failure_candidates"
```

若任一資料集不存在，**停止執行**並回報缺失項目，請使用者先補齊。

---

## Phase 1：確認/取得「Ours」模型的 LOL-v1 訓練權重

`compare_models.py` 預設讀 `./experiments/loli_scratch_v1/best.h5` (LOLI-Street 訓練的)，但本論文 Results 章節需要的是 **LOL-v1 訓練的 Ours 權重**。

**步驟：**

1. 檢查 `TensorFlow/experiments/LOLv1/` 底下是否已有訓練好的權重 (`net_psnr_*.h5`)
2. 若有：選 PSNR 最高的那個檔案，記下路徑為 `OURS_WEIGHTS_LOLV1`
3. 若無：執行訓練（74 epochs 與論文一致；目前 `train.py` 寫死 1000 epoch，需臨時改成 74 或設定 early-stop）
   ```bash
   # 將 train.py L153 epochs 暫改為 74，或加 --epochs 參數
   # 先備份: cp scripts/train.py scripts/train.py.bak
   python scripts/train.py --dataset LOLv1
   ```
   訓練完成後從 `TensorFlow/experiments/LOLv1/` 找最佳權重。

**回報**：`OURS_WEIGHTS_LOLV1 = TensorFlow/experiments/LOLv1/<filename>.h5`

---

## Phase 2：新增統一評估腳本

目前 `scripts/test.py` 只支援 LOLv1 / LOLv2_Real / LOLv2_Synthetic 且只能用 `model_modify`。需擴充以支援雙架構與 LOLI-Street。

**建立 `TensorFlow/scripts/eval_all.py`**（新檔案）：

```python
"""
統一評估腳本：支援 model.arch (原版 LYT-Net) 與 model_modify.arch (改良版 Ours)
對 LOLv1 / LOLv2_Real / LOLI-Street 三個資料集分別計算 PSNR / SSIM / LPIPS。

Usage:
    python scripts/eval_all.py --arch original --weights pretrained_weights/LOLv1.h5
    python scripts/eval_all.py --arch modified --weights experiments/LOLv1/best.h5
    python scripts/eval_all.py --arch modified --weights ... --per_image_log
"""
import os, sys, glob, argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import torch, lpips
import data_loading as dl

DATASETS = {
    'LOLv1': {
        'low': './data/LOLv1/Test/input/*.png',
        'high': './data/LOLv1/Test/target/*.png',
        'loader': 'lol',
    },
    'LOLv2_Real': {
        'low': './data/LOLv2/Real_captured/Test/Low/*.png',
        'high': './data/LOLv2/Real_captured/Test/Normal/*.png',
        'loader': 'lol',
    },
    'LOLI-Street': {
        'low': '../dataset/LoLI-Street Dataset/Val/low/*.jpg',
        'high': '../dataset/LoLI-Street Dataset/Val/high/*.jpg',
        'loader': 'loli',
    },
}

def build(arch):
    if arch == 'original':
        from model.arch import LYT, Denoiser
    else:
        from model_modify.arch import LYT, Denoiser
    d_cb, d_cr = Denoiser(16), Denoiser(16)
    d_cb.build(input_shape=(None, None, None, 1))
    d_cr.build(input_shape=(None, None, None, 1))
    m = LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m.build(input_shape=(None, None, None, 3))
    return m

def eval_one(model, ds_cfg, lpips_model, per_image_log=False, ds_name=''):
    if ds_cfg['loader'] == 'loli':
        test_ds = dl.get_loli_datasets_metrics(ds_cfg['low'], ds_cfg['high'], crop_margin=0)
        file_list = sorted(glob.glob(ds_cfg['low']))
    else:
        test_ds = dl.get_datasets_metrics(ds_cfg['low'], ds_cfg['high'], 0)
        file_list = sorted(glob.glob(ds_cfg['low']))

    psnrs, ssims, lpipss = [], [], []
    per_image = []
    for i, (raw, gt) in enumerate(test_ds):
        pred = model(raw)
        pred = tf.clip_by_value((pred + 1.0) / 2.0, 0, 1)
        gt01 = (gt + 1.0) / 2.0
        p = float(tf.reduce_mean(tf.image.psnr(gt01, pred, max_val=1.0)))
        s = float(tf.reduce_mean(tf.image.ssim(gt01, pred, max_val=1.0)))
        p_pt = torch.from_numpy(pred.numpy()).permute(0, 3, 1, 2).float()
        t_pt = torch.from_numpy(gt01.numpy()).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            l = lpips_model(p_pt, t_pt).item()
        psnrs.append(p); ssims.append(s); lpipss.append(l)
        if per_image_log:
            fname = os.path.basename(file_list[i]) if i < len(file_list) else f'img_{i}'
            per_image.append((fname, p, s, l))

    if per_image_log:
        out_csv = f'./experiments/per_image_{ds_name}.csv'
        with open(out_csv, 'w') as f:
            f.write('filename,psnr,ssim,lpips\n')
            for fname, p, s, l in per_image:
                f.write(f'{fname},{p:.4f},{s:.4f},{l:.4f}\n')
        print(f'  → per-image log saved: {out_csv}')

    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['original', 'modified'], required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()))
    p.add_argument('--per_image_log', action='store_true',
                   help='Also save per-image PSNR/SSIM/LPIPS CSV (useful for failure-case selection)')
    args = p.parse_args()

    model = build(args.arch)
    model.load_weights(args.weights)
    import sys as _s
    orig = _s.stdout; _s.stdout = open(os.devnull, 'w')
    lpips_model = lpips.LPIPS(net='alex'); _s.stdout = orig

    print(f"\n{'='*60}\nArch: {args.arch}  |  Weights: {args.weights}\n{'='*60}")
    print(f"{'Dataset':18s} | {'PSNR':>8s} | {'SSIM':>8s} | {'LPIPS':>8s}")
    print('-' * 55)
    results = {}
    for name in args.datasets:
        try:
            psnr, ssim, lp = eval_one(model, DATASETS[name], lpips_model,
                                       per_image_log=args.per_image_log, ds_name=name)
            print(f"{name:18s} | {psnr:8.2f} | {ssim:8.4f} | {lp:8.4f}")
            results[name] = (psnr, ssim, lp)
        except Exception as e:
            print(f"{name:18s} | FAILED: {e}")

    os.makedirs('./experiments', exist_ok=True)
    out_csv = f'./experiments/eval_{args.arch}.csv'
    with open(out_csv, 'w') as f:
        f.write('dataset,psnr,ssim,lpips\n')
        for name, (p, s, l) in results.items():
            f.write(f'{name},{p:.4f},{s:.4f},{l:.4f}\n')
    print(f"\nSaved → {out_csv}")
```

**檢查 `TensorFlow/data_loading.py`** 是否已有 `get_loli_datasets_metrics`，若無則需新增（參考 `get_datasets_metrics` 但讀 `.jpg`）。

---

## Phase 3：跑出 18 個量化數字

```bash
cd "D:\人工智慧課程 1142\LYT-Net-main\TensorFlow"
conda activate LYTNet

# Baseline（LYT-Net 原版 + 官方 LOLv1 預訓練權重）
python scripts/eval_all.py --arch original --weights pretrained_weights/LOLv1.h5

# Ours（改良版 + 自訓 LOLv1 權重）
python scripts/eval_all.py --arch modified --weights <OURS_WEIGHTS_LOLV1>

# 額外：產出 Ours 對 LOLI-Street 的 per-image log，供 Phase 5 挑失敗案例
python scripts/eval_all.py --arch modified --weights <OURS_WEIGHTS_LOLV1> \
    --datasets LOLI-Street --per_image_log
```

預期產出：
- `TensorFlow/experiments/eval_original.csv` — baseline 三組數字
- `TensorFlow/experiments/eval_modified.csv` — Ours 三組數字
- `TensorFlow/experiments/per_image_LOLI-Street.csv` — 每張圖的指標

**例外處理：**
- 若 LOLv2_Real 或 LOLI-Street 評估失敗，**不要中斷整個流程**，把那一格標 N/A 並繼續。
- LPIPS 初次執行會下載 AlexNet 權重 (~5MB)，需要網路連線。

---

## Phase 4：產生定性比較圖 (Figure 1)

**目標**：3 個含交通標誌場景 × 4 欄 (Input | LYT-Net | Ours | GT) → 一張 `docs/figures/qualitative_traffic_sign.pdf`

**步驟：**

1. **修改 `TensorFlow/scripts/gen_comparison.py`** 使其支援雙架構：
   - 先備份：`cp scripts/gen_comparison.py scripts/gen_comparison.py.bak`
   - 在 `build_model()` 加 `arch` 參數
   - CLI 加 `--ours_arch modified --orig_arch original`

2. **從 LOLI-Street Val 集挑選含交通標誌的圖片**：
   ```bash
   cd "D:\人工智慧課程 1142\LYT-Net-main\TensorFlow"
   python scripts/gen_comparison.py \
       --dataset LOLI-Street \
       --ours <OURS_WEIGHTS_LOLV1> --ours_arch modified \
       --orig pretrained_weights/LOLv1.h5 --orig_arch original \
       --out "../docs/intermediate/qualitative_all" \
       --num 30
   ```

3. **挑選標準**（從 30 張中選 3 張）：
   - 場景 1：含明顯**紅色禁止標誌**（圓形紅邊白底）
   - 場景 2：含明顯**藍色指示標誌**（圓形或方形藍底）
   - 場景 3：**含文字或符號**的標誌（讓邊緣保留差異可見）
   - 視覺差異要肉眼可見（Ours 明顯比 LYT-Net 好）
   - 把選用的 3 張原始圖檔複製到 `docs/figures/sources/scene{1,2,3}_*.jpg`

4. **組合 3 張 → 一張 3 列 × 4 欄的 PDF**，建立 `TensorFlow/scripts/compose_qualitative.py`：
   ```python
   """組合 3 個 4 欄場景對比圖 → 單張 PDF"""
   import argparse, os
   from PIL import Image
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

   def main():
       p = argparse.ArgumentParser()
       p.add_argument('--scenes', nargs=3, required=True,
                      help='3 張單場景對比圖 (input+ours+orig+gt 已並排成一張)')
       p.add_argument('--out', required=True, help='輸出 PDF 路徑')
       args = p.parse_args()

       fig, axes = plt.subplots(3, 1, figsize=(14, 12))
       for ax, path in zip(axes, args.scenes):
           ax.imshow(Image.open(path))
           ax.axis('off')
       plt.tight_layout()
       os.makedirs(os.path.dirname(args.out), exist_ok=True)
       plt.savefig(args.out, dpi=150, bbox_inches='tight')
       print(f'Saved → {args.out}')

   if __name__ == '__main__':
       main()
   ```

   執行：
   ```bash
   python scripts/compose_qualitative.py \
       --scenes "../docs/intermediate/qualitative_all/<選擇1>.png" \
                "../docs/intermediate/qualitative_all/<選擇2>.png" \
                "../docs/intermediate/qualitative_all/<選擇3>.png" \
       --out "../docs/figures/qualitative_traffic_sign.pdf"
   ```

---

## Phase 5：找出並產出失敗案例 (Figure 2)

**目標**：2 個失敗案例 (a) 過暗失色 + (b) 強光 bloom → 一張 `docs/figures/failure_cases.pdf`

**步驟：**

1. **使用 Phase 3 產生的** `TensorFlow/experiments/per_image_LOLI-Street.csv` 排序找 PSNR 最低的 10 張：
   ```python
   import csv
   with open('experiments/per_image_LOLI-Street.csv') as f:
       rows = list(csv.DictReader(f))
   worst = sorted(rows, key=lambda r: float(r['psnr']))[:10]
   for r in worst:
       print(r['filename'], r['psnr'])
   ```

2. **對最差 10 張生成詳細對比圖**到 `docs/intermediate/failure_candidates/`：
   ```bash
   python scripts/gen_comparison.py \
       --images <worst10 paths separated by space> \
       --ours <OURS_WEIGHTS_LOLV1> --ours_arch modified \
       --orig pretrained_weights/LOLv1.h5 --orig_arch original \
       --out "../docs/intermediate/failure_candidates"
   ```

3. **人工/啟發式挑選**：
   - **(a) 過暗失色案例**：標誌區域在輸入圖的像素均值 < 0.1，但 GT 中是飽和色 (R/G/B 至少一通道 > 0.7)
   - **(b) 強光 bloom 案例**：輸入中有「車燈/路燈」(像素均值 > 0.9 的區塊 > 100 像素)，且 Ours 輸出有明顯光暈擴散到附近標誌

4. **把選用的 2 張原始圖檔複製到** `docs/figures/sources/failure_{a,b}_*.jpg`

5. **建立 `TensorFlow/scripts/compose_failure.py`** 組合失敗案例：
   ```python
   """組合 2 個失敗案例 (Input | Ours | GT) → 單張 PDF"""
   import argparse, os
   from PIL import Image
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

   def load(p):
       return Image.open(p)

   def main():
       p = argparse.ArgumentParser()
       p.add_argument('--case_a', nargs=3, required=True, help='input ours gt')
       p.add_argument('--case_b', nargs=3, required=True, help='input ours gt')
       p.add_argument('--out', required=True)
       args = p.parse_args()

       fig, axes = plt.subplots(2, 3, figsize=(15, 8))
       titles = ['Input (Low-light)', 'Ours', 'Ground Truth']
       for col, (img_path, title) in enumerate(zip(args.case_a, titles)):
           axes[0, col].imshow(load(img_path)); axes[0, col].axis('off')
           if col == 0: axes[0, col].set_ylabel('(a)', fontsize=14)
           axes[0, col].set_title(title)
       for col, img_path in enumerate(args.case_b):
           axes[1, col].imshow(load(img_path)); axes[1, col].axis('off')
           if col == 0: axes[1, col].set_ylabel('(b)', fontsize=14)
       plt.tight_layout()
       os.makedirs(os.path.dirname(args.out), exist_ok=True)
       plt.savefig(args.out, dpi=150, bbox_inches='tight')
       print(f'Saved → {args.out}')

   if __name__ == '__main__':
       main()
   ```

   執行：
   ```bash
   python scripts/compose_failure.py \
       --case_a "<a_input>.jpg" "<a_ours>.png" "<a_gt>.jpg" \
       --case_b "<b_input>.jpg" "<b_ours>.png" "<b_gt>.jpg" \
       --out "../docs/figures/failure_cases.pdf"
   ```

---

## Phase 6：產生數據總結 `docs/RESULTS_DATA.md`（**不動 results.tex**）

> ⚠️ **重要**：為節省 token，**不要**讀取或修改 `D:\人工智慧課程 1142\人工智慧課程 1142\results.tex`。所有結果統一寫入 `docs/RESULTS_DATA.md`，由使用者之後手動貼回。

**輸出檔案**：`D:\人工智慧課程 1142\LYT-Net-main\docs\RESULTS_DATA.md`

**動作**：

1. 讀 `TensorFlow/experiments/eval_original.csv` 和 `TensorFlow/experiments/eval_modified.csv`
2. 計算 Ours 相對 baseline 的 delta（每個資料集每個指標）
3. 依下列模板產出 `docs/RESULTS_DATA.md`：

````markdown
# Results 章節數據總結

> 產生時間：<YYYY-MM-DD HH:MM>
> 此檔記錄所有實驗數據與圖片路徑，由使用者手動貼回 `D:\人工智慧課程 1142\人工智慧課程 1142\results.tex`。

---

## 1. Table I (`tab:cross_dataset`) — 量化結果

| Method | LOL-v1 PSNR↑ | LOL-v1 SSIM↑ | LOL-v1 LPIPS↓ | LOL-v2-Real PSNR↑ | LOL-v2-Real SSIM↑ | LOL-v2-Real LPIPS↓ | LOLI-Street PSNR↑ | LOLI-Street SSIM↑ | LOLI-Street LPIPS↓ |
|---|---|---|---|---|---|---|---|---|---|
| LYT-Net (baseline) | XX.XX | X.XXX | X.XXX | XX.XX | X.XXX | X.XXX | XX.XX | X.XXX | X.XXX |
| **Ours** | **XX.XX** | **X.XXX** | **X.XXX** | **XX.XX** | **X.XXX** | **X.XXX** | **XX.XX** | **X.XXX** | **X.XXX** |

### LaTeX 替換用片段（直接複製貼到 results.tex 的 tabular 內）

```latex
LYT-Net (baseline) & XX.XX & X.XXX & X.XXX
                   & XX.XX & X.XXX & X.XXX
                   & XX.XX & X.XXX & X.XXX \\
Ours               & \textbf{XX.XX} & \textbf{X.XXX} & \textbf{X.XXX}
                   & \textbf{XX.XX} & \textbf{X.XXX} & \textbf{X.XXX}
                   & \textbf{XX.XX} & \textbf{X.XXX} & \textbf{X.XXX} \\
```

---

## 2. 段落中的 delta 數字（替換 `results.tex` 中 `\paragraph{...}` 的 `+X.XX`）

### In-Domain (LOL-v1)
- Ours: PSNR=XX.XX, SSIM=X.XXX, LPIPS=X.XXX
- Baseline: PSNR=XX.XX, SSIM=X.XXX, LPIPS=X.XXX

### LOL-v2-Real delta (Ours − Baseline)
- ΔPSNR = **+X.XX dB**
- ΔSSIM = **+X.XXX**
- ΔLPIPS = **−X.XXX**（注意 LPIPS 越低越好，所以是負號）

### LOLI-Street delta (Ours − Baseline)
- ΔPSNR = **+X.XX dB**
- ΔSSIM = **+X.XXX**
- ΔLPIPS = **−X.XXX**

---

## 3. Figure 1：`fig:qualitative` 定性比較圖

**檔案實際路徑**：`D:\人工智慧課程 1142\LYT-Net-main\docs\figures\qualitative_traffic_sign.pdf`

**選用的 3 個場景**：
1. 場景 1（紅色禁止標誌）：來源 `<原始 LOLI-Street 檔名>`
2. 場景 2（藍色指示標誌）：來源 `<原始 LOLI-Street 檔名>`
3. 場景 3（含文字符號）：來源 `<原始 LOLI-Street 檔名>`

**LaTeX 替換**：把 `results.tex` 中 `fig:qualitative` 那段的
```latex
\rule{\linewidth}{0.4pt}\\[-2pt]
\rule{0pt}{2.2cm}\textit{[Qualitative comparison figure to be inserted]}\\
\rule{\linewidth}{0.4pt}
```
改為（前提：使用者已把 PDF 複製到論文同目錄的 `figures/` 子資料夾）：
```latex
\includegraphics[width=\linewidth]{figures/qualitative_traffic_sign.pdf}
```

---

## 4. Figure 2：`fig:failure` 失敗案例圖

**檔案實際路徑**：`D:\人工智慧課程 1142\LYT-Net-main\docs\figures\failure_cases.pdf`

**選用的 2 個失敗案例**：
- (a) 過暗失色：來源 `<原始 LOLI-Street 檔名>`，Ours PSNR=XX.XX
- (b) 強光 bloom：來源 `<原始 LOLI-Street 檔名>`，Ours PSNR=XX.XX

**LaTeX 替換**：把 `results.tex` 中 `fig:failure` 那段的 `\rule{...}\textit{...}\rule{...}` 改為：
```latex
\includegraphics[width=\linewidth]{figures/failure_cases.pdf}
```

---

## 5. 完整檔案清單（給使用者）

**需要複製到主論文目錄的檔案**（建議放到 `D:\人工智慧課程 1142\人工智慧課程 1142\figures\`）：

- [ ] `docs/figures/qualitative_traffic_sign.pdf` → `figures/qualitative_traffic_sign.pdf`
- [ ] `docs/figures/failure_cases.pdf` → `figures/failure_cases.pdf`

**需要在 `results.tex` 中手動修改的位置**：

- [ ] tabular 內的 `XX.XX` × 18（見第 1 節 LaTeX 片段）
- [ ] `\paragraph{In-Domain Performance}` 內的數字（見第 2 節）
- [ ] `\paragraph{Cross-Dataset Generalization}` 內的 `+X.XX` × 6（見第 2 節 delta）
- [ ] `fig:qualitative` 的 `\rule` 佔位 → `\includegraphics`
- [ ] `fig:failure` 的 `\rule` 佔位 → `\includegraphics`

---

## 6. 原始 CSV 路徑（給開發者驗證用）

- `D:\人工智慧課程 1142\LYT-Net-main\TensorFlow\experiments\eval_original.csv`
- `D:\人工智慧課程 1142\LYT-Net-main\TensorFlow\experiments\eval_modified.csv`
- `D:\人工智慧課程 1142\LYT-Net-main\TensorFlow\experiments\per_image_LOLI-Street.csv`

````

**驗證**：
```bash
ls -la "D:\人工智慧課程 1142\LYT-Net-main\docs\RESULTS_DATA.md"
# 確認沒有未替換的 XX.XX（除了 LaTeX 範例片段內保留示意用的 XX.XX）
```

> 注意：**不要打開或修改 `results.tex`**，僅根據 CSV 數據與圖片路徑寫 `docs/RESULTS_DATA.md` 即可。

---

## Phase 7：產出最終交付清單

完成後輸出：

```
✅ 量化數據（raw）：
   - TensorFlow/experiments/eval_original.csv
   - TensorFlow/experiments/eval_modified.csv
   - TensorFlow/experiments/per_image_LOLI-Street.csv

✅ 圖片：
   - docs/figures/qualitative_traffic_sign.pdf
   - docs/figures/failure_cases.pdf
   - docs/figures/sources/  （5 張原檔備份）
   - docs/intermediate/     （所有候選對比圖，方便日後重挑）

✅ 數據總結文件（給使用者貼回 results.tex 用）：
   - docs/RESULTS_DATA.md
     ↑ 含所有要替換的數字、LaTeX 片段、圖檔路徑、與替換位置清單

✅ 新增的程式碼：
   - TensorFlow/scripts/eval_all.py
   - TensorFlow/scripts/compose_qualitative.py
   - TensorFlow/scripts/compose_failure.py

✅ 修改過的程式碼（含 .bak 備份）：
   - TensorFlow/scripts/gen_comparison.py (+ .bak)
   - TensorFlow/data_loading.py (if patched, + .bak)
   - TensorFlow/scripts/train.py (if patched, + .bak)

❌ 不應該動到的檔案（節省 token / 保護論文）：
   - D:\人工智慧課程 1142\人工智慧課程 1142\results.tex
   - D:\人工智慧課程 1142\人工智慧課程 1142\methodology.tex
   - D:\人工智慧課程 1142\人工智慧課程 1142\methodology_modify.tex
   - D:\人工智慧課程 1142\人工智慧課程 1142\currentLatex.txt

📋 給使用者的下一步：
   1. 開啟 docs/RESULTS_DATA.md，依照「LaTeX 替換用片段」與「替換位置清單」手動貼進 results.tex
   2. 把 docs/figures/qualitative_traffic_sign.pdf 與 failure_cases.pdf 複製到主論文目錄的 figures/ 子資料夾
   3. pdflatex / xelatex 編譯確認
```

---

## 關鍵風險與例外處理

| 情境 | 處理方式 |
|---|---|
| LPIPS 安裝失敗或下載 weights 失敗 | 改用 `--metrics psnr ssim` 跳過 LPIPS，告知使用者後續手動補 |
| LOL-v2-Real 資料集不存在 | 該欄填 "N/A — dataset not available"，繼續其他評估 |
| LOLI-Street 資料集不存在 | **必須中止**並提示下載連結（這是論文核心對比） |
| 訓練 74 epochs 太久 | 給使用者選擇：(a) 改用既有權重 (b) 暫時用 baseline 數字 + Ours 留 N/A，但要在文中說明 |
| `model_modify` 權重與 `model.arch` 結構不相容 | `eval_all.py` 已用 `--arch` 旗標處理；若仍報錯，dump 兩邊 layer name 比對 |
| 沒有交通標誌場景符合 | 改用一般低光場景，但要在 figure caption 註明 |
| `data_loading.py` 缺 `get_loli_datasets_metrics` | 仿照 `get_datasets_metrics` 新增一個讀 `.jpg` 的版本，先備份再改 |

---

## 建議執行順序

1. Phase 0 (5 分鐘) → 環境/檔案/資料夾驗證
2. Phase 1 (依是否需訓練：0 或數小時)
3. Phase 2 (15 分鐘) → 寫 `eval_all.py` + 視需要改 `data_loading.py`
4. Phase 3 (20-60 分鐘) → 跑 2 次評估 + per-image log
5. Phase 4 (30 分鐘) → 產 30 張候選定性圖 + 人工挑 3 張 + 組合 PDF
6. Phase 5 (30 分鐘) → 找 worst-10 失敗候選 + 挑 2 案例 + 組合 PDF
7. Phase 6 (10 分鐘) → 寫 `docs/RESULTS_DATA.md`
8. Phase 7 (5 分鐘) → 交付驗證 + 清單輸出

**預估總時數**：若 Phase 1 已有權重，2-3 小時；若需從頭訓練 LOL-v1，加 2-6 小時 GPU 時間。

---

## 給 Claude Code 的指示

- ⚠️ **絕對不要讀取 `D:\人工智慧課程 1142\人工智慧課程 1142\` 底下的任何 `.tex` 或 `.txt` 檔**（檔案很大、會吃掉大量 context token）。所有要寫進論文的內容統一輸出到 `docs/RESULTS_DATA.md`，由使用者自己貼回。
- ⚠️ **絕對不要修改使用者論文資料夾** (`D:\人工智慧課程 1142\人工智慧課程 1142\`) **底下的任何檔案**。
- **所有新檔案放在以下指定位置**：
  - 新 Python 腳本 → `TensorFlow/scripts/`
  - 評估產出 CSV → `TensorFlow/experiments/`
  - 最終圖檔 → `docs/figures/`
  - 候選/中間圖檔 → `docs/intermediate/`
  - 數據總結 .md → `docs/RESULTS_DATA.md`
- **修改既有檔案前一定要先備份成 `*.bak`**（適用於 `gen_comparison.py`, `data_loading.py`, `train.py`）
- **每完成一個 Phase 就回報進度**與產出檔案路徑
- **遇到不確定就停下來問**，不要擅自跳過評估步驟
- **保留所有中間檔案**（CSV、所有候選對比圖、per-image log）方便日後重挑
- **產生 `RESULTS_DATA.md` 時**，把所有實際數字（PSNR 兩位小數、SSIM/LPIPS 三位小數）填好，包含已算好的 delta 值，使用者就不用再手算
