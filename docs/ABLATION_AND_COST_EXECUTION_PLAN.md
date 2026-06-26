# Ablation Study + Cost Table 實驗執行計畫 (for Claude Code)

> **目的**：為 IEEE 論文 "Road sign recognition in low-light conditions"
> 補上 `tab:ablation` (Table II) 與 `tab:cost` (Table III) 的數字。
>
> **本計畫位置**：`D:\人工智慧課程 1142\LYT-Net-main\docs\ABLATION_AND_COST_EXECUTION_PLAN.md`
>
> **論文檔位置（不可修改）**：`D:\人工智慧課程 1142\人工智慧課程 1142\paper_v2_filled.tex`
>
> **程式碼根目錄**：`D:\人工智慧課程 1142\LYT-Net-main\TensorFlow`
>
> **GPU**：RTX 4060 (8 GB)
>
> **執行階段**：本計畫分 Phase A (Cost) 與 Phase B (Ablation) 兩階段，可獨立執行。Phase A 不需訓練，先做完。Phase B 需 GPU 跑 2 次訓練。

---

## 任務總覽

| 階段 | 產出 | 時間預估 | 訓練 |
|---|---|---|---|
| Phase A | Table III 6 個數字 (params / FLOPs / latency × 2 模型) + delta | 30 分鐘 | 否 |
| Phase B | Table II 6 個數字 (PSNR/SSIM/LPIPS × 2 新配置) | 4–6 小時 | 是 (×2 次) |
| Phase C | `docs/ABLATION_AND_COST_DATA.md`（給使用者貼回 paper_v2_filled.tex） | 10 分鐘 | 否 |

---

## Phase 0：環境檢查（與 RESULTS_EXECUTION_PLAN 相同，若該計畫已完成可跳過）

```bash
cd "D:\人工智慧課程 1142\LYT-Net-main\TensorFlow"
conda activate LYTNet
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
python -c "import torch, lpips; print('torch:', torch.__version__); print('lpips OK')"
pip show keras-flops || pip install keras-flops  # 用於 FLOPs 量測
```

**必要檔案：**
- [ ] `TensorFlow/model/arch.py` (原版 LYT-Net)
- [ ] `TensorFlow/model_modify/arch.py` (D1+D2+D3 版)
- [ ] `TensorFlow/pretrained_weights/LOLv1.h5` (baseline weights)
- [ ] `TensorFlow/experiments/LOLv1/<best>.h5` (Ours weights，記下檔名)
- [ ] `TensorFlow/scripts/eval_all.py` (RESULTS_EXECUTION_PLAN 已產出)

---

# Phase A：Cost Table 量測（先做，無需訓練）

## A.1 建立 `TensorFlow/scripts/measure_cost.py`

```python
"""
量測 LYT-Net (model.arch) 與 Ours (model_modify.arch) 在固定解析度
下的 #params / FLOPs / 平均推理延遲。

Usage:
    python scripts/measure_cost.py \
        --orig pretrained_weights/LOLv1.h5 \
        --ours experiments/LOLv1/<best>.h5 \
        --h 400 --w 600
"""
import os, sys, time, argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf


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


def count_flops(model, h, w):
    """用 tf.compat.v1.profiler 估 FLOPs。也可改用 keras-flops。"""
    try:
        from keras_flops import get_flops
        return get_flops(model, batch_size=1) / 1e9  # GFLOPs
    except Exception:
        # fallback: tf profiler
        concrete = tf.function(lambda x: model(x)).get_concrete_function(
            tf.TensorSpec([1, h, w, 3], tf.float32)
        )
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(graph_def, name='')
            flops = tf.compat.v1.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops / 1e9


def measure_latency(model, h, w, warmup=10, n=100):
    x = tf.random.uniform([1, h, w, 3], -1.0, 1.0)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    # timed
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = model(x)
        # 同步 GPU
        if tf.config.list_physical_devices('GPU'):
            tf.experimental.numpy.experimental_enable_numpy_behavior  # no-op to keep import alive
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ts)), float(np.std(ts))


def one_model(arch, weights, h, w):
    print(f"\n--- {arch} | weights: {weights} ---")
    m = build(arch)
    m.load_weights(weights)
    n_params = m.count_params() / 1e6  # M
    flops = count_flops(m, h, w)
    lat_mean, lat_std = measure_latency(m, h, w)
    print(f"#Params: {n_params:.3f} M")
    print(f"FLOPs:   {flops:.2f} G  (@ {h}x{w})")
    print(f"Latency: {lat_mean:.1f} +/- {lat_std:.1f} ms")
    return n_params, flops, lat_mean, lat_std


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--orig', required=True)
    p.add_argument('--ours', required=True)
    p.add_argument('--h', type=int, default=400)
    p.add_argument('--w', type=int, default=600)
    p.add_argument('--out', default='./experiments/cost_table.csv')
    args = p.parse_args()

    rows = []
    p0 = one_model('original', args.orig, args.h, args.w)
    rows.append(('LYT-Net (baseline)', *p0))
    p1 = one_model('modified', args.ours, args.h, args.w)
    rows.append(('Ours', *p1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('method,params_M,flops_G,latency_ms_mean,latency_ms_std\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')
    print(f"\nSaved {args.out}")
```

## A.2 執行

```bash
cd "D:\人工智慧課程 1142\LYT-Net-main\TensorFlow"
python scripts/measure_cost.py \
    --orig pretrained_weights/LOLv1.h5 \
    --ours experiments/LOLv1/<OURS_BEST>.h5 \
    --h 400 --w 600
```

**產出**：`TensorFlow/experiments/cost_table.csv`（2 列數字）。

**例外處理**：
- 若 `keras-flops` 安裝失敗，script 會 fallback 用 `tf.profiler`。若兩者皆失敗，至少 params 與 latency 可成功。
- 解析度可調 `--h 256 --w 256` 對齊訓練 patch size，但建議 400×600 較接近 LOLI-Street 實際畫面。
- Latency 受 GPU 同步影響，可額外用 `tf.config.experimental.set_synchronous_execution(True)` 強化精度。

---

# Phase B：Ablation Study（需訓練 2 次）

## B.1 Patch `model_modify/arch.py` 加入 D1/D2/D3 可控旗標

**先備份**：`cp model_modify/arch.py model_modify/arch.py.bak`

**目標**：讓 `LYT(...)` 與 `Denoiser(...)` 接受三個旗標，控制是否啟用 D1/D2/D3：

| 旗標 | 預設 | 行為 |
|---|---|---|
| `enable_d1` | `True` | 啟用 → φ 是 2 層 Conv；停用 → φ 退化為 1 層 Conv（同 baseline） |
| `dropout_rate` | `0.1` | `0.0` 時等效停用 D2 |
| `learnable_lum` | `True` | `False` 時 `w_lum` 固定為常數 0.2 |

**需要改的地方（請 Claude Code 讀完 arch.py 後實作）**：

1. **`Denoiser` class**：把 bottleneck 後的 `Dropout(0.1)` 改成 `Dropout(self.dropout_rate)`，並在 `__init__` 加 `dropout_rate=0.1` 參數。
2. **`LYT` class**：
   - `__init__` 加 `enable_d1=True, dropout_rate=0.1, learnable_lum=True` 參數。
   - φ 抽取器：若 `enable_d1=False`，只用一層 Conv3×3+ReLU（同 `model.arch`）。
   - `w_lum` 參數：若 `learnable_lum=False`，改用 `tf.constant(0.2, dtype=tf.float32)`；否則維持 `tf.Variable(0.2, trainable=True)`。
   - 把 `dropout_rate` 傳進 `Denoiser`。

**測試**：改完後執行下列 sanity check（確保不會破壞 Ours 預設行為）：
```bash
python -c "
from model_modify.arch import LYT, Denoiser
m = LYT(filters=32,
        denoiser_cb=Denoiser(16, dropout_rate=0.1),
        denoiser_cr=Denoiser(16, dropout_rate=0.1),
        enable_d1=True, dropout_rate=0.1, learnable_lum=True)
m.build(input_shape=(None,None,None,3))
print('Param count:', m.count_params())
"
```
這個 param count 應該跟原本的 Ours 模型一致。

## B.2 建立 `TensorFlow/scripts/train_ablation.py`

```python
"""
跑指定 ablation 配置的 LOL-v1 訓練（74 epochs）。
Usage:
    python scripts/train_ablation.py --config d1_only
    python scripts/train_ablation.py --config d1_d2
"""
import os, sys, argparse, shutil
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

CONFIGS = {
    'd1_only': dict(enable_d1=True,  dropout_rate=0.0, learnable_lum=False),
    'd1_d2':   dict(enable_d1=True,  dropout_rate=0.1, learnable_lum=False),
    # 'd1_d2_d3' 已是 Ours，不必再跑
}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=list(CONFIGS.keys()), required=True)
    p.add_argument('--epochs', type=int, default=74)
    args = p.parse_args()

    cfg = CONFIGS[args.config]
    save_dir = f'./experiments/ablation_{args.config}'
    os.makedirs(save_dir, exist_ok=True)

    # 借用既有 train.py 邏輯但用 ablation 配置
    # 最簡單：複製 scripts/train.py → scripts/train_ablation_runner.py
    # 然後在 build_model 處改用 LYT(..., **cfg)
    # 並把 save_dir 設成 experiments/ablation_<config>/
    #
    # 由於 train.py 的細節 (data loading, loss, scheduler) 與 ablation 完全相同，
    # 為了不重複實作，建議用下列做法（請 Claude Code 自行判斷實作方式）：
    #
    # 1. import train.py 內的 build_model / build_loss / run_training 函式
    # 2. 改寫 build_model 接受 cfg
    # 3. 呼叫 run_training(save_dir, epochs=args.epochs)
    #
    # 若 train.py 沒有結構化函式（一團 script），則直接 sed-replace 兩處：
    #   LYT(filters=32, ...) → LYT(filters=32, **cfg, ...)
    #   './experiments/LOLv1' → save_dir
    raise NotImplementedError(
        '請 Claude Code 依專案實際 train.py 結構實作，'
        '保留 74 epochs、學習率、loss weights 等所有原配置，只改 LYT(...) 的 kwargs。'
    )
```

> ⚠️ 注意：以上 `train_ablation.py` 是骨架。實際做法視 `scripts/train.py` 結構而定。Claude Code 應先 `Read` train.py，再決定是 (a) 重構成可呼叫函式、或 (b) 用 monkey-patch 改 LYT() 呼叫。優先選擇 (b) 較不破壞既有腳本。

## B.3 執行兩組訓練

```bash
# 第一組：+D1 only（74 epochs，預估 1-3 小時）
python scripts/train_ablation.py --config d1_only

# 第二組：+D1+D2（74 epochs，預估 1-3 小時）
python scripts/train_ablation.py --config d1_d2

# 兩組產出：
# experiments/ablation_d1_only/<best>.h5
# experiments/ablation_d1_d2/<best>.h5
```

**建議**：兩組可序列跑，總 GPU 時間 2–6 小時。可掛夜間。

**Sanity check 訓練是否正常**（前 1–2 epoch 後）：
- validation PSNR 應在 1–2 epoch 內爬過 15 dB
- loss 應穩定下降，沒有 NaN

## B.4 用 eval_all.py 評估這兩個 checkpoint 在 LOLI-Street 上的表現

```bash
# 為了讓 eval_all.py 拿到對的 arch 配置，需要在 build() 時帶入 ablation 旗標
# 最簡單作法：先在 model_modify.arch 加一個 module-level 環境變數讀取器，
# 或者在 eval 時直接 import 模型並 set attribute 後 load_weights。

# 推薦做法：建立 scripts/eval_ablation.py，類似 eval_all.py 但
# 在 build() 加 --config 旗標：
python scripts/eval_ablation.py --config d1_only \
    --weights experiments/ablation_d1_only/<best>.h5 \
    --datasets LOLI-Street

python scripts/eval_ablation.py --config d1_d2 \
    --weights experiments/ablation_d1_d2/<best>.h5 \
    --datasets LOLI-Street
```

> 注意：load_weights 要在用相同 cfg build 過的 model 上做，否則 layer name 不一致會報錯。

**產出**：`TensorFlow/experiments/eval_ablation_d1_only.csv`、`eval_ablation_d1_d2.csv`。

---

# Phase C：產出數據總結 `docs/ABLATION_AND_COST_DATA.md`

> ⚠️ **不要修改 paper_v2_filled.tex**，所有數字統一寫到本檔，由使用者手動貼回。

**範本**：

````markdown
# Ablation + Cost 章節數據總結

> 產生時間：<YYYY-MM-DD>
> 對應論文：`D:\人工智慧課程 1142\人工智慧課程 1142\paper_v2_filled.tex`

---

## 1. Table II (`tab:ablation`)

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| LYT-Net (baseline) | 14.64 | 0.774 | 0.106 |
| +D1 (deeper per-channel) | XX.XX | X.XXX | X.XXX |
| +D1+D2 (bottleneck Dropout) | XX.XX | X.XXX | X.XXX |
| +D1+D2+D3 (Ours, full) | **17.22** | **0.842** | **0.080** |

### LaTeX 替換片段

```latex
LYT-Net (baseline)                & 14.64 & 0.774 & 0.106 \\
\,+\,D1 (deeper per-channel)        & XX.XX & X.XXX & X.XXX \\
\,+\,D1\,+\,D2 (bottleneck Dropout) & XX.XX & X.XXX & X.XXX \\
\,+\,D1\,+\,D2\,+\,D3 (Ours, full)  & \textbf{17.22} & \textbf{0.842} & \textbf{0.080} \\
```

---

## 2. Table III (`tab:cost`) — @ 400×600 input

| Method | #Params (M) | FLOPs (G) | Latency (ms) |
|---|---|---|---|
| LYT-Net (baseline) | X.XXX | XX.XX | XX.X |
| Ours | X.XXX | XX.XX | XX.X |
| Δ | +X.XXX | +XX.XX | +XX.X |

### LaTeX 替換片段

```latex
LYT-Net (baseline) & X.XXX & XX.XX & XX.X \\
Ours               & X.XXX & XX.XX & XX.X \\
$\Delta$           & $+$X.XXX & $+$XX.XX & $+$XX.X \\
```

---

## 3. 給使用者的下一步

- [ ] 開啟 paper_v2_filled.tex，找 `tab:ablation`，將 4 個 XX.XX 替換成上表
- [ ] 找 `tab:cost`，將 6 個佔位替換成上表
- [ ] 刪掉表格上方的 `%% [TODO]` 註解
- [ ] 重新 pdflatex 編譯

---

## 4. 原始 CSV

- `TensorFlow/experiments/cost_table.csv`
- `TensorFlow/experiments/eval_ablation_d1_only.csv`
- `TensorFlow/experiments/eval_ablation_d1_d2.csv`
````

---

# 風險與例外

| 情境 | 處理方式 |
|---|---|
| `keras-flops` 安裝失敗 | 用 tf.profiler fallback；或只報 params + latency |
| `arch.py` patch 後 ours 既有 weights 載入失敗 | 確保 layer name 完全一致；若失敗，新訓練 Ours 加上 cfg |
| RTX 4060 8GB 訓練 OOM | 把 batch size 從預設降一半；patch size 仍維持 256×256 |
| 訓練後 PSNR 異常（< 12 dB on LOLI-Street） | 檢查 cfg 是否正確傳入 LYT()；確認 dropout_rate=0.0 真的不算 layer |
| Ablation 跑出來 +D1 比 baseline 更差 | 在論文中如實報告，並在 Discussion 補一句解釋 |

---

# 給 Claude Code 的指示

- ⚠️ **絕對不要修改** `D:\人工智慧課程 1142\人工智慧課程 1142\` 底下任何檔案
- 所有新腳本放在 `TensorFlow/scripts/`
- 所有產出 CSV 放在 `TensorFlow/experiments/`
- 修改 `arch.py` 與 `train.py` 前一定要 `.bak`
- **每完成一個 Phase 回報進度**並貼出 CSV 內容
- **Phase A 與 Phase B 可獨立執行**（建議先做 Phase A 看到 cost 數字後，再決定要不要花 GPU 時間做 Phase B）
- 最終把 `docs/ABLATION_AND_COST_DATA.md` 寫好，讓使用者一個動作就能把數字貼回論文

---

# 建議執行順序

1. **Phase 0** (5 min) — 環境 / 檔案檢查
2. **Phase A.1 → A.2** (30 min) — Cost table，先看數字確認 lightweight 論點是否站得住
3. **回報 Phase A 結果，等使用者確認**
4. **Phase B.1** (15 min) — patch `arch.py`，跑 sanity check
5. **Phase B.2** (15 min) — 寫 `train_ablation.py`
6. **Phase B.3** (2–6 hr, 可掛夜間) — 跑兩次訓練
7. **Phase B.4** (15 min) — 評估
8. **Phase C** (10 min) — 寫 `ABLATION_AND_COST_DATA.md`

**總時間**：Phase A 約 30 min；Phase B 約 4–6 hr GPU + 30 min 人工。
