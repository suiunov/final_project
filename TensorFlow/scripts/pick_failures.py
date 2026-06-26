"""
從 per_image_LOLI-Street.csv 找 PSNR 最低的 10 張，
生成對比圖到 docs/intermediate/failure_candidates/，
並自動挑選「過暗失色」和「強光 bloom」兩個失敗案例，
複製原始圖到 docs/figures/sources/failure_{a,b}*.jpg。
"""
import os, sys, csv, shutil, glob
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.join(script_dir, '..')
sys.path.append(root_dir)
sys.path.append(script_dir)

csv_path = os.path.join(root_dir, 'experiments', 'per_image_LOLI-Street.csv')
low_dir  = os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'low')
cand_dir = os.path.join(root_dir, '..', 'docs', 'intermediate', 'failure_candidates')
src_dir  = os.path.join(root_dir, '..', 'docs', 'figures', 'sources')
os.makedirs(cand_dir, exist_ok=True)

# 讀 CSV
with open(csv_path) as f:
    rows = list(csv.DictReader(f))

# 排序取最差 10 張
worst10 = sorted(rows, key=lambda r: float(r['psnr']))[:10]
print('PSNR 最低 10 張：')
for r in worst10:
    print(f"  {r['filename']:30s}  PSNR={float(r['psnr']):.2f}  SSIM={float(r['ssim']):.4f}")

print(f'\n正在生成 {len(worst10)} 張失敗候選對比圖...')

import os, sys, glob as _glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model_modify.arch import LYT, Denoiser
from model.arch import LYT as LYT_orig, Denoiser as Denoiser_orig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 建立模型
def make_model(arch_cls_lyt, arch_cls_den, weights):
    d_cb = arch_cls_den(16); d_cr = arch_cls_den(16)
    d_cb.build((None,None,None,1)); d_cr.build((None,None,None,1))
    m = arch_cls_lyt(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m.build((None,None,None,3))
    m.load_weights(weights)
    return m

ours_w = os.path.join(root_dir, 'experiments', 'LOLv1',
    'net_psnr_26.41_ssim_0.852_epoch_156_dataset_LOLv1.h5')
orig_w = os.path.join(root_dir, 'pretrained_weights', 'LOLv1.h5')

model_ours = make_model(LYT, Denoiser, ours_w)
model_orig = make_model(LYT_orig, Denoiser_orig, orig_w)

generated_paths = []
for r in worst10:
    fname = r['filename']
    img_path = os.path.join(low_dir, fname)
    gt_path  = img_path.replace(os.sep + 'low' + os.sep, os.sep + 'high' + os.sep)

    if not os.path.exists(img_path):
        print(f'  SKIP {fname}: not found')
        continue

    raw = tf.io.read_file(img_path)
    raw = tf.image.decode_image(raw, channels=3, expand_animations=False)
    raw.set_shape([None, None, 3])
    raw = (tf.cast(raw, tf.float32) / 127.5) - 1.0
    inp = tf.expand_dims(raw, 0)

    if os.path.exists(gt_path):
        gt_raw = tf.io.read_file(gt_path)
        gt_raw = tf.image.decode_image(gt_raw, channels=3, expand_animations=False)
        gt_raw.set_shape([None, None, 3])
        gt_t = (tf.cast(gt_raw, tf.float32) / 127.5) - 1.0
        gt_exp = tf.expand_dims(gt_t, 0)
    else:
        gt_exp = None

    out_ours = model_ours(inp)
    out_orig = model_orig(inp)

    def to_np(t): return tf.clip_by_value((t[0]+1)/2, 0, 1).numpy()

    if gt_exp is not None:
        fig, axes = plt.subplots(1, 4, figsize=(28, 6.5))
        titles = ['Input (Low-light)', 'Ours', 'LYT-Net (Baseline)', 'Ground Truth']
        imgs   = [to_np(inp), to_np(out_ours), to_np(out_orig), to_np(gt_exp)]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
        titles = ['Input (Low-light)', 'Ours', 'LYT-Net (Baseline)']
        imgs   = [to_np(inp), to_np(out_ours), to_np(out_orig)]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    psnr_val = float(r['psnr'])
    fig.suptitle(f'{fname}  PSNR(Ours)={psnr_val:.2f} dB', fontsize=13)
    plt.tight_layout()
    out_png = os.path.join(cand_dir, fname.replace('.jpg', '_failure.png'))
    plt.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out_png}')
    generated_paths.append((fname, float(r['psnr']), float(r['ssim']), out_png, img_path))

# --- 自動挑選 2 個失敗案例 ---
# (a) 過暗失色：PSNR 最低
case_a = generated_paths[0]
# (b) 若 LOLI-Street 有 "light_" 前綴，優先從 light_ 類找（強光場景）；否則次低
light_cases = [x for x in generated_paths[1:] if x[0].startswith('light_')]
case_b = light_cases[0] if light_cases else generated_paths[1]

print('\n=== 自動選定的 2 個失敗案例 ===')
for tag, case in [('failure_a_dark', case_a), ('failure_b_glare', case_b)]:
    fname, psnr, ssim, cmp_png, src_jpg = case
    print(f"  {tag}: {fname}  PSNR={psnr:.2f} dB")
    dst = os.path.join(src_dir, tag + '.jpg')
    shutil.copy2(src_jpg, dst)
    print(f"    Copied → {dst}")

print('\n案例對比圖路徑（給 compose_failure.py 用）：')
print(f"  Case (a): {case_a[3]}")
print(f"  Case (b): {case_b[3]}")
