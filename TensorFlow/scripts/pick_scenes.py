"""
分析 30 張候選圖，找含紅色/藍色/文字標誌的場景，輸出排名並自動挑選 3 張。
"""
import os, glob, shutil
import numpy as np
from PIL import Image

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'intermediate', 'qualitative_all')
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'low')
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'figures', 'sources')

files = sorted(glob.glob(os.path.join(img_dir, '*_comparison.png')))

results = []
for f in files:
    img = np.array(Image.open(f).convert('RGB'))
    h, w = img.shape[:2]
    # Input 欄（左 1/4），GT 欄（右 1/4）
    inp = img[:, :w//4, :]
    gt  = img[:, 3*w//4:, :]

    r_i, g_i, b_i = inp[:,:,0].astype(float), inp[:,:,1].astype(float), inp[:,:,2].astype(float)
    r_g, g_g, b_g = gt[:,:,0].astype(float),  gt[:,:,1].astype(float),  gt[:,:,2].astype(float)

    total_px = r_i.size

    # 紅色標誌 (在 GT 中明顯，在 input 中暗淡)
    red_gt   = ((r_g > 150) & (r_g > g_g * 1.5) & (r_g > b_g * 1.5)).sum()
    red_inp  = ((r_i > 100) & (r_i > g_i * 1.5) & (r_i > b_i * 1.5)).sum()
    red_score = float(red_gt) / total_px * 1000

    # 藍色標誌 (在 GT 中明顯)
    blue_gt  = ((b_g > 150) & (b_g > r_g * 1.3) & (b_g > g_g * 1.1)).sum()
    blue_score = float(blue_gt) / total_px * 1000

    # PSNR gap (Ours - Original): 用 Ours 欄（左中 1/4）vs Original 欄（右中 1/4）
    # 用相同 shape 的切片避免 off-by-one
    q = h * (w // 4)  # 以行數 * 欄寬 計算，改為直接切 reshape 會更穩定
    ours_col = img[:, w//4: 2*(w//4), :]
    orig_col = img[:, 2*(w//4): 3*(w//4), :]
    gt_crop  = img[:, 3*(w//4): 4*(w//4), :]
    min_w = min(ours_col.shape[1], gt_crop.shape[1])
    ours_col = ours_col[:, :min_w, :]
    orig_col = orig_col[:, :min_w, :]
    gt_crop  = gt_crop[:, :min_w, :]
    mse_ours = float(np.mean((ours_col.astype(float) - gt_crop.astype(float))**2))
    mse_orig = float(np.mean((orig_col.astype(float) - gt_crop.astype(float))**2))
    psnr_ours = 10 * np.log10(255**2 / (mse_ours + 1e-6))
    psnr_orig = 10 * np.log10(255**2 / (mse_orig + 1e-6))
    psnr_gap = psnr_ours - psnr_orig

    name = os.path.basename(f).replace('_comparison.png', '')
    results.append({
        'name': name,
        'file': f,
        'red_score': red_score,
        'blue_score': blue_score,
        'psnr_gap': psnr_gap,
        'psnr_ours': psnr_ours,
    })

print(f"{'filename':35s} | red   | blue  | gap(Ours-Orig) | psnr_ours")
print('-' * 80)
for r in sorted(results, key=lambda x: -(x['red_score'] + x['blue_score'])):
    print(f"{r['name']:35s} | {r['red_score']:5.2f} | {r['blue_score']:5.2f} | {r['psnr_gap']:+.2f}           | {r['psnr_ours']:.2f}")

# --- 自動選 3 張 ---
# 場景 1：紅色得分最高
scene1 = sorted(results, key=lambda x: -x['red_score'])[0]
# 場景 2：藍色得分最高（排除已選）
scene2 = sorted([r for r in results if r['name'] != scene1['name']],
                key=lambda x: -x['blue_score'])[0]
# 場景 3：Ours 明顯優於 Original（psnr_gap 最大，排除已選）
used = {scene1['name'], scene2['name']}
scene3 = sorted([r for r in results if r['name'] not in used],
                key=lambda x: -x['psnr_gap'])[0]

selected = [
    (scene1, 'scene1_red_sign'),
    (scene2, 'scene2_blue_sign'),
    (scene3, 'scene3_text_sign'),
]

print('\n=== 自動選定的 3 個場景 ===')
for scene, tag in selected:
    print(f"  {tag}: {scene['name']}  (red={scene['red_score']:.2f}, blue={scene['blue_score']:.2f}, gap={scene['psnr_gap']:+.2f} dB)")
    src_jpg = os.path.join(src_dir, scene['name'] + '.jpg')
    dst_jpg = os.path.join(out_dir, tag + '.jpg')
    if os.path.exists(src_jpg):
        shutil.copy2(src_jpg, dst_jpg)
        print(f"    Copied {src_jpg} → {dst_jpg}")
    else:
        print(f"    WARNING: source jpg not found at {src_jpg}")

print('\n場景對比圖路徑（給 compose_qualitative.py 用）：')
for scene, tag in selected:
    print(f"  {scene['file']}")
