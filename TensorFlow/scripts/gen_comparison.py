"""
LYT-Net 模型對比工具（通用版）
================================
支援兩種模式：
  1. 資料集模式：載入 LOLv1/LOLv2 測試集（有 GT，4 格對比）
  2. 自訂圖片模式：指定任意圖片路徑（無 GT，3 格對比）

Usage:
  # 模式 1：用 LOLv1 測試集
  python scripts/gen_comparison.py --dataset LOLv1

  # 模式 2：指定任意圖片
  python scripts/gen_comparison.py --images path1.jpg path2.jpg path3.jpg

  # 可選：換模型權重、輸出目錄、最大解析度
  python scripts/gen_comparison.py --dataset LOLv1 --ours ./experiments/loli_scratch_v1/best.h5
  python scripts/gen_comparison.py --images *.jpg --max_dim 640 --out ./my_results
"""
import os, sys, glob, argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lpips")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import lpips

# Initialize LPIPS globally to avoid reloading
_lpips_model = None

# ======================== Helpers ========================

def get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        _lpips_model = lpips.LPIPS(net='alex')
        sys.stdout = original_stdout
    return _lpips_model

def build_model(weights_path, arch='original'):
    """建立 LYT-Net 並載入權重；arch='original' 用 model.arch，'modified' 用 model_modify.arch"""
    if arch == 'modified':
        from model_modify.arch import LYT, Denoiser
    else:
        from model.arch import LYT, Denoiser
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None, None, None, 1))
    denoiser_cr.build(input_shape=(None, None, None, 1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None, None, None, 3))
    model.load_weights(weights_path)
    return model


def load_img(path):
    """讀取 jpg/png，正規化到 [-1, 1]"""
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return tf.expand_dims(img, 0)


def safe_resize(img, max_dim):
    """Resize 到 max_dim 以內（8 的倍數），避免 OOM"""
    h, w = img.shape[1], img.shape[2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    new_h = (int(h * scale) // 8) * 8
    new_w = (int(w * scale) // 8) * 8
    return tf.image.resize(img, [new_h, new_w]), h, w


def to_np(tensor):
    """[-1,1] → [0,1] numpy"""
    return tf.clip_by_value((tensor[0] + 1.0) / 2.0, 0, 1).numpy()


def compute_metrics(pred, target):
    """計算 PSNR, SSIM, LPIPS。輸入範圍皆為 [-1, 1] 的 tensor"""
    p_01 = (pred + 1.0) / 2.0
    t_01 = (target + 1.0) / 2.0
    
    psnr = tf.image.psnr(t_01, p_01, max_val=1.0)
    ssim = tf.image.ssim(t_01, p_01, max_val=1.0)
    
    # LPIPS calculation
    p_pt = torch.from_numpy(p_01.numpy()).permute(0, 3, 1, 2).float()
    t_pt = torch.from_numpy(t_01.numpy()).permute(0, 3, 1, 2).float()
    
    lpips_model = get_lpips_model()
    with torch.no_grad():
        lpips_dist = lpips_model(p_pt, t_pt)
        
    return psnr.numpy()[0], ssim.numpy()[0], lpips_dist.item()


def infer_and_plot(inp, model_ours, model_orig, gt, name, out_dir, max_dim):
    """推論 + 產生對比圖"""
    inp_small, orig_h, orig_w = safe_resize(inp, max_dim)

    out_ours = model_ours(inp_small)
    out_orig = model_orig(inp_small)

    # Resize 回原始尺寸做顯示
    out_ours = tf.image.resize(out_ours, [orig_h, orig_w])
    out_orig = tf.image.resize(out_orig, [orig_h, orig_w])

    if gt is not None:
        # 計算指標
        psnr_ours, ssim_ours, lpips_ours = compute_metrics(out_ours, gt)
        psnr_orig, ssim_orig, lpips_orig = compute_metrics(out_orig, gt)

        # 4 格：Input | Ours | Original | GT
        fig, axes = plt.subplots(1, 4, figsize=(28, 6.5))
        
        t_ours = f"Ours (LoLI-Street)\nPSNR: {psnr_ours:.2f} | SSIM: {ssim_ours:.3f} | LPIPS: {lpips_ours:.3f}"
        t_orig = f"Original (LOLv1.h5)\nPSNR: {psnr_orig:.2f} | SSIM: {ssim_orig:.3f} | LPIPS: {lpips_orig:.3f}"
        
        titles = ['Input (Low-light)', t_ours, t_orig, 'Ground Truth']
        images = [to_np(inp), to_np(out_ours), to_np(out_orig), to_np(gt)]
    else:
        # 3 格：Input | Ours | Original
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        titles = ['Input', 'Ours (LoLI-Street)', 'Original (LOLv1.h5)']
        images = [to_np(inp), to_np(out_ours), to_np(out_orig)]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{name}_comparison.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return save_path


# ======================== Dataset Configs ========================

DATASET_PATHS = {
    'LOLv1': {
        'input': './data/LOLv1/Test/input/*.png',
        'target': './data/LOLv1/Test/target/*.png',
    },
    'LOLv2_Real': {
        'input': './data/LOLv2/Real_captured/Test/Low/*.png',
        'target': './data/LOLv2/Real_captured/Test/Normal/*.png',
    },
    'LOLv2_Synthetic': {
        'input': './data/LOLv2/Synthetic/Test/Low/*.png',
        'target': './data/LOLv2/Synthetic/Test/Normal/*.png',
    },
    'LOLI-Street': {
        'input': '../dataset/LoLI-Street Dataset/Val/low/*.jpg',
        'target': '../dataset/LoLI-Street Dataset/Val/high/*.jpg',
    },
}


# ======================== Main ========================

def main():
    p = argparse.ArgumentParser(description='LYT-Net Model Comparison Tool')

    # 模式選擇（二擇一）
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dataset', choices=list(DATASET_PATHS.keys()),
                      help='使用預設資料集（有 GT，4 格對比）')
    mode.add_argument('--images', nargs='+',
                      help='指定圖片路徑（無 GT，3 格對比），支援 glob')

    # 模型權重與架構
    p.add_argument('--ours', default='./experiments/loli_scratch_v1_5epoch/best.h5',
                   help='自訓模型權重路徑')
    p.add_argument('--orig', default='./pretrained_weights/LOLv1.h5',
                   help='原始模型權重路徑')
    p.add_argument('--ours_arch', choices=['original', 'modified'], default='modified',
                   help='自訓模型架構（預設 modified = model_modify.arch）')
    p.add_argument('--orig_arch', choices=['original', 'modified'], default='original',
                   help='原始模型架構（預設 original = model.arch）')

    # 輸出設定
    p.add_argument('--out', default=None,
                   help='輸出資料夾（預設自動命名）')
    p.add_argument('--max_dim', type=int, default=512,
                   help='推論最大解析度（避免 OOM，預設 512）')
    p.add_argument('--num', type=int, default=None,
                   help='最多處理幾張（預設全部）')

    args = p.parse_args()

    # 決定輸出目錄
    if args.out:
        out_dir = args.out
    elif args.dataset:
        out_dir = f'./results/comparison_{args.dataset}'
    else:
        out_dir = './results/comparison_custom'
    os.makedirs(out_dir, exist_ok=True)

    # 載入模型
    print('Loading models...')
    print(f'  Ours:     {args.ours}  (arch={args.ours_arch})')
    print(f'  Original: {args.orig}  (arch={args.orig_arch})')
    model_ours = build_model(args.ours, arch=args.ours_arch)
    model_orig = build_model(args.orig, arch=args.orig_arch)

    if args.dataset:
        # ===== 資料集模式（有 GT）=====
        paths = DATASET_PATHS[args.dataset]
        inputs = sorted(glob.glob(paths['input']))
        targets = sorted(glob.glob(paths['target']))
        assert len(inputs) == len(targets), \
            f'Mismatch: {len(inputs)} inputs vs {len(targets)} targets'

        if args.num:
            step = max(1, len(inputs) // args.num)
            indices = list(range(0, len(inputs), step))[:args.num]
        else:
            indices = list(range(len(inputs)))

        print(f'\nDataset: {args.dataset} ({len(inputs)} images, showing {len(indices)})')
        for i in indices:
            name = os.path.splitext(os.path.basename(inputs[i]))[0]
            print(f'  [{i+1}/{len(inputs)}] {name}...', end=' ')
            inp = load_img(inputs[i])
            gt = load_img(targets[i])
            path = infer_and_plot(inp, model_ours, model_orig, gt, name, out_dir, args.max_dim)
            print(f'→ {path}')

    else:
        # ===== 自訂圖片模式（無 GT）=====
        # 展開 glob patterns
        all_files = []
        for pattern in args.images:
            expanded = glob.glob(pattern)
            if expanded:
                all_files.extend(expanded)
            elif os.path.isfile(pattern):
                all_files.append(pattern)
        all_files = sorted(set(all_files))

        if args.num:
            all_files = all_files[:args.num]

        print(f'\nCustom images: {len(all_files)} files')
        for i, img_path in enumerate(all_files):
            name = os.path.splitext(os.path.basename(img_path))[0]
            print(f'  [{i+1}/{len(all_files)}] {name}...', end=' ')
            inp = load_img(img_path)
            path = infer_and_plot(inp, model_ours, model_orig, None, name, out_dir, args.max_dim)
            print(f'→ {path}')

    print(f'\nDone! Results saved to {out_dir}/')


if __name__ == '__main__':
    main()
