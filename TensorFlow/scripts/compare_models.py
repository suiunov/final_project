"""
Compare Models: 新訓練模型 vs 原始預訓練模型
==============================================
在同一組測試影像上評估多個模型的 PSNR / SSIM，
並產生視覺對比圖。

Usage:
    conda activate LYTNet
    cd TensorFlow
    python scripts/compare_models.py
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import data_loading as dl
from model.arch import LYT, Denoiser


def build_model():
    """Build a fresh LYT-Net model instance."""
    d_cb = Denoiser(16)
    d_cr = Denoiser(16)
    d_cb.build(input_shape=(None, None, None, 1))
    d_cr.build(input_shape=(None, None, None, 1))
    m = LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m.build(input_shape=(None, None, None, 3))
    return m


def evaluate_model(model, test_ds, name):
    """Evaluate PSNR and SSIM on a test dataset."""
    psnr_list = []
    ssim_list = []
    for raw, gt in test_ds:
        pred = model(raw)
        pred = (pred + 1.0) / 2.0
        gt   = (gt + 1.0)   / 2.0
        pred = tf.clip_by_value(pred, 0, 1)

        p = tf.image.psnr(gt, pred, max_val=1.0)
        s = tf.image.ssim(gt, pred, max_val=1.0)
        psnr_list.append(float(tf.reduce_mean(p)))
        ssim_list.append(float(tf.reduce_mean(s)))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f'  [{name:20s}]  PSNR={avg_psnr:.2f} dB  |  SSIM={avg_ssim:.4f}')
    return avg_psnr, avg_ssim, psnr_list, ssim_list


def save_comparison_image(raw, preds_dict, gt, save_path, idx):
    """Save a side-by-side comparison of multiple models on a single image."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [warning] matplotlib not installed, skipping visual comparison')
        return

    n_models = len(preds_dict)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4))

    def to_uint8(t):
        t = (t + 1.0) / 2.0
        return tf.clip_by_value(t * 255, 0, 255).numpy().astype(np.uint8)

    # Input
    axes[0].imshow(to_uint8(raw[0]))
    axes[0].set_title('Input (Low-light)', fontsize=10)
    axes[0].axis('off')

    # Model outputs
    for i, (name, pred) in enumerate(preds_dict.items()):
        pred_clipped = tf.clip_by_value(pred, -1, 1)
        axes[i + 1].imshow(to_uint8(pred_clipped[0]))
        axes[i + 1].set_title(name, fontsize=10)
        axes[i + 1].axis('off')

    # GT
    axes[-1].imshow(to_uint8(gt[0]))
    axes[-1].set_title('Ground Truth', fontsize=10)
    axes[-1].axis('off')

    plt.tight_layout()
    fname = os.path.join(save_path, f'compare_{idx:04d}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()


def main(args):
    print('=' * 60)
    print('Model Comparison: Custom vs Pretrained')
    print('=' * 60)

    # ---- Test data ----
    print('\nLoading test data...')
    test_ds = dl.get_loli_datasets_metrics(
        args.test_low, args.test_high, crop_margin=0)

    # ---- Define models to compare ----
    models = {}

    # New model (from scratch)
    if args.custom and os.path.exists(args.custom):
        print(f'Loading custom model: {args.custom}')
        m = build_model()
        m.load_weights(args.custom)
        models['Ours (LoLI-Street)'] = m
    elif args.custom:
        print(f'[SKIP] Custom model not found: {args.custom}')

    # Pretrained models
    pretrained_dir = os.path.join(root_dir, 'pretrained_weights')
    pretrained_files = {
        'LOLv1 (Original)': 'LOLv1.h5',
        'LOLv2-Real (Original)': 'LOLv2_Real.h5',
        'LOLv2-Synth (Original)': 'LOLv2_Synthetic.h5',
    }
    for name, fname in pretrained_files.items():
        fpath = os.path.join(pretrained_dir, fname)
        if os.path.exists(fpath):
            print(f'Loading pretrained: {fpath}')
            m = build_model()
            m.load_weights(fpath)
            models[name] = m
        else:
            print(f'[SKIP] Not found: {fpath}')

    if not models:
        print('No models loaded. Exiting.')
        return

    # ---- Evaluate all models ----
    print(f'\n{"=" * 60}')
    print(f'Evaluating on {args.test_low}')
    print(f'{"=" * 60}')

    results = {}
    for name, m in models.items():
        avg_psnr, avg_ssim, _, _ = evaluate_model(m, test_ds, name)
        results[name] = {'PSNR': avg_psnr, 'SSIM': avg_ssim}

    # ---- Print summary table ----
    print(f'\n{"=" * 60}')
    print(f'{"Model":25s} | {"PSNR (dB)":>10s} | {"SSIM":>8s}')
    print('-' * 50)
    for name, r in sorted(results.items(), key=lambda x: -x[1]['PSNR']):
        print(f'{name:25s} | {r["PSNR"]:10.2f} | {r["SSIM"]:8.4f}')

    # ---- Visual comparison ----
    if args.save_visual:
        out_dir = os.path.join('./experiments', 'comparison')
        os.makedirs(out_dir, exist_ok=True)
        print(f'\nGenerating visual comparisons → {out_dir}/')

        for idx, (raw, gt) in enumerate(test_ds.take(args.n_visual)):
            preds = {name: m(raw) for name, m in models.items()}
            save_comparison_image(raw, preds, gt, out_dir, idx)
        print(f'Saved {args.n_visual} comparison images.')


if __name__ == '__main__':
    dataset_base = os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset')

    p = argparse.ArgumentParser(description='Compare LYT-Net models')
    p.add_argument('--custom',
                   default='./experiments/loli_scratch_v1/best.h5',
                   help='Path to your custom-trained model weights')
    p.add_argument('--test_low',
                   default=os.path.join(dataset_base, 'Val', 'low', '*.jpg'))
    p.add_argument('--test_high',
                   default=os.path.join(dataset_base, 'Val', 'high', '*.jpg'))
    p.add_argument('--save_visual', action='store_true', default=True,
                   help='Save visual comparison images')
    p.add_argument('--n_visual', type=int, default=20,
                   help='Number of visual comparison images to save')
    args = p.parse_args()
    main(args)
