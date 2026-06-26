"""
Evaluate all ablation variants and print a paper-ready table.

Usage (from TensorFlow/ directory) — simplest form, auto-finds all checkpoints:
    python scripts/eval_ablation.py --auto

Or with explicit weight paths:
    python scripts/eval_ablation.py \
        --baseline_weights pretrained_weights/LOLv1.h5 \
        --full_weights     experiments/LOLv1/net_psnr_26.41_ssim_0.852_epoch_156_dataset_LOLv1.h5
    (single/combo variants are picked up automatically from experiments/ablation_*/）
"""

import os, sys, glob, argparse, importlib
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.join(script_dir, '..')
sys.path.append(root_dir)
sys.path.append(script_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import torch
import lpips as lpips_lib

import data_loading as dl
from find_gamma import find_optimal_gamma, adjust_gamma

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASETS = {
    'LOLv1': {
        'low':  os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'input', '*.png'),
        'high': os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'target', '*.png'),
        'loader': 'lol',
    },
    'LOLv2_Real': {
        'low':  os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Low', '*.png'),
        'high': os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Normal', '*.png'),
        'loader': 'lol',
    },
    'LOLI-Street': {
        'low':  os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'low', '*.jpg'),
        'high': os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'high', '*.jpg'),
        'loader': 'loli',
    },
}

# ── Variant → module mapping ───────────────────────────────────────────────────
VARIANT_MODULE = {
    'baseline': 'model.arch',
    'd1':       'model_modify.arch_d1_only',
    'd2':       'model_modify.arch_d2_only',
    'd3':       'model_modify.arch_d3_only',
    'd1d2':     'model_modify.arch_d1d2',
    'd1d3':     'model_modify.arch_d1d3',
    'd2d3':     'model_modify.arch_d2d3',
    'd1d2d3':   'model_modify.arch',
}

VARIANT_LABEL = {
    'baseline': 'Baseline (no mod.)',
    'd1':       '+D1 only',
    'd2':       '+D2 only',
    'd3':       '+D3 only',
    'd1d2':     '+D1+D2',
    'd1d3':     '+D1+D3',
    'd2d3':     '+D2+D3',
    'd1d2d3':   'D1+D2+D3 (Ours)',
}

# Display order for the paper table
TABLE_ORDER = ['baseline', 'd1', 'd2', 'd3', 'd1d2', 'd1d3', 'd2d3', 'd1d2d3']


def best_h5(path_or_dir):
    """Return best .h5 checkpoint: accepts a file path or a directory."""
    if os.path.isfile(path_or_dir) and path_or_dir.endswith('.h5'):
        return path_or_dir
    candidates = glob.glob(os.path.join(path_or_dir, '*.h5'))
    if not candidates:
        return None
    def _psnr(p):
        try: return float(os.path.basename(p).split('psnr_')[1].split('_')[0])
        except: return 0.0
    return max(candidates, key=_psnr)


def build_model(variant, weights_path):
    mod = importlib.import_module(VARIANT_MODULE[variant])
    d_cb = mod.Denoiser(16)
    d_cr = mod.Denoiser(16)
    m = mod.LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m(tf.zeros((1, 256, 256, 3)), training=False)
    m.load_weights(weights_path)
    return m


def eval_model(model, ds_cfg, lpips_fn):
    low_g = ds_cfg['low']; high_g = ds_cfg['high']
    ds = (dl.get_loli_datasets_metrics(low_g, high_g, crop_margin=0)
          if ds_cfg['loader'] == 'loli'
          else dl.get_datasets_metrics(low_g, high_g, 0))
    psnrs, ssims, lpipss = [], [], []
    for raw, gt in ds:
        pred = tf.clip_by_value((model(raw, training=False) + 1.0) / 2.0, 0.0, 1.0)
        gt01 = (gt + 1.0) / 2.0
        p = float(tf.reduce_mean(tf.image.psnr(gt01, pred, max_val=1.0)))
        s = float(tf.reduce_mean(tf.image.ssim(gt01, pred, max_val=1.0)))
        p_pt = torch.from_numpy(pred.numpy()).permute(0,3,1,2).float()
        t_pt = torch.from_numpy(gt01.numpy()).permute(0,3,1,2).float()
        with torch.no_grad(): l = lpips_fn(p_pt, t_pt).item()
        psnrs.append(p); ssims.append(s); lpipss.append(l)
    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss)


def print_table(results):
    """results: dict  variant → {ds_name: (psnr,ssim,lpips) or None}"""
    DS = list(DATASETS.keys())
    sep = '-' * 106

    # Header
    print(f"\n{sep}")
    print(f"{'Variant':<22}  {'LOLv1':^30}  {'LOLv2_Real':^30}  {'LOLI-Street':^30}")
    print(f"{'':22}  {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}  "
          f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}  "
          f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(sep)

    # Find best per column for bolding (printed with asterisk)
    def _has(v, ds):
        return v in results and results[v].get(ds) is not None
    best = {}
    for ds in DS:
        best[(ds,'psnr')] = max((results[v][ds][0] for v in TABLE_ORDER if _has(v, ds)), default=0)
        best[(ds,'ssim')] = max((results[v][ds][1] for v in TABLE_ORDER if _has(v, ds)), default=0)
        best[(ds,'lpips')]= min((results[v][ds][2] for v in TABLE_ORDER if _has(v, ds)), default=99)

    def fmt(val, key, better='higher'):
        if val is None: return f"{'N/A':>8}"
        mark = '*' if (better == 'higher' and abs(val - best[key]) < 1e-4) or \
                      (better == 'lower'  and abs(val - best[key]) < 1e-4) else ' '
        return f"{val:7.2f}{mark}"

    for v in TABLE_ORDER:
        if v not in results: continue
        label = VARIANT_LABEL[v]
        line  = f"{label:<22}"
        for ds in DS:
            r = results[v].get(ds)
            if r is not None:
                p, s, l = r
                line += (f"  {fmt(p,(ds,'psnr'),'higher'):>9}"
                         f" {fmt(s,(ds,'ssim'),'higher'):>9}"
                         f" {fmt(l,(ds,'lpips'),'lower'):>9}")
            else:
                line += "  " + "     N/A " * 3
        print(line)
    print(sep)
    print("* = best in column")

    # Save CSV
    exp_dir  = os.path.join(root_dir, 'experiments')
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, 'ablation_results.csv')
    with open(csv_path, 'w') as f:
        f.write('variant,dataset,psnr,ssim,lpips\n')
        for v in TABLE_ORDER:
            if v not in results: continue
            for ds in DS:
                r = results[v].get(ds)
                if r is not None:
                    f.write(f'{VARIANT_LABEL[v]},{ds},{r[0]:.4f},{r[1]:.4f},{r[2]:.4f}\n')
    print(f"\n→ CSV saved: {csv_path}")
    return csv_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--auto', action='store_true',
                   help='Auto-discover all checkpoints in experiments/ablation_*/')
    p.add_argument('--baseline_weights', default=None)
    p.add_argument('--full_weights',     default=None)
    # individual overrides (optional — --auto handles these automatically)
    for v in ['d1','d2','d3','d1d2','d1d3','d2d3','d1d2d3']:
        p.add_argument(f'--{v}_weights', default=None)
    args = p.parse_args()

    # Build weight map
    weight_map = {}

    if args.baseline_weights:
        weight_map['baseline'] = args.baseline_weights
    else:
        w = best_h5(os.path.join(root_dir, 'pretrained_weights', 'LOLv1.h5'))
        if w: weight_map['baseline'] = w

    if args.full_weights:
        weight_map['d1d2d3'] = args.full_weights
    elif getattr(args, 'd1d2d3_weights', None):
        weight_map['d1d2d3'] = getattr(args, 'd1d2d3_weights')
    else:
        w = best_h5(os.path.join(root_dir, 'experiments', 'ablation_d1d2d3'))
        if w: weight_map['d1d2d3'] = w
        else:
            # fall back to the existing LOLv1 experiment folder
            w = best_h5(os.path.join(root_dir, 'experiments', 'LOLv1'))
            if w: weight_map['d1d2d3'] = w

    for v in ['d1','d2','d3','d1d2','d1d3','d2d3']:
        override = getattr(args, f'{v}_weights', None)
        if override:
            weight_map[v] = override
        elif args.auto:
            w = best_h5(os.path.join(root_dir, 'experiments', f'ablation_{v}'))
            if w: weight_map[v] = w

    if not weight_map:
        print("No weights found. Run train_ablation.py first, then re-run with --auto.")
        sys.exit(1)

    print(f"\nVariants to evaluate: {list(weight_map.keys())}")
    for v, w in weight_map.items():
        print(f"  {VARIANT_LABEL.get(v,v):<25} {w}")

    # Load LPIPS
    orig_stdout = sys.stdout; sys.stdout = open(os.devnull,'w')
    lpips_fn    = lpips_lib.LPIPS(net='alex', version='0.1', lpips=True, eval_mode=True, verbose=False)
    sys.stdout  = orig_stdout

    # Evaluate
    all_results = {}
    for v, w_path in weight_map.items():
        label = VARIANT_LABEL.get(v, v)
        print(f"\n[Evaluating] {label}")
        try:
            model = build_model(v, w_path)
        except Exception as e:
            print(f"  Failed to build model: {e}"); continue
        all_results[v] = {}
        for ds_name, ds_cfg in DATASETS.items():
            try:
                psnr, ssim, lp = eval_model(model, ds_cfg, lpips_fn)
                all_results[v][ds_name] = (psnr, ssim, lp)
                print(f"  {ds_name:<18} PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lp:.4f}")
            except Exception as e:
                print(f"  {ds_name:<18} FAILED: {e}")
                all_results[v][ds_name] = None

    print_table(all_results)
