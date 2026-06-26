"""
Evaluate all dataset-ablation checkpoints and print a paper-ready table.

Usage (from TensorFlow/ directory):
    python scripts/eval_dataset_ablation.py --auto
"""

import os, sys, glob, argparse
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

DATASETS = {
    'LOLv1': {
        'low':    os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'input',  '*.png'),
        'high':   os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'target', '*.png'),
        'loader': 'lol',
    },
    'LOLv2_Real': {
        'low':    os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Low',    '*.png'),
        'high':   os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Normal', '*.png'),
        'loader': 'lol',
    },
    'LOLI-Street': {
        'low':    os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'low',  '*.jpg'),
        'high':   os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'high', '*.jpg'),
        'loader': 'loli',
    },
}

TABLE_ORDER = ['D1', 'D2', 'D3', 'D1D2', 'D1D3', 'D2D3', 'D1D2D3']
COMBO_LABEL = {
    'D1':     'D1 (LOLv1)',
    'D2':     'D2 (LOLv2-Real)',
    'D3':     'D3 (LOLI-Street)',
    'D1D2':   'D1 + D2',
    'D1D3':   'D1 + D3',
    'D2D3':   'D2 + D3',
    'D1D2D3': 'D1 + D2 + D3 (Ours)',
}


def best_h5(directory):
    candidates = glob.glob(os.path.join(directory, '*.h5'))
    if not candidates:
        return None
    def _psnr(p):
        try: return float(os.path.basename(p).split('psnr_')[1].split('_')[0])
        except: return 0.0
    return max(candidates, key=_psnr)


def build_model(weights_path):
    from model_modify.arch import LYT, Denoiser
    d_cb = Denoiser(16); d_cr = Denoiser(16)
    m = LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m(tf.zeros((1, 256, 256, 3)), training=False)
    m.load_weights(weights_path)
    return m


def eval_model(model, ds_cfg, lpips_fn):
    if ds_cfg['loader'] == 'loli':
        ds = dl.get_loli_datasets_metrics(ds_cfg['low'], ds_cfg['high'], crop_margin=0)
    else:
        ds = dl.get_datasets_metrics(ds_cfg['low'], ds_cfg['high'], 0)
    psnrs, ssims, lpipss = [], [], []
    for raw, gt in ds:
        pred = tf.clip_by_value((model(raw, training=False) + 1.0) / 2.0, 0.0, 1.0)
        gt01 = (gt + 1.0) / 2.0
        psnrs.append(float(tf.reduce_mean(tf.image.psnr(gt01, pred, max_val=1.0))))
        ssims.append(float(tf.reduce_mean(tf.image.ssim(gt01, pred, max_val=1.0))))
        p_pt = torch.from_numpy(pred.numpy()).permute(0, 3, 1, 2).float()
        t_pt = torch.from_numpy(gt01.numpy()).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            lpipss.append(lpips_fn(p_pt, t_pt).item())
    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss)


def print_table(results):
    DS  = list(DATASETS.keys())
    sep = '-' * 108

    print(f"\n{sep}")
    print(f"{'Training Data':<22}  {'LOLv1':^30}  {'LOLv2_Real':^30}  {'LOLI-Street':^30}")
    print(f"{'':22}  {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}  "
          f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}  "
          f"{'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(sep)

    def _has(v, ds):
        return v in results and results[v].get(ds) is not None

    best = {}
    for ds in DS:
        best[(ds,'psnr')]  = max((results[v][ds][0] for v in TABLE_ORDER if _has(v,ds)), default=0)
        best[(ds,'ssim')]  = max((results[v][ds][1] for v in TABLE_ORDER if _has(v,ds)), default=0)
        best[(ds,'lpips')] = min((results[v][ds][2] for v in TABLE_ORDER if _has(v,ds)), default=99)

    def fmt(val, key, better='higher'):
        if val is None: return f"{'N/A':>8}"
        mark = '*' if (better=='higher' and abs(val-best[key])<1e-4) or \
                      (better=='lower'  and abs(val-best[key])<1e-4) else ' '
        return f"{val:7.2f}{mark}"

    for v in TABLE_ORDER:
        if v not in results: continue
        label = COMBO_LABEL[v]
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

    csv_path = os.path.join(root_dir, 'experiments', 'dataset_ablation_results.csv')
    with open(csv_path, 'w') as f:
        f.write('training_data,dataset,psnr,ssim,lpips\n')
        for v in TABLE_ORDER:
            if v not in results: continue
            for ds in DS:
                r = results[v].get(ds)
                if r is not None:
                    f.write(f'{COMBO_LABEL[v]},{ds},{r[0]:.4f},{r[1]:.4f},{r[2]:.4f}\n')
    print(f"\n→ CSV saved: {csv_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--auto', action='store_true')
    args = p.parse_args()

    weight_map = {}
    if args.auto:
        # D1 is the existing Ours model trained on LOLv1
        w = best_h5(os.path.join(root_dir, 'experiments', 'dataset_abl_D1'))
        if not w:
            # Fall back to ablation_d1d2d3 (Ours arch trained on LOLv1)
            w = best_h5(os.path.join(root_dir, 'experiments', 'ablation_d1d2d3'))
        if w: weight_map['D1'] = w

        for combo in ['D2', 'D3', 'D1D2', 'D1D3', 'D2D3', 'D1D2D3']:
            w = best_h5(os.path.join(root_dir, 'experiments', f'dataset_abl_{combo}'))
            if w: weight_map[combo] = w

    if not weight_map:
        print("No checkpoints found. Run train_dataset_ablation.py first.")
        sys.exit(1)

    print(f"\nVariants to evaluate: {list(weight_map.keys())}")
    for v, w in weight_map.items():
        print(f"  {COMBO_LABEL.get(v,v):<25} {os.path.basename(w)}")

    orig_stdout = sys.stdout; sys.stdout = open(os.devnull,'w')
    lpips_fn    = lpips_lib.LPIPS(net='alex', version='0.1', lpips=True, eval_mode=True, verbose=False)
    sys.stdout  = orig_stdout

    all_results = {}
    for v, w_path in weight_map.items():
        label = COMBO_LABEL.get(v, v)
        print(f"\n[Evaluating] {label}")
        try:
            model = build_model(w_path)
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
