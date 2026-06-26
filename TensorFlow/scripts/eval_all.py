"""
統一評估腳本：支援 model.arch (原版 LYT-Net) 與 model_modify.arch (改良版 Ours)
對 LOLv1 / LOLv2_Real / LOLI-Street 三個資料集分別計算 PSNR / SSIM / LPIPS。

Usage:
    python scripts/eval_all.py --arch original --weights pretrained_weights/LOLv1.h5
    python scripts/eval_all.py --arch modified --weights experiments/LOLv1/best.h5
    python scripts/eval_all.py --arch modified --weights ... --datasets LOLI-Street --per_image_log
"""
import os, sys, glob, argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.join(script_dir, '..')
sys.path.append(root_dir)
sys.path.append(script_dir)   # fix: data_loading.py lives in scripts/

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import torch
import lpips as lpips_lib

import data_loading as dl
from find_gamma import find_optimal_gamma, adjust_gamma

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


def build(arch):
    if arch == 'original':
        from model.arch import LYT, Denoiser
    else:
        from model_modify.arch import LYT, Denoiser
    d_cb = Denoiser(16)
    d_cr = Denoiser(16)
    d_cb.build(input_shape=(None, None, None, 1))
    d_cr.build(input_shape=(None, None, None, 1))
    m = LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m.build(input_shape=(None, None, None, 3))
    return m


def eval_one(model, ds_cfg, lpips_model, per_image_log=False, ds_name='', gtmean=False):
    low_glob  = ds_cfg['low']
    high_glob = ds_cfg['high']

    if ds_cfg['loader'] == 'loli':
        test_ds   = dl.get_loli_datasets_metrics(low_glob, high_glob, crop_margin=0)
        file_list = sorted(glob.glob(low_glob))
    else:
        test_ds   = dl.get_datasets_metrics(low_glob, high_glob, 0)
        file_list = sorted(glob.glob(low_glob))

    psnrs, ssims, lpipss = [], [], []
    per_image = []

    for i, (raw, gt) in enumerate(test_ds):
        pred   = model(raw)
        pred   = tf.clip_by_value((pred + 1.0) / 2.0, 0.0, 1.0)
        gt01   = (gt + 1.0) / 2.0

        if gtmean:
            gamma_values = np.linspace(0.4, 2.5, 100)
            optimal_gamma = find_optimal_gamma(pred, gt01, gamma_values, 1.0)
            pred = adjust_gamma(pred, optimal_gamma)

        p = float(tf.reduce_mean(tf.image.psnr(gt01, pred, max_val=1.0)))
        s = float(tf.reduce_mean(tf.image.ssim(gt01, pred, max_val=1.0)))

        p_pt = torch.from_numpy(pred.numpy()).permute(0, 3, 1, 2).float()
        t_pt = torch.from_numpy(gt01.numpy()).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            l = lpips_model(p_pt, t_pt).item()

        psnrs.append(p)
        ssims.append(s)
        lpipss.append(l)

        if per_image_log:
            fname = os.path.basename(file_list[i]) if i < len(file_list) else f'img_{i}'
            per_image.append((fname, p, s, l))

    if per_image_log:
        os.makedirs(os.path.join(root_dir, 'experiments'), exist_ok=True)
        out_csv = os.path.join(root_dir, 'experiments', f'per_image_{ds_name}.csv')
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
                   help='Save per-image PSNR/SSIM/LPIPS CSV')
    p.add_argument('--gtmean', action='store_true',
                   help='Apply GT-mean gamma scan (matches official LYT-Net eval)')
    args = p.parse_args()

    model = build(args.arch)
    model.load_weights(args.weights)

    orig_stdout = sys.stdout
    sys.stdout  = open(os.devnull, 'w')
    lpips_model = lpips_lib.LPIPS(net='alex')
    sys.stdout  = orig_stdout

    print(f"\n{'='*60}")
    print(f"Arch: {args.arch}  |  Weights: {args.weights}")
    print(f"{'='*60}")
    print(f"{'Dataset':18s} | {'PSNR':>8s} | {'SSIM':>8s} | {'LPIPS':>8s}")
    print('-' * 55)

    results = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"{name:18s} | UNKNOWN dataset")
            continue
        try:
            psnr, ssim, lp = eval_one(
                model, DATASETS[name], lpips_model,
                per_image_log=args.per_image_log, ds_name=name, gtmean=args.gtmean)
            print(f"{name:18s} | {psnr:8.2f} | {ssim:8.4f} | {lp:8.4f}")
            results[name] = (psnr, ssim, lp)
        except Exception as e:
            print(f"{name:18s} | FAILED: {e}")
            results[name] = None

    os.makedirs(os.path.join(root_dir, 'experiments'), exist_ok=True)
    arch_tag = 'original' if args.arch == 'original' else 'modified'
    out_csv  = os.path.join(root_dir, 'experiments', f'eval_{arch_tag}.csv')
    with open(out_csv, 'w') as f:
        f.write('dataset,psnr,ssim,lpips\n')
        for name in args.datasets:
            if name not in results or results[name] is None:
                f.write(f'{name},N/A,N/A,N/A\n')
            else:
                pv, sv, lv = results[name]
                f.write(f'{name},{pv:.4f},{sv:.4f},{lv:.4f}\n')
    print(f"\nSaved → {out_csv}")
