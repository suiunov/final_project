"""
Ablation training script — supports all 7 variants.

Usage (run from TensorFlow/ directory):
    python scripts/train_ablation.py --variant d1
    python scripts/train_ablation.py --variant d2
    python scripts/train_ablation.py --variant d3
    python scripts/train_ablation.py --variant d1d2
    python scripts/train_ablation.py --variant d1d3
    python scripts/train_ablation.py --variant d2d3
    python scripts/train_ablation.py --variant d1d2d3
    python scripts/train_ablation.py --variant all     # all 6 new variants sequentially

Weights saved to:  experiments/ablation_<variant>/
"""

import os, sys, argparse, datetime
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.join(script_dir, '..')
sys.path.append(root_dir)
sys.path.append(script_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.random.set_seed(1)

import data_loading as dl
from model_modify.losses import load_vgg, loss
from model_modify.scheduler import CosineDecayWithRestartsLearningRateSchedule
from find_gamma import find_optimal_gamma, adjust_gamma

LOLv1_TRAIN_LOW  = os.path.join(root_dir, 'data', 'LOLv1', 'Train', 'input',  '*.png')
LOLv1_TRAIN_HIGH = os.path.join(root_dir, 'data', 'LOLv1', 'Train', 'target', '*.png')
LOLv1_TEST_LOW   = os.path.join(root_dir, 'data', 'LOLv1', 'Test',  'input',  '*.png')
LOLv1_TEST_HIGH  = os.path.join(root_dir, 'data', 'LOLv1', 'Test',  'target', '*.png')

TOTAL_EPOCHS = 10

# ── Variant registry ───────────────────────────────────────────────────────────
VARIANT_MAP = {
    'd1':     'model_modify.arch_d1_only',
    'd2':     'model_modify.arch_d2_only',
    'd3':     'model_modify.arch_d3_only',
    'd1d2':   'model_modify.arch_d1d2',
    'd1d3':   'model_modify.arch_d1d3',
    'd2d3':   'model_modify.arch_d2d3',
    'd1d2d3': 'model_modify.arch',          # full model = arch.py
}
ALL_VARIANTS = list(VARIANT_MAP.keys())

VARIANT_LABEL = {
    'd1':     '+D1 only',
    'd2':     '+D2 only',
    'd3':     '+D3 only',
    'd1d2':   '+D1+D2',
    'd1d3':   '+D1+D3',
    'd2d3':   '+D2+D3',
    'd1d2d3': 'D1+D2+D3 (Ours)',
}


def get_time():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def build_model(variant):
    import importlib
    mod = importlib.import_module(VARIANT_MAP[variant])
    LYT, Denoiser = mod.LYT, mod.Denoiser
    d_cb = Denoiser(16)
    d_cr = Denoiser(16)
    m = LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
    m(tf.zeros((1, 256, 256, 3)), training=False)
    return m


@tf.function
def train_step(raw, gt, model, loss_model, optimizer):
    with tf.GradientTape() as tape:
        pred     = model(raw, training=True)
        loss_val = loss(gt, pred, loss_model)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_val


def run_training(variant):
    label    = VARIANT_LABEL[variant]
    save_dir = os.path.join(root_dir, 'experiments', f'ablation_{variant}')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{get_time()}] Training variant: {label}")
    print(f"{'='*60}")

    train_ds   = dl.get_datasets(LOLv1_TRAIN_LOW, LOLv1_TRAIN_HIGH)
    test_ds    = dl.get_datasets_metrics(LOLv1_TEST_LOW, LOLv1_TEST_HIGH, 0)
    model      = build_model(variant)
    loss_model = load_vgg()

    initial_lr        = 2e-4
    min_lr            = 1e-6
    steps_per_epoch   = len(train_ds)
    total_steps       = TOTAL_EPOCHS * steps_per_epoch
    first_decay_steps = 150 * steps_per_epoch

    lr_schedule = CosineDecayWithRestartsLearningRateSchedule(
        initial_lr, min_lr, total_steps, first_decay_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    best_gt_psnr = 0.0
    best_ckpt    = None

    for epoch in range(1, TOTAL_EPOCHS + 1):
        for raw, gt in train_ds:
            loss_val = train_step(raw, gt, model, loss_model, optimizer)

        total_psnr = total_ssim = total_gt_psnr = 0.0
        n = 0
        for raw, gt in test_ds:
            pred = model(raw, training=False)
            pred = tf.clip_by_value((pred + 1.0) / 2.0, 0.0, 1.0)
            gt01 = (gt + 1.0) / 2.0
            psnr = float(tf.reduce_mean(tf.image.psnr(gt01, pred, max_val=1.0)))
            opt_g   = find_optimal_gamma(pred, gt01, np.linspace(0.4,2.5,100), 1.0)
            pred_g  = adjust_gamma(pred, opt_g)
            gt_psnr = float(tf.reduce_mean(tf.image.psnr(gt01, pred_g, max_val=1.0)))
            ssim    = float(tf.reduce_mean(tf.image.ssim(gt01, pred_g, max_val=1.0)))
            total_psnr += psnr; total_gt_psnr += gt_psnr; total_ssim += ssim; n += 1

        avg_psnr    = total_psnr    / n
        avg_gt_psnr = total_gt_psnr / n
        avg_ssim    = total_ssim    / n

        print(f"[{get_time()}] Epoch {epoch:4d} | GT-PSNR: {avg_gt_psnr:.2f} | "
              f"PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f} | loss={loss_val:.6f}")

        if avg_gt_psnr > best_gt_psnr:
            best_gt_psnr = avg_gt_psnr
            ckpt_name    = f"net_{variant}_psnr_{avg_gt_psnr:.2f}_ssim_{avg_ssim:.3f}_epoch_{epoch}.weights.h5"
            model.save_weights(os.path.join(save_dir, ckpt_name))
            best_ckpt = ckpt_name
            print(f"  → Saved: {ckpt_name}")

    print(f"\n[{get_time()}] Finished {label}. Best GT-PSNR = {best_gt_psnr:.2f}")
    if best_ckpt is None:
        print(f"  WARNING: no checkpoint saved for {label} (all epochs produced NaN?)")
        return None
    return os.path.join(save_dir, best_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',
                        choices=ALL_VARIANTS + ['all'],
                        required=True)
    args = parser.parse_args()

    to_run = ALL_VARIANTS if args.variant == 'all' else [args.variant]
    for v in to_run:
        run_training(v)

    print("\nAll requested variants finished.")
    print("Next: python scripts/eval_ablation.py  (see ABLATION_COLAB_CELLS.md)")
