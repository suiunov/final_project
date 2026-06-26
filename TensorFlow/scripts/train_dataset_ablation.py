"""
Dataset ablation training script.

Trains the Ours architecture (model_modify.arch) on 7 dataset combinations:
  D1        = LOLv1
  D2        = LOLv2-Real
  D3        = LOLI-Street (subsampled to LOLI_SAMPLE images)
  D1+D2     = LOLv1 + LOLv2-Real
  D1+D3     = LOLv1 + LOLI-Street
  D2+D3     = LOLv2-Real + LOLI-Street
  D1+D2+D3  = all three

Usage (from TensorFlow/ directory):
    python scripts/train_dataset_ablation.py --combo D2
    python scripts/train_dataset_ablation.py --combo all
"""

import os, sys, argparse, datetime, glob, random
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

# ── Dataset paths ──────────────────────────────────────────────────────────────
D1_LOW  = os.path.join(root_dir, 'data', 'LOLv1', 'Train', 'input',  '*.png')
D1_HIGH = os.path.join(root_dir, 'data', 'LOLv1', 'Train', 'target', '*.png')

D2_LOW  = os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Train', 'Low',    '*.png')
D2_HIGH = os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Train', 'Normal', '*.png')

D3_LOW  = os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Train', 'low',  '*.jpg')
D3_HIGH = os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Train', 'high', '*.jpg')

# Eval test sets
EVAL_LOLv1 = {
    'low':  os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'input',  '*.png'),
    'high': os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'target', '*.png'),
    'loader': 'lol',
}
EVAL_LOLv2 = {
    'low':  os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Low',    '*.png'),
    'high': os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Normal', '*.png'),
    'loader': 'lol',
}
EVAL_LOLI = {
    'low':  os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'low',  '*.jpg'),
    'high': os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset', 'Val', 'high', '*.jpg'),
    'loader': 'loli',
}

LOLI_SAMPLE = 1000   # subsample D3 to this many pairs (full=30000, too slow)
TOTAL_EPOCHS = 10

ALL_COMBOS = ['D1', 'D2', 'D3', 'D1D2', 'D1D3', 'D2D3', 'D1D2D3']


def get_time():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def sample_glob(low_g, high_g, n):
    """Return (low_files, high_files) sorted and paired, then subsampled to n."""
    lows  = sorted(glob.glob(low_g))
    highs = sorted(glob.glob(high_g))
    assert len(lows) == len(highs), f'Mismatch: {len(lows)} low vs {len(highs)} high'
    if n and len(lows) > n:
        idx   = sorted(random.sample(range(len(lows)), n))
        lows  = [lows[i]  for i in idx]
        highs = [highs[i] for i in idx]
    return lows, highs


def build_combined_dataset(combo):
    """Build a tf.data training dataset for the given combo string."""
    all_low, all_high = [], []

    if 'D1' in combo:
        l, h = sample_glob(D1_LOW, D1_HIGH, None)
        all_low.extend(l); all_high.extend(h)
        print(f'  D1 (LOLv1):      {len(l)} pairs')

    if 'D2' in combo:
        l, h = sample_glob(D2_LOW, D2_HIGH, None)
        all_low.extend(l); all_high.extend(h)
        print(f'  D2 (LOLv2-Real): {len(l)} pairs')

    if 'D3' in combo:
        l, h = sample_glob(D3_LOW, D3_HIGH, LOLI_SAMPLE)
        all_low.extend(l); all_high.extend(h)
        print(f'  D3 (LOLI-Street): {len(l)} pairs  (subsampled from 30 000)')

    print(f'  Total: {len(all_low)} training pairs')

    # Build tf.data pipeline
    ds = tf.data.Dataset.from_tensor_slices((all_low, all_high))
    ds = ds.shuffle(buffer_size=len(all_low), seed=1)
    ds = ds.map(lambda r, g: dl._load_and_preprocess_loli(r, g, 256),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    from model_modify.arch import LYT, Denoiser
    d_cb = Denoiser(16); d_cr = Denoiser(16)
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


def eval_on_dataset(model, ds_cfg):
    if ds_cfg['loader'] == 'loli':
        ds = dl.get_loli_datasets_metrics(ds_cfg['low'], ds_cfg['high'], crop_margin=0)
    else:
        ds = dl.get_datasets_metrics(ds_cfg['low'], ds_cfg['high'], 0)
    psnrs, ssims = [], []
    for raw, gt in ds:
        pred  = tf.clip_by_value((model(raw, training=False) + 1.0) / 2.0, 0.0, 1.0)
        gt01  = (gt + 1.0) / 2.0
        opt_g = find_optimal_gamma(pred, gt01, np.linspace(0.4, 2.5, 100), 1.0)
        pred_g = adjust_gamma(pred, opt_g)
        psnrs.append(float(tf.reduce_mean(tf.image.psnr(gt01, pred_g, max_val=1.0))))
        ssims.append(float(tf.reduce_mean(tf.image.ssim(gt01, pred_g, max_val=1.0))))
    return np.mean(psnrs), np.mean(ssims)


def run_training(combo):
    save_dir = os.path.join(root_dir, 'experiments', f'dataset_abl_{combo}')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{get_time()}] Dataset combo: {combo}")
    print(f"{'='*60}")

    train_ds   = build_combined_dataset(combo)
    model      = build_model()
    loss_model = load_vgg()

    steps_per_epoch   = len(train_ds)
    total_steps       = TOTAL_EPOCHS * steps_per_epoch
    first_decay_steps = max(150 * steps_per_epoch, total_steps)

    lr_schedule = CosineDecayWithRestartsLearningRateSchedule(
        2e-4, 1e-6, total_steps, first_decay_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    best_psnr = 0.0
    best_ckpt = None

    # Use LOLv1 test set for checkpoint selection (consistent across all combos)
    ref_ds = EVAL_LOLv1

    for epoch in range(1, TOTAL_EPOCHS + 1):
        loss_val = None
        for raw, gt in train_ds:
            loss_val = train_step(raw, gt, model, loss_model, optimizer)

        psnr, ssim = eval_on_dataset(model, ref_ds)
        print(f"[{get_time()}] Epoch {epoch:3d} | LOLv1 GT-PSNR: {psnr:.2f} | SSIM: {ssim:.3f} | loss={loss_val:.6f}")

        if psnr > best_psnr:
            best_psnr = psnr
            ckpt_name = f"net_{combo}_psnr_{psnr:.2f}_ssim_{ssim:.3f}_epoch_{epoch}.weights.h5"
            model.save_weights(os.path.join(save_dir, ckpt_name))
            best_ckpt = ckpt_name
            print(f"  → Saved: {ckpt_name}")

    print(f"\n[{get_time()}] Finished {combo}. Best LOLv1 GT-PSNR = {best_psnr:.2f}")
    if best_ckpt is None:
        print(f"  WARNING: no checkpoint saved for {combo}")
        return None
    return os.path.join(save_dir, best_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', choices=ALL_COMBOS + ['all'], required=True)
    args = parser.parse_args()

    combos = ALL_COMBOS if args.combo == 'all' else [args.combo.upper()]
    for c in combos:
        run_training(c)

    print("\nAll combos finished. Next: python scripts/eval_dataset_ablation.py --auto")
