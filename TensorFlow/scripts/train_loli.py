"""
LYT-Net From-Scratch Training on LoLI-Street Dataset
=====================================================
從零訓練 LYT-Net，使用 LoLI-Street 33K 配對街景影像。
訓練完成後可用 compare_models.py 與原始預訓練模型對比。

Usage:
    conda activate LYTNet
    cd TensorFlow
    python scripts/train_loli.py --epochs 200 --tag loli_scratch_v1
"""

import os
import sys
import argparse
import datetime
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import tensorflow as tf
import data_loading as dl
from model.arch import LYT, Denoiser
from model.losses import load_vgg, loss   # 使用原始 loss function，保持一致
from model.scheduler import CosineDecayWithRestartsLearningRateSchedule
from find_gamma import find_optimal_gamma, adjust_gamma

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(1)


def get_time():
    return datetime.datetime.now().strftime("%m-%d %H:%M:%S")


# ======================== Train Step ========================

@tf.function
def train_step(raw_images, corrected_images, model, loss_model, optimizer):
    with tf.GradientTape() as tape:
        generated_images = model(raw_images, training=True)
        loss_val = loss(generated_images, corrected_images, loss_model)
    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del tape
    return loss_val


# ======================== Main ========================

def main(args):
    print('=' * 60)
    print('LYT-Net From-Scratch Training on LoLI-Street')
    print('=' * 60)
    print(f'({get_time()}) Config:')
    print(f'  Tag:        {args.tag}')
    print(f'  Epochs:     {args.epochs}')
    print(f'  Batch size: {args.bs}')
    print(f'  Crop size:  {args.crop}')
    print(f'  LR:         {args.lr}')
    print(f'  GT-Mean:    {not args.no_gtmean}')
    print()

    # ---- Data ----
    print(f'({get_time()}) Loading LoLI-Street training data...')
    train_ds = dl.get_loli_datasets(
        args.train_low, args.train_high,
        crop_size=args.crop, batch_size=args.bs)

    print(f'({get_time()}) Loading LoLI-Street validation data...')
    val_ds = dl.get_loli_datasets_metrics(
        args.val_low, args.val_high, crop_margin=0)

    # ---- Model (from scratch — NO pretrained weights) ----
    print(f'({get_time()}) Building model from scratch...')
    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None, None, None, 1))
    denoiser_cr.build(input_shape=(None, None, None, 1))
    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None, None, None, 3))

    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f'  Total trainable params: {int(n_params):,}')

    # ---- VGG for perceptual loss ----
    loss_model = load_vgg()

    # ---- Scheduler & Optimizer (same as original) ----
    steps_per_epoch = len(train_ds)
    total_steps = args.epochs * steps_per_epoch
    first_decay_steps = 150 * steps_per_epoch
    print(f'  Steps/epoch: {steps_per_epoch}')
    print(f'  Total steps: {total_steps}')

    sched = CosineDecayWithRestartsLearningRateSchedule(
        initial_lr=args.lr, min_lr=1e-6,
        total_steps=total_steps,
        first_decay_steps=first_decay_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=sched)

    # ---- Output directory ----
    exp_dir = os.path.join('./experiments', args.tag)
    os.makedirs(exp_dir, exist_ok=True)

    # ---- Training loop ----
    print(f'\n({get_time()}) Starting training...\n')
    best_gt_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for raw_images, corrected_images in train_ds:
            loss_val = train_step(raw_images, corrected_images,
                                  model, loss_model, optimizer)
            epoch_loss += float(loss_val)
            n_batches += 1
            if n_batches % 500 == 0:
                print(f'\r  [{n_batches:5d}/{steps_per_epoch}] loss={epoch_loss/n_batches:.4f}', end='', flush=True)
        print('\r' + ' ' * 60 + '\r', end='', flush=True)  # clear progress line

        avg_loss = epoch_loss / max(n_batches, 1)

        # ---- Validate (same as original train.py) ----
        total_psnr = 0.0
        total_gt_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        for raw_images, corrected_images in val_ds:
            generated_images = model(raw_images)
            generated_images = (generated_images + 1.0) / 2.0
            corrected_images = (corrected_images + 1.0) / 2.0

            psnr = tf.image.psnr(corrected_images, generated_images, max_val=1.0)

            if not args.no_gtmean:
                gamma_values = np.linspace(0.4, 2.5, 100)
                optimal_gamma = find_optimal_gamma(
                    generated_images, corrected_images, gamma_values, 1.0)
                generated_adj = adjust_gamma(generated_images, optimal_gamma)
                gt_psnr = tf.image.psnr(corrected_images, generated_adj, max_val=1.0)
                ssim = tf.image.ssim(corrected_images, generated_adj, max_val=1.0)
            else:
                gt_psnr = psnr
                ssim = tf.image.ssim(corrected_images, generated_images, max_val=1.0)

            total_psnr += tf.reduce_mean(psnr)
            total_gt_psnr += tf.reduce_mean(gt_psnr)
            total_ssim += tf.reduce_mean(ssim)
            num_samples += 1

        avg_psnr = total_psnr / num_samples
        avg_gt_psnr = total_gt_psnr / num_samples
        avg_ssim = total_ssim / num_samples

        print(f'({get_time()}) Epoch {epoch:03d}/{args.epochs} | '
              f'loss={avg_loss:.4f} | '
              f'PSNR={avg_psnr:.2f} | GT-PSNR={avg_gt_psnr:.2f} | SSIM={avg_ssim:.3f}')

        # ---- Save best model ----
        if avg_gt_psnr > best_gt_psnr:
            best_gt_psnr = avg_gt_psnr
            model_name = (f"net_psnr_{best_gt_psnr:.2f}_ssim_{avg_ssim:.3f}"
                          f"_epoch_{epoch}_dataset_{args.tag}.h5")
            model.save_weights(os.path.join(exp_dir, model_name))
            # Also save as best.h5 for easy reference
            model.save_weights(os.path.join(exp_dir, 'best.h5'))
            print(f'  → Saved best model: {model_name}')

        # ---- Save checkpoint every 50 epochs ----
        if epoch % 50 == 0:
            ckpt_name = f'checkpoint_ep{epoch}.h5'
            model.save_weights(os.path.join(exp_dir, ckpt_name))
            print(f'  → Checkpoint: {ckpt_name}')

    # ---- Final save ----
    model.save_weights(os.path.join(exp_dir, 'final.h5'))
    print(f'\n({get_time()}) Training complete!')
    print(f'  Best GT-PSNR: {best_gt_psnr:.2f}')
    print(f'  Weights saved to: {exp_dir}/')


if __name__ == '__main__':
    # Resolve default paths relative to TensorFlow/ directory
    dataset_base = os.path.join(root_dir, '..', 'dataset', 'LoLI-Street Dataset')

    p = argparse.ArgumentParser(description='LYT-Net from-scratch on LoLI-Street')
    p.add_argument('--tag',        default='loli_scratch_v1',
                   help='Experiment name (saves to ./experiments/<tag>/)')
    p.add_argument('--train_low',
                   default=os.path.join(dataset_base, 'Train', 'low', '*.jpg'))
    p.add_argument('--train_high',
                   default=os.path.join(dataset_base, 'Train', 'high', '*.jpg'))
    p.add_argument('--val_low',
                   default=os.path.join(dataset_base, 'Val', 'low', '*.jpg'))
    p.add_argument('--val_high',
                   default=os.path.join(dataset_base, 'Val', 'high', '*.jpg'))
    p.add_argument('--epochs',     type=int, default=200)
    p.add_argument('--bs',         type=int, default=2)
    p.add_argument('--crop',       type=int, default=256)
    p.add_argument('--lr',         type=float, default=2e-4,
                   help='Initial learning rate (same as original paper)')
    p.add_argument('--no_gtmean',  action='store_true',
                   help='Disable GT-Mean PSNR (faster validation, skip gamma search)')
    args = p.parse_args()
    main(args)
