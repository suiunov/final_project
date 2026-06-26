"""
Generate side-by-side visual comparison for all ablation variants.

Usage (from TensorFlow/ directory):
    python scripts/gen_ablation_visual.py --dataset LOLv1 --num 5
    python scripts/gen_ablation_visual.py --dataset LOLv1 --num 5 --out ./results/ablation_visual
"""
import os, sys, glob, argparse, importlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir   = os.path.join(script_dir, '..')
sys.path.insert(0, root_dir)
sys.path.insert(0, script_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

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
    'baseline': 'Baseline',
    'd1':       '+D1',
    'd2':       '+D2',
    'd3':       '+D3',
    'd1d2':     '+D1+D2',
    'd1d3':     '+D1+D3',
    'd2d3':     '+D2+D3',
    'd1d2d3':   'Full (Ours)',
}

DATASET_PATHS = {
    'LOLv1': {
        'input':  os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'input', '*.png'),
        'target': os.path.join(root_dir, 'data', 'LOLv1', 'Test', 'target', '*.png'),
    },
    'LOLv2_Real': {
        'input':  os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Low', '*.png'),
        'target': os.path.join(root_dir, 'data', 'LOLv2', 'Real_captured', 'Test', 'Normal', '*.png'),
    },
}


def best_h5(path_or_dir):
    if os.path.isfile(path_or_dir) and path_or_dir.endswith('.h5'):
        return path_or_dir
    candidates = glob.glob(os.path.join(path_or_dir, '*.h5'))
    if not candidates:
        return None
    def _psnr(p):
        try: return float(os.path.basename(p).split('psnr_')[1].split('_')[0])
        except: return 0.0
    return max(candidates, key=_psnr)


def load_models():
    models = {}
    weight_sources = {
        'baseline': os.path.join(root_dir, 'pretrained_weights', 'LOLv1.h5'),
        'd1d2d3':   best_h5(os.path.join(root_dir, 'experiments', 'ablation_d1d2d3'))
                    or best_h5(os.path.join(root_dir, 'experiments', 'LOLv1')),
    }
    for v in ['d1', 'd2', 'd3', 'd1d2', 'd1d3', 'd2d3']:
        w = best_h5(os.path.join(root_dir, 'experiments', f'ablation_{v}'))
        if w:
            weight_sources[v] = w

    for v, w_path in weight_sources.items():
        if not w_path or not os.path.isfile(w_path):
            print(f'  Skipping {v}: no weights at {w_path}')
            continue
        try:
            mod  = importlib.import_module(VARIANT_MODULE[v])
            d_cb = mod.Denoiser(16)
            d_cr = mod.Denoiser(16)
            m    = mod.LYT(filters=32, denoiser_cb=d_cb, denoiser_cr=d_cr)
            m(tf.zeros((1, 256, 256, 3)), training=False)
            m.load_weights(w_path)
            models[v] = m
            print(f'  Loaded {VARIANT_LABEL[v]:<16} ← {os.path.basename(w_path)}')
        except Exception as e:
            print(f'  Failed {v}: {e}')
    return models


def load_img(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32)
    return tf.expand_dims((img / 127.5) - 1.0, 0)


def to_np(t):
    return tf.clip_by_value((t[0] + 1.0) / 2.0, 0, 1).numpy()


def psnr_val(pred, gt):
    p = tf.clip_by_value((pred + 1.0) / 2.0, 0, 1)
    g = (gt + 1.0) / 2.0
    return float(tf.image.psnr(g, p, max_val=1.0))


def render_row(inp_path, gt_path, models, out_path, max_dim=400):
    inp = load_img(inp_path)
    gt  = load_img(gt_path)

    # Resize for inference
    h, w = inp.shape[1], inp.shape[2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    nh, nw = (int(h * scale) // 8) * 8, (int(w * scale) // 8) * 8
    inp_s = tf.image.resize(inp, [nh, nw])

    ORDER = ['baseline', 'd1', 'd2', 'd3', 'd1d2', 'd1d3', 'd2d3', 'd1d2d3']
    available = [v for v in ORDER if v in models]

    n_cols = 2 + len(available)  # Input + variants + GT
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))

    # Input
    axes[0].imshow(to_np(inp))
    axes[0].set_title('Input\n(Low-light)', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    # Each variant
    for i, v in enumerate(available):
        pred   = models[v](inp_s, training=False)
        pred_f = tf.image.resize(pred, [h, w])
        psnr   = psnr_val(pred_f, gt)
        axes[i + 1].imshow(to_np(pred_f))
        axes[i + 1].set_title(f'{VARIANT_LABEL[v]}\nPSNR {psnr:.2f}', fontsize=9, fontweight='bold')
        axes[i + 1].axis('off')

    # GT
    axes[-1].imshow(to_np(gt))
    axes[-1].set_title('Ground Truth', fontsize=9, fontweight='bold')
    axes[-1].axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(DATASET_PATHS.keys()), default='LOLv1')
    p.add_argument('--num',     type=int, default=5, help='Number of images to render')
    p.add_argument('--out',     default=None)
    p.add_argument('--max_dim', type=int, default=400)
    args = p.parse_args()

    out_dir = args.out or os.path.join(root_dir, 'results', f'ablation_visual_{args.dataset}')
    os.makedirs(out_dir, exist_ok=True)

    print('Loading models...')
    models = load_models()
    if not models:
        print('No models loaded. Train variants first, then re-run.')
        return

    paths = DATASET_PATHS[args.dataset]
    inputs  = sorted(glob.glob(paths['input']))
    targets = sorted(glob.glob(paths['target']))
    assert len(inputs) == len(targets), f'Mismatch: {len(inputs)} inputs vs {len(targets)} targets'

    step    = max(1, len(inputs) // args.num)
    indices = list(range(0, len(inputs), step))[:args.num]

    print(f'\nRendering {len(indices)} images from {args.dataset}...')
    for i in indices:
        name = os.path.splitext(os.path.basename(inputs[i]))[0]
        out  = os.path.join(out_dir, f'{name}_ablation.png')
        print(f'  {name}...', end=' ', flush=True)
        render_row(inputs[i], targets[i], models, out, args.max_dim)
        print(f'→ {out}')

    print(f'\nDone. Saved to {out_dir}/')


if __name__ == '__main__':
    main()
