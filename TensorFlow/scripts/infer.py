"""
LYT-Net 單張推理腳本
====================
用法：
  python scripts/infer.py --weights <權重路徑.h5> --image <圖片路徑>

選項：
  --weights   模型權重檔 (.h5) 路徑（必填）
  --image     輸入圖片路徑（必填，支援 jpg/png/bmp）
  --output    輸出圖片路徑（選填，預設為 <原檔名>_enhanced.png）
  --max_dim   推論最大邊長（選填，預設 0 = 不縮放）
  --compare   是否同時輸出原圖與增強圖的並排對比（選填）

範例：
  python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./test.jpg
  python scripts/infer.py --weights ./experiments/loli_scratch_v1/best.h5 --image ./dark.png --output result.png
  python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./test.jpg --compare
"""

import os
import sys
import argparse

# 設定路徑，確保能正確 import model
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 關閉 TensorFlow 冗餘日誌

import tensorflow as tf
import numpy as np


def build_model(weights_path):
    """建立 LYT-Net 模型並載入權重"""
    from model_modify.arch import LYT, Denoiser

    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None, None, None, 1))
    denoiser_cr.build(input_shape=(None, None, None, 1))

    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None, None, None, 3))
    model.load_weights(weights_path)

    return model


def load_image(image_path):
    """讀取圖片並正規化到 [-1, 1]，回傳 (batch_tensor, original_h, original_w)"""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    h, w = img.shape[0], img.shape[1]
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0  # [0, 255] → [-1, 1]
    return tf.expand_dims(img, 0), h, w


def safe_resize(img, max_dim):
    """將圖片縮放至 max_dim 以內（8 的倍數），回傳 (resized_img, new_h, new_w)"""
    h, w = img.shape[1], img.shape[2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    new_h = (int(h * scale) // 8) * 8
    new_w = (int(w * scale) // 8) * 8
    if new_h == 0:
        new_h = 8
    if new_w == 0:
        new_w = 8
    return tf.image.resize(img, [new_h, new_w]), new_h, new_w


def save_image(tensor, save_path):
    """將 [-1, 1] 的 tensor 存為圖片"""
    img = tf.clip_by_value((tensor[0] + 1.0) / 2.0, 0.0, 1.0)  # → [0, 1]
    img = tf.cast(img * 255.0, tf.uint8)
    encoded = tf.image.encode_png(img)
    tf.io.write_file(save_path, encoded)


def save_comparison(input_tensor, output_tensor, save_path):
    """產生原圖 vs 增強圖的並排對比圖"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inp_np = tf.clip_by_value((input_tensor[0] + 1.0) / 2.0, 0, 1).numpy()
    out_np = tf.clip_by_value((output_tensor[0] + 1.0) / 2.0, 0, 1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(inp_np)
    axes[0].set_title('Input (Low-light)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(out_np)
    axes[1].set_title('Enhanced', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='LYT-Net 單張推理腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./test.jpg
  python scripts/infer.py --weights ./experiments/best.h5 --image ./dark.png --output result.png
  python scripts/infer.py --weights ./pretrained_weights/LOLv1.h5 --image ./test.jpg --compare
        """
    )
    parser.add_argument('--weights', type=str, required=True,
                        help='模型權重檔路徑 (.h5)')
    parser.add_argument('--image', type=str, required=True,
                        help='輸入圖片路徑 (jpg/png/bmp)')
    parser.add_argument('--output', type=str, default=None,
                        help='輸出圖片路徑（預設為 <原檔名>_enhanced.png）')
    parser.add_argument('--max_dim', type=int, default=0,
                        help='推論最大邊長（預設 0 = 不縮放，建議 512 或 640）')
    parser.add_argument('--compare', action='store_true',
                        help='同時輸出原圖與增強圖的並排對比')
    parser.add_argument('--postprocess', action='store_true',
                        help='套用後處理（CLAHE + Guided Filter）')

    args = parser.parse_args()

    # ── 驗證輸入 ──
    if not os.path.isfile(args.weights):
        print(f'[錯誤] 權重檔不存在: {args.weights}')
        sys.exit(1)

    if not os.path.isfile(args.image):
        print(f'[錯誤] 圖片檔不存在: {args.image}')
        sys.exit(1)

    # ── 決定輸出路徑 ──
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        output_path = f'{base}_enhanced.png'

    # ── 載入模型 ──
    print(f'[1/3] 載入模型權重: {args.weights}')
    model = build_model(args.weights)
    print(f'      模型建立完成 ✓')

    # ── 讀取圖片 ──
    print(f'[2/3] 讀取圖片: {args.image}')
    img, orig_h, orig_w = load_image(args.image)
    print(f'      原始尺寸: {orig_h} x {orig_w}')

    # ── 推理 ──
    print(f'[3/3] 推理中...')

    if args.max_dim > 0:
        img_infer, new_h, new_w = safe_resize(img, args.max_dim)
        print(f'      推論尺寸: {new_h} x {new_w} (max_dim={args.max_dim})')
    else:
        # 確保尺寸是 8 的倍數（模型有 8x pooling）
        h, w = img.shape[1], img.shape[2]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            img_infer = tf.image.resize(img, [new_h, new_w])
            print(f'      自動調整尺寸至 8 的倍數: {new_h} x {new_w}')
        else:
            img_infer = img

    output = model(img_infer)

    # ── 後處理（可選）──
    if args.postprocess:
        from postprocess import postprocess
        out_01 = tf.clip_by_value((output[0] + 1.0) / 2.0, 0.0, 1.0).numpy()
        out_01 = postprocess(out_01)
        # 轉回 [-1, 1] 以保持後續儲存邏輯一致
        output = tf.expand_dims(tf.constant(out_01 * 2.0 - 1.0, dtype=tf.float32), 0)
        print(f'      後處理已套用 (CLAHE + Guided Filter) ✓')

    # Resize 回原始尺寸
    if output.shape[1] != orig_h or output.shape[2] != orig_w:
        output = tf.image.resize(output, [orig_h, orig_w])

    # ── 儲存結果 ──
    save_image(output, output_path)
    print(f'      增強圖片已儲存: {output_path} ✓')

    # ── 對比圖（可選）──
    if args.compare:
        compare_path = os.path.splitext(output_path)[0] + '_comparison.png'
        # 用原始尺寸的 input 做對比
        if img.shape[1] != orig_h or img.shape[2] != orig_w:
            img_display = tf.image.resize(img, [orig_h, orig_w])
        else:
            img_display = img
        save_comparison(img_display, output, compare_path)
        print(f'      對比圖已儲存: {compare_path} ✓')

    print('\n完成！')


if __name__ == '__main__':
    main()
