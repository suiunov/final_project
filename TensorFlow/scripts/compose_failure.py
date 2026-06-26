"""組合 2 個失敗案例 (Input | Ours | GT) → 單張 PDF"""
import argparse, os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case_a', nargs=3, required=True,
                   help='案例 a 的 3 張圖：input ours gt')
    p.add_argument('--case_b', nargs=3, required=True,
                   help='案例 b 的 3 張圖：input ours gt')
    p.add_argument('--out', required=True, help='輸出 PDF 路徑')
    args = p.parse_args()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = ['Input (Low-light)', 'Ours', 'Ground Truth']

    for col, (img_path, title) in enumerate(zip(args.case_a, titles)):
        axes[0, col].imshow(Image.open(img_path))
        axes[0, col].axis('off')
        axes[0, col].set_title(title, fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('(a)', fontsize=14, rotation=0, labelpad=20)

    for col, img_path in enumerate(args.case_b):
        axes[1, col].imshow(Image.open(img_path))
        axes[1, col].axis('off')
    axes[1, 0].set_ylabel('(b)', fontsize=14, rotation=0, labelpad=20)

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
