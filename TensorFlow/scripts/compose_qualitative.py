"""組合 3 個 4 欄場景對比圖 → 單張 PDF"""
import argparse, os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs=3, required=True,
                   help='3 張單場景對比圖（已並排成 Input|LYT-Net|Ours|GT 的一張圖）')
    p.add_argument('--out', required=True, help='輸出 PDF 路徑')
    args = p.parse_args()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, path in zip(axes, args.scenes):
        ax.imshow(Image.open(path))
        ax.axis('off')
    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
