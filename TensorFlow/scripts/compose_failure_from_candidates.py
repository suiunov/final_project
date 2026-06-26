"""
把 pick_failures.py 生成的兩張 4 欄對比圖直接組合成 failure_cases.pdf (2 列)
"""
import os, sys, argparse
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case_a', required=True, help='案例 (a) 的 comparison PNG')
    p.add_argument('--case_b', required=True, help='案例 (b) 的 comparison PNG')
    p.add_argument('--out', required=True, help='輸出 PDF 路徑')
    args = p.parse_args()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    labels = ['(a) Over-dark / color loss', '(b) Strong light / bloom']
    for ax, path, label in zip(axes, [args.case_a, args.case_b], labels):
        ax.imshow(Image.open(path))
        ax.axis('off')
        ax.set_title(label, fontsize=11, loc='left', pad=4)
    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
