# Cross-Domain Low-Light Image Enhancement for Traffic Scenes via a YUV Transformer

> Built on [LYT-Net](https://github.com/albrateanu/LYT-Net) — Lightweight YUV Transformer-based Network for Low-light Image Enhancement

**Nooruzbek Suiunov, Ezra, Y.C. Chuang** — Department of Computer Science and Engineering, Yuan Ze University, Taiwan

---

## Overview

This project extends LYT-Net with three targeted architectural modifications aimed at improving **cross-domain generalization** for low-light image enhancement in real-world traffic scenarios. The model is trained on LOL-v1 and evaluated on three test sets spanning an increasing distributional gap: the in-domain LOL-v1 set, the cross-domain LOL-v2-Real set, and the outdoor LOLI-Street benchmark — the most semantically relevant testbed for nighttime traffic-sign scenes.

We introduce three modifications — **D1**, **D2**, and **D3** — that are mutually stabilizing: each individually disrupts the LYT-Net balance, but together they recover in-domain quality while producing cross-domain gains that no subset achieves alone.

---

## Modifications over LYT-Net

| ID | Name | Description |
|----|------|-------------|
| **D1** | Deeper Per-Channel Feature Extractor | Two stacked 3×3 Conv + ReLU layers per YUV channel branch, enlarging the receptive field and capturing richer low-level features (edges, textures, local contrast) before the global attention stage |
| **D2** | Bottleneck Dropout Regularization | Dropout (p=0.1) inserted after the Multi-Head Self-Attention module at the chrominance denoiser bottleneck, discouraging memorization of LOL-v1's indoor noise distribution |
| **D3** | Learnable Luminance Injection Weight | The luminance-to-chrominance coupling coefficient is parameterized as a trainable scalar w_lum (initialized to 0.2), allowing adaptive cross-channel coupling strength to be learned from data |

---

## Results

### Cross-Dataset Quantitative Comparison

All models trained on **LOL-v1** and evaluated on three test sets without fine-tuning. ↑ higher is better, ↓ lower is better. **Bold** = best in column.

| Method | LOL-v1 PSNR↑ | LOL-v1 SSIM↑ | LOL-v1 LPIPS↓ | LOL-v2-Real PSNR↑ | LOL-v2-Real SSIM↑ | LOL-v2-Real LPIPS↓ | LOLI-Street PSNR↑ | LOLI-Street SSIM↑ | LOLI-Street LPIPS↓ |
|--------|-------------|-------------|--------------|------------------|------------------|------------------|------------------|------------------|------------------|
| LYT-Net (baseline) | **22.38** | 0.826 | **0.076** | 22.25 | 0.854 | 0.082 | 14.64 | 0.774 | 0.106 |
| Ours (D1+D2+D3) | 22.20 | **0.831** | 0.077 | **23.11** | **0.860** | **0.069** | **17.22** | **0.842** | **0.080** |

Key takeaways:
- **In-domain (LOL-v1):** Comparable performance — only −0.18 dB PSNR, within measurement noise, with improved SSIM (+0.005)
- **Cross-domain (LOL-v2-Real):** +0.86 dB PSNR / +0.006 SSIM / −0.013 LPIPS
- **Out-of-domain (LOLI-Street):** **+2.58 dB PSNR / +0.068 SSIM / −0.027 LPIPS**

---

## Ablation Study

All variants trained on LOL-v1. Bold = best in column.

| Variant | LOL-v1 PSNR↑ | LOL-v1 SSIM↑ | LOL-v1 LPIPS↓ | LOL-v2-Real PSNR↑ | LOLI-Street PSNR↑ | LOLI-Street SSIM↑ | LOLI-Street LPIPS↓ |
|---------|-------------|-------------|--------------|------------------|------------------|------------------|------------------|
| Baseline (LYT-Net) | **22.38** | 0.826 | **0.076** | 22.25 | 14.64 | 0.774 | 0.106 |
| +D1 only | 19.63 | 0.772 | 0.148 | 19.11 | 14.71 | 0.772 | 0.130 |
| +D2 only | 19.12 | 0.773 | 0.149 | 20.12 | 17.65 | 0.835 | 0.104 |
| +D3 only | 18.99 | 0.768 | 0.158 | 19.27 | **18.91** | 0.816 | 0.164 |
| +D1+D2 | 19.50 | 0.778 | 0.144 | 20.62 | 16.05 | 0.806 | 0.139 |
| +D1+D3 | 18.96 | 0.764 | 0.160 | 18.32 | 18.67 | 0.817 | 0.128 |
| +D2+D3 | 18.92 | 0.765 | 0.163 | 19.15 | 17.36 | 0.821 | 0.146 |
| **D1+D2+D3 (Ours)** | 22.20 | **0.831** | 0.077 | **23.11** | 17.22 | **0.842** | **0.080** |

Notable findings:
- **D2 alone** yields the most consistent out-of-domain gain (+3.01 dB on LOLI-Street)
- **D3 alone** achieves the highest single LOLI-Street PSNR (18.91 dB) but at the cost of severe in-domain regression
- **Only D1+D2+D3 together** simultaneously recovers in-domain quality and improves on both cross-domain sets

---

## Datasets

| Dataset | Description | Split Used |
|---------|-------------|------------|
| [LOL-v1](https://daooshee.github.io/BMVC2018website/) | 485 paired low/normal-light images, predominantly indoor | Training + test (15 pairs) |
| LOL-v2-Real | 689 training / 100 testing pairs, broader indoor/outdoor scenes | Test only (cross-domain) |
| [LOLI-Street](https://github.com/md-islam/LoLI-Street) | Outdoor urban driving scenes with traffic signs; 3 illumination subsets (dense, light, moderate) | Validation split (out-of-domain) |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- TensorFlow 2.x
- numpy
- Pillow
- scikit-image

> The model is implemented in **TensorFlow** and was trained on a single NVIDIA GeForce RTX 4060 (8 GB) GPU.

---

## Usage

### Training

```bash
python train.py \
  --data_dir data/LOL \
  --epochs 200 \
  --batch_size 8 \
  --lr 2e-4
```

### Evaluation

```bash
# Evaluate on LOL-v1 (in-domain)
python evaluate.py \
  --dataset LOL-v1 \
  --data_dir data/LOL/eval15 \
  --checkpoint checkpoints/best_model.h5

# Evaluate on LOLI-Street (out-of-domain)
python evaluate.py \
  --dataset LOLI-Street \
  --data_dir data/LOLI-Street \
  --checkpoint checkpoints/best_model.h5
```

### Inference on a single image

```bash
python infer.py \
  --input path/to/low_light_image.jpg \
  --output path/to/output.jpg \
  --checkpoint checkpoints/best_model.h5
```

---

## Project Structure

```
├── data/
│   ├── LOL/
│   ├── LOL-v2-Real/
│   └── LOLI-Street/
├── models/
│   ├── lyt_net.py              # Original LYT-Net architecture
│   └── lyt_net_modified.py     # Our modified architecture (D1, D2, D3)
├── checkpoints/
├── train.py
├── evaluate.py
├── infer.py
├── requirements.txt
└── README.md
```

---

## Citation

If you use this work, please also cite the original LYT-Net paper:

```bibtex
@article{brateanu2024lytnet,
  title={LYT-Net: Lightweight YUV Transformer-based Network for Low-light Image Enhancement},
  author={Brateanu, Alexandru and Balmez, Raul and Avram, Adrian and Orhei, Ciprian},
  journal={arXiv preprint arXiv:2401.15204},
  year={2024}
}
```

---

## Acknowledgements

This project is built upon [LYT-Net](https://github.com/albrateanu/LYT-Net). We thank the original authors for their open-source contribution.

This work was completed as part of a course project at **Yuan Ze University**, Department of Computer Science and Engineering.

---

## License

This project is released for educational use only.
# final_project_Generative_AI_class

