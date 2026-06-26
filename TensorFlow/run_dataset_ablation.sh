#!/usr/bin/env bash
# Dataset ablation study — trains Ours on 7 dataset combinations then evaluates.
# D1=LOLv1  D2=LOLv2-Real  D3=LOLI-Street(1000 samples)
# Note: D1 (LOLv1-only) is skipped if ablation_d1d2d3 checkpoint already exists.
#
# Run from TensorFlow/ directory:
#   bash run_dataset_ablation.sh 2>&1 | tee experiments/logs/dataset_ablation.log

set -e
PYTHON=./venv/bin/python3
SCRIPTS=scripts
mkdir -p experiments/logs

for COMBO in D1 D2 D3 D1D2 D1D3 D2D3 D1D2D3; do
    CKPT_DIR="experiments/dataset_abl_${COMBO}"

    # For D1, also accept the existing ablation_d1d2d3 checkpoint
    if [ "$COMBO" = "D1" ]; then
        if ls experiments/dataset_abl_D1/net_D1_*.h5 2>/dev/null | grep -q . || \
           ls experiments/ablation_d1d2d3/net_d1d2d3_*.h5 2>/dev/null | grep -q .; then
            echo "Skipping D1 — checkpoint already exists"
            continue
        fi
    else
        if ls "${CKPT_DIR}"/net_${COMBO}_psnr_*_epoch_10.weights.h5 2>/dev/null | grep -q .; then
            echo "Skipping ${COMBO} — epoch 10 checkpoint found"
            continue
        fi
    fi

    echo ""
    echo "========================================"
    echo "Training combo: ${COMBO}"
    echo "========================================"
    $PYTHON $SCRIPTS/train_dataset_ablation.py --combo "$COMBO"
done

echo ""
echo "========================================"
echo "Running evaluation..."
echo "========================================"
$PYTHON $SCRIPTS/eval_dataset_ablation.py --auto

echo ""
echo "Done. Results saved to experiments/dataset_ablation_results.csv"
