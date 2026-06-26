#!/usr/bin/env bash
# Run all remaining ablation variants then evaluate.
# d1 is already trained; d2 is running separately.
# Run from the TensorFlow/ directory:
#   bash run_ablation_remaining.sh 2>&1 | tee experiments/logs/ablation_remaining.log

set -e
PYTHON=./venv/bin/python3
SCRIPTS=scripts

for VARIANT in d3 d1d2 d1d3 d2d3 d1d2d3; do
    CKPT_DIR="experiments/ablation_${VARIANT}"
    # Skip if this variant already has a completed checkpoint (10 epochs finished = file with epoch_10 or highest epoch ≥ 10)
    if ls "${CKPT_DIR}"/net_${VARIANT}_psnr_*_epoch_10.weights.h5 2>/dev/null | grep -q .; then
        echo ""
        echo "========================================"
        echo "Skipping $VARIANT — already trained (epoch 10 checkpoint found)"
        echo "========================================"
        continue
    fi
    echo ""
    echo "========================================"
    echo "Training variant: $VARIANT"
    echo "========================================"
    $PYTHON $SCRIPTS/train_ablation.py --variant "$VARIANT"
done

echo ""
echo "========================================"
echo "Running evaluation (all variants)..."
echo "========================================"
$PYTHON $SCRIPTS/eval_ablation.py --auto

echo ""
echo "All done. Results saved to experiments/ablation_results.csv"
