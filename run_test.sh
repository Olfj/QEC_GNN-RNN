#!/usr/bin/env bash
#SBATCH -A NAISS2025-5-525
#SBATCH -p alvis
#SBATCH -t 1-00:00:00
#SBATCH -o logs_alvis/test_%j.out
#SBATCH --gpus-per-node=A40:1

module purge
module load PyTorch-Geometric/2.5.0-foss-2023a-PyTorch-2.1.2-CUDA-12.1.1
source .venv/bin/activate

MODEL_DIR=./models

# Find the latest model file matching a pattern (most recently modified)
latest_model() {
    ls -t "$MODEL_DIR"/$1 2>/dev/null | head -1
}

for d in 3 5; do
    M_LAST=$(latest_model "d${d}_p0.001_t50_dt2_last_*.pt")
    M_EC=$(latest_model "d${d}_p0.001_t50_dt2_error_chain_*.pt")
    M_MPP=$(latest_model "d${d}_p0.001_t50_dt2_mpp_*.pt")

    echo "=== d=$d ==="
    echo "  last:        $M_LAST"
    echo "  error_chain: $M_EC"
    echo "  mpp:         $M_MPP"

    python examples/test_nn.py --d "$d" --p 0.001 \
        --model_last "$M_LAST" \
        --model_error_chain "$M_EC" \
        --model_mpp "$M_MPP" \
        --out "eval_d${d}_p0.001"
done
