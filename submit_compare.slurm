#!/bin/bash
#SBATCH --job-name=docling-compare
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/compare/compare-%j.out
#SBATCH --error=logs/compare/compare-%j.err

WORK_DIR="/home/desrochest_wit_edu/Documents/superfund-docling/"
SCRIPT="$WORK_DIR/scripts/compare_evaluations.py"
cd "$WORK_DIR"
mkdir -p logs/compare


echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

source "$WORK_DIR/.venv/bin/activate"

# Force offline execution mode for cluster isolation
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Env Variable to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check input arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch submit_compare.slurm <path_to_pdf> <ground_truth_json>"
    exit 1
fi

# Run the comparative evaluation (Pictures ON vs OFF)
uv run --active "$SCRIPT" "$1" "$2"

echo "Job finished: $(date)"