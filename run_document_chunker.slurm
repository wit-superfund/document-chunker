#!/bin/bash
#SBATCH --job-name=rag_demo_05262026
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rag_docling_demo/demo_%j.out
#SBATCH --error=logs/rag_docling_demo/demo_%j.err

set -e

WORK_DIR="/home/desrochest_wit_edu/Documents/superfund-docling/"
SCRIPT="$WORK_DIR/main.py"
cd "$WORK_DIR"
mkdir -p logs/rag_docling_demo

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"

source "$WORK_DIR/.venv/bin/activate"

uv run "$SCRIPT"

echo "Job finished: $(date)"
