#!/bin/bash
#SBATCH --job-name=docling-evaluate
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/evaluate-%j.out
#SBATCH --error=logs/evaluate-%j.err

# ponytail: Ensure logs directory exists
mkdir -p logs

# Load virtual environment
source .venv/bin/activate

# Check input arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch submit_evaluate.slurm <ground_truth_json> <markdown_file>"
    exit 1
fi

# Run the standalone quality evaluation
python3 scripts/docling-evaluate.py "$1" --markdown "$2"
