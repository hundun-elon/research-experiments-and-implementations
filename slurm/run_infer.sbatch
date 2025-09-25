#!/bin/bash
#SBATCH --job-name=grader_infer
#SBATCH --partition=biggpu          # GPU partition
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --output=logs/infer-%j.out

# Load CUDA
module load cuda/12.0

# Activate conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

# Move into repo
cd /home-mscluster/smbuyazi/research/llm-grading

# Input/output
INPUT=data/pilot/pilot.jsonl
OUTPUT=runs/results/pilot_mistral.jsonl

# Run inference
python scripts/infer.py \
  --model-name "mistral-7b" \
  --input-jsonl $INPUT \
  --output-jsonl $OUTPUT \
  --strategy rubric_guided \
  --seed 13

# Evaluate predictions
python scripts/metrics.py --input $OUTPUT
