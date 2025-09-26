#!/bin/bash
#SBATCH --job-name=grader_infer
#SBATCH --partition=biggpu          # GPU partition
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH --output=logs/infer-%j.out

# Ensure environment modules work
source /etc/profile

# Load CUDA if needed (optional, cluster may already have it)
# module load cuda/12.0

# Activate conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

# Move into repo
cd /home-mscluster/smbuyazi/research/llm-grading

# Make sure output folder exists
mkdir -p runs/results

# Input/output files (use absolute path if needed)
INPUT=/home-mscluster/smbuyazi/research/llm-grading/data/pilot/pilot.jsonl
OUTPUT=/home-mscluster/smbuyazi/research/llm-grading/runs/results/pilot_mistral.jsonl

# Run inference
python scripts/infer.py \
  --model-name "mistral-7b" \
  --input-jsonl $INPUT \
  --output-jsonl $OUTPUT \
  --strategy rubric_guided \
  --seed 13

# Evaluate predictions
python scripts/metrics.py --input $OUTPUT
