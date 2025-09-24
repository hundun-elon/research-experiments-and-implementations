#!/bin/bash
#SBATCH --job-name=pilot_infer
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --output=runs/logs/pilot_infer-%j.out   # better: keep logs inside runs/logs/

module load cuda/12.0
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

# Workdir in home
cd /home-mscluster/smbuyazi/research/llm-grading

# Copy small dataset to local scratch (optional)
rsync -av /datasets/smbuyazi/llm-grading/samples/ $SCRATCH/pilot_data/

# Run inference script
python scripts/infer.py \
  --model-name "mistral-7b" \
  --input-jsonl $SCRATCH/pilot_data/pilot.jsonl \
  --output-jsonl runs/results/pilot_mistral.jsonl \
  --strategy rubric_guided \
  --seed 13
