#!/bin/bash
#SBATCH --job-name=eval_log
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --output=logs/eval_log-%j.out

eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

cd /home-mscluster/smbuyazi/research/llm-grading

python scripts/metrics.py runs/results/pilot_mistral.jsonl --log-wandb --project "llm-grading"
