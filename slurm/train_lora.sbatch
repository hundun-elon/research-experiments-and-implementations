#!/bin/bash
#SBATCH --job-name=train_lora
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH --output=logs/train_lora-%j.out

module load cuda/12.0
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

cd /home-mscluster/smbuyazi/research/llm-grading

# copy data to node-local scratch for fast IO
rsync -av /datasets/smbuyazi/llm-grading/train/ $SCRATCH/train_data/

python scripts/train_peft.py \
  --model-name "codellama-7b" \
  --train-jsonl $SCRATCH/train_data/train.jsonl \
  --output-dir models/lora/codellama_r8 \
  --lora-rank 8 \
  --batch-size 8 \
  --epochs 3 \
  --lr 1e-4 \
  --seed 13 \
  --wandb-project "llm-grading"
# When done, copy adapter weights back to persistent storage
rsync -av models/lora/codellama_r8/ /datasets/smbuyazi>/llm-grading/models/lora/codellama_r8/
