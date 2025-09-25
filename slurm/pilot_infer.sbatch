#!/bin/bash
#SBATCH --job-name=pilot_infer
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH --output=runs/logs/pilot_infer-%j.out

module load cuda/12.0
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

cd $HOME/research/llm-grading

python scripts/infer.py \
  --model-name "meta-llama/Llama-2-7b-instruct" \
  --input-jsonl data/samples/pilot.jsonl \
  --output-jsonl runs/results/pilot_mistral.jsonl \
  --strategy dsp_rubric_program \
  --log-wandb \
  --wandb-project "llm-grading"
