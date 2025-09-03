# Grading Experiments

This project evaluates open-source Large Language Models (LLMs) for automated grading of computer science student work. It includes baselines, DSPy and LangChain pipelines, and parameter-efficient fine-tuning (PEFT) methods such as LoRA and P-Tuning v2.

## Folder Structure

- `configs/` → Experiment grids and hyperparameters.
- `data/` → Datasets.
  - `raw/` holds original datasets (DO NOT commit sensitive data).
  - `processed/` holds rubric-aligned, preprocessed JSONL/CSV splits.
  - `samples/` holds small synthetic/anonymized sets for pilots.
- `scripts/` → Main Python scripts (training, inference, evaluation, perturbations).
- `prompts/` → Prompt templates for each strategy (Direct, Few-shot, Rubric-guided, etc.).
- `dsp_langchain/` → DSPy and LangChain pipeline implementations.
- `models/` → Trained PEFT adapters (LoRA, P-Tuning).
- `runs/` → Experiment outputs:
  - `logs/` = SLURM logs
  - `results/` = JSONL outputs of model runs
  - `metrics/` = Aggregated evaluation tables
- `slurm/` → SLURM job submission scripts for each experiment type.
- `docs/` → Documentation and figures for reporting results.

## How to start

1. Place datasets under `data/raw/` and preprocess into JSONL under `data/processed/`.
2. Define experiments in `configs/grid.yaml`.
3. Submit jobs using SLURM scripts in `slurm/`.
4. Collect outputs from `runs/` and evaluate with `scripts/evaluate.py`.

## Weights & Biases.

This repo uses [Weights & Biases](https://wandb.ai/) for experiment tracking.

### Setup

1. Install W&B:
   ```bash
   pip install wandb
   wandb login
   ```
