#!/bin/bash
set -euo pipefail
# Lightweight install on login node (CPU packages)
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

pip install --upgrade pip
# Install pure-Python packages that won't break GPU/CUDA
pip install numpy scipy pandas scikit-learn tqdm pyyaml transformers datasets dspy-ai langchain wandb mlflow sentence-transformers
#install Torch with CUDA in a GPU job as shown below.
echo "Done CPU-side installation. Install CUDA-aware torch inside a GPU job or container."
