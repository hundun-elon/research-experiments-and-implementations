#!/bin/bash
#SBATCH --job-name=install_gpu_deps
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --output=logs/install_gpu_deps-%j.out

module load cuda/12.0  # or set PATH as mscluster suggests
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate llm-exp

# Install torch matching the compute node CUDA (example for CUDA 12.0, PyTorch build must exist)
pip install --upgrade pip
pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121
pip install accelerate
pip install peft
# optionally quantify libs like bitsandbytes if using 4/8-bit quantization
pip install bitsandbytes

echo "GPU packages installed on node."
