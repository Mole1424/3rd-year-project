#!/bin/bash
#
#SBATCH --job-name=transferability
#SBATCH --partition=gecko,falcon
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load CUDA
export TF_GPU_ALLOCATOR=cuda_malloc_async

source ../venv/bin/activate
python3.12 -u transferability.py