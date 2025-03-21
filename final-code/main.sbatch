#!/bin/bash
#
#SBATCH --job-name=main_loop
#SBATCH --partition=gecko
#SBATCH --gres=gpu:3
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load CUDA
export TF_GPU_ALLOCATOR=cuda_malloc_async

source ../venv/bin/activate
python3.12 -u main.py /dcs/large/u2204489/faceforensics/ /dcs/large/u2204489/