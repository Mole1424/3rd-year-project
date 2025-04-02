#!/bin/bash
#
#SBATCH --job-name=main_loop
#SBATCH --partition=gecko,falcon,eagle,kestrel
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
python3.12 -u main.py /dcs/large/u2204489/faceforensics/ /dcs/large/u2204489/