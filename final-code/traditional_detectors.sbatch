#!/bin/bash
#
#SBATCH --job-name=traditional_detectors
#SBATCH --partition=gecko,falcon
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --output=joboutput_%j.out
#SBATCH --error=joboutput_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

module load CUDA
export TF_GPU_ALLOCATOR=cuda_malloc_async
source ../venv/bin/activate
python3.12 -u traditional_detectors.py