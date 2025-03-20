#!/bin/bash
#SBATCH --job-name=main_loop
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --time=2-00:00:00

module purge
module load GCCcore/13.3.0 Python/3.12.3
export TF_GPU_ALLOCATOR=cuda_malloc_async

source ../venv/bin/activate
python3.12 -u main.py ../datasets/faceforensics/ ../models/